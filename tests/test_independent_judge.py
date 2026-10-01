"""Independent answer calls, parallelism, bounded retries and automatic Critic IDs."""

from contextvars import ContextVar
import json
from threading import Barrier, Lock

import httpx
import pytest

from verge.calculator import calculate_rubric_judgment
from verge.llm import JUDGE_RESPONSE_SIDE, LLMRequestError, OpenAICompatibleBackend
from verge.models import Criterion, CriterionJudgment, Rubric
from verge.modules import JudgeOutputError, LLMCritic, LLMJudge
from verge.prompts import NumberedIDCodec, critic_payload


def rubric():
    return Rubric('private-rubric', None, [Criterion('private-first', '第一项要求？', 6, []),
                                          Criterion('private-second', '第二项要求？', 4, [])])


def response(value):
    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(value)},
                                                'finish_reason': 'stop'}]})


def test_real_http_judge_concurrent_private_answers_and_copied_context(monkeypatch, wire_payload):
    monkeypatch.setenv('JUDGE_KEY', 'test-key')
    rendezvous = Barrier(2)
    context = ContextVar('test_sample', default=None)
    events, calls = [], []
    lock = Lock()

    def handler(request):
        body = json.loads(request.content)
        payload = wire_payload(body)
        assert set(payload) == {'task', 'criteria', 'response'}
        assert payload['task'] == '测试任务'
        assert [c['criterion_id'] for c in payload['criteria']] == ['1', '2']
        assert all(set(c) == {'criterion_id', 'question'} for c in payload['criteria'])
        assert 'chosen' not in body['messages'][0]['content']
        assert 'rejected' not in body['messages'][0]['content']
        with lock:
            calls.append(payload['response'])
        rendezvous.wait(timeout=3)  # Sequential implementation cannot pass.
        return response({'criterion_judgments': [
            {'criterion_id': c['criterion_id'], 'satisfied': payload['response'] == 'answer A'}
            for c in reversed(payload['criteria'])]})

    def log(event):
        with lock:
            events.append((context.get(), event))

    client = httpx.Client(base_url='https://judge.invalid/v1/', transport=httpx.MockTransport(handler))
    backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY',
                                     max_retries=0, metadata_hook=log, trace_hook=log)
    token = context.set('task-private')
    try:
        result = LLMJudge(backend).judge(rubric(), 'answer A', 'answer B', task="测试任务")
    finally:
        context.reset(token)
        client.close()
    assert sorted(calls) == ['answer A', 'answer B']
    assert result.chosen_score == 1 and result.rejected_score == 0
    assert [c.criterion_id for c in result.criterion_judgments] == ['private-first', 'private-second']
    assert all(sample == 'task-private' for sample, _ in events)
    assert {event['response_side'] for _, event in events} == {'chosen', 'rejected'}
    assert JUDGE_RESPONSE_SIDE.get() is None


def test_retry_only_invalid_side_retains_other_result(monkeypatch, wire_payload):
    monkeypatch.setenv('JUDGE_KEY', 'test-key')
    calls = {'A': [], 'B': []}
    lock = Lock()

    def handler(request):
        payload = wire_payload(json.loads(request.content))
        side = payload['response']
        with lock:
            calls[side].append(payload)
            count = len(calls[side])
        if side == 'A' and count == 1:
            return response({'criterion_judgments': [{'criterion_id': '9', 'satisfied': True}]})
        return response({'criterion_judgments': [
            {'criterion_id': c['criterion_id'], 'satisfied': side == 'A'} for c in payload['criteria']]})

    with httpx.Client(base_url='https://judge.invalid/v1/', transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY', max_retries=0)
        result = LLMJudge(backend, invalid_output_retries=1).judge(rubric(), 'A', 'B', task="测试任务")
    assert result.success
    assert len(calls['A']) == 2 and len(calls['B']) == 1
    assert 'validation_feedback' in calls['A'][1]
    assert 'validation_feedback' not in calls['B'][0]


@pytest.mark.parametrize('fatal_other_side', [False, True])
def test_exhaustion_is_unknown_and_transport_error_takes_precedence(monkeypatch, wire_payload,
                                                                     fatal_other_side):
    monkeypatch.setenv('JUDGE_KEY', 'test-key')
    calls = {'A': 0, 'B': 0}
    lock = Lock()

    def handler(request):
        payload = wire_payload(json.loads(request.content))
        with lock:
            calls[payload['response']] += 1
        if payload['response'] == 'B':
            if fatal_other_side:
                return httpx.Response(401, json={'error': 'unauthorized'})
            return response({'criterion_judgments': [
                {'criterion_id': c['criterion_id'], 'satisfied': False} for c in payload['criteria']]})
        return response({'criterion_judgments': [{'criterion_id': '9', 'satisfied': True}]})

    with httpx.Client(base_url='https://judge.invalid/v1/', transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY', max_retries=0)
        expected = LLMRequestError if fatal_other_side else JudgeOutputError
        with pytest.raises(expected):
            LLMJudge(backend, invalid_output_retries=1).judge(rubric(), 'A', 'B', task="测试任务")
    assert calls == {'A': 2, 'B': 1}


def test_all_reused_judgments_require_zero_calls():
    class Unused:
        def generate(self, *args):
            pytest.fail('unchanged criteria must reuse both sides')

    base = rubric()
    judgment = calculate_rubric_judgment(base, [CriterionJudgment(c.criterion_id, True, False)
                                               for c in base.criteria])
    revised = Rubric('revised', base.rubric_id, base.criteria)
    result = LLMJudge(Unused()).judge(revised, 'A', 'B', previous=base, previous_judgment=judgment, task="测试任务")
    assert result.criterion_judgments == judgment.criterion_judgments


def test_same_response_is_independent_and_can_tie(monkeypatch, wire_payload):
    monkeypatch.setenv('JUDGE_KEY', 'test-key')
    calls = []

    def handler(request):
        payload = wire_payload(json.loads(request.content))
        calls.append(payload)
        return response({'criterion_judgments': [{'criterion_id': c['criterion_id'], 'satisfied': True}
                                                 for c in payload['criteria']]})

    with httpx.Client(base_url='https://judge.invalid/v1/', transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY', max_retries=0)
        result = LLMJudge(backend).judge(rubric(), 'same answer', 'same answer', task="测试任务")
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result.margin == 0 and not result.success


def test_critic_http_returns_content_system_assigns_ids_across_rounds(monkeypatch):
    monkeypatch.setenv('JUDGE_KEY', 'test-key')
    output = {'analysis': '根据任务调整标准。', 'operations': [
        {'operation_type': 'MODIFY', 'target_criterion_id': 1,
         'new_or_updated_criterion': {'question': '修订后的要求？', 'importance_score': 8}},
        {'operation_type': 'ADD', 'target_criterion_id': None,
         'new_or_updated_criterion': {'question': '新增的要求？', 'importance_score': 5}},
        {'operation_type': 'DELETE', 'target_criterion_id': 2, 'new_or_updated_criterion': None}]}
    base = rubric()
    judgment = calculate_rubric_judgment(base, [CriterionJudgment(c.criterion_id, False, True)
                                               for c in base.criteria])
    with httpx.Client(base_url='https://judge.invalid/v1/',
                      transport=httpx.MockTransport(lambda _: response(output))) as client:
        backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY', max_retries=0)
        proposal = LLMCritic(backend).critique('任务', 'A', 'B', base, judgment, 1)
    assert [o.operation_id for o in proposal.operations] == ['op1_1', 'op1_2', 'op1_3']
    assert proposal.operations[0].new_or_updated_criterion.criterion_id == 'private-first'
    assert proposal.operations[1].new_or_updated_criterion.criterion_id == 'added1_2'
    assert all(o.before_criterion is None for o in proposal.operations)
    assert proposal.operations[2].target_criterion_id == 'private-second'

    # Restore the same content for a new round: IDs are round-local, not model-generated.
    from verge.models import InvalidRevisionAttempt
    second_input = critic_payload('任务', 'A', 'B', base, judgment, 2,
        previous_invalid_attempt=InvalidRevisionAttempt(1, '诊断', ['结构错误']))
    codec = NumberedIDCodec('critic', second_input)
    codec.project(second_input)
    second = codec.restore({**output, 'operations': [
        {**o, 'target_criterion_id': str(o['target_criterion_id']) if o['target_criterion_id'] else None}
        for o in output['operations']]})
    assert [o['operation_id'] for o in second['operations']] == ['op2_1', 'op2_2', 'op2_3']
    assert second['operations'][1]['new_or_updated_criterion']['criterion_id'] == 'added2_2'


def test_critic_system_new_id_avoids_existing_base_id():
    base = Rubric('base', None, [Criterion('added1_1', '旧问题？', 5, []),
                                Criterion('added1_1_new', '其它问题？', 5, [])])
    judgment = calculate_rubric_judgment(base, [CriterionJudgment(c.criterion_id, False, True)
                                               for c in base.criteria])
    codec = NumberedIDCodec('critic', critic_payload('任务', 'A', 'B', base, judgment, 1))
    restored = codec.restore({'operations': [{'operation_type': 'ADD', 'target_criterion_id': None,
        'new_or_updated_criterion': {'question': '新问题？', 'importance_score': 5}}]})
    assert restored['operations'][0]['new_or_updated_criterion']['criterion_id'] == 'added1_1_new_new'


def test_judge_can_copy_numeric_labels_without_format_retry(monkeypatch):
    import re
    monkeypatch.setenv("JUDGE_KEY", "test-key")
    calls = []
    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][1]["content"]
        copied_ids = re.findall(r"^\s*- (\d+)：", text, re.MULTILINE)
        assert copied_ids == ["1", "2"]
        assert "标准 1" not in text and "标准 2" not in text
        calls.append(text)
        return response({"criterion_judgments": [
            {"criterion_id": cid, "satisfied": True} for cid in copied_ids]})
    with httpx.Client(base_url="https://judge.invalid/v1/",
                      transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend("test", base_url="https://judge.invalid/v1/",
                                          client=client, api_key_env="JUDGE_KEY", max_retries=0)
        result = LLMJudge(backend).judge(rubric(), "A", "B", task="核验任务")
    assert len(calls) == 2  # Each side succeeds immediately.
    assert result.chosen_score == result.rejected_score == 1
