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
    return Rubric(
        "private-rubric",
        [
            Criterion("private-first", "第一项要求？", 6, []),
            Criterion("private-second", "第二项要求？", 4, []),
        ],
    )


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
            {"reason": f"evidence:{payload['response']}:{c['criterion_id']}", 'criterion_id': c['criterion_id'], 'satisfied': payload['response'] == 'answer A'}
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
    for i, verdict in enumerate(result.criterion_judgments, 1):
        assert verdict.chosen_reason == f"evidence:answer A:{i}"
        assert verdict.rejected_reason == f"evidence:answer B:{i}"
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
            return response({'criterion_judgments': [{"reason": "fixture evidence", 'criterion_id': '9', 'satisfied': True}]})
        return response({'criterion_judgments': [
            {"reason": "fixture evidence", 'criterion_id': c['criterion_id'], 'satisfied': side == 'A'} for c in payload['criteria']]})

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
                {"reason": "fixture evidence", 'criterion_id': c['criterion_id'], 'satisfied': False} for c in payload['criteria']]})
        return response({'criterion_judgments': [{"reason": "fixture evidence", 'criterion_id': '9', 'satisfied': True}]})

    with httpx.Client(base_url='https://judge.invalid/v1/', transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY', max_retries=0)
        expected = LLMRequestError if fatal_other_side else JudgeOutputError
        with pytest.raises(expected):
            LLMJudge(backend, invalid_output_retries=1).judge(rubric(), 'A', 'B', task="测试任务")
    assert calls == {'A': 2, 'B': 1}


def test_all_reused_judgments_require_zero_calls():
    class Unused:
        def generate(self, *args):
            pytest.fail("unchanged criteria must reuse both sides")

    base = rubric()
    judgment = calculate_rubric_judgment(
        base, [CriterionJudgment(c.criterion_id, True, False, "chosen fixture evidence", "rejected fixture evidence") for c in base.criteria]
    )
    revised = Rubric("revised", base.criteria)
    result = LLMJudge(Unused()).judge(
        revised, "A", "B", previous=base, previous_judgment=judgment, task="测试任务"
    )
    assert result.criterion_judgments == judgment.criterion_judgments


def test_reuse_changed_ids_and_weights_preserves_original_reasons():
    class Unused:
        def generate(self, *args):
            pytest.fail("same questions must reuse judgments and reasons")

    base = rubric()
    judgment = calculate_rubric_judgment(base, [
        CriterionJudgment(c.criterion_id, True, False, f"chosen:{i}", f"rejected:{i}")
        for i, c in enumerate(base.criteria)
    ])
    revised = Rubric("revised", [
        Criterion(f"new{i}", c.question, weight, [])
        for i, (c, weight) in enumerate(zip(base.criteria, [.01, .99]))
    ])
    result = LLMJudge(Unused()).judge(revised, "A", "B", task="task",
        previous=base, previous_judgment=judgment)
    assert result.normalized_weights == {"new0": .01, "new1": .99}
    for i, verdict in enumerate(result.criterion_judgments):
        assert verdict.criterion_id == f"new{i}"
        assert (verdict.chosen_reason, verdict.rejected_reason) == (f"chosen:{i}", f"rejected:{i}")


@pytest.mark.parametrize("bad_reason", [None, "", "  ", 7])
def test_missing_or_invalid_reason_uses_bounded_format_retry(monkeypatch, wire_payload, bad_reason):
    monkeypatch.setenv("JUDGE_KEY", "test-key")
    calls = {"A": 0, "B": 0}
    lock = Lock()

    def handler(request):
        payload = wire_payload(json.loads(request.content))
        side = payload["response"]
        with lock:
            calls[side] += 1
        rows = []
        for c in payload["criteria"]:
            row = {"criterion_id": c["criterion_id"], "satisfied": True, "reason": "valid B evidence"}
            if side == "A":
                if bad_reason is None:
                    del row["reason"]
                else:
                    row["reason"] = bad_reason
            rows.append(row)
        return response({"criterion_judgments": rows})

    with httpx.Client(base_url="https://judge.invalid/v1/", transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend("test", base_url="https://judge.invalid/v1/",
            client=client, api_key_env="JUDGE_KEY", max_retries=1, sleep=lambda _: None)
        with pytest.raises(JudgeOutputError, match="format retries exhausted"):
            LLMJudge(backend, invalid_output_retries=3).judge(rubric(), "A", "B", task="task")
    assert calls == {"A": 2, "B": 1}  # No stacked adapter retry or retry of the valid side.


def test_same_response_is_independent_and_can_tie(monkeypatch, wire_payload):
    monkeypatch.setenv('JUDGE_KEY', 'test-key')
    calls = []

    def handler(request):
        payload = wire_payload(json.loads(request.content))
        calls.append(payload)
        return response({'criterion_judgments': [{"reason": "fixture evidence", 'criterion_id': c['criterion_id'], 'satisfied': True}
                                                 for c in payload['criteria']]})

    with httpx.Client(base_url='https://judge.invalid/v1/', transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend('test', base_url='https://judge.invalid/v1/', client=client, api_key_env='JUDGE_KEY', max_retries=0)
        result = LLMJudge(backend).judge(rubric(), 'same answer', 'same answer', task="测试任务")
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result.margin == 0 and not result.success


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
            {"reason": "fixture evidence", "criterion_id": cid, "satisfied": True} for cid in copied_ids]})
    with httpx.Client(base_url="https://judge.invalid/v1/",
                      transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend("test", base_url="https://judge.invalid/v1/",
                                          client=client, api_key_env="JUDGE_KEY", max_retries=0)
        result = LLMJudge(backend).judge(rubric(), "A", "B", task="核验任务")
    assert len(calls) == 2  # Each side succeeds immediately.
    assert result.chosen_score == result.rejected_score == 1
