"""同一 Generator 的真实多轮消息、最新基线、错误隔离与候选信息边界。"""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest

from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMRequestError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import PreferenceSample, Rubric
from verge.modules import LLMInitialRubricGenerator, make_fake_modules
from verge.pipeline import PipelineRunner
from verge.prompts import reflection_payload, llm_messages


def run(tmp_path, scenario='attempt_2_success', modules=None, **kwargs):
    modules = modules or make_fake_modules(scenario)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path), **kwargs)
    sample = PreferenceSample('one', '任务文本', '私有优选全文', '私有非优选全文', {})
    return runner.run_sample(sample), modules, runner


def test_two_rounds_revise_latest_rubric_keep_one_generator_context(tmp_path):
    record, modules, _ = run(tmp_path)
    first, second = record.revision_attempts
    assert second.rubric.base_rubric_id == first.rubric.rubric_id
    assert second.previous_rubric == first.rubric and second.previous_judgment == first.judgment
    assert first.operations == second.operations == []
    calls = [p for role, p in modules.backend.requests if role == 'initial_generator']
    assert len(calls) == 3
    first_messages, second_messages = calls[1]['_conversation'], calls[2]['_conversation']
    assert [m['role'] for m in first_messages] == ['system', 'user', 'assistant', 'user']
    assert [m['role'] for m in second_messages] == ['system', 'user', 'assistant', 'user', 'assistant', 'user']
    assert second_messages[:len(first_messages)] == first_messages
    assert 'improvement_advice' not in second_messages[2]['content']  # actual R0 reply, not Critic
    for p in calls:
        assert '私有优选全文' not in json.dumps(p, ensure_ascii=False)
        assert '私有非优选全文' not in json.dumps(p, ensure_ascii=False)
        assert '当前标准对关键要求的判别不足' not in json.dumps(p, ensure_ascii=False)
    critics = [p for role, p in modules.backend.requests if role == 'critic']
    assert critics[1]['current_rubric']['rubric_id'] == first.rubric.rubric_id
    payload = reflection_payload('任务', record.initial_rubric, record.initial_judgment,
                                 second, chosen='好', rejected='坏')
    text = llm_messages('reflector', payload)[1]['content']
    assert second.analysis in text and second.feedback in text
    assert '本次修订前标准' in text and 'operations' not in json.dumps(payload)


def test_generator_exact_raw_assistant_reply_and_backend_format_retry(monkeypatch):
    monkeypatch.setenv('DIALOGUE_KEY', 'fake')
    bodies = []
    reply = '{ "analysis": "任务分析", "criteria": [{"question":"要求甲？","importance_score":5,"source_experiences":[]},{"question":"要求乙？","importance_score":3,"source_experiences":[]}] }'
    def handler(request):
        bodies.append(json.loads(request.content))
        content = 'bad JSON' if len(bodies) == 2 else reply
        return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]})
    client = httpx.Client(base_url='https://api.openai.com/v1/', transport=httpx.MockTransport(handler))
    backend = OpenAICompatibleBackend('mock', api_key_env='DIALOGUE_KEY', client=client, sleep=lambda _: None)
    generator = LLMInitialRubricGenerator(backend)
    r0 = generator.generate('原始任务', [])
    r1 = generator.revise(r0, '调整重要性', 1)
    generator.revise(r1, '补全覆盖', 2)
    assert bodies[1]['messages'][2] == {'role': 'assistant', 'content': reply}
    assert bodies[2]['messages'][-1]['role'] == 'user' and '上次输出格式无效' in bodies[2]['messages'][-1]['content']
    assert bodies[3]['messages'][:len(bodies[2]['messages'])] == bodies[2]['messages']
    assert bodies[3]['messages'][-2] == {'role': 'assistant', 'content': reply}
    generator.generate('另一条任务', [])
    assert len(bodies[-1]['messages']) == 2
    assert '原始任务' not in json.dumps(bodies[-1]['messages'], ensure_ascii=False)
    client.close()


def test_sample_and_thread_contexts_are_isolated():
    modules = make_fake_modules('attempt_1_success')
    generator = modules.initial_generator
    barrier = Barrier(2)
    def worker(task):
        r0 = generator.generate(task, [])
        barrier.wait(timeout=3)
        generator.revise(r0, '修订', 1)
        messages = generator._conversation.get()
        other = '任务乙' if task == '任务甲' else '任务甲'
        assert task in json.dumps(messages, ensure_ascii=False)
        assert other not in json.dumps(messages, ensure_ascii=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, ['任务甲', '任务乙']))
    with pytest.raises(ValueError, match='initial conversation'):
        generator.revise(Rubric('unused', None, []), '修订', 1)


@pytest.mark.parametrize('bad_count, expected', [(1, 'revision_success_1'), (3, 'revision_success_2'), (6, 'final_failure')])
def test_invalid_full_rubric_retries_or_moves_to_next_round(tmp_path, bad_count, expected):
    modules = make_fake_modules('attempt_1_success' if bad_count == 1 else 'attempt_2_success')
    original = modules.backend.generate
    count = 0
    def bad(role, payload):
        nonlocal count
        response = original(role, payload)
        if role == 'initial_generator' and payload.get('revision_number'):
            count += 1
            if count <= bad_count:
                response['rubric']['criteria'] = response['rubric']['criteria'][:1]
        return response
    modules.backend.generate = bad
    record, _, _ = run(tmp_path, modules=modules)
    assert record.state == expected
    if bad_count >= 3:
        assert len(record.invalid_revision_attempts[0].validation_failures) == 3
        assert record.invalid_revision_attempts[0].feedback
        if record.revision_attempts:
            assert record.revision_attempts[0].rubric.base_rubric_id == record.initial_rubric.rubric_id
    else:
        assert len(record.revision_attempts[0].validation_failures) == 1
    assert record.judge_error is None and record.sample_error is None


def test_generator_format_exhaustion_does_not_stack_retries(tmp_path):
    modules = make_fake_modules('attempt_2_success')
    original = modules.initial_generator.revise
    calls = []
    def fail_first(previous, advice, number, **kwargs):
        calls.append(number)
        if number == 1:
            raise LLMFormatError('generator malformed')
        return original(previous, advice, number, **kwargs)
    modules.initial_generator.revise = fail_first
    record, _, _ = run(tmp_path, modules=modules)
    assert calls == [1, 2]
    assert record.state == 'revision_success_2'
    assert record.invalid_revision_attempts[0].analysis


def test_generator_network_failure_still_aborts(tmp_path):
    modules = make_fake_modules('attempt_1_success')
    def fail(*args, **kwargs):
        raise LLMRequestError('network')
    modules.initial_generator.revise = fail
    with pytest.raises(LLMRequestError):
        run(tmp_path, modules=modules)
