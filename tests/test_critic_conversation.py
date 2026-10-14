"""Critic消息继承、真实失败回复及样本局部隔离；不证明语义效果。"""

from copy import deepcopy
from dataclasses import replace
import json

import httpx
import pytest

from test_pipeline import runner, sample
from verge.calculator import calculate_rubric_judgment
from verge.llm import LLMFormatError, OpenAICompatibleBackend
from verge.models import Criterion, CriterionJudgment, Rubric
from verge.modules import LLMCritic


@pytest.mark.parametrize("retries", [0, 1])
def test_raw_format_failure_is_retained_without_extra_business_rounds(monkeypatch, retries):
    monkeypatch.setenv("MOCK_KEY", "test-secret")
    requests, traces = [], []
    raw = '{ "analysis": "新分析", "candidates": [{"content":"方法","trigger":"条件"}] }'

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": "不是JSON的真实回复" if len(requests) == 1 else raw,
        }}]})

    rubric = Rubric("r0", [Criterion("a", "要求一？", .5, []), Criterion("b", "要求二？", .5, [])])
    judgment = calculate_rubric_judgment(rubric, [
        CriterionJudgment(c.criterion_id, False, False, "原始依据甲", "原始依据乙")
        for c in rubric.criteria
    ])
    history = []
    with httpx.Client(base_url="https://mock.invalid/v1/", transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend(
            "mock", base_url="https://mock.invalid/v1/", client=client,
            api_key_env="MOCK_KEY", max_retries=retries, trace_hook=traces.append,
        )
        critic = LLMCritic(backend, invalid_output_retries=3)
        if retries == 0:
            with pytest.raises(LLMFormatError):
                critic.propose("原任务", "回答甲", "回答乙", rubric, judgment, 1, conversation=history)
            assert len(requests) == 1  # adapter does not multiply exhausted format retries
            assert [m["role"] for m in history] == ["system", "user", "assistant"]
            assert history[-1]["content"] == "不是JSON的真实回复"
            proposal = critic.propose(
                "原任务", "回答甲", "回答乙", rubric, judgment, 2,
                invalid_error="格式重试耗尽", conversation=history,
            )
            feedback = requests[-1]["messages"][-1]["content"]
            assert "没有新的Judge判断" in feedback and "格式重试耗尽" in feedback
            assert "仍未成功" not in feedback
        else:
            proposal = critic.propose("原任务", "回答甲", "回答乙", rubric, judgment, 1, conversation=history)
            assert len(requests) == 2
            assert requests[1]["messages"][:2] == requests[0]["messages"]
            assert requests[1]["messages"][2] == {"role": "assistant", "content": "不是JSON的真实回复"}
            assert "输出格式无效" in requests[1]["messages"][-1]["content"]
        assert proposal.analysis == "新分析"
        assert history[-1] == {"role": "assistant", "content": raw}  # preserve raw spacing
        assert traces[-1]["messages"] == requests[-1]["messages"]


def test_conversation_is_sample_local_even_after_exhaustion(tmp_path):
    r = runner(tmp_path, "final_failure")
    snapshots = []
    original = r.modules.critic.propose

    def capture(*args, **kwargs):
        snapshots.append(deepcopy(kwargs["conversation"]))
        return original(*args, **kwargs)

    r.modules.critic.propose = capture
    for i in (1, 2):
        result = r.run_sample(replace(sample(f"s{i}"), task=f"独立任务{i}"))
        assert result.state == "validation_exhausted"
    assert [len(messages) for messages in snapshots] == [0, 3, 0, 3]
    assert "独立任务1" in snapshots[1][1]["content"]
    assert "独立任务1" not in str(snapshots[3])
    assert "独立任务2" in snapshots[3][1]["content"]


def test_invalid_generator_round_appends_unknown_result_feedback(tmp_path):
    r = runner(tmp_path, "attempt_2_success")
    apply = r.modules.initial_generator.apply
    histories = []
    propose = r.modules.critic.propose

    def fail_first(task, selected, candidate, number):
        if number == 1:
            raise ValueError("应用标准结构无效")
        return apply(task, selected, candidate, number)

    def capture(*args, **kwargs):
        result = propose(*args, **kwargs)
        histories.append(deepcopy(kwargs["conversation"]))
        return result

    r.modules.initial_generator.apply = fail_first
    r.modules.critic.propose = capture
    result = r.run_sample(sample())
    assert result.state == "application_success"
    assert result.application_attempts[0].judgment is None
    assert histories[1][:3] == histories[0]
    feedback = histories[1][3]["content"]
    assert "应用标准结构无效" in feedback and "没有新的Judge判断" in feedback
    assert "仍未成功" not in feedback
