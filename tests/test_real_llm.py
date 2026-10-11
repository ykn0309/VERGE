"""真实 backend 用 MockTransport 测试；不调用外网、不依赖真实凭据。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from verge.cli import main
from verge.llm import JUDGE_RESPONSE_SIDE, FakeLLMBackend, LLMFormatError, LLMRequestError, OpenAICompatibleBackend
from verge.prompts import PROMPT_VERSION, _ROLE_GUIDANCE, _ROLE_INSTRUCTIONS, llm_messages


def result(value, *, status=200, reason="stop", headers=None):
    return httpx.Response(status, headers=headers, json={
        "choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)},
                     "finish_reason": reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    })


def numbered_fake_response(fake, role, payload):
    """MockTransport 的模型桩遵循 v9 数字协议；Fake 本体仍用于内部 ID 测试。"""
    if role == "judge":
        internal = {
            **payload,
            "criteria": [
                {
                    **item,
                    "criterion_id": (
                        "c2"
                        if item["question"] == "是否逐项满足关键要求？"
                        else "c" + item["criterion_id"]
                    ),
                }
                for item in payload["criteria"]
            ],
        }
        result_value = fake.generate(role, internal)
        for answer, requested in zip(
            result_value["criterion_judgments"], payload["criteria"]
        ):
            answer["criterion_id"] = requested["criterion_id"]
        return result_value
    result_value = fake.generate(role, payload)
    if role == "initial_generator":
        result_value = {
            "analysis": "根据任务拆解核心要求。",
            "criteria": [
                {k: v for k, v in item.items() if k != "criterion_id"}
                for item in result_value["rubric"]["criteria"]
            ],
        }
        return result_value


def backend(monkeypatch, handler, **options):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-log")
    client = httpx.Client(base_url="https://api.openai.com/v1/",
                          transport=httpx.MockTransport(handler))
    return OpenAICompatibleBackend("mock-model", client=client,
                                   sleep=options.pop("sleep", lambda _: None), **options)


def test_structured_success_metadata_prompt_version_and_privacy(monkeypatch):
    events = []
    observed = []

    def handler(request):
        observed.append(request)
        return result(
            {
                "analysis": "诊断覆盖不足",
                "candidates": [
                    {"content": "按任务要求完善标准", "trigger": "任务条件"}
                ],
            },
            headers={"x-request-id": "req-123"},
        )

    llm = backend(monkeypatch, handler, metadata_hook=events.append)
    assert llm.generate("critic", {"task": "private task"}) == {
        "analysis": "诊断覆盖不足",
        "candidates": [{"content": "按任务要求完善标准", "trigger": "任务条件"}],
    }
    assert observed[0].url.path == "/v1/chat/completions"
    assert observed[0].headers["authorization"] == "Bearer test-secret-never-log"
    sent = json.loads(observed[0].content)
    assert sent["model"] == "mock-model"
    assert sent["response_format"] == {"type": "json_object"}
    assert len(sent["messages"]) == 2
    assert "任务：" in sent["messages"][1]["content"]
    assert "private task" in sent["messages"][1]["content"]
    assert events == [
        {
            "role": "critic",
            "model": "mock-model",
            "prompt_version": PROMPT_VERSION,
            "attempt": 1,
            "status": 200,
            "request_id": "req-123",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        }
    ]
    assert "test-secret-never-log" not in json.dumps(events)

    # 即使不可信服务把凭据反射进头部/元数据，也不能写入本地 sidecar。
    reflected = []
    echo = backend(
        monkeypatch,
        lambda _: result(
            {
                "analysis": "诊断覆盖不足",
                "candidates": [
                    {"content": "按任务要求完善标准", "trigger": "任务条件"}
                ],
            },
            reason="test-secret-never-log",
            headers={"x-request-id": "test-secret-never-log"},
        ),
        metadata_hook=reflected.append,
    )
    assert echo.generate("critic", {"task": "x"}) == {
        "analysis": "诊断覆盖不足",
        "candidates": [{"content": "按任务要求完善标准", "trigger": "任务条件"}],
    }
    assert "test-secret-never-log" not in json.dumps(reflected)


@pytest.mark.parametrize(
    "role, output, expected",
    [
        (
            "selector",
            {"selected_experience_ids": [1, 2]},
            {"selected_experience_ids": ["1", "2"]},
        ),
        (
            "initial_generator",
            {
                "analysis": "任务分析",
                "criteria": [
                    {
                        "question": "准则？",
                        "weight": 8,
                        "source_experiences": [3],
                    }
                ],
            },
            {
                "analysis": "任务分析",
                "criteria": [
                    {
                        "question": "准则？",
                        "weight": 8,
                        "source_experiences": ["3"],
                    }
                ],
            },
        ),
        (
            "judge",
            {"criterion_judgments": [{"reason": "fixture evidence", "criterion_id": 1, "satisfied": True}]},
            {"criterion_judgments": [{"reason": "fixture evidence", "criterion_id": "1", "satisfied": True}]},
        ),
        (
            "critic",
            {
                "analysis": "私有诊断",
                "candidates": [{"content": "经验", "trigger": "条件"}],
            },
            {
                "analysis": "私有诊断",
                "candidates": [{"content": "经验", "trigger": "条件"}],
            },
        ),
    ],
)
def test_model_can_return_positive_integer_ids_for_any_role(
    monkeypatch, role, output, expected
):
    llm = backend(monkeypatch, lambda _: result(output), max_retries=0)
    assert llm.generate(role, {"task": "x"}) == expected


def test_format_retry_on_missing_nested_field_and_limit(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        value = (
            {"criterion_judgments": [{"criterion_id": "c1"}]}
            if len(calls) == 1
            else {"criterion_judgments": [{"reason": "fixture evidence", "criterion_id": "c1", "satisfied": True}]}
        )
        return result(value)

    llm = backend(monkeypatch, handler, max_retries=1)
    assert (
        llm.generate("judge", {"criteria": [{"criterion_id": "c1"}]})[
            "criterion_judgments"
        ][0]["satisfied"]
        is True
    )
    assert len(calls) == 2 and "上次输出格式无效" in calls[1]["messages"][-1]["content"]

    bad = backend(monkeypatch, lambda _: result({"wrong": 1}), max_retries=1)
    with pytest.raises(LLMFormatError, match="after 2 attempts"):
        bad.generate("critic", {"task": "x"})

    # 多余的模型字段也是格式错误（领域 Model.from_dict 会严格拒绝）。
    extra = backend(
        monkeypatch,
        lambda _: result(
            {
                "candidates": [
                    {
                        "candidate_id": "candidate_1",
                        "content": "建议",
                        "trigger": "条件",
                        "credit": 0.9,
                    }
                ]
            }
        ),
        max_retries=1,
    )
    with pytest.raises(LLMFormatError, match="after 2 attempts"):
        extra.generate("critic", {"task": "x"})


def test_invalid_json_retries_then_succeeds(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "not JSON"}, "finish_reason": "stop"}
                    ]
                },
            )
        return result(
            {
                "analysis": "诊断覆盖不足",
                "candidates": [
                    {"content": "按任务要求完善标准", "trigger": "任务条件"}
                ],
            }
        )

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate("critic", {"task": "x"}) == {
        "analysis": "诊断覆盖不足",
        "candidates": [{"content": "按任务要求完善标准", "trigger": "任务条件"}],
    }
    assert len(calls) == 2


def test_rate_limit_timeout_and_terminal_errors(monkeypatch):
    waits = []
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="private error body")
        if len(attempts) == 2:
            raise httpx.ReadTimeout("private network details")
        return result({"analysis": "证据不足", "candidates": []})

    llm = backend(monkeypatch, handler, max_retries=2, sleep=waits.append)
    assert llm.generate("critic", {"task": "任务"}) == {"analysis": "证据不足", "candidates": []}
    assert len(attempts) == 3 and waits == [0.0, 2]
    assert len(attempts) == 3  # 空候选是合法结果，不重试。

    unauthorized = backend(monkeypatch, lambda _: httpx.Response(401, text="test-secret-never-log"))
    with pytest.raises(LLMRequestError, match="HTTP 401") as exc:
        unauthorized.generate("critic", {"task": "x"})
    assert "test-secret-never-log" not in str(exc.value)

    exhausted = []

    def unavailable(request):
        exhausted.append(request)
        return httpx.Response(503, text="private service body")

    flaky = backend(monkeypatch, unavailable, max_retries=1)
    with pytest.raises(LLMRequestError, match="HTTP 503 after 2 attempts"):
        flaky.generate("critic", {"task": "x"})
    assert len(exhausted) == 2




def test_candidate_is_not_a_format_failure(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return result(
            {
                "analysis": "私有诊断",
                "candidates": [{"content": "方法", "trigger": "条件"}],
            }
        )

    llm = backend(monkeypatch, handler)
    assert llm.generate("critic", {"task": "任务"}) == {
        "analysis": "私有诊断",
        "candidates": [{"content": "方法", "trigger": "条件"}],
    }
    assert len(calls) == 1


def test_missing_key_and_unsafe_url_fail_before_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing"):
        OpenAICompatibleBackend("mock-model")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with pytest.raises(ValueError, match="base_url requires HTTPS"):
        OpenAICompatibleBackend("mock-model", base_url="http://remote.example/v1")
    with pytest.raises(ValueError, match="base_url requires HTTPS"):
        OpenAICompatibleBackend(
            "mock-model", base_url="https://user:secret@api.example/v1"
        )
    # 测试注入的 client 也不能绕过目标地址检查或自动转发凭据。
    with httpx.Client(base_url="http://remote.example/v1") as unsafe:
        with pytest.raises(ValueError, match="client must use"):
            OpenAICompatibleBackend("mock-model", client=unsafe)
    with httpx.Client(
        base_url="https://api.openai.com/v1", follow_redirects=True
    ) as redirecting:
        with pytest.raises(ValueError, match="client must use"):
            OpenAICompatibleBackend("mock-model", client=redirecting)
    with pytest.raises(ValueError, match="enable_thinking must be Boolean"):
        backend(
            monkeypatch,
            lambda _: result(
                {
                    "analysis": "诊断覆盖不足",
                    "candidates": [
                        {"content": "按任务要求完善标准", "trigger": "任务条件"}
                    ],
                }
            ),
            enable_thinking="false",
        )


def test_role_trace_preserves_actual_prompt_response_and_format_retry(monkeypatch):
    trace = []
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "not JSON"}, "finish_reason": "stop"}
                    ]
                },
            )
        return result(
            {
                "analysis": "诊断覆盖不足",
                "candidates": [
                    {"content": "按任务要求完善标准", "trigger": "任务条件"}
                ],
            }
        )

    llm = backend(monkeypatch, handler, max_retries=1, trace_hook=trace.append)
    assert llm.generate("critic", {"task": "private task"}) == {
        "analysis": "诊断覆盖不足",
        "candidates": [{"content": "按任务要求完善标准", "trigger": "任务条件"}],
    }
    assert len(trace) == 2
    assert trace[0]["input"] == {"task": "private task"}
    assert trace[0]["messages"] == calls[0]["messages"]
    assert (
        trace[0]["raw_output"] == "not JSON"
        and trace[0]["error"] == "invalid structured output"
    )
    assert trace[1]["messages"] == calls[1]["messages"]
    assert "上次输出格式无效" in trace[1]["messages"][-1]["content"]
    assert json.loads(trace[1]["raw_output"]) == trace[1]["output"]
    assert trace[1]["output"] == {
        "analysis": "诊断覆盖不足",
        "candidates": [{"content": "按任务要求完善标准", "trigger": "任务条件"}],
    }
    assert "test-secret-never-log" not in json.dumps(trace, ensure_ascii=False)


def test_role_trace_does_not_save_http_error_bodies_or_headers(monkeypatch):
    trace = []
    llm = backend(monkeypatch, lambda _: httpx.Response(401, text="secret error body",
                   headers={"x-request-id": "private-id"}), trace_hook=trace.append)
    with pytest.raises(LLMRequestError, match="HTTP 401"):
        llm.generate("critic", {"task": "x"})
    assert trace[0]["raw_output"] is None and trace[0]["error"] == "HTTP 401"
    assert "secret error body" not in json.dumps(trace) and "private-id" not in json.dumps(trace)
