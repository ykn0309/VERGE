"""Current HTTP transport contracts, tested without sending any API calls."""

import json
import httpx
import pytest
from verge.llm import (
    OpenAICompatibleBackend,
    OpenAICompatibleEmbedding,
    LLMRequestError,
    LLMFormatError,
)
from verge.models import Rubric, Criterion
from verge.modules import LLMGenerator


def backend(monkeypatch, handler, **options):
    monkeypatch.setenv("VERGE_TEST_TOKEN", "offline-token")
    client = httpx.Client(
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return OpenAICompatibleBackend(
        "fixture",
        base_url="https://example.test/v1",
        api_key_env="VERGE_TEST_TOKEN",
        client=client,
        max_retries=0,
        enable_thinking=False,
        **options,
    )


def response(value, **extra):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}
            ],
            **extra,
        },
    )


def test_json_output_and_actual_cache_metadata_are_retained(monkeypatch):
    events = []

    def handler(request):
        payload = json.loads(request.content)
        assert payload["enable_thinking"] is False
        assert payload["messages"][0]["role"] == "system"
        return response(
            {"selected_experience_ids": [], "selection_reasons": []},
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        )

    b = backend(monkeypatch, handler, metadata_hook=events.append)
    try:
        result = b.generate(
            "selector", {"task": "task", "candidates": [], "max_selected": 5}
        )
        assert result["selected_experience_ids"] == []
        assert events and "offline-token" not in json.dumps(events)
        assert events[-1]["usage"]["prompt_tokens"] == 10
    finally:
        b.close()


def test_auth_error_is_fatal_instead_of_a_bad_training_observation(monkeypatch):
    b = backend(
        monkeypatch, lambda request: httpx.Response(401, json={"error": "unauthorized"})
    )
    try:
        with pytest.raises(LLMRequestError):
            b.generate(
                "selector", {"task": "task", "candidates": [], "max_selected": 5}
            )
    finally:
        b.close()


def test_invalid_role_is_rejected_before_network(monkeypatch):
    def handler(request):
        raise AssertionError("unexpected request")

    b = backend(monkeypatch, handler)
    try:
        with pytest.raises(ValueError, match="unknown"):
            b.generate("experience_updater", {})
    finally:
        b.close()


def test_format_retry_keeps_task_experience_and_modified_guide(monkeypatch):
    bodies = []

    def handler(request):
        payload = json.loads(request.content)
        bodies.append(json.loads(payload["messages"][-1]["content"]))
        if len(bodies) == 1:
            return response({"analysis": "x", "criteria": []})
        return response(
            {
                "analysis": "x",
                "criteria": [
                    {
                        "question": "核心要求是否满足？",
                        "weight": 1,
                    }
                ],
            }
        )

    b = backend(monkeypatch, handler)
    try:
        rubric = LLMGenerator(b, retries=1).revise(
            "原任务", Rubric("r", [Criterion("c", "初稿", 1)]), []
        )
        assert rubric.criteria and len(bodies) == 2
        assert bodies[0]["task"] == bodies[1]["task"] == "原任务"
        assert bodies[1]["experiences"] == [] and "validation_feedback" in bodies[1]
    finally:
        b.close()


def test_real_http_transport_does_not_append_critic_history_twice(monkeypatch):
    from verge.modules import LLMCritic, CriticSession

    messages = []

    def handler(request):
        messages.append(json.loads(request.content)["messages"])
        return response({"analysis": "没有足够依据提出经验", "candidates": []})

    b = backend(monkeypatch, handler)
    try:
        session = CriticSession()
        critic = LLMCritic(b)
        critic.propose(
            {"task": "唯一初始任务"},
            source_task_id="t",
            allowed_parent_ids=[],
            session=session,
        )
        assert [m["role"] for m in session.history] == ["user", "assistant"]
        critic.propose(
            {"previous_attempts": []},
            source_task_id="t",
            allowed_parent_ids=[],
            session=session,
        )
        assert [m["role"] for m in messages[1]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert sum("唯一初始任务" in m["content"] for m in messages[1]) == 1
        assert [m["role"] for m in session.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
    finally:
        b.close()
