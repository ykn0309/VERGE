"""Local IDs and actual HTTP messages validate E/C isolation without paid requests."""

import json
import httpx
import pytest
from verge.llm import OpenAICompatibleBackend
from verge.models import CandidateExperience, Criterion, CriterionJudgment, Rubric
from verge.modules import LLMInitialRubricGenerator, LLMCritic
from verge.prompts import ExperienceView
from verge.calculator import calculate_rubric_judgment


def make_backend(monkeypatch, handler, trace=None):
    monkeypatch.setenv("MOCK_KEY", "test-secret")
    client = httpx.Client(
        base_url="https://mock.invalid/v1/", transport=httpx.MockTransport(handler)
    )
    return (
        OpenAICompatibleBackend(
            "mock",
            base_url="https://mock.invalid/v1/",
            client=client,
            api_key_env="MOCK_KEY",
            max_retries=0,
            trace_hook=trace,
        ),
        client,
    )


def response(value):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}
            ]
        },
    )


def views():
    return [ExperienceView("exp_real", "已有指导", "已有条件", 0.7, 0.8)]


def candidate():
    return CandidateExperience(
        "candidate_current", "本轮指导", "本轮场景"
    )


def test_http_application_sources_retained_and_no_dialogue_history(
    monkeypatch, wire_payload
):
    bodies = []
    traces = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        p = wire_payload(body)
        assert set(p) == {"task", "selected_experiences", "candidate_experiences"}
        assert p["selected_experiences"][0]["experience_id"] == "1"
        assert p["candidate_experiences"][0]["experience_id"] == "2"
        assert set(p["candidate_experiences"][0]) == {
            "experience_id",
            "content",
            "trigger",
            "status",
        }
        assert len(body["messages"]) == 2
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert all(
            s not in str(body)
            for s in (
                "candidate_current",
                "exp_real",
                "chosen",
                "rejected",
                "private analysis",
                "revision_number",
            )
        )
        return response(
            {
                "analysis": "结合task设计标准",
                "criteria": [
                    {
                        "question": "准确？",
                        "weight": 2,
                        "source_experiences": ["1", "2"],
                    },
                    {
                        "question": "满足要求？",
                        "weight": 8,
                        "source_experiences": [],
                    },
                ],
            }
        )

    backend, client = make_backend(monkeypatch, handler, traces.append)
    try:
        gen = LLMInitialRubricGenerator(backend)
        a = gen.apply("任务", views(), [candidate()], 1)
        c2 = CandidateExperience(
            "candidate_new", "第二轮独立指导", "第二轮场景"
        )
        b = gen.apply("任务", views(), [c2], 2)
    finally:
        client.close()
    assert a.criteria[0].source_experiences == ["exp_real", "candidate_current"]
    assert b.criteria[0].source_experiences == ["exp_real", "candidate_new"]
    assert (
        a.rubric_id != b.rubric_id
        and a.criteria[0].criterion_id != b.criteria[0].criterion_id
    )
    assert traces[0]["output"]["criteria"][0]["source_experiences"] == ["1", "2"]
    assert json.loads(traces[0]["raw_output"])["criteria"][0]["source_experiences"] == [
        "1",
        "2",
    ]
    assert "test-secret" not in str(traces)
    assert bodies[0]["messages"][0] == bodies[1]["messages"][0]
    assert "本轮指导" not in bodies[1]["messages"][1]["content"]


@pytest.mark.parametrize("ids", [["3"], ["2", "2"], ["candidate_old"], [True]])
def test_invalid_application_sources_exhaust_without_clearing(
    monkeypatch, wire_payload, ids
):
    calls = []

    def handler(request):
        calls.append(wire_payload(json.loads(request.content)))
        return response(
            {
                "analysis": "任务分析",
                "criteria": [
                    {
                        "question": "要求？",
                        "weight": 2,
                        "source_experiences": ids,
                    },
                    {
                        "question": "完整？",
                        "weight": 8,
                        "source_experiences": [],
                    },
                ],
            }
        )

    backend, client = make_backend(monkeypatch, handler)
    try:
        with pytest.raises(ValueError):
            LLMInitialRubricGenerator(backend, invalid_output_retries=1).apply(
                "任务", views(), [candidate()], 1
            )
    finally:
        client.close()
    assert len(calls) == (1 if ids == [True] else 2)
    # bool is a backend shape failure; that layer owns its retry, no adapter stacking.
    if ids != [True]:
        assert "validation_feedback" in calls[-1]


def test_r0_cannot_reference_candidate(monkeypatch):
    backend, client = make_backend(
        monkeypatch,
        lambda _: response(
            {
                "analysis": "任务分析",
                "criteria": [
                    {
                        "question": "要求？",
                        "weight": 2,
                        "source_experiences": ["2"],
                    },
                    {
                        "question": "完整？",
                        "weight": 8,
                        "source_experiences": [],
                    },
                ],
            }
        ),
    )
    try:
        with pytest.raises(ValueError, match="unknown numbered experience"):
            LLMInitialRubricGenerator(backend, invalid_output_retries=0).generate(
                "任务", views()
            )
    finally:
        client.close()
