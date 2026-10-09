"""Local IDs and actual HTTP messages validate E/C isolation without paid requests."""

import json
import httpx
import pytest
from verge.llm import OpenAICompatibleBackend
from verge.models import CandidateExperience, Criterion, CriterionJudgment, Rubric
from verge.modules import LLMInitialRubricGenerator, LLMCritic, LLMAdmissionChecker
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
        "candidate_current", "本轮指导", "本轮场景", "PASS", "可用"
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
        assert set(p) == {"task", "selected_experiences", "candidate_experience"}
        assert p["selected_experiences"][0]["experience_id"] == "1"
        assert p["candidate_experience"]["experience_id"] == "2"
        assert set(p["candidate_experience"]) == {
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
                        "importance_score": 2,
                        "source_experiences": ["1", "2"],
                    },
                    {
                        "question": "满足要求？",
                        "importance_score": 8,
                        "source_experiences": [],
                    },
                ],
            }
        )

    backend, client = make_backend(monkeypatch, handler, traces.append)
    try:
        gen = LLMInitialRubricGenerator(backend)
        a = gen.apply("任务", views(), candidate(), 1)
        c2 = CandidateExperience(
            "candidate_new", "第二轮独立指导", "第二轮场景", "PASS", "可用"
        )
        b = gen.apply("任务", views(), c2, 2)
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
                        "importance_score": 2,
                        "source_experiences": ids,
                    },
                    {
                        "question": "完整？",
                        "importance_score": 8,
                        "source_experiences": [],
                    },
                ],
            }
        )

    backend, client = make_backend(monkeypatch, handler)
    try:
        with pytest.raises(ValueError):
            LLMInitialRubricGenerator(backend, invalid_output_retries=1).apply(
                "任务", views(), candidate(), 1
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
                        "importance_score": 2,
                        "source_experiences": ["2"],
                    },
                    {
                        "question": "完整？",
                        "importance_score": 8,
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


def test_critic_rejected_feedback_and_frozen_admission_http(monkeypatch, wire_payload):
    observed = []

    def handler(request):
        p = wire_payload(json.loads(request.content))
        observed.append(p)
        if "current_rubric" in p:
            assert p["rejected_candidate"]["content"] == "被拒原文"
            assert p["rejected_candidate"]["admission_reason"] == "具体拒绝原因"
            assert "analysis" not in p
            return response(
                {
                    "analysis": "私有事实核对",
                    "candidates": [{"content": "新指导", "trigger": "事前条件"}],
                }
            )
        assert set(p) == {"task", "candidate"} and set(p["candidate"]) == {
            "candidate_id",
            "content",
            "trigger",
        }
        return response({"decision": "PASS", "reason": "可用于生成"})

    backend, client = make_backend(monkeypatch, handler)
    r = Rubric("old_r", [Criterion("a", "条件", 2, []), Criterion("b", "要求", 8, [])])
    j = calculate_rubric_judgment(
        r, [CriterionJudgment(c.criterion_id, False, False) for c in r.criteria]
    )
    rejected = CandidateExperience(
        "rejected_c", "被拒原文", "事后条件", "FAIL", "具体拒绝原因"
    )
    try:
        proposal = LLMCritic(backend).propose(
            "任务", "A", "B", r, j, 1, 2, rejected_candidate=rejected
        )
        approved = LLMAdmissionChecker(backend).check("任务", proposal.candidate)
    finally:
        client.close()
    assert approved.content == "新指导" and approved.candidate_id.startswith(
        "candidate_"
    )
    assert proposal.candidate.admission is None and approved.admission == "PASS"
    assert len(observed) == 2
