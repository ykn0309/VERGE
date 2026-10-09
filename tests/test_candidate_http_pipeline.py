"""Worst business path through production HTTP adapters, with immutable raw model traces."""

import json
from threading import Lock
import httpx
import pytest
from verge.llm import OpenAICompatibleBackend
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, Credit
from verge.modules import make_llm_modules
from verge.pipeline import PipelineRunner
from test_pipeline import sample
from test_numbered_ids import response


def test_http_four_candidates_two_fresh_applications_exact_candidate_fusion(
    monkeypatch, tmp_path, wire_payload
):
    monkeypatch.setenv("MOCK_KEY", "test-secret")
    lock = Lock()
    counts = {"critic": 0, "admission_checker": 0, "generator": 0}
    requests = []
    traces = []

    def handler(request):
        body = json.loads(request.content)
        p = wire_payload(body)
        with lock:
            requests.append((body, p))
        if "max_selected" in p:
            return response({"selected_experience_ids": ["1"]})
        if "current_rubric" in p:
            counts["critic"] += 1
            n = counts["critic"]
            assert p["generation_number"] == (1 if n % 2 else 2)
            if n in (2, 4):
                assert p["rejected_candidate"]["content"] == f"指导{n-1}"
                assert p["rejected_candidate"]["admission_reason"] == f"拒绝原因{n-1}"
            if n == 3:
                assert p["adopted"] is True
                assert p["previous_candidate"]["content"] == "指导2"
                assert p["previous_candidate"]["candidate_id"] in p["current_rubric"]["criteria"][0]["source_experiences"]
                assert "previous_rubric" in p and "previous_judgment" in p
                assert p["previous_rubric"]["criteria"][0]["question"] == "R0问题1"
                assert p["current_rubric"]["criteria"][0]["question"] == "R1问题1"
                assert "analysis" not in p
            return response(
                {
                    "analysis": f"PRIVATE诊断{n}",
                    "candidates": [{"content": f"指导{n}", "trigger": "事前任务条件"}],
                }
            )
        if "selected_experiences" in p:
            n = counts["generator"]
            counts["generator"] += 1
            assert len(body["messages"]) == 2 and set(p) <= {
                "task",
                "selected_experiences",
                "candidate_experience",
            }
            assert (
                p["selected_experiences"][0]["confidence"] == 0.5
            )  # fixed before R0 update
            assert all(
                s not in str(body)
                for s in ("PRIVATE", "answer chosen private", "answer rejected private")
            )
            if n:
                assert p["candidate_experience"]["content"] == f"指导{2*n}"
                assert "confidence" not in p["candidate_experience"]
                assert "指导1" not in str(body) and "指导3" not in str(body)
                assert f"R{n-1}问题" not in str(body)
            return response(
                {
                    "analysis": "task分析",
                    "criteria": [
                        {
                            "question": f"R{n}问题1",
                            "importance_score": 2,
                            "source_experiences": ["1", "2"] if n else ["1"],
                        },
                        {
                            "question": f"R{n}问题2",
                            "importance_score": 8,
                            "source_experiences": ["2"] if n else [],
                        },
                    ],
                }
            )
        if "response" in p:
            preferred = p["response"] == "answer chosen private"
            return response(
                {
                    "criterion_judgments": [
                        {
                            "criterion_id": x["criterion_id"],
                            "satisfied": (
                                x["question"] == "R2问题2"
                                if preferred
                                else x["question"].endswith("问题1")
                            ),
                        }
                        for x in p["criteria"]
                    ]
                }
            )
        if "candidate" in p and "existing" not in p:
            n = counts["admission_checker"]
            counts["admission_checker"] += 1
            assert set(p) == {"task", "candidate"}
            return response(
                {
                    "decision": "FAIL" if n % 2 == 0 else "PASS",
                    "reason": f"拒绝原因{n+1}" if n % 2 == 0 else "候选表达可用",
                }
            )
        assert (
            set(p) == {"candidate", "existing"} and p["candidate"]["content"] == "指导4"
        )
        assert p["existing"][0]["content"] == "种子指导"
        return response(
            {"match_index": 0, "content": "融合指导4", "trigger": "兼容事前条件"}
        )

    with httpx.Client(
        base_url="https://mock.invalid/v1/", transport=httpx.MockTransport(handler)
    ) as client:
        backend = OpenAICompatibleBackend(
            "mock",
            base_url="https://mock.invalid/v1/",
            client=client,
            api_key_env="MOCK_KEY",
            max_retries=0,
            trace_hook=traces.append,
        )
        memory = ExperienceMemory(tmp_path)
        seed = memory.fuse(
            CandidateExperience("seed_c", "种子指导", "*", "PASS", "种子"), "seed"
        )
        rec = PipelineRunner(
            make_llm_modules(backend), memory, TrajectoryLog(tmp_path)
        ).run_sample(sample())
    assert counts == {"critic": 4, "admission_checker": 4, "generator": 3}
    assert rec.state == "application_success" and len(rec.application_attempts) == 2
    assert rec.application_attempts[0].credit == Credit(0, 0.1)
    assert rec.application_attempts[1].credit == Credit(0.8, 0.1)
    assert rec.proposal_attempts[-1].candidate.content == "指导4"
    assert rec.proposal_attempts[-1].candidate.fusion == "MERGE"
    e = memory.get(seed.experience_id)
    assert (e.alpha, e.beta) == pytest.approx((1.8, 1.3))
    assert len(e.evidence_history) == 2  # R0 once, successful current candidate once
    assert e.fusion_history[-1].credit == Credit(0.8, 0.1)
    assert all("test-secret" not in str(t) for t in traces)
    rawgens = [t for t in traces if t["role"] == "initial_generator"]
    assert json.loads(rawgens[-1]["raw_output"])["criteria"][0][
        "source_experiences"
    ] == ["1", "2"]
    assert rawgens[-1]["output"]["criteria"][0]["source_experiences"] == ["1", "2"]
    assert rec.application_attempts[-1].rubric.criteria[0].source_experiences == [
        seed.experience_id,
        rec.application_attempts[-1].candidate_id,
    ]
    critic_messages = [
        b["messages"][1]["content"] for b, p in requests if "current_rubric" in p
    ]
    assert (
        critic_messages[2].count("R0问题1") == 1
        and critic_messages[2].count("R1问题1") == 1
    )
    assert "PRIVATE诊断" not in critic_messages[2]
