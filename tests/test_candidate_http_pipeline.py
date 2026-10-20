"""Worst business path through production HTTP adapters, with immutable raw model traces."""

import json
from threading import Lock
from experience_helpers import seed
import httpx
import pytest
from verge.llm import OpenAICompatibleBackend
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, Credit
from verge.modules import make_llm_modules
from verge.pipeline import PipelineRunner
from test_pipeline import sample
from test_numbered_ids import response


def test_http_two_candidates_fresh_applications_single_candidate_admission(
    monkeypatch, tmp_path, wire_payload
):
    monkeypatch.setenv("MOCK_KEY", "test-secret")
    lock = Lock()
    counts = {"critic": 0, "generator": 0}
    requests = []
    traces = []

    def handler(request):
        body = json.loads(request.content)
        p = wire_payload(body)
        with lock:
            requests.append((body, p))
        if "draft" in p:
            return response({"experiences":[p["draft"]]})
        if "max_selected" in p:
            return response({"selected_experience_ids": ["1"]})
        if "current_rubric" in p:
            counts["critic"] += 1
            n = counts["critic"]
            assert p["round_number"] == n
            for judgment_key, prefix in [("current_judgment", f"R{n-1}")]+(
                [("previous_judgment", "R0")] if n == 2 else []
            ):
                for i, verdict in enumerate(p[judgment_key]["criterion_judgments"], 1):
                    for side in ("chosen", "rejected"):
                        expected = f"JUDGE_EVIDENCE:answer {side} private:{prefix}问题{i}"
                        assert verdict[f"{side}_reason"] == expected
                        assert expected in str(body["messages"])
            if n == 2:
                assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
                first = [b for b, p in requests if "current_rubric" in p][0]
                assert body["messages"][:2] == first["messages"]
                assert body["messages"][2]["content"] == next(
                    t["raw_output"] for t in traces if t["role"] == "critic"
                )
                feedback = body["messages"][-1]["content"]
                assert "仍未成功" in feedback and "重新分析" in feedback
                assert "R0问题" not in feedback and "\n优选回答：" not in feedback
                assert p["adopted"] is True
                assert p["previous_candidate"]["content"] == "指导1"
                assert p["atomic_feedback"]["atomic_candidates"][0]["candidate_id"] in p["current_rubric"]["criteria"][0]["source_experiences"]
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
                "candidate_experiences",
            }
            assert set(p["selected_experiences"][0]) == {"experience_id", "content", "trigger"}
            assert "证据量" not in body["messages"][1]["content"]
            assert all(
                s not in str(body)
                for s in ("PRIVATE", "answer chosen private", "answer rejected private", "JUDGE_EVIDENCE")
            )
            user = body["messages"][1]["content"]
            assert user.count("本次可用经验") == 1
            assert "本次选中的经验" not in user
            assert "admission" not in p
            if n:
                assert "当前候选、尚未验证，无历史置信度" in user
                assert f"2：指导{n}" in user
                assert p["candidate_experiences"][0]["content"] == f"指导{n}"
                assert "confidence" not in p["candidate_experiences"]
                if n == 2:
                    assert "指导1" not in str(body)
                assert f"R{n-1}问题" not in str(body)
            return response(
                {
                    "analysis": "task分析",
                    "criteria": [
                        {
                            "question": f"R{n}问题1",
                            "weight": .2,
                            "source_experiences": ["1", "2"] if n else ["1"],
                        },
                        {
                            "question": f"R{n}问题2",
                            "weight": .8,
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
                        {"reason": f"JUDGE_EVIDENCE:{p['response']}:{x['question']}",
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
        pytest.fail("unexpected extra LLM role")

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
        seeded = seed(memory,
            CandidateExperience("seed_c", "种子指导", "*"), "seed"
        )
        rec = PipelineRunner(
            make_llm_modules(backend), memory, TrajectoryLog(tmp_path)
        ).run_sample(sample())
    assert counts == {"critic": 2, "generator": 3}
    saved = list(TrajectoryLog(tmp_path).all())[0]
    assert saved.to_dict() == rec.to_dict()
    assert saved.schema_version == 10
    for judgment, prefix in [(rec.initial_judgment, "R0")]+[
        (a.judgment, f"R{i}") for i, a in enumerate(rec.application_attempts, 1)
    ]:
        for i, verdict in enumerate(judgment.criterion_judgments, 1):
            assert verdict.chosen_reason == f"JUDGE_EVIDENCE:answer chosen private:{prefix}问题{i}"
            assert verdict.rejected_reason == f"JUDGE_EVIDENCE:answer rejected private:{prefix}问题{i}"
    assert rec.state == "application_success" and len(rec.application_attempts) == 2
    assert rec.application_attempts[0].credit == Credit(0, 0.1)
    assert rec.application_attempts[1].credit == Credit(0.8, 0.1)
    assert rec.proposal_attempts[-1].candidate.content == "指导2"
    assert rec.admitted_candidate_ids == [rec.proposal_attempts[-1].atomic_candidates[0].candidate_id]
    e = memory.get(seeded.experience_id)
    assert (e.alpha, e.beta) == pytest.approx((1.2, 1.2))
    assert len(e.evidence_history) == 2  # original seed and R0 only
    new = memory.get(rec.admitted_experience_ids[0])
    assert (new.alpha, new.beta) == pytest.approx((1.8, 1.1))
    assert new.cases[0].credit == Credit(.8, .1)
    assert all("test-secret" not in str(t) for t in traces)
    rawgens = [t for t in traces if t["role"] == "initial_generator"]
    assert json.loads(rawgens[-1]["raw_output"])["criteria"][0][
        "source_experiences"
    ] == ["1", "2"]
    assert rawgens[-1]["output"]["criteria"][0]["source_experiences"] == ["1", "2"]
    assert rec.application_attempts[-1].rubric.criteria[0].source_experiences == [
        seeded.experience_id,
        rec.application_attempts[-1].candidate_id,
    ]
    critic_messages = [
        "\n".join(m["content"] for m in b["messages"])
        for b, p in requests if "current_rubric" in p
    ]
    assert (
        critic_messages[1].count("：R0问题1\n") == 1
        and critic_messages[1].count("：R1问题1\n") == 1
    )
    assert json.loads([b for b, p in requests if "current_rubric" in p][1]["messages"][2]["content"])["analysis"] == "PRIVATE诊断1"
    # Check actual messages across selector, generator, both judges and Critic history.
    internal_ids = {seeded.experience_id, *rec.admitted_experience_ids}
    internal_ids.update(p.candidate.candidate_id for p in rec.proposal_attempts)
    for rubric in [rec.initial_rubric, *(a.rubric for a in rec.application_attempts)]:
        internal_ids.add(rubric.rubric_id)
        internal_ids.update(c.criterion_id for c in rubric.criteria)
    assert {t['role'] for t in traces} == {'selector', 'initial_generator', 'judge', 'critic', 'atomizer'}
    assert all(real_id not in json.dumps(body['messages'], ensure_ascii=False)
               for body, _ in requests for real_id in internal_ids)
