from copy import deepcopy
import json
import pytest
from verge.models import Experience, ExperienceReview
from verge.modules import make_fake_modules, LLMCritic, LLMExperienceChecker
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.pipeline import PipelineRunner
from verge.prompts import NumberedIDCodec
from verge.llm import FakeLLMBackend, LLMFormatError
from test_v2 import sample, rubric, judged


def test_initial_review_merged_and_no_repair_checker_or_source():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        m = make_fake_modules()
        mem = ExperienceMemory(root)

        def forbidden(*args, **kwargs):
            raise AssertionError("repair Judge Checker removed")

        m.checkers.check_judgment = forbidden
        r = PipelineRunner(
            m, mem, TrajectoryLog(root), revision_enabled=False
        ).run_sample(sample())
        roles = [role for role, _ in m.backend.calls]
        assert roles.count("critic_repair") == roles.count("critic_assess") == 1
        assert roles.count("judge") == 4 and roles.count("initial_generator") == 1
        assert "judge_checker" not in roles
        assert r.baseline_judgment and r.initial_checker is None
        assert r.proposal_attempts[0].repair_checker is None
        assert roles[-1] == "experience_checker_batch"


def test_second_attempt_keeps_one_task_copy_and_fixed_baseline(tmp_path):
    m = make_fake_modules("attempt_2_success")
    events = []
    m.backend.trace_hook = events.append
    r = PipelineRunner(
        m, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    calls = [x for x in events if x.get("role") == "critic_repair" and "messages" in x]
    assert len(calls) == 2
    second = calls[-1]
    text = "\n".join(x["content"] for x in second["messages"])
    assert text.count('"' + sample().task + '"') == 1
    assert text.count('"' + sample().chosen + '"') == 1
    assert "task" not in second["input"] and "rubric" not in second["input"]
    assert [a.margin_before for a in r.proposal_attempts] == [
        r.baseline_judgment.margin
    ] * 2
    assert "checker" not in second["input"]["previous_feedback"]


def test_three_extracted_candidates_are_not_silently_truncated():
    class TooMany(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            d = super().generate(role, payload, **kw)
            if role == "critic_extract":
                d["candidates"] = [deepcopy(d["candidates"][0]) for _ in range(3)]
            return d

    r = rubric()
    with pytest.raises(LLMFormatError, match="0..2"):
        LLMCritic(TooMany(), retries=0).extract(
            sample(), r, judged(r), [{"change": "c", "basis": "b"}], {}
        )


def test_batch_reviews_two_then_rewrites_only_rejected_and_no_source(tmp_path):
    m = make_fake_modules()
    original = m.critic.extract

    def extract(*args, **kw):
        xs = original(*args, **kw)
        e, refs, reason = deepcopy(xs[0])
        e.experience_id = "bad"
        e.method.principle = "第二个独立方法"
        return xs + [(e, refs, reason)]

    m.critic.extract = extract
    calls = []

    def check(s, ev, es):
        calls.append([e.experience_id for e in es])
        return {
            e.experience_id: ExperienceReview(
                e.experience_id != "bad" or len(calls) == 2, "逐条核对"
            )
            for e in es
        }

    m.experience_checker.check_many = check
    mem = ExperienceMemory(tmp_path)
    r = PipelineRunner(
        m, mem, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    assert len(calls[0]) == 2 and calls[1] == ["bad"] and len(mem.all()) == 2
    rewrites = [p for role, p in m.backend.calls if role == "critic_rewrite_batch"]
    assert len(rewrites) == 1 and len(rewrites[0]["candidates"]) == 1
    assert all(not e.applications for e in mem.all())


def test_first_pass_approval_survives_other_candidate_rewrite_failure(tmp_path):
    m = make_fake_modules()
    orig = m.critic.extract

    def extract(*a, **kw):
        xs = orig(*a, **kw)
        e, refs, reason = deepcopy(xs[0])
        e.experience_id = "bad"
        e.method.principle = "第二条"
        return xs + [(e, refs, reason)]

    m.critic.extract = extract
    m.experience_checker.check_many = lambda s, ev, es: {
        e.experience_id: ExperienceReview(e.experience_id != "bad", "逐条结论")
        for e in es
    }

    def fail(*a, **kw):
        raise LLMFormatError("rewrite transport failed")

    m.critic.rewrite_many = fail
    mem = ExperienceMemory(tmp_path)
    r = PipelineRunner(
        m, mem, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    assert len(mem.all()) == 1 and r.state == "experience_admitted"
    assert r.proposal_attempts[0].extractions[1]["error"]


@pytest.mark.parametrize("ids", [["1", "1"], ["1"], ["1", "3"]])
def test_batch_review_requires_complete_distinct_whitelisted_ids(ids):
    class B:
        def generate(self, *args, **kw):
            return {
                "reviews": [
                    {"experience_id": i, "approved": True, "reason": "ok"} for i in ids
                ]
            }

    from test_v2 import experience

    with pytest.raises(LLMFormatError):
        LLMExperienceChecker(B(), retries=0).check_many(
            sample(), {}, [experience("first"), experience("second")]
        )


def test_cache_metadata_uses_actual_provider_report(monkeypatch):
    import httpx
    from verge.llm import OpenAICompatibleBackend

    monkeypatch.setenv("VERGE_TEST_KEY", "test")
    client = httpx.Client(
        base_url="https://example.test/v1/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "{}"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "total_tokens": 101,
                        "prompt_tokens_details": {"cached_tokens": 64},
                    },
                },
            )
        ),
    )
    events = []
    b = OpenAICompatibleBackend(
        "test",
        base_url="https://example.test/v1",
        api_key_env="VERGE_TEST_KEY",
        client=client,
        enable_thinking=False,
        metadata_hook=events.append,
    )
    b.generate("critic_assess", {})
    assert events[0]["cache"] == {"reported": True, "cached_input_tokens": 64}
    client.close()


def test_empty_extraction_stops_without_batch_review(tmp_path):
    m = make_fake_modules()
    m.critic.extract = lambda *a, **kw: []
    mem = ExperienceMemory(tmp_path)
    r = PipelineRunner(
        m, mem, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    assert r.state == "no_admitted_experience" and not mem.all()
    assert not any(role == "experience_checker_batch" for role, _ in m.backend.calls)


def test_batch_rewrite_cannot_drop_or_add_ids():
    class B:
        def generate(self, *a, **kw):
            return {"rewrites": []}

    from test_v2 import experience

    e = experience()
    with pytest.raises(LLMFormatError, match="cover every"):
        LLMCritic(B(), retries=0).rewrite_many(
            [e], {e.experience_id: ExperienceReview(False, "错误")}, {}
        )


def test_unknown_cache_is_not_reported_as_zero_hits(monkeypatch):
    import httpx
    from verge.llm import OpenAICompatibleBackend

    monkeypatch.setenv("VERGE_TEST_KEY", "test")
    client = httpx.Client(
        base_url="https://example.test/v1/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "{}"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "total_tokens": 101,
                    },
                },
            )
        ),
    )
    events = []
    b = OpenAICompatibleBackend(
        "test",
        base_url="https://example.test/v1",
        api_key_env="VERGE_TEST_KEY",
        client=client,
        enable_thinking=False,
        metadata_hook=events.append,
    )
    b.generate("critic_assess", {})
    assert events[0]["cache"] == {"reported": False, "cached_input_tokens": None}
    client.close()


def test_sparse_correction_leaves_other_judge_results_and_original_unchanged():
    from verge.modules import review_judge_evidence

    r = rubric()
    j = judged(r, ((False, False), (True, True)))
    original = j.to_json()
    diagnosis = {
        "issues": [
            {
                "kind": "judge_error",
                "criterion_ids": ["c1"],
                "reason": "甲确实满足第一项",
            }
        ],
        "judge_corrections": [
            {
                "criterion_id": "c1",
                "side": "chosen",
                "satisfied": True,
                "reason": "回答甲中的具体依据",
            }
        ],
    }
    final = review_judge_evidence(r, j, diagnosis)
    assert final.criterion_judgments[0].chosen is True
    assert final.criterion_judgments[0].rejected is False
    assert final.criterion_judgments[1] == j.criterion_judgments[1]
    assert j.to_json() == original


@pytest.mark.parametrize(
    "corrections",
    [
        [
            {
                "criterion_id": "c1",
                "side": "chosen",
                "satisfied": True,
                "reason": "无对应误判",
            }
        ],
        [
            {
                "criterion_id": "not_supplied",
                "side": "chosen",
                "satisfied": False,
                "reason": "虚构标准",
            }
        ],
    ],
)
def test_corrections_need_diagnosed_real_error(corrections):
    from verge.modules import review_judge_evidence

    with pytest.raises(ValueError):
        review_judge_evidence(
            rubric(), judged(rubric()), {"issues": [], "judge_corrections": corrections}
        )


def test_judge_error_alone_does_not_force_rubric_change_or_extraction(tmp_path):
    class B(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            if role == "critic_repair":
                cid = payload["initial_rubric"]["criteria"][0]["criterion_id"]
                result.update(
                    issues=[
                        {
                            "kind": "judge_error",
                            "criterion_ids": [cid],
                            "reason": "Judge误读甲",
                        }
                    ],
                    judge_corrections=[
                        {
                            "criterion_id": cid,
                            "side": "chosen",
                            "satisfied": True,
                            "reason": "原文支持",
                        }
                    ],
                    changes=[],
                    criteria=[],
                )
            return result

    from verge.modules import make_llm_modules

    m = make_llm_modules(B())
    mem = ExperienceMemory(tmp_path)
    r = PipelineRunner(
        m, mem, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    assert r.state == "no_rubric_revision" and not mem.all()
    assert r.baseline_diagnosis["issues"][0]["kind"] == "judge_error"
    assert r.baseline_judgment.margin > r.initial_judgment.margin
    assert not any(
        role in ("critic_extract", "experience_checker_batch")
        for role, _ in m.backend.calls
    )
    assert sum(role == "judge" for role, _ in m.backend.calls) == 2


def test_second_diagnosis_can_reference_previous_repair_but_not_rewrite_baseline(
    tmp_path,
):
    m = make_fake_modules("attempt_2_success")
    # Existing replay history ensures the second diagnosis still returns no full baseline array.
    r = PipelineRunner(
        m, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    assert len(r.proposal_attempts) == 2 and r.initial_checker is None
    assert all(
        a.diagnosis["repair"]["judge_corrections"] == [] for a in r.proposal_attempts
    )
    assert r.proposal_attempts[1].margin_before == r.proposal_attempts[0].margin_before


def test_valid_batch_item_survives_invalid_sibling_field():
    class B:
        def generate(self, *a, **kw):
            return {
                "reviews": [
                    {"experience_id": "1", "approved": True, "reason": "有依据"},
                    {"experience_id": "2", "approved": "false", "reason": "字段错误"},
                ]
            }

    from test_v2 import experience

    checker = LLMExperienceChecker(B(), retries=0)
    result = checker.check_many(
        sample(), {}, [experience("first"), experience("second")]
    )
    assert result["first"].approved and result["second"] is None
    assert "second" in checker.last_errors


def test_same_batch_invalid_sibling_does_not_prevent_admission(tmp_path):
    class B(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            if role == "experience_checker_batch":
                result["reviews"][1]["approved"] = "invalid"
            return result

    from verge.modules import make_llm_modules

    m = make_llm_modules(B())
    orig = m.critic.extract

    def extract(*a, **kw):
        xs = orig(*a, **kw)
        e, refs, why = deepcopy(xs[0])
        e.experience_id = "second"
        e.method.principle = "另一个方法"
        return xs + [(e, refs, why)]

    m.critic.extract = extract
    mem = ExperienceMemory(tmp_path)
    r = PipelineRunner(
        m, mem, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    assert len(mem.all()) == 1 and r.state == "experience_admitted"
    entries = r.proposal_attempts[0].extractions
    assert entries[0]["admitted"] and entries[1]["error"] and not entries[1]["reviews"]
    assert not any(role == "critic_rewrite_batch" for role, _ in m.backend.calls)


def test_second_diagnosis_id_mapping_is_explicit_and_valid(tmp_path):
    class B(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            if role == "critic_repair" and payload["round"] == 2:
                mapping = payload["previous_feedback"]["previous_criteria"]
                assert len(mapping) == 2 and mapping[0]["question_index"] == 0
                assert "threshold" in payload["previous_feedback"]
                result["issues"] = [
                    {
                        "kind": "rubric_defect",
                        "criterion_ids": [mapping[0]["criterion_id"]],
                        "reason": "上一修订标准仍缺任务检查",
                    }
                ]
            return result

    from verge.modules import make_llm_modules

    m = make_llm_modules(B("attempt_2_success"))
    r = PipelineRunner(
        m, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    last = r.proposal_attempts[1]
    assert last.diagnosis["repair"]["issues"][0]["criterion_ids"] == [
        r.proposal_attempts[0].repair_rubric.criteria[0].criterion_id
    ]
    assert last.margin_before == r.proposal_attempts[0].margin_before


def test_invalid_json_still_records_reported_cache(monkeypatch):
    import httpx
    from verge.llm import OpenAICompatibleBackend

    monkeypatch.setenv("VERGE_TEST_KEY", "test")
    client = httpx.Client(
        base_url="https://example.test/v1/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "not JSON"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "total_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 50},
                    },
                },
            )
        ),
    )
    events = []
    b = OpenAICompatibleBackend(
        "test",
        base_url="https://example.test/v1",
        api_key_env="VERGE_TEST_KEY",
        client=client,
        max_retries=0,
        metadata_hook=events.append,
    )
    with pytest.raises(LLMFormatError):
        b.generate("critic_assess", {})
    assert events[0]["cache"] == {"reported": True, "cached_input_tokens": 50}
    client.close()
