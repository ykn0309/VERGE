from verge.models import ExperienceReview
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
import json
import pytest
from verge.models import *
from verge.calculator import *
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import *
from verge.pipeline import *
from verge.prompts import *
from verge.llm import *


def sample(n=0):
    return PreferenceSample("s" + str(n), "任务" + str(n), "完整回答", "不完整回答", {})


def experience(eid="exp_long_real_id"):
    return Experience(eid, Method("任务有明确要求", "核对任务要求"), "source")


def rubric(eid="exp_long_real_id", weights=(1, 1)):
    return Rubric(
        "r",
        [
            Criterion("c1", "核验一", weights[0], [eid]),
            Criterion("c2", "核验二", weights[1], [eid]),
        ],
    )


def judged(r, states=((True, False), (True, False))):
    return calculate_rubric_judgment(
        r,
        [
            CriterionJudgment(c.criterion_id, a, b, "a原文", "b原文")
            for c, (a, b) in zip(r.criteria, states)
        ],
    )


def app(
    e=None,
    n=0,
    states=((True, False), (True, False)),
    phase="downstream",
    weights=(1, 1),
):
    e = e or experience()
    r = rubric(e.experience_id, weights)
    return application(sample(n), e, r, judged(r, states), phase)


def reports(statuses=("correct", "correct", "correct")):
    return [
        CheckerReport(s, v, "独立报告")
        for s, v in zip(("selector", "generator", "judge"), statuses)
    ]


def params(version="test-parameters"):
    # Fictional unit-test fixture ONLY; never shipped as usable run parameters.
    return ProbabilityModel(
        version,
        0.6,
        0.8,
        {
            "1:1": {"positive": 0.8, "reverse": 0.1, "tie": 0.1},
            "1:0": {"positive": 0.3, "reverse": 0.4, "tie": 0.3},
            "0:1": {"positive": 0.2, "reverse": 0.5, "tie": 0.3},
            "0:0": {"positive": 0.3, "reverse": 0.4, "tie": 0.3},
        },
        {
            "1": {"correct": 0.8, "error": 0.1, "unknown": 0.1},
            "0": {"correct": 0.1, "error": 0.8, "unknown": 0.1},
        },
        {
            "scope": "available_reports",
            "estimation_source": "fictional TEST fixture",
            "feedback_protocol": "net_direction_tie_v1",
            "admission_protocol": "experience_checker_v1",
        },
        "usable",
    )


@pytest.mark.parametrize(
    "statuses", list(product(("correct", "error", "unknown", "failed"), repeat=3))
)
def test_checker_priority_all_64_combinations(statuses):
    expected = (
        "error"
        if "error" in statuses
        else (
            "unavailable"
            if "failed" in statuses
            else "unknown" if "unknown" in statuses else "correct"
        )
    )
    assert aggregate_reports(reports(statuses)) == expected


def test_missing_checker_never_all_correct():
    assert aggregate_reports(reports()[:2]) == "unavailable"
    assert aggregate_reports([CheckerReport("judge", "error", "明确错误")]) == "error"


@pytest.mark.parametrize(
    "states,expected",
    [
        (((1, 0), (1, 0)), "positive"),
        (((0, 1), (0, 1)), "reverse"),
        (((1, 1), (0, 0)), "tie"),
        (((1, 0), (0, 1)), "tie"),
    ],
)
def test_feedback_exact(states, expected):
    states = tuple(tuple(bool(x) for x in pair) for pair in states)
    assert app(states=states).feedback == expected


def test_fraction_and_zero_weight():
    e = experience()
    r = rubric(e.experience_id, (0.1, 0.2))
    r.criteria.append(Criterion("c3", "第三项", 0.1, [e.experience_id]))
    j = calculate_rubric_judgment(
        r,
        [
            CriterionJudgment(c.criterion_id, a, b, "a", "b")
            for c, (a, b) in zip(
                r.criteria, [(True, False), (False, True), (True, False)]
            )
        ],
    )
    assert application(sample(), e, r, j).feedback == "tie"
    assert app(weights=(0, 0.5), states=((True, False), (True, True))).feedback == "tie"
    r = rubric()
    r.criteria[0].weight = r.criteria[1].weight = 0
    with pytest.raises(ValueError):
        calculate_rubric_judgment(r, [])


def test_uncited_and_other_sources_do_not_count():
    e = experience()
    r = rubric()
    r.criteria[0].source_experiences = []
    r.criteria[1].source_experiences = ["other"]
    assert application(sample(), e, r, judged(r)).feedback == "unused"


def test_single_source_and_strict_contract():
    with pytest.raises(ValueError):
        Criterion("c", "问题", 1, ["a", "b"])
    with pytest.raises(ValueError):
        Experience.from_dict({"content": "old", "alpha": 1, "beta": 1})
    with pytest.raises(ValueError):
        ExperienceUpdate.from_dict({"action": "split", "method": None, "reason": "x"})


@pytest.mark.parametrize("extra", [{"template": None}, {"parameters": {}}])
def test_method_rejects_removed_fields(extra):
    with pytest.raises(ValueError, match="unknown"):
        Method.from_dict({"trigger": "t", "principle": "p", **extra})


def test_two_field_method_and_source_whitelist():
    e = experience()
    e.method = Method("指定材料", "核对回答事实是否得到指定材料支持")

    class Backend:
        def generate(self, role, payload, **kw):
            return {
                "analysis": "取舍",
                "criteria": [
                    {
                        "question": "回答是否有材料2支持？",
                        "weight": 0.8,
                        "source_experiences": ["1"],
                    },
                    {"question": "是否清晰？", "weight": 0.2, "source_experiences": []},
                ],
            }

    g = LLMInitialRubricGenerator(Backend(), retries=0)
    assert (
        g.generate("请概括材料2", [e]).criteria[0].question == "回答是否有材料2支持？"
    )


def test_bn_matches_eight_state_enumeration():
    p = params()
    prior = 0.6
    for f, o in product(("positive", "reverse", "tie"), ("correct", "error")):
        ws = {
            (q, c): (prior if q else 1 - prior)
            * (p.execution_prior if c else 1 - p.execution_prior)
            * p.feedback[f"{q}:{c}"][f]
            * p.observation[str(c)][o]
            for q, c in product((0, 1), repeat=2)
        }
        expected = sum(v for (q, c), v in ws.items() if q) / sum(ws.values())
        assert p.infer(prior, f, o)[0] == pytest.approx(expected)


def test_sequential_equals_batch_and_reestimate_no_double_count():
    e = experience()
    p = params()
    for n in range(8):
        a = app(e, n)
        a.reports = reports()
        update_quality(e, a, p)
        e.applications.append(a)
    expected = e.q_posterior
    rebuild_quality(e, p)
    assert e.q_posterior == pytest.approx(expected)
    rebuild_quality(e, p)
    assert e.q_posterior == pytest.approx(expected)
    e2 = experience()
    a = app(e2)
    a.reports = reports()
    update_quality(e2, a, p)
    l1, l0 = [
        sum(
            (p.execution_prior if c else 1 - p.execution_prior)
            * p.feedback[f"{q}:{c}"]["positive"]
            * p.observation[str(c)]["correct"]
            for c in (0, 1)
        )
        for q in (1, 0)
    ]
    assert expected == pytest.approx(0.6 * l1**8 / (0.6 * l1**8 + 0.4 * l0**8))


@pytest.mark.parametrize("f,o", [("tie", "unknown"), ("unused", "correct")])
def test_uninformative_not_updated(f, o):
    e = experience()
    a = app(e)
    a.feedback = f
    a.reports = (
        reports(("unknown", "correct", "correct")) if o == "unknown" else reports()
    )
    update_quality(e, a, params())
    assert e.q_posterior is None


def test_partial_failure_with_error_does_update():
    e = experience()
    a = app(e)
    a.reports = reports(("error", "failed", "unknown"))
    update_quality(e, a, params())
    assert a.observation == "error" and a.update_reason == "updated"
    e2 = experience()
    a2 = app(e2)
    a2.reports = reports()
    update_quality(e2, a2, None)
    assert a2.update_reason == "parameters_unavailable" and e2.q_posterior is None


def test_no_fictitious_parameters_and_global_parameter_change():
    p = params()
    p.feedback["1:1"]["positive"] = 2
    with pytest.raises(ProbabilityError):
        p.__post_init__()
    e = experience()
    a = app(e)
    a.reports = reports()
    update_quality(e, a, params())
    changed = app(e, 1)
    changed.reports = reports()
    with pytest.raises(ProbabilityError):
        update_quality(e, changed, params("new"))


def test_memory_roundtrip_dedupe_frozen_and_prior_only(tmp_path):
    p = params()
    m = ExperienceMemory(tmp_path, parameters=p)
    e = experience()
    a = app(e, "source", phase="source_validation")
    a.sample = replace(a.sample, sample_id="source")
    m.add(e, a.sample, ExperienceReview(True, "测试核对通过"))
    assert m.get(e.experience_id).q_posterior == p.q_prior
    x = app(e, 1)
    x.reports = reports()
    m.record(x)
    q = m.get(e.experience_id).q_posterior
    duplicate = deepcopy(x)
    m.record(duplicate)
    assert (
        duplicate.update_reason == "duplicate"
        and m.get(e.experience_id).q_posterior == q
    )
    reread = ExperienceMemory(tmp_path, parameters=p)
    assert reread.all() == m.all()
    read = reread.get(e.experience_id)
    read.method.principle = "tamper"
    assert m.get(e.experience_id).method.principle != "tamper"
    frozen = ExperienceMemory(tmp_path, read_only=True)
    with pytest.raises(ValueError):
        frozen.mark_checked(e.experience_id)


def test_unknown_counts_towards_check_and_source_does_not():
    e = experience()
    source = app(e, "source", phase="source_validation")
    source.sample = replace(source.sample, sample_id=e.source_sample_id)
    e.applications.append(source)
    assert not e.downstream
    for n in range(3):
        a = app(e, n)
        a.reports = reports(("unknown", "correct", "correct"))
        e.applications.append(a)
    assert revision_reasons(e) == []
    e.last_checked_count = 3
    assert revision_reasons(e) is None
    for n in range(3, 6):
        a = app(e, n)
        a.reports = reports(("correct", "error", "correct"))
        e.applications.append(a)
    assert revision_reasons(e) == ["generator_errors"]


def test_three_stage_errors_do_not_combine():
    e = experience()
    for n, stage in enumerate(("selector", "generator", "judge")):
        a = app(e, n)
        a.reports = [CheckerReport(stage, "error", "x")]
        e.applications.append(a)
    assert revision_reasons(e) == []


def test_sampler_3_1_1_1_and_dedup():
    e = experience()
    source = app(e, -1, phase="source_validation")
    source.sample = replace(source.sample, sample_id="source")
    e.applications = [source]
    for n in range(1, 9):
        a = app(e, n)
        a.reports = reports()
        if n <= 3:
            a.reports = reports(("correct", "error", "correct"))
        if n == 4:
            a.reports = reports(("unknown", "correct", "correct"))
        e.applications.append(a)
    selected, kinds = select_revision_cases(e, {e.experience_id: e}, window=8)
    assert len(selected) == 6 and len(kinds) == 6
    assert list(kinds.values()).count("problem") == 3
    assert set(kinds.values()) == {
        "problem",
        "source",
        "positive",
        "applicability_dispute",
    }


@pytest.mark.parametrize(
    "scenario,state,rounds",
    [
        ("initial_success", "initial_success", 0),
        ("attempt_1_success", "application_success", 1),
        ("attempt_2_success", "application_success", 2),
        ("final_failure", "validation_exhausted", 1),
    ],
)
def test_candidate_only_flow_and_critic_conversation(tmp_path, scenario, state, rounds):
    m = make_fake_modules(scenario)
    mem = ExperienceMemory(tmp_path)
    r = PipelineRunner(m, mem, TrajectoryLog(tmp_path)).run_sample(sample())
    assert r.state == state and len(r.proposal_attempts) == rounds
    assert len(mem.all()) == (0 if scenario == "initial_success" else 1)
    calls = m.backend.calls
    assert sum(role == "experience_checker" for role, p in calls) == rounds
    candidate_calls = [
        p for role, p in calls if role == "initial_generator" and p.get("_round")
    ]
    assert all(len(p["experiences"]) == 1 for p in candidate_calls)
    assert all(e.q_posterior is None for e in mem.all())
    assert list(TrajectoryLog(tmp_path).all())[0].to_dict() == r.to_dict()


def test_downstream_checkers_and_frozen_no_mutation(tmp_path):
    mods = make_fake_modules()
    mem = ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(mods, mem, log)
    runner.run_sample(sample(0))
    runner.run_sample(sample(1))
    before = mem.path.read_bytes()
    assert any(role.endswith("_checker") for role, p in mods.backend.calls)
    start = len(mods.backend.calls)
    PipelineRunner(
        mods, ExperienceMemory(tmp_path, read_only=True), log, training=False
    ).run_sample(sample(2))
    assert mem.path.read_bytes() == before
    assert not any(
        role.endswith("_checker") or role == "critic"
        for role, p in mods.backend.calls[start:]
    )


def test_checker_visibility_and_error_isolation():
    a = app()
    b = FakeLLMBackend()
    check = LLMCheckers({s: b for s in ("selector", "generator", "judge")})
    check.check(a)
    for role, p in b.calls:
        text = json.dumps(p)
        assert (
            "chosen" not in text
            and "rejected" not in text
            and "feedback" not in text
            and "q_posterior" not in text
        )
        if role != "judge_checker":
            assert a.sample.chosen not in text and a.sample.rejected not in text

    class Failed:
        def generate(self, *args, **kw):
            raise LLMSampleError("timeout")

    check.backends["selector"] = Failed()
    rs = check.check(a)
    assert rs[0].status == "failed"


def test_short_ids_and_natural_text_not_rewritten():
    codec = NumberedIDCodec()
    raw = {
        "sample_id": "sample-long",
        "experience_id": "exp-long",
        "source_experiences": ["exp-long"],
        "task": "exp-long is quoted text",
    }
    wire = codec.project(raw)
    assert (
        wire["sample_id"] == "1"
        and wire["experience_id"] == "1"
        and wire["task"] == raw["task"]
    )
    assert codec.restore(wire) == raw
    with pytest.raises(ValueError):
        codec.restore({"selected_experience_ids": ["99"]})


def test_hybrid_direct_active_only_and_no_keyword_arbitrary_bonus():
    class Embed:
        def embed(self, text):
            return (1.0, 1.0)

    es = [experience("b"), experience("a"), experience("retired")]
    es[-1].status = "retired"
    es[-1].retired_by_event_id = "event"
    r = HybridRetriever(Embed())
    found = r.retrieve("unmatchedenglishword", es)
    assert [e.experience_id for e in found] == ["a", "b"]
    assert all(t["keyword_rank"] is None for t in r.last_trace)
    assert r.fork().cache == r.cache and r.fork().cache is not r.cache


def test_sample_failure_continues_but_program_error_fails_fast(tmp_path):
    mods = make_fake_modules()
    original = mods.initial_generator.generate

    def generate(task, *a, **kw):
        if task == "任务0":
            raise LLMSampleError("exhausted timeout")
        return original(task, *a, **kw)

    mods.initial_generator.generate = generate
    rs = PipelineRunner(mods, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path)).run(
        [sample(0), sample(1)]
    )
    assert rs[0].state == "sample_error" and rs[1].state == "application_success"
    mods.initial_generator.generate = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("bug")
    )
    with pytest.raises(RuntimeError):
        PipelineRunner(
            mods, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path)
        ).run_sample(sample(3))


def test_revision_replay_replaces_only_when_strictly_better(tmp_path):
    mem = ExperienceMemory(tmp_path)
    e = experience()
    source = app(e, 0, phase="source_validation")
    source.sample = replace(source.sample, sample_id="source")
    mem.add(e, source.sample, ExperienceReview(True, "测试核对通过"))
    for n in range(1, 6):
        a = app(e, n, states=((False, True), (False, True)))
        a.reports = reports(("correct", "error", "correct"))
        mem.record(a)
    mods = make_fake_modules()

    class Updater:
        def update(self, *args, **kwargs):
            return ExperienceUpdate(
                "revise", Method("精确条件", "改进方法"), ["principle"], "修正歧义"
            )

    mods.experience_updater = Updater()
    runner = PipelineRunner(mods, mem, TrajectoryLog(tmp_path))

    def replay(s, v, pool):
        return app(
            v,
            s.sample_id,
            states=(
                ((True, False), (True, False))
                if v.parent_id
                else ((False, True), (False, True))
            ),
            phase="replay",
        )

    runner._replay = replay
    event = runner._revise(mem.get(e.experience_id), "s5", ["generator_errors"])
    assert event.accepted and len(mem.all(active_only=True)) == 1
    assert mem.get(e.experience_id).status == "retired"
    child = mem.all(active_only=True)[0]
    assert child.q_posterior is None and child.parent_id == e.experience_id
    assert (
        mem.get(e.experience_id).applications
        == ExperienceMemory(tmp_path).get(e.experience_id).applications
    )
    assert ExperienceMemory(tmp_path).all() == mem.all()


def test_revision_minimum_rejects_and_parent_pool_unchanged(tmp_path):
    mem = ExperienceMemory(tmp_path)
    e = experience()
    a = app(e, 0, phase="source_validation")
    a.sample = replace(a.sample, sample_id="source")
    mem.add(e, a.sample, ExperienceReview(True, "测试核对通过"))
    mods = make_fake_modules()
    mods.experience_updater.update = lambda *args, **kwargs: ExperienceUpdate(
        "revise", Method("t", "p"), ["principle"], "修订"
    )
    runner = PipelineRunner(mods, mem, TrajectoryLog(tmp_path))
    runner._replay = lambda s, v, pool: app(v, s.sample_id, phase="replay")
    event = runner._revise(mem.get(e.experience_id), "s5", ["quality_risk"])
    assert (
        not event.accepted
        and mem.get(e.experience_id).status == "active"
        and len(mem.all()) == 1
    )


def test_valid_unselected_replay_scores_zero():
    e = experience()
    a = app(e)
    a.feedback = "unused"
    assert replay_score(a) == 0 and replay_score(None) is None


def test_seeded_e_never_leaks_and_rejected_candidate_is_not_applied(
    tmp_path,
):
    class Backend(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            if role == "critic":
                answer = super().generate(role, payload, **kw)
                answer["candidates"][0]["principle"] = "C" + str(payload["round"])
                return answer
            return super().generate(role, payload, **kw)

    backend = Backend("attempt_2_success")
    mods = make_llm_modules(backend)
    memory = ExperienceMemory(tmp_path)
    e = experience()
    e.method.principle = "old-E"
    source = app(e, 0, phase="source_validation")
    source.sample = replace(source.sample, sample_id="source")
    memory.add(e, source.sample, ExperienceReview(True, "测试核对通过"))
    rec = PipelineRunner(
        mods, memory, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample(5))
    calls = [payload for role, payload in backend.calls if role == "initial_generator"]
    assert [x["principle"] for x in calls[0]["experiences"]] == ["old-E"]
    assert len(calls) == 2
    assert [x["principle"] for x in calls[1]["experiences"]] == ["C2"]
    admitted = memory.get(rec.admitted_experience_ids[0])
    assert admitted.method.principle == "C2"


def test_partial_failure_error_counts_but_unknown_never_counts_as_error():
    e = experience()
    for n in range(3):
        a = app(e, n)
        a.reports = reports(("error", "failed", "unknown"))
        a.observation = aggregate_reports(a.reports)
        e.applications.append(a)
    assert revision_reasons(e) == ["selector_errors"]
    for a in e.applications:
        a.reports = reports(("unknown", "failed", "unknown"))
    assert revision_reasons(e) == []


def test_updater_or_replay_request_error_keeps_parent_and_records_event(tmp_path):
    memory = ExperienceMemory(tmp_path)
    e = experience()
    source = app(e, 0, phase="source_validation")
    source.sample = replace(source.sample, sample_id="source")
    memory.add(e, source.sample, ExperienceReview(True, "测试核对通过"))
    mods = make_fake_modules()

    def fail(*a, **kwargs):
        raise LLMSampleError("timeout after finite retries")

    mods.experience_updater.update = fail
    runner = PipelineRunner(mods, memory, TrajectoryLog(tmp_path))
    before = memory.get(e.experience_id)
    event = runner._revise(before, "s1", ["quality_risk"])
    assert not event.accepted and event.error and memory.get(e.experience_id) == before
    assert len(memory.revisions) == 1


def test_reviewed_candidate_stays_admitted_even_when_whole_task_fails(
    tmp_path,
):
    mods = make_fake_modules("final_failure")

    def generate(task, selected, *, round_number=0):
        eid = selected[0].experience_id if selected else None
        return Rubric(
            "round" + str(round_number),
            [
                Criterion(
                    "a",
                    "pass" if round_number == 1 else "fail",
                    0.2,
                    [eid] if eid else [],
                ),
                Criterion("b", "fail", 0.8, []),
            ],
        )

    def judge(r, chosen, rejected, *, task):
        return calculate_rubric_judgment(
            r,
            [
                CriterionJudgment(
                    c.criterion_id,
                    c.question == "pass",
                    c.question != "pass",
                    "原文a",
                    "原文b",
                )
                for c in r.criteria
            ],
        )

    mods.initial_generator.generate = generate
    mods.judge.judge = judge
    memory = ExperienceMemory(tmp_path)
    rec = PipelineRunner(mods, memory, TrajectoryLog(tmp_path)).run_sample(sample())
    assert all(not p.judgment.success for p in rec.proposal_attempts)
    assert rec.proposal_attempts[0].applications[0].feedback == "positive"
    assert len(rec.proposal_attempts) == 1
    assert rec.admitted_experience_ids == [
        rec.proposal_attempts[0].proposal.candidates[0].experience_id
    ]
    assert len(memory.all()) == 1
