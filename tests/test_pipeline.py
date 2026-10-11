"""Candidate-first business budgets and adoption branches (offline, no semantics claims)."""

from dataclasses import replace
from copy import deepcopy
import pytest

from verge.calculator import calculate_rubric_judgment
from verge.llm import FakeLLMBackend, LLMFormatError, LLMRequestError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import (
    Criterion,
    CriterionJudgment,
    PreferenceSample,
    Rubric,
    TrajectoryRecord,
)
from verge.modules import JudgeOutputError, make_fake_modules
from verge.pipeline import PipelineRunner


def sample(sid="s"):
    return PreferenceSample(
        sid,
        "任务包含明确关键要求。",
        "answer chosen private",
        "answer rejected private",
        {},
    )


def runner(tmp_path, scenario="attempt_1_success", **options):
    modules = make_fake_modules(scenario, **options)
    memory = ExperienceMemory(tmp_path)
    return PipelineRunner(modules, memory, TrajectoryLog(tmp_path))


@pytest.mark.parametrize(
    "scenario,state,rounds,pool",
    [
        ("initial_success", "initial_success", 0, 0),
        ("attempt_1_success", "application_success", 1, 1),
        ("attempt_2_success", "application_success", 2, 1),
        ("final_failure", "validation_exhausted", 2, 0),
    ],
)
def test_business_outcomes(tmp_path, scenario, state, rounds, pool):
    r = runner(tmp_path, scenario)
    result = r.run_sample(sample())
    assert result.state == state
    assert len(result.application_attempts) == rounds
    assert len(r.memory.all()) == pool
    assert TrajectoryRecord.from_json(result.to_json()) == result
    assert len(list(r.trajectories.all())) == 1
    assert not hasattr(r.modules, "reflector")
    assert not hasattr(r.modules, "admission_checker")
    assert len(result.proposal_attempts) == rounds
    assert all(role != "admission_checker" for role, _ in r.modules.backend.requests)
    if pool:
        c = result.proposal_attempts[-1].candidate
        assert (r.memory.all()[0].content, r.memory.all()[0].trigger) == (
            c.content,
            c.trigger,
        )
        assert c.fusion_record.credit == result.application_attempts[-1].credit
        assert r.memory.all()[0].alpha == pytest.approx(1.8)




def test_two_proposals_two_direct_applications(tmp_path):
    r = runner(tmp_path, "attempt_2_success")
    rec = r.run_sample(sample())
    assert rec.state == "application_success"
    assert [p.round_number for p in rec.proposal_attempts] == [1, 2]
    assert len(rec.application_attempts) == 2
    assert len({p.candidate.candidate_id for p in rec.proposal_attempts}) == 2
    cs = [p for role, p in r.modules.backend.requests if role == "critic"]
    assert cs[1]["current_rubric"] == rec.application_attempts[0].rubric.to_dict()
    assert cs[1]["current_judgment"] == rec.application_attempts[0].judgment.to_dict()
    assert (
        cs[1]["previous_candidate"]["candidate_id"]
        == rec.application_attempts[0].candidate_id
    )
    assert cs[1]["adopted"] is True and "rejected_candidate" not in cs[1]
    assert "analysis" not in cs[1]
    gs = [p for role, p in r.modules.backend.requests if role == "initial_generator"]
    assert (
        gs[1]["candidate_experience"]["experience_id"]
        != gs[2]["candidate_experience"]["experience_id"]
    )
    for p in gs:
        assert set(p) <= {
            "task",
            "selected_experiences",
            "candidate_experience",
            "_application_round",
            "validation_feedback",
        }
        assert p["selected_experiences"] == gs[0]["selected_experiences"]
        assert "private" not in str(p)
    assert len([1 for role, _ in r.modules.backend.requests if role == "selector"]) <= 1




@pytest.mark.parametrize(
    "scenario,state",
    [
        ("attempt_1_success", "success_without_candidate"),
        ("final_failure", "validation_exhausted"),
    ],
)
def test_unadopted_candidate_is_not_fused_or_negative_evidence(
    tmp_path, scenario, state
):
    r = runner(tmp_path, scenario, adopt_candidate=False)
    rec = r.run_sample(sample())
    assert rec.state == state and not r.memory.all()
    assert all(
        a.adopted is False and a.credit.amount == 0 for a in rec.application_attempts
    )
    assert all(role != "fusion_judge" for role, _ in r.modules.backend.requests)
    if scenario == "final_failure":
        cs = [p for role, p in r.modules.backend.requests if role == "critic"]
        assert cs[1]["adopted"] is False


def test_no_candidate_ends_without_inventing_admission(tmp_path):
    r = runner(tmp_path, candidate_count=0)
    rec = r.run_sample(sample())
    assert rec.state == "no_candidate" and not rec.application_attempts
    assert all(
        role not in ("admission_checker", "fusion_judge")
        for role, _ in r.modules.backend.requests
    )


@pytest.mark.parametrize(
    "role,method,stage",
    [
        ("selector", "select", "selection"),
        ("initial_generator", "generate", "initial_generator"),
        ("fusion_judge", "find_match", "fusion_judge"),
    ],
)
def test_role_contract_isolation_continues_next_sample(tmp_path, role, method, stage):
    r = runner(tmp_path)
    obj = getattr(r.modules, role)
    original = getattr(obj, method)
    first = True

    def call(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise ValueError("external output invalid")
        return original(*args, **kwargs)

    setattr(obj, method, call)
    records = r.run([sample("a"), sample("b")])
    assert (
        records[0].state == "sample_error" and records[0].sample_error["stage"] == stage
    )
    assert records[1].state == "application_success"


@pytest.mark.parametrize(
    "role,method", [("critic", "propose"), ("initial_generator", "apply")]
)
@pytest.mark.parametrize("error", [ValueError, LLMFormatError])
def test_invalid_round_uses_remaining_opportunity_without_fake_judge(
    tmp_path, role, method, error
):
    r = runner(tmp_path, "attempt_2_success")
    obj = getattr(r.modules, role)
    original = getattr(obj, method)
    n = 0

    def fail(*args, **kwargs):
        nonlocal n
        n += 1
        if n == 1:
            raise error("bad model output")
        return original(*args, **kwargs)

    setattr(obj, method, fail)
    rec = r.run_sample(sample())
    assert rec.state == "application_success"
    if role == "critic":
        assert rec.proposal_attempts[0].error and len(rec.application_attempts) == 1
    else:
        assert (
            rec.application_attempts[0].judgment is None
            and rec.application_attempts[0].adopted is None
        )
    calls = [p for name, p in r.modules.backend.requests if name == "critic"]
    assert calls[-1]["invalid_error"] == "bad model output"
    assert calls[-1]["current_judgment"] == rec.initial_judgment.to_dict()


@pytest.mark.parametrize(
    "role,method", [("critic", "propose"), ("initial_generator", "apply")]
)
def test_two_invalid_rounds_end_and_next_sample_continues(tmp_path, role, method):
    r = runner(tmp_path)
    obj = getattr(r.modules, role)
    original = getattr(obj, method)
    count = 0

    def fail(*args, **kw):
        nonlocal count
        count += 1
        if count <= 2:
            raise LLMFormatError("format exhausted")
        return original(*args, **kw)

    setattr(obj, method, fail)
    a, b = r.run([sample("a"), sample("b")])
    assert (
        a.state == "invalid_application_exhausted" and b.state == "application_success"
    )
    assert len(a.proposal_attempts) == 2 and not r.memory.all()[
        0
    ].source_trajectories == ["a"]


@pytest.mark.parametrize("application", [False, True])
def test_judge_error_no_candidate_write_and_next_sample(tmp_path, application):
    r = runner(tmp_path)
    judge = r.modules.judge.judge
    n = 0

    def fail(*args, **kwargs):
        nonlocal n
        n += 1
        if n == (2 if application else 1):
            raise JudgeOutputError("judge exhausted")
        return judge(*args, **kwargs)

    r.modules.judge.judge = fail
    a, b = r.run([sample("a"), sample("b")])
    assert (
        a.state == "judge_error_application"
        if application
        else a.state == "judge_error_initial"
    )
    assert a.judge_error and b.state == "application_success"
    assert all("a" not in e.source_trajectories for e in r.memory.all())
    if application:
        assert (
            a.application_attempts[0].rubric
            and a.application_attempts[0].judgment is None
        )


@pytest.mark.parametrize(
    "component,method,error",
    [
        ("critic", "propose", LLMRequestError),
        ("initial_generator", "apply", RuntimeError),
        ("judge", "judge", TypeError),
    ],
)
def test_infrastructure_or_program_error_is_not_swallowed(
    tmp_path, component, method, error
):
    r = runner(tmp_path)
    setattr(
        getattr(r.modules, component),
        method,
        lambda *a, **kw: (_ for _ in ()).throw(error("fatal")),
    )
    with pytest.raises(error):
        r.run_sample(sample())
    assert not list(r.trajectories.all())


def test_storage_and_calculator_errors_are_global(tmp_path, monkeypatch):
    r = runner(tmp_path)
    monkeypatch.setattr(
        "verge.pipeline.calculate_experience_credits",
        lambda *a: (_ for _ in ()).throw(ValueError("internal mismatch")),
    )
    with pytest.raises(ValueError, match="internal mismatch"):
        r.run_sample(sample())
    assert not list(r.trajectories.all())


def test_r0_credit_survives_later_failure_and_e_is_not_recredited(tmp_path):
    r = runner(tmp_path)
    seed = r.memory.fuse(
        __import__(
            "verge.models", fromlist=["CandidateExperience"]
        ).CandidateExperience("seed_c", "经验", "*"),
        "seed",
    )
    original = r.modules.critic.propose

    def fail(*a, **kw):
        raise ValueError("invalid critic output")

    r.modules.critic.propose = fail
    rec = r.run_sample(sample())
    e = r.memory.get(seed.experience_id)
    assert rec.state == "invalid_application_exhausted" and e.beta == pytest.approx(1.2)
    r.modules.critic.propose = original
    nxt = r.run_sample(sample("next"))
    e = r.memory.get(seed.experience_id)
    assert nxt.state == "application_success" and e.beta == pytest.approx(1.4)
    assert len(e.evidence_history) == 2
    for call in [
        p for role, p in r.modules.backend.requests if role == "initial_generator"
    ]:
        if "candidate_experience" in call:
            assert call["selected_experiences"] == nxt.selected_experiences


def test_success_with_negative_candidate_credit_still_fuses(tmp_path):
    r = runner(tmp_path)
    original = r.modules.initial_generator.apply

    def apply(task, selected, candidate, n):
        return Rubric(
            "applied",
            [
                Criterion("bad", "基础条件", 2, [candidate.candidate_id]),
                Criterion("good", "关键条件", 8, []),
            ],
        )

    r.modules.initial_generator.apply = apply
    judge = r.modules.judge.judge

    def judged(rubric, *args, **kw):
        if rubric.rubric_id == "applied":
            return calculate_rubric_judgment(
                rubric,
                [
                    CriterionJudgment("bad", False, True, "chosen fixture evidence", "rejected fixture evidence"),
                    CriterionJudgment("good", True, False, "chosen fixture evidence", "rejected fixture evidence"),
                ],
            )
        return judge(rubric, *args, **kw)

    r.modules.judge.judge = judged
    rec = r.run_sample(sample())
    e = r.memory.all()[0]
    assert (
        rec.state == "application_success"
        and e.alpha == 1
        and e.beta == pytest.approx(1.2)
    )
    assert rec.application_attempts[0].credit.net < 0


def test_heldout_freezes_memory_text_beta_and_no_candidate_roles(tmp_path):
    r = runner(tmp_path)
    r.run_sample(sample("train"))
    before = r.memory.path.read_bytes()
    r.modules.backend.requests.clear()
    evaluation = PipelineRunner(
        r.modules,
        ExperienceMemory(tmp_path, read_only=True),
        r.trajectories,
        training=False,
    )
    rec = evaluation.run_sample(sample("test"))
    assert (
        not rec.proposal_attempts
        and not rec.application_attempts
        and not rec.downstream_credits
    )
    assert r.memory.path.read_bytes() == before
    assert all(
        role in ("selector", "initial_generator", "judge")
        for role, _ in r.modules.backend.requests
    )


def test_max_rounds_one_preserves_finite_bound(tmp_path):
    r = runner(tmp_path, "attempt_2_success")
    r.max_rounds = 1
    rec = r.run_sample(sample())
    assert rec.state == "validation_exhausted" and len(rec.application_attempts) == 1


def test_r0_balanced_channels_update_before_success_branch(tmp_path):
    from verge.models import CandidateExperience

    r = runner(tmp_path, "initial_success")
    seeded = r.memory.fuse(
        CandidateExperience("seed_c", "经验", "*"), "seed"
    )
    eid = seeded.experience_id
    r.modules.initial_generator.generate = lambda *a: Rubric(
        "r0",
        [
            Criterion("p", "正向要求", 2, [eid]),
            Criterion("n", "另一个正向要求", 2, [eid]),
            Criterion("t", "task要求", 6, []),
        ],
    )
    r.modules.judge.judge = lambda rubric, *a, **kw: calculate_rubric_judgment(
        rubric,
        [
            CriterionJudgment("p", True, False, "chosen fixture evidence", "rejected fixture evidence"),
            CriterionJudgment("n", False, True, "chosen fixture evidence", "rejected fixture evidence"),
            CriterionJudgment("t", True, False, "chosen fixture evidence", "rejected fixture evidence"),
        ],
    )
    rec = r.run_sample(sample())
    e = r.memory.get(eid)
    assert rec.state == "initial_success" and rec.downstream_credits[eid].net == 0
    assert (e.alpha, e.beta) == pytest.approx((1.2, 1.2)) and len(
        e.evidence_history
    ) == 1
    assert not rec.proposal_attempts


def test_adopted_zero_candidate_credit_still_adds_without_beta_evidence(tmp_path):
    r = runner(tmp_path)
    r.modules.initial_generator.apply = lambda task, e, c, n: Rubric(
        "a",
        [
            Criterion("c", "候选影响的要求", 2, [c.candidate_id]),
            Criterion("t", "task要求", 8, []),
        ],
    )
    original = r.modules.judge.judge

    def judged(rubric, *a, **kw):
        if rubric.rubric_id == "a":
            return calculate_rubric_judgment(
                rubric,
                [
                    CriterionJudgment("c", True, True, "chosen fixture evidence", "rejected fixture evidence"),
                    CriterionJudgment("t", True, False, "chosen fixture evidence", "rejected fixture evidence"),
                ],
            )
        return original(rubric, *a, **kw)

    r.modules.judge.judge = judged
    rec = r.run_sample(sample())
    e = r.memory.all()[0]
    assert rec.state == "application_success" and rec.application_attempts[0].adopted
    assert (
        e.alpha == e.beta == 1 and not e.evidence_history and len(e.fusion_history) == 1
    )
