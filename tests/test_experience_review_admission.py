"""Content review admits candidates; source application cannot retroactively reject them."""

from copy import deepcopy
from dataclasses import replace
import pytest
from verge.models import ExperienceReview, Method
from verge.modules import make_fake_modules
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.pipeline import PipelineRunner
from verge.calculator import rebuild_quality, ProbabilityError
from verge.llm import LLMFormatError
from test_v2 import sample, params, experience, app, reports


def runner(tmp_path, scenario="final_failure", parameters=None):
    mods = make_fake_modules(scenario)
    memory = ExperienceMemory(tmp_path, parameters=parameters)
    return (
        mods,
        memory,
        PipelineRunner(mods, memory, TrajectoryLog(tmp_path), revision_enabled=False),
    )


def test_review_rejection_only_feedback_and_no_application_before_pass(tmp_path):
    mods, mem, run = runner(tmp_path, "attempt_2_success")
    r = run.run_sample(sample())
    assert [a.review.approved for a in r.proposal_attempts] == [False, True]
    critic = [p for role, p in mods.backend.calls if role == "critic"]
    assert critic[1]["previous_feedback"] == {
        "experience_review": r.proposal_attempts[0].review.to_dict()
    }
    assert critic[0]["criteria"] == critic[1]["criteria"]
    calls = [role for role, _ in mods.backend.calls]
    assert calls.index("experience_checker") < calls.index("initial_generator", 1)
    generators = [p for role, p in mods.backend.calls if role == "initial_generator"]
    assert len(generators) == 2 and len(generators[-1]["experiences"]) == 1
    assert len(mem.all()) == 1


def test_rejected_twice_no_admission_or_candidate_application(tmp_path):
    mods, mem, run = runner(tmp_path)
    mods.experience_checker.check = lambda *args: ExperienceReview(
        False, "包含本题专名，需去除"
    )
    r = run.run_sample(sample())
    assert r.state == "candidate_rejected" and len(r.proposal_attempts) == 2
    assert not mem.all()
    assert sum(role == "initial_generator" for role, _ in mods.backend.calls) == 1


@pytest.mark.parametrize("scenario", ["final_failure", "attempt_1_success"])
def test_admitted_source_feedback_updates_q_and_reestimate_does_not_double_count(
    tmp_path, scenario
):
    p = params()
    mods, mem, run = runner(tmp_path, scenario, p)
    r = run.run_sample(sample())
    assert len(r.proposal_attempts) == 1 and len(mem.all()) == 1
    e = mem.get(r.admitted_experience_ids[0])
    a = e.applications[0]
    assert a.phase == "source_validation" and a.update_reason == "updated"
    assert e.q_posterior != p.q_prior and e.evidence_counts["source_validation"] == 1
    assert e.downstream == [] and e.admission_review.approved
    before = e.q_posterior
    mem.reestimate(p)
    assert mem.get(e.experience_id).q_posterior == pytest.approx(before)
    dup = deepcopy(a)
    mem.record(dup)
    assert (
        dup.update_reason == "duplicate"
        and mem.get(e.experience_id).q_posterior == before
    )
    downstream = deepcopy(a)
    downstream.phase = "downstream"
    downstream.sample = replace(a.sample, sample_id="alias")
    mem.record(downstream)
    assert downstream.update_reason == "duplicate_task"


def test_generator_failure_after_admission_keeps_experience(tmp_path):
    mods, mem, run = runner(tmp_path)
    original = mods.initial_generator.generate

    def generate(task, selected, *, round_number=0):
        if round_number:
            raise LLMFormatError("bad bindings type")
        return original(task, selected, round_number=round_number)

    mods.initial_generator.generate = generate
    r = run.run_sample(sample())
    assert r.state == "source_validation_failed" and len(mem.all()) == 1
    assert len(r.proposal_attempts) == 1
    e = mem.all()[0]
    assert not e.applications and e.source_sample.task == sample().task
    assert e.admission_review.approved and r.proposal_attempts[0].error


def test_unused_candidate_still_admitted(tmp_path):
    mods, mem, run = runner(tmp_path)
    original = mods.initial_generator.generate

    def generate(task, selected, *, round_number=0):
        r = original(task, selected, round_number=round_number)
        for c in r.criteria:
            c.source_experiences = []
        return r

    mods.initial_generator.generate = generate
    r = run.run_sample(sample())
    assert len(mem.all()) == 1
    assert r.proposal_attempts[0].applications[0].feedback == "unused"
    assert not any(
        role in ("selector_checker", "generator_checker", "judge_checker")
        for role, _ in mods.backend.calls
    )


def test_review_technical_failure_never_becomes_content_feedback(tmp_path):
    mods, mem, run = runner(tmp_path)

    def fail(*args):
        raise LLMFormatError("invalid review JSON")

    mods.experience_checker.check = fail
    r = run.run_sample(sample())
    assert r.state == "candidate_review_failed" and len(r.proposal_attempts) == 1
    assert not mem.all() and r.proposal_attempts[0].review is None


def test_old_admission_prior_table_rejected():
    p = params().to_dict()
    del p["metadata"]["admission_protocol"]
    from verge.calculator import ProbabilityModel

    with pytest.raises(ValueError, match="admission_protocol"):
        ProbabilityModel.from_dict(p)


def test_rejected_review_cannot_write_memory(tmp_path):
    mods, mem, run = runner(tmp_path)
    e = experience()
    s = replace(sample(), sample_id=e.source_sample_id)
    with pytest.raises(ValueError, match="approved"):
        mem.add(e, s, ExperienceReview(False, "不合理"))


def test_source_unknown_preserves_prior_but_keeps_ledger(tmp_path):
    p = params()
    mods, mem, run = runner(tmp_path, "attempt_1_success", p)
    mods.checkers.check = lambda a: reports(("unknown", "correct", "correct"))
    r = run.run_sample(sample())
    e = mem.all()[0]
    assert e.q_posterior == p.q_prior
    assert e.applications[0].update_reason == "observation_unknown"
    assert e.evidence_counts["source_validation"] == 0
    assert e.downstream == []


def test_review_roundtrip_in_trajectory_and_memory(tmp_path):
    mods, mem, run = runner(tmp_path, "attempt_1_success")
    r = run.run_sample(sample())
    restored = list(TrajectoryLog(tmp_path).all())[0]
    e = ExperienceMemory(tmp_path).all()[0]
    assert restored.proposal_attempts[0].review == r.proposal_attempts[0].review
    assert e.admission_review == r.proposal_attempts[0].review


def test_transport_and_cli_register_new_role():
    from verge.llm import _OUTPUT_FIELDS
    from verge.cli import ROLES

    assert "experience_checker" in _OUTPUT_FIELDS and "experience_checker" in ROLES


@pytest.mark.parametrize("states", [(True, False), (False, True), (True, True)])
def test_source_record_accepts_all_valid_feedback_directions(tmp_path, states):
    p = params()
    mods, mem, run = runner(tmp_path, parameters=p)
    e = experience()
    s = replace(sample(), sample_id=e.source_sample_id)
    mem.add(e, s, ExperienceReview(True, "方法可试用"))
    a = app(e, states=[states, states], phase="source_validation")
    a.sample = s
    a.reports = reports()
    mem.record(a)
    assert a.update_reason == "updated" and len(mem.all()) == 1
    q = mem.get(e.experience_id).q_posterior
    mem.reestimate(p)
    assert mem.get(e.experience_id).q_posterior == pytest.approx(q)
