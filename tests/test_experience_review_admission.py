"""Rubric improvement gates extraction; review/source validation have separate budgets."""

from copy import deepcopy
from dataclasses import replace
import pytest
from verge.models import ExperienceReview
from verge.modules import make_fake_modules
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.pipeline import PipelineRunner
from verge.calculator import ProbabilityModel
from verge.llm import LLMFormatError
from test_v2 import sample, params, experience, reports


def runner(tmp_path, scenario="attempt_1_success", parameters=None):
    mods = make_fake_modules(scenario)
    mem = ExperienceMemory(tmp_path, parameters=parameters)
    return (
        mods,
        mem,
        PipelineRunner(mods, mem, TrajectoryLog(tmp_path), revision_enabled=False),
    )


def test_second_repair_uses_fixed_initial_baseline_then_one_experience_rewrite(
    tmp_path,
):
    m, mem, r = runner(tmp_path, "attempt_2_success")
    x = r.run_sample(sample())
    assert len(x.proposal_attempts) == 2
    assert [a.margin_before for a in x.proposal_attempts] == [0, 0]
    assert [a.margin_gain for a in x.proposal_attempts] == [0, 1]
    assert len(x.proposal_attempts[-1].extractions[0]["reviews"]) == 2
    calls = [role for role, _ in m.backend.calls]
    assert calls.count("critic_repair") == 2 and calls.count("critic_rewrite_batch") == 1
    assert calls.index("critic_extract") > calls.index("critic_assess")
    assert len(mem.all()) == 1


def test_no_improvement_never_extracts_or_admits(tmp_path):
    m, mem, r = runner(tmp_path, "final_failure")
    x = r.run_sample(sample())
    assert x.state == "repair_not_improved" and len(x.proposal_attempts) == 2
    assert not mem.all() and not any(
        role == "critic_extract" for role, _ in m.backend.calls
    )


def test_rejected_twice_no_admission_or_source_application(tmp_path):
    m, mem, r = runner(tmp_path)
    m.experience_checker.check_many = lambda sample,evidence,es: {e.experience_id:ExperienceReview(False,"多个独立方法") for e in es}
    x = r.run_sample(sample())
    entry = x.proposal_attempts[0].extractions[0]
    assert x.state == "no_admitted_experience" and len(entry["reviews"]) == 2
    assert (
        not mem.all()
        and sum(role == "initial_generator" for role, _ in m.backend.calls) == 1
    )


def test_admission_keeps_prior_and_does_not_run_source_generator_judge(tmp_path):
    p = params()
    m, mem, r = runner(tmp_path, parameters=p)
    x = r.run_sample(sample())
    e = mem.all()[0]
    assert x.state == "experience_admitted"
    assert e.q_prior == e.q_posterior == p.q_prior and e.q_log_odds is None
    assert not e.applications and not x.proposal_attempts[0].applications
    assert (
        x.proposal_attempts[0].rubric is None
        and x.proposal_attempts[0].judgment is None
    )
    roles = [role for role, _ in m.backend.calls]
    assert roles.count("initial_generator") == 1
    assert (
        roles.count("judge") == 4
    )  # initial rubric + Critic repaired rubric, two sides each
    assert roles.count("judge_checker") == 0  # removed from the Critic chain
    assert "selector_checker" not in roles and "generator_checker" not in roles
    assert roles[-1] == "experience_checker_batch"
    mem.reestimate(p)
    assert mem.all()[0].q_posterior == p.q_prior


def test_new_experience_first_quality_update_comes_from_later_task(tmp_path):
    p = params()
    m, mem, r = runner(tmp_path, parameters=p)
    r.run_sample(sample())
    e = mem.all()[0]
    assert e.q_posterior == p.q_prior and not e.applications
    # The next task uses its actual public method; no source/admission evidence is added.
    m.backend.scenario = "initial_success"
    result = r.run_sample(sample(1))
    e = mem.get(e.experience_id)
    assert result.state == "initial_success" and len(e.applications) == 1
    assert (
        e.applications[0].phase == "downstream"
        and e.applications[0].update_reason == "updated"
    )
    assert (
        e.evidence_counts["source_validation"] == 0
        and e.evidence_counts["downstream"] == 1
    )
    expected = e.q_posterior
    mem.reestimate(p)
    assert mem.get(e.experience_id).q_posterior == pytest.approx(expected)


def test_review_technical_failure_not_content_rejection(tmp_path):
    m, mem, r = runner(tmp_path)

    def fail(*a, **kw):
        raise LLMFormatError("invalid review")

    m.experience_checker.check_many = fail
    x = r.run_sample(sample())
    assert not mem.all() and x.proposal_attempts[0].error
    assert not x.proposal_attempts[0].extractions[0]["reviews"]


def test_old_admission_prior_table_rejected():
    p = params().to_dict()
    p["metadata"]["admission_protocol"] = "experience_checker_v1"
    with pytest.raises(ValueError, match="admission_protocol"):
        ProbabilityModel.from_dict(p)


def test_rejected_review_cannot_write_memory(tmp_path):
    m, mem, r = runner(tmp_path)
    e = experience()
    s = replace(sample(), sample_id=e.source_sample_id)
    with pytest.raises(ValueError, match="approved"):
        mem.add(e, s, ExperienceReview(False, "不合理"))


def test_review_roundtrip_in_trajectory_and_memory(tmp_path):
    m, mem, r = runner(tmp_path)
    x = r.run_sample(sample())
    assert list(TrajectoryLog(tmp_path).all())[0].to_dict() == x.to_dict()
    assert ExperienceMemory(tmp_path).all()[0].admission_review.approved
