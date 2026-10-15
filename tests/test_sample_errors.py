"""Exhausted model requests isolate samples; storage and program failures stay global."""

import pytest
from test_pipeline import runner, sample
from verge.llm import LLMFormatError, LLMRefusalError, LLMSampleError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import TrajectoryRecord


@pytest.mark.parametrize(
    "role,method",
    [
        ("critic", "propose"),
        ("initial_generator", "apply"),
    ],
)
def test_model_refusal_preserves_prior_results_and_continues(tmp_path, role, method):
    r = runner(tmp_path)
    obj = getattr(r.modules, role)
    original = getattr(obj, method)
    n = 0

    def refusing(*a, **kw):
        nonlocal n
        n += 1
        if n == 1:
            raise LLMRefusalError("model refusal")
        return original(*a, **kw)

    setattr(obj, method, refusing)
    a, b = r.run([sample("a"), sample("b")])
    assert a.state == "sample_error" and a.initial_judgment is not None
    assert b.state == "application_success"


@pytest.mark.parametrize("role,method,stage", [
    ("retriever", "retrieve", "retrieval"),
    ("selector", "select", "selection"),
    ("initial_generator", "generate", "initial_generator"),
    ("judge", "judge", "initial_judge"),
    ("critic", "propose", "application_1.critic"),
    ("initial_generator", "apply", "application_1.generator"),
])
def test_exhausted_request_is_logged_and_next_sample_runs(tmp_path, role, method, stage):
    r = runner(tmp_path)
    component = getattr(r.modules, role)
    original = getattr(component, method)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMSampleError("request failed after 3 attempts (ReadTimeout)")
        return original(*args, **kwargs)

    setattr(component, method, fail_once)
    failed, following = r.run([sample("failed"), sample("following")])
    assert failed.state == "sample_error" and failed.sample_error["stage"] == stage
    assert failed.admitted_candidate_id is None
    assert following.state == "application_success"
    assert list(TrajectoryLog(tmp_path).all()) == [failed, following]
    assert TrajectoryRecord.from_json(failed.to_json()) == failed
    if stage.endswith(".critic"):
        assert failed.proposal_attempts[-1].error
    if stage.endswith(".generator"):
        assert failed.application_attempts[-1].error


def test_application_timeout_saves_r0_credit_once_and_continues(tmp_path):
    from test_bounded_experience import existing_runner
    r, eid = existing_runner(tmp_path, overall_success=False)
    original = r.modules.judge.judge
    failed = False

    def judge(rubric, *args, **kwargs):
        nonlocal failed
        if rubric.rubric_id != "existing_r0" and not failed:
            failed = True
            raise LLMSampleError("judge: ReadTimeout after 3 attempts")
        return original(rubric, *args, **kwargs)

    r.modules.judge.judge = judge
    first, second = r.run([sample("failed"), sample("following")])
    assert first.state == "sample_error" and first.sample_error["stage"] == "application_1.judge"
    assert first.initial_judgment and first.downstream_credits and not first.admitted_experience_id
    assert first.application_attempts[-1].error
    assert second.state != "sample_error"
    restored = ExperienceMemory(tmp_path).get(eid)
    assert (restored.alpha, restored.beta) == pytest.approx((1.8, 1.2))
    assert [e.sample_id for e in restored.evidence_history] == ["seed", "failed", "following"]
    assert [c.sample_id for c in restored.cases] == ["seed", "failed", "following"]
    assert list(TrajectoryLog(tmp_path).all()) == [first, second]


def test_embedding_output_failure_skips_only_current_sample(tmp_path):
    r = runner(tmp_path)
    original = r.modules.retriever.retrieve
    r.modules.retriever.retrieve = lambda *args, **kwargs: (_ for _ in ()).throw(LLMFormatError("embedding: invalid vector response"))
    failed = r.run_sample(sample("failed"))
    r.modules.retriever.retrieve = original
    following = r.run_sample(sample("following"))
    assert failed.sample_error["stage"] == "retrieval"
    assert following.state == "application_success"


@pytest.mark.parametrize("stage", ["retrieval", "credit", "memory", "trajectory"])
def test_non_role_valueerrors_never_become_sample_error(tmp_path, monkeypatch, stage):
    r = runner(tmp_path)

    def fatal(*a, **kw):
        raise ValueError("internal failure")

    if stage == "retrieval":
        r.modules.retriever.retrieve = fatal
    elif stage == "credit":
        monkeypatch.setattr("verge.pipeline.calculate_experience_credits", fatal)
    elif stage == "memory":
        r.memory.apply_downstream_credits = fatal
    else:
        r.trajectories.append = fatal
    with pytest.raises(ValueError, match="internal failure"):
        r.run_sample(sample())
