"""Output failures isolate samples; storage, retrieval and programming failures stay global."""

import pytest
from test_pipeline import runner, sample
from verge.llm import LLMRefusalError
from verge.models import CandidateExperience


@pytest.mark.parametrize(
    "role,method",
    [
        ("critic", "propose"),
        ("initial_generator", "apply"),
        ("fusion_judge", "find_match"),
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


def test_illegal_fusion_cannot_write_and_leaves_actual_success_audit(tmp_path):
    r = runner(tmp_path)
    from verge.models import FusionResult

    r.modules.fusion_judge.find_match = lambda *a: FusionResult(10, "合并", "条件")
    rec = r.run_sample(sample())
    assert rec.state == "sample_error" and rec.sample_error["stage"] == "fusion_judge"
    assert rec.application_attempts[0].judgment.success and not r.memory.all()
