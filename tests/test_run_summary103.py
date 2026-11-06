"""A repaired rubric's outcome must not disappear from CLI/overview totals."""
from verge.cli import _completed_judgment, run
from verge.models import TrajectoryRecord, ProposalAttempt
from verge.memory import TrajectoryLog
from test_v2 import rubric, judged
from test_v2_io import config, write_config


def test_latest_completed_review_wins_and_incomplete_retry_preserves_it():
    r = rubric()
    bad = judged(r, ((False, True), (False, True)))
    good = judged(r)
    record = TrajectoryRecord("summary", initial_judgment=bad)
    assert _completed_judgment(record) is bad
    record.baseline_judgment = good
    assert _completed_judgment(record) is good
    record.proposal_attempts = [ProposalAttempt(1, final_judgment=bad)]
    assert _completed_judgment(record) is bad
    record.proposal_attempts.append(ProposalAttempt(2, final_judgment=good))
    assert _completed_judgment(record) is good
    record.proposal_attempts.append(ProposalAttempt(3, error="incomplete"))
    assert _completed_judgment(record) is good


def test_admission_does_not_imply_correct_and_empty_is_not_a_prediction():
    record = TrajectoryRecord("summary", state="experience_admitted")
    assert _completed_judgment(record) is None
    record.initial_judgment = judged(rubric(), ((False, True), (False, True)))
    assert not _completed_judgment(record).success


def test_cli_totals_use_completed_critic_results(tmp_path):
    c = config(tmp_path)
    c["dataset"]["config"]["paths"].pop("test")
    summary = run(write_config(tmp_path, c))
    records = list(TrajectoryLog(tmp_path / "run" / "artifacts").all())
    assert any(a.final_judgment for r in records for a in r.proposal_attempts)
    assert summary["final_correct"] == sum(
        bool(_completed_judgment(r) and _completed_judgment(r).success) for r in records
    )
    assert summary["final_correct"] > summary["initial_correct"]
