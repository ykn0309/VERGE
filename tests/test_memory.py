"""Persistent dual evidence and text versions; held-out is read-only."""

from dataclasses import replace
import json
import pytest
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, Credit, TrajectoryRecord


def candidate(sid="candidate_1"):
    return CandidateExperience(
        sid,
        "独立关键要求标准",
        "明确关键要求",
        admission="PASS",
        admission_reason="表达可用",
    )


def test_add_and_merge_preserve_id_and_append_both_channels(tmp_path):
    m = ExperienceMemory(tmp_path)
    added = m.fuse(candidate(), "s1", application_credit=Credit(0.2, 0.2))
    m.apply_downstream_credits("s2", {added.experience_id: Credit(0.3, 0.2)})
    merged = m.fuse(
        candidate("candidate_2"),
        "s3",
        added.experience_id,
        merged_content="完整融合指导",
        merged_trigger="更精确条件",
        application_credit=Credit(0, 0.1),
    )
    e = ExperienceMemory(tmp_path).get(added.experience_id)
    assert merged.experience_id == added.experience_id
    assert (e.alpha, e.beta) == pytest.approx((1.5, 1.5))
    assert e.evidence_amount == pytest.approx(1)
    assert e.source_trajectories == ["s1", "s3"]
    assert [ev.evidence_type for ev in e.evidence_history] == [
        "application_credit",
        "downstream_credit",
        "application_credit",
    ]
    assert len(e.fusion_history) == 2
    assert e.fusion_history[1].previous_content == candidate().content
    assert e.fusion_history[1].content == "完整融合指导"
    assert e.fusion_history[1].credit == Credit(0, 0.1)
    assert json.loads(m.path.read_text())["schema_version"] == 2


def test_zero_application_still_adds_text_without_evidence(tmp_path):
    m = ExperienceMemory(tmp_path)
    f = m.fuse(candidate(), "s")
    e = m.get(f.experience_id)
    assert (
        e.alpha == e.beta == 1 and not e.evidence_history and len(e.fusion_history) == 1
    )
    before = m.path.read_bytes()
    m.apply_downstream_credits("s2", {e.experience_id: Credit()})
    assert m.path.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: replace(c, admission=None, admission_reason=None),
        lambda c: replace(c, admission="FAIL"),
    ],
)
def test_unadmitted_candidate_cannot_write(tmp_path, mutation):
    m = ExperienceMemory(tmp_path)
    with pytest.raises(ValueError):
        m.fuse(mutation(candidate()), "s")
    assert not m.all()


def test_readonly_does_not_write_and_rejects_mutations(tmp_path):
    m = ExperienceMemory(tmp_path)
    e = m.fuse(candidate(), "seed")
    before = m.path.read_bytes()
    frozen = ExperienceMemory(tmp_path, read_only=True)
    for action in [
        lambda: frozen.apply_downstream_credits("s", {e.experience_id: Credit(0.2, 0)}),
        lambda: frozen.fuse(candidate(), "s"),
        lambda: frozen.fuse(candidate("c2"), "s"),
    ]:
        with pytest.raises(ValueError):
            action()
    assert m.path.read_bytes() == before


def test_old_memory_rejected_and_trajectory_roundtrip(tmp_path):
    (tmp_path / "experience_memory.json").write_text('{"experiences": []}')
    with pytest.raises(ValueError):
        ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    r = TrajectoryRecord("s", "initial_failure")
    log.append(r)
    assert log.get("s") == r


def test_merge_validation_happens_before_persistence(tmp_path):
    m = ExperienceMemory(tmp_path)
    e = m.fuse(candidate(), "s")
    before = m.path.read_bytes()
    with pytest.raises(ValueError):
        m.fuse(candidate(), "next", e.experience_id, merged_content="missing trigger")
    assert m.path.read_bytes() == before
