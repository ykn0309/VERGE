"""Experience Memory 与 trajectory 的本地写入契约。"""

from dataclasses import replace
import json

import pytest

from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, Criterion, CriterionJudgment, Rubric, TrajectoryRecord
from verge.calculator import calculate_rubric_judgment


def candidate(name: str, credit: float = 0.25) -> CandidateExperience:
    return CandidateExperience(name, f"原则 {name}", "可迁移条件", [], None, "PASS")


def test_empty_memory_creation_loading_and_snapshot(tmp_path):
    memory = ExperienceMemory(tmp_path)
    assert memory.all() == []
    assert (tmp_path / "experience_memory.json").exists()
    assert ExperienceMemory(tmp_path).all() == []
    with pytest.raises(KeyError):
        memory.get("unknown")




def test_zero_credit_add_merge_preserves_sources_without_zero_evidence(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = memory.fuse(candidate("zero", 0), "sample-1")
    stored = memory.get(added.experience_id)
    assert (stored.alpha, stored.beta, stored.evidence_history) == (1, 1, [])
    memory.fuse(candidate("again", 0), "sample-2", added.experience_id,
                merged_content="融合原则", merged_trigger="可迁移条件")
    stored = ExperienceMemory(tmp_path).get(added.experience_id)
    assert (stored.alpha, stored.beta, stored.evidence_history) == (1, 1, [])
    assert stored.source_trajectories == ["sample-1", "sample-2"]


def test_downstream_update_persists_at_initial_stage_even_if_revision_fails(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = memory.fuse(candidate("base", 0.2), "origin")
    memory.apply_downstream_credits("training-1", {added.experience_id: -0.5},
                                    evidence_weight=0.5)
    # 尚未写入 trajectory；后续修订失败不会回滚已经落盘的下游信用。
    assert not (tmp_path / "trajectories.jsonl").exists()
    loaded = ExperienceMemory(tmp_path).get(added.experience_id)
    assert (loaded.alpha, loaded.beta) == pytest.approx((1, 1.25))
    assert loaded.source_trajectories == ["origin"]
    assert [item.evidence_type for item in loaded.evidence_history] == [
        "downstream_credit"]
    assert loaded.evidence_history[-1].to_dict() == {
        "sample_id": "training-1", "evidence_type": "downstream_credit",
        "credit": -0.5, "evidence_weight": 0.5,
    }
    memory.apply_downstream_credits("training-2", {added.experience_id: 0.3})
    latest = ExperienceMemory(tmp_path).get(added.experience_id)
    assert (latest.alpha, latest.beta) == pytest.approx((1.3, 1.25))
    assert latest.evidence_history[-1].sample_id == "training-2"


def test_add_merges_within_same_trajectory_without_dedup(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = memory.fuse(candidate("a", 0.3), "same-sample")
    memory.fuse(candidate("b", 0.2), "same-sample", added.experience_id,
                merged_content="融合原则", merged_trigger="可迁移条件")
    stored = memory.get(added.experience_id)
    assert stored.source_trajectories == ["same-sample", "same-sample"]
    assert stored.alpha == pytest.approx(1)
    assert len(stored.evidence_history) == 0


def test_fusion_rejects_invalid_input_without_mutation(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = memory.fuse(candidate("valid"), "origin")
    original = (tmp_path / "experience_memory.json").read_bytes()
    bad = replace(candidate("invalid"), admission="FAIL")
    with pytest.raises(ValueError, match="admitted"):
        memory.fuse(bad, "sample")
    with pytest.raises(ValueError, match="admitted"):
        memory.fuse(added, "sample")
    with pytest.raises(ValueError, match="uncredited"):
        memory.fuse(replace(candidate("invalid"), linked_operation_ids=["op"], credit=0.2), "sample")
    with pytest.raises(KeyError):
        memory.fuse(candidate("unknown"), "sample", "not-in-memory")
    assert (tmp_path / "experience_memory.json").read_bytes() == original
    assert len(memory.all()) == 1


def test_downstream_invalid_or_zero_evidence_has_no_partial_update(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = memory.fuse(candidate("first"), "origin")
    key = added.experience_id
    with pytest.raises(KeyError):
        memory.apply_downstream_credits("sample", {key: 0.2, "missing": -0.1})
    assert ExperienceMemory(tmp_path).get(key).alpha == 1
    memory.apply_downstream_credits("sample", {key: 0})
    memory.apply_downstream_credits("sample", {key: -0.1}, evidence_weight=0)
    assert len(ExperienceMemory(tmp_path).get(key).evidence_history) == 0
    with pytest.raises(ValueError, match="evidence_weight"):
        memory.apply_downstream_credits("sample", {key: 0.3}, evidence_weight=2)
    assert ExperienceMemory(tmp_path).get(key).alpha == 1


def test_read_only_memory_is_frozen_and_returns_copies(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperienceMemory(tmp_path, read_only=True)
    writable = ExperienceMemory(tmp_path)
    added = writable.fuse(candidate("base"), "sample")
    frozen = ExperienceMemory(tmp_path, read_only=True)
    snapshot = frozen.get(added.experience_id)
    snapshot.alpha = 999
    snapshot.source_trajectories.append("not-stored")
    assert frozen.get(added.experience_id).alpha == 1
    assert frozen.get(added.experience_id).source_trajectories == ["sample"]
    for operation in (
        lambda: frozen.fuse(candidate("new"), "sample"),
        lambda: frozen.apply_downstream_credits("sample", {added.experience_id: 0.1}),
    ):
        with pytest.raises(ValueError, match="read-only"):
            operation()
    assert ExperienceMemory(tmp_path).get(added.experience_id).alpha == 1


def test_jsonl_trajectory_round_trip_and_inspect_last_sample(tmp_path):
    log = TrajectoryLog(tmp_path)
    assert list(log.all()) == []
    assert log.get("sample") is None
    rubric = Rubric("r0", None, [Criterion("c", "问题？", 1, ["exp"])])
    judgment = calculate_rubric_judgment(rubric, [CriterionJudgment("c", True, False)])
    first = TrajectoryRecord("sample", "initial_judged", ["exp"], ["exp"], rubric,
                             judgment, [], [], {})
    second = replace(first, state="complete")
    log.append(first)
    log.append(second)
    log.append(replace(first, sample_id="another"))
    loaded = list(TrajectoryLog(tmp_path).all())
    assert [item.to_dict() for item in loaded] == [item.to_dict() for item in
                                                  (first, second, replace(first, sample_id="another"))]
    assert log.get("sample") == second
    assert log.get("absent") is None
    lines = (tmp_path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["initial_rubric"]["criteria"][0]["question"] == "问题？"


def test_trajectory_log_rejects_malformed_lines(tmp_path):
    log = TrajectoryLog(tmp_path)
    log.path.write_text('{"unknown": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        list(log.all())
