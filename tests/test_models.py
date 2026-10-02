"""领域数据边界和 JSON 往返测试。"""

import json
from dataclasses import replace

import pytest

from verge.models import (
    CandidateExperience,
    Criterion,
    CriterionJudgment,
    EvidenceRecord,
    Experience,
    InvalidRevisionAttempt,
    PreferenceSample,
    RevisionAttempt,
    RevisionOperation,
    Rubric,
    RubricJudgment,
    TrajectoryRecord,
)


def criterion(criterion_id="c1", score=8):
    return Criterion(criterion_id, "是否满足任务？", score, [])


def judgment(rubric_id="r0", margin=0.0):
    return RubricJudgment(rubric_id, {"c1": 1.0}, [CriterionJudgment("c1", True, False)],
                         1.0, 0.0, margin, margin > 0)


def test_core_objects_json_round_trip():
    sample = PreferenceSample("sample_1", "任务", "优", "劣", {"split": "train", "level": 1})
    evidence = EvidenceRecord("sample_0", "revision_credit", 0.3, 1.0)
    downstream = EvidenceRecord("sample_1", "downstream_credit", 0.4, 1.0)
    exp = Experience("exp_1", "原则", "触发条件", 1.7, 1.0, ["sample_0"],
                     [evidence, downstream])
    rubric = Rubric("r0", None, [criterion()])
    revised = Rubric("r1", "r0", [criterion(score=9), criterion("c2", 10)])
    op = RevisionOperation("o1", "ADD", None, criterion("c2", 10))
    modified = RevisionOperation("o2", "MODIFY", "c1", criterion(score=9), criterion())
    candidate = CandidateExperience("candidate_1", "原则", "条件", ["o1"], 0.3,
                                    "PASS", "MERGE", "exp_1")
    attempt = RevisionAttempt(1, "缺少条件", [op, modified], revised, judgment("r1", 0.2))
    trajectory = TrajectoryRecord(
        "sample_1", "ready", ["exp_1"], ["exp_1"], rubric, judgment(),
        [attempt], [candidate], {"o1": 0.4},
    )
    for obj in (sample, evidence, exp, rubric, judgment(), op, modified, candidate,
                downstream, attempt, trajectory):
        assert type(obj).from_json(obj.to_json()) == obj
        assert type(obj).from_dict(json.loads(obj.to_json())) == obj
    assert exp.confidence == pytest.approx(1.7 / 2.7)
    assert exp.evidence_amount == pytest.approx(0.7)
    assert set(evidence.to_dict()) == {"sample_id", "evidence_type", "credit", "evidence_weight"}
    assert "committed" not in trajectory.to_dict()
    assert "executed_operations" not in attempt.to_dict()
    assert "operation_bindings" not in trajectory.to_dict()
    assert "existing_experience_credits" not in trajectory.to_dict()
    assert "provisional_revision_credits" not in trajectory.to_dict()
    assert "pending_memory_changes" not in trajectory.to_dict()
    assert "admission_results" not in trajectory.to_dict()
    assert "fusion_results" not in trajectory.to_dict()
    assert "source_criterion_ids" not in downstream.to_dict()
    assert "normalized_weight" not in criterion().to_dict()
    assert "margin" not in criterion().to_dict()
    assert "credit" not in criterion().to_dict()


def test_invalid_revision_audit_round_trip_and_legacy_trajectory_default():
    failed = InvalidRevisionAttempt(1, "原始反馈", ["revised rubric must contain 2..8 criteria"])
    trajectory = TrajectoryRecord("sample", "final_failure", [], [], None, None, [], [], {}, [failed])
    assert TrajectoryRecord.from_json(trajectory.to_json()) == trajectory
    legacy = trajectory.to_dict()
    legacy.pop("invalid_revision_attempts")
    assert TrajectoryRecord.from_dict(legacy).invalid_revision_attempts == []
    with pytest.raises(ValueError, match="validation errors"):
        InvalidRevisionAttempt(1, "反馈", [])


@pytest.mark.parametrize("score", [0, 11, 1.5, True, "5", None])
def test_invalid_importance_score(score):
    with pytest.raises(ValueError, match="importance_score"):
        criterion(score=score)


@pytest.mark.parametrize("operation_type", ["UPDATE", "add", "", None])
def test_invalid_operation_type(operation_type):
    with pytest.raises(ValueError, match="operation_type"):
        RevisionOperation("o1", operation_type, None, criterion("c2"))


@pytest.mark.parametrize("op_type,target,updated", [
    ("ADD", "c1", criterion("c2")),
    ("ADD", None, None),
    ("DELETE", None, None),
    ("DELETE", "c1", criterion("c2")),
    ("MODIFY", "c1", None),
    ("MODIFY", "c1", criterion("c2")),
])
def test_invalid_revision_operation_shapes(op_type, target, updated):
    with pytest.raises(ValueError):
        RevisionOperation("o1", op_type, target, updated)


@pytest.mark.parametrize("op_type,target,updated,before", [
    ("ADD", None, criterion("c2"), criterion()),
    ("DELETE", "c1", None, criterion("c2")),
    ("MODIFY", "c1", criterion(), criterion("c2")),
])
def test_invalid_before_criterion(op_type, target, updated, before):
    with pytest.raises(ValueError):
        RevisionOperation("o1", op_type, target, updated, before)


def test_valid_revision_operation_shapes():
    assert RevisionOperation("add", "ADD", None, criterion("c2"))
    assert RevisionOperation("delete", "DELETE", "c1", None)
    assert RevisionOperation("modify", "MODIFY", "c1", criterion())
    assert RevisionOperation("add", "ADD", None, criterion("c2"), None)
    assert RevisionOperation("delete", "DELETE", "c1", None, criterion())
    assert RevisionOperation("modify", "MODIFY", "c1", criterion(score=9), criterion())


def test_optional_fields_and_empty_collections():
    rubric = Rubric("r0", None, [])
    assert Rubric.from_json(rubric.to_json()) == rubric
    assert TrajectoryRecord("sample_1", "new", [], [], None, None, [], [], {}).to_dict()["initial_rubric"] is None
    candidate = CandidateExperience.from_dict({"candidate_id": "candidate_1", "content": "原则", "trigger": "条件"})
    assert candidate.linked_operation_ids == []
    assert candidate.credit is None
    assert candidate.admission is None
    assert candidate.fusion is None
    assert candidate.experience_id is None
    assert CandidateExperience.from_json(candidate.to_json()) == candidate
    with pytest.raises(ValueError, match="linked_operation_ids"):
        CandidateExperience("candidate_1", "原则", "条件", ["o1", "o1"])
    for invalid in (float("nan"), True, "0.3"):
        with pytest.raises(ValueError, match="credit"):
            CandidateExperience("candidate_1", "原则", "条件", ["o1"], invalid)
    with pytest.raises(ValueError, match="duplicate criterion_id"):
        Rubric("r0", None, [criterion(), criterion()])


def test_rejects_invalid_payloads_at_deserialization_boundary():
    data = criterion().to_dict()
    for mutation in ({**data, "importance_score": True},
                     {**data, "unknown": 1},
                     {key: val for key, val in data.items() if key != "question"}):
        with pytest.raises(ValueError):
            Criterion.from_dict(mutation)
    with pytest.raises(ValueError):
        Rubric.from_dict({"rubric_id": "r0", "base_rubric_id": None,
                          "criteria": [{**data, "importance_score": 11}]})
    with pytest.raises(ValueError, match="unknown"):
        EvidenceRecord.from_dict({**EvidenceRecord("s", "revision_credit", 0.2, 1).to_dict(),
                                  "source_id": "candidate_1"})
    with pytest.raises(ValueError, match="confidence"):
        Experience.from_dict({**Experience("e", "x", "t", 1, 1, [], []).to_dict(),
                              "confidence": 0.9})


def test_invalid_result_values():
    with pytest.raises(ValueError, match="success"):
        RubricJudgment("r0", {}, [], 0, 0, 0, True)
    ready = CandidateExperience("c", "principle", "trigger", ["o1"], 0.2)
    with pytest.raises(ValueError, match="admission"):
        replace(ready, admission="UNKNOWN")
    with pytest.raises(ValueError, match="fusion"):
        replace(ready, admission="PASS", fusion="UNKNOWN")
    assert CandidateExperience("c", "principle", "trigger", admission="PASS").credit is None
    with pytest.raises(ValueError, match="linked_operation_ids"):
        CandidateExperience("c", "principle", "trigger", credit=0.2)
    with pytest.raises(ValueError, match="PASS"):
        replace(ready, admission="FAIL", fusion="MERGE", experience_id="e")
    with pytest.raises(ValueError, match="experience_id"):
        replace(ready, admission="PASS", fusion="MERGE")
    with pytest.raises(ValueError, match="fusion"):
        replace(ready, experience_id="e")
    assert replace(ready, admission="FAIL").fusion is None
    assert replace(ready, admission="PASS", fusion="ADD", experience_id="e").credit == 0.2
    with pytest.raises(ValueError, match="admission_reason"):
        replace(ready, admission_reason="没有准入决定")
    with pytest.raises(ValueError, match="admission_reason"):
        replace(ready, admission="FAIL", admission_reason="  ")
    reviewed = replace(ready, admission="FAIL", admission_reason="不满足可迁移性。")
    assert CandidateExperience.from_dict(reviewed.to_dict()) == reviewed
    # 旧实验轨迹没有理由字段；回放时保留判断，但不得伪造历史理由。
    old = reviewed.to_dict()
    old.pop("admission_reason")
    assert CandidateExperience.from_dict(old).admission_reason is None
    with pytest.raises(ValueError, match="alpha"):
        Experience("e", "x", "t", 0, 1, [], [])
