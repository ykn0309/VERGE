"""New-only JSON models reject old protocols and malformed provenance."""

import pytest
from verge.models import (
    CandidateExperience,
    Criterion,
    CriterionJudgment,
    Credit,
    EvidenceRecord,
    Experience,
    PreferenceSample,
    Rubric,
    TrajectoryRecord,
)


@pytest.mark.parametrize("score", [True, 0, -1, float("nan"), float("inf"), float("-inf"), "2"])
def test_weight_is_finite_positive_number(score):
    with pytest.raises(ValueError):
        Criterion("c", "问题", score, [])


@pytest.mark.parametrize("weight", [.001, .999, 1.2, 100])
def test_relative_weight_round_trip_without_integer_range_limit(weight):
    c = Criterion("c", "问题", weight, [])
    assert Criterion.from_json(c.to_json()) == c


def test_old_importance_and_boolean_only_judgment_rejected():
    with pytest.raises(ValueError, match="importance_score"):
        Criterion.from_dict({"criterion_id": "c", "question": "问题", "importance_score": 8,
                             "source_experiences": []})
    with pytest.raises(ValueError, match="reason"):
        CriterionJudgment.from_dict({"criterion_id": "c", "chosen": True, "rejected": False})


@pytest.mark.parametrize("reason", [None, "", "  ", 1, True])
def test_both_answer_reasons_are_required_nonempty_text(reason):
    for field in ("chosen_reason", "rejected_reason"):
        data = dict(criterion_id="c", chosen=True, rejected=False,
                    chosen_reason="有依据", rejected_reason="缺少依据")
        data[field] = reason
        with pytest.raises(ValueError):
            CriterionJudgment.from_dict(data)


@pytest.mark.parametrize("sources", [["x", "x"], [""], [1], None])
def test_sources_are_distinct_ids(sources):
    with pytest.raises(ValueError):
        Criterion("c", "问题", 1, sources)


@pytest.mark.parametrize(
    "p,n", [(-0.1, 0), (0, -0.1), (0.6, 0.5), (float("nan"), 0), (True, 0)]
)
def test_credit_validation(p, n):
    with pytest.raises(ValueError):
        Credit(p, n)


def test_new_json_round_trips_and_no_net_field():
    models = [
        Credit(0.2, 0.2),
        CandidateExperience("candidate_x", "指导", "适用场景"),
        Experience("e", "指导", "*", 1, 1, [], []),
        TrajectoryRecord("s"),
        EvidenceRecord("s", "application_credit", 0.2, 0.3, "candidate_x"),
        Rubric(
            "r", [Criterion("c1", "要求", 2, []), Criterion("c2", "准确", 8, ["e"])]
        ),
    ]
    for model in models:
        assert type(model).from_json(model.to_json()) == model
    assert Credit(0.2, 0.2).to_dict() == {
        "positive_credit": 0.2,
        "negative_credit": 0.2,
    }


@pytest.mark.parametrize(
    "data",
    [
        {
            "candidate_id": "c",
            "content": "x",
            "trigger": "*",
            "credit": None,
            "linked_operation_ids": [],
        },
        {"candidate_id": "c", "content": "x", "trigger": "*", "admission": "PASS"},
        {
            "candidate_id": "c",
            "content": "x",
            "trigger": "*",
            "admission": "WRONG",
            "admission_reason": "x",
        },
    ],
)
def test_candidate_rejects_old_or_incomplete_schema(data):
    with pytest.raises(ValueError):
        CandidateExperience.from_dict(data)


def test_old_rubric_trajectory_and_evidence_rejected():
    with pytest.raises(ValueError):
        Rubric.from_dict({"rubric_id": "r", "base_rubric_id": None, "criteria": []})
    with pytest.raises(ValueError):
        TrajectoryRecord.from_dict({"sample_id": "s", "revision_attempts": []})
    with pytest.raises(ValueError):
        EvidenceRecord.from_dict(
            {
                "sample_id": "s",
                "evidence_type": "fusion_credit",
                "credit": 0.2,
                "evidence_weight": 1,
            }
        )
    with pytest.raises(ValueError):
        EvidenceRecord("s", "downstream_credit", 0, 0)
    with pytest.raises(ValueError):
        TrajectoryRecord("s", schema_version=1)


def test_confidence_derived_and_verified():
    e = Experience("e", "指导", "*", 1.3, 1.2, [], [])
    assert e.confidence == pytest.approx(0.52)
    data = e.to_dict()
    data["confidence"] = 0.8
    with pytest.raises(ValueError):
        Experience.from_dict(data)


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_old_trajectory_version_cannot_be_read_as_weighted_application(version):
    old = TrajectoryRecord("s").to_dict()
    old["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version 10"):
        TrajectoryRecord.from_dict(old)
