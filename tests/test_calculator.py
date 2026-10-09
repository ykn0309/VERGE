"""Normalized Boolean scores and undiscounted, separate evidence channels."""

from dataclasses import replace
import pytest
from verge.calculator import (
    calculate_experience_credits,
    calculate_rubric_judgment,
    normalize_weights,
    apply_beta_evidence,
)
from verge.models import (
    Criterion,
    CriterionJudgment,
    Credit,
    EvidenceRecord,
    Experience,
    Rubric,
)


def rubric_case(scores, signs, sources):
    rubric = Rubric(
        "r",
        [
            Criterion(str(i), f"问题{i}", s, ids)
            for i, (s, ids) in enumerate(zip(scores, sources))
        ],
    )
    judgment = calculate_rubric_judgment(
        rubric,
        [
            CriterionJudgment(str(i), sign == 1, sign == -1)
            for i, sign in enumerate(signs)
        ],
    )
    return rubric, judgment


@pytest.mark.parametrize(
    "scores,signs,expected",
    [
        ([2, 2, 6], [1, -1, 0], Credit(0.2, 0.2)),
        ([3, 2, 5], [1, -1, 0], Credit(0.3, 0.2)),
        ([2, 8], [1, 0], Credit(0.2, 0)),
        ([2, 8], [-1, 0], Credit(0, 0.2)),
    ],
)
def test_separate_channels_never_net_before_beta_update(scores, signs, expected):
    r, j = rubric_case(scores, signs, [["e"]] * len(scores))
    credit = calculate_experience_credits(r, j)["e"]
    assert credit.positive_credit == pytest.approx(expected.positive_credit)
    assert credit.negative_credit == pytest.approx(expected.negative_credit)
    e = Experience("e", "指导", "*", 1, 1, [], [])
    updated = apply_beta_evidence(
        e,
        EvidenceRecord(
            "s", "downstream_credit", credit.positive_credit, credit.negative_credit
        ),
    )
    assert updated.alpha == pytest.approx(1 + expected.positive_credit)
    assert updated.beta == pytest.approx(1 + expected.negative_credit)
    assert updated.evidence_amount == pytest.approx(expected.amount)
    assert e.alpha == e.beta == 1


def test_shared_sources_conserve_total_mass():
    r, j = rubric_case([3, 2, 5], [1, -1, 1], [["a", "b"], ["a"], []])
    credits = calculate_experience_credits(r, j)
    assert credits == {"a": Credit(0.15, 0.2), "b": Credit(0.15, 0)}
    assert sum(c.amount for c in credits.values()) == pytest.approx(0.5)
    assert j.margin == pytest.approx(0.6)


@pytest.mark.parametrize("sources", [[[], []], [["e"], ["e"]]])
def test_zero_channels_no_record(sources):
    r, j = rubric_case([2, 8], [0, 0], sources)
    assert calculate_experience_credits(r, j) == {}


def test_balanced_evidence_pulls_asymmetric_beta_toward_half():
    e = Experience("e", "指导", "*", 3, 1, [], [])
    updated = apply_beta_evidence(e, EvidenceRecord("s", "downstream_credit", 0.2, 0.2))
    assert 0.5 < updated.confidence < e.confidence
    assert updated.evidence_amount == pytest.approx(e.evidence_amount + 0.4)


@pytest.mark.parametrize(
    "signs,margin,success",
    [([1, -1], 0, False), ([1, 0], 0.5, True), ([-1, 0], -0.5, False)],
)
def test_scoring_tie_and_boolean_interface(signs, margin, success):
    r, j = rubric_case([1, 1], signs, [[], []])
    assert j.margin == margin and j.success == success
    assert sum(normalize_weights(r).values()) == 1


def test_precision_tie_does_not_become_success():
    r, j = rubric_case([1, 2, 3], [1, 1, -1], [[], [], []])
    assert j.margin == 0 and not j.success


@pytest.mark.parametrize(
    "field,value",
    [
        ("margin", 0.8),
        ("normalized_weights", {"0": 0.8, "1": 0.2}),
        ("success", False),
        ("rubric_id", "other"),
    ],
)
def test_calculator_rejects_corrupt_judgment(field, value):
    r, j = rubric_case([2, 8], [1, 0], [["e"], []])
    with pytest.raises(ValueError):
        calculate_experience_credits(r, replace(j, **{field: value}))


@pytest.mark.parametrize(
    "answers",
    [
        [CriterionJudgment("0", True, False)],
        [CriterionJudgment("0", True, False)] * 2,
        [CriterionJudgment("0", True, False), CriterionJudgment("other", False, False)],
    ],
)
def test_exact_criterion_coverage(answers):
    r, _ = rubric_case([1, 1], [0, 0], [[], []])
    with pytest.raises(ValueError):
        calculate_rubric_judgment(r, answers)
