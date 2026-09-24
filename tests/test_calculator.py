"""确定性公式、输入边界与信用预算测试。"""

from dataclasses import replace

import pytest

from verge.calculator import (
    allocate_revision_credit,
    apply_beta_evidence,
    calculate_existing_experience_credit,
    calculate_operation_contributions,
    calculate_revision_budget,
    calculate_rubric_judgment,
    normalize_weights,
)
from verge.models import (
    CandidateExperience,
    Criterion,
    CriterionJudgment,
    EvidenceRecord,
    Experience,
    RevisionOperation,
    Rubric,
)


def c(key, score=1, sources=None, question=None):
    return Criterion(key, question or key, score, sources or [])


def j(rubric, *answers):
    return calculate_rubric_judgment(
        rubric, [CriterionJudgment(key, chosen, rejected)
                 for key, chosen, rejected in answers],
    )


def test_weight_normalization_is_local_to_full_rubric():
    a = Rubric("a", None, [c("c1", 5), c("c2", 3), c("c3", 2)])
    b = Rubric("b", "a", [c("c1", 5), c("c3", 2)])
    assert normalize_weights(a) == {"c1": 0.5, "c2": 0.3, "c3": 0.2}
    assert normalize_weights(b)["c1"] == pytest.approx(5 / 7)
    with pytest.raises(ValueError, match="empty rubric"):
        normalize_weights(Rubric("empty", None, []))


@pytest.mark.parametrize("answers,expected_margin,success", [
    ([("c1", True, False)], 1.0, True),
    ([("c1", False, True)], -1.0, False),
    ([("c1", True, True)], 0.0, False),
])
def test_judgment_success_failure_and_tie(answers, expected_margin, success):
    rubric = Rubric("r0", None, [c("c1")])
    result = j(rubric, *answers)
    assert result.margin == expected_margin
    assert result.success is success
    assert result.chosen_score - result.rejected_score == pytest.approx(result.margin)


def test_multicriterion_tie_cannot_round_into_success_or_unlock_budget():
    rubric = Rubric("tie", None, [c("a", 1), c("b", 1), c("c", 4), c("d", 5)])
    tied = j(rubric, ("a", True, False), ("b", False, False),
             ("c", True, False), ("d", False, True))
    assert tied.margin == 0
    assert not tied.success
    revised = Rubric("revision", "tie", rubric.criteria)
    fabricated = replace(tied, rubric_id="revision", margin=1e-10, success=True)
    with pytest.raises(ValueError, match="does not match"):
        calculate_revision_budget(rubric, tied, revised, fabricated)


def test_judgment_rejects_missing_extra_or_duplicate_criteria():
    rubric = Rubric("r0", None, [c("c1"), c("c2")])
    for answers in ([("c1", True, False)],
                    [("c1", True, False), ("c1", False, True)],
                    [("c1", True, False), ("extra", True, False)]):
        with pytest.raises(ValueError, match="match the rubric"):
            j(rubric, *answers)


def test_existing_credit_splits_provenance_and_ignores_unreferenced_and_zero():
    rubric = Rubric("r0", None, [c("c1", 5, ["a", "b"]),
                                 c("c2", 3, ["a"]), c("c3", 2)])
    judgment = j(rubric, ("c1", False, True), ("c2", True, True),
                 ("c3", True, False))
    assert calculate_existing_experience_credit(rubric, judgment) == {"a": -0.25, "b": -0.25}
    with pytest.raises(ValueError, match="duplicate source"):
        dup = Rubric("dup", None, [c("x", 1, ["a", "a"])])
        calculate_existing_experience_credit(dup, j(dup, ("x", True, False)))
    with pytest.raises(ValueError, match="does not match"):
        calculate_existing_experience_credit(rubric, replace(judgment, rubric_id="other"))
    with pytest.raises(ValueError, match="does not match"):
        calculate_existing_experience_credit(rubric, replace(judgment, chosen_score=0))


def test_existing_credit_cancels_to_zero_without_evidence():
    rubric = Rubric("r0", None, [c("a", 1, ["e"]), c("b", 1, ["e"])])
    assert calculate_existing_experience_credit(
        rubric, j(rubric, ("a", True, False), ("b", False, True))) == {}


def test_add_delete_modify_use_direct_contributions_not_total_margin_change():
    initial = Rubric("r0", None, [c("gone", 2), c("edited", 2), c("same", 1)])
    revised = Rubric("r1", "r0", [c("edited", 4, question="changed"),
                                   c("same", 1), c("new", 1)])
    first = j(initial, ("gone", False, True), ("edited", False, True), ("same", True, False))
    second = j(revised, ("edited", True, False), ("same", True, False),
               ("new", True, False))
    ops = [RevisionOperation("del", "DELETE", "gone", None, initial.criteria[0]),
           RevisionOperation("mod", "MODIFY", "edited", revised.criteria[0], initial.criteria[1]),
           RevisionOperation("add", "ADD", None, revised.criteria[2])]
    delta = calculate_operation_contributions(initial, first, revised, second, ops)
    assert delta == pytest.approx({"del": 0.4, "mod": 4 / 6 + 0.4, "add": 1 / 6})
    assert sum(delta.values()) != pytest.approx(second.margin - first.margin)
    with pytest.raises(ValueError, match="before_criterion"):
        calculate_operation_contributions(initial, first, revised, second,
                                          [replace(ops[0], before_criterion=None), *ops[1:]])
    with pytest.raises(ValueError, match="unrecorded"):
        calculate_operation_contributions(initial, first, revised, second, ops[:2])
    with pytest.raises(ValueError, match="duplicate operation"):
        calculate_operation_contributions(initial, first, revised, second, [*ops, ops[-1]])
    changed_answers = j(revised, ("edited", True, False), ("same", False, False),
                        ("new", True, False))
    with pytest.raises(ValueError, match="reuse Boolean judgments"):
        calculate_operation_contributions(initial, first, revised, changed_answers, ops)
    score_only = Rubric("score_only", "r0", [c("gone", 2), c("edited", 4), c("same", 1)])
    drift = j(score_only, ("gone", False, True), ("edited", True, False),
              ("same", True, False))
    with pytest.raises(ValueError, match="reuse Boolean judgments"):
        calculate_operation_contributions(
            initial, first, score_only, drift,
            [RevisionOperation("score", "MODIFY", "edited", score_only.criteria[1], initial.criteria[1])])
    identical = RevisionOperation("noop", "MODIFY", "edited", initial.criteria[1], initial.criteria[1])
    with pytest.raises(ValueError, match="no-op MODIFY"):
        calculate_operation_contributions(initial, first, Rubric("copy", "r0", initial.criteria),
                                          replace(first, rubric_id="copy"), [identical])


def test_revision_ignores_provenance_but_rejects_source_only_modify():
    initial = Rubric("r0", None, [c("bad", 1, ["old"]), c("good", 1),
                                  c("neutral1", 1), c("neutral2", 1)])
    revised = Rubric("r1", "r0", [c("bad", 1, ["new"]), c("good", 2, ["unrelated"]),
                                   c("neutral1", 1, ["unrelated"]), c("neutral2", 1)])
    first = j(initial, ("bad", False, True), ("good", True, False),
              ("neutral1", False, False), ("neutral2", False, False))
    second = j(revised, ("bad", False, True), ("good", True, False),
               ("neutral1", False, False), ("neutral2", False, False))
    assert calculate_existing_experience_credit(initial, first) == {"old": -0.25}
    score = RevisionOperation("score", "MODIFY", "good", c("good", 2, ["ignored"]),
                              c("good", 1, ["ignored"]))
    # 来源字段可不同，但不会触发未记录的 MODIFY，也不会影响归一化后的收益。
    assert calculate_operation_contributions(initial, first, revised, second, [score]) == {
        "score": pytest.approx(2 / 5 - 1 / 4),
    }
    source_only = RevisionOperation("metadata", "MODIFY", "bad",
                                    c("bad", 1, ["new"]), c("bad", 1, ["old"]))
    # good 提分会让 bad 的权重从 1/4 降为 1/5，但 bad 只换来源，不能领取 +0.05。
    with pytest.raises(ValueError, match="no-op MODIFY"):
        calculate_operation_contributions(initial, first, revised, second,
                                          [score, source_only])


def test_add_and_delete_do_not_depend_on_revision_provenance():
    initial = Rubric("r0", None, [c("old", 1, ["e"]), c("same", 1)])
    revised = Rubric("r1", "r0", [c("new", 1, ["whatever"]), c("same", 1, ["other"])])
    first = j(initial, ("old", False, True), ("same", False, False))
    second = j(revised, ("new", True, False), ("same", False, False))
    ops = [RevisionOperation("del", "DELETE", "old", None, c("old", 1)),
           RevisionOperation("add", "ADD", None, c("new", 1))]
    assert calculate_operation_contributions(initial, first, revised, second, ops) == {
        "del": pytest.approx(0.5), "add": pytest.approx(0.5),
    }


def test_revision_budget_and_invalid_transitions():
    first_r = Rubric("r0", None, [c("c1", 1), c("c2", 1)])
    next_r = Rubric("r1", "r0", [c("c1", 1, question="new question"), c("c2", 1)])
    first = j(first_r, ("c1", False, True), ("c2", True, True))
    second = j(next_r, ("c1", True, False), ("c2", True, True))
    assert calculate_revision_budget(first_r, first, next_r, second) == pytest.approx(1)
    assert calculate_revision_budget(first_r, first, next_r, second, 0.5) == pytest.approx(0.5)
    weighted = allocate_revision_credit([CandidateExperience("candidate", "x", "t", ["op"])],
                                        {"op": 1.0},
                                        calculate_revision_budget(first_r, first, next_r, second, 0.5))
    exp = Experience("e", "x", "t", 1, 1, [], [])
    assert apply_beta_evidence(exp, EvidenceRecord("s", "revision_credit", weighted[0].credit, 1)).alpha == 1.5
    with pytest.raises(ValueError, match="already weighted"):
        apply_beta_evidence(exp, EvidenceRecord("s", "revision_credit", weighted[0].credit, 0.5))
    for bad in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="evidence_weight"):
            calculate_revision_budget(first_r, first, next_r, second, bad)
    failure_r = Rubric("r2", "r1", next_r.criteria)
    failure = j(failure_r, ("c1", False, True), ("c2", True, True))
    with pytest.raises(ValueError, match="initial failure"):
        calculate_revision_budget(next_r, second, failure_r, failure)
    with pytest.raises(ValueError, match="initial failure"):
        calculate_revision_budget(first_r, first, next_r,
                                  j(next_r, ("c1", False, True), ("c2", True, True)))
    with pytest.raises(ValueError, match="does not match"):
        calculate_revision_budget(first_r, first, next_r, replace(second, margin=0.4))


def test_shared_operation_zero_contribution_and_unbound_candidates():
    candidates = [CandidateExperience("a", "a", "t", ["shared", "zero"]),
                  CandidateExperience("b", "b", "t", ["shared"]),
                  CandidateExperience("unbound", "c", "t")]
    output = allocate_revision_credit(candidates, {"shared": 0.6, "zero": 0.0}, 0.3)
    assert [item.credit for item in output] == [pytest.approx(0.15), pytest.approx(0.15), None]
    assert candidates[0].credit is None
    assert sum(abs(item.credit or 0) for item in output) <= 0.3
    assert allocate_revision_credit(candidates, {"shared": 0, "zero": 0}, 0.3) == candidates
    assert allocate_revision_credit([CandidateExperience("z", "z", "t", ["zero"])],
                                    {"zero": 0.0, "positive": 1.0}, 1)[0].credit == 0


def test_negative_credit_cancellation_and_budget_bound():
    candidates = [CandidateExperience("a", "a", "t", ["positive", "negative"]),
                  CandidateExperience("b", "b", "t", ["negative"])]
    output = allocate_revision_credit(candidates, {"positive": 0.6, "negative": -0.4}, 0.5)
    assert output[0].credit == pytest.approx(0.2)
    assert output[1].credit == pytest.approx(-0.1)
    assert sum(abs(item.credit or 0) for item in output) <= 0.5
    for invalid in (float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="finite"):
            allocate_revision_credit(candidates, {"positive": invalid, "negative": -0.4}, 0.5)
    with pytest.raises(ValueError, match="unknown operation"):
        allocate_revision_credit(candidates, {"positive": 0.5}, 0.5)
    with pytest.raises(ValueError, match="already"):
        allocate_revision_credit([replace(candidates[0], credit=0.1)],
                                 {"positive": 0.6, "negative": -0.4}, 0.5)
    with pytest.raises(ValueError, match="duplicate candidate"):
        allocate_revision_credit([candidates[0], candidates[0]],
                                 {"positive": 0.6, "negative": -0.4}, 0.5)


def test_beta_update_weight_zero_and_negative_credit():
    exp = Experience("e", "principle", "trigger", 1, 1, [], [])
    positive = EvidenceRecord("s1", "revision_credit", 0.25, 1)
    negative = EvidenceRecord("s2", "downstream_credit", -0.5, 0.5)
    updated = apply_beta_evidence(apply_beta_evidence(exp, positive), negative)
    assert (updated.alpha, updated.beta) == pytest.approx((1.25, 1.25))
    assert updated.confidence == 0.5
    assert updated.evidence_amount == 0.5
    assert updated.evidence_history == [positive, negative]
    assert (exp.alpha, exp.beta, exp.evidence_history) == (1, 1, [])
    assert apply_beta_evidence(exp, EvidenceRecord("s", "downstream_credit", 0, 1)) is exp
    assert apply_beta_evidence(exp, EvidenceRecord("s", "downstream_credit", 0.5, 0)) is exp
    with pytest.raises(ValueError, match="credit"):
        apply_beta_evidence(exp, EvidenceRecord("s", "downstream_credit", 2, 1))
    with pytest.raises(ValueError, match="evidence_weight"):
        apply_beta_evidence(exp, EvidenceRecord("s", "downstream_credit", 0.1, 1.5))
