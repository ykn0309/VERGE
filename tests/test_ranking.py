"""Individual ranking must preserve group-first weighting and unknown evidence."""

from dataclasses import asdict

import pytest

from verge.feedback import FeedbackLedger, CreditFeedback
from verge.ranking import rank_experiences


def feedback(rid, task, reward, ids=("E",), **kwargs):
    ids = tuple(dict.fromkeys(ids))
    return CreditFeedback(rid, task, "test", ids, reward < 0, reward > 0, (reward > 0,) * len(ids), **kwargs)


def test_weight_group_means_without_multiplying_by_sample_counts():
    ledger = FeedbackLedger(
        "test",
        [feedback("a", "a", 1)] + [feedback(f"b{i}", f"b{i}", -1) for i in range(9)],
    )
    assignments = {"a": "A", **{f"b{i}": "B" for i in range(9)}}
    result = rank_experiences(
        ["E"], {"A": 0.9, "B": 0.6}, ledger.by_task_group(assignments)
    )
    score = result.scored[0]
    assert score.score == pytest.approx((0.9 - 0.6) / (0.9 + 0.6))
    assert score.score != pytest.approx((0.9 - 9 * 0.6) / (0.9 + 9 * 0.6))
    assert score.tasks == 10
    assert [item.statistics.tasks for item in score.evidence] == [1, 9]


def test_missing_group_evidence_does_not_enter_the_denominator():
    ledger = FeedbackLedger("test", [feedback("a", "t", 1)])
    result = rank_experiences(
        ["E"], {"A": 0.2, "B": 0.9}, ledger.by_task_group({"t": "A"})
    )
    assert result.scored[0].score == 1
    assert len(result.scored[0].evidence) == 1


def test_unknown_is_separate_from_zero_and_negative_scores():
    ledger = FeedbackLedger(
        "test", [feedback("z", "z", 0, ("zero",)), feedback("n", "n", -1, ("neg",))]
    )
    result = rank_experiences(
        ["new", "zero", "neg"],
        {"A": 0.7},
        ledger.by_task_group({"z": "A", "n": "A"}),
    )
    assert [(item.experience_id, item.score) for item in result.scored] == [
        ("zero", 0),
        ("neg", -1),
    ]
    assert result.unknown_ids == ("new",)


def test_candidate_ids_and_repeated_citations_do_not_duplicate_ranking():
    ledger = FeedbackLedger("test", [feedback("a", "t", 1, ("E", "E"))])
    result = rank_experiences(["E", "E"], {"A": 1}, ledger.by_task_group({"t": "A"}))
    assert len(result.scored) == 1 and result.scored[0].tasks == 1


def test_only_queried_groups_and_requested_experiences_are_used():
    ledger = FeedbackLedger(
        "test", [feedback("a", "a", 1), feedback("b", "b", -1, ("other",))]
    )
    result = rank_experiences(
        ["E"], {"B": 1}, ledger.by_task_group({"a": "A", "b": "B"})
    )
    assert result.scored == () and result.unknown_ids == ("E",)


def test_development_failures_and_duplicate_tasks_do_not_affect_score():
    ledger = FeedbackLedger(
        "test",
        [
            CreditFeedback("err", "t", "test", ("E",), None, None, (None,), error="timeout"),
            feedback("ok", "t", 1),
            feedback("repeat", "t", -1),
        ],
    )
    result = rank_experiences(["E"], {"A": 1}, ledger.by_task_group({"t": "A"}))
    assert result.scored[0].score == 1 and result.scored[0].tasks == 1
    assert result.scored[0].evidence[0].statistics.record_ids == ("ok",)


def test_scores_do_not_depend_on_an_experience_group():
    ledger = FeedbackLedger("test", [feedback("a", "t", 1, ("E1",))])
    result = rank_experiences(["E1", "E2"], {"A": 1}, ledger.by_task_group({"t": "A"}))
    assert [item.experience_id for item in result.scored] == ["E1"]
    assert result.unknown_ids == ("E2",)


def test_reassignment_recomputes_ranking_without_changing_raw_records():
    ledger = FeedbackLedger("test", [feedback("a", "t", 1)])
    original = ledger.records
    old = rank_experiences(["E"], {"A": 1}, ledger.by_task_group({"t": "A"}))
    new = rank_experiences(["E"], {"A": 1}, ledger.by_task_group({"t": "B"}))
    assert old.scored[0].score == 1 and new.scored == ()
    assert ledger.records == original


def test_equal_scores_use_id_order_and_inputs_are_unchanged():
    ledger = FeedbackLedger("test", [feedback("a", "t", 1, ("Z", "A"))])
    stats = ledger.by_task_group({"t": "G"})
    weights = {"G": 0.8}
    before = {key: asdict(value) for key, value in stats.items()}
    result = rank_experiences(["Z", "A"], weights, stats)
    assert [item.experience_id for item in result.scored] == ["A", "Z"]
    assert before == {key: asdict(value) for key, value in stats.items()}
    assert weights == {"G": 0.8}


@pytest.mark.parametrize("weights", [{}, {"A": 0}])
def test_no_positive_similarity_returns_unknown(weights):
    ledger = FeedbackLedger("test", [feedback("a", "t", 1)])
    result = rank_experiences(["E"], weights, ledger.by_task_group({"t": "A"}))
    assert result.scored == () and result.unknown_ids == ("E",)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_similarities_fail(weight):
    with pytest.raises(ValueError, match="similarity"):
        rank_experiences(["E"], {"A": weight}, {})


@pytest.mark.parametrize("ids", [[""], [None], "E"])
def test_invalid_candidate_ids_fail(ids):
    with pytest.raises(ValueError):
        rank_experiences(ids, {}, {})


def test_duplicate_task_in_two_groups_is_rejected():
    ledger = FeedbackLedger("test", [feedback("a", "t", 1)])
    evidence = ledger.for_experience("E")
    with pytest.raises(ValueError, match="multiple task groups"):
        rank_experiences(
            ["E"], {"A": 1, "B": 1}, {("A", "E"): evidence, ("B", "E"): evidence}
        )
