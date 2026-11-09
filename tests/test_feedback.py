"""Complete subset credit, missing outcomes and task-level aggregation."""

from dataclasses import replace
from itertools import product
import pytest
from verge.feedback import CreditFeedback, FeedbackLedger


def record(outcomes=(False, False, False, True), *, task="t", rid="r"):
    r0, r1, r2, r12 = outcomes
    return CreditFeedback(rid, task, "p", ("a", "b"), r0, r12, (r1, r2))


@pytest.mark.parametrize("values", list(product([False, True], repeat=4)))
def test_all_sixteen_outcomes_preserve_total_and_average_marginals(values):
    r0, r1, r2, r12 = map(int, values)
    credits = record(values).credits
    assert credits == {
        "a": ((r1 - r0) + (r12 - r2)) / 2,
        "b": ((r2 - r0) + (r12 - r1)) / 2,
    }
    assert sum(credits.values()) == r12 - r0


def test_joint_success_and_redundancy_are_not_individual_full_rewards():
    assert record().credits == {"a": 0.5, "b": 0.5}
    assert record((False, True, True, True)).credits == {"a": 0.5, "b": 0.5}
    assert record((False, True, False, False)).credits == {"a": 0.5, "b": -0.5}


@pytest.mark.parametrize("values", list(product([False, True], repeat=2)))
def test_one_experience(values):
    r0, r1 = values
    assert CreditFeedback("r", "t", "p", ("a",), r0, r1, (r1,)).credits == {
        "a": int(r1) - int(r0)
    }


def test_missing_any_subset_is_unknown_and_does_not_consume_task():
    missing = CreditFeedback(
        "err", "t", "p", ("a", "b"), False, True, (None, False), "timeout"
    )
    ledger = FeedbackLedger("p", [missing, record()])
    assert missing.credits is None
    assert ledger.for_experience("a").mean_reward == 0.5
    assert ledger.for_experience("a").tasks == 1


def test_duplicate_tasks_and_groups_do_not_duplicate_credit():
    ledger = FeedbackLedger(
        "p", [record(), record((True, False, False, False), rid="repeat")]
    )
    stats = ledger.by_task_group({"t": "G"})
    assert stats["G", "a"].mean_reward == 0.5 and stats["G", "a"].tasks == 1
    group = ledger.by_group_pair({"t": "G"}, {"a": "H", "b": "H"})["G", "H"]
    assert group.mean_reward == 1 and group.tasks == 1
    assert ledger.for_experience("new").mean_reward is None


def test_append_is_idempotent_and_conflicts_fail():
    ledger = FeedbackLedger("p", [record()])
    assert ledger.append(record()) is False
    with pytest.raises(ValueError, match="conflicting"):
        ledger.append(record((True, True, True, True)))
    with pytest.raises(ValueError, match="protocol"):
        ledger.append(replace(record(), protocol_id="other"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experience_ids": ("a", "a")},
        {"experience_ids": ()},
        {"singleton_correct": (True,)},
        {"baseline_correct": 1},
        {"actual_correct": None},
        {"experience_ids": ("a", "b", "c")},
    ],
)
def test_invalid_credit_contracts_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(record(), **kwargs)
