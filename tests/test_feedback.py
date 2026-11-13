"""Per-experience reward differences, task deduplication, and unknowns."""

import pytest
from verge.feedback import CreditFeedback, FeedbackLedger


@pytest.mark.parametrize(
    "baseline,trial,credit",
    [(False, True, 1), (True, False, -1), (True, True, 0), (False, False, 0)],
)
def test_credit(baseline, trial, credit):
    row = CreditFeedback("r", "t", "p", "E", baseline, trial)
    assert row.credits == {"E": credit}


def test_unknown_and_group_remapping():
    ledger = FeedbackLedger(
        "p",
        [
            CreditFeedback("r1", "t1", "p", "E", False, True),
            CreditFeedback("r2", "t2", "p", "E", True, False),
            CreditFeedback("r3", "t3", "p", "E", None, True, "baseline timeout"),
        ],
    )
    assert ledger.for_experience("E").tasks == 2
    assert ledger.for_experience("E").mean_reward == 0
    assert ledger.by_task_group({"t1": "A", "t2": "B"})[("A", "E")].mean_reward == 1
    assert ledger.by_task_group({"t1": "B", "t2": "B"})[("B", "E")].mean_reward == 0
    assert ledger.records[-1].credits is None


def test_invalid_missing_and_protocol():
    with pytest.raises(ValueError):
        CreditFeedback("r", "t", "p", "E", None, False)
    ledger = FeedbackLedger("p")
    with pytest.raises(ValueError):
        ledger.append(CreditFeedback("r", "t", "other", "E", False, True))
