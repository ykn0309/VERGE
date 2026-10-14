"""Check preference orientation, input preservation, and conservative selection."""
from copy import deepcopy

import pytest

from scripts.prepare_helpsteer3 import select_samples


def row(task="  Explain this.\n", score=-2, individual=(-2, -3, -2)):
    return {
        "domain": "general", "language": "english",
        "context": [{"role": "user", "content": task}],
        "response1": "  Response one.\n", "response2": "Response two.",
        "overall_preference": score,
        "individual_preference": [{"score": s, "reasoning": "Human reason."} for s in individual],
    }


def test_preference_sign_orients_pair_without_rewriting_text_or_evidence():
    negative = row()
    positive = row("Another task", 3, (3, 2, 3))
    original = deepcopy([negative, positive])
    samples, counts = select_samples([negative, positive])
    assert [negative, positive] == original
    by_task = {s["task"]: s for s in samples}
    assert by_task[negative["context"][0]["content"]]["chosen"] == negative["response1"]
    assert by_task["Another task"]["chosen"] == positive["response2"]
    assert by_task["Another task"]["rejected"] == positive["response1"]
    assert by_task["Another task"]["metadata"]["individual_preference"] == positive["individual_preference"]
    assert counts["retained"] == 2


def test_weak_disagreeing_and_incomplete_contexts_are_not_strict_pairs():
    weak = row(score=-1)
    disagreement = row(individual=(-3, -2, 1))
    tie = row(score=0)
    multi = row()
    multi["context"].insert(0, {"role": "assistant", "content": "Earlier answer."})
    identical = row()
    identical["response2"] = identical["response1"].strip()
    samples, counts = select_samples([weak, disagreement, tie, multi, identical])
    assert not samples
    assert counts == {"source_rows": 5, "weak_or_tied_preference": 2,
                      "individual_direction_disagreement": 1, "not_single_user_turn": 1,
                      "identical_responses": 1, "retained": 0}


def test_order_and_task_dedup_are_independent_of_source_row_order():
    first, second = row("Explain this."), row(" Explain   this. \n")
    other = row("Different task.", 2, (2, 2, 3))
    forward, counts = select_samples([first, other, second])
    backward, _ = select_samples([second, other, first])
    assert [s["sample_id"] for s in forward] == [s["sample_id"] for s in backward]
    assert len(forward) == 2 and counts["duplicate_task"] == 1


def test_strong_average_does_not_hide_individually_weak_preference():
    samples, counts = select_samples([row(score=-2, individual=(-3, -1, -1))])
    assert samples == []
    assert counts["weak_individual_preference"] == 1


@pytest.mark.parametrize("change", [
    {"overall_preference": True}, {"overall_preference": 4},
    {"response1": ""}, {"individual_preference": []},
])
def test_invalid_eligible_rows_fail_instead_of_silently_changing_dataset(change):
    sample = row()
    sample.update(change)
    with pytest.raises(ValueError, match="train row 0"):
        select_samples([sample])
