"""Mathematical invariants of constrained, alternating modularity search."""

from copy import deepcopy
import numpy as np
import pytest
from verge.clustering import ClusterItem, OnlineClustering
from verge.partitioning import ModularitySearch


def test_modularity_preserves_two_blocks_and_singleton():
    matrix = np.array(
        [
            [0, 0.9, 0.1, 0.1, 0],
            [0.9, 0, 0.1, 0.1, 0],
            [0.1, 0.1, 0, 0.9, 0],
            [0.1, 0.1, 0.9, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    search = ModularitySearch(matrix, 0.35)
    result = search.run([(0,), (1,), (2,), (3,), (4,)])
    assert result.groups == ((0, 1), (2, 3), (4,))
    assert result.quality == pytest.approx(search.quality(result.groups))
    assert result.quality > search.quality([(i,) for i in range(5)])
    assert all(search.feasible(c) for c in result.groups)


def test_member_floor_not_only_group_average_and_repair():
    matrix = np.array([[0, 0.95, 0.1], [0.95, 0, 0.1], [0.1, 0.1, 0]])
    search = ModularitySearch(matrix, 0.35)
    assert not search.feasible([0, 1, 2])
    repaired, events = search.repair([(0, 1, 2)])
    assert repaired == ((0, 1), (2,)) and events[0]["object_index"] == 2
    assert search.run([(0, 1, 2)]).groups == repaired


def test_group_constraint_is_average_not_complete_link():
    matrix = np.array([[0, 0.9, 0.3], [0.9, 0, 0.9], [0.3, 0.9, 0]])
    assert ModularitySearch(matrix, 0.5).feasible([0, 1, 2])


@pytest.mark.parametrize("n", [0, 1, 4])
def test_zero_weight_keeps_singletons(n):
    r = ModularitySearch(np.zeros((n, n)), 0.35).run([tuple(range(n))] if n else [])
    assert r.groups == tuple((i,) for i in range(n)) and r.quality == 0


def test_category_profile_normalizes_counts_and_uses_raw_support():
    c = OnlineClustering()
    items = [
        ClusterItem("a", "experience", "a", (1, 0)),
        ClusterItem("b", "experience", "b", (0, 1)),
    ]
    items += [ClusterItem(str(i), "task", str(i), (1, 0)) for i in range(4)]
    c.update(items, [("0", "a"), ("1", "a"), ("2", "b"), ("3", "b")])
    # No shared task ID; identical task group profiles nevertheless overlap fully.
    assert not c.references("experience", "a") & c.references("experience", "b")
    assert len(c.groups("task")) == 1
    assert c.similarity("experience", "a", "b") == pytest.approx(0.5 * 2 / 3)
    assert c._profile(("experience", "a")) == c._profile(("experience", "b"))


def test_online_roundtrip_query_no_threshold_and_invalid_input_atomicity():
    c = OnlineClustering()
    c.update([ClusterItem("t", "task", "t", (0, 1))])
    assert c.query_task_groups((1, 0)) == [(next(iter(c.groups("task"))), 0)]
    before = deepcopy(c.to_dict())
    with pytest.raises(ValueError):
        c.update([], [("missing", "E")])
    assert c.to_dict() == before
    restored = OnlineClustering.from_dict(before)
    assert restored.to_dict() == before
    assert restored.last_update.status == "stable"
    restored.update([ClusterItem("e", "experience", "e", (1, 0))], [("t", "e")])
    assert restored.references("task", "t") == {"e"}
    assert all(h["current_opposite"] for h in restored.last_update.half_rounds[-2:])


def test_budget_and_determinism():
    a = np.array(
        [[0, 0.9, 0.1, 0.1], [0.9, 0, 0.1, 0.1], [0.1, 0.1, 0, 0.9], [0.1, 0.1, 0.9, 0]]
    )
    search = ModularitySearch(a, 0.35, budget=1)
    r = search.run([(i,) for i in range(4)])
    assert r.budget and all(search.feasible(c) for c in r.groups)
    assert search.run([(i,) for i in range(4)]).groups == r.groups


def test_query_top_three_readonly_and_no_group_count_setting():
    c = OnlineClustering()
    c.update(
        [
            ClusterItem(str(i), "task", str(i), tuple(int(i == j) for j in range(5)))
            for i in range(5)
        ]
    )
    before = c.to_dict()
    assert len(c.query_task_groups((1, 0, 0, 0, 0))) == 3
    assert c.to_dict() == before
    assert len(c.groups("task")) == 5


@pytest.mark.parametrize("seed", range(8))
def test_random_fixed_matrix_search_never_publishes_illegal_or_worse_than_singles(seed):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(15, 5))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    matrix = np.clip(vectors @ vectors.T, 0, 1)
    np.fill_diagonal(matrix, 0)
    singles = [(i,) for i in range(15)]
    search = ModularitySearch(matrix, 0.35)
    result = search.run([tuple(range(15))])
    assert all(search.feasible(c) for c in result.groups)
    assert sorted(x for c in result.groups for x in c) == list(range(15))
    assert result.quality >= search.quality(singles) - 1e-12
    assert result.groups == search.run([tuple(range(15))]).groups
