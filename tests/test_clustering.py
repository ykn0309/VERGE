"""State and boundary checks for the new online grouping algorithm."""

from math import cos, radians, sin, sqrt

import pytest

from verge.clustering import ClusterItem, OnlineClustering


def clustering(**changes):
    values = dict(reference_weight=0.3, task_threshold=0.75, experience_threshold=0.75)
    return OnlineClustering(**(values | changes))


def item(name, vector=(1, 0), kind="task"):
    return ClusterItem(name, kind, name, vector)


def angle(name, degrees):
    return item(name, (cos(radians(degrees)), sin(radians(degrees))))


def test_new_objects_join_or_create_groups_and_types_never_mix():
    c = clustering()
    result = c.update(
        [item("a"), item("b"), item("c", (0, 1)), item("a", kind="experience")]
    )
    groups = c.assignments("task")
    assert groups["a"] == groups["b"] != groups["c"]
    assert len(c.groups("task")) == 2 and len(c.groups("experience")) == 1
    assert result.stable and result.new_items == 4


def test_group_uses_all_members_instead_of_nearest_neighbor_only():
    c = clustering(task_threshold=0.8)
    c.update([angle("a", 0), angle("b", 25), angle("c", 50)])
    groups = c.assignments("task")
    assert groups["a"] == groups["b"] != groups["c"]


def test_citation_sets_are_bidirectional_deduplicated_and_exclude_missing_evidence():
    c = clustering()
    c.update([item("a"), item("e", kind="experience")], [("a", "e"), ("a", "e")])
    assert c.references("task", "a") == {"e"}
    assert c.references("experience", "e") == {"a"}
    result = c.update(citations=[("a", "e")])
    assert result.new_edges == 0


def test_missing_references_fall_back_to_text_and_weight_uses_support():
    c = clustering(task_threshold=0.99)
    c.update([item("a"), item("b", (0.82, sqrt(1 - 0.82**2)))])
    assert c.similarity("task", "a", "b") == pytest.approx(0.82)
    c.update(
        [item(str(i), kind="experience") for i in range(4)],
        [("a", "0"), ("a", "1"), *(("b", str(i)) for i in range(4))],
    )
    assert c.similarity("task", "a", "b") == pytest.approx(0.756)


def test_updated_references_invalidate_cached_similarity_and_can_split_members():
    c = clustering(task_threshold=0.9)
    c.update([item("a"), item("b")])
    assert c.assignments("task")["a"] == c.assignments("task")["b"]
    assert c.similarity("task", "a", "b") == 1
    result = c.update(
        [item("e1", kind="experience"), item("e2", kind="experience")],
        [("a", "e1"), ("b", "e2")],
    )
    assert c.similarity("task", "a", "b") == pytest.approx(0.85)
    assert c.assignments("task")["a"] != c.assignments("task")["b"]
    assert result.moves > 0 and result.stable


def test_shared_tasks_raise_experience_similarity_without_using_rewards():
    c = clustering()
    c.update(
        [
            item("t"),
            item("a", kind="experience"),
            item("b", (0.7, sqrt(0.51)), "experience"),
        ]
    )
    before = c.similarity("experience", "a", "b")
    c.update(citations=[("t", "a"), ("t", "b")])
    assert c.similarity("experience", "a", "b") > before


# A fixed geometry in which greedy insertion requires two subsequent moves.
MULTI_ROUND_VECTORS = [
    (0.354252, 0.569823, 0.040932),
    (0.022983, -0.212931, 0.993634),
    (-0.421270, -0.703480, -0.477843),
    (-0.479126, -0.345266, -0.464178),
    (-0.784712, -0.349001, -0.377870),
    (0.138480, -0.596738, -0.858425),
    (-0.594896, 0.084885, -0.222876),
    (0.466941, 0.606119, -0.171125),
]


def test_multiple_rounds_and_cap_are_reported_honestly():
    items = [item(str(i), vector) for i, vector in enumerate(MULTI_ROUND_VECTORS)]
    capped = clustering(task_threshold=0.5, max_rounds=1)
    full = clustering(task_threshold=0.5, max_rounds=5)
    first = capped.update(items)
    result = full.update(items)
    assert first.rounds == 1 and not first.stable
    assert result.rounds == 3 and result.stable and result.moves == 2
    for _ in range(5):
        if capped.update().stable:
            break
    assert sorted(capped.groups("task").values()) == sorted(
        full.groups("task").values()
    )


def test_query_returns_multiple_groups_without_assigning_or_mutating():
    c = clustering(task_threshold=0.7)
    c.update([angle("a", -30), angle("b", 30)])
    before = c.assignments("task")
    matches = c.query_task_groups((1, 0))
    assert len(matches) == 2
    assert all(score == pytest.approx(cos(radians(30))) for _, score in matches)
    assert c.assignments("task") == before
    assert c.query_task_groups((1, 0), limit=1) == matches[:1]
    assert c.query_task_groups((-1, 0)) == []


def test_input_order_is_canonical_within_an_update():
    items = [angle("b", 25), angle("c", 50), angle("a", 0)]
    a, b = clustering(), clustering()
    a.update(items)
    b.update(reversed(items))
    assert a.assignments("task") == b.assignments("task")


def test_invalid_snapshot_does_not_partially_add_objects_or_edges():
    c = clustering()
    c.update([item("t")])
    before = c.groups("task")
    with pytest.raises(ValueError, match="endpoints"):
        c.update([item("new")], [("new", "missing")])
    assert c.groups("task") == before
    with pytest.raises(ValueError, match="dimension"):
        c.update([item("new"), item("bad", (1, 0, 0))])
    assert c.groups("task") == before
    with pytest.raises(ValueError, match="immutable"):
        c.update([item("t", (0, 1))])
    assert c.groups("task") == before


def test_returned_membership_is_a_copy_and_repeated_items_are_idempotent():
    c = clustering()
    value = item("a")
    c.update([value])
    mapping = c.assignments("task")
    mapping.clear()
    assert c.assignments("task")
    assert c.update([value]).new_items == 0


@pytest.mark.parametrize(
    "vector", [(), (0, 0), (float("nan"), 1), (float("inf"), 0), (True, 1)]
)
def test_invalid_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        item("a", vector)


@pytest.mark.parametrize(
    "changes",
    [
        dict(reference_weight=2),
        dict(task_threshold=-1),
        dict(experience_threshold=True),
        dict(max_rounds=0),
        dict(max_rounds=True),
    ],
)
def test_invalid_parameters_are_rejected(changes):
    with pytest.raises(ValueError):
        clustering(**changes)
