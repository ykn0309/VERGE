"""Whole-pool semantic search remains available with and without history."""

import pytest

from verge.clustering import ClusterItem, OnlineClustering
from verge.feedback import FeedbackLedger, CreditFeedback
from verge.retrieval import CandidateQuota, merge_candidates, retrieve_candidates
from verge.ranking import ExperienceRanking, ExperienceScore


def clusters(*, experience_threshold=0.75):
    return OnlineClustering(
        reference_weight=0.3,
        task_threshold=0.75,
        experience_threshold=experience_threshold,
    )


def retrieve(state, ledger=None, vector=(1, 0), **kwargs):
    return retrieve_candidates(
        vector,
        state,
        ledger if ledger is not None else FeedbackLedger("test"),
        task_group_limit=kwargs.get("task_group_limit", 3),
        quota=kwargs.get("quota", CandidateQuota()),
    )


def test_no_matching_task_group_uses_semantic_search_over_all_experiences():
    state = clusters()
    state.update(
        [
            ClusterItem("past", "task", "different task", (0, 1)),
            ClusterItem("close", "experience", "relevant method", (1, 0)),
            ClusterItem("far", "experience", "other method", (0, 1)),
        ]
    )
    result = retrieve(state)
    assert result.semantic_only
    assert result.task_groups == () and result.historical.scored == ()
    assert result.semantic == (("close", 1), ("far", 0))
    assert result.historical.unknown_ids == ("close", "far")


def test_empty_history_still_searches_new_experiences():
    state = clusters()
    state.update([ClusterItem("new", "experience", "new method", (1, 0))])
    result = retrieve(state)
    assert result.semantic_only and result.semantic == (("new", 1),)


def test_matched_history_and_unlinked_semantic_experiences_are_both_available():
    state = clusters()
    state.update(
        [
            ClusterItem("past", "task", "similar task", (1, 0)),
            ClusterItem("known", "experience", "used method", (0.8, 0.6)),
            ClusterItem("new", "experience", "unlinked method", (1, 0)),
        ],
        [("past", "known")],
    )
    ledger = FeedbackLedger(
        "test", [CreditFeedback("r", "past", "test", ("known",), False, True, (True,))]
    )
    result = retrieve(state, ledger)
    assert not result.semantic_only
    assert result.historical.scored[0].experience_id == "known"
    assert result.historical.scored[0].score == 1
    assert result.historical.unknown_ids == ("new",)
    assert [eid for eid, _ in result.semantic] == ["new", "known"]
    assert state.references("experience", "new") == frozenset()


def test_matching_group_without_usage_does_not_manufacture_historical_scores():
    state = clusters()
    state.update(
        [
            ClusterItem("past", "task", "similar task", (1, 0)),
            ClusterItem("new", "experience", "new method", (1, 0)),
        ]
    )
    result = retrieve(state)
    assert result.task_groups and result.historical.scored == ()
    assert result.semantic == (("new", 1),)


def test_experience_cluster_membership_cannot_change_the_routes():
    results = []
    for threshold in (0.5, 0.99):
        state = clusters(experience_threshold=threshold)
        state.update(
            [
                ClusterItem("past", "task", "task", (1, 0)),
                ClusterItem("E1", "experience", "method1", (1, 0)),
                ClusterItem("E2", "experience", "method2", (0.8, 0.6)),
            ],
            [("past", "E1")],
        )
        ledger = FeedbackLedger(
            "test", [CreditFeedback("r", "past", "test", ("E1",), False, True, (True,))]
        )
        result = retrieve(state, ledger)
        results.append(
            (
                result.semantic,
                [(e.experience_id, e.score) for e in result.historical.scored],
            )
        )
    assert results[0] == results[1]


def test_retrieval_is_read_only_and_accepts_a_query_iterator():
    state = clusters()
    state.update(
        [
            ClusterItem("past", "task", "task", (1, 0)),
            ClusterItem("E", "experience", "method", (1, 0)),
        ],
        [("past", "E")],
    )
    ledger = FeedbackLedger(
        "test", [CreditFeedback("r", "past", "test", ("E",), False, True, (True,))]
    )
    before = state.assignments("task"), state.assignments("experience"), ledger.records
    first = retrieve(state, ledger, vector=iter((1, 0)))
    second = retrieve(state, ledger)
    assert first == second
    assert before == (
        state.assignments("task"),
        state.assignments("experience"),
        ledger.records,
    )
    assert state.references("task", "past") == frozenset({"E"})


def test_semantic_order_is_stable_and_respects_the_explicit_limit():
    state = clusters()
    state.update([ClusterItem(eid, "experience", eid, (1, 0)) for eid in ("Z", "A")])
    assert retrieve(state, quota=CandidateQuota(0, 1)).semantic == (("A", 1),)


def test_empty_pool_returns_no_candidates_without_error():
    result = retrieve(clusters())
    assert result.semantic_only and result.semantic == ()
    assert result.historical.scored == () and result.historical.unknown_ids == ()


@pytest.mark.parametrize("field", ["task_group_limit"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits_fail(field, value):
    with pytest.raises(ValueError, match="limit"):
        retrieve(clusters(), **{field: value})


def test_semantic_query_rejects_wrong_embedding_dimension():
    state = clusters()
    state.update([ClusterItem("E", "experience", "method", (1, 0))])
    with pytest.raises(ValueError, match="dimension"):
        retrieve(state, vector=(1, 0, 0))


def history_ranking(count):
    return ExperienceRanking(
        tuple(ExperienceScore(f"H{i}", 1 - i / 10, ()) for i in range(count)), ()
    )


def test_default_quota_six_history_four_semantic():
    result = merge_candidates(
        history_ranking(8), [(f"S{i}", 1 - i / 10) for i in range(10)]
    )
    assert len(result) == 10
    assert [item.experience_id for item in result] == [f"H{i}" for i in range(6)] + [
        f"S{i}" for i in range(4)
    ]
    assert [item.source for item in result] == ["historical"] * 6 + ["semantic"] * 4


def test_overlap_does_not_consume_semantic_slots():
    semantic = [(f"H{i}", 1 - i / 20) for i in range(6)] + [
        (f"S{i}", 0.5 - i / 10) for i in range(4)
    ]
    result = merge_candidates(history_ranking(6), semantic)
    assert len(result) == 10 and len({item.experience_id for item in result}) == 10
    assert [item.experience_id for item in result[6:]] == [f"S{i}" for i in range(4)]


def test_fewer_historical_candidates_are_filled_by_semantics():
    result = merge_candidates(
        history_ranking(2), [(f"S{i}", 1 - i / 10) for i in range(10)]
    )
    assert len(result) == 10
    assert sum(item.source == "historical" for item in result) == 2
    assert sum(item.source == "semantic" for item in result) == 8


def test_no_history_uses_ten_semantic_candidates_with_unknown_historical_scores():
    result = merge_candidates(
        history_ranking(0), [(f"S{i}", 1 - i / 10) for i in range(10)]
    )
    assert len(result) == 10
    assert all(
        item.source == "semantic" and item.historical_score is None for item in result
    )


def test_small_pool_is_not_padded_with_duplicates():
    result = merge_candidates(history_ranking(1), [("H0", 0.8), ("new", 0.7)])
    assert len(result) == 2 and result[0].semantic_similarity == 0.8
    assert result[1].historical_score is None


def test_semantic_candidate_can_retain_a_known_negative_historical_score():
    historical = ExperienceRanking((ExperienceScore("E", -0.5, ()),), ())
    result = merge_candidates(historical, [("E", 0.8)], quota=CandidateQuota(0, 1))
    assert result[0].source == "semantic" and result[0].historical_score == -0.5


def test_history_candidate_can_be_outside_the_semantic_top_ten():
    result = merge_candidates(
        history_ranking(6), [(f"S{i}", 1 - i / 10) for i in range(10)]
    )
    assert all(item.semantic_similarity is None for item in result[:6])


def test_full_retrieval_applies_the_default_quota_after_global_semantic_search():
    state = clusters()
    state.update(
        [ClusterItem("past", "task", "past", (1, 0))]
        + [
            ClusterItem(f"E{i:02}", "experience", f"method{i}", (1, 0))
            for i in range(15)
        ],
        [("past", f"E{i:02}") for i in range(8)],
    )
    ledger = FeedbackLedger(
        "test",
        [
            CreditFeedback(f"r{i}", "past", "test", (f"E{i:02}",), False, True, (True,))
            for i in range(8)
        ],
    )
    result = retrieve(state, ledger)
    assert len(result.candidates) == 10
    assert [c.experience_id for c in result.candidates] == [
        f"E{i:02}" for i in range(10)
    ]
    assert sum(c.source == "historical" for c in result.candidates) == 6
    assert sum(c.source == "semantic" for c in result.candidates) == 4


def test_full_retrieval_with_no_groups_assigns_all_ten_slots_to_semantics():
    state = clusters()
    state.update(
        [ClusterItem(f"E{i:02}", "experience", f"method{i}", (1, 0)) for i in range(15)]
    )
    result = retrieve(state)
    assert result.semantic_only and len(result.candidates) == 10
    assert all(c.source == "semantic" for c in result.candidates)


@pytest.mark.parametrize(
    "historical,semantic", [(0, 0), (-1, 4), (6, -1), (True, 4), (6, 1.5)]
)
def test_invalid_candidate_quotas_fail(historical, semantic):
    with pytest.raises(ValueError, match="quota"):
        CandidateQuota(historical, semantic)
