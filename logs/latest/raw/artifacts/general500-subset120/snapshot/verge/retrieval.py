"""Retrieve and combine individual experiences for subsequent LLM screening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .clustering import OnlineClustering
from .feedback import FeedbackLedger
from .ranking import ExperienceRanking, rank_experiences


@dataclass(frozen=True)
class CandidateQuota:
    historical: int = 6
    semantic: int = 4

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 0 for n in (self.historical, self.semantic)):
            raise ValueError("candidate quotas must be nonnegative integers")
        if self.total == 0:
            raise ValueError("total candidate quota must be positive")

    @property
    def total(self) -> int:
        return self.historical + self.semantic


@dataclass(frozen=True)
class ExperienceCandidate:
    experience_id: str
    source: Literal["historical", "semantic"]
    historical_score: float | None
    semantic_similarity: float | None


@dataclass(frozen=True)
class CandidateRoutes:
    task_groups: tuple[tuple[str, float], ...]
    historical: ExperienceRanking
    semantic: tuple[tuple[str, float], ...]
    candidates: tuple[ExperienceCandidate, ...]
    quota: CandidateQuota

    @property
    def semantic_only(self) -> bool:
        return not self.task_groups


def merge_candidates(
    historical: ExperienceRanking,
    semantic: Iterable[tuple[str, float]],
    *,
    quota: CandidateQuota = CandidateQuota(),
) -> tuple[ExperienceCandidate, ...]:
    """Take history first, then fill all remaining slots from global semantics.

    A duplicate cannot consume a slot twice. Fewer historical candidates leave
    more semantic slots; a small pool returns its actual available population.
    """
    semantic = tuple(semantic)
    semantic_scores = dict(semantic)
    historical_scores = {item.experience_id: item.score for item in historical.scored}
    result: list[ExperienceCandidate] = []
    seen: set[str] = set()

    def add(experience_id: str, source: Literal["historical", "semantic"]) -> None:
        if experience_id not in seen:
            seen.add(experience_id)
            result.append(
                ExperienceCandidate(
                    experience_id,
                    source,
                    historical_scores.get(experience_id),
                    semantic_scores.get(experience_id),
                )
            )

    for item in historical.scored[: quota.historical]:
        add(item.experience_id, "historical")
    for experience_id, _ in semantic:
        if len(result) == quota.total:
            break
        add(experience_id, "semantic")
    return tuple(result)


def retrieve_candidates(
    vector: Iterable[float],
    clusters: OnlineClustering,
    ledger: FeedbackLedger,
    *,
    task_group_limit: int,
    quota: CandidateQuota = CandidateQuota(),
) -> CandidateRoutes:
    """Retrieve from one read-only history snapshot before observing this task.

    A cold task with no matching groups receives the semantic route only.
    The semantic route always searches the whole pool, including experiences
    without citations. An empty pool returns empty candidates for the caller.
    """
    query = tuple(vector)
    groups = clusters.query_task_groups(query, limit=task_group_limit)
    # Request enough semantic results to fill the complete budget after dedup.
    semantic = clusters.query_experiences(query, limit=quota.total)
    historical = rank_experiences(
        clusters.assignments("experience"),
        dict(groups),
        ledger.by_task_group(clusters.assignments("task")),
    )
    candidates = merge_candidates(historical, semantic, quota=quota)
    return CandidateRoutes(
        tuple(groups), historical, tuple(semantic), candidates, quota
    )
