"""Rank individual experiences using task-group participation means.

The group mean is computed before query-similarity weighting. Sample counts
remain evidence, not extra weights. Experience clusters are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Iterable, Mapping

from .feedback import CreditStats


@dataclass(frozen=True)
class GroupEvidence:
    group_id: str
    similarity: float
    statistics: CreditStats


@dataclass(frozen=True)
class ExperienceScore:
    experience_id: str
    score: float
    evidence: tuple[GroupEvidence, ...]

    @property
    def tasks(self) -> int:
        return sum(item.statistics.tasks for item in self.evidence)


@dataclass(frozen=True)
class ExperienceRanking:
    """Scored results and unknown candidates are separate; unknown is not zero."""

    scored: tuple[ExperienceScore, ...]
    unknown_ids: tuple[str, ...]


def rank_experiences(
    experience_ids: Iterable[str],
    group_similarities: Mapping[str, float],
    group_statistics: Mapping[tuple[str, str], CreditStats],
) -> ExperienceRanking:
    """Apply the confirmed formula to deduplicated task-group statistics.

    Input statistics normally come from FeedbackLedger.by_task_group(). A task
    may belong to only one group in this snapshot. This function neither chooses
    an exploration budget nor decides whether a scored experience is applicable.
    """
    if isinstance(experience_ids, str):
        raise ValueError("experience_ids must contain individual experience IDs")
    ids = set()
    for experience_id in experience_ids:
        _validate_id(experience_id)
        ids.add(experience_id)
    for group_id, similarity in group_similarities.items():
        _validate_id(group_id)
        if (
            type(similarity) not in (int, float)
            or not isfinite(similarity)
            or not 0 <= similarity <= 1
        ):
            raise ValueError("group similarity must be a finite number in [0, 1]")

    scored: list[ExperienceScore] = []
    unknown: list[str] = []
    for experience_id in sorted(ids):
        evidence: list[GroupEvidence] = []
        seen_tasks: set[str] = set()
        for group_id, similarity in sorted(group_similarities.items()):
            statistics = group_statistics.get((group_id, experience_id))
            if similarity == 0 or statistics is None or statistics.tasks == 0:
                continue
            if seen_tasks.intersection(statistics.task_ids):
                raise ValueError("a task appears in multiple task groups")
            seen_tasks.update(statistics.task_ids)
            evidence.append(GroupEvidence(group_id, similarity, statistics))
        if not evidence:
            unknown.append(experience_id)
            continue
        weighted_sum = fsum(
            item.similarity * item.statistics.mean_reward for item in evidence
        )
        total_weight = fsum(item.similarity for item in evidence)
        scored.append(
            ExperienceScore(experience_id, weighted_sum / total_weight, tuple(evidence))
        )
    return ExperienceRanking(
        tuple(sorted(scored, key=lambda item: (-item.score, item.experience_id))),
        tuple(unknown),
    )


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("IDs must be nonempty strings")
