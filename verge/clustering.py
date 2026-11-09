"""Incremental task/experience grouping with fixed vectors and citation sets.

Task groups support individual ranking; experience groups are analysis only.
Neither grouping is an applicability or quality judgment.
No embeddings are trained here, and querying never changes the learning state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isclose, isfinite
from typing import Iterable, Literal


Kind = Literal["task", "experience"]
Key = tuple[Kind, str]


def _unit_vector(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(values)
    if not vector or any(
        type(v) not in (int, float) or not isfinite(v) for v in vector
    ):
        raise ValueError("vector must contain finite numbers")
    norm = hypot(*vector)
    if norm == 0 or not isfinite(norm):
        raise ValueError("vector must have a finite positive norm")
    return tuple(value / norm for value in vector)


def _strictly_better(left: float, right: float) -> bool:
    # Equal geometric scores can differ by a few floating-point rounding bits.
    return left > right and not isclose(left, right, rel_tol=0, abs_tol=1e-12)


def _meets_threshold(score: float, threshold: float) -> bool:
    return not _strictly_better(threshold, score)


@dataclass(frozen=True)
class ClusterItem:
    object_id: str
    kind: Kind
    text: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("task", "experience"):
            raise ValueError("kind must be task or experience")
        if any(
            not isinstance(v, str) or not v.strip() for v in (self.object_id, self.text)
        ):
            raise ValueError("object_id and text must be nonempty strings")
        object.__setattr__(self, "vector", _unit_vector(self.vector))

    @property
    def key(self) -> Key:
        return self.kind, self.object_id


@dataclass(frozen=True)
class ClusterUpdate:
    rounds: int
    moves: int
    stable: bool
    new_items: int
    new_edges: int


class OnlineClustering:
    """Exact member-average grouping; parameters are supplied by the caller.

    An update validates all inputs before changing state, inserts new objects,
    and makes bounded sequential relocation sweeps. Group IDs are never reused.
    """

    def __init__(
        self,
        *,
        reference_weight: float,
        task_threshold: float,
        experience_threshold: float,
        max_rounds: int = 5,
    ) -> None:
        for value in (reference_weight, task_threshold, experience_threshold):
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "weights and thresholds must be finite numbers in [0, 1]"
                )
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        self.reference_weight = reference_weight
        self.thresholds = {"task": task_threshold, "experience": experience_threshold}
        self.max_rounds = max_rounds
        self._items: dict[Key, ClusterItem] = {}
        self._references: dict[Key, set[str]] = {}
        self._membership: dict[Key, str] = {}
        self._groups: dict[str, set[Key]] = {}
        self._next_group = 1
        self._pair_scores: dict[tuple[Key, Key], float] = {}
        self._dimension: int | None = None

    def assignments(self, kind: Kind) -> dict[str, str]:
        return {
            key[1]: group for key, group in self._membership.items() if key[0] == kind
        }

    def groups(self, kind: Kind) -> dict[str, tuple[str, ...]]:
        return {
            group: tuple(sorted(key[1] for key in members))
            for group, members in self._groups.items()
            if next(iter(members))[0] == kind
        }

    def references(self, kind: Kind, object_id: str) -> frozenset[str]:
        return frozenset(self._references[(kind, object_id)])

    def similarity(self, kind: Kind, left: str, right: str) -> float:
        a, b = (kind, left), (kind, right)
        pair = tuple(sorted((a, b)))
        if pair not in self._pair_scores:
            self._pair_scores[pair] = self._score(
                self._items[a].vector,
                self._references[a],
                self._items[b].vector,
                self._references[b],
            )
        return self._pair_scores[pair]

    def _score(
        self,
        left: tuple[float, ...],
        left_refs: set[str],
        right: tuple[float, ...],
        right_refs: set[str],
    ) -> float:
        text = max(0.0, min(1.0, fsum(a * b for a, b in zip(left, right))))
        if not left_refs or not right_refs:
            return text
        support = min(len(left_refs), len(right_refs))
        weight = self.reference_weight * support / (support + 1)
        overlap = len(left_refs & right_refs) / len(left_refs | right_refs)
        return (1 - weight) * text + weight * overlap

    def _group_score(self, key: Key, group: str) -> float:
        others = sorted(self._groups[group] - {key})
        if not others:
            return self.thresholds[key[0]]
        return fsum(
            self.similarity(key[0], key[1], other[1]) for other in others
        ) / len(others)

    def _best_group(self, key: Key, *, exclude: str | None = None) -> str | None:
        candidates = [
            group
            for group, members in self._groups.items()
            if group != exclude and next(iter(members))[0] == key[0]
        ]
        if not candidates:
            return None
        best = min(candidates)
        for group in sorted(candidates):
            if _strictly_better(
                self._group_score(key, group), self._group_score(key, best)
            ):
                best = group
        return best

    def _place(self, key: Key, group: str | None) -> None:
        previous = self._membership.get(key)
        if previous is not None:
            self._groups[previous].remove(key)
            if not self._groups[previous]:
                del self._groups[previous]
        if group is None:
            group = f"{key[0]}_{self._next_group:012d}"
            self._next_group += 1
            self._groups[group] = set()
        self._groups[group].add(key)
        self._membership[key] = group

    def update(
        self,
        items: Iterable[ClusterItem] = (),
        citations: Iterable[tuple[str, str]] = (),
    ) -> ClusterUpdate:
        """Commit item/citation additions, then check every item for relocation.

        Each citation is (task ID, experience ID). Resubmitting an immutable
        object or existing edge is safe; changing an existing item is rejected.
        """
        additions: dict[Key, ClusterItem] = {}
        dimension = self._dimension
        for item in items:
            if dimension is not None and len(item.vector) != dimension:
                raise ValueError("all vectors must use the same embedding dimension")
            dimension = len(item.vector)
            previous = additions.get(item.key, self._items.get(item.key))
            if previous is not None and previous != item:
                raise ValueError(f"immutable cluster item changed: {item.object_id}")
            additions[item.key] = item
        edges = set(citations)
        known = self._items.keys() | additions.keys()
        for task_id, experience_id in edges:
            if ("task", task_id) not in known or (
                "experience",
                experience_id,
            ) not in known:
                raise ValueError("citation endpoints must exist in the same snapshot")
        # All external input validation completes before mutation.
        new_keys = sorted(additions.keys() - self._items.keys())
        self._dimension = dimension
        self._items.update(additions)
        for key in new_keys:
            self._references[key] = set()
        changed: set[Key] = set()
        new_edges = 0
        for task_id, experience_id in sorted(edges):
            task, experience = ("task", task_id), ("experience", experience_id)
            if experience_id not in self._references[task]:
                self._references[task].add(experience_id)
                self._references[experience].add(task_id)
                changed.update((task, experience))
                new_edges += 1
        self._pair_scores = {
            pair: score
            for pair, score in self._pair_scores.items()
            if not changed.intersection(pair)
        }
        for key in new_keys:
            group = self._best_group(key)
            if group is not None and _meets_threshold(
                self._group_score(key, group), self.thresholds[key[0]]
            ):
                self._place(key, group)
            else:
                self._place(key, None)
        moves = 0
        for round_number in range(1, self.max_rounds + 1):
            moved = 0
            for key in sorted(self._items):
                current = self._membership[key]
                current_score = self._group_score(key, current)
                best = self._best_group(key, exclude=current)
                if (
                    best is not None
                    and _meets_threshold(
                        self._group_score(key, best), self.thresholds[key[0]]
                    )
                    and _strictly_better(self._group_score(key, best), current_score)
                ):
                    self._place(key, best)
                elif len(self._groups[current]) > 1 and _strictly_better(
                    self.thresholds[key[0]], current_score
                ):
                    self._place(key, None)
                else:
                    continue
                moved += 1
            moves += moved
            if not moved:
                return ClusterUpdate(
                    round_number, moves, True, len(new_keys), new_edges
                )
        return ClusterUpdate(self.max_rounds, moves, False, len(new_keys), new_edges)

    def query_task_groups(
        self, vector: Iterable[float], *, limit: int = 3
    ) -> list[tuple[str, float]]:
        """Rank matching task groups without inserting the query or new edges."""
        query = _unit_vector(vector)
        if self._dimension is not None and len(query) != self._dimension:
            raise ValueError("query uses a different embedding dimension")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        matches = []
        for group, members in self._groups.items():
            if next(iter(members))[0] != "task":
                continue
            score = fsum(
                self._score(
                    query, set(), self._items[key].vector, self._references[key]
                )
                for key in sorted(members)
            ) / len(members)
            if _meets_threshold(score, self.thresholds["task"]):
                matches.append((group, score))
        return sorted(matches, key=lambda match: (-match[1], match[0]))[:limit]

    def query_experiences(
        self, vector: Iterable[float], *, limit: int
    ) -> list[tuple[str, float]]:
        """Search all individual experience texts, regardless of group or edges."""
        query = _unit_vector(vector)
        if self._dimension is not None and len(query) != self._dimension:
            raise ValueError("query uses a different embedding dimension")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        matches = [
            (
                key[1],
                self._score(query, set(), item.vector, set()),
            )
            for key, item in self._items.items()
            if key[0] == "experience"
        ]
        return sorted(matches, key=lambda match: (-match[1], match[0]))[:limit]

    def to_dict(self) -> dict:
        """Persist exact membership; loading must not silently recluster history."""
        from dataclasses import asdict

        return {
            "parameters": {
                "reference_weight": self.reference_weight,
                "task_threshold": self.thresholds["task"],
                "experience_threshold": self.thresholds["experience"],
                "max_rounds": self.max_rounds,
            },
            "items": [asdict(self._items[key]) for key in sorted(self._items)],
            "citations": [
                (key[1], eid)
                for key in sorted(self._items)
                if key[0] == "task"
                for eid in sorted(self._references[key])
            ],
            "assignments": {
                kind: self.assignments(kind) for kind in ("task", "experience")
            },
            "next_group": self._next_group,
        }

    @classmethod
    def from_dict(cls, data: dict):
        if set(data) != {
            "parameters",
            "items",
            "citations",
            "assignments",
            "next_group",
        }:
            raise ValueError("invalid cluster snapshot fields")
        state = cls(**data["parameters"])
        for raw in data["items"]:
            item = ClusterItem(**raw)
            if abs(hypot(*raw["vector"]) - 1) > 1e-12:
                raise ValueError("stored vector is not normalized")
            # Preserve serialized normalized bits for immutable-item comparisons.
            object.__setattr__(item, "vector", tuple(raw["vector"]))
            if item.key in state._items:
                raise ValueError("duplicate cluster item")
            if state._dimension is not None and len(item.vector) != state._dimension:
                raise ValueError("inconsistent embedding dimensions")
            state._dimension = len(item.vector)
            state._items[item.key] = item
            state._references[item.key] = set()
        for kind, assignments in data["assignments"].items():
            if kind not in ("task", "experience"):
                raise ValueError("unknown cluster kind")
            for oid, group in assignments.items():
                key = (kind, oid)
                if (
                    key not in state._items
                    or not isinstance(group, str)
                    or not group.startswith(kind + "_")
                ):
                    raise ValueError("invalid group assignment")
                state._membership[key] = group
                state._groups.setdefault(group, set()).add(key)
        if state._membership.keys() != state._items.keys():
            raise ValueError("incomplete group assignments")
        for task, eid in data["citations"]:
            if ("task", task) not in state._items or (
                "experience",
                eid,
            ) not in state._items:
                raise ValueError("unknown citation endpoint")
            state._references[("task", task)].add(eid)
            state._references[("experience", eid)].add(task)
        state._next_group = data["next_group"]
        if type(state._next_group) is not int or state._next_group <= max(
            (int(group.rsplit("_", 1)[1]) for group in state._groups), default=0
        ):
            raise ValueError("invalid group counter")
        return state
