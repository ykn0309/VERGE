"""Online, alternating task/experience clustering from text and provision edges."""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict, field
from math import hypot, isfinite
from typing import Iterable, Literal
import hashlib
import json
import numpy as np

from verge.partitioning import ModularitySearch, canonical

Kind = Literal["task", "experience"]
Key = tuple[Kind, str]
KINDS = ("task", "experience")


def _unit_vector(values):
    vector = tuple(values)
    if not vector or any(
        type(v) not in (int, float) or not isfinite(v) for v in vector
    ):
        raise ValueError("vector must contain finite numbers")
    norm = hypot(*vector)
    if not norm or not isfinite(norm):
        raise ValueError("vector must have finite positive norm")
    return tuple(v / norm for v in vector)


@dataclass(frozen=True)
class ClusterItem:
    object_id: str
    kind: Kind
    text: str
    vector: tuple[float, ...]

    def __post_init__(self):
        if self.kind not in KINDS or any(
            not isinstance(v, str) or not v.strip() for v in (self.object_id, self.text)
        ):
            raise ValueError("invalid cluster item identity")
        object.__setattr__(self, "vector", _unit_vector(self.vector))

    @property
    def key(self):
        return self.kind, self.object_id


@dataclass(frozen=True)
class ClusterUpdate:
    rounds: int
    moves: int
    stable: bool
    new_items: int
    new_edges: int
    status: str
    half_rounds: list
    initializations: list = field(default_factory=list)


class OnlineClustering:
    def __init__(
        self,
        *,
        reference_weight=0.5,
        task_threshold=0.35,
        experience_threshold=0.50,
        max_rounds=5,
        search_rounds=100,
    ):
        for value in (reference_weight, task_threshold, experience_threshold):
            if (
                type(value) not in (float, int)
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("weights and thresholds must be finite in [0,1]")
        if min(task_threshold, experience_threshold) <= 0:
            raise ValueError("group floors must be positive")
        if (
            any(type(v) is not int or v < 1 for v in (max_rounds, search_rounds))
            or max_rounds > 5
        ):
            raise ValueError("round budgets must be positive integers")
        self.reference_weight = reference_weight
        self.thresholds = dict(task=task_threshold, experience=experience_threshold)
        self.max_rounds, self.search_rounds = max_rounds, search_rounds
        self._items, self._references, self._membership = {}, {}, {}
        self._dimension = None
        self.last_update = None

    def assignments(self, kind):
        return {oid: group for (k, oid), group in self._membership.items() if k == kind}

    def groups(self, kind):
        groups = {}
        for oid, group in self.assignments(kind).items():
            groups.setdefault(group, []).append(oid)
        return {g: tuple(sorted(m)) for g, m in sorted(groups.items())}

    def references(self, kind, object_id):
        return frozenset(self._references[(kind, object_id)])

    def _profile(self, key):
        other = "experience" if key[0] == "task" else "task"
        refs = self._references[key]
        counts = Counter(self._membership[(other, oid)] for oid in refs)
        return {g: n / len(refs) for g, n in counts.items()} if refs else {}

    def similarity(self, kind, left, right):
        a, b = (kind, left), (kind, right)
        text = float(
            np.clip(np.dot(self._items[a].vector, self._items[b].vector), 0, 1)
        )
        pa, pb = self._profile(a), self._profile(b)
        if not pa or not pb:
            return text
        keys = pa.keys() | pb.keys()
        overlap = sum(min(pa.get(g, 0), pb.get(g, 0)) for g in keys) / sum(
            max(pa.get(g, 0), pb.get(g, 0)) for g in keys
        )
        m = min(len(self._references[a]), len(self._references[b]))
        weight = self.reference_weight * m / (m + 1)
        return (1 - weight) * text + weight * overlap

    def _matrix(self, kind, *, text_only=False):
        ids = sorted(self.assignments(kind))
        if not ids:
            return ids, np.zeros((0, 0))
        vectors = np.array([self._items[(kind, oid)].vector for oid in ids])
        matrix = np.clip(vectors @ vectors.T, 0, 1)
        if not text_only:
            profiles = [self._profile((kind, oid)) for oid in ids]
            counts = [len(self._references[(kind, oid)]) for oid in ids]
            for i, pa in enumerate(profiles):
                if not pa:
                    continue
                for j in range(i):
                    pb = profiles[j]
                    if not pb:
                        continue
                    keys = pa.keys() | pb.keys()
                    overlap = sum(min(pa.get(g, 0), pb.get(g, 0)) for g in keys) / sum(
                        max(pa.get(g, 0), pb.get(g, 0)) for g in keys
                    )
                    m = min(counts[i], counts[j])
                    weight = self.reference_weight * m / (m + 1)
                    matrix[i, j] = matrix[j, i] = (1 - weight) * matrix[
                        i, j
                    ] + weight * overlap
        np.fill_diagonal(matrix, 0)
        return ids, matrix

    def _partition(self, kind, ids):
        positions = {oid: i for i, oid in enumerate(ids)}
        return canonical(
            [positions[oid] for oid in members]
            for members in self.groups(kind).values()
        )

    def _publish(self, kind, ids, groups):
        for group in groups:
            # Canonical ID based on members; no arbitrary cluster numbering.
            members = [ids[i] for i in group]
            gid = (
                kind
                + "_"
                + hashlib.sha256(json.dumps(members).encode()).hexdigest()[:20]
            )
            for oid in members:
                self._membership[(kind, oid)] = gid

    def _signature(self):
        return tuple(tuple(sorted(self.groups(kind).values())) for kind in KINDS)

    def _half(self, kind, number):
        other = "experience" if kind == "task" else "task"
        opposite = self.assignments(other)
        ids, matrix = self._matrix(kind)
        search = ModularitySearch(matrix, self.thresholds[kind], self.search_rounds)
        result = search.run(self._partition(kind, ids))
        self._publish(kind, ids, result.groups)
        minimums = {}
        for group in result.groups:
            if len(group) > 1:
                gid = self._membership[(kind, ids[group[0]])]
                minimums[gid] = float(
                    (matrix[np.ix_(group, group)].sum(axis=1) / (len(group) - 1)).min()
                )
        return {
            "round": number,
            "kind": kind,
            "object_ids": ids,
            "opposite_assignments": opposite,
            "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "quality": result.quality,
            "minimum_member_averages": minimums,
            "repairs": [
                {**r, "object_id": ids[r["object_index"]]} for r in result.repairs
            ],
            "moves": result.moves,
            "budget": result.budget,
        }

    def update(self, items=(), citations=()):
        additions = {}
        dimension = self._dimension
        for item in items:
            if dimension is not None and len(item.vector) != dimension:
                raise ValueError("inconsistent embedding dimensions")
            dimension = len(item.vector)
            previous = additions.get(item.key, self._items.get(item.key))
            if previous is not None and previous != item:
                raise ValueError("immutable cluster item changed")
            additions[item.key] = item
        edges = set(citations)
        known = self._items.keys() | additions.keys()
        if any(
            ("task", t) not in known or ("experience", e) not in known for t, e in edges
        ):
            raise ValueError("citation endpoints must exist")
        new = sorted(additions.keys() - self._items.keys())
        self._dimension = dimension
        self._items.update(additions)
        for key in new:
            self._references[key] = set()
            self._membership[key] = key[0] + "_new_" + key[1]
        count = 0
        for t, e in sorted(edges):
            if e not in self._references[("task", t)]:
                self._references[("task", t)].add(e)
                self._references[("experience", e)].add(t)
                count += 1
        initializations = []
        # New objects are text-initialized without reassigning old objects.
        for kind in KINDS:
            ids, matrix = self._matrix(kind, text_only=True)
            movable = [i for i, oid in enumerate(ids) if (kind, oid) in new]
            if movable:
                search = ModularitySearch(
                    matrix, self.thresholds[kind], self.search_rounds
                )
                first = len(movable) == len(ids)
                result = search.run(
                    self._partition(kind, ids), movable=None if first else movable
                )
                self._publish(kind, ids, result.groups)
                initializations.append(
                    {
                        "kind": kind,
                        "quality": result.quality,
                        "budget": result.budget,
                        "moves": result.moves,
                        "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                    }
                )
        seen, halves, moves = {self._signature()}, [], 0
        status = "round_cap"
        for number in range(1, self.max_rounds + 1):
            old = self._signature()
            for kind in KINDS:
                diagnostic = self._half(kind, number)
                halves.append(diagnostic)
                moves += diagnostic["moves"]
            signature = self._signature()
            if signature == old:
                status = "stable"
                break
            if signature in seen:
                status = "cycle"
                break
            seen.add(signature)
        if any(h["budget"] for h in [*initializations, *halves]):
            status = "budget"
        for h in halves:
            other = "experience" if h["kind"] == "task" else "task"
            h["current_opposite"] = h["opposite_assignments"] == self.assignments(other)
        self.last_update = ClusterUpdate(
            number,
            moves,
            status == "stable",
            len(new),
            count,
            status,
            halves,
            initializations,
        )
        return self.last_update

    def _query(self, vector):
        query = _unit_vector(vector)
        if self._dimension is not None and len(query) != self._dimension:
            raise ValueError("query embedding dimension differs")
        return np.array(query)

    def query_task_groups(self, vector, *, limit=3):
        query = self._query(vector)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be positive")
        matches = [
            (
                g,
                float(
                    np.mean(
                        [
                            np.clip(
                                np.dot(query, self._items[("task", oid)].vector), 0, 1
                            )
                            for oid in members
                        ]
                    )
                ),
            )
            for g, members in self.groups("task").items()
        ]
        return sorted(matches, key=lambda m: (-m[1], m[0]))[:limit]

    def query_experiences(self, vector, *, limit):
        query = self._query(vector)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be positive")
        matches = [
            (oid, float(np.clip(np.dot(query, item.vector), 0, 1)))
            for (kind, oid), item in self._items.items()
            if kind == "experience"
        ]
        return sorted(matches, key=lambda m: (-m[1], m[0]))[:limit]

    def to_dict(self):
        return {
            "parameters": {
                "reference_weight": self.reference_weight,
                "task_threshold": self.thresholds["task"],
                "experience_threshold": self.thresholds["experience"],
                "max_rounds": self.max_rounds,
                "search_rounds": self.search_rounds,
            },
            "items": [asdict(self._items[key]) for key in sorted(self._items)],
            "citations": [
                (oid, e)
                for (kind, oid) in sorted(self._items)
                if kind == "task"
                for e in sorted(self._references[(kind, oid)])
            ],
            "assignments": {kind: self.assignments(kind) for kind in KINDS},
            "last_update": asdict(self.last_update) if self.last_update else None,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {
            "parameters",
            "items",
            "citations",
            "assignments",
            "last_update",
        }:
            raise ValueError("invalid cluster snapshot fields")
        state = cls(**data["parameters"])
        for raw in data["items"]:
            item = ClusterItem(**raw)
            if abs(hypot(*raw["vector"]) - 1) > 1e-12 or item.key in state._items:
                raise ValueError("invalid stored vector or duplicate item")
            if state._dimension is not None and len(item.vector) != state._dimension:
                raise ValueError("inconsistent embedding dimensions")
            object.__setattr__(item, "vector", tuple(raw["vector"]))
            state._dimension = len(item.vector)
            state._items[item.key] = item
            state._references[item.key] = set()
        for kind, assignments in data["assignments"].items():
            if kind not in KINDS:
                raise ValueError("unknown cluster kind")
            for oid, group in assignments.items():
                if (
                    (kind, oid) not in state._items
                    or not isinstance(group, str)
                    or not group.startswith(kind + "_")
                ):
                    raise ValueError("invalid group assignment")
                state._membership[(kind, oid)] = group
        if state._membership.keys() != state._items.keys():
            raise ValueError("incomplete group assignments")
        for t, e in data["citations"]:
            if ("task", t) not in state._items or ("experience", e) not in state._items:
                raise ValueError("unknown citation endpoint")
            if e in state._references[("task", t)]:
                raise ValueError("duplicate citation")
            state._references[("task", t)].add(e)
            state._references[("experience", e)].add(t)
        state.last_update = (
            ClusterUpdate(**data["last_update"]) if data["last_update"] else None
        )
        return state
