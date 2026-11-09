"""Atomic checkpoints for subset trajectories and independent downstream credit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import json
import os
import tempfile

from verge.clustering import ClusterItem, OnlineClustering
from verge.feedback import CreditFeedback, FeedbackLedger
from verge.models import Experience, TrajectoryRecord

SCHEMA = "verge-subset-learning-v2"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant")


def downstream_feedback(record, protocol_id):
    """Only the original selection receives credit; never inspect Critic rounds."""
    if record.actual is None:
        return None
    ids = tuple(record.actual.provided_experience_ids)
    indexed = {frozenset(p.provided_experience_ids): p for p in record.subset_paths}
    baseline = indexed.get(frozenset())
    singles = [indexed.get(frozenset([eid])) for eid in ids]
    paths = [baseline, *singles, record.actual]
    errors = [
        p.error if p is not None else "missing subset"
        for p in paths
        if p is None or p.correct is None
    ]
    return CreditFeedback(
        record.sample.sample_id,
        record.sample.task_id,
        protocol_id,
        ids,
        baseline.correct if baseline else None,
        record.actual.correct,
        tuple(p.correct if p else None for p in singles),
        error="; ".join(e or "missing outcome" for e in errors) or None,
    )


def _initial_edges(record, pool):
    if record.actual is None:
        return []
    task = record.sample.task_id
    edges = []
    for eid in record.actual.provided_experience_ids:
        if eid not in pool:
            raise ValueError("initial path refers to an unknown experience")
        if pool[eid].source_task_id != task:
            edges.append((task, eid))
    return edges


def _validate_record(record):
    if record.used_path is None:
        if not record.error or record.subset_paths:
            raise ValueError("missing execution without a task error")
        return
    ids = record.actual.provided_experience_ids if record.actual else []
    if len(ids) > 2 or len(ids) != len(set(ids)):
        raise ValueError("invalid selected experience set")
    expected = {
        frozenset(c) for size in range(len(ids) + 1) for c in combinations(ids, size)
    }
    indexed = {frozenset(p.provided_experience_ids): p for p in record.subset_paths}
    if len(indexed) != len(record.subset_paths) or set(indexed) != expected:
        raise ValueError("initial subsets are incomplete or duplicated")
    if (
        record.baseline is None
        or record.baseline.to_dict() != indexed[frozenset()].to_dict()
    ):
        raise ValueError("baseline differs from saved subset")
    if record.actual and record.actual.to_dict() != indexed[frozenset(ids)].to_dict():
        raise ValueError("actual differs from predetermined full subset")
    if set(record.selection_reasons) != set(ids):
        raise ValueError("selection reasons differ from selected experience set")
    for path in record.subset_paths:
        if path.error is None and (path.rubric is None or path.judgment is None):
            raise ValueError("execution has no outcome and no error")
    for round_ in record.critic_rounds:
        if any(
            a.experience.source_task_id != record.sample.task_id
            for a in round_.attempts
        ):
            raise ValueError("candidate source differs from its validation task")
        candidate_ids = [a.experience.experience_id for a in round_.attempts]
        if len(candidate_ids) > 2 or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("invalid Critic candidates")
        if not candidate_ids:
            continue
        expected = {
            frozenset(c)
            for size in range(len(candidate_ids) + 1)
            for c in combinations(candidate_ids, size)
        }
        paths = {frozenset(p.provided_experience_ids): p for p in round_.subset_paths}
        if set(paths) != expected or len(paths) != len(round_.subset_paths):
            raise ValueError("candidate validation subsets are incomplete")
        if paths[frozenset()].to_dict() != record.baseline.to_dict():
            raise ValueError("candidate baseline was replaced")
        for attempt in round_.attempts:
            eid = attempt.experience.experience_id
            if attempt.path.to_dict() != paths[frozenset([eid])].to_dict():
                raise ValueError("candidate singleton differs from its validation path")
            if not attempt.admitted:
                if attempt.admission_subset:
                    raise ValueError("failed candidate claims admission")
                continue
            subset = frozenset(attempt.admission_subset)
            if (
                eid not in subset
                or subset not in paths
                or paths[subset].correct is not True
            ):
                raise ValueError("admission lacks a successful validation subset")
            if len(subset) == 2:
                partners = [
                    a for a in round_.attempts if a.experience.experience_id in subset
                ]
                if any(
                    not a.admitted or frozenset(a.admission_subset) != subset
                    for a in partners
                ):
                    raise ValueError(
                        "joint admission must publish both candidates together"
                    )
                if any(
                    paths[frozenset([partner])].correct is True for partner in subset
                ):
                    raise ValueError(
                        "joint admission cannot bypass successful singleton policy"
                    )


class ExperienceMemory:
    def __init__(self, protocol_id, clustering, *, path=None, read_only=False):
        self.protocol_id = protocol_id
        self.clusters = clustering
        self.ledger = FeedbackLedger(protocol_id)
        self.experiences = {}
        self.trajectories = {}
        self.path = Path(path) if path else None
        self.read_only = read_only

    def snapshot(self):
        result = deepcopy(self)
        result.read_only, result.path = True, None
        return result

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "protocol_id": self.protocol_id,
            "clustering": self.clusters.to_dict(),
            "experiences": [e.to_dict() for e in self.experiences.values()],
            "feedback": [asdict(r) for r in self.ledger.records],
            "trajectories": [r.to_dict() for r in self.trajectories.values()],
        }

    @classmethod
    def load(cls, path, *, protocol_id=None, read_only=True):
        data = json.loads(Path(path).read_text(), parse_constant=_invalid_constant)
        if (
            set(data)
            != {
                "schema",
                "protocol_id",
                "clustering",
                "experiences",
                "feedback",
                "trajectories",
            }
            or data["schema"] != SCHEMA
        ):
            raise ValueError(
                "unsupported learning state; use a new v2 run, never resume legacy evidence"
            )
        if protocol_id is not None and data["protocol_id"] != protocol_id:
            raise ValueError("execution protocol changed")
        state = cls(
            data["protocol_id"],
            OnlineClustering.from_dict(data["clustering"]),
            path=path,
            read_only=read_only,
        )
        for raw in data["experiences"]:
            e = Experience.from_dict(raw)
            if e.experience_id in state.experiences:
                raise ValueError("duplicate experience")
            state.experiences[e.experience_id] = e
        for raw in data["trajectories"]:
            record = TrajectoryRecord.from_dict(raw)
            _validate_record(record)
            if record.sample.task_id in state.trajectories:
                raise ValueError("duplicate task trajectory")
            state.trajectories[record.sample.task_id] = record
        if len({r.sample.sample_id for r in state.trajectories.values()}) != len(
            state.trajectories
        ):
            raise ValueError("duplicate sample IDs")
        if set(state.experiences) != set(state.clusters.assignments("experience")):
            raise ValueError("experience pool and cluster objects differ")
        item_texts = {
            (i["kind"], i["object_id"]): i["text"] for i in data["clustering"]["items"]
        }
        admitted = {
            a.experience.experience_id: a.experience
            for r in state.trajectories.values()
            for cr in r.critic_rounds
            for a in cr.attempts
            if a.admitted
        }
        if admitted != state.experiences:
            raise ValueError("experience pool differs from admitted candidates")
        for e in state.experiences.values():
            if (
                e.source_task_id not in state.trajectories
                or set(e.parent_ids) - state.experiences.keys()
                or e.experience_id in e.parent_ids
            ):
                raise ValueError("invalid experience provenance")
            if item_texts[("experience", e.experience_id)] != json.dumps(
                e.method.to_dict(), ensure_ascii=False, sort_keys=True
            ):
                raise ValueError("experience text differs from its embedded text")
        for (kind, oid), text in item_texts.items():
            if kind == "task" and (
                oid not in state.trajectories
                or state.trajectories[oid].sample.task != text
            ):
                raise ValueError("task cluster lacks original trajectory")
        expected_feedback = []
        expected_edges = set()
        for record in state.trajectories.values():
            expected_edges.update(_initial_edges(record, state.experiences))
            feedback = downstream_feedback(record, state.protocol_id)
            if feedback:
                # An initial result cannot credit a candidate from its source task.
                if any(
                    state.experiences[e].source_task_id == record.sample.task_id
                    for e in feedback.experience_ids
                ):
                    raise ValueError("source task entered downstream credit")
                expected_feedback.append(feedback)
        supplied = [CreditFeedback(**raw) for raw in data["feedback"]]
        if len({r.record_id for r in supplied}) != len(supplied):
            raise ValueError("duplicate feedback events")
        if expected_feedback != supplied:
            raise ValueError("feedback differs from saved subset execution")
        if set(map(tuple, data["clustering"]["citations"])) != expected_edges:
            raise ValueError("associations differ from independent initial provision")
        state.ledger = FeedbackLedger(state.protocol_id, supplied)
        return state

    def save(self):
        if self.read_only:
            raise ValueError("frozen state is read-only")
        if self.path:
            atomic_json(self.path, self.to_dict())

    def commit(self, result):
        if self.read_only:
            raise ValueError("frozen state is read-only")
        record = result.record
        _validate_record(record)
        if record.sample.task_id in self.trajectories:
            raise ValueError("task already committed")
        if any(
            r.sample.sample_id == record.sample.sample_id
            for r in self.trajectories.values()
        ):
            raise ValueError("sample ID already committed")
        next_state = deepcopy(self)
        items = []
        if result.task_vector is not None:
            items.append(
                ClusterItem(
                    record.sample.task_id,
                    "task",
                    record.sample.task,
                    result.task_vector,
                )
            )
        expected = {
            a.experience.experience_id: a.experience
            for cr in record.critic_rounds
            for a in cr.attempts
            if a.admitted
        }
        if {e.experience_id: e for e, _ in result.admitted} != expected:
            raise ValueError("published candidates differ from admission decisions")
        for e, vector in result.admitted:
            if (
                e.source_task_id != record.sample.task_id
                or set(e.parent_ids) - self.experiences.keys()
                or e.experience_id in self.experiences
            ):
                raise ValueError("invalid new experience provenance")
            next_state.experiences[e.experience_id] = deepcopy(e)
            items.append(
                ClusterItem(
                    e.experience_id,
                    "experience",
                    json.dumps(e.method.to_dict(), ensure_ascii=False, sort_keys=True),
                    vector,
                )
            )
        edges = _initial_edges(record, self.experiences)
        next_state.clusters.update(items, edges)
        feedback = downstream_feedback(record, self.protocol_id)
        if feedback:
            next_state.ledger.append(feedback)
        next_state.trajectories[record.sample.task_id] = deepcopy(record)
        next_state.save()
        self.__dict__.update(next_state.__dict__)
