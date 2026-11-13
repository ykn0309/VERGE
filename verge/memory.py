"""Atomic checkpoints for shared-draft singleton trials and independent downstream credit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import tempfile

from verge.clustering import ClusterItem, OnlineClustering
from verge.feedback import CreditFeedback, FeedbackLedger
from verge.models import Experience, TrajectoryRecord

SCHEMA = "verge-v3.1"


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
    """Independent pairs only; Critic source attempts never enter the ledger."""
    records = []
    baseline = record.baseline
    for path in record.paths:
        if not path.provided_experience_ids:
            continue
        eid = path.provided_experience_ids[0]
        errors = [
            p.error or "missing judgment"
            for p in (baseline, path)
            if p is not None and p.correct is None
        ]
        if baseline is None:
            errors.append("missing baseline")
        records.append(
            CreditFeedback(
                record.sample.sample_id + ":" + eid,
                record.sample.task_id,
                protocol_id,
                eid,
                baseline.correct if baseline else None,
                path.correct,
                "; ".join(errors) or None,
            )
        )
    return records


def _initial_edges(record, pool):
    edges = []
    task = record.sample.task_id
    for path in record.paths:
        for eid in path.provided_experience_ids:
            if eid not in pool or pool[eid].source_task_id == task:
                raise ValueError(
                    "initial provision must reference an independent existing experience"
                )
            edges.append((task, eid))
    return edges


def _validate_path(path):
    if len(path.provided_experience_ids) > 1:
        raise ValueError("combination paths are not supported")
    if path.error is None and (path.rubric is None or path.judgment is None):
        raise ValueError("execution has no outcome and no error")
    if path.judgment is not None:
        from verge.calculator import calculate_rubric_judgment

        if (
            path.rubric is None
            or calculate_rubric_judgment(
                path.rubric, path.judgment.criterion_judgments
            ).to_dict()
            != path.judgment.to_dict()
        ):
            raise ValueError("saved judgment differs from program calculation")


def _validate_record(record):
    if record.mode != "train":
        raise ValueError("inference cannot enter training memory")
    if record.draft is None:
        if not record.error or record.paths or record.critic_rounds:
            raise ValueError("missing draft requires task failure without paths")
        return
    ids = [eid for p in record.paths for eid in p.provided_experience_ids]
    if (
        record.error
        or not record.paths
        or record.paths[0].provided_experience_ids
        or len(record.paths) != 1 + len(ids)
        or len(ids) > 3
        or len(set(ids)) != len(ids)
        or set(record.selection_reasons) != set(ids)
    ):
        raise ValueError("initial singleton trials are incomplete or duplicated")
    for path in record.paths:
        _validate_path(path)
    if record.critic_rounds and not all(p.correct is False for p in record.paths):
        raise ValueError("Critic requires all initial paths explicitly incorrect")
    if len(record.critic_rounds) > 2:
        raise ValueError("Critic round budget exceeded")
    for number, round_ in enumerate(record.critic_rounds, 1):
        if round_.round_number != number or len(round_.attempts) > 1:
            raise ValueError("invalid Critic round or candidate count")
        if number > 1:
            previous = record.critic_rounds[number - 2]
            if (
                previous.error
                or not previous.attempts
                or previous.attempts[0].admitted
                or previous.attempts[0].path.correct is not False
            ):
                raise ValueError("second proposal requires explicit candidate failure")
        for attempt in round_.attempts:
            _validate_path(attempt.path)
            if (
                attempt.experience.source_task_id != record.sample.task_id
                or attempt.path.provided_experience_ids
                != [attempt.experience.experience_id]
            ):
                raise ValueError("candidate provenance differs from validation path")
            if attempt.admitted and attempt.path.correct is not True:
                raise ValueError("admission requires successful source validation")


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
                "unsupported learning state; start a new v3.1 run, never resume legacy evidence"
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
            expected_feedback.extend(downstream_feedback(record, state.protocol_id))
        supplied = [CreditFeedback(**raw) for raw in data["feedback"]]
        if len({r.record_id for r in supplied}) != len(supplied):
            raise ValueError("duplicate feedback events")
        if expected_feedback != supplied:
            raise ValueError("feedback differs from saved independent trials")
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
        for feedback in downstream_feedback(record, self.protocol_id):
            next_state.ledger.append(feedback)
        next_state.trajectories[record.sample.task_id] = deepcopy(record)
        next_state.save()
        self.__dict__.update(next_state.__dict__)
