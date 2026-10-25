"""V2 immutable-version ledger. One atomic file commit per state transition."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json, os, tempfile
from verge.models import Experience, ApplicationRecord, RevisionRecord, TrajectoryRecord
from verge.calculator import update_quality, rebuild_quality


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, encoding="utf8"
        ) as f:
            name = f.name
            os.chmod(name, 0o600)
            json.dump(data, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


class ExperienceMemory:
    SCHEMA = 11

    def __init__(
        self, root_dir, *, read_only=False, parameters=None, allow_reestimate=False
    ):
        self.root_dir = Path(root_dir)
        self.path = self.root_dir / "experience_memory.json"
        self.read_only = read_only
        self.parameters = parameters
        self._experiences = {}
        self.revisions = []
        self.parameter_events = []
        self.parameter_snapshot = parameters.to_dict() if parameters else None
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if (
                set(data)
                != {
                    "schema_version",
                    "experiences",
                    "revisions",
                    "parameters",
                    "parameter_events",
                }
                or data["schema_version"] != self.SCHEMA
            ):
                raise ValueError("Memory schema 11 required; V1 data cannot be resumed")
            self.parameter_snapshot = data["parameters"]
            self.parameter_events = data["parameter_events"]
            items = [Experience.from_dict(x) for x in data["experiences"]]
            if len({x.experience_id for x in items}) != len(items):
                raise ValueError("duplicate experience ID")
            self._experiences = {x.experience_id: x for x in items}
            self.revisions = [RevisionRecord.from_dict(x) for x in data["revisions"]]
            self._validate(self._experiences, self.revisions)
            incoming = parameters.to_dict() if parameters else None
            if (
                not read_only
                and not allow_reestimate
                and incoming != self.parameter_snapshot
            ):
                raise ValueError(
                    "parameter contents/status changed; use explicit reestimate"
                )
        elif not read_only:
            self._commit({}, [])

    @staticmethod
    def _validate(experiences, revisions):
        event_ids = {r.event_id for r in revisions}
        if len(event_ids) != len(revisions):
            raise ValueError("duplicate revision event")
        for e in experiences.values():
            e.__post_init__()
            if e.parent_id is not None and e.parent_id not in experiences:
                raise ValueError("missing parent")
            if e.retired_by_event_id:
                retirement = next(
                    (r for r in revisions if r.event_id == e.retired_by_event_id), None
                )
                if (
                    retirement is None
                    or not retirement.accepted
                    or retirement.parent_id != e.experience_id
                    or e.status != "retired"
                ):
                    raise ValueError("invalid retirement event")
            if e.parent_id is not None and not e.created_by_event_id:
                raise ValueError("child requires creation event")
            for a in e.applications:
                if a.version_id != e.experience_id or a.method != e.method:
                    raise ValueError("application version/method mismatch")
            lineage = set()
            current = e
            while current.parent_id:
                if current.experience_id in lineage:
                    raise ValueError("cyclic version lineage")
                lineage.add(current.experience_id)
                current = experiences[current.parent_id]
            if e.created_by_event_id:
                event = next(
                    (r for r in revisions if r.event_id == e.created_by_event_id), None
                )
                if (
                    event is None
                    or not event.accepted
                    or event.candidate is None
                    or event.candidate.experience_id != e.experience_id
                    or event.candidate.method != e.method
                ):
                    raise ValueError("invalid creation event")
        for r in revisions:
            if r.parent_id not in experiences:
                raise ValueError("missing revision parent")
            if r.accepted:
                child = r.candidate
                parent = experiences[r.parent_id]
                if (
                    child is None
                    or child.experience_id not in experiences
                    or child.parent_id != parent.experience_id
                    or child.status != "active"
                    or child.created_by_event_id != r.event_id
                    or parent.retired_by_event_id != r.event_id
                ):
                    raise ValueError("inconsistent accepted revision")
        roots = {}
        for e in experiences.values():
            current = e
            while current.parent_id:
                current = experiences[current.parent_id]
            if e.status == "active":
                if current.experience_id in roots:
                    raise ValueError("multiple active versions in lineage")
                roots[current.experience_id] = e.experience_id

    def _commit(self, experiences, revisions):
        if self.read_only:
            raise ValueError("memory is frozen")
        self._validate(experiences, revisions)
        atomic_json(
            self.path,
            {
                "schema_version": self.SCHEMA,
                "experiences": [e.to_dict() for e in experiences.values()],
                "revisions": [r.to_dict() for r in revisions],
                "parameters": self.parameter_snapshot,
                "parameter_events": self.parameter_events,
            },
        )
        self._experiences = deepcopy(experiences)
        self.revisions = deepcopy(revisions)

    def all(self, *, active_only=False):
        return deepcopy(
            [
                e
                for e in self._experiences.values()
                if not active_only or e.status == "active"
            ]
        )

    def get(self, eid):
        return deepcopy(self._experiences[eid])

    def add(self, e, source):
        if e.experience_id in self._experiences:
            raise ValueError("version already exists")
        if (
            source.phase != "admission"
            or source.feedback != "positive"
            or source.version_id != e.experience_id
            or source.sample.sample_id != e.source_sample_id
            or source.method != e.method
        ):
            raise ValueError(
                "admission requires actually referenced net-positive candidate evidence"
            )
        e = deepcopy(e)
        e.status = "active"
        e.source_task_id = source.sample.task_id
        e.parameter_digest = self.parameters.fingerprint if self.parameters else None
        e.applications = [deepcopy(source)]
        update_quality(e, e.applications[0], self.parameters)
        if self.parameters and self.parameters.status == "usable":
            e.q_prior = e.q_posterior = self.parameters.q_prior
            e.parameter_version = self.parameters.version
        versions = deepcopy(self._experiences)
        versions[e.experience_id] = e
        self._commit(versions, self.revisions)

    def record(self, app):
        if app.phase != "downstream":
            raise ValueError(
                "admission/replay evidence must use their explicit lifecycle commits"
            )
        versions = deepcopy(self._experiences)
        e = versions[app.version_id]
        if e.status != "active" or e.method != app.method:
            raise ValueError("application must use the active immutable version")
        if any(
            a.phase == app.phase and a.sample.sample_id == app.sample.sample_id
            for a in e.applications
        ):
            app.update_reason = "duplicate"
            return app
        duplicate = (
            any(a.sample.task_id == app.sample.task_id for a in e.applications)
            or app.sample.sample_id == e.source_sample_id
            or app.sample.task_id == e.source_task_id
        )
        if duplicate:
            app.q_before = app.q_after = e.q_posterior
            app.update_reason = "duplicate_task"
        else:
            update_quality(e, app, self.parameters)
        e.applications.append(deepcopy(app))
        self._commit(versions, self.revisions)
        return app

    def mark_checked(self, eid, reasons=(), window=6):
        versions = deepcopy(self._experiences)
        e = versions[eid]
        for stage in ("selector", "generator", "judge"):
            if stage + "_errors" in reasons:
                used = set(e.consumed_error_tasks.get(stage, []))
                used.update(
                    a.sample.task_id
                    for a in e.downstream[-window:]
                    if any(r.stage == stage and r.status == "error" for r in a.reports)
                )
                e.consumed_error_tasks[stage] = sorted(used)
        e.last_checked_count = len(e.downstream)
        self._commit(versions, self.revisions)

    def finish_revision(self, event):
        versions = deepcopy(self._experiences)
        if event.event_id in {r.event_id for r in self.revisions}:
            raise ValueError("duplicate revision")
        parent = versions[event.parent_id]
        if parent.status != "active":
            raise ValueError("parent is not active")
        if event.accepted:
            child = deepcopy(event.candidate)
            if (
                child is None
                or child.parent_id != parent.experience_id
                or child.experience_id in versions
            ):
                raise ValueError("invalid revision child")
            child.source_task_id = parent.source_task_id
            child.status = "active"
            child.created_by_event_id = event.event_id
            parent.status = "retired"
            parent.retired_by_event_id = event.event_id
            versions[child.experience_id] = child
            event = deepcopy(event)
            event.candidate = deepcopy(child)
        self._commit(versions, [*self.revisions, event])
        return deepcopy(event)

    def reestimate(self, parameters):
        versions = deepcopy(self._experiences)
        for e in versions.values():
            rebuild_quality(e, parameters)
        previous = self.parameter_snapshot
        incoming = parameters.to_dict() if parameters else None
        if previous != incoming:
            self.parameter_events.append(
                {
                    "event": "reestimate",
                    "sequence": len(self.parameter_events) + 1,
                    "previous": previous,
                    "parameters": incoming,
                    "parameter_digest": parameters.fingerprint if parameters else None,
                }
            )
        self.parameter_snapshot = incoming
        try:
            self._commit(versions, self.revisions)
        except Exception:
            self.parameter_snapshot = previous
            if previous != incoming:
                self.parameter_events.pop()
            raise
        self.parameters = parameters


class TrajectoryLog:
    def __init__(self, root_dir):
        self.path = Path(root_dir) / "trajectories.jsonl"

    def append(self, record):
        record = TrajectoryRecord.from_dict(record.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf8") as f:
            f.write(record.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def all(self):
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        yield TrajectoryRecord.from_json(line)

    def get(self, sample_id):
        return next((x for x in self.all() if x.sample_id == sample_id), None)
