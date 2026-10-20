"""Immutable experience versions, independent evidence and bounded evolution windows."""
from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from verge.calculator import (apply_beta_evidence, positive_experience_sources, calculate_experience_usage,
                              allocate_evolution_credit, EVOLUTION_PENALTY_LAMBDA)
from math import fsum, isclose
from verge.models import (CandidateExperience, Credit, EvidenceRecord, Experience, ExperienceCase,
    ExperienceUpdate, AtomicResult, EvolutionRecord, TrajectoryRecord, UsageRecord,
    ExperienceCategory, CategoryAssignment, CategoryDescription, OrganizationRecord)


class ExperienceMemory:
    SCHEMA = 8

    def __init__(self, root_dir: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(root_dir) / "experience_memory.json"
        self.read_only = read_only
        self._experiences: dict[str, Experience] = {}
        self._categories: dict[str, ExperienceCategory] = {}
        self.pending_organization: dict[str, int] = {}
        self.organization_history: list[OrganizationRecord] = []
        self.dirty_categories: dict[str, list[dict]] = {}
        self.organization_attempts: dict[str, str] = {}
        self.description_attempts: dict[str, str] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            expected = {"schema_version", "experiences", "categories", "pending_organization",
                        "organization_history", "dirty_categories", "organization_attempts", "description_attempts"}
            if set(data) != expected or data["schema_version"] != self.SCHEMA:
                raise ValueError("unsupported experience memory schema")
            for value in data["experiences"]:
                e = Experience.from_dict(value)
                if e.experience_id in self._experiences:
                    raise ValueError("duplicate experience ID")
                self._experiences[e.experience_id] = e
            for value in data["categories"]:
                c = ExperienceCategory.from_dict(value)
                if c.category_id in self._categories:
                    raise ValueError("duplicate category ID")
                self._categories[c.category_id] = c
            self.pending_organization = data["pending_organization"]
            self.organization_history = [OrganizationRecord.from_dict(v) for v in data["organization_history"]]
            self.dirty_categories = data["dirty_categories"]
            self.organization_attempts = data["organization_attempts"]
            self.description_attempts = data["description_attempts"]
            self._validate()
        elif read_only:
            raise FileNotFoundError(self.path)
        else:
            self._save()

    def _validate(self):
        active = {e.experience_id: e for e in self._experiences.values() if e.status == "active"}
        if any(k not in active or v != 1 for k, v in self.pending_organization.items()):
            raise ValueError("invalid classification queue")
        if set(self._categories) != {e.category_id for e in active.values() if e.category_id}:
            raise ValueError("categories must match active membership")
        if set(self.dirty_categories) - set(self._categories):
            raise ValueError("unknown dirty category")
        for e in self._experiences.values():
            if abs(e.alpha - 1 - e.positive_credit) > 1e-8 or abs(e.beta - 1 - e.negative_credit - e.penalty_credit) > 1e-8:
                raise ValueError("experience credit ledger disagrees")
            for case in e.downstream_cases:
                evidence = [v for v in e.evidence_history
                            if v.evidence_type == "downstream_credit" and v.sample_id == case.sample_id]
                if ((case.content, case.trigger) != (e.content, e.trigger) or len(evidence) != 1
                        or (evidence[0].positive_credit, evidence[0].negative_credit) != (
                            case.credit.positive_credit, case.credit.negative_credit)):
                    raise ValueError("case disagrees with version or actual downstream credit")
            for event in e.update_history:
                if event.parent_id != e.experience_id or any(k not in self._experiences for k in event.child_ids):
                    raise ValueError("invalid evolution linkage")
                prefix = [c for c in e.downstream_cases]
                try:
                    end = next(i + 1 for i, c in enumerate(prefix) if c.sample_id == event.sample_id)
                except StopIteration as exc:
                    raise ValueError("evolution trigger missing from cases") from exc
                window = prefix[max(0, end-5):end]
                if [c.sample_id for c in window] != event.case_ids or not isclose(
                        event.window_credit_amount, fsum(c.credit.amount for c in window), abs_tol=1e-9):
                    raise ValueError("evolution window credit disagrees")
                if event.child_ids:
                    if e.retired_by_event_id != event.event_id:
                        raise ValueError("replacement must retire its parent")
                    _, penalty, allocations = allocate_evolution_credit(e, event.event_id, len(event.child_ids))
                    if not isclose(event.penalty_credit, penalty, abs_tol=1e-9):
                        raise ValueError("evolution penalty disagrees")
                    for child_id, allocation in zip(event.child_ids, allocations):
                        child = self._experiences[child_id]
                        if child.inherited_credit != allocation:
                            raise ValueError("child inherited credit disagrees with parent allocation")
            if e.origin_kind == "evolution" and (not e.created_by_event_id or len(e.origin_refs) != 1 or e.origin_refs[0] not in self._experiences):
                raise ValueError("evolution origin missing")
            if e.origin_kind == "evolution":
                parent = self._experiences[e.origin_refs[0]]
                births = [u for u in parent.update_history if u.event_id == e.created_by_event_id]
                if len(births) != 1 or e.experience_id not in births[0].child_ids:
                    raise ValueError("evolution origin disagrees with parent event")
            if e.status == "retired":
                ends = [u for u in e.update_history if u.event_id == e.retired_by_event_id]
                if len(ends) != 1 or not ends[0].child_ids:
                    raise ValueError("retired version requires a completed replacement")

    def _ensure_writable(self):
        if self.read_only:
            raise ValueError("experience memory is read-only")

    def _save(self):
        self._ensure_writable()
        self._validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": self.SCHEMA, "experiences": [x.to_dict() for x in self._experiences.values()],
                   "categories": [x.to_dict() for x in self._categories.values()],
                   "pending_organization": self.pending_organization,
                   "organization_history": [x.to_dict() for x in self.organization_history],
                   "dirty_categories": self.dirty_categories, "organization_attempts": self.organization_attempts,
                   "description_attempts": self.description_attempts}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def all(self, *, active_only: bool = False):
        return [Experience.from_dict(x.to_dict()) for x in self._experiences.values()
                if not active_only or x.status == "active"]

    def _active(self, eid):
        e = self._experiences[eid]
        if e.status != "active":
            raise ValueError("cannot apply feedback to retired version")
        return e

    def get(self, eid):
        return Experience.from_dict(self._experiences[eid].to_dict())

    def categories(self):
        return [ExperienceCategory.from_dict(c.to_dict()) for c in self._categories.values()]

    def apply_downstream_credits(self, sample_id: str, credits: dict[str, Credit], *, usage=None):
        self._ensure_writable()
        updates = {}
        for eid, credit in credits.items():
            current = self._active(eid)
            if any(x.sample_id == sample_id and x.evidence_type == "downstream_credit" for x in current.evidence_history):
                raise ValueError("duplicate downstream credit")
            if credit.amount:
                updates[eid] = apply_beta_evidence(current, EvidenceRecord(sample_id, "downstream_credit",
                    credit.positive_credit, credit.negative_credit, text_version=1))
        for eid, event in (usage or {}).items():
            current = updates.get(eid, self._active(eid))
            if event.sample_id != sample_id or event.text_version != 1 or any(x.sample_id == sample_id for x in current.usage_history):
                raise ValueError("duplicate or mismatched usage")
            updates[eid] = replace(current, usage_history=[*current.usage_history, event])
        self._experiences.update(updates)
        if updates:
            self._save()

    def add(self, candidate: CandidateExperience, case: ExperienceCase):
        self._ensure_writable()
        if candidate.experience_id is not None or (case.source_id, case.content, case.trigger) != (candidate.candidate_id, candidate.content, candidate.trigger):
            raise ValueError("origin must match actually applied candidate")
        if any(candidate.candidate_id in e.origin_refs for e in self._experiences.values()):
            raise ValueError("candidate already admitted")
        if candidate.candidate_id not in positive_experience_sources(case.rubric, case.judgment):
            raise ValueError("origin requires net-positive candidate credit")
        eid = "exp_" + uuid4().hex
        e = Experience(eid, candidate.content, candidate.trigger, 1, 1, [case.sample_id], [], [case],
                       origin_refs=[candidate.candidate_id])
        e = apply_beta_evidence(e, EvidenceRecord(case.sample_id, "application_credit", case.credit.positive_credit,
                    case.credit.negative_credit, candidate.candidate_id))
        e.usage_history = [calculate_experience_usage(case.rubric, case.judgment, case.sample_id,
                {source: 1 for c in case.rubric.criteria for source in c.source_experiences})[candidate.candidate_id]]
        self._experiences[eid] = e
        self.pending_organization[eid] = 1
        self._save()
        return replace(candidate, experience_id=eid)

    def collect_case(self, eid, case):
        self._ensure_writable()
        e = self._active(eid)
        if (case.source_id, case.content, case.trigger, case.text_version) != (eid, e.content, e.trigger, 1):
            raise ValueError("case must refer to actually used immutable version")
        if any(c.sample_id == case.sample_id for c in e.cases):
            return False
        evidence = [v for v in e.evidence_history
                    if v.sample_id == case.sample_id and v.evidence_type == "downstream_credit"]
        if len(evidence) != 1 or (evidence[0].positive_credit, evidence[0].negative_credit) != (
                case.credit.positive_credit, case.credit.negative_credit):
            raise ValueError("case requires matching actual downstream credit")
        self._experiences[eid] = replace(e, cases=[*e.cases, case], source_trajectories=[*e.source_trajectories, case.sample_id])
        self._save()
        return True

    def evolve(self, eid, update: ExperienceUpdate | None, atomic: AtomicResult | None, *, error=None):
        self._ensure_writable()
        e = self._active(eid)
        if not e.update_due:
            raise ValueError("no pending evolution")
        if error is None and (update is None or (update.action == "revise" and atomic is None)):
            raise ValueError("incomplete evolution output")
        changed = error is None and update.action == "revise" and not (
                len(atomic.experiences) == 1 and (atomic.experiences[0].content, atomic.experiences[0].trigger) == (e.content, e.trigger))
        event_id = "evolution_" + uuid4().hex
        children = []
        window_amount = fsum(c.credit.amount for c in e.working_cases)
        penalty = 0.0
        if changed:
            window_amount, penalty, allocations = allocate_evolution_credit(e, event_id, len(atomic.experiences))
            for method, allocation in zip(atomic.experiences, allocations):
                children.append(Experience("exp_" + uuid4().hex, method.content, method.trigger,
                       1 + allocation.positive_credit,
                       1 + allocation.negative_credit + allocation.inherited_penalty + allocation.evolution_penalty,
                       [], [], origin_kind="evolution", origin_refs=[eid], created_by_event_id=event_id,
                       inherited_credit=allocation))
        event = EvolutionRecord(event_id, e.working_cases[-1].sample_id, eid,
                  [x.experience_id for x in children], [c.sample_id for c in e.working_cases],
                  "error" if error else "split" if len(children)>1 else "replace" if children else "keep",
                  update.draft if update else None, atomic, error, window_amount,
                  EVOLUTION_PENALTY_LAMBDA, penalty)
        seq = len(e.downstream_cases)
        parent = replace(e, update_history=[*e.update_history, event], last_attempted_feedback_seq=seq,
                  last_consumed_feedback_seq=seq if not error else e.last_consumed_feedback_seq,
                  status="retired" if changed else "active", category_id=None if changed else e.category_id,
                  retired_by_event_id=event_id if changed else None)
        if changed:
            self.pending_organization.pop(eid, None)
            if e.category_id:
                self._membership_change(e.category_id, e, "removed")
        self._experiences[eid] = parent
        for child in children:
            self._experiences[child.experience_id] = child
            self.pending_organization[child.experience_id] = 1
        self._delete_empty()
        self._save()
        return event

    def _membership_change(self, cid, e, action):
        self.dirty_categories.setdefault(cid, []).append({"action": action, "experience_id": e.experience_id,
                                                        "content": e.content, "trigger": e.trigger})

    def _delete_empty(self):
        occupied = {e.category_id for e in self._experiences.values() if e.status == "active" and e.category_id}
        for cid in set(self._categories) - occupied:
            del self._categories[cid]
            self.dirty_categories.pop(cid, None)
            self.description_attempts.pop(cid, None)

    def organization_due(self, batch_size, *, flush=False):
        return bool(self.pending_organization) and (flush or len(self.pending_organization) >= batch_size)

    def assign(self, sample_id, eid, plan: CategoryAssignment | None, index_version, *, error=None,
               retrieval_trace=None, retrieved_categories=None):
        self._ensure_writable()
        e = self._active(eid)
        cid = e.category_id
        if error is None:
            if plan is None:
                raise ValueError("missing assignment")
            if plan.action == "assign":
                if plan.category_id not in self._categories:
                    raise ValueError("unknown category")
                cid = plan.category_id
            elif plan.action == "create":
                cid = "cat_" + uuid4().hex
                self._categories[cid] = ExperienceCategory(cid, plan.name, plan.description)
            else:
                cid = None
            if cid != e.category_id:
                if e.category_id:
                    self._membership_change(e.category_id, e, "removed")
                if cid:
                    self._membership_change(cid, e, "added")
                self._experiences[eid] = replace(e, category_id=cid)
            self.pending_organization.pop(eid, None)
        self.organization_attempts[eid] = index_version
        event = OrganizationRecord(sample_id, eid, cid, plan.action if plan else "assign_error", index_version, error,
                                   retrieval_trace=list(retrieval_trace or []),
                                   retrieved_categories=list(retrieved_categories or []))
        self.organization_history.append(event)
        self._delete_empty()
        self._save()
        return event

    def dirty_version(self, cid):
        return hashlib.sha256(json.dumps(self.dirty_categories[cid], sort_keys=True).encode()).hexdigest()

    def describe(self, sample_id, cid, description: CategoryDescription | None, *, error=None):
        self._ensure_writable()
        old = self._categories[cid]
        version = self.dirty_version(cid)
        if error is None:
            if description is None:
                raise ValueError("missing category description")
            self._categories[cid] = ExperienceCategory(cid, description.name, description.description)
            del self.dirty_categories[cid]
        event = OrganizationRecord(sample_id, None, cid, "describe", version, error,
                    old.name, old.description, description.name if description else old.name,
                    description.description if description else old.description)
        self.description_attempts[cid] = version
        self.organization_history.append(event)
        self._save()
        return event


class TrajectoryLog:
    """按处理顺序追加完整 TrajectoryRecord；不去重、覆盖或回滚。"""

    def __init__(self, root_dir: str | Path) -> None:
        self.path = Path(root_dir) / "trajectories.jsonl"

    def append(self, record: TrajectoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(record.to_json() + "\n")

    def all(self) -> Iterator[TrajectoryRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield TrajectoryRecord.from_json(line)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"invalid trajectory at line {number}") from exc

    def get(self, sample_id: str) -> TrajectoryRecord | None:
        """用于 inspect-sample；同一 ID 多次写入时返回最后一条，不做去重。"""
        result = None
        for record in self.all():
            if record.sample_id == sample_id:
                result = record
        return result
