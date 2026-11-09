"""Task-deduplicated credit from complete, program-controlled subset outcomes.

Source validation never enters this ledger. Model citation declarations are not
used. A record is one task, not four independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable, Mapping


def _require_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


@dataclass(frozen=True)
class CreditFeedback:
    record_id: str
    task_id: str
    protocol_id: str
    experience_ids: tuple[str, ...]
    baseline_correct: bool | None
    actual_correct: bool | None
    singleton_correct: tuple[bool | None, ...] = ()
    error: str | None = None

    def __post_init__(self):
        for field in ("record_id", "task_id", "protocol_id"):
            _require_id(getattr(self, field), field)
        ids = tuple(self.experience_ids)
        if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
            raise ValueError("credit requires one or two distinct experience IDs")
        for eid in ids:
            _require_id(eid, "experience_id")
        object.__setattr__(self, "experience_ids", ids)
        singles = tuple(self.singleton_correct)
        if len(singles) != len(ids):
            raise ValueError("singleton outcomes must match the experience order")
        object.__setattr__(self, "singleton_correct", singles)
        values = (self.baseline_correct, self.actual_correct, *singles)
        if any(type(v) is not bool and v is not None for v in values):
            raise ValueError("outcomes must be Boolean or null")
        if self.error is not None:
            _require_id(self.error, "error")
        if any(v is None for v in values) and not self.error:
            raise ValueError("missing outcome requires an error")
        if len(ids) == 1 and singles[0] != self.actual_correct:
            raise ValueError("single experience outcome differs from actual")

    @property
    def credits(self) -> dict[str, float] | None:
        if self.error is not None:
            return None
        r0, full = int(self.baseline_correct), int(self.actual_correct)
        if len(self.experience_ids) == 1:
            return {self.experience_ids[0]: float(full - r0)}
        left, right = map(int, self.singleton_correct)
        return dict(
            zip(
                self.experience_ids,
                (
                    ((left - r0) + (full - right)) / 2,
                    ((right - r0) + (full - left)) / 2,
                ),
            )
        )

    @property
    def outcome(self):
        if self.error:
            return None
        if self.baseline_correct:
            return "both_correct" if self.actual_correct else "regressed"
        return "improved" if self.actual_correct else "both_wrong"


@dataclass(frozen=True)
class CreditStats:
    values: tuple[float, ...] = ()
    record_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()

    @property
    def tasks(self):
        return len(self.task_ids)

    @property
    def reward_sum(self):
        return fsum(self.values)

    @property
    def mean_reward(self):
        return self.reward_sum / self.tasks if self.tasks else None

    @property
    def improved(self):
        return sum(v > 0 for v in self.values)

    @property
    def regressed(self):
        return sum(v < 0 for v in self.values)


def _statistics(records, experience_ids):
    retained, values, seen = [], [], set()
    for record in records:
        credit = record.credits
        if credit is None or record.task_id in seen:
            continue
        relevant = experience_ids.intersection(credit)
        if not relevant:
            continue
        seen.add(record.task_id)
        retained.append(record)
        values.append(fsum(credit[eid] for eid in relevant))
    return CreditStats(
        tuple(values),
        tuple(r.record_id for r in retained),
        tuple(r.task_id for r in retained),
    )


class FeedbackLedger:
    def __init__(self, protocol_id: str, records: Iterable[CreditFeedback] = ()):
        _require_id(protocol_id, "protocol_id")
        self.protocol_id = protocol_id
        self._records = {}
        for record in records:
            self.append(record)

    @property
    def records(self):
        return tuple(self._records.values())

    def append(self, record: CreditFeedback):
        if record.protocol_id != self.protocol_id:
            raise ValueError("cannot mix execution protocols")
        if record.record_id in self._records:
            if self._records[record.record_id] != record:
                raise ValueError("conflicting feedback record")
            return False
        self._records[record.record_id] = record
        return True

    def for_experience(self, experience_id, *, task_ids=None):
        _require_id(experience_id, "experience_id")
        allowed = None if task_ids is None else frozenset(task_ids)
        return _statistics(
            (r for r in self.records if allowed is None or r.task_id in allowed),
            {experience_id},
        )

    def by_task_group(self, task_groups: Mapping[str, str]):
        buckets = {}
        for record in self.records:
            if record.credits is None:
                continue
            group = _group(task_groups, record.task_id)
            for eid in record.experience_ids:
                buckets.setdefault((group, eid), []).append(record)
        return {key: _statistics(records, {key[1]}) for key, records in buckets.items()}

    def by_group_pair(self, task_groups, experience_groups):
        """Analysis only: sum member credit once per task, not duplicated paths."""
        buckets, members = {}, {}
        for record in self.records:
            if record.credits is None:
                continue
            task_group = _group(task_groups, record.task_id)
            for eid in record.experience_ids:
                eg = _group(experience_groups, eid)
                members.setdefault(eg, set()).add(eid)
                bucket = buckets.setdefault((task_group, eg), {})
                bucket[record.record_id] = record
        return {
            key: _statistics(records.values(), members[key[1]])
            for key, records in buckets.items()
        }


def _group(assignments, object_id):
    try:
        group = assignments[object_id]
    except KeyError as exc:
        raise ValueError(f"missing group for {object_id}") from exc
    _require_id(group, "group_id")
    return group
