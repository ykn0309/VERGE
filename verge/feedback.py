"""Task-deduplicated credit from complete, independent single-experience trials.

Source validation never enters this ledger. Model citation declarations are not
used. Each record is one task–experience pair, sharing the task baseline.
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
    experience_id: str
    baseline_correct: bool | None
    experience_correct: bool | None
    error: str | None = None

    def __post_init__(self):
        for field in ("record_id", "task_id", "protocol_id", "experience_id"):
            _require_id(getattr(self, field), field)
        values = (self.baseline_correct, self.experience_correct)
        if any(type(v) is not bool and v is not None for v in values):
            raise ValueError("outcomes must be Boolean or null")
        if self.error is not None:
            _require_id(self.error, "error")
        if any(v is None for v in values) != bool(self.error):
            raise ValueError(
                "unknown pair requires an error; valid pair cannot have one"
            )

    @property
    def credits(self) -> dict[str, float] | None:
        if self.error is not None:
            return None
        return {
            self.experience_id: float(
                int(self.experience_correct) - int(self.baseline_correct)
            )
        }

    @property
    def outcome(self):
        if self.error:
            return None
        if self.baseline_correct:
            return "both_correct" if self.experience_correct else "regressed"
        return "improved" if self.experience_correct else "both_wrong"


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
            for eid in [record.experience_id]:
                buckets.setdefault((group, eid), []).append(record)
        return {key: _statistics(records, {key[1]}) for key, records in buckets.items()}


def _group(assignments, object_id):
    try:
        group = assignments[object_id]
    except KeyError as exc:
        raise ValueError(f"missing group for {object_id}") from exc
    _require_id(group, "group_id")
    return group
