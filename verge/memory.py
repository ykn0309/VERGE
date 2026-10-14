"""本地 Experience Memory 和逐样本 JSONL trajectory 存储。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from verge.calculator import apply_beta_evidence, positive_experience_sources
from verge.models import (
    CandidateExperience,
    Credit,
    EvidenceRecord,
    Experience,
    ExperienceCase,
    ExperienceUpdate,
    TextUpdateRecord,
    TrajectoryRecord,
)


class ExperienceMemory:
    """保存经验、有限案例和文本版本；调用方计算信用并决定候选新增与文本更新。

    每次有效更新立即写入单个 JSON 文件。不处理重复运行、跨文件事务或崩溃恢复。
    """

    def __init__(self, root_dir: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(root_dir) / "experience_memory.json"
        self.read_only = read_only
        self._experiences: dict[str, Experience] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict)
                or set(data) != {"schema_version", "experiences"}
                or data.get("schema_version") != 5
                or not isinstance(data["experiences"], list)
            ):
                raise ValueError("invalid experience memory format")
            for value in data["experiences"]:
                experience = Experience.from_dict(value)
                if not experience.cases or experience.source_trajectories != [c.sample_id for c in experience.cases]:
                    raise ValueError("memory requires a recorded origin and bounded source cases")
                if experience.experience_id in self._experiences:
                    raise ValueError("duplicate experience ID in memory")
                self._experiences[experience.experience_id] = experience
        elif read_only:
            raise FileNotFoundError(self.path)
        else:
            self._save()

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise ValueError("experience memory is read-only")

    def _save(self) -> None:
        self._ensure_writable()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 5,
            "experiences": [item.to_dict() for item in self._experiences.values()],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def all(self) -> list[Experience]:
        """Return detached copies; there are no retired/split experiences."""
        return [Experience.from_dict(item.to_dict()) for item in self._experiences.values()]

    def _active(self, experience_id: str) -> Experience:
        return self._experiences[experience_id]

    def get(self, experience_id: str) -> Experience:
        return Experience.from_dict(self._experiences[experience_id].to_dict())

    def apply_downstream_credits(
        self, sample_id: str, credits: dict[str, Credit]
    ) -> None:
        self._ensure_writable()
        updates = {}
        for key, credit in credits.items():
            current = self._active(key)
            if credit.amount:
                updates[key] = apply_beta_evidence(
                    current,
                    EvidenceRecord(
                        sample_id,
                        "downstream_credit",
                        credit.positive_credit,
                        credit.negative_credit,
                        text_version=current.text_version,
                    ),
                )
        if updates:
            self._experiences.update(updates)
            self._save()

    def add(self, candidate: CandidateExperience, case: ExperienceCase) -> CandidateExperience:
        """New candidate only; no matching, MERGE, or text rewriting at admission."""
        self._ensure_writable()
        if candidate.experience_id is not None:
            raise ValueError("candidate must not already be admitted")
        if (case.source_id != candidate.candidate_id
                or (case.content, case.trigger) != (candidate.content, candidate.trigger)):
            raise ValueError("origin case must match actually applied candidate")
        if any(case.sample_id == e.source_trajectories[0] for e in self._experiences.values()
               if e.source_trajectories):
            raise ValueError("at most one candidate per sample")
        if case.source_id not in positive_experience_sources(case.rubric, case.judgment):
            raise ValueError("origin requires net-positive candidate credit")
        experience_id = "exp_" + uuid4().hex
        experience = Experience(
            experience_id, candidate.content, candidate.trigger, 1, 1,
            [case.sample_id], [], [case],
        )
        experience = apply_beta_evidence(experience, EvidenceRecord(
            case.sample_id, "application_credit", case.credit.positive_credit,
            case.credit.negative_credit, candidate.candidate_id, text_version=1,
        ))
        self._experiences[experience_id] = experience
        self._save()
        return replace(candidate, experience_id=experience_id)

    def collect_case(self, experience_id: str, case: ExperienceCase) -> bool:
        """Collect positive, reversed and neutral R0 cases; ten includes the origin."""
        self._ensure_writable()
        current = self._active(experience_id)
        if case.source_id != experience_id or (case.content, case.trigger, case.text_version) != (
            current.content, current.trigger, current.text_version,
        ):
            raise ValueError("case must refer to the actually used current text version")
        if current.text_frozen or any(c.sample_id == case.sample_id for c in current.cases):
            return False
        self._experiences[experience_id] = replace(
            current, cases=[*current.cases, case],
            source_trajectories=[*current.source_trajectories, case.sample_id],
        )
        self._save()
        return True

    def update_text(self, experience_id: str, content: str, trigger: str,
                    *, error: str | None = None) -> TextUpdateRecord:
        self._ensure_writable()
        current = self._pending_update(experience_id)
        changed = (current.content, current.trigger) != (content, trigger)
        event = TextUpdateRecord(
            current.cases[-1].sample_id, [c.sample_id for c in current.cases],
            current.text_version, current.text_version + int(changed),
            current.content, current.trigger, content, trigger, error,
        )
        updated = replace(current, content=content, trigger=trigger,
                          text_version=event.text_version,
                          update_history=[*current.update_history, event])
        self._experiences[experience_id] = updated
        self._save()
        return event

    def _pending_update(self, experience_id: str) -> Experience:
        current = self._active(experience_id)
        if not current.update_due:
            raise ValueError("no pending text update")
        return current

    def apply_update(self, experience_id: str, update: ExperienceUpdate) -> TextUpdateRecord:
        """Keep ID and accumulated credit; only revise content/trigger."""
        return self.update_text(experience_id, update.content, update.trigger)


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
