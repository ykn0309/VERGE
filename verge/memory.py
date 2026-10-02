"""本地 Experience Memory 和逐样本 JSONL trajectory 存储。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from verge.calculator import apply_beta_evidence
from verge.models import CandidateExperience, EvidenceRecord, Experience, TrajectoryRecord


class ExperienceMemory:
    """保存 canonical experience；调用方负责计算信用和决定 Fusion 匹配对象。

    每次有效更新立即写入单个 JSON 文件。不处理重复运行、跨文件事务或崩溃恢复。
    """

    def __init__(self, root_dir: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(root_dir) / "experience_memory.json"
        self.read_only = read_only
        self._experiences: dict[str, Experience] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or set(data) != {"experiences"} or not isinstance(data["experiences"], list):
                raise ValueError("invalid experience memory format")
            for value in data["experiences"]:
                experience = Experience.from_dict(value)
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
        payload = {"experiences": [item.to_dict() for item in self._experiences.values()]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                             encoding="utf-8")

    def all(self) -> list[Experience]:
        """返回独立副本；修改查询结果不会悄悄改变池内状态。"""
        return [Experience.from_dict(item.to_dict()) for item in self._experiences.values()]

    def get(self, experience_id: str) -> Experience:
        return Experience.from_dict(self._experiences[experience_id].to_dict())

    def apply_downstream_credits(
        self, sample_id: str, credits: dict[str, float], *, evidence_weight: float = 1.0,
    ) -> None:
        """R0 判断后立即写入 U_t(e)；零信用/零权重不产生 evidence。"""
        self._ensure_writable()
        updates: dict[str, Experience] = {}
        for experience_id, credit in credits.items():
            current = self._experiences[experience_id]
            evidence = EvidenceRecord(sample_id, "downstream_credit", credit, evidence_weight)
            updates[experience_id] = apply_beta_evidence(current, evidence)
        changed = any(updates[key] is not self._experiences[key] for key in updates)
        if changed:
            self._experiences.update(updates)
            self._save()

    def fuse(
        self, candidate: CandidateExperience, sample_id: str,
        matching_experience_id: str | None = None,
    ) -> CandidateExperience:
        """Admission PASS 后按外部 FusionJudge 的匹配结果 ADD 或 MERGE。"""
        self._ensure_writable()
        if (candidate.admission != "PASS" or candidate.credit is not None
                or candidate.linked_operation_ids or candidate.fusion is not None):
            raise ValueError("fusion requires an admitted, uncredited, unfused candidate")
        # 先验证候选和匹配 ID，避免非法调用改变内存后才失败。
        if matching_experience_id is None:
            experience_id = f"exp_{uuid4().hex}"
            while experience_id in self._experiences:
                experience_id = f"exp_{uuid4().hex}"
            prior = Experience(experience_id, candidate.content, candidate.trigger, 1.0, 1.0,
                               [sample_id], [])
            updated = prior
            decision = "ADD"
        else:
            prior = self._experiences[matching_experience_id]
            updated = Experience(prior.experience_id, prior.content, prior.trigger,
                                 prior.alpha, prior.beta,
                                 [*prior.source_trajectories, sample_id],
                                 prior.evidence_history)
            experience_id = prior.experience_id
            decision = "MERGE"
        result = CandidateExperience(candidate.candidate_id, candidate.content, candidate.trigger,
                                     list(candidate.linked_operation_ids), candidate.credit,
                                     candidate.admission, decision, experience_id,
                                     candidate.admission_reason)
        self._experiences[experience_id] = updated
        self._save()
        return result


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
