"""各角色独立的输入投影；隔离 preference、provenance 和 Fusion 数值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from verge.models import CandidateExperience, Criterion, Experience, RevisionAttempt, RevisionOperation, Rubric, RubricJudgment


@dataclass(frozen=True)
class ExperienceView:
    """Selector/初始 Generator 可见的经验字段，不暴露来源轨迹或证据记录。"""

    experience_id: str
    content: str
    trigger: str
    confidence: float
    evidence_amount: float

    @classmethod
    def from_experience(cls, experience: Experience) -> ExperienceView:
        return cls(experience.experience_id, experience.content, experience.trigger,
                   experience.confidence, experience.evidence_amount)

    def to_dict(self) -> dict[str, Any]:
        return {"experience_id": self.experience_id, "content": self.content,
                "trigger": self.trigger, "confidence": self.confidence,
                "evidence_amount": self.evidence_amount}


def selector_payload(task: str, candidates: Sequence[ExperienceView], *,
                     max_selected: int, training: bool) -> dict[str, Any]:
    return {"task": task, "candidates": [item.to_dict() for item in candidates],
            "max_selected": max_selected, "training": training}


def generator_payload(task: str, selected: Sequence[ExperienceView]) -> dict[str, Any]:
    return {"task": task, "selected_experiences": [item.to_dict() for item in selected]}


def judge_payload(rubric: Rubric, pending: Sequence[Criterion],
                  chosen: str, rejected: str) -> dict[str, Any]:
    def criterion_fields(item: Criterion) -> dict[str, Any]:
        return {"criterion_id": item.criterion_id, "question": item.question,
                "importance_score": item.importance_score}

    return {"rubric": [criterion_fields(item) for item in rubric.criteria],
            "criteria": [criterion_fields(item) for item in pending],
            "chosen": chosen, "rejected": rejected}


def _criterion_fields(criterion: Criterion) -> dict[str, Any]:
    return {"criterion_id": criterion.criterion_id, "question": criterion.question,
            "importance_score": criterion.importance_score}


def _rubric_semantics(rubric: Rubric) -> dict[str, Any]:
    return {"rubric_id": rubric.rubric_id, "base_rubric_id": rubric.base_rubric_id,
            "criteria": [_criterion_fields(item) for item in rubric.criteria]}


def _operation_summary(operation: RevisionOperation) -> dict[str, Any]:
    return {"operation_id": operation.operation_id, "operation_type": operation.operation_type,
            "target_criterion_id": operation.target_criterion_id,
            "before": _criterion_fields(operation.before_criterion) if operation.before_criterion else None,
            "after": _criterion_fields(operation.new_or_updated_criterion)
                     if operation.new_or_updated_criterion else None}


def _attempt_semantics(attempt: RevisionAttempt) -> dict[str, Any]:
    return {"attempt_number": attempt.attempt_number, "feedback": attempt.feedback,
            "operations": [_operation_summary(item) for item in attempt.operations],
            "rubric": _rubric_semantics(attempt.rubric), "judgment": attempt.judgment.to_dict()}


def critic_payload(task: str, chosen: str, rejected: str, initial: Rubric,
                   initial_judgment: RubricJudgment,
                   failed_attempt: RevisionAttempt | None) -> dict[str, Any]:
    return {"task": task, "chosen": chosen, "rejected": rejected,
            "initial_rubric": _rubric_semantics(initial),
            "initial_judgment": initial_judgment.to_dict(),
            "failed_attempt": _attempt_semantics(failed_attempt) if failed_attempt else None}


def revision_payload(task: str, initial: Rubric, feedback: str,
                     attempt_number: int) -> dict[str, Any]:
    return {"task": task, "initial_rubric": _rubric_semantics(initial),
            "feedback": feedback, "attempt_number": attempt_number}


def reflection_payload(task: str, initial: Rubric, initial_judgment: RubricJudgment,
                       successful_attempt: RevisionAttempt) -> dict[str, Any]:
    return {"task": task, "initial_rubric": _rubric_semantics(initial),
            "initial_judgment": initial_judgment.to_dict(),
            "successful_attempt": _attempt_semantics(successful_attempt)}


def binding_payload(candidates: Sequence[CandidateExperience],
                    operations: Sequence[RevisionOperation]) -> dict[str, Any]:
    return {"candidates": [{"candidate_id": item.candidate_id, "content": item.content,
                            "trigger": item.trigger} for item in candidates],
            "operations": [_operation_summary(item) for item in operations]}


def admission_payload(task: str, candidate: CandidateExperience,
                      successful_attempt: RevisionAttempt,
                      operation_contributions: dict[str, float]) -> dict[str, Any]:
    return {"task": task, "candidate": candidate.to_dict(),
            "successful_attempt": _attempt_semantics(successful_attempt),
            "operation_contributions": dict(operation_contributions)}


def fusion_payload(content: str, trigger: str,
                   existing: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """只包含待比较的文本，不发送 ID、confidence、credit 或来源。"""
    return {"candidate": {"content": content, "trigger": trigger},
            "existing": [{"content": old_content, "trigger": old_trigger}
                         for old_content, old_trigger in existing]}
