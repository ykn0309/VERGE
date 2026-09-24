"""各角色独立的输入投影；隔离 preference、provenance 和 Fusion 数值。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from verge.models import CandidateExperience, Criterion, Experience, RevisionAttempt, RevisionOperation, Rubric, RubricJudgment


PROMPT_VERSION = "v3"

# 各角色只接受其投影后的 payload；不把其他阶段/内部证据混入上下文。
_ROLE_INSTRUCTIONS = {
    "selector": "按 task 选择真正适用且能帮助生成 rubric 的经验；检查 trigger、可执行性和互补性，不同时选冲突经验。适用性相近时参考 confidence 与 evidence_amount，不为凑数选无关经验。返回 {\"selected_experience_ids\":[候选 experience_id]}；最多 max_selected 条，不得虚构 ID；无适用经验返回空数组。",
    "initial_generator": "仅根据 task 与 selected_experiences 生成 R0，每条标准只判断一个问题、可用是/否评价回答，不引用特定响应。返回 {\"rubric\":{\"rubric_id\":唯一字符串,\"base_rubric_id\":null,\"criteria\":[{\"criterion_id\":唯一字符串,\"question\":可独立判定的问题,\"importance_score\":1到10的整数,\"source_experiences\":[所选经验ID]}]}}。生成4到8条标准；分数代表对 task 的重要性，不能预估区分当前回答的能力。仅引用实质影响标准的经验，未使用经验时 source_experiences 为空。",
    "judge": "只回答输入 criteria 数组中每条标准对 chosen/rejected 是否满足；返回 {\"criterion_judgments\":[{\"criterion_id\":criteria中的对应ID,\"chosen\":true或false,\"rejected\":true或false}]}。只对列出的 ID 各返回一次，不得返回其他 ID；只有一条 criteria 时只返回一条判断。不要计算权重、margin 或 success。",
    "critic": "分析 initial_rubric 对当前偏好对失败的原因，必要时参考 failed_attempt；返回 {\"feedback\":非空的具体修订建议}，不要生成新的 rubric。",
    "revision_generator": "从 initial_rubric（不是失败尝试）提出最小原子操作；返回 {\"operations\":[{\"operation_id\":唯一字符串,\"operation_type\":\"ADD\"或\"DELETE\"或\"MODIFY\",\"target_criterion_id\":已有ID或ADD时null,\"new_or_updated_criterion\":ADD/MODIFY时的完整criterion或DELETE时null}]}。新标准包含 criterion_id、question、importance_score(1-10)、source_experiences:[]；执行后保持4到8条。不要输出 before_criterion，旧标准由执行器自行确定。",
    "reflector": "只从 successful_attempt 中抽象可迁移到类似任务的 rubric 生成方法，不复述当前回答或偏好；返回 {\"candidates\":[{\"candidate_id\":唯一字符串,\"content\":单一、具体、可复用建议,\"trigger\":清晰的适用条件}]}，0到3条。不要输出 credit、operation 绑定或数值。",
    "revision_credit_assigner": "只把 candidates 与真正支撑它的 operations 绑定；返回 {\"bindings\":{候选candidate_id:[operation_id]}}，每个候选必须有键，无依据时使用空数组。不要计算 credit；无绑定是合法结果。",
    "admission_checker": "判断 candidate 是否有绑定操作的修改语义支撑，且是单一、可迁移的 rubric 生成方法、trigger 清楚、无当前回答细节或偏好泄漏；否则 FAIL。只返回 {\"decision\":\"PASS\"或\"FAIL\"}，不得改写候选、绑定、信用或其他数值。",
    "fusion_judge": "只比较 candidate 与 existing 的 content、trigger：生成原则、指导动作及 trigger 适用范围都实质相同才是语义等价；冲突不等价。返回 {\"match_index\":existing中匹配项的从0开始整数下标，或null}。不要使用任何数值/来源/偏好信息。"
}


def llm_messages(role: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """OpenAI-compatible chat 消息；各角色使用独立的输出约束。"""
    if role not in _ROLE_INSTRUCTIONS:
        raise ValueError(f"unknown LLM role: {role}")
    return [
        {"role": "system", "content": f"你是 VERGE 的结构化输出模块，当前角色：{role}。仅输出一个有效 JSON 对象，不加 Markdown 或额外文本。" + _ROLE_INSTRUCTIONS[role]},
        {"role": "user", "content": json.dumps(dict(payload), ensure_ascii=False, allow_nan=False)},
    ]


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


def judge_payload(pending: Sequence[Criterion], chosen: str, rejected: str) -> dict[str, Any]:
    # 修订时只传需要新判断的 criterion，避免模型把复用的旧 ID 也返回。
    return {"criteria": [{"criterion_id": item.criterion_id, "question": item.question}
                         for item in pending], "chosen": chosen, "rejected": rejected}


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
