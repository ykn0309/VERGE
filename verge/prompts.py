"""各角色独立的输入投影；隔离 preference、provenance 和 Fusion 数值。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from verge.models import (CandidateExperience, Criterion, Experience, InvalidRevisionAttempt,
                          RevisionAttempt, RevisionOperation, Rubric, RubricJudgment)


PROMPT_VERSION = "v10-critic-atomic"

# 各角色只接受其投影后的 payload；不把其他阶段/内部证据混入上下文。
_ROLE_INSTRUCTIONS = {
    "selector": (
        '仅根据当前 task 逐条筛选候选经验：先检查 trigger 的全部条件是否能由 task 本身核验且成立（如题型、显式要求）；'
        '依赖回答内容、已生成 rubric、Judge 结果等事后信息的条件不可当作匹配，也不能只凭关键词相似选择。'
        '再判断经验能否为此任务新增具体、可执行的判断问题或实质改变判断方式；若只重复 task 已足以推出的标准且无其他增益则不选。排除冲突/重复经验；'
        '仅在适用性与帮助相近时参考 confidence 和 evidence_amount，高信度不能覆盖不匹配。不为凑数选经验，可选零条。'
        '所有 ID 均使用输入中的短数字编号（字符串，如 "1"、"2"），不得拼接或缩写；'
        '返回 {"selected_experience_ids":[候选 experience_id]}；最多 max_selected 条，不得虚构 ID；'
        '无适用经验返回空数组。'
    ),
    "initial_generator": (
        '仅根据 task 与 selected_experiences 生成 R0：先根据 task 的显式要求拟定标准，再考虑适用经验能否真正新增或改变判断问题。'
        '每条标准只判断一个问题、可用是/否评价回答，不引用特定响应。'
        'rubric_id 固定为 "1"；标准 criterion_id 依次使用 "1"、"2"、…，不得生成复杂 ID。'
        '返回 {"rubric":{"rubric_id":"1","base_rubric_id":null,'
        '"criteria":[{"criterion_id":标准数字编号,"question":可独立判定的问题,'
        '"importance_score":1到10的整数,"source_experiences":[所选经验ID]}]}}。'
        '生成2到8条标准；分数代表对 task 的重要性，不能预估区分当前回答的能力。'
        "若仅凭 task 就足以提出该标准（例如 task 要求直接回答时的'是否直接回答'），即使经验也有类似建议，source_experiences 仍为 []；"
        '仅在经验实质新增或改变标准时引用其 ID，逐一判断多个来源。即使一条经验因帮助其他标准而被选中，也不能把它挂在仅凭 task 就能提出的标准上；未使用的经验不得挂名，'
        '也不要一律清空真正有用的来源。'
    ),
    "judge": (
        '只回答输入 criteria 数组中每条标准对 chosen/rejected 是否满足；'
        '返回 {"criterion_judgments":[{"criterion_id":criteria中的对应ID,'
        '"chosen":true或false,"rejected":true或false}]}。'
        '只对列出的数字编号各返回一次，严格复制，不得返回其他 ID；只有一条 criteria 时只返回一条判断。'
        '不要计算权重、margin 或 success。'
    ),
    "critic": (
        '目标是让给定偏好标签中的 chosen 得分高于 rejected。根据 task、两份回答、R0 每条标准的 Judge 布尔判断、'
        '权重与 margin，在私有 analysis 中具体解释当前 R0 失败的原因；允许引用当前回答内容，'
        '对可疑 Judge 判断或偏好标签可说明不确定性，不要捏造 chosen 优势。'
        '第二次还须对照第一次的 analysis、已执行的操作、R1/Judge（如存在）和校验失败摘要，'
        '解释第一次哪里没奏效；第一次操作不会自动继承，须针对 R0 重新提出可独立执行的完整操作集，'
        '不能依赖 R1 的新标准。若第一次仅有无效操作，不得虚构 R1 或 Judge。'
        'operations 中每条标准必须仅从 task 自身要求推导，适用于未来回答；'
        '不要把 chosen 或 rejected 的特有事实/措辞变成通用标准，除非 task 本身明确要求。'
        'operation_id 每次从 "1" 起重新编号（可与上次操作同号）；ADD 的新 criterion_id 大于 R0 最大编号，'
        'MODIFY 沿用目标编号；只用输入中的短数字编号，不缩写。'
        '返回 {"analysis":非空中文诊断,"operations":[{"operation_id":数字编号,'
        '"operation_type":"ADD"或"DELETE"或"MODIFY","target_criterion_id":已有ID或ADD时null,'
        '"new_or_updated_criterion":ADD/MODIFY时完整criterion或DELETE时null}]}。'
        '新标准包含 criterion_id、question、importance_score(1-10)、source_experiences:[]；执行后须保持2到8条。'
        '不要输出 before_criterion，旧标准由执行器自行确定。'
    ),
    "reflector": (
        '只从 successful_attempt 抽象可迁移的 rubric 生成原则(content)，不复述当前回答或偏好。'
        'trigger 只能描述未来筛选时仅凭当前 task 可核验的题型、显式要求或上下文；'
        "不要写'初始标准未区分回答'、'Judge 判断错误'、'生成的回答出现幻觉'等依赖回答、rubric 或判断结果的事后条件，也不要为凑数编造'所有任务'式空泛条件。"
        '无法写出明确的事前适用条件时不输出该候选。'
        'candidate_id 从 "1" 起依次编号。'
        '返回 {"candidates":[{"candidate_id":数字编号,"content":单一、具体、可复用建议,'
        '"trigger":清晰的 task 可检验条件}]}，0到3条。不要输出 credit、operation 绑定或数值。'
    ),
    "revision_credit_assigner": (
        '只把 candidates 与真正支撑它的 operations 绑定；'
        '候选与操作 ID 仅使用输入中的数字编号。'
        '返回 {"bindings":{候选candidate_id:[operation_id]}}，每个候选必须有键，无依据时使用空数组。不要计算 credit；'
        '无绑定是合法结果。'
    ),
    "admission_checker": (
        '判断 candidate 是否有绑定操作的修改语义支撑，且是单一、可迁移的 rubric 生成方法、无当前回答细节或偏好泄漏。'
        'trigger 必须清楚、非空泛且所有条件仅凭未来当前 task（题型、显式要求、任务上下文）即可核验；'
        '只要有一项依赖未来回答、初始/修订 rubric、Judge 结果、critic 反馈或偏好标签，即使混有 task 条件也 FAIL。'
        '当前成功修订只能证明 content 的依据，不能使事后条件成为未来任务可检验的 trigger。无合法条件则 FAIL，不改写 trigger。'
        '返回 {"decision":"PASS"或"FAIL","reason":用中文简要说明关键准入依据或违反的条件}。PASS 和 FAIL 都必须给出具体理由，'
        '指出相关 content 或 trigger；不得改写候选、绑定、信用或其他数值。'
    ),
    "fusion_judge": (
        '只比较 candidate 与 existing 的 content、trigger：生成原则、指导动作及 trigger 适用范围都实质相同才是语义等价；'
        '冲突不等价。返回 {"match_index":existing中匹配项的从0开始整数下标，或null}。不要使用任何数值/来源/偏好信息。'
    )
}


def llm_messages(role: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """OpenAI-compatible chat 消息；各角色使用独立的输出约束。"""
    if role not in _ROLE_INSTRUCTIONS:
        raise ValueError(f"unknown LLM role: {role}")
    return [
        {"role": "system", "content": (
            f"你是 VERGE 的结构化输出模块，当前角色：{role}。仅输出一个有效 JSON 对象，不加 Markdown 或额外文本。"
            "所有供人阅读的生成文本（标准问题、critic 私有分析、经验 content、trigger 等）必须以中文表达；"
            "英文输入需转述为中文，但可保留专有名词、代码、必要的原文引用。"
            "JSON 字段名、ID、操作枚举、PASS/FAIL 等机器协议值保持规定形式，不要翻译。"
            + _ROLE_INSTRUCTIONS[role]
        )},
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


def generator_payload(task: str, selected: Sequence[ExperienceView], *,
                      validation_feedback: str | None = None) -> dict[str, Any]:
    payload = {"task": task, "selected_experiences": [item.to_dict() for item in selected]}
    if validation_feedback is not None:
        payload["validation_feedback"] = validation_feedback
    return payload


def judge_payload(pending: Sequence[Criterion], chosen: str, rejected: str, *,
                  validation_feedback: str | None = None) -> dict[str, Any]:
    # 修订时只传需要新判断的 criterion，避免模型把复用的旧 ID 也返回。
    payload = {"criteria": [{"criterion_id": item.criterion_id, "question": item.question}
                            for item in pending], "chosen": chosen, "rejected": rejected}
    if validation_feedback is not None:
        payload["validation_feedback"] = validation_feedback
    return payload


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
    return {"attempt_number": attempt.attempt_number,
            "operations": [_operation_summary(item) for item in attempt.operations],
            "rubric": _rubric_semantics(attempt.rubric), "judgment": attempt.judgment.to_dict()}


def critic_payload(task: str, chosen: str, rejected: str, initial: Rubric,
                   initial_judgment: RubricJudgment, attempt_number: int, *,
                   previous_attempt: RevisionAttempt | None = None,
                   previous_invalid_attempt: InvalidRevisionAttempt | None = None,
                   validation_feedback: str | None = None) -> dict[str, Any]:
    payload = {"task": task, "chosen": chosen, "rejected": rejected,
               "initial_rubric": _rubric_semantics(initial),
               "initial_judgment": initial_judgment.to_dict(),
               "attempt_number": attempt_number}
    if attempt_number == 2:
        payload["previous_attempt"] = ({**_attempt_semantics(previous_attempt),
                                        "analysis": previous_attempt.analysis,
                                        "validation_failures": list(previous_attempt.validation_failures)}
                                       if previous_attempt else None)
        payload["previous_invalid_attempt"] = (
            {"attempt_number": previous_invalid_attempt.attempt_number,
             "analysis": previous_invalid_attempt.analysis,
             "validation_failures": list(previous_invalid_attempt.validation_failures)}
            if previous_invalid_attempt else None)
    if validation_feedback is not None:
        payload["validation_feedback"] = validation_feedback
    return payload


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


_ID_FIELDS = {"criterion_id": "criterion", "target_criterion_id": "criterion",
              "rubric_id": "rubric", "base_rubric_id": "rubric",
              "experience_id": "experience", "operation_id": "operation",
              "candidate_id": "candidate"}
_ID_LISTS = {"source_experiences": "experience", "selected_experience_ids": "experience",
             "linked_operation_ids": "operation"}
_ID_DICTS = {"normalized_weights": "criterion", "operation_contributions": "operation"}


class NumberedIDCodec:
    """单次模型调用的局部 ID 表；真实内部 ID 从不进入结构化提示词。

    不跨调用缓存：每次只根据本次白名单输入建立编号，再把模型输出还原到内部 ID。
    不改写 task、chosen/rejected 或任意自然语言文本。
    """

    def __init__(self, role: str, payload: Mapping[str, Any]) -> None:
        self.role = role
        self.attempt_number = payload.get("attempt_number", 1)
        self.base_criteria = {item["criterion_id"] for item in
                              payload.get("initial_rubric", {}).get("criteria", [])
                              if isinstance(item, dict) and isinstance(item.get("criterion_id"), str)}
        self.forward: dict[str, dict[str, str]] = {name: {} for name in
                                                     ("criterion", "rubric", "experience", "operation", "candidate")}
        self.reverse: dict[str, dict[str, str]] = {name: {} for name in self.forward}

    def _encode(self, name: str, value: str) -> str:
        known = self.forward[name]
        if value not in known:
            number = str(len(known) + 1)
            known[value] = number
            self.reverse[name][number] = value
        return known[value]

    def project(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """递归编号字段及字典键，保持相同命名空间的重复引用一致。"""
        def visit(value: Any) -> Any:
            if isinstance(value, dict):
                result: dict[str, Any] = {}
                for key, item in value.items():
                    if key in _ID_FIELDS and isinstance(item, str):
                        result[key] = self._encode(_ID_FIELDS[key], item)
                    elif key in _ID_LISTS and isinstance(item, list):
                        result[key] = [self._encode(_ID_LISTS[key], entry)
                                       if isinstance(entry, str) else entry for entry in item]
                    elif key in _ID_DICTS and isinstance(item, dict):
                        result[key] = {self._encode(_ID_DICTS[key], entry): visit(entry_value)
                                       for entry, entry_value in item.items()}
                    elif key == "bindings" and isinstance(item, dict):
                        result[key] = {self._encode("candidate", entry):
                                       [self._encode("operation", op) for op in operations]
                                       for entry, operations in item.items()}
                    else:
                        result[key] = visit(item)
                return result
            if isinstance(value, list):
                return [visit(item) for item in value]
            return value
        return visit(dict(payload))

    def _existing(self, name: str, value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError(f"{self.role}: {name} must be a numbered ID")
        if value not in self.reverse[name]:
            raise ValueError(f"{self.role}: unknown numbered {name}")
        return self.reverse[name][value]

    def _new(self, name: str, value: Any, prefix: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError(f"{self.role}: new {name} must be a numbered ID")
        if value in self.reverse[name] and not (
                self.role == "critic" and (
                    name == "operation" or
                    (name == "criterion" and self.attempt_number == 2
                     and self.reverse[name][value] not in self.base_criteria))):
            raise ValueError(f"{self.role}: new {name} must use an unused number")
        return f"{prefix}{value}"

    def restore(self, response: Mapping[str, Any]) -> dict[str, Any]:
        """只反解角色协议字段；错误引用直接拒绝，不猜测或按位置悄悄匹配。"""
        result = dict(response)
        if self.role == "selector" and isinstance(result.get("selected_experience_ids"), list):
            result["selected_experience_ids"] = [self._existing("experience", value)
                                                 for value in result["selected_experience_ids"]]
        elif self.role == "initial_generator" and isinstance(result.get("rubric"), dict):
            rubric = dict(result["rubric"])
            rubric["rubric_id"] = self._new("rubric", rubric.get("rubric_id"), "r")
            if isinstance(rubric.get("criteria"), list):
                rubric["criteria"] = [self._restore_criterion(item, new=True)
                                      if isinstance(item, dict) else item for item in rubric["criteria"]]
            result["rubric"] = rubric
        elif self.role == "judge" and isinstance(result.get("criterion_judgments"), list):
            result["criterion_judgments"] = [
                {**item, "criterion_id": self._existing("criterion", item.get("criterion_id"))}
                if isinstance(item, dict) else item for item in result["criterion_judgments"]]
        elif self.role == "critic" and isinstance(result.get("operations"), list):
            operations = []
            for item in result["operations"]:
                if not isinstance(item, dict):
                    operations.append(item)
                    continue
                operation = dict(item)
                operation["operation_id"] = self._new("operation", item.get("operation_id"),
                                                     f"op{self.attempt_number}_")
                if item.get("target_criterion_id") is not None:
                    operation["target_criterion_id"] = self._existing(
                        "criterion", item["target_criterion_id"])
                if isinstance(item.get("new_or_updated_criterion"), dict):
                    operation["new_or_updated_criterion"] = self._restore_criterion(
                        item["new_or_updated_criterion"], new=item.get("operation_type") == "ADD")
                operations.append(operation)
            result["operations"] = operations
        elif self.role == "reflector" and isinstance(result.get("candidates"), list):
            result["candidates"] = [
                {**item, "candidate_id": self._new("candidate", item.get("candidate_id"), "candidate_")}
                if isinstance(item, dict) else item for item in result["candidates"]]
        elif self.role == "revision_credit_assigner" and isinstance(result.get("bindings"), dict):
            result["bindings"] = {
                self._existing("candidate", candidate):
                [self._existing("operation", operation) for operation in operations]
                if isinstance(operations, list) else operations
                for candidate, operations in result["bindings"].items()
            }
        return result

    def _restore_criterion(self, item: dict[str, Any], *, new: bool) -> dict[str, Any]:
        criterion = dict(item)
        value = item.get("criterion_id")
        criterion["criterion_id"] = (self._new("criterion", value,
                                                "c" if self.role == "initial_generator"
                                                else f"added{self.attempt_number}_")
                                     if new else self._existing("criterion", value))
        if isinstance(item.get("source_experiences"), list):
            criterion["source_experiences"] = [self._existing("experience", value)
                                               for value in item["source_experiences"]]
        return criterion
