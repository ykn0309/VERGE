"""各角色独立的输入投影；隔离 preference、provenance 和 Fusion 数值。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from verge.models import (CandidateExperience, Criterion, Experience, InvalidRevisionAttempt,
                          RevisionAttempt, Rubric, RubricJudgment)


PROMPT_VERSION = "generator-dialogue-concise-revision-feedback"

# 专属身份和任务说明；Generator/Reflector 的对象编号由系统分配。
_ROLE_GUIDANCE = {
    "selector": """你是经验适用性筛选员。你的任务是：针对给定 task，筛选出可用于生成该 task 回答评价标准的经验""",
    "initial_generator": """你是评分标准设计者。你的任务是依据 task 和候选经验，设计用于评价回答的是/否判断标准。
先输出 analysis：结合候选经验分析给定 task 的一个高质量回答应该满足哪些要求，如何区分高质量和低质量的回答，以及不同评价维度的重要程度。
再依据分析输出 criteria：每条标准表达一个准确具体、可独立核验的是/否问题，“是”表示满足要求。importance_score 表达任务重要性，source_experiences 表达实际影响该标准的经验来源。""",
    "judge": """你是逐标准回答核验员。你的任务是结合 task 和评价标准，依据每条提供的标准和当前回答，独立判断这份回答是否满足要求。
每条标准对应一个 satisfied Boolean 判断，各项结果由这份回答的实际内容确定。""",
    "critic": """你是rubric质量诊断员。结合 task、偏好对、当前标准与逐项 Judge 判断，输出 analysis 和 improvement_advice。
analysis 是私有诊断，可以讨论当前评分标准、候选回答、Judge 错误及判断的不确定性。如果修订轮次为 2，需要分析上轮修订为何未奏效。
improvement_advice 是唯一交给rubric生成器的公开反馈：说明当前标准存在的问题及改进建议，严禁引用、复述或暗示与候选回答相关的任何信息。""",
    "reflector": """你是 rubric 生成经验提炼者。你的任务是思考为什么rubric在经过修订后，由judge不通过变成了judge通过， 从修订轨迹中提炼rubric生成经验，使rubric生成器在以后遇到类似任务时能够更有效地生成评分标准。提炼出的经验不能仅局限于当前 task 的场景，必须是更通用、更高层次的，也不能在经验中包含当前 task 或候选回答的任何细节。
content 说明如何设计评价标准；trigger 说明该经验的适用条件。
最多提炼一条最有迁移价值、证据清晰的设计机制；没有值得保留的方法时返回空数组。""",
    "admission_checker": """你是候选 rubric 经验的准入审核员。你的任务是根据 task、候选 content 和 trigger，审核供 rubric generator 首次生成标准使用的具体、可复用设计方法，并给出具体理由；实际有效性由后续下游任务验证。
content 说明如何设计评价标准；trigger 说明该经验的适用条件。
允许设计评价维度、安排标准重要性、排除不当要求；完整 rubric 和回答生成指南属于其他产物。任务类别有限、需要语义理解或机制具体，本身均不足以拒绝。
审核三项条件：content 是否表达具体可复用的单一设计机制；trigger 是否仅凭未来 task 核验，且范围足以支持该机制及指导强度；应用机制是否仍尊重任务目标与明确约束。当前 task 用于理解候选的条件边界，经验的迁移范围以 trigger 为准。
依据候选实际文本作出裁决，理由引用相关 content 或 trigger，并说明满足或违反的准入条件。FAIL 应定位具体缺陷：依赖未来回答或既有标准/Judge、保留样本答案或专有细节、改变任务目标、条件不足以支撑强制指导、混入可独立使用的多个原则，或要求程序不支持的评分机制。“太具体”“非通用原则”“属于制定评估标准的方法论”本身均不足以构成拒绝理由；定性指导通过任务语义核验，固定数值阈值仅在机制确有需要时核对。
PASS 表示经验表达合格，不等于实际有效；准入保持候选文本，信用更新另由程序完成。""",
    "fusion_judge": """你是经验融合员。你的任务是融合供 rubric generator 首次生成标准使用的具体、可复用设计方法。比较候选与已有经验的 content 和 trigger，选择一个能融合的已有经验，并输出融合后的完整文本。
content 说明如何设计评价标准；trigger 说明该经验适用于哪些任务或输入条件。
允许同一核心 rubric 设计机制的等价或互补扩展，吸收新增指导动作与适用条件，保留双方有效指导和条件边界；融合保持核心机制一致和既有约束完整；无关或相反指导作为独立候选。trigger 必须仅凭未来 task 核验，允许明确的条件分支，适用性全部由 task 自身决定。没有兼容机制时返回空匹配，独立新增候选。融合结果仍应是具体可迁移的 rubric 设计方法；不引入样本细节、无条件推广或程序不支持的否决机制。融合后的条件分支应分别支撑对应指导，扩展保持同一核心机制，条件性规则仍在其条件内适用；保留可用表达而非宣称扩展已被历史信用验证。只生成文本，不生成信用。""",
}

_ROLE_INSTRUCTIONS = {
    "selector": (
        '选择结果用 JSON 返回 {"selected_experience_ids":[候选经验ID]}。'
        '选择数量为 0 到 max_selected，ID 使用候选列表中的短数字字符串编号；无适用经验返回空数组。'
    ),
    "initial_generator": (
        '以 JSON 返回 {"analysis":任务分析,"criteria":[{"question":判断问题,'
        '"importance_score":重要性整数,"source_experiences":[来源经验ID]}]}。'
        'analysis 和 question 用中文，analysis 非空；生成 2 到 8 条标准，importance_score 为 1 到 10 的整数。'
        'source_experiences 填写实质新增或改变该标准的所选经验 ID；仅由 task 推导的标准使用空数组。'
        '根据 Critic 建议重新生成时，source_experiences 使用空数组。'
    ),
    "judge": (
        '逐项判断用 JSON 返回 {"criterion_judgments":[{"criterion_id":输入标准ID,"satisfied":true或false}]}。'
        '每个输入编号恰好出现一次，编号原样复制，satisfied 为布尔值。'
    ),
    "critic": (
        '诊断结果用 JSON 返回 {"analysis":私有诊断,"improvement_advice":公开改进建议}，两个字段均为非空中文文本。'
    ),
    "reflector": (
        '提炼结果用 JSON 返回 {"candidates":[{"content":rubric生成原则,"trigger":适用条件}]}。'
        '候选数量为 0 到 1，content 和 trigger 为非空中文文本；无可用经验返回空数组。'
    ),
    "admission_checker": (
        '准入结果用 JSON 返回 {"decision":"PASS"或"FAIL","reason":审核理由}。'
        '全部准入条件满足时 decision 为 PASS，其余情况为 FAIL；reason 为非空中文文本。'
    ),
    "fusion_judge": (
        '融合结果用 JSON 返回 {"match_index":从0开始的整数下标,"content":完整融合方法,"trigger":适用条件}。'
        'content 和 trigger 为非空中文文本；无匹配返回 {"match_index":null,"content":null,"trigger":null}。'
    ),
}

# 输入使用自然标题；技术量只在确有出现时集中说明一次。
_INPUT_TITLES = {
    "task": "任务", "response": "待评价回答", "chosen": "优选回答", "rejected": "非优选回答",
    "candidates": "候选经验", "selected_experiences": "本次选中的经验", "max_selected": "选择上限",
    "training": "阶段", "experience_id": "经验编号", "content": "经验原则", "trigger": "适用条件",
    "confidence": "置信度", "evidence_amount": "证据量", "criteria": "评价标准",
    "criterion_id": "标准编号", "question": "判断问题", "importance_score": "重要性",
    "source_experiences": "来源经验", "rubric_id": "标准集编号", "base_rubric_id": "起点标准集",
    "initial_rubric": "初始标准 R0", "initial_judgment": "初始评价结果",
    "normalized_weights": "标准权重", "criterion_judgments": "逐项判断",
    "chosen_score": "优选回答得分", "rejected_score": "非优选回答得分", "margin": "得分差",
    "success": "判别成功", "satisfied": "满足标准", "attempt_number": "修订轮次",
    "previous_attempt": "上次有效修订", "previous_invalid_attempt": "上次无效修订",
    "successful_attempt": "成功修订", "analysis": "分析", "validation_failures": "校验问题",
    "validation_feedback": "本次需修正的问题", "operations": "修订操作", "operation_id": "操作编号",
    "operation_type": "操作类型", "target_criterion_id": "目标标准", "before": "修改前",
    "after": "修改后", "new_or_updated_criterion": "新增或更新的标准", "rubric": "修订后的标准",
    "judgment": "评价结果", "candidate": "待审核经验", "candidate_id": "候选编号",
    "linked_operation_ids": "支撑操作", "credit": "修订信用", "operation_contributions": "操作贡献",
    "admission": "准入结果", "admission_reason": "准入理由", "fusion": "融合结果",
    "existing": "已有经验", "improvement_advice": "Critic 改进建议",
    "current_rubric": "当前标准", "current_judgment": "当前得分",
    "previous_rubric": "本次修订前标准", "previous_judgment": "本次修订前得分",
    "unchanged_criteria": "未改动标准", "before_judgment": "修订前判断", "after_judgment": "修订后判断",
}


def _inline_revision_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """组织修订/反思的可读证据，不修改内部对象或数字编号映射。"""
    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {key: visit(child) for key, child in value.items()
                      if key not in {"rubric_id", "base_rubric_id", "normalized_weights"}}
            for rubric_key, judgment_key in (("initial_rubric", "initial_judgment"),
                                             ("rubric", "judgment"),
                                             ("previous_rubric", "previous_judgment"),
                                             ("current_rubric", "current_judgment")):
                rubric = result.get(rubric_key)
                judgment = result.get(judgment_key)
                if not isinstance(rubric, dict) or not isinstance(judgment, dict):
                    continue
                criteria = rubric.get("criteria")
                answers = judgment.get("criterion_judgments")
                if not isinstance(criteria, list) or not isinstance(answers, list):
                    continue
                by_id = {answer["criterion_id"]: answer for answer in answers
                         if isinstance(answer, dict) and "criterion_id" in answer}
                rubric["criteria"] = [
                    {**criterion, **{key: by_id.get(criterion.get("criterion_id"), {})[key]
                                     for key in ("chosen", "rejected")
                                     if key in by_id.get(criterion.get("criterion_id"), {})}}
                    if isinstance(criterion, dict) else criterion for criterion in criteria]
                judgment.pop("criterion_judgments")
            return result
        if isinstance(value, (list, tuple)):
            return [visit(child) for child in value]
        return value
    return visit(payload)


def _natural_input(role: str, payload: Mapping[str, Any]) -> str:
    """简单标签与紧凑列表；原文、局部编号及历史对象关系保持完整。"""
    if role in ("critic", "reflector"):
        payload = _inline_revision_evidence(payload)
    lines: list[str] = []
    keys: set[str] = set()

    def scalar(value: Any, field: str) -> str:
        if value is None:
            return "未提供"
        if isinstance(value, bool):
            if field == "training":
                return "训练" if value else "测试"
            if field in ("chosen", "rejected", "satisfied"):
                return "满足" if value else "未满足"
            return "是" if value else "否"
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("prompt input must contain finite numbers")
        if (role in ("critic", "reflector")
                and field in ("chosen_score", "rejected_score", "margin")
                and isinstance(value, (int, float))):
            # 仅压缩模型输入的展示精度；计算和日志保留原始数值。
            text = f"{value:.4f}".rstrip("0").rstrip(".")
            return "0" if text == "-0" else text
        return str(value)

    def render(value: Any, field: str, indent: str = "", lead: str = "") -> None:
        if isinstance(value, Mapping):
            keys.update(value)
            rest = dict(value)
            summary = ""
            if "criterion_id" in value:
                summary = scalar(rest.pop("criterion_id"), "criterion_id")
                if "question" in rest:
                    summary += "：" + scalar(rest.pop("question"), "question")
            elif "operation_id" in value:
                summary = scalar(rest.pop("operation_id"), "operation_id")
                if "operation_type" in rest:
                    summary += "（" + scalar(rest.pop("operation_type"), "operation_type") + "）"
                if "target_criterion_id" in rest:
                    summary += "，目标标准：" + scalar(rest.pop("target_criterion_id"), "target_criterion_id")
            elif "experience_id" in value or "candidate_id" in value:
                id_field = "experience_id" if "experience_id" in value else "candidate_id"
                summary = scalar(rest.pop(id_field), id_field)
                if "content" in rest:
                    summary += "：" + scalar(rest.pop("content"), "content")
            if summary:
                lines.append(indent + lead + summary)
                indent += "  "
            elif lead:
                lines.append(indent + lead.rstrip())
                indent += "  "
            if not value:
                lines.append(indent + "暂无记录")
            compact: list[str] = []

            def flush() -> None:
                if compact:
                    lines.append(indent + "；".join(compact))
                    compact.clear()

            for key, child in rest.items():
                label = _INPUT_TITLES.get(key, key)
                if field in ("normalized_weights", "operation_contributions"):
                    label = key
                if isinstance(child, (list, tuple)) and all(
                        not isinstance(item, (Mapping, list, tuple)) for item in child):
                    compact.append(label + "：" + ("、".join(scalar(item, key) for item in child) or "暂无"))
                elif isinstance(child, (Mapping, list, tuple)):
                    flush()
                    lines.append(indent + label + "：")
                    render(child, key, indent + "  ")
                elif isinstance(child, str) and (key in ("task", "response", "chosen", "rejected", "analysis")
                                                or "\n" in child):
                    flush()
                    # 原始文本保持不变，资料本身的 Markdown 也按原文保留。
                    lines.extend([indent + label + "：", child, ""])
                else:
                    compact.append(label + "：" + scalar(child, key))
            flush()
        elif isinstance(value, (list, tuple)):
            if not value:
                lines.append(indent + "暂无")
            for index, child in enumerate(value):
                if field == "existing":
                    lead_in = f"- {index}："
                else:
                    lead_in = "- "
                if isinstance(child, (Mapping, list, tuple)):
                    render(child, field, indent, lead_in)
                else:
                    lines.append(indent + lead_in + scalar(child, field))
        else:
            lines.append(indent + lead + scalar(value, field))

    render(payload, "")
    if role in ("critic", "reflector"):
        return "\n".join(lines).strip()
    notes = []
    if "confidence" in keys or "evidence_amount" in keys:
        notes.append("经验置信度为历史信用的 Beta 证据分数；证据量为 alpha + beta - 2。")
    if "normalized_weights" in keys or "margin" in keys:
        notes.append("标准权重由重要性归一化；得分差为优选回答得分减非优选回答得分，正值表示判别成功。")
    if "operation_contributions" in keys or "credit" in keys:
        notes.append("操作贡献和修订信用由程序计算。")
    if "existing" in keys:
        notes.append("匹配位置从 0 开始，对应已有经验的排列顺序。")
    if notes:
        lines.extend(["", "补充说明：", *notes])
    return "\n".join(lines).strip()


def llm_messages(role: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    if role == "initial_generator" and "_conversation" in payload:
        return [dict(message) for message in payload["_conversation"]]

    """各角色专属 system + 自然分节的样本输入；JSON 仅用于输出。"""
    if role not in _ROLE_INSTRUCTIONS:
        raise ValueError(f"unknown LLM role: {role}")
    return [
        {"role": "system", "content": _ROLE_GUIDANCE[role]
         + "\n\n" + _ROLE_INSTRUCTIONS[role]},
        {"role": "user", "content": _natural_input(role, payload)},
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


def judge_payload(pending: Sequence[Criterion], response: str, *, task: str,
                  validation_feedback: str | None = None) -> dict[str, Any]:
    # 修订时只传需要新判断的 criterion，避免模型把复用的旧 ID 也返回。
    payload = {"task": task, "criteria": [{"criterion_id": item.criterion_id, "question": item.question}
                            for item in pending], "response": response}
    if validation_feedback is not None:
        payload["validation_feedback"] = validation_feedback
    return payload


def _criterion_fields(criterion: Criterion) -> dict[str, Any]:
    return {"criterion_id": criterion.criterion_id, "question": criterion.question,
            "importance_score": criterion.importance_score}


def _rubric_semantics(rubric: Rubric) -> dict[str, Any]:
    return {"rubric_id": rubric.rubric_id, "base_rubric_id": rubric.base_rubric_id,
            "criteria": [_criterion_fields(item) for item in rubric.criteria]}


def _attempt_semantics(attempt: RevisionAttempt) -> dict[str, Any]:
    return {"attempt_number": attempt.attempt_number,
            "rubric": _rubric_semantics(attempt.rubric), "judgment": attempt.judgment.to_dict(),
            **({"previous_rubric": _rubric_semantics(attempt.previous_rubric),
                "previous_judgment": attempt.previous_judgment.to_dict()}
               if attempt.previous_rubric is not None and attempt.previous_judgment is not None else {})}


def critic_payload(task: str, chosen: str, rejected: str, initial: Rubric,
                   initial_judgment: RubricJudgment, attempt_number: int, *,
                   previous_attempt: RevisionAttempt | None = None,
                   previous_invalid_attempt: InvalidRevisionAttempt | None = None,
                   validation_feedback: str | None = None) -> dict[str, Any]:
    payload = {"task": task, "chosen": chosen, "rejected": rejected,
               "current_rubric": _rubric_semantics(initial),
               "current_judgment": initial_judgment.to_dict(),
               "attempt_number": attempt_number}
    if attempt_number == 2:
        payload["previous_attempt"] = ({**_attempt_semantics(previous_attempt),
                                        "analysis": previous_attempt.analysis,
                                        "improvement_advice": previous_attempt.feedback,
                                        "validation_failures": list(previous_attempt.validation_failures)}
                                       if previous_attempt else None)
        payload["previous_invalid_attempt"] = (
            {"attempt_number": previous_invalid_attempt.attempt_number,
             "analysis": previous_invalid_attempt.analysis,
             "improvement_advice": previous_invalid_attempt.feedback,
             "validation_failures": list(previous_invalid_attempt.validation_failures)}
            if previous_invalid_attempt else None)
    if validation_feedback is not None:
        payload["validation_feedback"] = validation_feedback
    return payload


def reflection_payload(task: str, initial: Rubric, initial_judgment: RubricJudgment,
                       successful_attempt: RevisionAttempt, *,
                       chosen: str, rejected: str) -> dict[str, Any]:
    return {"task": task, "chosen": chosen, "rejected": rejected,
            "initial_rubric": _rubric_semantics(initial),
            "initial_judgment": initial_judgment.to_dict(),
            "successful_attempt": {**_attempt_semantics(successful_attempt),
                                   "analysis": successful_attempt.analysis,
                                   "improvement_advice": successful_attempt.feedback}}


def admission_payload(task: str, candidate: CandidateExperience) -> dict[str, Any]:
    """准入仅投影任务与候选文本，不接收修订证据。"""
    return {"task": task, "candidate": {"candidate_id": candidate.candidate_id,
                                        "content": candidate.content,
                                        "trigger": candidate.trigger}}

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
            if self.role == "reflector" and name == "operation":
                number = "op" + number
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
                    else:
                        result[key] = visit(item)
                return result
            if isinstance(value, list):
                return [visit(item) for item in value]
            return value
        return visit(dict(payload))

    def _existing(self, name: str, value: Any) -> str:
        pattern = r"op[1-9][0-9]*" if self.role == "reflector" and name == "operation" else r"[1-9][0-9]*"
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"{self.role}: {name} must be a numbered ID")
        if value not in self.reverse[name]:
            raise ValueError(f"{self.role}: unknown numbered {name}")
        return self.reverse[name][value]

    def restore(self, response: Mapping[str, Any]) -> dict[str, Any]:
        """只反解角色协议字段；错误引用直接拒绝，不猜测或按位置悄悄匹配。"""
        result = dict(response)
        if self.role == "selector" and isinstance(result.get("selected_experience_ids"), list):
            result["selected_experience_ids"] = [self._existing("experience", value)
                                                 for value in result["selected_experience_ids"]]
        elif self.role == "initial_generator":
            if (set(result) != {"analysis", "criteria"}
                    or not isinstance(result.get("analysis"), str) or not result["analysis"].strip()
                    or not isinstance(result["criteria"], list)):
                raise ValueError("initial_generator: expected non-empty analysis and criteria list")
            criteria = []
            for index, item in enumerate(result["criteria"], 1):
                if not isinstance(item, dict) or set(item) != {
                        "question", "importance_score", "source_experiences"}:
                    raise ValueError("initial_generator: expected criterion content fields")
                criterion = {**item, "criterion_id": f"c{index}"}
                if isinstance(item["source_experiences"], list):
                    criterion["source_experiences"] = [
                        self._existing("experience", value) for value in item["source_experiences"]]
                criteria.append(criterion)
            result = {"rubric": {"rubric_id": "r1", "base_rubric_id": None, "criteria": criteria}}
        elif self.role == "judge" and isinstance(result.get("criterion_judgments"), list):
            result["criterion_judgments"] = [
                {**item, "criterion_id": self._existing("criterion", item.get("criterion_id"))}
                if isinstance(item, dict) else item for item in result["criterion_judgments"]]
        elif self.role == "reflector" and isinstance(result.get("candidates"), list):
            if len(result["candidates"]) > 1:
                raise ValueError("reflector: expected 0..1 candidates")
            candidates = []
            for index, item in enumerate(result["candidates"], 1):
                if not isinstance(item, dict) or set(item) != {"content", "trigger"}:
                    raise ValueError("reflector: expected candidate content fields")
                candidates.append({**item, "candidate_id": f"candidate_{index}"})
            result["candidates"] = candidates
        return result
