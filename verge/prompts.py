"""各角色独立的输入投影；隔离 preference、provenance 和 Fusion 数值。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from verge.models import (
    CandidateExperience,
    Criterion,
    Experience,
    Rubric,
    RubricJudgment,
)


PROMPT_VERSION = "critic-candidate-admission-fresh-application"

# 专属身份和任务说明；Generator/Critic 的对象编号由系统分配。
_ROLE_GUIDANCE = {
    "selector": """你是经验适用性筛选员。你的任务是针对给定 task，筛选可用于首次生成评价标准的经验。
结合任务语义、实际输入和上下文分析适用条件，允许合理推理，不要求条件由 task 逐字声明。经验须有实质设计帮助；任务目标和明确约束优先，高置信度不能代替适用性。""",
    "initial_generator": """你是评分标准设计者。依据task和本次提供的经验，设计用于评价回答的是/否标准。
先分析任务目标、实际输入与完整性、明确约束及评价重要性。任务要求优先；已有经验与已准入候选都是可选指导，自主采用、组合或忽略，不因来自Critic就优先或强制采用，不新增无依据要求。
每条问题独立核验，“是”表示满足要求，不用“是否存在错误”等缺陷问题搭配正向加分。重要性表示task重要程度。来源只标注实质新增或改变该标准的经验；task足以推导的标准不挂名，实际采用也不一律删除。""",
    "judge": """你是逐标准回答核验员。你的任务是结合 task 和评价标准，依据每条提供的标准和当前回答，独立判断这份回答是否满足要求。
每条标准对应一个 satisfied Boolean 判断，各项结果由这份回答的实际内容确定。""",
    "critic": """你是rubric生成经验提出者。核对task、偏好对、当前失败标准和实际Judge结果，先输出私有analysis，再提出0到1条供首次生成rubric使用的候选经验。
失败不证明标准有错；区分设计缺陷、Judge疑点与证据不足。后续轮核对上条候选、实际生成和采用记录，旧私有分析不代替事实。准入重生成时依据被拒候选和具体拒绝原因重新提出指导。
content表达具体连贯的标准设计方法；trigger表达生成前可识别的任务或输入条件，允许上下文、领域知识推理及同一目的的协同动作，不依赖已有rubric、Judge或已观察到的回答特征。
候选不能包含候选回答特有内容、偏好身份、私有评分或比较结论。task及明确约束优先，不为偏好分差放宽约束，不把加严或增加细节当作自动改进。不要求程序支持部分分、否决、自动跳过或修改已有Judge状态。没有可信可复用指导返回空候选，不强行解释成功或伪造经验。""",
    "admission_checker": """你是候选 rubric 经验的准入审核员。你的任务是审核经验能否指导未来任务首次生成标准，判断表达是否可用，而非提前证明实际有效。
content 是具体、可复用且指导连贯的标准设计方法；trigger 是生成标准之前可识别的适用场景。允许结合任务语义、实际输入、上下文和领域知识进行合理推理；限定任务类型、具体指导或多个协同动作本身不足以拒绝。
content 可以指导核验未来回答的性质，但 trigger 不应以已经观察到这些性质为前提，也不应依赖已有 rubric 或 Judge 结果。允许设计评价维度、安排重要性及排除不当要求，应用仍尊重任务目标与明确约束。
只因明确缺陷返回 FAIL：事后触发条件、保留样本答案细节或偏好身份、改变任务目标、指导与适用条件不匹配、拼接互不相关的指导，或要求程序不支持的评分机制。理由引用实际 content 或 trigger 并指出具体缺陷，不能仅因复杂、具体或属于评分标准设计方法而拒绝。PASS 保持候选文本，不等于实际有效。""",
    "fusion_judge": """你是经验融合员。你的任务是识别候选与已有经验是否重复，而非尽量合并或压缩经验池；默认独立新增候选。
只有两者解决同一评价设计问题、指导等价或属于同一方法的必要细化，且适用条件兼容时才 MERGE。共享质量目标、主题或关键词不足以合并；服务不同设计目的、需在各自场景独立使用的指导分别保留。不确定是否重复时返回空匹配。
融合结果应能指导首次生成标准，适用条件允许任务语义与上下文分析，但不依赖已有 rubric、Judge 结果或已观察到的候选特征。去重提炼为统一指导，保留必要条件边界，不逐条追加无关原则；若必须拼接不同设计问题、扩展无关场景或修复事后触发条件才能合并，则独立新增。
同一设计指导可含多个协同动作。输出完整 content 和 trigger，只生成文本，不生成信用；历史证据不认证新增内容。""",
}

_ROLE_INSTRUCTIONS = {
    "selector": (
        '选择结果用 JSON 返回 {"selected_experience_ids":[候选经验ID]}。'
        "选择数量为 0 到 max_selected，ID 使用候选列表中的短数字字符串编号；无适用经验返回空数组。"
    ),
    "initial_generator": (
        '以 JSON 返回 {"analysis":任务分析,"criteria":[{"question":判断问题,'
        '"importance_score":重要性整数,"source_experiences":[来源经验ID]}]}。'
        "analysis 和 question 用中文，analysis 非空；生成 2 到 8 条标准，importance_score 为 1 到 10 的整数。"
        "source_experiences 填写实质新增或改变该标准的本次提供的经验 ID；仅由 task 推导的标准使用空数组。"
    ),
    "judge": (
        '逐项判断用 JSON 返回 {"criterion_judgments":[{"criterion_id":输入标准ID,"satisfied":true或false}]}。'
        "每个输入编号恰好出现一次，编号原样复制，satisfied 为布尔值。"
    ),
    "critic": (
        '用 JSON 返回 {"analysis":私有诊断,"candidates":[{"content":设计指导,"trigger":适用条件}]}。analysis为非空中文，候选0到1条，content/trigger为非空中文，无可信经验返回空数组。'
    ),
    "admission_checker": (
        '准入结果用 JSON 返回 {"decision":"PASS"或"FAIL","reason":审核理由}。'
        "全部准入条件满足时 decision 为 PASS，其余情况为 FAIL；reason 为非空中文文本。"
    ),
    "fusion_judge": (
        '融合结果用 JSON 返回 {"match_index":从0开始的整数下标,"content":完整融合方法,"trigger":适用条件}。'
        'content 和 trigger 为非空中文文本；无匹配返回 {"match_index":null,"content":null,"trigger":null}。'
    ),
}

# 输入使用自然标题；技术量只在确有出现时集中说明一次。
_INPUT_TITLES = {
    "task": "任务",
    "response": "待评价回答",
    "chosen": "优选回答",
    "rejected": "非优选回答",
    "candidates": "候选经验",
    "selected_experiences": "本次选中的经验",
    "max_selected": "选择上限",
    "training": "阶段",
    "experience_id": "经验编号",
    "content": "经验原则",
    "trigger": "适用条件",
    "confidence": "置信度",
    "evidence_amount": "证据量",
    "criteria": "评价标准",
    "criterion_id": "标准编号",
    "question": "判断问题",
    "importance_score": "重要性",
    "source_experiences": "来源经验",
    "rubric_id": "标准集编号",
    "normalized_weights": "标准权重",
    "criterion_judgments": "逐项判断",
    "chosen_score": "优选回答得分",
    "rejected_score": "非优选回答得分",
    "margin": "得分差",
    "success": "判别成功",
    "satisfied": "满足标准",
    "analysis": "分析",
    "validation_feedback": "本次需修正的问题",
    "candidate": "待审核经验",
    "candidate_id": "候选编号",
    "admission_reason": "准入理由",
    "existing": "已有经验",
    "current_rubric": "当前失败标准",
    "current_judgment": "当前判断",
    "previous_rubric": "上轮应用之前的标准",
    "previous_judgment": "上轮应用之前的判断",
    "adopted_criterion_ids": "上条候选声明引用标准",
    "status": "证据状态",
}


def _inline_critic_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """组织实际失败与应用前的可读证据，不修改内部对象或数字编号映射。"""

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {
                key: visit(child)
                for key, child in value.items()
                if key not in {"rubric_id", "normalized_weights"}
            }
            for rubric_key, judgment_key in (
                ("initial_rubric", "initial_judgment"),
                ("rubric", "judgment"),
                ("previous_rubric", "previous_judgment"),
                ("current_rubric", "current_judgment"),
            ):
                rubric = result.get(rubric_key)
                judgment = result.get(judgment_key)
                if not isinstance(rubric, dict) or not isinstance(judgment, dict):
                    continue
                criteria = rubric.get("criteria")
                answers = judgment.get("criterion_judgments")
                if not isinstance(criteria, list) or not isinstance(answers, list):
                    continue
                by_id = {
                    answer["criterion_id"]: answer
                    for answer in answers
                    if isinstance(answer, dict) and "criterion_id" in answer
                }
                rubric["criteria"] = [
                    (
                        {
                            **criterion,
                            **{
                                key: by_id.get(criterion.get("criterion_id"), {})[key]
                                for key in ("chosen", "rejected")
                                if key in by_id.get(criterion.get("criterion_id"), {})
                            },
                        }
                        if isinstance(criterion, dict)
                        else criterion
                    )
                    for criterion in criteria
                ]
                judgment.pop("criterion_judgments")
            return result
        if isinstance(value, (list, tuple)):
            return [visit(child) for child in value]
        return value

    return visit(payload)


def _natural_input(role: str, payload: Mapping[str, Any]) -> str:
    """简单标签与紧凑列表；原文、局部编号及历史对象关系保持完整。"""
    if role == "critic":
        payload = _inline_critic_evidence(payload)
    lines: list[str] = []
    keys: set[str] = set()
    titles = dict(_INPUT_TITLES)
    titles.update(
        {
            "round_number": "本次应用轮次",
            "generation_number": "本轮候选生成次数",
            "candidate_experience": "本轮可选候选经验",
            "previous_candidate": "上条已准入候选",
            "rejected_candidate": "本轮被拒候选",
            "adopted": "上条候选声明采用",
            "invalid_error": "上次技术错误",
            "judgment_status": "判断状态",
        }
    )

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
        if (
            role == "critic"
            and field in ("chosen_score", "rejected_score", "margin")
            and isinstance(value, (int, float))
        ):
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
            elif "experience_id" in value or "candidate_id" in value:
                id_field = (
                    "experience_id" if "experience_id" in value else "candidate_id"
                )
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
                label = titles.get(key, key)
                if role == "critic" and key == "attempt_number":
                    label = "应用轮次"
                if field == "normalized_weights":
                    label = key
                if isinstance(child, (list, tuple)) and all(
                    not isinstance(item, (Mapping, list, tuple)) for item in child
                ):
                    compact.append(
                        label
                        + "："
                        + ("、".join(scalar(item, key) for item in child) or "暂无")
                    )
                elif isinstance(child, (Mapping, list, tuple)):
                    flush()
                    lines.append(indent + label + "：")
                    render(child, key, indent + "  ")
                elif isinstance(child, str) and (
                    key in ("task", "response", "chosen", "rejected", "analysis")
                    or "\n" in child
                ):
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
    if role == "critic":
        return "\n".join(lines).strip()
    notes = []
    if "confidence" in keys or "evidence_amount" in keys:
        notes.append(
            "经验置信度为历史信用的 Beta 证据分数；证据量为 alpha + beta - 2。"
        )
    if "normalized_weights" in keys or "margin" in keys:
        notes.append(
            "标准权重由重要性归一化；得分差为优选回答得分减非优选回答得分，正值表示判别成功。"
        )
    if notes:
        lines.extend(["", "补充说明：", *notes])
    return "\n".join(lines).strip()


def llm_messages(role: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """各角色专属 system + 自然分节的样本输入；JSON 仅用于输出。"""
    if role not in _ROLE_INSTRUCTIONS:
        raise ValueError(f"unknown LLM role: {role}")
    return [
        {
            "role": "system",
            "content": _ROLE_GUIDANCE[role] + "\n\n" + _ROLE_INSTRUCTIONS[role],
        },
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


def generator_payload(
    task: str,
    selected: Sequence[ExperienceView],
    *,
    candidate: CandidateExperience | None = None,
    validation_feedback: str | None = None,
) -> dict[str, Any]:
    payload = {
        "task": task,
        "selected_experiences": [item.to_dict() for item in selected],
    }
    if candidate is not None:
        payload["candidate_experience"] = {
            "experience_id": candidate.candidate_id,
            "content": candidate.content,
            "trigger": candidate.trigger,
            "status": "已准入、尚未验证，无历史置信度",
        }
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
    return {
        "rubric_id": rubric.rubric_id,
        "criteria": [_criterion_fields(item) for item in rubric.criteria],
    }


def critic_payload(
    task: str,
    chosen: str,
    rejected: str,
    current: Rubric,
    judgment: RubricJudgment,
    round_number: int,
    generation_number: int,
    *,
    previous_candidate: CandidateExperience | None = None,
    rejected_candidate: CandidateExperience | None = None,
    adopted: bool | None = None,
    invalid_error: str | None = None,
    previous_rubric: Rubric | None = None,
    previous_judgment: RubricJudgment | None = None,
) -> dict[str, Any]:
    payload = {
        "task": task,
        "chosen": chosen,
        "rejected": rejected,
        "round_number": round_number,
        "generation_number": generation_number,
        "current_rubric": current.to_dict(),
        "current_judgment": judgment.to_dict(),
    }
    if previous_rubric is not None and previous_judgment is not None:
        payload["previous_rubric"] = previous_rubric.to_dict()
        payload["previous_judgment"] = previous_judgment.to_dict()
    if previous_candidate is not None:
        payload["previous_candidate"] = {
            "candidate_id": previous_candidate.candidate_id,
            "content": previous_candidate.content,
            "trigger": previous_candidate.trigger,
        }
        payload["adopted"] = adopted
        payload["adopted_criterion_ids"] = (
            [
                c.criterion_id
                for c in current.criteria
                if previous_candidate.candidate_id in c.source_experiences
            ]
            if adopted is not None
            else None
        )
    if rejected_candidate is not None:
        payload["rejected_candidate"] = {
            "candidate_id": rejected_candidate.candidate_id,
            "content": rejected_candidate.content,
            "trigger": rejected_candidate.trigger,
            "admission_reason": rejected_candidate.admission_reason,
        }
    if invalid_error is not None:
        payload["invalid_error"] = invalid_error
        payload["judgment_status"] = "上次未形成有效应用结果，无新的Judge判断"
    return payload


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


_ID_FIELDS = {
    "criterion_id": "criterion",
    "rubric_id": "rubric",
    "experience_id": "experience",
    "candidate_id": "experience",
}
_ID_LISTS = {
    "source_experiences": "experience",
    "selected_experience_ids": "experience",
    "adopted_criterion_ids": "criterion",
}
_ID_DICTS = {"normalized_weights": "criterion"}


class NumberedIDCodec:
    """单次模型调用的局部 ID 表；真实内部 ID 从不进入结构化提示词。

    不跨调用缓存：每次只根据本次白名单输入建立编号，再把模型输出还原到内部 ID。
    不改写 task、chosen/rejected 或任意自然语言文本。
    """

    def __init__(self, role: str, payload: Mapping[str, Any]) -> None:
        self.role = role
        self.forward: dict[str, dict[str, str]] = {
            name: {} for name in ("criterion", "rubric", "experience")
        }
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
                    else:
                        result[key] = visit(item)
                return result
            if isinstance(value, list):
                return [visit(item) for item in value]
            return value
        return visit(dict(payload))

    def _existing(self, name: str, value: Any) -> str:
        pattern = r"[1-9][0-9]*"
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"{self.role}: {name} must be a numbered ID")
        if value not in self.reverse[name]:
            raise ValueError(f"{self.role}: unknown numbered {name}")
        return self.reverse[name][value]

    def restore(self, response: Mapping[str, Any]) -> dict[str, Any]:
        """只反解角色协议字段；错误引用直接拒绝，不猜测或按位置悄悄匹配。"""
        result = dict(response)
        if self.role == "selector" and isinstance(
            result.get("selected_experience_ids"), list
        ):
            result["selected_experience_ids"] = [
                self._existing("experience", value)
                for value in result["selected_experience_ids"]
            ]
        elif self.role == "initial_generator":
            if (
                set(result) != {"analysis", "criteria"}
                or not isinstance(result.get("analysis"), str)
                or not result["analysis"].strip()
                or not isinstance(result["criteria"], list)
            ):
                raise ValueError(
                    "initial_generator: expected non-empty analysis and criteria list"
                )
            criteria = []
            for index, item in enumerate(result["criteria"], 1):
                if not isinstance(item, dict) or set(item) != {
                    "question",
                    "importance_score",
                    "source_experiences",
                }:
                    raise ValueError(
                        "initial_generator: expected criterion content fields"
                    )
                criterion = {**item, "criterion_id": f"c{index}"}
                if isinstance(item["source_experiences"], list):
                    criterion["source_experiences"] = [
                        self._existing("experience", value)
                        for value in item["source_experiences"]
                    ]
                criteria.append(criterion)
            result = {"rubric": {"rubric_id": "r1", "criteria": criteria}}
        elif self.role == "judge" and isinstance(
            result.get("criterion_judgments"), list
        ):
            result["criterion_judgments"] = [
                (
                    {
                        **item,
                        "criterion_id": self._existing(
                            "criterion", item.get("criterion_id")
                        ),
                    }
                    if isinstance(item, dict)
                    else item
                )
                for item in result["criterion_judgments"]
            ]
        elif self.role == "critic" and isinstance(result.get("candidates"), list):
            if len(result["candidates"]) > 1:
                raise ValueError("critic: expected 0..1 candidates")
            candidates = []
            for index, item in enumerate(result["candidates"], 1):
                if not isinstance(item, dict) or set(item) != {"content", "trigger"}:
                    raise ValueError("critic: expected candidate content fields")
                candidates.append({**item, "candidate_id": f"candidate_{index}"})
            result["candidates"] = candidates
        return result
