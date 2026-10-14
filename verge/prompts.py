"""角色输入投影：Generator不见偏好/轨迹；Critic与归纳员核对实际证据。"""

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


PROMPT_VERSION = "local-credit-delayed-experience-signed-cases"

# 专属身份和任务说明；Generator/Critic 的对象编号由系统分配。
_ROLE_GUIDANCE = {
    "selector": """你是经验适用性筛选员。你的任务是针对给定 task，筛选可用于首次生成评价标准的经验。
结合任务语义、实际输入和上下文分析适用条件，允许合理推理，不要求条件由 task 逐字声明。经验须有实质设计帮助；任务目标和明确约束优先，高置信度不能代替适用性。""",
    "initial_generator": """你是评分标准设计者。根据task和本次可用经验，设计用于评价回答的是/否标准。
先理解任务目标、明确约束、实际输入及上下文，确定当前任务中合理的评价要求。任务要求优先；经验是可选的标准设计指导，依据适用条件自主采用、组合或忽略，不因来自Critic而优先。
把任务要求组织成2到8条可独立核验的正向问题，“是”表示满足要求，避免用缺陷问题搭配正向加分。采用经验时落实其设计动作：可能只改变一项，也可能影响多个问题的表达、关系或权重。检查整套标准与task及所采用经验是否相容，避免相互冲突、重复计权或仅追加引用而保留与经验冲突的要求；不要求每条经验都重构全部标准。
对整套标准共同分配总预算1，weight表示相对权重。依据task主次权衡核心与辅助要求，允许权重差距较大；拆分同一维度时共同占据该维度的预算，不因条目更多而扩大重要性。所有weight为有限正数，系统按总和归一化。不引入task无依据的要求、阈值、总分否决、部分分或自动跳过。
来源只标注实质新增或改变问题表达或权重分配的经验；task足以推导且经验未实质改变的标准不挂名，实际采用也不一律删除。analysis简述设计取舍，具体标准与权重应落实这些取舍。""",
    "judge": """你是逐标准回答核验员。结合task、标准问题与当前这份回答，独立判断每条要求是否满足。
逐项定位回答中的相关内容或缺失信息，核对问题要求的性质、范围、数量及否定条件，然后给出satisfied和简短reason。reason记录可核对的判断依据：指出相关表述或缺失内容；涉及计数时给出计数结果及必要的计数对象说明，不只复述结论，也不需要长篇推理。
以原文和task提供的事实为依据，不补造回答内容或外部核验结果。每项按其问题判断，不用其他条目的结论替代当前项。""",
    "critic": """你从失败证据中提炼供未来首次设计评分标准使用的经验。目标是形成有依据、可复用的设计动作。
私有analysis先核对task要求、标准文本、回答事实、两侧Judge判断及理由。Judge理由是待核对的模型解释，不是事实证明；区分标准设计缺陷、表达歧义、Judge执行错误、偏好冲突与证据不足。分差不理想本身不意味着标准需要改变；只有Judge误判且没有可信设计动作时返回空候选。
从可确认的问题中提炼如何设计标准：明确核验性质、澄清语义边界、组织互相一致的要求或分配相对权重。可以使正确要求更明确、更容易被Judge核验，但不能迁就错误事实、放宽task明确约束或为了偏好分差创造无依据要求。保持加权Boolean协议，不提出部分分、总分否决、自动跳过或修改Judge的指令。
公开content表达面向一类问题的标准设计方法，trigger表达生成前能从task、实际输入、上下文或领域知识识别的条件。把本题主体转换为语义角色，把数量、时长等转换为从新task读取的参数，保留方法必需的专业边界。content与trigger范围一致；方法可以包含同一目的的协同动作，不必覆盖全部标准。
公开经验不依赖已有评分标准、Judge或已观察回答特征，不保留本题专名、答案片段、固定数值、偏好身份或私有分差，也不转成给回答者的成稿/拒绝命令。必要领域条件与通用参数说明可以保留，抽象不等于删除所有专业含义。
第二轮接续本样本的消息历史，对照自己上一轮的分析和候选、实际标准、来源与前后Judge依据，辨别诊断错误、方法不足、未落实或Judge执行问题；历史分析和来源声明均不证明已落实。根据新增证据重新分析并提出经验，避免仅换说法重复前候选；仍无可信设计动作可返回空候选。""",
    "experience_updater": """你是经验归纳员。根据同一条经验的有限来源和后续正负使用案例，重新归纳供未来首次设计评分标准使用的方法及适用条件。
输入包含当前经验、完整task/chosen/rejected，以及仅引用本经验的标准和两侧Judge判断/理由。每条标准的feedback由程序给出：positive表示优选满足而非优选不满足，negative表示反向区分，neutral表示同判、记较轻负信用。反馈是应用线索，不是事实证明，也不能从标准子集推断整体结果。applied_experience只在当时使用的文本与当前文本不同时出现。
结合task和两回答核对案例，综合成功与失败的使用：正向案例帮助保留有依据的方法，负向案例帮助识别方法内容不足、适用条件过宽或不恰当的应用。同判仅表示本次没有区分作用，不自动证明方法错误；Judge解释可能有误，不据此新增无依据规则，也不因一次失败机械否定方法。
只归纳一条清晰的设计方法，允许修正content或收窄trigger。保留任务必要边界及合理专业限定，把专名、固定数字和情节转为语义角色或从新task读取的参数，不追加每题特殊规则，不通过举例拼接案例，不让最后一例覆盖其他证据。证据不足以支持改写时保留当前方法，不拆分经验。
content写如何设计标准，trigger只依据生成前task、实际输入、上下文或领域知识识别，不依赖已观察回答或Judge结果。不放宽task明确约束，不提出总分否决、部分分或修改Judge的指令。
content不超过150个字符，trigger不超过60个字符，仅输出content/trigger，不输出analysis或案例分组。""",

}

_ROLE_INSTRUCTIONS = {
    "selector": (
        '选择结果用 JSON 返回 {"selected_experience_ids":[候选经验ID]}。'
        "选择数量为 0 到 max_selected，ID 使用候选列表中的短数字字符串编号；无适用经验返回空数组。"
    ),
    "initial_generator": (
        '以 JSON 返回 {"analysis":任务分析,"criteria":[{"question":判断问题,'
        '"weight":相对权重,"source_experiences":[来源经验ID]}]}。'
        "analysis 和 question 用中文，analysis 非空；生成 2 到 8 条标准，weight为有限正数，按总预算1分配，系统统一归一化。"
        "source_experiences 填写实质新增或改变该标准的本次提供的经验 ID；仅由 task 推导的标准使用空数组。"
    ),
    "judge": (
        '逐项判断用 JSON 返回 {"criterion_judgments":[{"criterion_id":输入标准ID,"satisfied":true或false,"reason":简短判断依据}]}。'
        "每个输入编号恰好出现一次，编号原样复制，satisfied为布尔值，reason为非空字符串。"
    ),
    "critic": (
        '用 JSON 返回 {"analysis":私有诊断,"candidates":[{"content":设计指导,"trigger":适用条件}]}。analysis为非空中文，简洁组织事实核对、问题定位与方法提炼，候选0到1条，content/trigger为非空中文，无可信经验返回空数组。'
    ),
    "experience_updater": (
        '仅返回JSON {"content":"设计方法","trigger":"生成前可识别的适用条件"}。'
        '两字段为非空中文，content≤150字符、trigger≤60字符；无需修改时保留当前方法。'
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
    "weight": "分配权重",
    "source_experiences": "来源经验",
    "rubric_id": "标准集编号",
    "normalized_weights": "标准权重",
    "criterion_judgments": "逐项判断",
    "chosen_score": "优选回答得分",
    "rejected_score": "非优选回答得分",
    "margin": "得分差",
    "success": "判别成功",
    "satisfied": "满足标准",
    "chosen_reason": "优选回答判断依据",
    "rejected_reason": "非优选回答判断依据",
    "reason": "判断依据",
    "analysis": "分析",
    "validation_feedback": "本次需修正的问题",
    "candidate": "待融合经验",
    "candidate_id": "候选编号",
    "current_rubric": "当前评分标准",
    "current_judgment": "本次Judge结果",
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
                                for key in ("chosen", "rejected", "chosen_reason", "rejected_reason")
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


def _natural_input(role: str, payload: Mapping[str, Any], *,
                   critic_labels: bool = False, critic_boolean: bool = False,
                   critic_reasons: str = "inline") -> str:
    """简单标签与紧凑列表；原文、局部编号及历史对象关系保持完整。"""
    if role == "critic":
        payload = _inline_critic_evidence(payload)
    lines: list[str] = []
    keys: set[str] = set()
    titles = dict(_INPUT_TITLES)
    if critic_reasons not in ("inline", "separate", "absent"):
        raise ValueError("unknown Critic reason presentation")
    if role == "critic" and critic_labels:
        titles.update(chosen="回答A", rejected="回答B",
                      chosen_score="回答A得分", rejected_score="回答B得分")
    titles.update(
        {
            "round_number": "本次应用轮次",
            "candidate_experience": "本轮可选候选经验",
            "previous_candidate": "上条候选",
            "adopted": "上条候选声明采用",
            "invalid_error": "上次技术错误",
            "judgment_status": "判断状态",
        }
    )

    if role == "initial_generator":
        # E/C share one usable list in the actual message; audit fields stay distinct.
        payload = dict(payload)
        available = list(payload.get("selected_experiences", []))
        if "candidate_experience" in payload:
            available.append(payload.pop("candidate_experience"))
        if "selected_experiences" in payload:
            payload["selected_experiences"] = available
        titles["selected_experiences"] = "本次可用经验"

    def scalar(value: Any, field: str) -> str:
        if value is None:
            return "未提供"
        if isinstance(value, bool):
            if field == "training":
                return "训练" if value else "测试"
            if field in ("chosen", "rejected", "satisfied"):
                if role == "critic" and critic_boolean:
                    return "true" if value else "false"
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
            sides = {}
            if role == "critic" and "criterion_id" in value:
                for side in ("chosen", "rejected"):
                    if isinstance(rest.get(side), bool):
                        sides[side] = (rest.pop(side), rest.pop(side + "_reason", None))

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
            for side, (judgment, reason) in sides.items():
                lines.append(indent + titles[side] + "：")
                lines.append(indent + "  Judge判定：" + scalar(judgment, side))
                if reason is not None and critic_reasons == "inline":
                    lines.extend([indent + "  Judge解释（待核对）：", str(reason)])
            if sides and critic_reasons == "separate":
                lines.append(indent + "Judge解释（待核对，分别对应上述判定）：")
                for side, (_, reason) in sides.items():
                    if reason is not None:
                        lines.extend([indent + "  " + titles[side] + "：", str(reason)])
        elif isinstance(value, (list, tuple)):
            if not value:
                lines.append(indent + "暂无")
            for child in value:
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
            "标准权重由分配权重归一化；得分差为优选回答得分减非优选回答得分，正值表示判别成功。"
        )
    if notes:
        lines.extend(["", "补充说明：", *notes])
    return "\n".join(lines).strip()


def llm_messages(
    role: str, payload: Mapping[str, Any], *,
    history: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, str]]:
    """Critic接续样本内历史；其他角色保持独立system/user。"""
    if role not in _ROLE_INSTRUCTIONS:
        raise ValueError(f"unknown LLM role: {role}")
    if history:
        if role != "critic":
            raise ValueError("only critic accepts conversation history")
        if "validation_feedback" in payload:
            feedback = {"validation_feedback": payload["validation_feedback"]}
            instruction = "请修正上条输出，仍只返回analysis及0到1条候选。"
        else:
            feedback = {key: value for key, value in payload.items() if key not in {
                "task", "chosen", "rejected", "previous_rubric", "previous_judgment",
            }}
            instruction = (
                "上次未形成有效应用结果，没有新的Judge判断。请结合技术错误重新分析并提出经验。"
                if "invalid_error" in payload else
                "上条候选的独立应用仍未成功。请对照实际结果重新分析并提出新的经验（0到1条）。"
            )
        return [
            *[dict(message) for message in history],
            {"role": "user", "content": instruction + "\n\n"
             + "以下对象编号仅在本条反馈内解释。\n" + _natural_input(role, feedback)},
        ]
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
        "selected_experiences": [
            {"experience_id": item.experience_id, "content": item.content, "trigger": item.trigger}
            for item in selected
        ],
    }
    if candidate is not None:
        payload["candidate_experience"] = {
            "experience_id": candidate.candidate_id,
            "content": candidate.content,
            "trigger": candidate.trigger,
            "status": "当前候选、尚未验证，无历史置信度",
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
            "weight": criterion.weight}


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
    *,
    previous_candidate: CandidateExperience | None = None,
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
    if invalid_error is not None:
        payload["invalid_error"] = invalid_error
        payload["judgment_status"] = "上次未形成有效应用结果，无新的Judge判断"
    return payload




def experience_update_payload(experience: Experience) -> dict[str, Any]:
    cases = []
    for case in experience.cases:
        verdicts = {j.criterion_id: j for j in case.judgment.criterion_judgments}
        card = {
            "sample_id": case.sample_id,
            "task": case.task, "chosen": case.chosen, "rejected": case.rejected,
            "criteria": [{
                "question": c.question,
                "feedback": ("positive" if verdicts[c.criterion_id].chosen > verdicts[c.criterion_id].rejected
                             else "negative" if verdicts[c.criterion_id].chosen < verdicts[c.criterion_id].rejected
                             else "neutral"),
                "chosen": verdicts[c.criterion_id].chosen,
                "rejected": verdicts[c.criterion_id].rejected,
                "chosen_reason": verdicts[c.criterion_id].chosen_reason,
                "rejected_reason": verdicts[c.criterion_id].rejected_reason,
            } for c in case.rubric.criteria if case.source_id in c.source_experiences],
        }
        if (case.content, case.trigger) != (experience.content, experience.trigger):
            card["applied_experience"] = {"content": case.content, "trigger": case.trigger}
        cases.append(card)
    return {"current_experience": {
        "content": experience.content, "trigger": experience.trigger,
    }, "cases": cases}


_ID_FIELDS = {
    "criterion_id": "criterion",
    "rubric_id": "rubric",
    "experience_id": "experience",
    "candidate_id": "experience",
    "sample_id": "sample",
}
_ID_LISTS = {
    "source_experiences": "experience",
    "selected_experience_ids": "experience",
    "adopted_criterion_ids": "criterion",
    "sample_ids": "sample",
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
            name: {} for name in ("criterion", "rubric", "experience", "sample")
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
                    "weight",
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
