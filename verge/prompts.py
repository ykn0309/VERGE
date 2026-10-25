"""V2 role contracts, short local identifiers, and visibility projections."""

from __future__ import annotations
from typing import Any
import json

PROMPT_VERSION = "verge-v2-semistructured-bn-4"
METHOD_SCHEMA = {
    "trigger": "生成前可识别的适用条件",
    "principle": "一条独立的标准设计方法",
    "template": None,
    "parameters": {},
}
_GUIDANCE = {
    "selector": """你是经验适用性筛选员。只根据任务和Trigger选择有实质设计帮助的方法，最多max_selected条；不适用则不选，允许空集合。先看适用性，再参考q_posterior、后续应用与修订回放的独立证据数；未估计不是0分，高Q不能抵消不适用，回放证据不等于后续验证。""",
    "initial_generator": """你是评分标准设计者。仅根据task和本次可用经验设计2至8条Boolean标准，是表示满足。一个标准最多引用一条经验，也可不引用；仅在方法实质影响标准时引用。经验条件不是当前事实，不为区分未知回答创造任务之外的要求。共同分配相对权重预算1，保留任务主次，不按条数放大维度。可同一次补充任务需要的其他标准。存在模板时用task中的信息绑定全部参数，保留binding_evidence原文，question必须为模板填充后的结果；无模板时自行实现Principle，bindings和binding_evidence均为空。""",
    "judge": """你是独立的逐标准核验员。只根据task、标准和当前一份回答逐项输出Boolean和简短原文依据，覆盖所有标准，不评价另一个回答，不猜测偏好。核对范围、否定、数量和隐含但等价的表达，不补造事实。""",
    "critic": """你从真实失败轨迹中提炼可供未来首次生成rubric使用的原子方法。先在简短analysis中核对任务、两回答的内容和Judge实际Boolean；Judge理由可能错误，不能为了符合偏好创造任务未要求的格式或数量。仅Judge误判、两回答均达标或证据不足时可无候选。直接输出0至max_candidates条互相独立的方法，每条一个可独立采用/忽略/修订的动作，保留必要边界。Trigger用于生成前判断，Principle描述rubric设计方法；Template可选，用简单{parameter}占位，每个参数定义description和source，不把专名、固定数值、当前回答片段或偏好身份带入经验。不是给答题者的答案指令。不新增analysis字段到经验中。第二轮沿用本题真实消息，依最新实际反馈纠正先前诊断，重新提出当前候选集合，不要求维持错误提案。文字简洁，Trigger尽量60字内、Principle尽量150字内，仅为提示要求。""",
    "experience_updater": """你是经验修订员。根据当前固定版本、触发原因及最多6个真实案例决定Keep或Revise。原始反馈和Checker报告均可能错，不能将反向/同判机械当内容缺陷。选择反复错误优先核对Trigger；应用反复错误优先核对Principle/Template/Parameters；Judge报告错误核对歧义与可判定性，不迎合错误事实；Q风险结合内容反证。只修正当前核心方法，不追加独立规则、不拆分、不输出列表。保留已有合理方法和必要边界；明确指令仅被执行者违反或证据不足可以Keep。Revise返回恰好一个完整新方法、目标字段和短理由。""",
    "selector_checker": """你是独立适用性核对员。核对给定任务是否符合经验Trigger，不看回答偏好或其他Checker报告。仅检查当前单条经验与task的匹配，不判断多经验间依赖或冲突。判断本次选择是否合理，返回correct/error/unknown及简短可核对依据。不能把主题相关当方法适用，不凭经验效果反推适用性。""",
    "generator_checker": """你是独立应用核对员。逐项检查引用标准是否忠实执行经验的Principle、可选Template和参数绑定，是否增加无依据要求或仅挂名引用。核对对象是Generator生成的criterion本身是否落实方法，不是检查答题者的回答是否达标。因此无需也不会提供实际回答，不能仅因缺少实际回答而报告unknown。只看task、经验及其引用标准，不看回答偏好、实际反馈或其他核对器结论。返回整体correct/error/unknown和短证据，可在details列逐标准问题。清楚指令被错误落实应报应用错误，不能凭空认定原方法错误。""",
    "judge_checker": """你是独立评价执行核对员。依据task、引用标准、两个匿名回答及Judge实际Boolean和理由，检查Judge是否忠实执行标准。不知道哪个回答被偏好，不看应用正反方向或其他Checker报告。判断的是Judge是否执行正确，不是标准应该如何设计。返回correct/error/unknown、短证据及可选逐标准details。""",
}
_SCHEMA = {
    "selector": {"selected_experience_ids": ["短编号"]},
    "initial_generator": {
        "analysis": "简短设计取舍",
        "criteria": [
            {
                "question": "是/否标准",
                "weight": 1,
                "source_experiences": ["1"],
                "bindings": {},
                "binding_evidence": {},
            }
        ],
    },
    "judge": {
        "criterion_judgments": [
            {"criterion_id": "短编号", "satisfied": True, "reason": "原文依据"}
        ]
    },
    "critic": {"analysis": "简短事实诊断", "candidates": [METHOD_SCHEMA]},
    "experience_updater": {
        "action": "keep或revise",
        "method": None,
        "target_fields": [],
        "reason": "短理由；keep时method为null，revise时为完整四字段方法",
    },
}
for role in ("selector_checker", "generator_checker", "judge_checker"):
    _SCHEMA[role] = {
        "status": "correct/error/unknown",
        "evidence": "短依据",
        "details": [],
    }


def llm_messages(role, payload, *, history=None):
    if role not in _GUIDANCE:
        raise ValueError("unknown role " + role)
    schema = json.dumps(_SCHEMA[role], ensure_ascii=False)
    system = (
        _GUIDANCE[role]
        + "\n严格只返回以下字段的JSON，示意值需替换为实际结果："
        + schema
    )
    if role in ("critic", "experience_updater"):
        system += (
            "\n方法结构："
            + json.dumps(METHOD_SCHEMA, ensure_ascii=False)
            + '；参数值结构：{"description":"含义","source":"task中的绑定来源约束"}。'
        )
    if role == "initial_generator":
        system += '\nsource_experiences只能为[]或["1"]这样的短编号字符串数组，最多一个元素，绝不能放经验对象或新造编号；没有适用经验就用[]，仍生成2至8条task标准。'
    if role in ("generator_checker", "judge_checker"):
        system += '\ndetails可为[]，或[{"criterion_id":"提供的短编号","status":"correct/error/unknown","evidence":"短依据"}]。不得使用其他字段；任一detail为error则整体status必须error；含unknown时不能整体correct。'
    if role == "selector_checker":
        system += "\ndetails必须为[]，本角色只做单条经验对任务的适用性判断。"
    if role == "experience_updater":
        system += "\ntarget_fields的每项只能是小写英文trigger、principle、template、parameters，不能填content、method、criteria、chosen等。keep用空数组；revise填写实际修改的字段。template为null时parameters必须是{}；只有模板的{参数名}才在parameters中定义。不要为追求区分创造逐字复述或固定数量要求。"
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    user = "\n\n".join(
        k + "：\n" + json.dumps(v, ensure_ascii=False, indent=2)
        for k, v in clean.items()
    )
    if history:
        return [*history, {"role": "user", "content": user}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class NumberedIDCodec:
    """Stable within a Critic conversation; independent for every other request."""

    fields = {
        "experience_id": "experience",
        "version_id": "experience",
        "parent_id": "experience",
        "source_sample_id": "sample",
        "sample_id": "sample",
        "task_id": "task",
        "criterion_id": "criterion",
        "rubric_id": "rubric",
        "trigger_sample_id": "sample",
        "created_by_event_id": "event",
        "retired_by_event_id": "event",
    }
    lists = {
        "selected_experience_ids": "experience",
        "source_experiences": "experience",
        "case_ids": "sample",
    }

    def __init__(self, role="", payload=None):
        self.forward = {}
        self.reverse = {}

    def _encode(self, namespace, value):
        f = self.forward.setdefault(namespace, {})
        r = self.reverse.setdefault(namespace, {})
        if value not in f:
            short = str(len(f) + 1)
            f[value] = short
            r[short] = value
        return f[value]

    def project(self, payload):
        def walk(x):
            if isinstance(x, list):
                return [walk(v) for v in x]
            if not isinstance(x, dict):
                return x
            out = {}
            for k, v in x.items():
                if k in self.fields and isinstance(v, str):
                    out[k] = self._encode(self.fields[k], v)
                elif k in self.lists and isinstance(v, list):
                    out[k] = [self._encode(self.lists[k], n) for n in v]
                else:
                    out[k] = walk(v)
            return out

        return walk(payload)

    def restore(self, payload):
        def lookup(namespace, value):
            if type(value) is int:
                value = str(value)
            if not isinstance(value, str) or value not in self.reverse.get(
                namespace, {}
            ):
                raise ValueError("unknown short " + namespace + " ID")
            return self.reverse[namespace][value]

        def walk(x):
            if isinstance(x, list):
                return [walk(v) for v in x]
            if not isinstance(x, dict):
                return x
            return {
                k: (
                    lookup(self.fields[k], v)
                    if k in self.fields and v is not None
                    else (
                        [lookup(self.lists[k], n) for n in v]
                        if k in self.lists and isinstance(v, list)
                        else walk(v)
                    )
                )
                for k, v in x.items()
            }

        return walk(payload)


def experience_payload(e, *, quality=False):
    data = {"experience_id": e.experience_id, **e.method.to_dict()}
    if quality:
        data.update(
            q_posterior=e.q_posterior,
            evidence_counts=e.evidence_counts,
            effective_tasks=len(
                {
                    a.sample.task_id
                    for a in e.applications
                    if a.phase == "downstream"
                    and a.update_reason not in ("duplicate", "duplicate_task")
                    and a.feedback in ("positive", "reverse", "tie")
                }
            ),
            replay_tasks=len(
                {
                    a.sample.task_id
                    for a in e.applications
                    if a.phase == "replay"
                    and a.update_reason not in ("duplicate", "duplicate_task")
                    and a.feedback in ("positive", "reverse", "tie")
                }
            ),
            quality_state=(
                "unestimated"
                if e.q_posterior is None
                else (
                    "prior_only" if not sum(e.evidence_counts.values()) else "estimated"
                )
            ),
        )
    return data


def evidence_payload(a):
    """Updater/Critic may see preference; Checkers use separate projections."""
    answers = {j.criterion_id: j for j in a.judgment.criterion_judgments}
    criteria = [
        {**c.to_dict(), **answers[c.criterion_id].to_dict()}
        for c in a.rubric.criteria
        if a.version_id in c.source_experiences
    ]
    return {
        "sample_id": a.sample.sample_id,
        "task": a.sample.task,
        "chosen": a.sample.chosen,
        "rejected": a.sample.rejected,
        "actual_version": {"version_id": a.version_id, **a.method.to_dict()},
        "criteria": criteria,
        "feedback": a.feedback,
        "reports": [r.to_dict() for r in a.reports],
        "q_before": a.q_before,
        "q_after": a.q_after,
        "phase": a.phase,
        "task_id": a.sample.task_id,
        "round_number": a.round_number,
        "observation": a.observation,
        "outcomes": a.outcomes,
        "positive_weight": a.positive_weight,
        "reverse_weight": a.reverse_weight,
        "update_reason": a.update_reason,
        "counted_as_quality_evidence": a.update_reason == "updated",
        "application_state": {
            "recalled": a.recalled,
            "selected": a.selected,
            "cited": a.cited,
            "no_evidence_reason": a.no_evidence_reason,
        },
    }
