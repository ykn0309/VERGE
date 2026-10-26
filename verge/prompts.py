"""V2 role contracts, short local identifiers, and visibility projections."""

from __future__ import annotations
from typing import Any
import json

PROMPT_VERSION = "verge-v2-semistructured-bn-12"
METHOD_SCHEMA = {
    "trigger": "任务明确要求将给定量换算为指定单位",
    "principle": "标准应核验换算后数值与目标单位是否正确对应原始量",
    "template": "回答是否将{source_quantity}正确换算为{target_unit}？",
    "parameters": {
        "source_quantity": {
            "description": "题目给出的待换算量及其原单位",
        },
        "target_unit": {
            "description": "题目要求使用的目标单位",
        },
    },
}
_GUIDANCE = {
    "experience_checker": "你是经验质量核对员。依据task、初始失败轨迹及候选方法，检查合理性、原子性、可复用性，以及是否夹带无必要的本题人名、数值、剧情或答案。原轨迹与Judge理由可能错误，不把偏好标签当任务要求。只对明确缺陷拒绝：方法引入任务外强制要求、包含多个可独立采用的方法、把偶然细节写成通用规则，或参数只能从未来回答而非task获得。场景限定和必要边界可以保留；不因适用范围窄、没有模板、措辞不完美或仅有潜在风险拒绝。不要求经验适用于所有任务。通过代表可入池试用，不保证实际有效。拒绝时reason简短指出具体缺陷和修改方向，不要求增加独立规则。",
    "selector": """你是经验适用性筛选员。只根据任务和Trigger选择有实质设计帮助的方法，最多max_selected条；不适用则不选，允许空集合。先看适用性，再参考q_posterior、后续应用与修订回放的独立证据数；未估计不是0分，高Q不能抵消不适用，回放证据不等于后续验证。""",
    "initial_generator": """你是评分标准设计者。仅根据task和本次可用经验设计2至8条Boolean标准，是表示满足。一个标准最多引用一条经验，也可不引用；仅在方法实质影响标准时引用。经验条件不是当前事实，不为区分未知回答创造任务之外的要求。整套标准共同分配总权重1，每条weight是该总预算中的份额，不是每条各自分配1。依据task中要求的重要性分配，保留主次；拆出更多标准时重新分配已有预算，不按条数放大维度。权重可以相同，但不能机械地全部填1或为制造差异强行设不同值。可同一次补充任务需要的其他标准。存在模板时用task中的信息绑定全部参数，question应保留模板核验语义，可根据上下文自然表述，无需逐字复现模板；无模板时自行实现Principle，bindings为空。""",
    "judge": """你是独立的逐标准核验员。只根据task、标准和当前一份回答逐项输出JSON布尔判断和简短判断理由，覆盖所有标准，不评价另一个回答，不猜测偏好。核对范围、否定、数量和隐含但等价的表达，不补造事实。""",
    "critic": """你从真实失败轨迹中提炼可供未来首次生成rubric使用的原子方法。先在简短analysis中核对任务、两回答的内容和Judge实际Boolean；Judge理由可能错误，不能为了符合偏好创造任务未要求的格式或数量。仅Judge误判、两回答均达标或证据不足时可无候选。每轮直接输出0或1条候选方法；只保留本轮最关键的一项可复用方法，不将多项独立要求拼成一条。候选只包含一个可独立采用/忽略/修订的动作，保留必要边界。Trigger用于生成前判断，Principle描述rubric设计方法；Template可选，用简单{parameter}占位，每个参数只定义description，不把专名、固定数值、当前回答片段或偏好身份带入经验。不是给答题者的答案指令。不新增analysis字段到经验中。第二轮沿用本题真实消息，只根据经验核对器的拒绝原因修订候选；初始失败轨迹固定，不接收候选重新应用的Generator/Judge结果。若核对器依据有误，应结合原始task合理纠正，不迎合无依据限制。文字简洁，Trigger尽量60字内、Principle尽量150字内，仅为提示要求。""",
    "experience_updater": """你是经验修订员。根据当前固定版本、触发原因及最多6个真实案例决定Keep或Revise。原始反馈和Checker报告均可能错，不能将反向/同判机械当内容缺陷。选择反复错误优先核对Trigger；应用反复错误优先核对Principle/Template/Parameters；Judge报告错误核对歧义与可判定性，不迎合错误事实；Q风险结合内容反证。只修正当前核心方法，不追加独立规则、不拆分、不输出列表。保留已有合理方法和必要边界；明确指令仅被执行者违反或证据不足可以Keep。Revise返回恰好一个完整新方法、目标字段和短理由。""",
    "selector_checker": """你是独立适用性核对员。核对给定任务是否符合经验Trigger，不看回答偏好或其他Checker报告。仅检查当前单条经验与task的匹配，不判断多经验间依赖或冲突。判断本次选择是否合理，返回correct/error/unknown及简短可核对依据。不能把主题相关当方法适用，不凭经验效果反推适用性。""",
    "generator_checker": """你是独立应用核对员。逐项检查引用标准是否忠实执行经验的Principle、可选Template和参数绑定，是否增加无依据要求或仅挂名引用。核对对象是Generator生成的criterion本身是否落实方法，不是检查答题者的回答是否达标。因此无需也不会提供实际回答，不能仅因缺少实际回答而报告unknown。只看task、经验及其引用标准，不看回答偏好、实际反馈或其他核对器结论。检查参数遗漏、额外绑定、依据不足及语义偏离；允许等价改写、翻译和可由task合理推导的绑定，不因措辞或键名差异机械报错，但缺失参数若使必要核验动作不清楚应报告错误。返回整体correct/error/unknown和短证据，可在details列逐标准问题。清楚指令被错误落实应报应用错误，不能凭空认定原方法错误。""",
    "judge_checker": """你是独立评价执行核对员。依据task、引用标准、两个匿名回答及Judge实际Boolean和理由，检查Judge是否忠实执行标准。不知道哪个回答被偏好，不看应用正反方向或其他Checker报告。判断的是Judge是否执行正确，不是标准应该如何设计。返回correct/error/unknown、短证据及可选逐标准details。""",
}
_SCHEMA = {
    "experience_checker": {"approved": True, "reason": "简短通过或拒绝理由"},
    "selector": {"selected_experience_ids": ["短编号"]},
    "initial_generator": {
        "analysis": "简短设计取舍",
        "criteria": [
            {
                "question": "根据task编写的核心核验问题？",
                "weight": 0.6,
                "source_experiences": ["1"],
                "bindings": {},
            },
            {
                "question": "根据task编写的另一项核验问题？",
                "weight": 0.25,
                "source_experiences": [],
                "bindings": {},
            },
            {
                "question": "根据task编写的其余必要核验问题？",
                "weight": 0.15,
                "source_experiences": [],
                "bindings": {},
            },
        ],
    },
    "judge": {
        "criterion_judgments": [
            {"criterion_id": "短编号", "satisfied": True, "reason": "简短判断理由"}
        ]
    },
    "critic": {"analysis": "简短事实诊断", "candidates": [METHOD_SCHEMA]},
    "experience_updater": {
        "action": "keep",
        "method": None,
        "target_fields": [],
        "reason": "短理由；keep时method为null，revise时为完整四字段方法",
    },
}
for role in ("selector_checker", "generator_checker", "judge_checker"):
    _SCHEMA[role] = {
        "status": "correct",
        "evidence": "短依据",
        "details": [],
    }


# Examples remain valid JSON; alternative values and branch constraints belong here.
_VALUE_RULES = {
    "experience_checker": "approved只能是JSON布尔值true或false，true表示可入池试用，false表示存在明确缺陷须重提；不能使用字符串、0/1或null。reason为非空简短字符串，解释通过或明确拒绝依据。只输出这两个字段，不增加评分、分析或子经验列表。",
    "selector": "selected_experience_ids为不重复的短编号字符串数组，只能选输入提供的经验编号，数量为0至max_selected；不选时为[]。",
    "initial_generator": "analysis为字符串；criteria为2至8个标准对象。question为非空的是/否问题，true表示满足、false表示不满足。weight为有限且非负的JSON数值（可为0），不可用字符串或布尔值；整套权重总和为1。source_experiences为[]或仅含一个输入经验短编号字符串的数组。bindings为参数名到填充值的字符串对象；有模板时绑定所需参数，无模板或无参数时为{}；不输出绑定依据字段。",
    "judge": "criterion_judgments为数组，必须覆盖输入全部标准，每个criterion_id恰好出现一次，且只能使用输入提供的短编号字符串。satisfied只能是JSON布尔值true或false：true表示当前回答满足该标准，false表示不满足；不能写成字符串、0/1或null。reason为非空的简短判断理由，说明当前回答为何满足或不满足；不要照抄示意文字，不要求逐字摘录回答，不补造事实。",
    "critic": "analysis为非空字符串；candidates只能为[]或包含一个完整四字段方法对象的数组，不能为null，也不能超过一条。方法的trigger/principle为非空字符串；template为非空字符串或JSON null；parameters为对象，每个参数定义只含非空字符串description。",
    "experience_updater": 'action只能是小写字符串"keep"或"revise"，不能输出"keep或revise"。keep时method必须为JSON null、target_fields为[]；revise时method必须为一个完整四字段方法对象，target_fields为非空字符串数组，每项只能是"trigger"、"principle"、"template"、"parameters"之一。reason为非空的简短修订或保留理由。方法的trigger/principle为非空字符串；template为非空字符串或JSON null；parameters为对象，每个参数定义只含非空字符串description。',
}
for _role in ("selector_checker", "generator_checker", "judge_checker"):
    _VALUE_RULES[_role] = (
        'status只能是小写字符串"correct"、"error"、"unknown"之一：correct表示核对对象执行正确，error表示发现执行错误，unknown表示依据不足无法确定；不能输出"correct/error/unknown"或"failed"。'
        'evidence为非空的简短可核对依据。details为数组，可为[]；若有逐标准细项，每项criterion_id只能使用输入提供的短编号字符串，不重复，status也只能为"correct"、"error"、"unknown"之一，evidence为非空字符串。'
    )


def llm_messages(role, payload, *, history=None):
    if role not in _GUIDANCE:
        raise ValueError("unknown role " + role)
    schema = json.dumps(_SCHEMA[role], ensure_ascii=False)
    system = (
        _GUIDANCE[role]
        + "\n字段取值规则："
        + _VALUE_RULES[role]
        + "\n严格只返回以下字段的JSON，示意值需替换为实际结果："
        + schema
    )
    if role in ("critic", "experience_updater"):
        system += (
            "\n以下是独立编写的结构示例，只示范格式和选择原则，不是要求你提出单位换算经验；不得复制到不相关任务中。"
            "Template是一条可直接填参成为rubric标准的是/否核验问题，不是回答正文、固定台词或仅含占位符的答案模板。"
            "若核心核验动作可稳定写成问题，且变化的对象、范围或量值能由生成前task明确绑定，应给出Template；"
            "不为了省略而一律填null，也不为凑字段编造task没有的参数。"
            "参数定义必须是以占位符名为键的对象，每个值只包含description；"
            "错误结构是parameters直接包含description，正确结构是parameters包含参数名再嵌套description。"
            "不要输出当前任务的绑定值；绑定由Generator在实际使用时完成。"
            "\n有模板的完整经验示例："
            + json.dumps(METHOD_SCHEMA, ensure_ascii=False)
            + "\n无需模板的完整经验示例："
            + json.dumps(
                {
                    "trigger": "任务没有要求特定礼貌开场或结束语",
                    "principle": "设计标准时不要将固定礼貌套语设为任务外的必选要求",
                    "template": None,
                    "parameters": {},
                },
                ensure_ascii=False,
            )
            + "；后一例约束的是标准设计取舍，强造一条固定套语核验问题反而会引入任务外要求。"
            '无模板时使用JSON空值null，绝不能写成字符串"null"或"None"，parameters={}。'
            "模板存在但无需填参时parameters={}也合法，不必为形式强加参数。"
            "仅将task中明确可识别的对象、范围、量值作为参数；不要把回答实际出现的行为、待判断的结果、"
            "凭常识推测的约束或你临时创造的示例设为待绑定参数。"
            "若这些内容无法由task确定，应保留在通用核验动作中，或使用无模板方法，不能捏造绑定来源。"
            "模板中仅使用简单{parameter_name}占位符，参数名与parameters键严格一一对应。"
        )
    if role == "initial_generator":
        system += "\n上面的三条标准及0.6/0.25/0.15只展示共享总预算1的输出格式，不是固定数量、权重配方或通用评价维度。按本题实际需要输出2至8条标准并重新分配，总和为1；不要照搬示例问题、数量或权重。仅当重要性相当时合理等权。程序归一化只用于兜底，不能替代你的任务主次判断。"
        system += '\nsource_experiences只能为[]或["1"]这样的短编号字符串数组，最多一个元素，绝不能放经验对象或新造编号；没有适用经验就用[]，仍生成2至8条task标准。'
    if role in ("generator_checker", "judge_checker"):
        system += '\ndetails可为[]，或[{"criterion_id":"提供的短编号","status":"correct","evidence":"短依据"}]。不得使用其他字段；任一detail为error则整体status必须error；含unknown时不能整体correct。'
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
