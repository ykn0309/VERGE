"""V2 role contracts, short local identifiers, and visibility projections."""

from __future__ import annotations
from typing import Any
import json

PROMPT_VERSION = "verge-compact-critic-102.3"
CRITIC_OUTPUT_REMINDER = """请像教学例一样，从本题中提炼关系而非压缩场景。候选不举例、不带题材或剧情，只保留适用条件、一个核验动作及必要边界；本题证据放analysis。"""
METHOD_SCHEMA = {
    "trigger": "任务要求根据指定材料概括内容",
    "principle": "核对概括中的事实是否得到指定材料支持，不能混入其他材料的信息",
}
_GUIDANCE = {
    "experience_checker": "你是经验质量核对员。依据task、初始失败轨迹及候选方法，检查合理性、原子性、可复用性，以及是否夹带无必要的本题人名、数值、剧情或答案。原轨迹与Judge理由可能错误，不把偏好标签当任务要求。只对明确缺陷拒绝：方法引入任务外强制要求、包含多个可独立采用的方法、把偶然细节写成通用规则，或方法要求使用生成标准前无法获得的信息。场景限定和必要边界可以保留；不因适用范围窄、措辞不完美或仅有潜在风险拒绝。不要求经验适用于所有任务。通过代表可入池试用，不保证实际有效。拒绝时reason简短指出具体缺陷和修改方向，不要求增加独立规则。",
    "selector": """你是经验适用性筛选员。只根据任务和Trigger选择有实质设计帮助的方法，最多max_selected条；不适用则不选，允许空集合。先看适用性，再参考q_posterior、后续应用与修订回放的独立证据数；未估计不是0分，高Q不能抵消不适用，回放证据不等于后续验证。""",
    "initial_generator": """你是评分标准设计者。仅根据task和本次可用经验设计2至8条Boolean标准，是表示满足。一个标准最多引用一条经验，也可不引用；仅在方法实质影响标准时引用。经验条件不是当前事实，不为区分未知回答创造任务之外的要求。整套标准共同分配总权重1，每条weight是该总预算中的份额，不是每条各自分配1。依据task中要求的重要性分配，保留主次；拆出更多标准时重新分配已有预算，不按条数放大维度。权重可以相同，但不能机械地全部填1或为制造差异强行设不同值。可同一次补充任务需要的其他标准。结合task将Principle落实为具体、可判定的标准，保留必要核验动作和边界，不增加任务外要求。""",
    "judge": """你是独立的逐标准核验员。只根据task、标准和当前一份回答逐项输出JSON布尔判断和简短判断理由，覆盖所有标准，不评价另一个回答，不猜测偏好。核对范围、否定、数量和隐含但等价的表达，不补造事实。""",
    "critic": """你为未来Generator提炼评分标准设计经验。中文JSON输出analysis与candidates；候选0或1条，仅含trigger、principle。analysis可以分析原题，经验必须独立于原题。Judge可能误判，偏好不是任务要求，无可复用方法可不提案。
方法不是原题标准的简写。先从错误中识别一种核验关系，再去掉承载该关系的人物、物品、行业活动和事件经过。trigger说明何时需要该检查，principle用一句话说明检查什么及必要边界。不举例、不用括号引文、不保留专名数字原句，不改名复述场景，也不写准确完整之类口号。
下面是独立编写的虚构教学例，不是本题，不要照搬其内容：
案例：河桥馆核验通行凭证，凭证已被撤销，回答仍依据旧记录允许通行。
要提炼的不是“核验河桥馆凭证是否撤销”，而是：
{"trigger":"任务依据会变化的状态记录作判断","principle":"核验判断所用记录是否反映当前有效状态。"}
案例：烘焙配方由一份调整到多份，只改变主料用量，辅料仍保持原值。
要提炼的不是“核验烘焙辅料用量”，而是：
{"trigger":"任务要求按比例调整多个分量","principle":"核验所有相关分量是否采用同一调整比例。"}
对真实输入也做这样的关系提炼，不给原题重新出题，不把案例细节带进经验。每字段一句短规则，trigger尽量20字、principle尽量40字，仅提示。第二轮保留真实消息，只用核对反馈与固定初始轨迹，不接候选应用后的Generator/Judge结果。""",
    "experience_updater": """你是经验修订员。根据当前固定版本、触发原因及最多6个真实案例决定Keep或Revise。原始反馈和Checker报告均可能错，不能将反向/同判机械当内容缺陷。选择反复错误优先核对Trigger；应用反复错误优先核对Principle；Judge报告错误核对歧义与可判定性，不迎合错误事实；Q风险结合内容反证。只修正当前核心方法，不追加独立规则、不拆分、不输出列表。保留已有合理方法和必要边界；明确指令仅被执行者违反或证据不足可以Keep。Revise返回恰好一个完整新方法、目标字段和短理由。""",
    "selector_checker": """你是独立适用性核对员。核对给定任务是否符合经验Trigger，不看回答偏好或其他Checker报告。仅检查当前单条经验与task的匹配，不判断多经验间依赖或冲突。判断本次选择是否合理，返回correct/error/unknown及简短可核对依据。不能把主题相关当方法适用，不凭经验效果反推适用性。""",
    "generator_checker": """你是独立应用核对员。逐项检查引用标准是否忠实执行经验的Principle、核验动作及必要边界，是否增加无依据要求或仅挂名引用。核对对象是Generator生成的criterion本身是否落实方法，不是检查答题者的回答是否达标。因此无需也不会提供实际回答，不能仅因缺少实际回答而报告unknown。只看task、经验及其引用标准，不看回答偏好、实际反馈或其他核对器结论。检查必要核验动作遗漏、无依据要求及语义偏离；允许结合task具体化、等价改写与翻译，不因措辞不同机械报错。返回整体correct/error/unknown和短证据，可在details列逐标准问题。清楚指令被错误落实应报应用错误，不能凭空认定原方法错误。""",
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
            },
            {
                "question": "根据task编写的另一项核验问题？",
                "weight": 0.25,
                "source_experiences": [],
            },
            {
                "question": "根据task编写的其余必要核验问题？",
                "weight": 0.15,
                "source_experiences": [],
            },
        ],
    },
    "judge": {
        "criterion_judgments": [
            {"criterion_id": "短编号", "satisfied": True, "reason": "简短判断理由"}
        ]
    },
    "critic": {
        "analysis": "简短事实诊断",
        "candidates": [
            {"trigger": "输入的适用条件", "principle": "一个具体的核验关系"}
        ],
    },
    "experience_updater": {
        "action": "keep",
        "method": None,
        "target_fields": [],
        "reason": "短理由；keep时method为null，revise时为完整双字段方法",
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
    "initial_generator": "analysis为字符串；criteria为2至8个标准对象。question为非空的是/否问题，true表示满足、false表示不满足。weight为有限且非负的JSON数值（可为0），不可用字符串或布尔值；整套权重总和为1。source_experiences为[]或仅含一个输入经验短编号字符串的数组。",
    "judge": "criterion_judgments为数组，必须覆盖输入全部标准，每个criterion_id恰好出现一次，且只能使用输入提供的短编号字符串。satisfied只能是JSON布尔值true或false：true表示当前回答满足该标准，false表示不满足；不能写成字符串、0/1或null。reason为非空的简短判断理由，说明当前回答为何满足或不满足；不要照抄示意文字，不要求逐字摘录回答，不补造事实。",
    "critic": "analysis为非空字符串；candidates只能为[]或包含一个完整双字段方法对象的数组，不能为null，也不能超过一条。方法的trigger/principle为非空字符串。",
    "experience_updater": 'action只能是小写字符串"keep"或"revise"，不能输出"keep或revise"。keep时method必须为JSON null、target_fields为[]；revise时method必须为一个完整双字段方法对象，target_fields为非空字符串数组，每项只能是"trigger"或"principle"之一。reason为非空的简短修订或保留理由。方法的trigger/principle为非空字符串。',
}
for _role in ("selector_checker", "generator_checker", "judge_checker"):
    _VALUE_RULES[_role] = (
        'status只能是小写字符串"correct"、"error"、"unknown"之一：correct表示核对对象执行正确，error表示发现执行错误，unknown表示依据不足无法确定；不能输出"correct/error/unknown"或"failed"。'
        'evidence为非空的简短可核对依据。details为数组，可为[]；若有逐标准细项，每项criterion_id只能使用输入提供的短编号字符串，不重复，status也只能为"correct"、"error"、"unknown"之一，evidence为非空字符串。'
    )


# Protocol100: role-specific guides and evidence-first rubric repair.
METHOD_SCHEMA.update(
    application_guide="将方法用于本题实际要求，不强加引用或固定格式",
    exclude_when=None,
    misuse=None,
)
_GUIDANCE["selector"] = (
    "你筛选可帮助本题设计标准的经验。trigger是选择指南，principle是核心核验方法；同时核对application_guide、exclude_when和misuse。主题相关不代表适用，经验条件不代表本题事实。每条选中经验说明任务依据；没有依据就不选，允许空。高Q不能抵消不适用。未知反向条件不是普遍适用，也不要求一律拒选。"
)
_SCHEMA["selector"] = {
    "selected_experience_ids": ["1"],
    "selection_reasons": [{"experience_id": "1", "reason": "本题要求及适用依据"}],
}
_VALUE_RULES[
    "selector"
] += " selection_reasons必须逐一覆盖选中ID且不重复，每项只有experience_id和非空reason；不选时两数组均为空。"
_GUIDANCE[
    "initial_generator"
] += " 对每条提供的经验逐条说明使用或未使用及原因。依据选择指南核查前提，再按application_guide应用principle；遵守已知不选条件和误用边界，不把历史条件当当前事实。经验可被舍弃，不为引用而添加要求。"
_SCHEMA["initial_generator"]["experience_usage"] = [
    {"experience_id": "1", "used": True, "reason": "对应本题哪项实际要求，怎样核验"}
]
_VALUE_RULES[
    "initial_generator"
] += " experience_usage覆盖全部输入经验且不重复，每项只有experience_id、used、reason；used只能true/false，须与criteria实际引用一致，reason非空；无经验时为[]。"
_GUIDANCE["critic_repair"] = (
    "你直接修订本题评分标准，输出完整rubric，可新增、删除、改写多项或调整权重。分析初始任务、两回答和判断依据，不把chosen标签或其措辞当作任务要求，不仅为翻转偏好任意调权重。每项修改写明变更及任务依据；允许参考上次未达门槛的尝试。本阶段不抽象经验，不输出答案。标准须覆盖任务、可判定、true表示满足，不强加格式数量或虚构前提。"
)
_SCHEMA["critic_repair"] = {
    "analysis": "具体诊断",
    "changes": [{"change": "增加事实核验", "basis": "任务要求依据材料"}],
    "criteria": [
        {
            "question": "核对当前任务要求的核心内容？",
            "weight": 0.7,
            "source_experiences": [],
        },
        {"question": "核对另一项必要要求？", "weight": 0.3, "source_experiences": []},
    ],
}
_VALUE_RULES["critic_repair"] = (
    "analysis非空；changes非空数组，每项只有非空change和basis；criteria为2..8项，每项仅question、weight、source_experiences。weight有限非负、总和1；所有source_experiences为[]，这是具体修复而非经验应用。"
)
_GUIDANCE["critic_assess"] = (
    "你结合task、两回答、rubric、Judge逐项Boolean和理由以及Judge Checker意见，作最终判定。Judge和Checker只是证据，不是裁判真值；逐项决定两回答是否满足，给简短可核对理由。不要因为chosen被偏好就推定其全部满足，不以期望margin为目标。标准本身错误须在analysis指出，不能通过曲解标准制造改善。不输出分数，程序计算。"
)
_SCHEMA["critic_assess"] = {
    "analysis": "采纳或更正的具体依据",
    "criterion_judgments": [
        {
            "criterion_id": "1",
            "chosen": True,
            "rejected": False,
            "chosen_reason": "依据",
            "rejected_reason": "依据",
        }
    ],
}
_VALUE_RULES["critic_assess"] = (
    "逐项覆盖输入标准，不增删ID；chosen/rejected只允许JSON true或false；chosen_reason/rejected_reason为非空理由。只输出analysis与criterion_judgments，不输出分数或margin。analysis说明对Judge/Checker哪些采纳或更正。"
)
_GUIDANCE["critic_extract"] = (
    "从已获整体margin改善的rubric修改中分析哪些改动真正有任务依据和可复用价值，然后提炼经验。整体成功不代表每项改动有效。允许多条，每条一个独立核验方法，不把多个方法挤在一起，不为每项操作凑经验。不得带原题专名、固定数值、原句、改名剧情或举例。principle写核心核验关系，trigger指导Selector如何核实适用，application_guide指导Generator如何落实；exclude_when和misuse没有真实证据时为null，不编造边界。仅调权重不自动产生有用经验。若无可复用关系返回空数组。"
)
_SCHEMA["critic_extract"] = {
    "analysis": "哪些修改有依据及局限",
    "candidates": [
        {"method": METHOD_SCHEMA, "change_indices": [0], "reason": "对应有效改动的依据"}
    ],
}
_VALUE_RULES["critic_extract"] = (
    "只输出analysis和candidates。analysis非空，candidates数组可为空；每项仅method、change_indices、reason。change_indices为不重复的有效修改下标整数（从0开始），至少一个。reason非空。method恰含trigger、principle、application_guide三个非空字符串以及exclude_when、misuse两个非空字符串或JSON null。不把整体margin改善称每条独立效果证明。"
)
_GUIDANCE["critic_rewrite"] = (
    "依据经验核对的具体反馈，重写当前这一条经验。保留来自有效修订的核验方法，不追加独立方法，不加入题目专名数值剧情或举例，不强造未知反向条件。本次只能返回一个method，不改rubric、不提出别的经验。"
)
_SCHEMA["critic_rewrite"] = {"method": METHOD_SCHEMA, "reason": "怎样修正明确问题"}
_VALUE_RULES["critic_rewrite"] = (
    "仅method和非空reason；method恰有trigger、principle、application_guide非空字符串，exclude_when和misuse为非空字符串或null。"
)
_GUIDANCE["experience_checker"] = (
    "检查从具体rubric有效改动提炼的经验是否忠实、合理、原子、通用及指南自洽。结合提供的修订证据，不能只看文字漂亮：不得凭空强化为必须/禁止，不把chosen措辞当要求，不带原题专名数值剧情，不合并可独立方法。未知反向条件允许null，不因适用窄或措辞不完美拒绝；可保留必要边界。原Judge和Critic分析也可能错。只拒绝明确缺陷并指出修改方向，通过仅代表初步入池，不保证泛化有效。"
)
_GUIDANCE["experience_updater"] = (
    "依据固定版本与真实案例Keep或Revise。各案例含选择/使用说明及Checker证据，均可能错误。选择问题优先检查trigger/exclude_when，应用问题检查application_guide/misuse及principle；清晰指南被忽略不必改方法，核心错误也允许纠正。反向边界从真实案例归纳，未知可null；不要机械叠加例外或加强为无依据强制要求。保留有效方法，每次只修一个完整方法，不拆分。"
)
_SCHEMA["experience_updater"]["method"] = None
_VALUE_RULES["experience_updater"] = (
    "action仅keep/revise；keep的method=null且target_fields=[]；revise的method是完整五字段方法，target_fields是实际改动字段的非空不重复数组，字段只能trigger/principle/application_guide/exclude_when/misuse；reason非空。"
)
_GUIDANCE[
    "selector_checker"
] += " 同时核对完整角色指南与本次selection_reason，理由本身不是真值。"
_GUIDANCE[
    "generator_checker"
] += " 核对application_guide、已知misuse及usage_reason是否有任务依据，不能只因理由自洽就通过。"


_GUIDANCE[
    "critic_extract"
] += " 未来Selector只能看task与经验，Generator只能看task与选中经验。trigger不能要求预先知道原rubric、两候选回答的差异、偏好标签、分数、同判或失败；这些仅放离线提炼依据。把‘修改旧标准’转成直接设计标准时可用的核验方法。可以让标准检查回答属性，但不能要求设计标准前先看到未知回答。不写例如/如式案例清单，不凭空发明比例阈值、权重或固定数量。"
_GUIDANCE[
    "experience_checker"
] += " 重点核对运行时可用信息：未来Selector/Generator看不到源rubric、两候选回答、偏好标签或历史分数；若选择条件依赖这些信息，或使用指南要求修改一个运行时不存在的旧标准，是明确缺陷，应要求改成仅凭未来task可选择、可直接用于设计标准的方法。标准可以核验回答属性，这不等于要求Generator预先知道回答。不要把源rubric/回答中的专名或数值误归到候选正文；拒绝时指出候选实际存在的问题片段。"


# Protocol102.2: diagnose Judge evidence; never request a second complete Judge pass.
_GUIDANCE[
    "critic_extract"
] += " 只提炼最重要的0至2条经验，每条一个独立方法；没有可靠方法可返回[]。不得为凑两条或规避限额将多个方法拼进一条。"
_VALUE_RULES["critic_extract"] += " candidates最多两项，超限无效。"
_GUIDANCE["experience_checker_batch"] = (
    _GUIDANCE["experience_checker"]
    + " 同次核对最多两条候选，共享修订背景但逐条判断，不让一条通过或失败替代另一条结论。候选ID必须完整覆盖，不增删、不重复。结合candidate_support的修改关联核对忠实性；候选的方法和源回答分开，不能把源rubric专名误归到经验正文。"
)
_SCHEMA["experience_checker_batch"] = {
    "reviews": [
        {"experience_id": "1", "approved": True, "reason": "这一条通过或拒绝的具体依据"}
    ]
}
_VALUE_RULES["experience_checker_batch"] = (
    "只返回reviews数组，恰好覆盖输入所有经验ID，每项仅experience_id/approved/reason；approved只能JSON true或false，reason为非空字符串，每条独立判断。"
)
_GUIDANCE["critic_rewrite_batch"] = (
    "根据每条拒绝理由重写对应经验，每条仍为一个独立、通用方法，不增加题目专名、数值、原句、剧情或任务外义务。只重写输入的未通过项，保持ID，不合并、不拆分、不新增；未知反向边界可null。共同背景只作依据，不在输出复述。"
)
_SCHEMA["critic_rewrite_batch"] = {
    "rewrites": [
        {"experience_id": "1", "method": METHOD_SCHEMA, "reason": "具体修正说明"}
    ]
}
_VALUE_RULES["critic_rewrite_batch"] = (
    "只返回rewrites，逐一覆盖给定ID且不重复；每项仅experience_id/method/reason。method完整五字段，trigger/principle/application_guide非空，exclude_when/misuse非空或null，reason非空。不返回已通过经验。"
)


_GUIDANCE["critic_repair"] = (
    "你是rubric改进者，不是第二个逐项Judge。读取task、两回答、初始rubric及Judge逐项结果和理由，分析真正问题：rubric_defect是标准缺失/过强/模糊等；judge_error是标准本身合理但Judge误读、算错或理由不成立；uncertain表示依据不足。只列发现的问题，不要求覆盖所有标准。根据rubric缺陷给出有任务依据的完整新rubric，不为迎合chosen措辞、标签或单纯翻转分数改权重。若只是Judge误判且无标准缺陷，不为了提炼经验强行改标准。初始Judge结果默认保留；只有发现明确误判才在judge_corrections列出需要更正的那一侧Boolean和具体依据，不重判或重复输出其他标准。第二次修改参考历史和反馈，但不得重写固定初始判定；只修标准。"
)
_issue = {
    "kind": "rubric_defect",
    "criterion_ids": ["1"],
    "reason": "任务依据和具体问题",
}
_correction = {
    "criterion_id": "1",
    "side": "chosen",
    "satisfied": True,
    "reason": "明确误判的原文或计算依据",
}
_SCHEMA["critic_repair"] = {
    "analysis": "区分标准缺陷、误判与不确定性",
    "issues": [_issue],
    "judge_corrections": [],
    "changes": [{"change": "针对标准缺陷的修改", "basis": "具体任务依据"}],
    "criteria": [
        {"question": "本题核心要求是否满足？", "weight": 0.7, "source_experiences": []},
        {"question": "另一必要要求是否满足？", "weight": 0.3, "source_experiences": []},
    ],
}
_diag_rules = "issues仅列发现的问题，可[]，每项只含kind/criterion_ids/reason；kind仅rubric_defect、judge_error、uncertain。criterion_ids为输入标准ID数组，首轮对应初始标准；第二轮也可引用previous_criteria给出的上一修订标准，question_index为上一轮输出criteria从0开始的位置。缺失标准可[]；reason非空。不强制给每条标准下结论。judge_corrections仅包含明确误判、确需改变的结果，可[]；每项只有criterion_id/side/satisfied/reason，side仅chosen或rejected，satisfied仅JSON true/false，理由非空；必须有对应judge_error，不重复同一标准同一侧，不输出不变结果。其他Boolean由程序沿用Judge原判。"
_VALUE_RULES["critic_repair"] = (
    _diag_rules
    + " 只返回analysis/issues/judge_corrections/changes/criteria五字段。analysis非空；changes每项只有非空change/basis。无rubric修改时changes=[]且criteria=[]；有修改时criteria为2..8条，question非空，weight有限非负且总和1，不输出source_experiences或任何来源ID，程序自动置空。round=2时judge_corrections必须[]，不能改写固定初始结果。"
)
_GUIDANCE["critic_assess"] = (
    "你分析修订后rubric的Judge结果和理由，不是重新评价所有标准。结合任务和原文，说明哪些结果可信、哪些是具体误判、哪些源于标准本身缺陷或依据不足。只列有问题项；判定不变的条目不输出新Boolean。明确误判才给对应一侧更正和依据，其他结果程序沿用Judge；不因chosen标签而更正，不为分差曲解标准。不输出分数，不复述全部输入。"
)
_SCHEMA["critic_assess"] = {
    "analysis": "基于Judge结果的判断",
    "issues": [],
    "judge_corrections": [],
}
_VALUE_RULES["critic_assess"] = (
    _diag_rules
    + " 只返回analysis/issues/judge_corrections，analysis非空。这里criterion_ids只能引用当前输入rubric。"
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
    if role == "critic":
        system += (
            "\n上面的双字段方法是独立编写的格式示例，不要求照搬到无关任务。"
            "trigger写适用条件和必要排除边界；principle写一个可独立采用的核验动作及判断依据。"
            "不要输出模板、参数、占位符或当前题目的标准答案。允许面向某类任务，不要求适用所有任务。"
        )
    if role == "initial_generator":
        system += "\n上面的三条标准及0.6/0.25/0.15只展示共享总预算1的输出格式，不是固定数量、权重配方或通用评价维度。按本题实际需要输出2至8条标准并重新分配，总和为1；不要照搬示例问题、数量或权重。仅当重要性相当时合理等权。程序归一化只用于兜底，不能替代你的任务主次判断。"
        system += '\nsource_experiences只能为[]或["1"]这样的短编号字符串数组，最多一个元素，绝不能放经验对象或新造编号；没有适用经验就用[]，仍生成2至8条task标准。'
    if role in ("generator_checker", "judge_checker"):
        system += '\ndetails可为[]，或[{"criterion_id":"提供的短编号","status":"correct","evidence":"短依据"}]。不得使用其他字段；任一detail为error则整体status必须error；含unknown时不能整体correct。'
    if role == "selector_checker":
        system += "\ndetails必须为[]，本角色只做单条经验对任务的适用性判断。"
    if role == "experience_updater":
        system += "\ntarget_fields的每项只能是trigger、principle、application_guide、exclude_when或misuse，不能填content、method、criteria、chosen等。keep用空数组；revise填写实际修改的字段。不要为追求区分创造逐字复述或固定数量要求。"
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    user = "\n\n".join(
        k + "：\n" + json.dumps(v, ensure_ascii=False, indent=2)
        for k, v in clean.items()
    )
    if role == "critic" and CRITIC_OUTPUT_REMINDER:
        user += "\n\n" + CRITIC_OUTPUT_REMINDER
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
        "criterion_ids": "criterion",
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
                if k == "normalized_weights" and isinstance(v, dict):
                    out[k] = {self._encode("criterion", cid): w for cid, w in v.items()}
                elif k in self.fields and isinstance(v, str):
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
        "selection_reason": a.selection_reason,
        "usage_reason": a.usage_reason,
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
