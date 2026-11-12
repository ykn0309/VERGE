"""Role contracts for task-conditional experience learning."""

from __future__ import annotations
from copy import deepcopy
import json
from typing import Any

PROMPT_VERSION = "verge-subset-learning-119.1"

_GUIDANCE = {
    "selector": """你的任务是从候选经验中，选出适合用来为当前task设计评分标准的经验。
判断时，要把任务的实际要求与经验的适用范围、必要前提逐一对照。主题相近并不代表经验适用，
也不能因为经验中提到了某个条件，就认定当前任务满足这个条件。

历史信用可以帮助你参考过去的使用情况，但这些记录可能有噪声，不能保证经验在本题也有用；
没有历史证据也不表示收益为0。请根据本题的实际依据决定选择哪些经验，最多选择max_selected条，也可以全部不选，
不必为了凑数量而选择。对每条选中的经验，说明任务中的哪些信息支持这次选择。""",
    "generator": """请根据当前task设计评分标准，使每条标准都能用于判断回答是否满足任务要求。
设计时应保留任务的核心要求，并区分背景信息与需要回答完成的内容，不要把背景本身变成新的输出义务。

提供的经验用于帮助你确定核验方法，而不是补充本题事实或给任务增加要求。请结合当前任务判断是否使用这些方法；
如果任务不支持某条经验的前提，可以不使用。请专注于标准本身，不输出经验引用或逐条使用声明。

每条question都应是含义明确的核验问题，回答满足该项要求时应判为true。
请按各项要求在任务中的重要程度分配权重，权重须为有限非负数，总和为1，不要依赖程序归一化来代替合理分配。""",
    "judge": """请根据task和提供的评分标准，逐项核验当前这份回答是否满足要求。
每一项都应针对其具体标准作出判断，不能补造事实，也不要猜测哪份回答更受偏好。

对每个criterion_id返回一次判断，并在reason中说明依据，避免用占位词代替解释。
满足该项要求时，satisfied填写JSON布尔值true；不满足时填写false，不使用字符串或其他取值。""",
    "critic": """你的任务是分析失败轨迹，并直接提出可供未来Generator使用的评分标准设计经验。
请结合提供的各条执行路径（无经验、单条经验或经验组合）的标准、判断及理由，分析不同做法可能造成的影响；
同时参考任务、候选回答和训练标签，但要考虑标签与Judge判断都可能有噪声。
这些分析是提出候选的依据，你不需要修改Rubric，也不能自行更正Judge结果来宣布修复成功。

你可以提出新的方法，也可以参考已有经验改写；改写后的内容同样作为一条独立的新经验。
每条候选只围绕一个核心核验方法展开，让适用范围、方法本身和生成指南相互对应。
trigger必须能仅凭未来任务的task判断，不能依赖候选回答、模型是否犯错、尚未生成的Rubric或评分结果。
application_guide应说明Generator如何把方法落实到评分标准中，而不是替答题者规定固定措辞、格式或操作，
也不能把用户允许的做法改成必须遵守的要求。没有证据支持时，不要编造不适用条件。

经验应当能复用于一类任务，而不是复述当前案例。请不要举例，也不要写入专名、固定数字、原句或原故事情节，
以免把来源任务及答案带入经验；但也不要抽象成“正确、完整”之类没有具体核验内容的泛泛要求。

最多提出max_candidates条候选；如果没有足够依据，可以不提出。
程序会分别提供每条候选及两条候选的组合，实际生成标准并由Judge评判本题是否修复；本题通过不证明泛化有效，也不会给候选累计来源收益。
如果收到失败反馈，请根据返回的实际轨迹重新分析，而不是因为输出格式合法就认定候选有效。""",
}
_SCHEMA = {
    "selector": {
        "selected_experience_ids": ["提供的经验编号"],
        "selection_reasons": [
            {
                "experience_id": "提供的经验编号",
                "reason": "说明本题哪些信息支持该经验的适用条件，而不是只说主题相近。",
            }
        ],
    },
    "generator": {
        "analysis": "设计依据",
        "criteria": [
            {
                "question": "本题明确要求的核验问题？",
                "weight": 1.0,
            }
        ],
    },
    "judge": {
        "criterion_judgments": [
            {"criterion_id": "提供的编号", "satisfied": True, "reason": "具体判断理由"}
        ]
    },
    "critic": {
        "analysis": "基于轨迹的分析",
        "candidates": [
            {
                "method": {
                    "trigger": "适用条件",
                    "principle": "一个核验关系",
                    "application_guide": "如何落实，必要边界是什么",
                },
                "parent_ids": [],
            }
        ],
    },
}
_RULES = {
    "selector": (
        "请将选中的经验编号写入selected_experience_ids，编号必须来自candidates，不能重复，最多max_selected条。"
        "在selection_reasons中为每个选中编号填写一条说明，包含同一experience_id和非空的reason；"
        "不要遗漏选中项，也不要为未选中的经验填写说明。"
        '如果没有适用经验，返回{"selected_experience_ids":[],"selection_reasons":[]}。'
        "下面仅展示选中一条时的字段结构，示例中的编号取自本次候选，并不表示推荐选择它；"
        "可以选择多条，也可以不选，数量和理由必须由当前任务决定。"
    ),
    "generator": (
        "仅返回analysis和criteria。analysis为非空字符串；criteria为1至8条标准，"
        "每条仅question和weight，question为非空字符串，weight为有限非负数且总和为1。"
        "不要返回经验编号、引用或使用声明。示例展示字段结构，实际问题和权重应根据task确定。"
    ),
    "judge": (
        "仅criterion_judgments数组，每项仅criterion_id、satisfied、reason；"
        "satisfied仅true/false，reason非空。"
    ),
    "critic": (
        "仅analysis和candidates。"
        "analysis非空；"
        "每个候选仅method和parent_ids。"
        "method恰好trigger、principle、application_guide三个非空字符串。"
        "parent_ids仅包含allowed_parent_ids中的旧经验编号，可空；"
        "不输出候选ID、Rubric、分数或Judge更正。"
    ),
}


def output_example(role, payload):
    """Show complete ID relationships using only this request's real IDs."""
    example = deepcopy(_SCHEMA[role])
    if role == "selector":
        candidates = payload.get("candidates", [])
        if not candidates or payload.get("max_selected") == 0:
            return {"selected_experience_ids": [], "selection_reasons": []}
        experience_id = candidates[0]["experience_id"]
        example["selected_experience_ids"] = [experience_id]
        example["selection_reasons"][0]["experience_id"] = experience_id
    return example


def llm_messages(role, payload, *, history=None):
    if role not in _GUIDANCE:
        raise ValueError("unknown role " + role)
    system = (
        _GUIDANCE[role]
        + "\n字段约束："
        + _RULES[role]
        + "\n严格只返回以下字段的JSON，示意值必须替换："
        + json.dumps(output_example(role, payload), ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system},
        *(history or []),
        {
            "role": "user",
            "content": json.dumps(
                {k: v for k, v in payload.items() if not k.startswith("_")},
                ensure_ascii=False,
            ),
        },
    ]


class NumberedIDCodec:
    """Stable within a Critic conversation; independent for every other request."""

    fields = {
        "experience_id": "experience",
        "sample_id": "sample",
        "task_id": "task",
        "criterion_id": "criterion",
        "rubric_id": "rubric",
    }
    lists = {
        "selected_experience_ids": "experience",
        "provided_experience_ids": "experience",
        "admission_subset": "experience",
        "actual_experience_ids": "experience",
        "parent_ids": "experience",
        "allowed_parent_ids": "experience",
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
