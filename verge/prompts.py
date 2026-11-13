"""Role contracts for task-conditional experience learning."""

from __future__ import annotations
from copy import deepcopy
import json
from typing import Any

PROMPT_VERSION = "verge-v3.1-123.1"

_GUIDANCE = {
    "selector": """你的任务是从候选经验中，选出适合用来为当前task设计评分标准的经验。
判断时，要把任务的实际要求与经验的适用范围、必要前提逐一对照。主题相近并不代表经验适用，
也不能因为经验中提到了某个条件，就认定当前任务满足这个条件。

历史信用可以帮助你参考过去的使用情况，但这些记录可能有噪声，不能保证经验在本题也有用；
没有历史证据也不表示收益为0。请根据本题的实际依据决定选择哪些经验，最多选择max_selected条，也可以全部不选，
不必为了凑数量而选择。对每条选中的经验，说明任务中的哪些信息支持这次选择。""",
    "generator": """请为当前task设计能逐项核验的评分标准。先确认当前需要交付什么、涉及哪些对象，
以及数量、范围和格式限制；区分背景与本次要求，保留用户允许的不同实现方式。

stage为draft时，只根据task生成初稿。stage为revise时，根据task检查并修订提供的draft，
可以修复初稿的遗漏或错误，也可以保留合理内容。每条修订路径都执行同样的检查；提供的经验至多一条，
它只是帮助核验的方法，不是本题事实，也不能新增输出义务、缩小范围或把一种允许的方式变成唯一方式。
前提需要核实时，应检查前提而不是直接认定它成立；没有适用依据时不采用该方法。

完整确定标准后，按任务重要性分配有限非负权重，总和为1。每条question应是明确的通过问题，
回答满足要求时应判true。只输出标准和设计分析，不输出经验引用或使用声明。""",
    "judge": """请根据task和提供的评分标准，逐项核验当前这份回答是否满足要求。
每一项都应针对其具体标准作出判断，不能补造事实，也不要猜测哪份回答更受偏好。

对每个criterion_id返回一次判断，并在reason中说明依据，避免用占位词代替解释。
满足该项要求时，satisfied填写JSON布尔值true；不满足时填写false，不使用字符串或其他取值。""",
    "critic": """请根据完整失败轨迹，判断是否有足够依据提出一条可复用的评分标准设计经验。
先核对task、两份回答及人工偏好理由，再比较共享初稿、各路修订和Judge的原判与理由。
分析应指出具体文本依据：Selector是否忽略适用前提，Generator是否遗漏要求、增加义务、反转条件或改变评价对象，
Judge是否漏看、读错、数错或未按当前标准执行。允许多种原因并存，也允许无法可靠归因。
人工理由保留原始回答顺序及chosen/rejected映射，标注者可能不同意，必须结合原文核对；
不能因为chosen使用了某种格式或rejected缺少某个特征，就把这个差异直接当成通用必要条件。

只有能指出评分标准设计问题及可复用改进依据时，才提出候选；如果仅是Judge误判或证据不足，
请说明原因并返回空candidates。不要修改Rubric，不自行更正Judge结果来宣布成功。
每轮最多一条，候选只表达一个核心核验方法。trigger应仅由未来task判断，不能依赖回答、
未生成的Rubric或评价结果。principle说明核验关系，application_guide说明Generator怎样设计标准及必要边界。
避免把来源答案、固定写法和案例特定取值带入经验；方法必需的领域概念可以保留，可变条件应由未来任务决定。
任务未要求时不增加格式或操作义务，任务明确要求时可以提出对应的核验方法。

程序会复用同一初稿，单独提供候选进行修订和Judge验证；来源判对仅允许入池，
不证明方法正确或跨任务有效，也不积累来源信用或关联。收到明确失败轨迹后重新分析，可以拒绝继续提案。""",
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
