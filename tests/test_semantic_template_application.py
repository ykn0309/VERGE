"""Template adherence is Checker evidence, not a generation hard gate."""

import pytest
from verge.models import Method, Parameter, Criterion
from verge.modules import LLMInitialRubricGenerator
from verge.prompts import llm_messages
from test_v2 import experience


@pytest.mark.parametrize(
    "bindings,evidence",
    [
        ({}, {}),
        ({"other": "12"}, {}),
        ({"count": "12"}, {"source": "由输入列表计数得到12项"}),
    ],
)
def test_semantic_binding_deviations_are_preserved_for_checker(bindings, evidence):
    e = experience()
    e.method = Method(
        "列出项目",
        "核对数量",
        "回答是否提供{count}项？",
        {"count": Parameter("数量")},
    )

    class Backend:
        def generate(self, *args, **kwargs):
            return {
                "analysis": "设计标准",
                "criteria": [
                    {
                        "question": "是否列出十二个项目？",
                        "weight": 0.8,
                        "source_experiences": ["1"],
                        "bindings": bindings,
                    },
                    {
                        "question": "是否清晰？",
                        "weight": 0.2,
                        "source_experiences": [],
                        "bindings": {},
                    },
                ],
            }

    c = (
        LLMInitialRubricGenerator(Backend(), retries=0)
        .generate("请提供12项", [e])
        .criteria[0]
    )
    assert c.bindings == bindings
    assert c.question == "是否列出十二个项目？"


def test_structure_types_still_rejected():
    with pytest.raises(ValueError):
        Criterion.from_dict(
            {
                "criterion_id": "1",
                "question": "达标？",
                "weight": 1,
                "source_experiences": ["e"],
                "bindings": [],
            }
        )


def test_checker_semantics_and_generator_guidance():
    checker = llm_messages("generator_checker", {})[0]["content"]
    assert "允许等价改写" in checker and "参数遗漏" in checker
    generator = llm_messages("initial_generator", {})[0]["content"]
    assert "无需逐字复现模板" in generator
