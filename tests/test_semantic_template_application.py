"""Natural-language methods are applied semantically, with no template contract."""

import pytest
from verge.models import Method, Criterion
from verge.modules import LLMInitialRubricGenerator
from verge.prompts import llm_messages
from test_v2 import experience


def test_generator_specializes_method_without_binding_fields():
    e = experience()
    e.method = Method("要求基于指定材料概括", "核对事实是否得到指定材料支持")

    class Backend:
        def generate(self, *args, **kwargs):
            return {
                "analysis": "任务指定材料2",
                "criteria": [
                    {
                        "question": "回答是否只包含材料2支持的事实？",
                        "weight": 0.8,
                        "source_experiences": ["1"],
                    },
                    {
                        "question": "表达是否清晰？",
                        "weight": 0.2,
                        "source_experiences": [],
                    },
                ],
            }

    result = LLMInitialRubricGenerator(Backend(), retries=0).generate("概括材料2", [e])
    assert result.criteria[0].source_experiences == [e.experience_id]
    assert "bindings" not in result.criteria[0].to_dict()


@pytest.mark.parametrize("field", ["bindings", "binding_evidence"])
def test_removed_criterion_fields_rejected(field):
    with pytest.raises(ValueError, match="unknown"):
        Criterion.from_dict(
            {"criterion_id": "1", "question": "符合？", "weight": 1, field: {}}
        )


def test_checker_semantics_and_generator_guidance():
    checker = llm_messages("generator_checker", {})[0]["content"]
    assert "核验动作遗漏" in checker and "等价改写" in checker
    generator = llm_messages("initial_generator", {})[0]["content"]
    assert "将Principle落实为具体" in generator and "bindings" not in generator


@pytest.mark.parametrize("field", ["template", "parameters"])
def test_updater_no_longer_accepts_removed_target_fields(field):
    from verge.models import ExperienceUpdate

    with pytest.raises(ValueError, match="revision field"):
        ExperienceUpdate("revise", Method("t", "p"), [field], "理由")
