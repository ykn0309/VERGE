"""Method-format examples are valid and do not force optional template adoption."""

import pytest
from verge.models import Method
from verge.prompts import METHOD_SCHEMA, llm_messages


def test_nonempty_example_has_nested_parameter_contract():
    method = Method.from_dict(METHOD_SCHEMA)
    assert method.template and set(method.parameters) == {
        "source_quantity",
        "target_unit",
    }
    bound = method.template.format(source_quantity="7米", target_unit="厘米")
    assert bound == "回答是否将7米正确换算为厘米？"


@pytest.mark.parametrize("role", ["critic", "experience_updater"])
def test_method_roles_see_both_examples_and_rubric_template_definition(role):
    system = llm_messages(role, {"task": "任务"})[0]["content"]
    assert "不是回答正文" in system and "无需模板的完整经验示例" in system
    assert '"source_quantity": {"description"' in system
    assert "错误结构是parameters直接包含description" in system
    assert "不得复制到不相关任务" in system
    assert "无模板时使用JSON空值null" in system
    assert "不能捏造绑定来源" in system


def test_optional_null_contract_is_preserved():
    method = Method("没有礼貌要求", "不将礼貌套语设为必选要求", None, {})
    assert Method.from_dict(method.to_dict()) == method


def test_critic_single_candidate_limit_is_enforced():
    from verge.modules import LLMCritic
    from verge.llm import FakeLLMBackend, LLMFormatError
    from verge.prompts import NumberedIDCodec
    from test_v2 import sample, rubric, judged

    class TooMany:
        def generate(self, *args, **kwargs):
            return {
                "analysis": "两个独立方法",
                "candidates": [METHOD_SCHEMA, METHOD_SCHEMA],
            }

    r = rubric()
    with pytest.raises(LLMFormatError, match="candidate limit"):
        LLMCritic(TooMany(), retries=0).propose(
            sample(), r, judged(r), 1, conversation=[], codec=NumberedIDCodec()
        )
    with pytest.raises(ValueError, match="one candidate"):
        LLMCritic(FakeLLMBackend(), max_candidates=2)


def test_shipped_configs_allow_only_one_critic_candidate():
    from pathlib import Path
    from verge.config import load_config

    for p in Path("configs").glob("*.yaml"):
        assert load_config(p)["modules"]["application"]["config"]["max_candidates"] == 1
