"""Role prompts describe alternatives separately from valid JSON examples."""

import json
import pytest
from verge.prompts import _GUIDANCE, _SCHEMA, _VALUE_RULES, llm_messages
from verge.models import ExperienceUpdate


@pytest.mark.parametrize("role", sorted(_GUIDANCE))
def test_every_role_explains_value_domains_and_preserves_json_example(role):
    system = llm_messages(role, {})[0]["content"]
    assert _VALUE_RULES[role] in system
    example = json.loads(json.dumps(_SCHEMA[role]))
    assert isinstance(example, dict)
    assert "示意值需替换为实际结果" in system


def test_judge_explains_both_boolean_values_and_reason():
    system = llm_messages("judge", {})[0]["content"]
    assert "satisfied只能是JSON布尔值true或false" in system
    assert "true表示当前回答满足该标准，false表示不满足" in system
    assert "不能写成字符串、0/1或null" in system
    assert "原文依据" not in system
    assert "简短判断理由" in system
    assert type(_SCHEMA["judge"]["criterion_judgments"][0]["satisfied"]) is bool


def test_updater_example_is_an_actual_branch_not_a_combined_enum():
    example = ExperienceUpdate.from_dict(_SCHEMA["experience_updater"])
    assert example.action == "keep" and example.method is None
    system = llm_messages("experience_updater", {})[0]["content"]
    assert 'action只能是小写字符串"keep"或"revise"' in system
    assert "target_fields为非空字符串数组" in system


@pytest.mark.parametrize(
    "role", ["selector_checker", "generator_checker", "judge_checker"]
)
def test_checker_enum_examples_are_single_valid_values(role):
    assert _SCHEMA[role]["status"] in ("correct", "error", "unknown")
    system = llm_messages(role, {})[0]["content"]
    assert '"status":"correct/error/unknown"' not in system
    assert '"correct"、"error"、"unknown"之一' in system
    assert '或"failed"' in system
    if role == "selector_checker":
        assert "details必须为[]" in system
    else:
        assert "任一detail为error则整体status必须error" in system
