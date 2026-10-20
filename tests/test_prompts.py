"""Fresh role-local prompts and prospective candidate requirements."""

import pytest
from verge.prompts import (
    _ROLE_GUIDANCE,
    _ROLE_INSTRUCTIONS,
    llm_messages,
    generator_payload,
    ExperienceView,
)
from verge.models import CandidateExperience


@pytest.mark.parametrize("role", list(_ROLE_GUIDANCE))
def test_dedicated_role_instructions_and_no_shared_definition(role):
    messages = llm_messages(role, {"task": "私有task"})
    assert (
        len(messages) == 2
        and messages[0]["role"] == "system"
        and messages[1]["role"] == "user"
    )
    assert _ROLE_GUIDANCE[role] in messages[0]["content"]
    assert _ROLE_INSTRUCTIONS[role] in messages[0]["content"]
    assert (
        "私有task" not in messages[0]["content"]
        and "私有task" in messages[1]["content"]
    )
    assert messages[0]["content"].count(_ROLE_INSTRUCTIONS[role]) == 1
    assert "reflector" not in _ROLE_GUIDANCE


def test_generator_projection_no_private_context_and_equal_optional_use():
    c = CandidateExperience("candidate_x", "指导", "适用场景")
    p = generator_payload(
        "任务", [ExperienceView("e", "原指导", "条件", 0.7, 1)], candidates=[c]
    )
    assert set(p) == {"task", "selected_experiences", "candidate_experiences"}
    assert set(p["selected_experiences"][0]) == {"experience_id", "content", "trigger"}
    assert "confidence" not in p["candidate_experiences"]
    assert "admission" not in str(p)
    assert "自主采用" in _ROLE_GUIDANCE["initial_generator"]
    assert "任务要求优先" in _ROLE_GUIDANCE["initial_generator"]
    assert "缺陷问题" in _ROLE_GUIDANCE["initial_generator"]
    assert "source_experiences" in _ROLE_INSTRUCTIONS["initial_generator"]
    user = llm_messages("initial_generator", p)[1]["content"]
    assert "置信度为历史信用" not in user and "证据量" not in user
    assert "0.7" not in user


def test_critic_prospective_method_and_private_analysis():
    guidance = _ROLE_GUIDANCE["critic"]
    for text in (
        "生成前",
        "偏好身份",
        "领域知识",
        "私有analysis",
        "证据不足",
        "空候选",
    ):
        assert text in guidance
    assert "improvement_advice" not in _ROLE_INSTRUCTIONS["critic"]
    assert "candidates" in _ROLE_INSTRUCTIONS["critic"]
    assert "不依赖已有评分标准" in guidance




def test_judge_boolean_interface_and_task_independence():
    p = {
        "task": "task",
        "response": "one answer",
        "criteria": [{"criterion_id": "1", "question": "要求？"}],
    }
    messages = llm_messages("judge", p)
    assert "优选" not in str(messages) and "非优选" not in str(messages)
    assert "satisfied" in messages[0]["content"] and "布尔值" in messages[0]["content"]


def test_natural_input_preserves_raw_multiline_and_no_old_conversation_inheritance():
    p = {"task": "# 自然文本\n第二行", "selected_experiences": []}
    m = llm_messages("initial_generator", p)
    assert "# 自然文本\n第二行" in m[1]["content"]
    assert len(m) == 2


def test_generator_actual_message_unifies_e_and_current_candidate():
    candidate = CandidateExperience("candidate_current", "本轮方法", "本轮条件")
    payload = generator_payload("任务", [ExperienceView("e", "已有方法", "原条件", .7, 1)], candidates=[candidate])
    user = llm_messages("initial_generator", payload)[1]["content"]
    assert user.count("本次可用经验") == 1
    assert user.index("已有方法") < user.index("本轮方法")
    assert "当前候选、尚未验证，无历史置信度" in user
    assert "已准入" not in user and "本次选中的经验" not in user
    assert "confidence" not in payload["candidate_experiences"]
    assert set(_ROLE_GUIDANCE) == {"selector", "initial_generator", "judge", "critic", "experience_updater", "category_organizer", "atomizer", "category_describer"}


def test_role_prompts_use_scoring_standard_language():
    for guidance in _ROLE_GUIDANCE.values():
        assert "rubric" not in guidance.lower()
