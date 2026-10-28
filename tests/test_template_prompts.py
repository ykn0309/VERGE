"""Experience prompts use only applicability and one verification method."""

import pytest
from verge.models import Method
from verge.prompts import METHOD_SCHEMA, llm_messages


def test_example_has_only_trigger_and_principle():
    assert set(METHOD_SCHEMA) == {"trigger", "principle"}
    assert Method.from_dict(METHOD_SCHEMA).principle


def test_critic_reminder_is_present_on_each_turn_without_changing_payload(monkeypatch):
    from verge import prompts

    monkeypatch.setattr(prompts, "CRITIC_OUTPUT_REMINDER", "test reminder")
    payload = {"task": "original task", "_private": "not visible"}
    original = dict(payload)
    first = llm_messages("critic", payload)
    history = first + [{"role": "assistant", "content": "previous proposal"}]
    second = llm_messages("critic", {"previous_feedback": "rejected"}, history=history)
    assert first[-1]["content"].endswith("test reminder")
    assert second[:-1] == history
    assert second[-1]["content"].endswith("test reminder")
    assert "not visible" not in first[-1]["content"]
    assert payload == original
    assert (
        "test reminder" not in llm_messages("initial_generator", payload)[-1]["content"]
    )


@pytest.mark.parametrize("role", ["critic", "experience_updater"])
def test_method_roles_only_propose_two_fields(role):
    system = llm_messages(role, {})[0]["content"]
    assert "完整双字段" in system
    assert "trigger写适用条件" in system
    assert "principle写一个可独立采用的核验动作" in system
    assert "Template" not in system and "description" not in system


def test_critic_single_candidate_limit_is_enforced():
    from verge.modules import LLMCritic
    from verge.llm import FakeLLMBackend, LLMFormatError
    from verge.prompts import NumberedIDCodec
    from test_v2 import sample, rubric, judged

    class TooMany:
        def generate(self, *args, **kwargs):
            return {
                "analysis": "两个方法",
                "candidates": [METHOD_SCHEMA, METHOD_SCHEMA],
            }

    r = rubric()
    with pytest.raises(LLMFormatError, match="candidate limit"):
        LLMCritic(TooMany(), retries=0).propose(
            sample(), r, judged(r), 1, conversation=[], codec=NumberedIDCodec()
        )
    with pytest.raises(ValueError, match="one candidate"):
        LLMCritic(FakeLLMBackend(), max_candidates=2)
