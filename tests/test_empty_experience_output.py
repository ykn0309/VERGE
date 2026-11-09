"""Generator's new contract never asks for model-reported attribution."""

from copy import deepcopy
import pytest
from verge.llm import FakeLLMBackend, LLMFormatError
from verge.models import Experience, Method
from verge.modules import LLMGenerator
from verge.prompts import llm_messages, output_example


def experiences(count):
    return [
        Experience(f"e{i}", Method("改写任务", "核验原意", "按task核验"), "source")
        for i in range(count)
    ]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_generator_example_and_model_output_have_no_attribution(count):
    payload = {
        "task": "保留原意改写",
        "experiences": [e.to_dict() for e in experiences(count)],
    }
    example = output_example("generator", payload)
    assert set(example) == {"analysis", "criteria"}
    assert set(example["criteria"][0]) == {"question", "weight"}
    rubric = LLMGenerator(FakeLLMBackend()).generate(
        payload["task"], experiences(count)
    )
    assert "experience_usage" not in rubric.to_dict()
    assert "source_experiences" not in rubric.criteria[0].to_dict()


def test_legacy_attribution_is_retried_with_same_input_then_recovers():
    class Recover(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            if len(self.calls) == 1:
                result["experience_usage"] = []
            return result

    b = Recover()
    rubric = LLMGenerator(b, retries=1).generate("task", experiences(1))
    assert rubric.criteria
    assert len(b.calls) == 2
    assert b.calls[0][1]["experiences"] == b.calls[1][1]["experiences"]
    assert "不要输出经验引用" in b.calls[1][1]["validation_feedback"]


def test_invalid_contract_exhausts_bounded_budget_without_silent_cleanup():
    class Bad(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            result["criteria"][0]["source_experiences"] = []
            return result

    b = Bad()
    with pytest.raises(LLMFormatError):
        LLMGenerator(b, retries=2).generate("task", [])
    assert len(b.calls) == 3


def test_generator_rejects_more_than_two_before_calling_model():
    b = FakeLLMBackend()
    with pytest.raises(ValueError):
        LLMGenerator(b).generate("task", experiences(3))
    assert b.calls == []
