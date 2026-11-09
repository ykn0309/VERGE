"""Selection examples must explain both outcomes without inventing IDs."""

import json

import pytest

from verge.prompts import llm_messages, output_example


def test_nonempty_example_uses_current_id_and_explains_each_selection():
    payload = {
        "task": "原任务",
        "candidates": [{"experience_id": "7"}, {"experience_id": "12"}],
        "max_selected": 5,
    }
    messages = llm_messages("selector", payload)
    system = messages[0]["content"]
    example = json.loads(system.rsplit("：", 1)[1])
    assert example["selected_experience_ids"] == ["7"]
    assert example["selection_reasons"][0]["experience_id"] == "7"
    assert example["selection_reasons"][0]["reason"].strip()
    assert "并不表示推荐选择它" in system
    assert '{"selected_experience_ids":[],"selection_reasons":[]}' in system
    assert json.loads(messages[-1]["content"]) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"candidates": [], "max_selected": 5},
        {"candidates": [{"experience_id": "7"}], "max_selected": 0},
    ],
)
def test_no_selection_available_has_no_placeholder_ids(payload):
    assert output_example("selector", payload) == {
        "selected_experience_ids": [],
        "selection_reasons": [],
    }


def test_rendering_examples_does_not_mutate_the_next_requests_template():
    first = output_example("selector", {"candidates": [{"experience_id": "7"}]})
    first["selection_reasons"][0]["reason"] = "mutation"
    second = output_example("selector", {"candidates": [{"experience_id": "12"}]})
    assert second["selected_experience_ids"] == ["12"]
    assert second["selection_reasons"][0]["experience_id"] == "12"
    assert second["selection_reasons"][0]["reason"] != "mutation"
