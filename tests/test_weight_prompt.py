from copy import deepcopy
import pytest
from verge.prompts import _SCHEMA, llm_messages
from verge.calculator import normalize_weights
from test_v2 import rubric


def test_generator_example_shares_unit_budget_without_mandating_distinct_weights():
    criteria = _SCHEMA["initial_generator"]["criteria"]
    assert len(criteria) >= 2 and sum(c["weight"] for c in criteria) == 1
    assert not all(c["weight"] == 1 for c in criteria)
    text = llm_messages("initial_generator", {})[0]["content"]
    assert "不是每条各自分配1" in text and "仅当重要性相当时合理等权" in text
    assert "不是固定数量、权重配方" in text


@pytest.mark.parametrize("weights", [(1, 1), (0.6, 0.25), (0.2, 0.8)])
def test_normalization_fallback_preserved(weights):
    r = rubric(weights=weights)
    raw = [c.weight for c in r.criteria]
    result = normalize_weights(r)
    assert sum(result.values()) == pytest.approx(1)
    assert result["c1"] == pytest.approx(weights[0] / sum(weights))
    assert [c.weight for c in r.criteria] == raw
