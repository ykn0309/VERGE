"""复现 docs/设计方案/13-完整示例.md 的完整数值链条。"""

from dataclasses import replace

import pytest

from verge.calculator import (
    allocate_revision_credit,
    apply_beta_evidence,
    calculate_existing_experience_credit,
    calculate_operation_contributions,
    calculate_revision_budget,
    calculate_rubric_judgment,
    normalize_weights,
)
from verge.models import (
    CandidateExperience,
    Criterion,
    CriterionJudgment,
    EvidenceRecord,
    Experience,
    RevisionOperation,
    Rubric,
)


def test_complete_example():
    c1 = Criterion("c1", "回答是否全面？", 5, ["exp_broad"])
    c2 = Criterion("c2", "回答是否清晰？", 3, [])
    c3 = Criterion("c3", "回答是否事实正确？", 2, [])
    initial = Rubric("r0", None, [c1, c2, c3])
    first = calculate_rubric_judgment(initial, [
        CriterionJudgment("c1", False, True),
        CriterionJudgment("c2", True, True),
        CriterionJudgment("c3", True, False),
    ])
    assert normalize_weights(initial) == {"c1": 0.5, "c2": 0.3, "c3": 0.2}
    assert first.margin == pytest.approx(-0.3)
    assert not first.success
    assert calculate_existing_experience_credit(initial, first) == {"exp_broad": -0.5}
    broad = Experience("exp_broad", "宽泛原则", "任务", 1, 1, [], [])
    broad = apply_beta_evidence(
        broad, EvidenceRecord("sample", "downstream_credit", -0.5, 1))
    assert broad.beta == 1.5

    # 失败的 R1 可用于诊断，但没有 revision credit；R2 仍以 R0 为 base。
    failed = Rubric("r1", "r0", [replace(c1, importance_score=3),
                                    replace(c2, importance_score=5), c3])
    failed_judgment = calculate_rubric_judgment(
        failed, [CriterionJudgment("c1", False, True),
                 CriterionJudgment("c2", True, True),
                 CriterionJudgment("c3", True, False)])
    assert failed_judgment.margin == pytest.approx(-0.1)
    assert not failed_judgment.success

    r1 = Criterion("r1", "解释过拟合？", 2, [])
    r2 = Criterion("r2", "比较误差？", 2, [])
    r3 = Criterion("r3", "缓解方法？", 1, [])
    r4 = Criterion("r4", "解释因果？", 3, [])
    revised = Rubric("r2", "r0", [r1, r2, r3, r4, c3])
    second = calculate_rubric_judgment(revised, [
        CriterionJudgment("r1", True, False),
        CriterionJudgment("r2", True, False),
        CriterionJudgment("r3", True, True),
        CriterionJudgment("r4", True, False),
        CriterionJudgment("c3", True, False),
    ])
    assert normalize_weights(revised) == {"r1": 0.2, "r2": 0.2, "r3": 0.1,
                                           "r4": 0.3, "c3": 0.2}
    assert second.margin == pytest.approx(0.9)
    assert second.success
    ops = [RevisionOperation("op_1", "DELETE", "c1", None, c1),
           RevisionOperation("op_2", "ADD", None, r1),
           RevisionOperation("op_3", "ADD", None, r2),
           RevisionOperation("op_4", "ADD", None, r3),
           RevisionOperation("op_5", "DELETE", "c2", None, c2),
           RevisionOperation("op_6", "ADD", None, r4)]
    contributions = calculate_operation_contributions(initial, first, revised, second, ops)
    assert contributions == pytest.approx({"op_1": 0.5, "op_2": 0.2, "op_3": 0.2,
                                           "op_4": 0, "op_5": 0, "op_6": 0.3})
    assert sum(abs(value) for value in contributions.values()) == pytest.approx(1.2)
    budget = calculate_revision_budget(initial, first, revised, second)
    assert budget == pytest.approx(1)
    candidates = [CandidateExperience("e1", "拆分显式要求", "多个要求",
                                      ["op_1", "op_2", "op_3", "op_4"]),
                  CandidateExperience("e2", "检查因果联系", "解释机制",
                                      ["op_5", "op_6"]),
                  CandidateExperience("e3", "加入误差变化标准", "过拟合",
                                      ["op_6"])]
    allocated = allocate_revision_credit(candidates, contributions, budget)
    assert [item.credit for item in allocated] == pytest.approx([0.75, 0.125, 0.125])
    assert sum(abs(item.credit) for item in allocated) <= budget

    e1 = replace(allocated[0], admission="PASS", fusion="MERGE", experience_id="e_old")
    e2 = replace(allocated[1], admission="PASS", fusion="ADD", experience_id="e2")
    e3 = replace(allocated[2], admission="FAIL")
    assert (e1.credit, e2.credit, e3.credit) == pytest.approx((0.75, 0.125, 0.125))
    assert sum(abs(item.credit) for item in (e1, e2)) == pytest.approx(0.875)
    assert e3.fusion is None
    old = Experience("e_old", "已有原则", "多个要求", 2.4, 1.6, [], [])
    merged = apply_beta_evidence(old, EvidenceRecord("sample", "revision_credit", e1.credit, 1))
    added = apply_beta_evidence(Experience("e2", e2.content, e2.trigger, 1, 1, [], []),
                                EvidenceRecord("sample", "revision_credit", e2.credit, 1))
    assert (merged.alpha, merged.beta) == pytest.approx((3.15, 1.6))
    assert (added.alpha, added.beta) == pytest.approx((1.125, 1))
