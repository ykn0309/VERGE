"""角色接口、固定场景、输出校验与数据隔离（无需网络）。"""

from dataclasses import replace
import json

import pytest

from verge.calculator import calculate_operation_contributions, calculate_revision_budget
from verge.llm import FakeLLMBackend
from verge.models import (
    CandidateExperience, EvidenceRecord, Experience,
    RevisionAttempt, Rubric,
)
from verge.modules import (
    LLMAdmissionChecker, LLMFusionJudge, LLMInitialRubricGenerator, LLMJudge,
    LLMReflector, LLMRevisionCreditAssigner, LLMRevisionGenerator, LLMSelector,
    make_fake_modules,
)
from verge.prompts import ExperienceView


TASK = "需要逐项满足关键要求的任务"
CHOSEN = "secret-chosen-response"
REJECTED = "secret-rejected-response"


def selected_experience():
    return Experience("exp_1", "为每个关键要求单设标准", "*", 2, 1,
                      ["secret-source-trajectory"],
                      [EvidenceRecord("secret-sample", "revision_credit", 0.4, 1)])


def initial_case(scenario="attempt_1_success"):
    modules = make_fake_modules(scenario)
    retrieved = modules.retriever.retrieve(TASK, [selected_experience()])
    candidates = [ExperienceView.from_experience(item) for item in retrieved]
    selected = modules.selector.select(TASK, candidates)
    initial = modules.initial_generator.generate(TASK, selected)
    judgment = modules.judge.judge(initial, CHOSEN, REJECTED)
    return modules, initial, judgment


def revise(modules, initial, initial_judgment, attempt_number, failed_attempt=None):
    feedback = modules.critic.critique(TASK, CHOSEN, REJECTED, initial,
                                      initial_judgment, failed_attempt)
    history = ({} if attempt_number == 1 else {
        "chosen": CHOSEN, "rejected": REJECTED,
        "initial_judgment": initial_judgment, "previous_attempt": failed_attempt,
    })
    intended = modules.revision_generator.generate(TASK, initial, feedback,
                                                    attempt_number, **history)
    assert all(item.before_criterion is None for item in intended)
    assert [item.target_criterion_id for item in intended] == ["c2"]
    # 阶段 06 实现真正的 Executor；当前 fixture 只补全这一条固定的 MODIFY。
    before = next(item for item in initial.criteria if item.criterion_id == "c2")
    operation = replace(intended[0], before_criterion=before)
    revised = Rubric(f"r{attempt_number}", initial.rubric_id,
                     [operation.new_or_updated_criterion if item.criterion_id == "c2" else item
                      for item in initial.criteria])
    judgment = modules.judge.judge(revised, CHOSEN, REJECTED,
                                   previous=initial, previous_judgment=initial_judgment)
    return RevisionAttempt(attempt_number, feedback, [operation], revised, judgment)


def test_initial_success_and_preference_blind_payloads():
    modules, initial, judgment = initial_case("initial_success")
    assert judgment.success and judgment.margin == pytest.approx(2 / 12)
    assert initial.criteria[0].source_experiences == ["exp_1"]
    for role, payload in modules.backend.requests:
        if role in ("selector", "initial_generator"):
            text = json.dumps(payload)
            assert CHOSEN not in text and REJECTED not in text
            assert "secret-source-trajectory" not in text
            assert "secret-sample" not in text
            views = payload["candidates"] if role == "selector" else payload["selected_experiences"]
            assert set(views[0]) == {
                "experience_id", "content", "trigger", "confidence", "evidence_amount",
            }
    with pytest.raises(ValueError, match="failed initial rubric"):
        modules.critic.critique(TASK, CHOSEN, REJECTED, initial, judgment)


def test_initial_generator_accepts_two_criteria_but_rejects_one_or_nine():
    _, initial, _ = initial_case()
    raw = initial.to_dict()
    for criterion in raw["criteria"]:
        criterion["source_experiences"] = []
    two = {**raw, "criteria": raw["criteria"][:2]}
    assert len(LLMInitialRubricGenerator(ConstantBackend({"rubric": two})).generate(TASK, []).criteria) == 2
    one = {**raw, "criteria": raw["criteria"][:1]}
    with pytest.raises(ValueError, match="invalid initial rubric"):
        LLMInitialRubricGenerator(ConstantBackend({"rubric": one})).generate(TASK, [])
    nine = {**raw, "criteria": raw["criteria"] + [
        {**raw["criteria"][0], "criterion_id": f"extra{i}", "source_experiences": []}
        for i in range(5)]}
    with pytest.raises(ValueError, match="invalid initial rubric"):
        LLMInitialRubricGenerator(ConstantBackend({"rubric": nine})).generate(TASK, [])


def test_initial_generator_retries_semantic_failure_with_only_validation_feedback():
    _, initial, _ = initial_case()
    valid = initial.to_dict()
    for item in valid["criteria"]:
        item["source_experiences"] = []
    invalid = {**valid, "criteria": valid["criteria"][:1]}

    class SequenceBackend:
        def __init__(self):
            self.payloads = []

        def generate(self, role, payload):
            assert role == "initial_generator"
            self.payloads.append(payload)
            return {"rubric": invalid if len(self.payloads) == 1 else valid}

    backend = SequenceBackend()
    result = LLMInitialRubricGenerator(backend, invalid_output_retries=2).generate(TASK, [])
    assert len(result.criteria) == len(valid["criteria"])
    assert len(backend.payloads) == 2
    assert "validation_feedback" not in backend.payloads[0]
    assert "criteria count" in backend.payloads[1]["validation_feedback"]
    assert CHOSEN not in json.dumps(backend.payloads) and REJECTED not in json.dumps(backend.payloads)
    with pytest.raises(ValueError, match="criteria count"):
        LLMInitialRubricGenerator(ConstantBackend({"rubric": invalid}),
                                  invalid_output_retries=1).generate(TASK, [])
    with pytest.raises(ValueError, match="invalid_output_retries"):
        LLMInitialRubricGenerator(backend, invalid_output_retries=True)


def test_attempt_one_success_credit_admission_and_fusion_privacy():
    modules, initial, initial_judgment = initial_case()
    assert not initial_judgment.success and initial_judgment.margin == pytest.approx(-2 / 12)
    attempt = revise(modules, initial, initial_judgment, 1)
    assert attempt.judgment.success and attempt.judgment.margin == pytest.approx(6 / 12)
    judge_requests = [payload for role, payload in modules.backend.requests if role == "judge"]
    assert len(judge_requests) == 2
    assert set(judge_requests[1]) == {"criteria", "chosen", "rejected"}
    assert [item["criterion_id"] for item in judge_requests[1]["criteria"]] == ["c2"]
    assert set(judge_requests[1]["criteria"][0]) == {"criterion_id", "question"}
    before = next(item for item in initial_judgment.criterion_judgments if item.criterion_id == "c1")
    after = next(item for item in attempt.judgment.criterion_judgments if item.criterion_id == "c1")
    assert before == after  # 同一 question 不向 LLM 重复询问
    contributions = calculate_operation_contributions(
        initial, initial_judgment, attempt.rubric, attempt.judgment, attempt.operations,
    )
    budget = calculate_revision_budget(initial, initial_judgment,
                                       attempt.rubric, attempt.judgment)
    candidates = modules.reflector.reflect(TASK, initial, initial_judgment, attempt)
    assert len(candidates) == 1 and candidates[0].credit is None
    credited = modules.revision_credit_assigner.assign(candidates, attempt.operations,
                                                        contributions, budget)
    assert credited[0].linked_operation_ids == ["op1"]
    assert credited[0].credit == pytest.approx(budget)
    assert candidates[0].credit is None  # 原输入不修改
    admitted = modules.admission_checker.check(TASK, credited[0], attempt, contributions)
    assert admitted.admission == "PASS"
    assert modules.fusion_judge.find_match(admitted.content, admitted.trigger, []) is None
    assert modules.fusion_judge.find_match(admitted.content, admitted.trigger,
                                           [("相反原则", admitted.trigger),
                                            (admitted.content, admitted.trigger)]) == 1
    role, payload = modules.backend.requests[-1]
    assert role == "fusion_judge"
    text = json.dumps(payload, ensure_ascii=False)
    for private in (CHOSEN, REJECTED, "exp_1", "secret-source-trajectory",
                    "credit", "confidence", "candidate_id", "sample_id", "operation"):
        assert private not in text
    assert set(payload) == {"candidate", "existing"}
    assert set(payload["existing"][0]) == {"content", "trigger"}
    for role, request in modules.backend.requests:
        if role in ("critic", "revision_generator", "reflector",
                    "revision_credit_assigner", "admission_checker"):
            text = json.dumps(request, ensure_ascii=False)
            assert "source_experiences" not in text and "exp_1" not in text


def test_attempt_two_still_uses_initial_rubric_and_failed_attempt_has_no_credit():
    modules, initial, judgment = initial_case("attempt_2_success")
    first = revise(modules, initial, judgment, 1)
    assert not first.judgment.success
    second = revise(modules, initial, judgment, 2, first)
    assert second.judgment.success
    assert first.rubric.base_rubric_id == second.rubric.base_rubric_id == initial.rubric_id
    assert second.operations[0].before_criterion == initial.criteria[1]
    assert second.operations[0].new_or_updated_criterion != first.operations[0].new_or_updated_criterion
    assert len(modules.reflector.reflect(TASK, initial, judgment, second)) == 1
    with pytest.raises(ValueError, match="successful revision"):
        modules.reflector.reflect(TASK, initial, judgment, first)


def test_final_failure_does_not_reflect():
    modules, initial, judgment = initial_case("final_failure")
    first = revise(modules, initial, judgment, 1)
    second = revise(modules, initial, judgment, 2, first)
    assert not first.judgment.success and not second.judgment.success
    assert all(role != "reflector" for role, _ in modules.backend.requests)
    with pytest.raises(ValueError, match="successful revision"):
        modules.reflector.reflect(TASK, initial, judgment, second)


def test_multiple_fake_candidates_share_one_operation_budget():
    modules, initial, judgment = initial_case()
    attempt = revise(modules, initial, judgment, 1)
    contributions = calculate_operation_contributions(
        initial, judgment, attempt.rubric, attempt.judgment, attempt.operations)
    budget = calculate_revision_budget(initial, judgment, attempt.rubric, attempt.judgment)
    multiple = make_fake_modules(reflect_count=2)
    reflected = multiple.reflector.reflect(TASK, initial, judgment, attempt)
    credited = multiple.revision_credit_assigner.assign(
        reflected, attempt.operations, contributions, budget)
    assert len(credited) == 2
    assert [item.credit for item in credited] == pytest.approx([budget / 2, budget / 2])
    assert all(item.linked_operation_ids == ["op1"] for item in credited)


def test_zero_candidates_unbound_and_failed_admission():
    modules, initial, judgment = initial_case()
    attempt = revise(modules, initial, judgment, 1)
    contributions = calculate_operation_contributions(
        initial, judgment, attempt.rubric, attempt.judgment, attempt.operations)
    budget = calculate_revision_budget(initial, judgment, attempt.rubric, attempt.judgment)
    empty_modules = make_fake_modules(reflect_count=0)
    assert empty_modules.reflector.reflect(TASK, initial, judgment, attempt) == []
    assert empty_modules.revision_credit_assigner.assign([], attempt.operations, contributions, budget) == []
    unbound_modules = make_fake_modules(unbound=True)
    candidate = unbound_modules.reflector.reflect(TASK, initial, judgment, attempt)
    failed = unbound_modules.revision_credit_assigner.assign(candidate, attempt.operations,
                                                             contributions, budget)
    assert failed[0].linked_operation_ids == [] and failed[0].credit is None
    with pytest.raises(ValueError, match="bound, credited"):
        unbound_modules.admission_checker.check(TASK, failed[0], attempt, contributions)
    rejected_modules = make_fake_modules(admission_pass=False)
    candidate = rejected_modules.reflector.reflect(TASK, initial, judgment, attempt)
    credited = rejected_modules.revision_credit_assigner.assign(candidate, attempt.operations,
                                                                 contributions, budget)
    refused = rejected_modules.admission_checker.check(TASK, credited[0], attempt, contributions)
    assert refused.admission == "FAIL" and refused.credit == credited[0].credit
    assert credited[0].admission is None


class ConstantBackend:
    def __init__(self, response):
        self.response = response

    def generate(self, role, payload):
        return self.response


def test_structured_response_and_domain_type_checks():
    modules, initial, judgment = initial_case()
    view = ExperienceView.from_experience(selected_experience())
    with pytest.raises(ValueError, match="structured result"):
        LLMSelector(ConstantBackend(["exp_1"])).select(TASK, [view])
    with pytest.raises(ValueError, match="list of strings"):
        LLMSelector(ConstantBackend({"selected_experience_ids": [1]})).select(TASK, [view])
    with pytest.raises(ValueError, match="projections"):
        modules.selector.select(TASK, [selected_experience()])  # type: ignore[list-item]
    bad_rubric = initial.to_dict()
    bad_rubric["criteria"][0]["importance_score"] = "2"
    with pytest.raises(ValueError, match="Criterion: expected"):
        LLMInitialRubricGenerator(ConstantBackend({"rubric": bad_rubric})).generate(TASK, [view])
    answers = [{"criterion_id": item.criterion_id, "chosen": 1, "rejected": False}
               for item in initial.criteria]
    with pytest.raises(ValueError, match="CriterionJudgment: expected"):
        LLMJudge(ConstantBackend({"criterion_judgments": answers})).judge(
            initial, CHOSEN, REJECTED)
    with pytest.raises(ValueError, match="operations must be a list"):
        LLMRevisionGenerator(ConstantBackend({"operations": "bad"})).generate(
            TASK, initial, "feedback", 1)
    operation = modules.revision_generator.generate(TASK, initial, "feedback", 1)[0].to_dict()
    operation["new_or_updated_criterion"].pop("source_experiences")
    operation["before_criterion"] = initial.criteria[1].to_dict()  # 模型来源不得覆盖 Executor 的 R0
    parsed = LLMRevisionGenerator(ConstantBackend({"operations": [operation]})).generate(
        TASK, initial, "feedback", 1)
    assert parsed[0].new_or_updated_criterion.source_experiences == []
    assert parsed[0].before_criterion is None
    attempt = revise(modules, initial, judgment, 1)
    with pytest.raises(ValueError, match="only unbound candidate"):
        LLMReflector(ConstantBackend({"candidates": [{
            "candidate_id": "x", "content": "c", "trigger": "t",
            "linked_operation_ids": ["op1"],
        }]})).reflect(TASK, initial, judgment, attempt)
    with pytest.raises(ValueError, match="match_index"):
        LLMFusionJudge(ConstantBackend({"match_index": True})).find_match("c", "t", [("c", "t")])


def test_revision_attempt_id_and_credit_range_checked_at_module_boundary():
    modules, initial, judgment = initial_case()
    attempt = revise(modules, initial, judgment, 1)
    wrong_id = replace(attempt, judgment=replace(attempt.judgment, rubric_id="wrong-rubric"))
    with pytest.raises(ValueError, match="successful revision"):
        modules.reflector.reflect(TASK, initial, judgment, wrong_id)
    candidate = CandidateExperience("x", "principle", "trigger", ["op1"], 0.2)
    with pytest.raises(ValueError, match="bound, credited"):
        modules.admission_checker.check(TASK, candidate, wrong_id, {"op1": 0.5})
    with pytest.raises(ValueError, match="bound, credited"):
        modules.admission_checker.check(TASK, replace(candidate, credit=2.0),
                                        attempt, {"op1": 0.5})
    failed_modules, failed_initial, failed_judgment = initial_case("attempt_2_success")
    failed_attempt = revise(failed_modules, failed_initial, failed_judgment, 1)
    wrong_failed = replace(failed_attempt, judgment=replace(failed_attempt.judgment,
                                                             rubric_id="wrong-rubric"))
    with pytest.raises(ValueError, match="failed revision from R0"):
        failed_modules.critic.critique(TASK, CHOSEN, REJECTED,
                                      failed_initial, failed_judgment, wrong_failed)


def test_invalid_binding_is_discarded_and_no_llm_numeric_output():
    modules, initial, judgment = initial_case()
    attempt = revise(modules, initial, judgment, 1)
    candidate = CandidateExperience("x", "principle", "trigger")
    assigner = LLMRevisionCreditAssigner(ConstantBackend({"bindings": {"x": ["not-an-operation"]}}))
    result = assigner.assign([candidate], attempt.operations, {"op1": 0.5}, 0.5)
    assert result[0].credit is None and not result[0].linked_operation_ids
    with pytest.raises(ValueError, match="PASS or FAIL"):
        LLMAdmissionChecker(ConstantBackend({"decision": "MAYBE", "reason": "不确定"})).check(
            TASK, replace(candidate, linked_operation_ids=["op1"], credit=0.2),
            attempt, {"op1": 0.5})
    roles = {role for role, _ in modules.backend.requests}
    assert "judge" in roles and "revision_generator" in roles
    for role, payload in modules.backend.requests:
        if role in ("judge", "revision_generator", "initial_generator"):
            assert "normalized_weights" not in json.dumps(payload)
