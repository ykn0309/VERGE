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
    LLMCritic, LLMReflector, LLMSelector,
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
    judgment = modules.judge.judge(initial, CHOSEN, REJECTED, task="测试任务")
    return modules, initial, judgment


def revise(modules, initial, initial_judgment, attempt_number, failed_attempt=None):
    proposal = modules.critic.critique(TASK, CHOSEN, REJECTED, initial,
                                       initial_judgment, attempt_number,
                                       previous_attempt=failed_attempt)
    previous = failed_attempt.rubric if failed_attempt else initial
    previous_judgment = failed_attempt.judgment if failed_attempt else initial_judgment
    revised = modules.initial_generator.revise(previous, proposal.improvement_advice, attempt_number)
    judgment = modules.judge.judge(revised, CHOSEN, REJECTED,
                                   previous=previous, previous_judgment=previous_judgment, task="测试任务")
    return RevisionAttempt(attempt_number, proposal.improvement_advice, [], revised, judgment,
                           analysis=proposal.analysis)


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
    with pytest.raises(ValueError, match="failed current rubric"):
        modules.critic.critique(TASK, CHOSEN, REJECTED, initial, judgment, 1)


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
    assert len(judge_requests) == 4
    assert set(judge_requests[2]) == {"task", "criteria", "response"}
    assert [item["criterion_id"] for item in judge_requests[2]["criteria"]] == ["revision1_c2"]
    assert set(judge_requests[2]["criteria"][0]) == {"criterion_id", "question"}
    before = next(item for item in initial_judgment.criterion_judgments if item.criterion_id == "c1")
    after = next(item for item in attempt.judgment.criterion_judgments if item.criterion_id == "revision1_c1")
    assert (before.chosen, before.rejected) == (after.chosen, after.rejected)  # 同一 question 不向 LLM 重复询问
    candidates = modules.reflector.reflect(TASK, initial, initial_judgment, attempt, chosen='preferred response', rejected='less preferred response')
    assert len(candidates) == 1 and candidates[0].credit is None
    admitted = modules.admission_checker.check(TASK, candidates[0], attempt)
    assert admitted.admission == "PASS"
    assert modules.fusion_judge.find_match(admitted.content, admitted.trigger, []).match_index is None
    assert modules.fusion_judge.find_match(admitted.content, admitted.trigger,
                                           [("相反原则", admitted.trigger),
                                            (admitted.content, admitted.trigger)]).match_index == 1
    role, payload = modules.backend.requests[-1]
    assert role == "fusion_judge"
    text = json.dumps(payload, ensure_ascii=False)
    for private in (CHOSEN, REJECTED, "exp_1", "secret-source-trajectory",
                    "credit", "confidence", "candidate_id", "sample_id", "operation"):
        assert private not in text
    assert set(payload) == {"candidate", "existing"}
    assert set(payload["existing"][0]) == {"content", "trigger"}
    for role, request in modules.backend.requests:
        if role in ("critic", "reflector",
                    "revision_credit_assigner", "admission_checker"):
            text = json.dumps(request, ensure_ascii=False)
            assert "source_experiences" not in text and "exp_1" not in text


def test_attempt_two_still_uses_initial_rubric_and_failed_attempt_has_no_credit():
    modules, initial, judgment = initial_case("attempt_2_success")
    first = revise(modules, initial, judgment, 1)
    assert not first.judgment.success
    second = revise(modules, initial, judgment, 2, first)
    assert second.judgment.success
    assert first.rubric.base_rubric_id == initial.rubric_id
    assert second.rubric.base_rubric_id == first.rubric.rubric_id
    assert first.operations == second.operations == []
    assert len(modules.reflector.reflect(TASK, initial, judgment, second, chosen='preferred response', rejected='less preferred response')) == 1
    with pytest.raises(ValueError, match="successful revision"):
        modules.reflector.reflect(TASK, initial, judgment, first, chosen='preferred response', rejected='less preferred response')


def test_final_failure_does_not_reflect():
    modules, initial, judgment = initial_case("final_failure")
    first = revise(modules, initial, judgment, 1)
    second = revise(modules, initial, judgment, 2, first)
    assert not first.judgment.success and not second.judgment.success
    assert all(role != "reflector" for role, _ in modules.backend.requests)
    with pytest.raises(ValueError, match="successful revision"):
        modules.reflector.reflect(TASK, initial, judgment, second, chosen='preferred response', rejected='less preferred response')






class ConstantBackend:
    def __init__(self, response):
        self.response = response

    def generate(self, role, payload):
        return self.response


def test_structured_response_and_domain_type_checks():
    modules, initial, judgment = initial_case()
    view = ExperienceView.from_experience(selected_experience())
    with pytest.raises(ValueError, match="structured fields"):
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
    with pytest.raises(ValueError, match="judge: expected"):
        LLMJudge(ConstantBackend({"criterion_judgments": answers})).judge(
            initial, CHOSEN, REJECTED, task="测试任务")
    with pytest.raises(ValueError, match="structured fields"):
        LLMCritic(ConstantBackend({"analysis": "诊断", "operations": "bad"})).critique(
            TASK, CHOSEN, REJECTED, initial, judgment, 1)
    parsed = LLMCritic(ConstantBackend({"analysis": "诊断", "improvement_advice": "完善标准"})).critique(
        TASK, CHOSEN, REJECTED, initial, judgment, 1)
    assert parsed.improvement_advice == "完善标准"
    attempt = revise(modules, initial, judgment, 1)
    with pytest.raises(ValueError, match="credit"):
        LLMReflector(ConstantBackend({"candidates": [{
            "candidate_id": "x", "content": "c", "trigger": "t",
            "credit": 0.2,
        }]})).reflect(TASK, initial, judgment, attempt, chosen='preferred response', rejected='less preferred response')
    with pytest.raises(ValueError, match="match_index"):
        LLMFusionJudge(ConstantBackend({"match_index": True, "content": "c", "trigger": "t"})).find_match("c", "t", [("c", "t")])






def test_response_context_in_real_reflection_binding_messages(monkeypatch):
    import json
    import httpx
    from verge.llm import OpenAICompatibleBackend
    from verge.modules import LLMReflector
    modules, initial, j0 = initial_case()
    attempt = revise(modules, initial, j0, 1)
    monkeypatch.setenv('CONTEXT_KEY', 'fake-key')
    captured = []
    outputs = iter([{'candidates': [{'content': '按关键要求设计标准', 'trigger': '任务明确列出关键要求', }]},
                    ])
    def handler(request):
        captured.append(json.loads(request.content)['messages'])
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(next(outputs))},
                                                    'finish_reason': 'stop'}]})
    backend = OpenAICompatibleBackend(model='fake', base_url='https://test.invalid/v1',
        api_key_env='CONTEXT_KEY', client=httpx.Client(base_url='https://test.invalid/v1',
            transport=httpx.MockTransport(handler)))
    chosen = '原始优选回答：甲\n  完整缩进'
    rejected = '原始非优选回答：乙\n另一行'
    candidates = LLMReflector(backend).reflect(TASK, initial, j0, attempt,
                                               chosen=chosen, rejected=rejected)
    assert candidates[0].credit is None and candidates[0].linked_operation_ids == []
    assert len(captured) == 1
    for messages in captured:
        text = messages[-1]['content']
        assert TASK in text and chosen in text and rejected in text
        assert '资料原值' not in text and '【角色标识】' not in text


@pytest.mark.parametrize('field', ['chosen', 'rejected'])
def test_reflector_requires_nonempty_answers(field):
    modules, initial, j0 = initial_case()
    attempt = revise(modules, initial, j0, 1)
    context = {'chosen': '优选回答', 'rejected': '非优选回答'}
    context[field] = ''
    with pytest.raises(ValueError, match=field):
        modules.reflector.reflect(TASK, initial, j0, attempt, **context)


@pytest.mark.parametrize('links', [['missing'], ['op1', 'op1']])
def test_reflector_rejects_invalid_successful_operation_links(links):
    modules, initial, j0 = initial_case()
    attempt = revise(modules, initial, j0, 1)
    output = {'candidates': [{'candidate_id': 'x', 'content': '原则', 'trigger': '条件',
                              'linked_operation_ids': links}]}
    with pytest.raises(ValueError):
        LLMReflector(ConstantBackend(output)).reflect(TASK, initial, j0, attempt,
                                                     chosen=CHOSEN, rejected=REJECTED)
