"""真实模型输入的数字编号映射、Judge 纠错及隔离边界。"""

import json

import pytest

from verge.llm import OpenAICompatibleBackend
from verge.models import Criterion, CriterionJudgment, Rubric
from verge.modules import (JudgeOutputError, LLMAdmissionChecker, LLMCritic,
                           LLMInitialRubricGenerator, LLMJudge, LLMSelector,
                           RevisionOperationOutputError, make_fake_modules)
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import PreferenceSample
from verge.pipeline import PipelineRunner, _check_reused_judgments
from verge.prompts import (ExperienceView, NumberedIDCodec, admission_payload,
                           binding_payload, critic_payload, reflection_payload)
from verge.models import CandidateExperience, RevisionAttempt, RevisionOperation
from verge.calculator import calculate_rubric_judgment


class ScriptedRealBackend(OpenAICompatibleBackend):
    """仅跳过 HTTP；保留真实 backend 类型，走生产环境的编号适配路径。"""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, role, payload):
        self.requests.append((role, payload))
        return next(self.responses)


def test_real_roles_only_send_numbers_and_restore_internal_ids():
    view = ExperienceView("exp_long_hash_123", "中文原则", "任务条件", 0.5, 1.0)
    backend = ScriptedRealBackend([
        {"selected_experience_ids": ["1"]},
        {"rubric": {"rubric_id": "1", "base_rubric_id": None, "criteria": [
            {"criterion_id": "1", "question": "是否符合？", "importance_score": 5,
             "source_experiences": ["1"]},
            {"criterion_id": "2", "question": "是否清楚？", "importance_score": 5,
             "source_experiences": []}]}},
        {"criterion_judgments": [
            {"criterion_id": "1", "chosen": True, "rejected": False},
            {"criterion_id": "2", "chosen": True, "rejected": False}]},
    ])
    selected = LLMSelector(backend).select("任务", [view])
    rubric = LLMInitialRubricGenerator(backend).generate("任务", selected)
    judgment = LLMJudge(backend).judge(rubric, "答案 A", "答案 B")
    assert rubric.rubric_id == "r1" and [item.criterion_id for item in rubric.criteria] == ["c1", "c2"]
    assert rubric.criteria[0].source_experiences == ["exp_long_hash_123"]
    assert judgment.success
    for _, payload in backend.requests:
        text = json.dumps(payload, ensure_ascii=False)
        assert "exp_long_hash_123" not in text and '"c1"' not in text and '"r1"' not in text
    assert backend.requests[2][1]["criteria"][0]["criterion_id"] == "1"


def test_nested_projections_and_generated_revision_ids_are_numbered():
    r0 = Rubric("rubric_extremely_long", None, [
        Criterion("criterion_12345678", "问题一", 4, []),
        Criterion("criterion_abcdefgh", "问题二", 6, []),
    ])
    j0 = calculate_rubric_judgment(r0, [
        CriterionJudgment("criterion_12345678", False, True),
        CriterionJudgment("criterion_abcdefgh", True, True),
    ])
    first_payload = critic_payload("任务", "答案 A", "答案 B", r0, j0, 1)
    codec = NumberedIDCodec("critic", first_payload)
    projected = codec.project(first_payload)
    assert [c["criterion_id"] for c in projected["initial_rubric"]["criteria"]] == ["1", "2"]
    restored_response = codec.restore({"analysis": "只对这道题的偏好做私有诊断。", "operations": [
        {"operation_id": "1", "operation_type": "MODIFY", "target_criterion_id": "1",
         "new_or_updated_criterion": {"criterion_id": "1", "question": "改善", "importance_score": 8,
                                      "source_experiences": []}},
        {"operation_id": "2", "operation_type": "ADD", "target_criterion_id": None,
         "new_or_updated_criterion": {"criterion_id": "3", "question": "新增", "importance_score": 5,
                                      "source_experiences": []}},
    ]})
    assert restored_response["analysis"] == "只对这道题的偏好做私有诊断。"
    restored = restored_response["operations"]
    assert restored[0]["operation_id"] == "op1_1"
    assert restored[0]["target_criterion_id"] == "criterion_12345678"
    assert restored[1]["new_or_updated_criterion"]["criterion_id"] == "added1_3"
    with pytest.raises(ValueError, match="unknown numbered criterion"):
        codec.restore({"operations": [{"operation_id": "1", "operation_type": "DELETE",
                                       "target_criterion_id": "9"}]})
    revised = Rubric("long_revision_id", r0.rubric_id, [
        Criterion("criterion_12345678", "改善", 8, []), r0.criteria[1],
    ])
    j1 = calculate_rubric_judgment(revised, [
        CriterionJudgment("criterion_12345678", True, False),
        CriterionJudgment("criterion_abcdefgh", True, True),
    ])
    op = RevisionOperation("operation_long_hash", "MODIFY", r0.criteria[0].criterion_id,
                           revised.criteria[0], r0.criteria[0])
    attempt = RevisionAttempt(1, "改进", [op], revised, j1)
    candidate = CandidateExperience("candidate_long_hash", "原则", "条件", [op.operation_id], 0.3)
    second_payload = critic_payload("任务", "答案 A", "答案 B", r0, j0, 2,
                                    previous_attempt=attempt)
    second_codec = NumberedIDCodec("critic", second_payload)
    second_projected = second_codec.project(second_payload)
    assert second_projected["previous_attempt"]["operations"][0]["operation_id"] == "1"
    second_op = second_codec.restore({"operations": [{
        "operation_id": "1", "operation_type": "MODIFY", "target_criterion_id": "1",
        "new_or_updated_criterion": {"criterion_id": "1", "question": "再次改进",
                                     "importance_score": 8, "source_experiences": []},
    }]})["operations"][0]
    assert second_op["operation_id"] == "op2_1"  # 上次操作也编号 1，本次独立重编。
    projections = [
        ("critic", second_payload),
        ("reflector", reflection_payload("任务", r0, j0, attempt)),
        ("revision_credit_assigner", binding_payload([candidate], [op])),
        ("admission_checker", admission_payload("任务", candidate, attempt, {op.operation_id: 1.0})),
    ]
    for role, payload in projections:
        text = json.dumps(NumberedIDCodec(role, payload).project(payload), ensure_ascii=False)
        for forbidden in ("rubric_extremely_long", "criterion_12345678", "criterion_abcdefgh",
                          "long_revision_id", "operation_long_hash", "candidate_long_hash"):
            assert forbidden not in text, (role, forbidden)
    credit_payload = projections[-2][1]
    credit_codec = NumberedIDCodec("revision_credit_assigner", credit_payload)
    projected_credit = credit_codec.project(credit_payload)
    assert projected_credit["candidates"][0]["candidate_id"] == "1"
    assert credit_codec.restore({"bindings": {"1": ["1"]}})["bindings"] == {
        "candidate_long_hash": ["operation_long_hash"]}


def test_critic_unknown_number_is_retryable_operation_error():
    initial = Rubric("r_original", None, [Criterion("c_old_1", "Q1", 5, []),
                                          Criterion("c_old_2", "Q2", 5, [])])
    judgment = calculate_rubric_judgment(initial, [
        CriterionJudgment("c_old_1", False, True),
        CriterionJudgment("c_old_2", False, True),
    ])
    backend = ScriptedRealBackend([{"analysis": "编号写错了。", "operations": [{
        "operation_id": "1", "operation_type": "DELETE", "target_criterion_id": "9",
        "new_or_updated_criterion": None,
    }]}])
    with pytest.raises(RevisionOperationOutputError, match="unknown numbered criterion") as error:
        LLMCritic(backend).critique("任务", "好", "坏", initial, judgment, 1)
    assert error.value.analysis == "编号写错了。"
    assert backend.requests[0][1]["initial_rubric"]["criteria"][0]["criterion_id"] == "1"


def test_runner_retries_real_critic_number_error_and_records_failure(tmp_path):
    good = {"analysis": "重新检查关键要求。", "operations": [{
        "operation_id": "1", "operation_type": "MODIFY", "target_criterion_id": "2",
        "new_or_updated_criterion": {"criterion_id": "2", "question": "是否逐项满足关键要求？",
                                     "importance_score": 8, "source_experiences": []},
    }]}
    bad = {"analysis": "上次操作目标引用有误。", "operations": [{
        "operation_id": "1", "operation_type": "DELETE", "target_criterion_id": "9",
        "new_or_updated_criterion": None,
    }]}
    item = PreferenceSample("sample", "任务包含明确的关键要求。", "优质回答", "较差回答", {})
    modules = make_fake_modules("attempt_1_success")
    backend = ScriptedRealBackend([bad, good])
    modules.critic = LLMCritic(backend)
    record = PipelineRunner(modules, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path)).run_sample(item)
    assert record.state == "revision_success_1"
    assert len(backend.requests) == 2
    assert "unknown numbered criterion" in backend.requests[1][1]["validation_feedback"]
    assert record.revision_attempts[0].validation_failures == ["critic: unknown numbered criterion"]
    assert record.revision_attempts[0].operations[0].target_criterion_id == "c2"
    assert "c2" not in json.dumps(backend.requests)

    failed_dir = tmp_path / "failed"
    modules = make_fake_modules("attempt_1_success")
    backend = ScriptedRealBackend([bad] * 6)
    modules.critic = LLMCritic(backend)
    log = TrajectoryLog(failed_dir)
    record = PipelineRunner(modules, ExperienceMemory(failed_dir), log).run_sample(item)
    assert record.state == "final_failure"
    assert len(backend.requests) == 6
    assert [len(attempt.validation_failures) for attempt in record.invalid_revision_attempts] == [3, 3]
    assert record.invalid_revision_attempts[0].analysis == bad["analysis"]
    assert backend.requests[3][1]["previous_invalid_attempt"]["analysis"] == bad["analysis"]
    assert log.get("sample") == record


def test_admission_checker_executes_numbered_projection():
    initial = Rubric("r_original", None, [Criterion("c_old_1", "Q1", 5, []),
                                          Criterion("c_old_2", "Q2", 5, [])])
    revised = Rubric("r_revision_long", initial.rubric_id, [
        Criterion("c_old_1", "改进", 8, []), initial.criteria[1],
    ])
    judgment = calculate_rubric_judgment(revised, [
        CriterionJudgment("c_old_1", True, False), CriterionJudgment("c_old_2", True, True),
    ])
    operation = RevisionOperation("op_internal_long", "MODIFY", "c_old_1",
                                  revised.criteria[0], initial.criteria[0])
    attempt = RevisionAttempt(1, "诊断", [operation], revised, judgment)
    candidate = CandidateExperience("candidate_internal_long", "原则", "条件",
                                    [operation.operation_id], 0.3)
    backend = ScriptedRealBackend([{"decision": "PASS", "reason": "操作支撑该原则。"}])
    checked = LLMAdmissionChecker(backend).check("任务", candidate, attempt,
                                                  {operation.operation_id: 0.5})
    assert checked.admission == "PASS" and checked.admission_reason == "操作支撑该原则。"
    payload = backend.requests[0][1]
    assert payload["candidate"]["candidate_id"] == "1"
    assert payload["candidate"]["linked_operation_ids"] == ["1"]
    assert payload["operation_contributions"] == {"1": 0.5}
    assert payload["successful_attempt"]["judgment"]["normalized_weights"].keys() == {"1", "2"}
    for internal in ("r_original", "r_revision_long", "c_old_1", "op_internal_long",
                     "candidate_internal_long"):
        assert internal not in json.dumps(payload)
    assert candidate.admission is None


def test_revision_judge_only_numbers_pending_criteria_and_reuses_old_judgment():
    initial = Rubric("r_initial_long", None, [
        Criterion("criterion_old_one", "不变", 5, []),
        Criterion("criterion_old_two", "要修改", 5, []),
    ])
    old = calculate_rubric_judgment(initial, [
        CriterionJudgment("criterion_old_one", True, False),
        CriterionJudgment("criterion_old_two", False, True),
    ])
    revised = Rubric("r_revision_long", initial.rubric_id, [
        initial.criteria[0], Criterion("criterion_old_two", "改后的问题", 5, []),
    ])
    backend = ScriptedRealBackend([{"criterion_judgments": [
        {"criterion_id": "1", "chosen": True, "rejected": False}]}])
    result = LLMJudge(backend).judge(revised, "好", "坏", previous=initial,
                                      previous_judgment=old)
    assert result.success
    assert len(backend.requests[0][1]["criteria"]) == 1
    assert backend.requests[0][1]["criteria"][0]["criterion_id"] == "1"
    assert [j.criterion_id for j in result.criterion_judgments] == [
        "criterion_old_one", "criterion_old_two"]


def test_duplicate_base_question_with_conflicting_old_judgments_does_not_abort():
    initial = Rubric("r_original", None, [Criterion("c1", "同一个问题", 5, []),
                                          Criterion("c2", "同一个问题", 5, []),
                                          Criterion("c3", "别的问题", 5, [])])
    old = calculate_rubric_judgment(initial, [
        CriterionJudgment("c1", True, False), CriterionJudgment("c2", False, True),
        CriterionJudgment("c3", False, False),
    ])
    revised = Rubric("r_new", initial.rubric_id, [initial.criteria[0], initial.criteria[1],
                     Criterion("c3", "同一个问题", 5, [])])
    backend = ScriptedRealBackend([{"criterion_judgments": [
        {"criterion_id": "1", "chosen": False, "rejected": False},
    ]}])
    new = LLMJudge(backend).judge(revised, "好", "坏", previous=initial, previous_judgment=old)
    assert [item.criterion_id for item in new.criterion_judgments] == ["c1", "c2", "c3"]
    assert len(backend.requests) == 1  # 同名旧判断矛盾，只对新增同名的 c3 请求。
    _check_reused_judgments(initial, old, revised, new)
    altered = calculate_rubric_judgment(revised, [
        CriterionJudgment("c1", False, False), new.criterion_judgments[1], new.criterion_judgments[2],
    ])
    with pytest.raises(ValueError, match="unchanged criterion question"):
        _check_reused_judgments(initial, old, revised, altered)


def test_judge_retries_mismatched_id_with_feedback_and_reuses_valid_result():
    rubric = Rubric("r_long", None, [Criterion("complex_aaa", "问题一", 5, []),
                                    Criterion("complex_bbb", "问题二", 5, [])])
    backend = ScriptedRealBackend([
        {"criterion_judgments": [{"criterion_id": "9", "chosen": True, "rejected": False}]},
        {"criterion_judgments": [{"criterion_id": "1", "chosen": True, "rejected": False},
                                 {"criterion_id": "2", "chosen": True, "rejected": False}]},
    ])
    result = LLMJudge(backend, invalid_output_retries=1).judge(rubric, "好", "坏")
    assert result.success and len(backend.requests) == 2
    assert "validation_feedback" not in backend.requests[0][1]
    assert "严格复制其数字编号" in backend.requests[1][1]["validation_feedback"]
    assert "complex_aaa" not in json.dumps(backend.requests)
    bad_backend = ScriptedRealBackend([{"criterion_judgments": [{"criterion_id": "9"}]}] * 3)
    with pytest.raises(JudgeOutputError, match="after 3 attempts"):
        LLMJudge(bad_backend).judge(rubric, "好", "坏")
    assert len(bad_backend.requests) == 3
