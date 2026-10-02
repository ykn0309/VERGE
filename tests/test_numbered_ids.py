"""真实模型输入的数字编号映射、Judge 纠错及隔离边界。"""

import json

import pytest

from verge.llm import JUDGE_RESPONSE_SIDE, OpenAICompatibleBackend
from verge.models import Criterion, CriterionJudgment, Rubric
from verge.modules import (JudgeOutputError, LLMAdmissionChecker, LLMCritic,
                           LLMInitialRubricGenerator, LLMJudge, LLMSelector,
                           make_fake_modules)
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import PreferenceSample
from verge.pipeline import PipelineRunner, _check_reused_judgments
from verge.prompts import (ExperienceView, NumberedIDCodec, admission_payload,
                           critic_payload, reflection_payload)
from verge.models import CandidateExperience, RevisionAttempt, RevisionOperation
from verge.calculator import calculate_rubric_judgment


class ScriptedRealBackend(OpenAICompatibleBackend):
    """仅跳过 HTTP；保留真实 backend 类型，走生产环境的编号适配路径。"""

    def __init__(self, responses):
        # Historical paired fixtures become separate scripts per answer side.
        from copy import deepcopy
        self.responses = iter([r for r in responses if "criterion_judgments" not in r])
        self.judge_responses = {side: iter([
            {"criterion_judgments": [
                {"criterion_id": item.get("criterion_id"), **({"satisfied": item[side]} if side in item else {})}
                for item in r["criterion_judgments"]]}
            for r in deepcopy(responses) if "criterion_judgments" in r])
            for side in ("chosen", "rejected")}
        self.requests = []

    def generate(self, role, payload):
        self.requests.append((role, payload))
        if role == "judge":
            return next(self.judge_responses[JUDGE_RESPONSE_SIDE.get()])
        result = next(self.responses)
        if role == "critic" and isinstance(result.get("operations"), list):
            result = {**result, "operations": [
                {k: v for k, v in item.items() if k not in ("operation_id", "before_criterion")}
                for item in result["operations"]]}
            for op in result["operations"]:
                criterion = op.get("new_or_updated_criterion")
                if isinstance(criterion, dict):
                    op["new_or_updated_criterion"] = {k: v for k, v in criterion.items()
                                                      if k not in ("criterion_id", "source_experiences")}
        if role == "initial_generator" and "criteria" in result:
            result = {"analysis": "根据任务分析核心要求。", **result}
        return result


def test_real_roles_only_send_numbers_and_restore_internal_ids():
    view = ExperienceView("exp_long_hash_123", "中文原则", "任务条件", 0.5, 1.0)
    backend = ScriptedRealBackend([
        {"selected_experience_ids": ["1"]},
        {"criteria": [
            {"question": "是否符合？", "importance_score": 5,
             "source_experiences": ["1"]},
            {"question": "是否清楚？", "importance_score": 5,
             "source_experiences": []}]},
        {"criterion_judgments": [
            {"criterion_id": "1", "chosen": True, "rejected": False},
            {"criterion_id": "2", "chosen": True, "rejected": False}]},
    ])
    selected = LLMSelector(backend).select("任务", [view])
    rubric = LLMInitialRubricGenerator(backend).generate("任务", selected)
    judgment = LLMJudge(backend).judge(rubric, "答案 A", "答案 B", task="测试任务")
    assert rubric.rubric_id == "r1" and [item.criterion_id for item in rubric.criteria] == ["c1", "c2"]
    assert rubric.criteria[0].source_experiences == ["exp_long_hash_123"]
    assert judgment.success
    for _, payload in backend.requests:
        text = json.dumps(payload, ensure_ascii=False)
        assert "exp_long_hash_123" not in text and '"c1"' not in text and '"r1"' not in text
    assert backend.requests[2][1]["criteria"][0]["criterion_id"] == "1"








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
                                    [], None)
    backend = ScriptedRealBackend([{"decision": "PASS", "reason": "操作支撑该原则。"}])
    checked = LLMAdmissionChecker(backend).check("任务", candidate, attempt)
    assert checked.admission == "PASS" and checked.admission_reason == "操作支撑该原则。"
    payload = backend.requests[0][1]
    assert payload["candidate"]["candidate_id"] == "1"
    assert set(payload) == {"task", "candidate"}
    assert set(payload["candidate"]) == {"candidate_id", "content", "trigger"}
    assert checked.linked_operation_ids == candidate.linked_operation_ids
    assert checked.credit == candidate.credit
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
                                      previous_judgment=old, task="测试任务")
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
    new = LLMJudge(backend).judge(revised, "好", "坏", previous=initial, previous_judgment=old, task="测试任务")
    assert [item.criterion_id for item in new.criterion_judgments] == ["c1", "c2", "c3"]
    assert len(backend.requests) == 2  # 双侧独立核验；同名旧判断矛盾，只对新增同名的 c3 请求。
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
    result = LLMJudge(backend, invalid_output_retries=1).judge(rubric, "好", "坏", task="测试任务")
    assert result.success and len(backend.requests) == 4
    assert sum("validation_feedback" in p for _, p in backend.requests) == 2
    assert all("严格复制其数字编号" in p["validation_feedback"]
               for _, p in backend.requests if "validation_feedback" in p)
    assert "complex_aaa" not in json.dumps(backend.requests)
    bad_backend = ScriptedRealBackend([{"criterion_judgments": [{"criterion_id": "9"}]}] * 3)
    with pytest.raises(JudgeOutputError, match="after 3 attempts"):
        LLMJudge(bad_backend).judge(rubric, "好", "坏", task="测试任务")
    assert len(bad_backend.requests) == 6




@pytest.mark.parametrize('role, response', [
    ('initial_generator', {"analysis": "任务分析", 'criteria': [{'criterion_id': '1', 'question': '问题',
                                         'importance_score': 5, 'source_experiences': []}]}),
    ('reflector', {'candidates': [{'candidate_id': '1', 'content': '方法', 'trigger': '条件'}]}),
])
def test_content_generation_codec_accepts_only_content_fields(role, response):
    codec = NumberedIDCodec(role, {})
    with pytest.raises(ValueError, match='content fields'):
        codec.restore(response)


def test_empty_reflection_remains_empty_after_system_numbering():
    assert NumberedIDCodec('reflector', {}).restore({'candidates': []}) == {'candidates': []}
