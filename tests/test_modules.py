"""New role boundaries, bounded output correction and immutable raw responses."""

import pytest
from verge.llm import FakeLLMBackend, LLMFormatError
from verge.models import CandidateExperience
from verge.modules import LLMInitialRubricGenerator, LLMCritic, LLMAdmissionChecker
from verge.prompts import ExperienceView


def test_fake_unknown_source_retries_then_exhausts_without_mutating_response():
    class Bad(FakeLLMBackend):
        def generate(self, role, payload):
            raw = super().generate(role, payload)
            if role == "initial_generator":
                raw["rubric"]["criteria"][0]["source_experiences"] = ["unknown"]
            return raw

    backend = Bad()
    gen = LLMInitialRubricGenerator(backend, invalid_output_retries=1)
    c = CandidateExperience("candidate_current", "方法", "条件", "PASS", "可用")
    with pytest.raises(ValueError, match="source_experiences"):
        gen.apply("task", [], c, 1)
    assert (
        len(backend.requests) == 2 and "validation_feedback" in backend.requests[-1][1]
    )


def test_backend_format_exhaustion_is_not_retried_by_adapter():
    class Bad(FakeLLMBackend):
        def generate(self, role, payload):
            self.requests.append((role, payload))
            raise LLMFormatError("format exhausted")

    backend = Bad()
    with pytest.raises(LLMFormatError):
        LLMInitialRubricGenerator(backend, invalid_output_retries=3).generate(
            "task", []
        )
    assert len(backend.requests) == 1


@pytest.mark.parametrize("count", [1, 9])
def test_generator_criterion_bounds_are_structural(count):
    class Bad(FakeLLMBackend):
        def generate(self, role, payload):
            raw = super().generate(role, payload)
            if role == "initial_generator":
                raw["rubric"]["criteria"] = [
                    {**raw["rubric"]["criteria"][0], "criterion_id": f"c{i}"}
                    for i in range(count)
                ]
            return raw

    with pytest.raises(ValueError, match="2..8"):
        LLMInitialRubricGenerator(Bad(), invalid_output_retries=0).generate("task", [])


def test_admission_does_not_rewrite_input_candidate():
    c = CandidateExperience("c", "具体指导", "task条件")
    b = FakeLLMBackend()
    out = LLMAdmissionChecker(b).check("task", c)
    assert c.admission is None and out.admission == "PASS"
    assert (out.content, out.trigger, out.candidate_id) == (
        c.content,
        c.trigger,
        c.candidate_id,
    )
    assert set(b.requests[0][1]) == {"task", "candidate"}


def test_missing_analysis_and_extra_old_fields_rejected():
    class Old(FakeLLMBackend):
        def generate(self, *a):
            return {"analysis": "x", "improvement_advice": "旧建议"}

    with pytest.raises(ValueError):
        from verge.models import Rubric, Criterion, CriterionJudgment
        from verge.calculator import calculate_rubric_judgment

        r = Rubric("r", [Criterion("a", "问题", 1, []), Criterion("b", "问题2", 1, [])])
        j = calculate_rubric_judgment(
            r, [CriterionJudgment(c.criterion_id, False, False) for c in r.criteria]
        )
        LLMCritic(Old()).propose("task", "A", "B", r, j, 1, 1)


@pytest.mark.parametrize(
    "bad", [1, None, "text", {}, [{"content": "x", "trigger": "*"}]]
)
def test_critic_candidate_type_failure_is_bounded_role_error(bad):
    class Bad(FakeLLMBackend):
        def generate(self, role, payload):
            self.requests.append((role, payload))
            return {"analysis": "分析", "candidates": [bad]}

    from verge.models import Rubric, Criterion, CriterionJudgment
    from verge.calculator import calculate_rubric_judgment

    r = Rubric("r", [Criterion("a", "问题", 1, []), Criterion("b", "问题2", 1, [])])
    j = calculate_rubric_judgment(
        r, [CriterionJudgment(c.criterion_id, False, False) for c in r.criteria]
    )
    b = Bad()
    with pytest.raises(ValueError):
        LLMCritic(b, invalid_output_retries=1).propose("task", "A", "B", r, j, 1, 1)
    assert len(b.requests) == 2 and "validation_feedback" in b.requests[-1][1]
