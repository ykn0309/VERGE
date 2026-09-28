"""方法模块契约和 LLM 角色适配器。

适配器负责限定各角色可见的输入、校验结构化输出；数值计算委托 Calculator。
将相同接口接到 FakeLLMBackend 即可离线运行，不需要另写一套假模块。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, sqrt
from typing import Any, Mapping, Protocol, Sequence

from verge.calculator import allocate_revision_credit, calculate_rubric_judgment
from verge.llm import FakeLLMBackend, FakeScenario, LLMBackend
from verge.models import (
    CandidateExperience, CriterionJudgment, Experience, RevisionAttempt,
    RevisionOperation, Rubric, RubricJudgment,
)
from verge.prompts import (
    ExperienceView, admission_payload, binding_payload, critic_payload, fusion_payload,
    generator_payload, judge_payload, reflection_payload, revision_payload, selector_payload,
)


# 检索、筛选和初始生成不接收 preference pair；只传递 task/经验投影。
class Retriever(Protocol):
    """广召回：输入 task 和经验池，输出候选经验；不得读取偏好回答。"""

    def retrieve(self, task: str, experiences: Sequence[Experience]) -> list[Experience]: ...


class Selector(Protocol):
    """经验筛选：根据 task 和召回候选的公开视图，选出适用经验；不见偏好回答。"""

    def select(self, task: str, candidates: Sequence[ExperienceView], *,
               training: bool = True) -> list[ExperienceView]: ...


class InitialRubricGenerator(Protocol):
    """初始标准生成：只根据 task 和选中经验生成 R0（含问题、分数和来源）。"""

    def generate(self, task: str, selected: Sequence[ExperienceView]) -> Rubric: ...


# Judge 及其后续角色才接触偏好对和修订结果。
class Judge(Protocol):
    """标准评价：读取偏好对，逐项产生 Boolean 判断并计算完整 RubricJudgment。"""

    def judge(self, rubric: Rubric, chosen: str, rejected: str, *,
              previous: Rubric | None = None,
              previous_judgment: RubricJudgment | None = None) -> RubricJudgment: ...


class Critic(Protocol):
    """失败诊断：基于 R0、Judge 结果及可选的失败尝试，输出修订反馈文本。"""

    def critique(self, task: str, chosen: str, rejected: str, initial: Rubric,
                 initial_judgment: RubricJudgment,
                 failed_attempt: RevisionAttempt | None = None) -> str: ...


class RevisionGenerator(Protocol):
    """修订意图：基于 R0 和反馈提出 ADD/DELETE/MODIFY，不负责执行操作。"""

    def generate(self, task: str, initial: Rubric, feedback: str,
                 attempt_number: int) -> list[RevisionOperation]: ...


class Reflector(Protocol):
    """经验反思：仅从成功修订提炼 0–3 条候选，不做绑定或信用计算。"""

    def reflect(self, task: str, initial: Rubric, initial_judgment: RubricJudgment,
                successful_attempt: RevisionAttempt) -> list[CandidateExperience]: ...


class RevisionCreditAssigner(Protocol):
    """修订信用：将候选绑定到已执行操作，再按收益和预算分配暂定信用。"""

    def assign(self, candidates: Sequence[CandidateExperience],
               operations: Sequence[RevisionOperation],
               operation_contributions: Mapping[str, float],
               budget: float) -> list[CandidateExperience]: ...


class AdmissionChecker(Protocol):
    """准入审核：审查已绑定且带信用的候选，只写入 PASS/FAIL 决策。"""

    def check(self, task: str, candidate: CandidateExperience,
              successful_attempt: RevisionAttempt,
              operation_contributions: Mapping[str, float]) -> CandidateExperience: ...


class FusionJudge(Protocol):
    """经验融合判断：仅比较 content/trigger，返回等价经验的位置或 None。"""

    def find_match(self, content: str, trigger: str,
                   existing: Sequence[tuple[str, str]]) -> int | None: ...


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _views(items: Sequence[ExperienceView]) -> None:
    """强制使用白名单投影，避免把 Experience 的轨迹/证据字段送进 prompt。"""
    if any(type(item) is not ExperienceView for item in items):
        raise ValueError("selected/retrieved experiences must be ExperienceView projections")


def _output(backend: LLMBackend, role: str, payload: dict[str, Any],
            field: str) -> Any:
    """所有角色共享的结构校验；字段值仍由对应模块按领域模型校验。"""
    response = backend.generate(role, payload)
    if not isinstance(response, Mapping) or set(response) != {field}:
        raise ValueError(f"{role}: expected a structured result containing only {field}")
    return response[field]


class FakeRetriever:
    """稳定取前 L 条，供离线测试使用；不声称提供向量相似度。"""

    def __init__(self, top_l: int = 20) -> None:
        if type(top_l) is not int or top_l != 20:
            raise ValueError("V1 retrieval top_l must be 20")
        self.top_l = top_l

    def retrieve(self, task: str, experiences: Sequence[Experience]) -> list[Experience]:
        _text(task, "task")
        return list(experiences[:self.top_l])


class EmbeddingBackend(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class VectorRetriever:
    """进程内缓存经验文本向量，用余弦相似度从 Memory 广召回 Top 20。"""

    def __init__(self, backend: EmbeddingBackend, top_l: int = 20) -> None:
        if type(top_l) is not int or top_l != 20:
            raise ValueError("V1 retrieval top_l must be 20")
        self.backend, self.top_l = backend, top_l
        self._vectors: dict[str, tuple[tuple[str, str], tuple[float, ...]]] = {}

    @staticmethod
    def _unit(vector: tuple[float, ...]) -> tuple[float, ...]:
        if (not isinstance(vector, (list, tuple)) or not vector
                or any(type(x) not in (int, float) or not isfinite(x) for x in vector)):
            raise ValueError("retrieval requires a finite embedding vector")
        norm = sqrt(sum(x * x for x in vector))
        if not isfinite(norm) or norm == 0:
            raise ValueError("retrieval requires a nonzero embedding vector")
        return tuple(x / norm for x in vector)

    def retrieve(self, task: str, experiences: Sequence[Experience]) -> list[Experience]:
        _text(task, "task")
        if not experiences:
            return []
        if len({item.experience_id for item in experiences}) != len(experiences):
            raise ValueError("retrieval requires unique experience IDs")
        query = self._unit(self.backend.embed(task))
        scored: list[tuple[float, str, Experience]] = []
        for item in experiences:
            text = (item.content, item.trigger)
            cached = self._vectors.get(item.experience_id)
            if cached is None or cached[0] != text:
                vector = self._unit(self.backend.embed(f"{item.content}\n{item.trigger}"))
                self._vectors[item.experience_id] = (text, vector)
            else:
                vector = cached[1]
            if len(query) != len(vector):
                raise ValueError("retrieval embedding dimensions disagree")
            scored.append((sum(a * b for a, b in zip(query, vector)), item.experience_id, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [row[2] for row in scored[:self.top_l]]


class LLMSelector:
    """让 LLM 返回召回候选的 ID；不允许凭空选出未召回的经验。"""

    def __init__(self, backend: LLMBackend, *, max_selected: int = 5) -> None:
        if type(max_selected) is not int or not 0 <= max_selected <= 5:
            raise ValueError("max_selected must be an integer in [0, 5]")
        self.backend = backend
        self.max_selected = max_selected

    def select(self, task: str, candidates: Sequence[ExperienceView], *,
               training: bool = True) -> list[ExperienceView]:
        _text(task, "task")
        _views(candidates)
        if type(training) is not bool:
            raise ValueError("training must be Boolean")
        if not candidates or not self.max_selected:
            return []
        identifiers = _output(self.backend, "selector", selector_payload(
            task, candidates, max_selected=self.max_selected, training=training,
        ), "selected_experience_ids")
        if not isinstance(identifiers, list) or any(type(key) is not str for key in identifiers):
            raise ValueError("selector: selected_experience_ids must be a list of strings")
        by_id = {item.experience_id: item for item in candidates}
        if (len(by_id) != len(candidates) or len(identifiers) > self.max_selected
                or len(set(identifiers)) != len(identifiers)
                or any(key not in by_id for key in identifiers)):
            raise ValueError("selector: selected experiences must be distinct retrieved candidates")
        return [by_id[key] for key in identifiers]


class LLMInitialRubricGenerator:
    """生成 R0；可对输出校验失败进行有限次数的完整重新生成。"""

    def __init__(self, backend: LLMBackend, *, invalid_output_retries: int = 0) -> None:
        if type(invalid_output_retries) is not int or not 0 <= invalid_output_retries <= 3:
            raise ValueError("invalid_output_retries must be an integer in [0, 3]")
        self.backend = backend
        self.invalid_output_retries = invalid_output_retries

    def generate(self, task: str, selected: Sequence[ExperienceView]) -> Rubric:
        _text(task, "task")
        _views(selected)
        allowed = {item.experience_id for item in selected}
        feedback = None
        for retry in range(self.invalid_output_retries + 1):
            raw = _output(self.backend, "initial_generator", generator_payload(
                task, selected, validation_feedback=feedback,
            ), "rubric")
            try:
                rubric = Rubric.from_dict(raw)
                errors = []
                if rubric.base_rubric_id is not None:
                    errors.append("base_rubric_id must be null")
                if not 2 <= len(rubric.criteria) <= 8:
                    errors.append("criteria count must be in [2, 8]")
                if any(len(set(item.source_experiences)) != len(item.source_experiences)
                       or set(item.source_experiences) - allowed for item in rubric.criteria):
                    errors.append("source_experiences must be distinct selected IDs")
                for criterion in rubric.criteria:
                    _text(criterion.question, "criterion question")
                if errors:
                    raise ValueError("initial_generator: invalid initial rubric or source_experiences: "
                                     + "; ".join(errors))
                return rubric
            except ValueError as exc:
                if retry == self.invalid_output_retries:
                    raise
                # 不回传原始生成文本或 preference pair，只告知验证约束。
                feedback = f"上次完整 R0 未通过校验：{str(exc)[:160]}。请重新生成 2 到 8 条标准；base_rubric_id 为 null，source_experiences 只引用所选经验且不得重复。"
        raise AssertionError("unreachable retry state")


class LLMJudge:
    """LLM 只判断 Boolean；权重、margin 和 success 均交给 Calculator。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def judge(self, rubric: Rubric, chosen: str, rejected: str, *,
              previous: Rubric | None = None,
              previous_judgment: RubricJudgment | None = None) -> RubricJudgment:
        if type(chosen) is not str or type(rejected) is not str:
            raise ValueError("chosen and rejected must be strings")
        if (previous is None) != (previous_judgment is None):
            raise ValueError("previous rubric and judgment must be supplied together")
        known: dict[str, CriterionJudgment] = {}
        if previous is not None and previous_judgment is not None:
            if (rubric.base_rubric_id != previous.rubric_id
                    or previous_judgment.rubric_id != previous.rubric_id):
                raise ValueError("previous judgment must belong to the revision base")
            old = {item.criterion_id: item for item in previous.criteria}
            old_answers = {item.criterion_id: item for item in previous_judgment.criterion_judgments}
            if set(old_answers) != set(old) or len(old_answers) != len(previous_judgment.criterion_judgments):
                raise ValueError("previous judgment does not cover the base rubric")
            # 问题未变时沿用 R0 判断；即使重要性分数变化也不重新询问 Judge。
            known = {item.criterion_id: old_answers[item.criterion_id]
                     for item in rubric.criteria if item.criterion_id in old
                     and item.question == old[item.criterion_id].question}
        # 后端仅回答待判断标准，返回值必须恰好覆盖这些 ID。
        pending = [item for item in rubric.criteria if item.criterion_id not in known]
        generated: list[CriterionJudgment] = []
        if pending:
            raw = _output(self.backend, "judge", judge_payload(pending, chosen, rejected),
                          "criterion_judgments")
            if not isinstance(raw, list):
                raise ValueError("judge: criterion_judgments must be a list")
            generated = [CriterionJudgment.from_dict(item) for item in raw]
        answers = {item.criterion_id: item for item in generated}
        requested = {item.criterion_id for item in pending}
        if len(answers) != len(generated) or set(answers) != requested:
            raise ValueError("judge: judgments must cover exactly the requested criteria "
                             f"(missing={len(requested - answers.keys())}, "
                             f"extra={len(answers.keys() - requested)}, "
                             f"duplicate={len(generated) - len(answers)})")
        return calculate_rubric_judgment(
            rubric, [known.get(item.criterion_id) or answers[item.criterion_id]
                     for item in rubric.criteria],
        )


class LLMCritic:
    """根据失败原因给修订器反馈；第二次尝试仍追溯同一个 R0。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def critique(self, task: str, chosen: str, rejected: str, initial: Rubric,
                 initial_judgment: RubricJudgment,
                 failed_attempt: RevisionAttempt | None = None) -> str:
        _text(task, "task")
        if type(chosen) is not str or type(rejected) is not str:
            raise ValueError("chosen and rejected must be strings")
        if initial_judgment.rubric_id != initial.rubric_id or initial_judgment.success:
            raise ValueError("critic requires a failed initial rubric")
        if (failed_attempt is not None and
                (failed_attempt.attempt_number != 1 or failed_attempt.judgment.success
                 or failed_attempt.rubric.base_rubric_id != initial.rubric_id
                 or failed_attempt.judgment.rubric_id != failed_attempt.rubric.rubric_id)):
            raise ValueError("critic second attempt requires a failed revision from R0")
        feedback = _output(self.backend, "critic", critic_payload(
            task, chosen, rejected, initial, initial_judgment, failed_attempt,
        ), "feedback")
        _text(feedback, "critic feedback")
        return feedback


class LLMRevisionGenerator:
    """只生成操作意图；before_criterion 和修订后 rubric 由 Executor 补全。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def generate(self, task: str, initial: Rubric, feedback: str,
                 attempt_number: int) -> list[RevisionOperation]:
        _text(task, "task")
        _text(feedback, "feedback")
        if initial.base_rubric_id is not None or type(attempt_number) is not int or attempt_number not in (1, 2):
            raise ValueError("revision_generator requires R0 and attempt 1 or 2")
        raw = _output(self.backend, "revision_generator", revision_payload(
            task, initial, feedback, attempt_number,
        ), "operations")
        if not isinstance(raw, list):
            raise ValueError("revision_generator: operations must be a list")
        result = []
        for item in raw:
            if isinstance(item, dict):
                # before_criterion 仅由 Executor 从 R0 补全，不能信任模型提供的旧标准。
                item = {**item, "before_criterion": None}
                if isinstance(item.get("new_or_updated_criterion"), dict):
                    # 修订不读取来源：模型提供的 source_experiences 不参与归因。
                    item["new_or_updated_criterion"] = {
                        **item["new_or_updated_criterion"], "source_experiences": [],
                    }
            result.append(RevisionOperation.from_dict(item))
        for item in result:
            if item.new_or_updated_criterion is not None:
                _text(item.new_or_updated_criterion.question, "revision criterion question")
        return result


class LLMReflector:
    """只从成功修订提取未绑定、未计算信用的候选经验。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def reflect(self, task: str, initial: Rubric, initial_judgment: RubricJudgment,
                successful_attempt: RevisionAttempt) -> list[CandidateExperience]:
        _text(task, "task")
        if (initial_judgment.rubric_id != initial.rubric_id or initial_judgment.success
                or not successful_attempt.judgment.success
                or successful_attempt.rubric.base_rubric_id != initial.rubric_id
                or successful_attempt.judgment.rubric_id != successful_attempt.rubric.rubric_id):
            raise ValueError("reflector requires a successful revision of a failed initial rubric")
        raw = _output(self.backend, "reflector", reflection_payload(
            task, initial, initial_judgment, successful_attempt,
        ), "candidates")
        if not isinstance(raw, list) or len(raw) > 3:
            raise ValueError("reflector: expected 0..3 candidates")
        result = [CandidateExperience.from_dict(item) for item in raw]
        if (len({item.candidate_id for item in result}) != len(result)
                or any(item.linked_operation_ids or item.credit is not None
                       or item.admission is not None or item.fusion is not None
                       or item.experience_id is not None for item in result)):
            raise ValueError("reflector must output only unbound candidate content and trigger")
        for item in result:
            _text(item.content, "candidate content")
            _text(item.trigger, "candidate trigger")
        return result


class LLMRevisionCreditAssigner:
    """LLM 只绑定语义；Calculator 计算并写入暂定信用。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def assign(self, candidates: Sequence[CandidateExperience],
               operations: Sequence[RevisionOperation],
               operation_contributions: Mapping[str, float],
               budget: float) -> list[CandidateExperience]:
        keys = [item.operation_id for item in operations]
        if len(set(keys)) != len(keys) or set(keys) != set(operation_contributions):
            raise ValueError("credit assigner requires contributions for exactly these operations")
        if not candidates:
            return []
        bindings = _output(self.backend, "revision_credit_assigner", binding_payload(
            candidates, operations,
        ), "bindings")
        if not isinstance(bindings, dict) or set(bindings) != {item.candidate_id for item in candidates}:
            raise ValueError("revision_credit_assigner: bindings must cover all candidate IDs")
        bound: list[CandidateExperience] = []
        for item in candidates:
            ids = bindings[item.candidate_id]
            if not isinstance(ids, list) or any(type(key) is not str for key in ids):
                raise ValueError("revision_credit_assigner: operation IDs must be strings")
            # 无效引用视为该 candidate 无法绑定；绝不凭空归因。
            selected = ids if len(set(ids)) == len(ids) and set(ids) <= set(keys) else []
            bound.append(replace(item, linked_operation_ids=selected))
        # 绑定失败的候选保留空绑定；其预算不会重新分配给其他候选。
        return allocate_revision_credit(bound, dict(operation_contributions), budget)


class LLMAdmissionChecker:
    """只裁决 PASS/FAIL，不改写候选内容、绑定或暂定信用。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def check(self, task: str, candidate: CandidateExperience,
              successful_attempt: RevisionAttempt,
              operation_contributions: Mapping[str, float]) -> CandidateExperience:
        _text(task, "task")
        operation_ids = {item.operation_id for item in successful_attempt.operations}
        if (candidate.credit is None or not -1 <= candidate.credit <= 1
                or not candidate.linked_operation_ids or candidate.admission is not None
                or not successful_attempt.judgment.success
                or successful_attempt.judgment.rubric_id != successful_attempt.rubric.rubric_id
                or not set(candidate.linked_operation_ids) <= operation_ids
                or set(operation_contributions) != operation_ids):
            raise ValueError("admission requires bound, credited candidate and successful revision")
        response = self.backend.generate("admission_checker", admission_payload(
            task, candidate, successful_attempt, dict(operation_contributions),
        ))
        if (not isinstance(response, Mapping) or set(response) != {"decision", "reason"}
                or type(response["decision"]) is not str
                or response["decision"] not in ("PASS", "FAIL")
                or not isinstance(response["reason"], str)
                or not response["reason"].strip()):
            raise ValueError("admission_checker: expected PASS or FAIL with a non-empty reason")
        return replace(candidate, admission=response["decision"],
                       admission_reason=response["reason"].strip())


class LLMFusionJudge:
    """只提交 content/trigger；匹配位置由调用方映射回 canonical experience ID。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def find_match(self, content: str, trigger: str,
                   existing: Sequence[tuple[str, str]]) -> int | None:
        _text(content, "content")
        _text(trigger, "trigger")
        if any(not isinstance(item, tuple) or len(item) != 2
               or any(not isinstance(text, str) for text in item) for item in existing):
            raise ValueError("fusion_judge: existing entries must be content/trigger pairs")
        if not existing:
            return None
        index = _output(self.backend, "fusion_judge", fusion_payload(
            content, trigger, existing,
        ), "match_index")
        if index is not None and (type(index) is not int or not 0 <= index < len(existing)):
            raise ValueError("fusion_judge: match_index must be an existing position or null")
        return index


@dataclass
class LLMModules:
    """角色适配器共享一个 backend；可由配置为各角色覆盖。"""

    backend: LLMBackend
    retriever: Retriever
    selector: LLMSelector
    initial_generator: LLMInitialRubricGenerator
    judge: LLMJudge
    critic: LLMCritic
    revision_generator: LLMRevisionGenerator
    reflector: LLMReflector
    revision_credit_assigner: LLMRevisionCreditAssigner
    admission_checker: LLMAdmissionChecker
    fusion_judge: LLMFusionJudge


FakeModules = LLMModules  # 向后兼容原有离线装配类型名。


def make_llm_modules(backend: LLMBackend, *, top_l: int = 20,
                     max_selected: int = 5, retriever: Retriever | None = None) -> LLMModules:
    """装配各角色与可替换检索实现，不修改 Pipeline。"""
    return LLMModules(backend, retriever if retriever is not None else FakeRetriever(top_l),
                      LLMSelector(backend, max_selected=max_selected),
                      LLMInitialRubricGenerator(backend), LLMJudge(backend),
                      LLMCritic(backend), LLMRevisionGenerator(backend),
                      LLMReflector(backend), LLMRevisionCreditAssigner(backend),
                      LLMAdmissionChecker(backend), LLMFusionJudge(backend))


def make_fake_modules(scenario: FakeScenario = "attempt_1_success", **options: Any) -> LLMModules:
    """组装固定场景离线模块，保留原有测试/API。"""
    return make_llm_modules(FakeLLMBackend(scenario, **options))
