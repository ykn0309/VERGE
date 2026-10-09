"""方法模块契约和 LLM 角色适配器。

适配器负责限定各角色可见的输入、校验结构化输出；数值计算委托 Calculator。
将相同接口接到 FakeLLMBackend 即可离线运行，不需要另写一套假模块。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from uuid import uuid4
from typing import Any, Mapping, Protocol, Sequence

from verge.calculator import calculate_rubric_judgment
from verge.llm import (
    FakeLLMBackend,
    FakeScenario,
    JUDGE_RESPONSE_SIDE,
    LLMBackend,
    LLMFormatError,
    OpenAICompatibleBackend,
)
from verge.models import (
    CandidateExperience,
    CriticProposal,
    Criterion,
    CriterionJudgment,
    Experience,
    Rubric,
    RubricJudgment,
    FusionResult,
)
from verge.prompts import (
    ExperienceView,
    NumberedIDCodec,
    admission_payload,
    critic_payload,
    fusion_payload,
    generator_payload,
    judge_payload,
    selector_payload,
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

    def apply(
        self,
        task: str,
        selected: Sequence[ExperienceView],
        candidate: CandidateExperience,
        round_number: int,
    ) -> Rubric: ...


# Judge 及其后续角色才接触偏好对和修订结果。
class Judge(Protocol):
    """标准评价：读取偏好对，逐项产生 Boolean 判断并计算完整 RubricJudgment。"""

    def judge(self, rubric: Rubric, chosen: str, rejected: str, *, task: str,
              previous: Rubric | None = None,
              previous_judgment: RubricJudgment | None = None) -> RubricJudgment: ...


class Critic(Protocol):
    def propose(
        self,
        task: str,
        chosen: str,
        rejected: str,
        current: Rubric,
        judgment: RubricJudgment,
        round_number: int,
        generation_number: int,
        **history,
    ) -> CriticProposal: ...


class AdmissionChecker(Protocol):
    def check(
        self, task: str, candidate: CandidateExperience
    ) -> CandidateExperience: ...


class FusionJudge(Protocol):
    """经验融合判断：仅看 content/trigger，返回兼容经验的位置及融合文本。"""

    def find_match(self, content: str, trigger: str,
                   existing: Sequence[tuple[str, str]]) -> FusionResult: ...


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _views(items: Sequence[ExperienceView]) -> None:
    """强制使用白名单投影，避免把 Experience 的轨迹/证据字段送进 prompt。"""
    if any(type(item) is not ExperienceView for item in items):
        raise ValueError("selected/retrieved experiences must be ExperienceView projections")


def _output(
    backend: LLMBackend, role: str, payload: dict[str, Any], field: str | None
) -> Any:
    """所有角色共享的结构校验；字段值仍由对应模块按领域模型校验。"""
    # Fake 后端沿用内部固定 ID；真实模型只接触本次调用的局部数字编号。
    codec = (
        NumberedIDCodec(role, payload)
        if isinstance(backend, OpenAICompatibleBackend)
        else None
    )
    projected = codec.project(payload) if codec else payload
    response = backend.generate(role, projected)
    expected = (
        {"analysis", "criteria"}
        if codec and role == "initial_generator"
        else (
            {field}
            if field is not None
            else (
                {"decision", "reason"}
                if role == "admission_checker"
                else (
                    {"match_index", "content", "trigger"}
                    if role == "fusion_judge"
                    else {"analysis", "candidates"}
                )
            )
        )
    )
    if not isinstance(response, Mapping) or set(response) != expected:
        raise ValueError(f"{role}: expected structured fields {sorted(expected)}")
    if codec:
        response = codec.restore(response)
    return response[field] if field is not None else response


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

    def fork(self) -> VectorRetriever:
        """为并发只读评估复制已缓存的经验向量，避免线程共享可变缓存。"""
        result = VectorRetriever(self.backend, self.top_l)
        result._vectors = self._vectors.copy()
        return result

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
    """Every generation is preference-blind and begins a fresh conversation."""

    def __init__(self, backend: LLMBackend, *, invalid_output_retries: int = 2) -> None:
        if (
            type(invalid_output_retries) is not int
            or not 0 <= invalid_output_retries <= 3
        ):
            raise ValueError("invalid_output_retries must be in [0,3]")
        self.backend = backend
        self.invalid_output_retries = invalid_output_retries
        self.initial_output_retries = invalid_output_retries

    def generate(self, task: str, selected: Sequence[ExperienceView]) -> Rubric:
        return self._generate(task, selected)

    def apply(
        self,
        task: str,
        selected: Sequence[ExperienceView],
        candidate: CandidateExperience,
        round_number: int,
    ) -> Rubric:
        if candidate.admission != "PASS" or candidate.fusion is not None:
            raise ValueError("application requires an admitted unfused candidate")
        if round_number not in (1, 2):
            raise ValueError("application round must be 1 or 2")
        return self._generate(task, selected, candidate, round_number)

    def _generate(self, task, selected, candidate=None, round_number=0) -> Rubric:
        _text(task, "task")
        _views(selected)
        allowed = {x.experience_id for x in selected}
        if len(allowed) != len(selected):
            raise ValueError("selected IDs must be unique")
        if candidate:
            if candidate.candidate_id in allowed:
                raise ValueError("candidate ID collides with existing experience")
            allowed.add(candidate.candidate_id)
        feedback = None
        retries = (
            self.invalid_output_retries if candidate else self.initial_output_retries
        )
        for retry in range(retries + 1):
            try:
                payload = generator_payload(
                    task, selected, candidate=candidate, validation_feedback=feedback
                )
                # Fake scenario metadata only; never sent in actual Generator messages.
                if isinstance(self.backend, FakeLLMBackend):
                    payload["_application_round"] = round_number
                raw = _output(self.backend, "initial_generator", payload, "rubric")
                rubric = Rubric.from_dict(raw)
                if any(set(c.source_experiences) - allowed for c in rubric.criteria):
                    raise ValueError(
                        "source_experiences must be distinct IDs from this call's experiences"
                    )
                rid = "rubric_" + uuid4().hex
                return Rubric(
                    rid,
                    [
                        replace(c, criterion_id=f"{rid}_c{i}")
                        for i, c in enumerate(rubric.criteria, 1)
                    ],
                )
            except LLMFormatError:
                raise
            except ValueError as exc:
                if retry == retries:
                    raise
                feedback = (
                    "上次输出校验失败："
                    + str(exc)[:160]
                    + "。请返回完整合法结果，来源只引用本次提供的经验。"
                )
        raise AssertionError("unreachable")


class JudgeOutputError(ValueError):
    """模型 Judge 输出经有限纠错后仍不能对应待判断标准；不得写成有效判断。"""


class LLMJudge:
    """LLM 只判断 Boolean；错误输出有限重试，最终失败交由 Runner 记账并继续。"""

    def __init__(self, backend: LLMBackend, *, invalid_output_retries: int = 2) -> None:
        if type(invalid_output_retries) is not int or not 0 <= invalid_output_retries <= 3:
            raise ValueError("invalid_output_retries must be an integer in [0, 3]")
        self.backend = backend
        self.invalid_output_retries = invalid_output_retries

    def judge(
        self,
        rubric: Rubric,
        chosen: str,
        rejected: str,
        *,
        task: str,
        previous: Rubric | None = None,
        previous_judgment: RubricJudgment | None = None,
    ) -> RubricJudgment:
        _text(task, "task")
        if type(chosen) is not str or type(rejected) is not str:
            raise ValueError("chosen and rejected must be strings")
        if (previous is None) != (previous_judgment is None):
            raise ValueError("previous rubric and judgment must be supplied together")
        known: dict[str, CriterionJudgment] = {}
        if previous is not None and previous_judgment is not None:
            if previous_judgment.rubric_id != previous.rubric_id:
                raise ValueError(
                    "previous judgment must belong to the supplied previous rubric"
                )
            old = {item.criterion_id: item for item in previous.criteria}
            old_answers = {
                item.criterion_id: item
                for item in previous_judgment.criterion_judgments
            }
            if set(old_answers) != set(old) or len(old_answers) != len(
                previous_judgment.criterion_judgments
            ):
                raise ValueError("previous judgment does not cover the base rubric")
            # 相同问题若 R0 判断一致，按文本复用 Boolean；包含 MODIFY 后撞名的情况。
            # R0 同名问题判断不一致时，不猜测答案，只沿用 ID 与问题均未变的判断。
            by_question: dict[str, set[tuple[bool, bool]]] = {}
            for item in previous.criteria:
                answer = old_answers[item.criterion_id]
                by_question.setdefault(item.question, set()).add(
                    (answer.chosen, answer.rejected)
                )
            for item in rubric.criteria:
                if (
                    item.criterion_id in old
                    and item.question == old[item.criterion_id].question
                ):
                    known[item.criterion_id] = old_answers[item.criterion_id]
                elif len(by_question.get(item.question, ())) == 1:
                    chosen_answer, rejected_answer = next(
                        iter(by_question[item.question])
                    )
                    known[item.criterion_id] = CriterionJudgment(
                        item.criterion_id,
                        chosen_answer,
                        rejected_answer,
                    )
        # 后端仅回答待判断标准，返回值必须恰好覆盖这些 ID。
        pending = [item for item in rubric.criteria if item.criterion_id not in known]
        generated: list[CriterionJudgment] = []
        requested = {item.criterion_id for item in pending}
        if pending:

            def score_response(response: str, side: str) -> dict[str, bool]:
                token = JUDGE_RESPONSE_SIDE.set(side)
                try:
                    feedback = None
                    for retry in range(self.invalid_output_retries + 1):
                        try:
                            raw = _output(
                                self.backend,
                                "judge",
                                judge_payload(
                                    pending,
                                    response,
                                    task=task,
                                    validation_feedback=feedback,
                                ),
                                "criterion_judgments",
                            )
                            if not isinstance(raw, list):
                                raise ValueError(
                                    "judge: criterion_judgments must be a list"
                                )
                            answers = {}
                            for item in raw:
                                if (
                                    not isinstance(item, dict)
                                    or set(item) != {"criterion_id", "satisfied"}
                                    or not isinstance(item["criterion_id"], str)
                                    or type(item["satisfied"]) is not bool
                                ):
                                    raise ValueError(
                                        "judge: expected criterion_id and Boolean satisfied"
                                    )
                                if item["criterion_id"] in answers:
                                    raise ValueError("judge: duplicate criterion_id")
                                answers[item["criterion_id"]] = item["satisfied"]
                            if set(answers) != requested:
                                raise ValueError(
                                    "judge: judgments must cover exactly the requested criteria "
                                    f"(missing={len(requested - answers.keys())}, "
                                    f"extra={len(answers.keys() - requested)})"
                                )
                            return answers
                        except ValueError as exc:
                            if retry == self.invalid_output_retries:
                                raise JudgeOutputError(
                                    f"judge ({side}): invalid output after {retry + 1} attempts: "
                                    f"{str(exc)[:160]}"
                                ) from exc
                            feedback = (
                                "上次 Judge 输出校验信息："
                                + str(exc)[:160]
                                + "。请严格复制其数字编号，每条标准返回一次判断，"
                                "完整覆盖本次标准；satisfied 使用布尔值。"
                            )
                    raise AssertionError("unreachable judge retry state")
                finally:
                    JUDGE_RESPONSE_SIDE.reset(token)

            # Each worker inherits sample context, with its own answer-side label only in logs.
            outcomes = []
            errors = []
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="verge-judge"
            ) as pool:
                futures = [
                    pool.submit(copy_context().run, score_response, response, side)
                    for response, side in ((chosen, "chosen"), (rejected, "rejected"))
                ]
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except Exception as exc:
                        outcomes.append(None)
                        errors.append(exc)
            if errors:
                # Transport/auth errors keep their abort semantics even if the other side is malformed.
                raise next(
                    (exc for exc in errors if not isinstance(exc, JudgeOutputError)),
                    errors[0],
                )
            chosen_answers, rejected_answers = outcomes
            generated = [
                CriterionJudgment(
                    item.criterion_id,
                    chosen_answers[item.criterion_id],
                    rejected_answers[item.criterion_id],
                )
                for item in pending
            ]
        answers = {item.criterion_id: item for item in generated}
        return calculate_rubric_judgment(
            rubric,
            [
                known.get(item.criterion_id) or answers[item.criterion_id]
                for item in rubric.criteria
            ],
        )


class LLMCritic:
    def __init__(self, backend: LLMBackend, *, invalid_output_retries: int = 2) -> None:
        if (
            type(invalid_output_retries) is not int
            or not 0 <= invalid_output_retries <= 3
        ):
            raise ValueError("invalid_output_retries must be in [0,3]")
        self.backend = backend
        self.invalid_output_retries = invalid_output_retries

    def propose(
        self,
        task,
        chosen,
        rejected,
        current,
        judgment,
        round_number,
        generation_number,
        *,
        previous_candidate=None,
        rejected_candidate=None,
        adopted=None,
        invalid_error=None,
        previous_rubric=None,
        previous_judgment=None,
    ):
        _text(task, "task")
        if judgment.rubric_id != current.rubric_id or judgment.success:
            raise ValueError("critic requires a failed valid rubric")
        if round_number not in (1, 2) or generation_number not in (1, 2):
            raise ValueError("critic budget exceeded")
        payload = critic_payload(
            task,
            chosen,
            rejected,
            current,
            judgment,
            round_number,
            generation_number,
            previous_candidate=previous_candidate,
            rejected_candidate=rejected_candidate,
            adopted=adopted,
            invalid_error=invalid_error,
            previous_rubric=previous_rubric,
            previous_judgment=previous_judgment,
        )
        for retry in range(self.invalid_output_retries + 1):
            try:
                raw = _output(self.backend, "critic", payload, None)
                _text(raw["analysis"], "critic analysis")
                if (
                    not isinstance(raw["candidates"], list)
                    or len(raw["candidates"]) > 1
                ):
                    raise ValueError("critic must propose 0..1 candidates")
                candidate = None
                if raw["candidates"]:
                    if not isinstance(raw["candidates"][0], Mapping):
                        raise ValueError("critic candidate must be an object")
                    item = dict(raw["candidates"][0])
                    if isinstance(self.backend, OpenAICompatibleBackend):
                        item.pop(
                            "candidate_id", None
                        )  # codec's placeholder; adapter owns canonical ID.
                    if set(item) != {"content", "trigger"}:
                        raise ValueError(
                            "critic candidate must contain content and trigger only"
                        )
                    candidate = CandidateExperience("candidate_" + uuid4().hex, **item)
                return CriticProposal(raw["analysis"], candidate)
            except LLMFormatError:
                raise  # backend format budget is already exhausted
            except ValueError as exc:
                if retry == self.invalid_output_retries:
                    raise
                payload = {
                    **payload,
                    "validation_feedback": "上次输出校验失败："
                    + str(exc)[:160]
                    + "。返回合法私有分析及0到1条content/trigger候选。",
                }
        raise AssertionError("unreachable")


class LLMAdmissionChecker:
    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def check(self, task: str, candidate: CandidateExperience) -> CandidateExperience:
        if candidate.admission is not None or candidate.fusion is not None:
            raise ValueError("admission requires an unreviewed candidate")
        raw = _output(
            self.backend, "admission_checker", admission_payload(task, candidate), None
        )
        if raw["decision"] not in ("PASS", "FAIL"):
            raise ValueError("admission expected PASS or FAIL")
        _text(raw["reason"], "admission reason")
        return replace(
            candidate, admission=raw["decision"], admission_reason=raw["reason"]
        )


class LLMFusionJudge:
    """只提交 content/trigger；匹配位置由调用方映射回 canonical experience ID。"""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def find_match(self, content: str, trigger: str,
                   existing: Sequence[tuple[str, str]]) -> FusionResult:
        _text(content, "content")
        _text(trigger, "trigger")
        if any(not isinstance(item, tuple) or len(item) != 2
               or any(not isinstance(text, str) for text in item) for item in existing):
            raise ValueError("fusion_judge: existing entries must be content/trigger pairs")
        if not existing:
            return FusionResult(None, None, None)
        raw = _output(self.backend, "fusion_judge", fusion_payload(content, trigger, existing), None)
        if raw["match_index"] is not None and type(raw["match_index"]) is not int:
            raise ValueError("fusion_judge: match_index must be an integer or null")
        result = FusionResult.from_dict(raw)
        if result.match_index is not None and result.match_index >= len(existing):
            raise ValueError("fusion_judge: match_index must be an existing position or null")
        return result


@dataclass
class LLMModules:
    """角色适配器共享一个 backend；可由配置为各角色覆盖。"""

    backend: LLMBackend
    retriever: Retriever
    selector: LLMSelector
    initial_generator: LLMInitialRubricGenerator
    judge: LLMJudge
    critic: LLMCritic
    admission_checker: LLMAdmissionChecker
    fusion_judge: LLMFusionJudge


def make_llm_modules(
    backend: LLMBackend,
    *,
    top_l: int = 20,
    max_selected: int = 5,
    retriever: Retriever | None = None,
) -> LLMModules:
    """装配各角色与可替换检索实现，不修改 Pipeline。"""
    return LLMModules(
        backend,
        retriever if retriever is not None else FakeRetriever(top_l),
        LLMSelector(backend, max_selected=max_selected),
        LLMInitialRubricGenerator(backend),
        LLMJudge(backend),
        LLMCritic(backend),
        LLMAdmissionChecker(backend),
        LLMFusionJudge(backend),
    )


def make_fake_modules(scenario: FakeScenario = "attempt_1_success", **options: Any) -> LLMModules:
    """组装固定场景离线模块，保留原有测试/API。"""
    return make_llm_modules(FakeLLMBackend(scenario, **options))
