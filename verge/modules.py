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
    LLMSampleError,
    OpenAICompatibleBackend,
)
from verge.models import (
    CandidateExperience,
    CriticProposal,
    Criterion,
    CriterionJudgment,
    Experience,
    ExperienceUpdate,
    ExperienceCategory,
    CategoryOrganization,
    Rubric,
    RubricJudgment,
)
from verge.prompts import (
    ExperienceView,
    NumberedIDCodec,
    critic_payload,
    experience_update_payload,
    generator_payload,
    judge_payload,
    selector_payload,
    category_organization_payload,
)


# 检索、筛选和初始生成不接收 preference pair；只传递 task/经验投影。
class Retriever(Protocol):
    """广召回：输入 task 和经验池，输出候选经验；不得读取偏好回答。"""

    def retrieve(self, task: str, experiences: Sequence[Experience], *,
                 categories: Sequence[ExperienceCategory] = ()) -> list[Experience]: ...


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
        **history,
    ) -> CriticProposal: ...


class ExperienceUpdater(Protocol):
    def update(self, experience: Experience) -> ExperienceUpdate: ...


class CategoryOrganizer(Protocol):
    def organize(self, experiences: Sequence[Experience], categories: Sequence[ExperienceCategory],
                 pending: Mapping[str, int]) -> CategoryOrganization: ...


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _views(items: Sequence[ExperienceView]) -> None:
    """强制使用白名单投影，避免把 Experience 的轨迹/证据字段送进 prompt。"""
    if any(type(item) is not ExperienceView for item in items):
        raise ValueError("selected/retrieved experiences must be ExperienceView projections")


def _output(
    backend: LLMBackend, role: str, payload: dict[str, Any], field: str | None,
    *, conversation: list[dict[str, str]] | None = None,
) -> Any:
    """所有角色共享的结构校验；字段值仍由对应模块按领域模型校验。"""
    # Fake 后端沿用内部固定 ID；真实模型只接触本次调用的局部数字编号。
    codec = (
        NumberedIDCodec(role, payload)
        if isinstance(backend, OpenAICompatibleBackend)
        else None
    )
    projected = codec.project(payload) if codec else payload
    response = (backend.generate(role, projected, conversation=conversation)
                if conversation is not None else backend.generate(role, projected))
    expected = (
        {"analysis", "criteria"}
        if codec and role == "initial_generator"
        else (
            {field}
            if field is not None
            else (
                {"categories", "ungrouped_experience_ids"}
                if role == "category_organizer"
                else {"content", "trigger"}
                if role == "experience_updater"
                else {"analysis", "candidates"}
            )
        )
    )
    if not isinstance(response, Mapping) or set(response) != expected:
        raise ValueError(f"{role}: expected structured fields {sorted(expected)}")
    if codec:
        response = codec.restore(response)
    return response[field] if field is not None else response


@dataclass
class RetrievalUnit:
    unit_id: str
    text: str
    members: list[Experience]
    category: ExperienceCategory | None = None


def retrieval_units(experiences: Sequence[Experience], categories: Sequence[ExperienceCategory]) -> list[RetrievalUnit]:
    by_category = {c.category_id: c for c in categories}
    if (len({e.experience_id for e in experiences}) != len(experiences)
            or len(by_category) != len(categories)):
        raise ValueError("retrieval requires unique experience and category IDs")
    units = []
    grouped = {}
    for e in experiences:
        if e.category_id is None:
            units.append(RetrievalUnit(e.experience_id, f"{e.content}\n{e.trigger}", [e]))
        else:
            if e.category_id not in by_category:
                raise ValueError("retrieval found unknown category")
            if e.category_id not in grouped:
                c = by_category[e.category_id]
                grouped[e.category_id] = RetrievalUnit(c.category_id, "", [], c)
                units.append(grouped[e.category_id])
            grouped[e.category_id].members.append(e)
    for unit in grouped.values():
        unit.text = "\n".join([unit.category.name, unit.category.description,
                              *(f"{e.trigger}\n{e.content}" for e in unit.members)])
    return units


class FakeRetriever:
    """稳定取前 L 条，供离线测试使用；不声称提供向量相似度。"""

    def __init__(self, top_l: int = 20) -> None:
        if type(top_l) is not int or top_l != 20:
            raise ValueError("V1 retrieval top_l must be 20")
        self.top_l = top_l

    def retrieve(self, task: str, experiences: Sequence[Experience], *,
                 categories: Sequence[ExperienceCategory] = ()) -> list[Experience]:
        _text(task, "task")
        return [e for unit in retrieval_units(experiences, categories)[:self.top_l] for e in unit.members]


class EmbeddingBackend(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class VectorRetriever:
    """进程内缓存经验文本向量，用余弦相似度从 Memory 广召回 Top 20。"""

    def __init__(self, backend: EmbeddingBackend, top_l: int = 20) -> None:
        if type(top_l) is not int or top_l != 20:
            raise ValueError("V1 retrieval top_l must be 20")
        self.backend, self.top_l = backend, top_l
        self._vectors: dict[str, tuple[str, tuple[float, ...]]] = {}

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

    def retrieve(self, task: str, experiences: Sequence[Experience], *,
                 categories: Sequence[ExperienceCategory] = ()) -> list[Experience]:
        _text(task, "task")
        if not experiences:
            return []
        units = retrieval_units(experiences, categories)
        query = self._unit(self.backend.embed(task))
        scored = []
        self._vectors = {key: value for key, value in self._vectors.items() if key in {u.unit_id for u in units}}
        for item in units:
            text = item.text
            cached = self._vectors.get(item.unit_id)
            if cached is None or cached[0] != text:
                vector = self._unit(self.backend.embed(text))
                self._vectors[item.unit_id] = (text, vector)
            else:
                vector = cached[1]
            if len(query) != len(vector):
                raise ValueError("retrieval embedding dimensions disagree")
            scored.append((sum(a * b for a, b in zip(query, vector)), item.unit_id, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [e for row in scored[:self.top_l] for e in row[2].members]


class LLMSelector:
    """让 LLM 返回召回候选的 ID；不允许凭空选出未召回的经验。"""

    def __init__(self, backend: LLMBackend, *, max_selected: int = 5, input_bytes: int = 100_000) -> None:
        if type(max_selected) is not int or not 0 <= max_selected <= 5:
            raise ValueError("max_selected must be an integer in [0, 5]")
        self.backend = backend
        self.max_selected = max_selected
        if type(input_bytes) is not int or input_bytes < 1:
            raise ValueError("selector input_bytes must be positive")
        self.input_bytes = input_bytes

    def select(self, task: str, candidates: Sequence[ExperienceView], *,
               training: bool = True) -> list[ExperienceView]:
        _text(task, "task")
        _views(candidates)
        if type(training) is not bool:
            raise ValueError("training must be Boolean")
        if not candidates or not self.max_selected:
            return []
        payload = selector_payload(
            task, candidates, max_selected=self.max_selected, training=training,
        )
        _input_budget(self.backend, "selector", payload, self.input_bytes)
        identifiers = _output(self.backend, "selector", payload, "selected_experience_ids")
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
        if candidate.experience_id is not None:
            raise ValueError("application requires a new candidate")
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
    """LLM返回Boolean与依据；错误输出有限重试，最终失败显式留档。"""

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
            # 相同问题若判断一致，按文本复用Boolean及原依据。
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
                    prior_answer = next(
                        old_answers[c.criterion_id]
                        for c in previous.criteria
                        if c.question == item.question
                    )
                    known[item.criterion_id] = replace(
                        prior_answer, criterion_id=item.criterion_id,
                    )
        # 后端仅回答待判断标准，返回值必须恰好覆盖这些 ID。
        pending = [item for item in rubric.criteria if item.criterion_id not in known]
        generated: list[CriterionJudgment] = []
        requested = {item.criterion_id for item in pending}
        if pending:

            def score_response(response: str, side: str) -> dict[str, tuple[bool, str]]:
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
                                    or set(item) != {"criterion_id", "satisfied", "reason"}
                                    or not isinstance(item["criterion_id"], str)
                                    or type(item["satisfied"]) is not bool
                                    or not isinstance(item["reason"], str)
                                    or not item["reason"].strip()
                                ):
                                    raise ValueError(
                                        "judge: expected criterion_id, Boolean satisfied and nonempty reason"
                                    )
                                if item["criterion_id"] in answers:
                                    raise ValueError("judge: duplicate criterion_id")
                                answers[item["criterion_id"]] = (item["satisfied"], item["reason"])
                            if set(answers) != requested:
                                raise ValueError(
                                    "judge: judgments must cover exactly the requested criteria "
                                    f"(missing={len(requested - answers.keys())}, "
                                    f"extra={len(answers.keys() - requested)})"
                                )
                            return answers
                        except LLMFormatError as exc:
                            # Backend格式预算已耗尽，不再叠加适配层结构重试。
                            raise JudgeOutputError(
                                f"judge ({side}): backend format retries exhausted: {exc}"
                            ) from exc
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
                                "完整覆盖本次标准；satisfied使用布尔值，reason给出非空判断依据。"
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
                # Fatal auth/program errors must win over a skippable error on the other side.
                raise next(
                    (exc for exc in errors if not isinstance(exc, (JudgeOutputError, LLMSampleError))),
                    next((exc for exc in errors if isinstance(exc, LLMSampleError)), errors[0]),
                )
            chosen_answers, rejected_answers = outcomes
            generated = [
                CriterionJudgment(
                    item.criterion_id,
                    chosen_answers[item.criterion_id][0],
                    rejected_answers[item.criterion_id][0],
                    chosen_answers[item.criterion_id][1],
                    rejected_answers[item.criterion_id][1],
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
        *,
        previous_candidate=None,
        adopted=None,
        invalid_error=None,
        previous_rubric=None,
        previous_judgment=None,
        conversation=None,
    ):
        _text(task, "task")
        if judgment.rubric_id != current.rubric_id or judgment.success:
            raise ValueError("critic requires a failed valid rubric")
        if round_number not in (1, 2):
            raise ValueError("critic budget exceeded")
        payload = critic_payload(
            task,
            chosen,
            rejected,
            current,
            judgment,
            round_number,
            previous_candidate=previous_candidate,
            adopted=adopted,
            invalid_error=invalid_error,
            previous_rubric=previous_rubric,
            previous_judgment=previous_judgment,
        )
        for retry in range(self.invalid_output_retries + 1):
            try:
                raw = _output(self.backend, "critic", payload, None, conversation=conversation)
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


class LLMExperienceUpdater:
    def __init__(self, backend: LLMBackend, *, invalid_output_retries: int = 2) -> None:
        if type(invalid_output_retries) is not int or not 0 <= invalid_output_retries <= 3:
            raise ValueError("invalid_output_retries must be in [0,3]")
        self.backend = backend
        self.invalid_output_retries = invalid_output_retries

    def update(self, experience: Experience) -> ExperienceUpdate:
        if not 2 <= len(experience.cases) <= 10:
            raise ValueError("experience update requires 2..10 cases")
        payload = experience_update_payload(experience)
        # Bound the full request in UTF-8 bytes; never truncate original cases.
        from verge.prompts import llm_messages
        budget_payload = (NumberedIDCodec("experience_updater", payload).project(payload)
                          if isinstance(self.backend, OpenAICompatibleBackend) else payload)
        if sum(len(m["content"].encode("utf-8")) for m in llm_messages("experience_updater", budget_payload)) > 100_000:
            raise ValueError("experience update context budget exceeded")
        for retry in range(self.invalid_output_retries + 1):
            try:
                raw = _output(self.backend, "experience_updater", payload, None)
                update = ExperienceUpdate.from_dict(dict(raw))
                return update
            except LLMFormatError:
                raise
            except ValueError as exc:
                if retry == self.invalid_output_retries:
                    raise
                payload = {**payload, "validation_feedback": "上次文本校验失败：" + str(exc)[:160]}
        raise AssertionError("unreachable")


def _input_budget(backend, role, payload, limit):
    from verge.prompts import llm_messages
    projected = NumberedIDCodec(role, payload).project(payload) if isinstance(backend, OpenAICompatibleBackend) else payload
    if sum(len(m["content"].encode("utf-8")) for m in llm_messages(role, projected)) > limit:
        raise ValueError(f"{role} context budget exceeded")


class LLMCategoryOrganizer:
    def __init__(self, backend: LLMBackend, *, invalid_output_retries: int = 2) -> None:
        if type(invalid_output_retries) is not int or not 0 <= invalid_output_retries <= 3:
            raise ValueError("invalid_output_retries must be in [0,3]")
        self.backend, self.invalid_output_retries = backend, invalid_output_retries

    def organize(self, experiences, categories, pending) -> CategoryOrganization:
        payload = category_organization_payload(experiences, categories, pending)
        for retry in range(self.invalid_output_retries + 1):
            try:
                _input_budget(self.backend, "category_organizer", payload, 100_000)
                plan = CategoryOrganization.from_dict(dict(_output(self.backend, "category_organizer", payload, None)))
                plan.validate_partition({e.experience_id for e in experiences}, {c.category_id for c in categories})
                return plan
            except LLMFormatError:
                raise
            except ValueError:
                if retry == self.invalid_output_retries:
                    raise
                # Do not echo long internal IDs into correction prompts.
                payload = {**payload, "validation_feedback": "上次分类结构或分配无效：请确保所有经验恰好分配一次，沿用输入类别编号或用null创建新类别，输出完整方案。"}
        raise AssertionError("unreachable")


@dataclass
class LLMModules:
    """角色适配器共享一个 backend；可由配置为各角色覆盖。"""

    backend: LLMBackend
    retriever: Retriever
    selector: LLMSelector
    initial_generator: LLMInitialRubricGenerator
    judge: LLMJudge
    critic: LLMCritic
    experience_updater: LLMExperienceUpdater
    category_organizer: LLMCategoryOrganizer


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
        LLMExperienceUpdater(backend),
        LLMCategoryOrganizer(backend),
    )


def make_fake_modules(scenario: FakeScenario = "attempt_1_success", **options: Any) -> LLMModules:
    """组装固定场景离线模块，保留原有测试/API。"""
    return make_llm_modules(FakeLLMBackend(scenario, **options))
