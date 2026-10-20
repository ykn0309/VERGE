"""VERGE candidate-validation 的领域数据契约；数值计算和持久化由后续模块负责。"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
import json
from math import isfinite
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints


ModelT = TypeVar("ModelT", bound="Model")


def _decode(value: Any, annotation: Any) -> Any:
    """按字段类型递归解码，并在输入边界拒绝类型不匹配的 JSON 值。"""
    origin = get_origin(annotation)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError("expected a list")
        return [_decode(item, get_args(annotation)[0]) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError("expected a dictionary")
        key_type, value_type = get_args(annotation)
        return {_decode(k, key_type): _decode(v, value_type) for k, v in value.items()}
    if origin is Literal:
        if value not in get_args(annotation):
            raise ValueError(f"expected one of {get_args(annotation)}")
        return value
    if annotation is type(None):
        if value is not None:
            raise ValueError("expected null")
        return None
    if annotation is Any:
        return value
    if origin is not None:  # Optional / Union：逐一尝试可选分支
        for option in get_args(annotation):
            try:
                return _decode(value, option)
            except (TypeError, ValueError):
                pass
        raise ValueError("value does not match any allowed type")
    if isinstance(annotation, type) and issubclass(annotation, Model):
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return annotation.from_dict(value)
    if annotation is float and type(value) in (int, float):
        return float(value)
    if annotation in (str, int, float, bool) and type(value) is annotation:
        return value
    raise ValueError(f"expected {annotation}")


def _encode(value: Any) -> Any:
    if isinstance(value, Model):
        return value.to_dict()
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


class Model:
    """统一 JSON-compatible dict 转换入口；不执行磁盘 I/O。"""

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _encode(getattr(self, field.name)) for field in fields(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls: type[ModelT], text: str) -> ModelT:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls: type[ModelT], data: dict[str, Any]) -> ModelT:
        # 严格检查字段集合，避免拼错字段名时静默丢失信息。
        if not isinstance(data, dict):
            raise ValueError(f"{cls.__name__}: expected an object")
        annotations = get_type_hints(cls)
        model_fields = {item.name: item for item in fields(cls)}
        names = set(model_fields)
        required = {name for name, item in model_fields.items()
                    if item.default is MISSING and item.default_factory is MISSING}
        if not required <= data.keys() or not data.keys() <= names:
            raise ValueError(f"{cls.__name__}: missing {sorted(required - data.keys())}, unknown {sorted(data.keys() - names)}")
        try:
            kwargs = {name: _decode(data[name], annotations[name]) for name in data}
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{cls.__name__}: {exc}") from exc


def _id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _number(value: float, name: str, *, minimum: float | None = None) -> None:
    if type(value) not in (int, float) or not isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be a finite number" + (f" >= {minimum}" if minimum is not None else ""))


@dataclass
class PreferenceSample(Model):
    """偏好样本。

    sample_id: 样本唯一标识；task: 待评价的任务描述；
    chosen/rejected: 偏好对中较优/较差的回答；metadata: 数据集附加信息。
    """

    sample_id: str
    task: str
    chosen: str
    rejected: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.sample_id, "sample_id")


@dataclass(frozen=True)
class Credit(Model):
    """两个非负通道；净方向仅供展示。"""

    positive_credit: float = 0.0
    negative_credit: float = 0.0

    def __post_init__(self) -> None:
        for name in ("positive_credit", "negative_credit"):
            _number(getattr(self, name), name, minimum=0)
        if self.positive_credit + self.negative_credit > 1 + 1e-9:
            raise ValueError("credit evidence amount must be at most 1")

    @property
    def net(self) -> float:
        return self.positive_credit - self.negative_credit

    @property
    def amount(self) -> float:
        return self.positive_credit + self.negative_credit


@dataclass
class EvidenceRecord(Model):
    sample_id: str
    evidence_type: Literal["downstream_credit", "application_credit"]
    positive_credit: float
    negative_credit: float
    candidate_id: str | None = None
    evidence_weight: float = 1.0
    text_version: int = 1

    def __post_init__(self) -> None:
        _id(self.sample_id, "sample_id")
        if type(self.text_version) is not int or self.text_version < 1:
            raise ValueError("invalid evidence text_version")
        if type(self.evidence_weight) not in (int, float) or self.evidence_weight != 1:
            raise ValueError("evidence_weight is fixed at 1")
        if self.evidence_type not in ("downstream_credit", "application_credit"):
            raise ValueError("unknown evidence type")
        if Credit(self.positive_credit, self.negative_credit).amount == 0:
            raise ValueError("zero evidence must not be appended")
        if self.evidence_type == "application_credit":
            _id(self.candidate_id, "application candidate_id")
        elif self.candidate_id is not None:
            raise ValueError("downstream evidence must not have a candidate")


@dataclass
class ExperienceCase(Model):
    sample_id: str
    task: str
    chosen: str
    rejected: str
    source_id: str
    text_version: int
    content: str
    trigger: str
    rubric: Rubric
    judgment: RubricJudgment
    credit: Credit
    analysis: str | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "task", "chosen", "rejected", "source_id", "content", "trigger"):
            _id(getattr(self, name), name)
        if type(self.text_version) is not int or self.text_version < 1:
            raise ValueError("invalid case text_version")
        from verge.calculator import calculate_experience_credits
        expected = calculate_experience_credits(self.rubric, self.judgment).get(self.source_id)
        if expected != self.credit or self.credit.amount == 0:
            raise ValueError("case requires actual nonzero source credit")


@dataclass
class AtomicMethod(Model):
    content: str
    trigger: str

    def __post_init__(self) -> None:
        _id(self.content, "content")
        _id(self.trigger, "trigger")


@dataclass
class AtomicResult(Model):
    experiences: list[AtomicMethod]

    def __post_init__(self) -> None:
        if not self.experiences or len({(x.content, x.trigger) for x in self.experiences}) != len(self.experiences):
            raise ValueError("atomic result requires distinct nonempty methods")


@dataclass
class ExperienceUpdate(Model):
    action: Literal["keep", "revise"]
    draft: AtomicMethod | None = None

    def __post_init__(self) -> None:
        if self.action not in ("keep", "revise") or (self.action == "revise") != (self.draft is not None):
            raise ValueError("keep requires null draft; revise requires one draft")


@dataclass
class UsageRecord(Model):
    """实际引用标准的次数，与加权P/N证据分开；同判不改变Beta。"""

    sample_id: str
    text_version: int
    positive: int = 0
    negative: int = 0
    both_true: int = 0
    both_false: int = 0

    def __post_init__(self) -> None:
        _id(self.sample_id, "usage sample_id")
        if type(self.text_version) is not int or self.text_version < 1:
            raise ValueError("invalid usage text_version")
        counts = [self.positive, self.negative, self.both_true, self.both_false]
        if any(type(n) is not int or n < 0 for n in counts) or sum(counts) == 0:
            raise ValueError("usage requires nonnegative counts and actual citations")


@dataclass
class ExperienceCategory(Model):
    category_id: str
    name: str
    description: str

    def __post_init__(self) -> None:
        for name in ("category_id", "name", "description"):
            _id(getattr(self, name), name)


@dataclass
class CategoryAssignment(Model):
    action: Literal["assign", "create", "unclassified"]
    category_id: str | None = None
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.action == "assign":
            _id(self.category_id, "category_id")
            if self.name is not None or self.description is not None:
                raise ValueError("assign cannot change category metadata")
        elif self.action == "create":
            _id(self.name, "name")
            _id(self.description, "description")
            if self.category_id is not None:
                raise ValueError("new category ID belongs to program")
        elif self.action != "unclassified" or any(x is not None for x in (self.category_id, self.name, self.description)):
            raise ValueError("invalid category assignment")


@dataclass
class CategoryDescription(Model):
    name: str
    description: str

    def __post_init__(self) -> None:
        _id(self.name, "name")
        _id(self.description, "description")


@dataclass
class OrganizationRecord(Model):
    sample_id: str
    experience_id: str | None
    category_id: str | None
    action: str
    index_version: str
    error: str | None = None
    previous_name: str | None = None
    previous_description: str | None = None
    name: str | None = None
    description: str | None = None
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    retrieved_categories: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InheritedCredit(Model):
    """A birth allocation, not observations of the child's own text."""

    parent_id: str
    event_id: str
    positive_credit: float
    negative_credit: float
    inherited_penalty: float
    evolution_penalty: float

    def __post_init__(self) -> None:
        _id(self.parent_id, "inheritance parent_id")
        _id(self.event_id, "inheritance event_id")
        for name in ("positive_credit", "negative_credit", "inherited_penalty", "evolution_penalty"):
            _number(getattr(self, name), name, minimum=0)


@dataclass
class EvolutionRecord(Model):
    event_id: str
    sample_id: str
    parent_id: str
    child_ids: list[str]
    case_ids: list[str]
    result: Literal["keep", "replace", "split", "error"]
    draft: AtomicMethod | None = None
    atomic_result: AtomicResult | None = None
    error: str | None = None
    window_credit_amount: float = 0.0
    penalty_lambda: float = 0.2
    penalty_credit: float = 0.0

    def __post_init__(self) -> None:
        for name in ("event_id", "sample_id", "parent_id"):
            _id(getattr(self, name), name)
        if len(self.case_ids) != 5 or len(set(self.case_ids)) != 5:
            raise ValueError("evolution requires five distinct downstream cases")
        if (self.result == "error") != (self.error is not None):
            raise ValueError("evolution error status disagrees")
        for name in ("window_credit_amount", "penalty_lambda", "penalty_credit"):
            _number(getattr(self, name), name, minimum=0)
        if self.penalty_lambda != 0.2:
            raise ValueError("evolution penalty lambda is fixed at 0.2")
        expected_penalty = self.penalty_lambda * self.window_credit_amount if self.child_ids else 0
        if abs(self.penalty_credit - expected_penalty) > 1e-9:
            raise ValueError("evolution penalty disagrees with window credit")
        expected = {"keep": 0, "error": 0, "replace": 1}
        if (len(set(self.child_ids)) != len(self.child_ids) or self.parent_id in self.child_ids
                or (self.result in expected and len(self.child_ids) != expected[self.result])
                or (self.result == "split" and len(self.child_ids) < 2)):
            raise ValueError("invalid evolution children")


@dataclass
class Experience(Model):
    experience_id: str
    content: str
    trigger: str
    alpha: float
    beta: float
    source_trajectories: list[str]
    evidence_history: list[EvidenceRecord]
    cases: list[ExperienceCase] = field(default_factory=list)
    text_version: int = 1
    update_history: list[EvolutionRecord] = field(default_factory=list)
    category_id: str | None = None
    usage_history: list[UsageRecord] = field(default_factory=list)
    status: Literal["active", "retired"] = "active"
    origin_kind: Literal["candidate", "evolution"] = "candidate"
    origin_refs: list[str] = field(default_factory=list)
    created_by_event_id: str | None = None
    retired_by_event_id: str | None = None
    last_consumed_feedback_seq: int = 0
    last_attempted_feedback_seq: int = 0
    inherited_credit: InheritedCredit | None = None

    def __post_init__(self) -> None:
        for name in ("experience_id", "content", "trigger"):
            _id(getattr(self, name), name)
        _number(self.alpha, "alpha", minimum=1)
        _number(self.beta, "beta", minimum=1)
        if self.category_id is not None:
            _id(self.category_id, "category_id")
        if type(self.text_version) is not int or self.text_version < 1:
            raise ValueError("invalid experience text_version")
        if self.status not in ("active", "retired") or self.origin_kind not in ("candidate", "evolution"):
            raise ValueError("invalid experience lifecycle")
        if (self.origin_kind == "evolution") != (self.inherited_credit is not None):
            raise ValueError("only evolution versions require inherited credit")
        if (self.status == "retired") != (self.retired_by_event_id is not None):
            raise ValueError("retirement requires an evolution event")
        if self.status == "retired" and self.category_id is not None:
            raise ValueError("retired experience cannot remain a category member")
        if self.text_version != 1 or any(v.text_version != 1 for v in [*self.cases, *self.evidence_history, *self.usage_history]):
            raise ValueError("each immutable version has a new ID and local text_version 1")
        if len({c.sample_id for c in self.cases}) != len(self.cases):
            raise ValueError("experience cases must have distinct tasks")
        n = len(self.downstream_cases)
        if not 0 <= self.last_consumed_feedback_seq <= self.last_attempted_feedback_seq <= n:
            raise ValueError("invalid feedback sequence")

    @property
    def downstream_cases(self) -> list[ExperienceCase]:
        return [c for c in self.cases if c.source_id == self.experience_id]

    @property
    def working_cases(self) -> list[ExperienceCase]:
        return self.downstream_cases[-5:]

    @property
    def pending_feedback_count(self) -> int:
        return len(self.downstream_cases) - self.last_consumed_feedback_seq

    @property
    def update_due(self) -> bool:
        return (self.status == "active" and self.pending_feedback_count >= 5
                and len(self.downstream_cases) > self.last_attempted_feedback_seq)

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence_amount(self) -> float:
        return self.positive_credit + self.negative_credit

    @property
    def positive_credit(self) -> float:
        return sum(v.positive_credit for v in self.evidence_history) + (
            self.inherited_credit.positive_credit if self.inherited_credit else 0)

    @property
    def negative_credit(self) -> float:
        return sum(v.negative_credit for v in self.evidence_history) + (
            self.inherited_credit.negative_credit if self.inherited_credit else 0)

    @property
    def penalty_credit(self) -> float:
        return (self.inherited_credit.inherited_penalty + self.inherited_credit.evolution_penalty
                if self.inherited_credit else 0)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experience:
        values = dict(data)
        confidence = values.pop("confidence", None)
        result = super().from_dict(values)
        if confidence is not None and (
            type(confidence) not in (float, int)
            or not isfinite(confidence)
            or abs(confidence - result.confidence) > 1e-9
        ):
            raise ValueError("Experience: confidence disagrees with alpha/beta")
        return result


@dataclass
class Criterion(Model):
    criterion_id: str
    question: str
    weight: float
    source_experiences: list[str]

    def __post_init__(self) -> None:
        _id(self.criterion_id, "criterion_id")
        _id(self.question, "question")
        _number(self.weight, "weight")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if (
            not isinstance(self.source_experiences, list)
            or any(type(key) is not str or not key for key in self.source_experiences)
            or len(set(self.source_experiences)) != len(self.source_experiences)
        ):
            raise ValueError("source_experiences must be distinct nonempty IDs")


@dataclass
class Rubric(Model):
    rubric_id: str
    criteria: list[Criterion]

    def __post_init__(self) -> None:
        _id(self.rubric_id, "rubric_id")
        if not 2 <= len(self.criteria) <= 8:
            raise ValueError("rubric must have 2..8 criteria")
        if len({c.criterion_id for c in self.criteria}) != len(self.criteria):
            raise ValueError("duplicate criterion_id in rubric")


@dataclass
class CriterionJudgment(Model):
    """一条标准对偏好对两侧回答的Boolean判断与原始依据。

    criterion_id: 被判断的标准ID；chosen/rejected: 对应回答是否满足该标准；
    chosen_reason/rejected_reason: 各侧原始判断依据，不等于事实认证。
    """

    criterion_id: str
    chosen: bool
    rejected: bool
    chosen_reason: str
    rejected_reason: str

    def __post_init__(self) -> None:
        _id(self.criterion_id, "criterion_id")
        if type(self.chosen) is not bool or type(self.rejected) is not bool:
            raise ValueError("criterion judgments must be Boolean")
        _id(self.chosen_reason, "chosen_reason")
        _id(self.rejected_reason, "rejected_reason")


@dataclass
class RubricJudgment(Model):
    """一次完整 rubric 的评价结果，不回写原始 criterion。

    rubric_id: 被评价的 rubric ID；normalized_weights: criterion ID 到权重的映射；
    criterion_judgments: 两侧回答的逐条 Boolean 判断；
    chosen_score/rejected_score: 两侧加权总分；
    margin: chosen_score 与 rejected_score 之差；success: margin 是否大于零。
    """

    rubric_id: str
    normalized_weights: dict[str, float]
    criterion_judgments: list[CriterionJudgment]
    chosen_score: float
    rejected_score: float
    margin: float
    success: bool

    def __post_init__(self) -> None:
        _id(self.rubric_id, "rubric_id")
        for name in ("chosen_score", "rejected_score", "margin"):
            _number(getattr(self, name), name)
        for weight in self.normalized_weights.values():
            _number(weight, "normalized_weight", minimum=0)
        if type(self.success) is not bool or self.success != (self.margin > 0):
            raise ValueError("success must equal margin > 0")


@dataclass
class CandidateExperience(Model):
    candidate_id: str
    content: str
    trigger: str
    experience_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_id", "content", "trigger"):
            _id(getattr(self, name), name)
        if self.experience_id is not None:
            _id(self.experience_id, "experience_id")


@dataclass
class CriticProposal(Model):
    analysis: str
    candidate: CandidateExperience | None

    def __post_init__(self) -> None:
        _id(self.analysis, "critic analysis")


@dataclass
class ProposalAttempt(Model):
    round_number: int
    analysis: str | None = None
    candidate: CandidateExperience | None = None
    error: str | None = None
    atomic_candidates: list[CandidateExperience] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.round_number not in (1, 2):
            raise ValueError("proposal budget exceeded")


@dataclass
class ApplicationAttempt(Model):
    round_number: int
    candidate_id: str
    rubric: Rubric | None = None
    judgment: RubricJudgment | None = None
    adopted: bool | None = None
    adopted_criterion_ids: list[str] = field(default_factory=list)
    credit: Credit | None = None
    eligible: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.round_number not in (1, 2):
            raise ValueError("application budget exceeded")
        _id(self.candidate_id, "candidate_id")
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be Boolean")
        if self.rubric is None:
            if (
                self.judgment is not None
                or self.adopted is not None
                or self.adopted_criterion_ids
                or self.credit is not None
                or self.eligible
            ):
                raise ValueError(
                    "invalid output must not invent adoption, judgment or credit"
                )
        else:
            used = [
                c.criterion_id
                for c in self.rubric.criteria
                if self.candidate_id in c.source_experiences
            ]
            if self.adopted is not bool(used) or self.adopted_criterion_ids != used:
                raise ValueError("adoption must match actual criterion sources")
            if self.judgment is None:
                if self.credit is not None or self.eligible:
                    raise ValueError("credit requires an effective judgment")
            elif (
                self.judgment.rubric_id != self.rubric.rubric_id or self.credit is None
            ):
                raise ValueError(
                    "application judgment and credit must match its rubric"
                )
            else:
                from verge.calculator import positive_experience_sources, calculate_experience_credits
                expected = calculate_experience_credits(self.rubric, self.judgment).get(self.candidate_id, Credit())
                if self.credit != expected or self.eligible != (
                    self.candidate_id in positive_experience_sources(self.rubric, self.judgment)
                ):
                    raise ValueError("eligibility and credit must match actual source judgments")


@dataclass
class TrajectoryRecord(Model):
    sample_id: str
    state: str = "in_progress"
    retrieved_experience_ids: list[str] = field(default_factory=list)
    retrieved_units: list[dict[str, Any]] = field(default_factory=list)
    selected_experiences: list[dict[str, Any]] = field(default_factory=list)
    initial_rubric: Rubric | None = None
    initial_judgment: RubricJudgment | None = None
    downstream_credits: dict[str, Credit] = field(default_factory=dict)
    proposal_attempts: list[ProposalAttempt] = field(default_factory=list)
    application_attempts: list[ApplicationAttempt] = field(default_factory=list)
    sample_error: dict[str, str] | None = None
    judge_error: str | None = None
    admitted_candidate_ids: list[str] = field(default_factory=list)
    admitted_experience_ids: list[str] = field(default_factory=list)
    experience_updates: dict[str, EvolutionRecord] = field(default_factory=dict)
    category_organization: list[OrganizationRecord] = field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = 10

    def __post_init__(self) -> None:
        _id(self.sample_id, "sample_id")
        if self.schema_version != 10:
            raise ValueError("unsupported trajectory schema")
        if self.state not in {
            "in_progress",
            "initial_success",
            "initial_failure",
            "application_success",
            "success_without_candidate",
            "no_candidate",
            "validation_exhausted",
            "invalid_application_exhausted",
            "sample_error",
            "judge_error_initial",
            "judge_error_application",
        }:
            raise ValueError("unknown trajectory state")
        pairs = [p.round_number for p in self.proposal_attempts]
        if pairs != sorted(set(pairs)) or len(pairs) > 2:
            raise ValueError(
                "proposal attempts must be unique and ordered within budget"
            )
        rounds = [a.round_number for a in self.application_attempts]
        if rounds != sorted(rounds) or any(n not in (1, 2) for n in rounds):
            raise ValueError("application rounds must be ordered within budget")
        candidates = {c.candidate_id: (p.round_number, c) for p in self.proposal_attempts for c in p.atomic_candidates}
        if len(candidates) != sum(len(p.atomic_candidates) for p in self.proposal_attempts):
            raise ValueError("atomic candidate IDs must be unique")
        app_ids = [a.candidate_id for a in self.application_attempts]
        if len(set(app_ids)) != len(app_ids):
            raise ValueError("duplicate candidate application")
        for app in self.application_attempts:
            if app.candidate_id not in candidates or candidates[app.candidate_id][0] != app.round_number:
                raise ValueError("application requires this round's atomic candidate")
            app.__post_init__()
        admitted = [c for p in self.proposal_attempts for c in p.atomic_candidates if c.experience_id]
        if ([c.candidate_id for c in admitted] != self.admitted_candidate_ids
                or [c.experience_id for c in admitted] != self.admitted_experience_ids):
            raise ValueError("admission IDs disagree with candidates")
        if admitted:
            if self.sample_error or self.judge_error:
                raise ValueError("error end cannot admit candidates")
            eligible = [a for a in self.application_attempts if a.eligible]
            latest = max(a.round_number for a in eligible)
            expected = [a.candidate_id for a in eligible if a.round_number == latest]
            if self.admitted_candidate_ids != expected:
                raise ValueError("admission must select latest eligible round set")
        for number in set(rounds):
            batch = [a for a in self.application_attempts if a.round_number == number]
            if any((a.rubric, a.judgment) != (batch[0].rubric, batch[0].judgment) for a in batch):
                raise ValueError("atomic candidates share one application rubric and judgment per round")
        if self.sample_error is not None and (
            set(self.sample_error) != {"stage", "type", "message"}
            or any(
                not isinstance(v, str) or not v.strip()
                for v in self.sample_error.values()
            )
        ):
            raise ValueError("invalid sample_error")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryRecord:
        if not isinstance(data, dict) or data.get("schema_version") != 10:
            raise ValueError("TrajectoryRecord requires explicit schema_version 10")
        return super().from_dict(data)
