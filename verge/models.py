"""VERGE candidate-validation 的领域数据契约；数值计算和持久化由后续模块负责。"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
import json
import hashlib
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
        return {
            field.name: _encode(getattr(self, field.name)) for field in fields(self)
        }

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
        required = {
            name
            for name, item in model_fields.items()
            if item.default is MISSING and item.default_factory is MISSING
        }
        if not required <= data.keys() or not data.keys() <= names:
            raise ValueError(
                f"{cls.__name__}: missing {sorted(required - data.keys())}, unknown {sorted(data.keys() - names)}"
            )
        try:
            kwargs = {}
            for name in data:
                try:
                    kwargs[name] = _decode(data[name], annotations[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name}: {exc}") from exc
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{cls.__name__}: {exc}") from exc


def _id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _number(value: float, name: str, *, minimum: float | None = None) -> None:
    if (
        type(value) not in (int, float)
        or not isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(
            f"{name} must be a finite number"
            + (f" >= {minimum}" if minimum is not None else "")
        )


def task_identity(task: str) -> str:
    return hashlib.sha256(" ".join(task.split()).encode("utf8")).hexdigest()


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
        _id(self.task, "task")

    @property
    def task_id(self):
        return task_identity(self.task)


@dataclass
class Method(Model):
    trigger: str
    principle: str

    def __post_init__(self):
        _id(self.trigger, "trigger")
        _id(self.principle, "principle")


@dataclass
class Criterion(Model):
    criterion_id: str
    question: str
    weight: float
    source_experiences: list[str] = field(default_factory=list)

    def __post_init__(self):
        _id(self.criterion_id, "criterion_id")
        _id(self.question, "question")
        _number(self.weight, "weight", minimum=0)
        if len(self.source_experiences) > 1:
            raise ValueError("one criterion can cite at most one experience")
        for value in self.source_experiences:
            _id(value, "experience reference")


@dataclass
class Rubric(Model):
    rubric_id: str
    criteria: list[Criterion]
    analysis: str = ""

    def __post_init__(self):
        _id(self.rubric_id, "rubric_id")
        if not 2 <= len(self.criteria) <= 8 or not any(
            c.weight > 0 for c in self.criteria
        ):
            raise ValueError("rubric needs 2..8 criteria and a positive weight budget")
        if len({c.criterion_id for c in self.criteria}) != len(self.criteria):
            raise ValueError("duplicate criterion IDs")


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
class CheckerDetail(Model):
    criterion_id: str
    status: Literal["correct", "error", "unknown"]
    evidence: str

    def __post_init__(self):
        _id(self.criterion_id, "criterion_id")
        _id(self.evidence, "detail evidence")
        if self.status not in ("correct", "error", "unknown"):
            raise ValueError("invalid detail status")


@dataclass
class CheckerReport(Model):
    stage: Literal["selector", "generator", "judge"]
    status: Literal["correct", "error", "unknown", "failed"]
    evidence: str
    details: list[CheckerDetail] = field(default_factory=list)

    def __post_init__(self):
        if self.stage not in ("selector", "generator", "judge") or self.status not in (
            "correct",
            "error",
            "unknown",
            "failed",
        ):
            raise ValueError("invalid checker stage/status")
        _id(self.evidence, "checker evidence")
        if len({d.criterion_id for d in self.details}) != len(self.details):
            raise ValueError("duplicate checker detail")
        if any(d.status == "error" for d in self.details) and self.status != "error":
            raise ValueError("detail error requires top-level error")
        if (
            any(d.status == "unknown" for d in self.details)
            and self.status == "correct"
        ):
            raise ValueError("unknown detail cannot be reported correct")


@dataclass
class ApplicationRecord(Model):
    sample: PreferenceSample
    version_id: str
    method: Method
    phase: Literal["downstream", "source_validation", "replay"]
    rubric: Rubric
    judgment: RubricJudgment
    feedback: Literal["positive", "reverse", "tie", "unused"]
    positive_weight: float
    reverse_weight: float
    outcomes: dict[str, int]
    reports: list[CheckerReport] = field(default_factory=list)
    observation: Literal["correct", "error", "unknown", "unavailable"] = "unavailable"
    q_before: float | None = None
    q_after: float | None = None
    parameter_version: str | None = None
    q_log_odds_before: float | None = None
    q_log_odds_after: float | None = None
    likelihoods: dict[str, float] = field(default_factory=dict)
    update_reason: str = "not_evaluated"
    parameter_digest: str | None = None
    recalled: bool = False
    selected: bool = False
    cited: bool = False
    retrieval_rank: int | None = None
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    no_evidence_reason: str | None = None
    round_number: int = 0

    def __post_init__(self):
        _id(self.version_id, "version_id")
        if len({x.stage for x in self.reports}) != len(self.reports):
            raise ValueError("duplicate checker stage")
        if self.phase not in ("downstream", "source_validation", "replay"):
            raise ValueError("invalid evidence phase")
        from verge.calculator import validate_rubric_judgment, _weight_ratios

        validate_rubric_judgment(self.rubric, self.judgment)
        answers = {j.criterion_id: j for j in self.judgment.criterion_judgments}
        weights = _weight_ratios(self.rubric)
        pos = neg = 0
        used = False
        counts = dict(positive=0, reverse=0, both_true=0, both_false=0)
        for c in self.rubric.criteria:
            if self.version_id not in c.source_experiences:
                continue
            j = answers[c.criterion_id]
            used |= weights[c.criterion_id] > 0
            kind = (
                "positive"
                if j.chosen and not j.rejected
                else (
                    "reverse"
                    if j.rejected and not j.chosen
                    else "both_true" if j.chosen else "both_false"
                )
            )
            counts[kind] += 1
            if kind == "positive":
                pos += weights[c.criterion_id]
            if kind == "reverse":
                neg += weights[c.criterion_id]
        f = (
            "unused"
            if not used
            else ("positive" if pos > neg else "reverse" if neg > pos else "tie")
        )
        if (
            self.cited != bool(sum(counts.values()))
            or (self.cited and not self.selected)
            or (self.selected and not self.recalled)
        ):
            raise ValueError("inconsistent recalled/selected/cited state")
        if type(self.round_number) is not int or self.round_number < 0:
            raise ValueError("invalid application round")
        if self.retrieval_rank is not None and (
            type(self.retrieval_rank) is not int
            or self.retrieval_rank < 1
            or not self.recalled
        ):
            raise ValueError("invalid retrieval rank")
        if (
            self.feedback != f
            or self.positive_weight != float(pos)
            or self.reverse_weight != float(neg)
            or self.outcomes != counts
        ):
            raise ValueError(
                "application feedback must match actual cited Boolean evidence"
            )


@dataclass
class ExperienceReview(Model):
    approved: bool
    reason: str

    def __post_init__(self):
        if type(self.approved) is not bool:
            raise ValueError("approved must be JSON boolean")
        _id(self.reason, "review reason")


@dataclass
class Experience(Model):
    experience_id: str
    method: Method
    source_sample_id: str
    source_task_id: str | None = None
    source_sample: PreferenceSample | None = None
    admission_review: ExperienceReview | None = None
    parent_id: str | None = None
    status: Literal["candidate", "active", "retired"] = "active"
    q_prior: float | None = None
    q_posterior: float | None = None
    q_log_odds: float | None = None
    parameter_version: str | None = None
    applications: list[ApplicationRecord] = field(default_factory=list)
    last_checked_count: int = 0
    consumed_error_tasks: dict[str, list[str]] = field(default_factory=dict)
    parameter_digest: str | None = None
    created_by_event_id: str | None = None
    retired_by_event_id: str | None = None

    def __post_init__(self):
        _id(self.experience_id, "experience_id")
        _id(self.source_sample_id, "source_sample_id")
        if self.status not in ("candidate", "active", "retired"):
            raise ValueError("invalid version status")
        for q in (self.q_prior, self.q_posterior):
            if q is not None and (
                type(q) not in (int, float) or not isfinite(q) or not 0 <= q <= 1
            ):
                raise ValueError("quality probability must be in [0,1] or null")
        if self.q_log_odds is not None:
            _number(self.q_log_odds, "q_log_odds")
        if self.q_prior is not None and (
            type(self.q_prior) not in (int, float) or not 0 < self.q_prior < 1
        ):
            raise ValueError("version prior must be strictly between 0 and 1")
        if (self.q_posterior is None) != (self.parameter_version is None):
            raise ValueError("quality requires parameter provenance")
        if self.status == "retired" and not self.retired_by_event_id:
            raise ValueError("retirement needs event")
        keys = [(a.sample.sample_id, a.phase) for a in self.applications]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task/phase evidence")
        if not 0 <= self.last_checked_count <= len(self.downstream):
            raise ValueError("invalid check cursor")

    @property
    def downstream(self):
        unique = {}
        for a in self.applications:
            if (
                a.phase == "downstream"
                and a.sample.sample_id != self.source_sample_id
                and a.sample.task_id != self.source_task_id
                and a.update_reason not in ("duplicate", "duplicate_task")
                and a.feedback in ("positive", "reverse", "tie")
            ):
                unique.setdefault(a.sample.task_id, a)
        return list(unique.values())

    @property
    def evidence_counts(self):
        return {
            phase: sum(
                a.phase == phase and a.update_reason == "updated"
                for a in self.applications
            )
            for phase in ("source_validation", "downstream", "replay")
        }


@dataclass
class CriticProposal(Model):
    analysis: str
    candidates: list[Experience]


@dataclass
class ExperienceUpdate(Model):
    action: Literal["keep", "revise"]
    method: Method | None = None
    target_fields: list[str] = field(default_factory=list)
    reason: str = ""

    def __post_init__(self):
        if self.action not in ("keep", "revise") or (self.action == "revise") != (
            self.method is not None
        ):
            raise ValueError(
                "keep requires no method; revise requires exactly one method"
            )
        if set(self.target_fields) - {"trigger", "principle"}:
            raise ValueError("unknown revision field")
        if self.action == "revise" and not self.target_fields:
            raise ValueError("revise requires target_fields")
        _id(self.reason, "update reason")


@dataclass
class ReplayResult(Model):
    sample_id: str
    old_score: int | None
    new_score: int | None
    old_application: ApplicationRecord | None = None
    new_application: ApplicationRecord | None = None
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class RevisionRecord(Model):
    event_id: str
    parent_id: str
    trigger_sample_id: str
    reasons: list[str]
    case_ids: list[str]
    case_kinds: dict[str, str]
    case_refs: list[dict[str, Any]] = field(default_factory=list)
    action: str = "pending"
    update: ExperienceUpdate | None = None
    candidate: Experience | None = None
    replays: list[ReplayResult] = field(default_factory=list)
    accepted: bool = False
    decision: str = ""
    error: str | None = None


@dataclass
class ProposalAttempt(Model):
    round_number: int
    proposal: CriticProposal | None = None
    review: ExperienceReview | None = None
    rubric: Rubric | None = None
    judgment: RubricJudgment | None = None
    applications: list[ApplicationRecord] = field(default_factory=list)
    error: str | None = None


@dataclass
class TrajectoryRecord(Model):
    sample_id: str
    state: str = "in_progress"
    sample: PreferenceSample | None = None
    split: str = "train"
    retrieved_experience_ids: list[str] = field(default_factory=list)
    selected_experiences: list[dict[str, Any]] = field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    initial_rubric: Rubric | None = None
    initial_judgment: RubricJudgment | None = None
    applications: list[ApplicationRecord] = field(default_factory=list)
    proposal_attempts: list[ProposalAttempt] = field(default_factory=list)
    admitted_experience_ids: list[str] = field(default_factory=list)
    experience_updates: list[RevisionRecord] = field(default_factory=list)
    sample_error: dict[str, str] | None = None
    schema_version: int = 16

    def __post_init__(self):
        _id(self.sample_id, "sample_id")
        if self.schema_version != 16:
            raise ValueError("unsupported trajectory schema; no V1 compatibility")

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 16:
            raise ValueError("explicit trajectory schema 16 required")
        return super().from_dict(data)
