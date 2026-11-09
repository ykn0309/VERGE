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
    if isinstance(value, (list, tuple)):
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


@dataclass(frozen=True)
class Method(Model):
    trigger: str
    principle: str
    application_guide: str

    def __post_init__(self):
        for name in ("trigger", "principle", "application_guide"):
            _id(getattr(self, name), name)


@dataclass
class Criterion(Model):
    criterion_id: str
    question: str
    weight: float

    def __post_init__(self):
        _id(self.criterion_id, "criterion_id")
        _id(self.question, "question")
        _number(self.weight, "weight", minimum=0)


@dataclass
class Rubric(Model):
    rubric_id: str
    criteria: list[Criterion]
    analysis: str = ""

    def __post_init__(self):
        _id(self.rubric_id, "rubric_id")
        if not 1 <= len(self.criteria) <= 8 or not any(
            c.weight > 0 for c in self.criteria
        ):
            raise ValueError("rubric needs 1..8 criteria and a positive weight budget")
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


@dataclass(frozen=True)
class Experience(Model):
    experience_id: str
    method: Method
    source_task_id: str
    parent_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        _id(self.experience_id, "experience_id")
        _id(self.source_task_id, "source_task_id")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("duplicate parent IDs")
        for parent in self.parent_ids:
            _id(parent, "parent_id")


@dataclass
class ExecutionPath(Model):
    provided_experience_ids: list[str]
    rubric: Rubric | None = None
    judgment: RubricJudgment | None = None
    error: str | None = None

    @property
    def correct(self):
        return (
            self.judgment.success
            if self.judgment is not None and self.error is None
            else None
        )


@dataclass
class CandidateAttempt(Model):
    experience: Experience
    path: ExecutionPath
    admitted: bool
    reason: str
    admission_subset: list[str] = field(default_factory=list)


@dataclass
class CriticRound(Model):
    round_number: int
    analysis: str
    attempts: list[CandidateAttempt]
    error: str | None = None
    subset_paths: list[ExecutionPath] = field(default_factory=list)


@dataclass
class TrajectoryRecord(Model):
    sample: PreferenceSample
    retrieval: dict[str, Any]
    selection_reasons: dict[str, str]
    baseline: ExecutionPath | None
    actual: ExecutionPath | None
    critic_rounds: list[CriticRound] = field(default_factory=list)
    error: str | None = None
    subset_paths: list[ExecutionPath] = field(default_factory=list)

    @property
    def used_path(self):
        return self.actual if self.actual is not None else self.baseline
