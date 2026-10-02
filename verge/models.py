"""VERGE V1 的领域数据契约；数值计算和持久化由后续模块负责。"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
import json
from math import isfinite
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints


ModelT = TypeVar("ModelT", bound="Model")
OperationType = Literal["ADD", "DELETE", "MODIFY"]
FusionDecision = Literal["MERGE", "ADD"]
AdmissionDecision = Literal["PASS", "FAIL"]


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


@dataclass
class EvidenceRecord(Model):
    """一次可追溯的可靠性证据。

    sample_id: 证据所属样本；evidence_type: 修订或下游信用来源；
    credit: 带符号的贡献；evidence_weight: 样本证据权重。
    V1 不在单条证据中保存更细的来源标识。
    """

    sample_id: str
    evidence_type: Literal["revision_credit", "downstream_credit"]
    credit: float
    evidence_weight: float

    def __post_init__(self) -> None:
        _id(self.sample_id, "sample_id")
        _number(self.credit, "credit")
        _number(self.evidence_weight, "evidence_weight", minimum=0)


@dataclass
class Experience(Model):
    """可复用的 rubric 生成原则及其累计证据。

    experience_id: 经验唯一标识；content: 生成原则；trigger: 适用条件；
    alpha/beta: 正/负证据累积（含基础先验）；
    source_trajectories: 生成或再次发现该经验的样本 ID；
    evidence_history: 可追溯的修订与下游证据记录。
    confidence 和 evidence_amount 均从 alpha/beta 派生，不独立存储。
    """

    experience_id: str
    content: str
    trigger: str
    alpha: float
    beta: float
    source_trajectories: list[str]
    evidence_history: list[EvidenceRecord]

    def __post_init__(self) -> None:
        _id(self.experience_id, "experience_id")
        _number(self.alpha, "alpha", minimum=1)
        _number(self.beta, "beta", minimum=1)

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence_amount(self) -> float:
        return self.alpha + self.beta - 2

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experience:
        # 历史记录中的 confidence 仅供读取和核验，不作为独立状态恢复。
        values = dict(data)
        has_confidence = "confidence" in values
        confidence = values.pop("confidence", None)
        result = super().from_dict(values)
        if has_confidence and (type(confidence) not in (float, int) or
                               not isfinite(confidence) or
                               abs(confidence - result.confidence) > 1e-9):
            raise ValueError("Experience: confidence disagrees with alpha/beta")
        return result


@dataclass
class Criterion(Model):
    """一条原子 Boolean 评价标准。

    criterion_id: 标准唯一标识；question: 可回答是/否的判断问题；
    importance_score: 原始重要性整数分数（1–10）；
    source_experiences: 实质影响初始标准的经验 ID，可为空；
    只在 initial rubric 的已有经验信用中使用，修订计算忽略该字段。
    归一化权重和贡献由 Calculator 计算，不写入此对象。
    """

    criterion_id: str
    question: str
    importance_score: int
    source_experiences: list[str]

    def __post_init__(self) -> None:
        _id(self.criterion_id, "criterion_id")
        if type(self.importance_score) is not int or not 1 <= self.importance_score <= 10:
            raise ValueError("importance_score must be an integer from 1 to 10")


@dataclass
class Rubric(Model):
    """一组完整的评价标准。

    rubric_id: 当前 rubric 的唯一标识；base_rubric_id: 修订起点的 rubric ID，
    初始 rubric 为 None；criteria: 组成 rubric 的标准列表。
    """

    rubric_id: str
    base_rubric_id: str | None
    criteria: list[Criterion]

    def __post_init__(self) -> None:
        _id(self.rubric_id, "rubric_id")
        if self.base_rubric_id is not None:
            _id(self.base_rubric_id, "base_rubric_id")
        ids = [criterion.criterion_id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate criterion_id in rubric")


@dataclass
class CriterionJudgment(Model):
    """一条标准对偏好对两侧回答的 Boolean 判断。

    criterion_id: 被判断的标准 ID；chosen/rejected: 对应回答是否满足该标准。
    """

    criterion_id: str
    chosen: bool
    rejected: bool

    def __post_init__(self) -> None:
        _id(self.criterion_id, "criterion_id")
        if type(self.chosen) is not bool or type(self.rejected) is not bool:
            raise ValueError("criterion judgments must be Boolean")


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
class RevisionOperation(Model):
    """生成与执行共用的原子修订操作。

    operation_id: 当前修订尝试内的操作标识；operation_type: ADD、DELETE 或 MODIFY；
    target_criterion_id: DELETE/MODIFY 的目标 ID，ADD 时为空；
    new_or_updated_criterion: ADD 的新标准或 MODIFY 的更新后标准，DELETE 时为空；
    before_criterion: Critic 留空；Executor 从初始 rubric 补全 DELETE/MODIFY 的旧标准。
    """

    operation_id: str
    operation_type: OperationType
    target_criterion_id: str | None
    new_or_updated_criterion: Criterion | None
    before_criterion: Criterion | None = None

    def __post_init__(self) -> None:
        _id(self.operation_id, "operation_id")
        if self.operation_type not in ("ADD", "DELETE", "MODIFY"):
            raise ValueError("operation_type must be ADD, DELETE or MODIFY")
        if self.operation_type == "ADD":
            if (self.target_criterion_id is not None or self.new_or_updated_criterion is None
                    or self.before_criterion is not None):
                raise ValueError("ADD needs only new_or_updated_criterion")
        elif self.operation_type == "DELETE":
            _id(self.target_criterion_id, "target_criterion_id")
            if self.new_or_updated_criterion is not None:
                raise ValueError("DELETE must not include a new criterion")
        else:
            _id(self.target_criterion_id, "target_criterion_id")
            if (self.new_or_updated_criterion is None or
                    self.new_or_updated_criterion.criterion_id != self.target_criterion_id):
                raise ValueError("MODIFY must preserve target_criterion_id")
        if (self.before_criterion is not None and
                self.before_criterion.criterion_id != self.target_criterion_id):
            raise ValueError("before_criterion must match target_criterion_id")


@dataclass
class CandidateExperience(Model):
    """从成功修订中反思出的待准入经验。

    candidate_id: 候选唯一标识；content: 可迁移的生成原则；
    trigger: 原则适用的任务条件；linked_operation_ids/credit: 历史记录解码字段；
    当前方法不赋值，分别固定为空列表/null；
    admission/admission_reason: 准入判断及模型给出的理由；
    fusion/experience_id: 融合决策及最终经验 ID。
    """

    candidate_id: str
    content: str
    trigger: str
    linked_operation_ids: list[str] = field(default_factory=list)
    credit: float | None = None
    admission: AdmissionDecision | None = None
    fusion: FusionDecision | None = None
    experience_id: str | None = None
    admission_reason: str | None = None

    def __post_init__(self) -> None:
        _id(self.candidate_id, "candidate_id")
        if len(set(self.linked_operation_ids)) != len(self.linked_operation_ids):
            raise ValueError("linked_operation_ids must not contain duplicates")
        for operation_id in self.linked_operation_ids:
            _id(operation_id, "linked_operation_ids item")
        if self.credit is not None:
            _number(self.credit, "credit")
            if not self.linked_operation_ids:
                raise ValueError("credit requires linked_operation_ids")
        if self.admission not in (None, "PASS", "FAIL"):
            raise ValueError("admission must be PASS or FAIL")
        if self.admission_reason is not None:
            if (self.admission is None or not isinstance(self.admission_reason, str)
                    or not self.admission_reason.strip()):
                raise ValueError("admission_reason requires a decision and non-empty text")
        if self.fusion not in (None, "MERGE", "ADD"):
            raise ValueError("fusion must be MERGE or ADD")
        if self.fusion is None:
            if self.experience_id is not None:
                raise ValueError("experience_id requires fusion")
        elif self.admission != "PASS":
            raise ValueError("fusion requires PASS admission")
        else:
            _id(self.experience_id, "experience_id")


@dataclass
class CriticFeedback(Model):
    """私有诊断与唯一交给 Generator 的公开建议。"""
    analysis: str
    improvement_advice: str

    def __post_init__(self) -> None:
        for name in ("analysis", "improvement_advice"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"critic {name} must be non-empty")


@dataclass
class RevisionAttempt(Model):
    """一次接续当前 rubric 的 Generator 对话修订。

    attempt_number: 第 1 或第 2 次；analysis: 新流程的私有失败分析；
    feedback: Critic 的公开 improvement_advice；
    operations: 历史字段，当前固定空列表；
    rubric/judgment: 修订产生的完整 rubric 及其评价结果；
    validation_failures: 有效修订前的结构校验错误，便于审计纠错次数与上下文。
    """

    attempt_number: int
    feedback: str | None
    operations: list[RevisionOperation]
    rubric: Rubric
    judgment: RubricJudgment
    validation_failures: list[str] = field(default_factory=list)
    analysis: str | None = None
    previous_rubric: Rubric | None = None
    previous_judgment: RubricJudgment | None = None

    def __post_init__(self) -> None:
        if self.analysis is not None and (not isinstance(self.analysis, str) or not self.analysis.strip()):
            raise ValueError("revision analysis must be non-empty")
        if type(self.attempt_number) is not int or self.attempt_number not in (1, 2):
            raise ValueError("attempt_number must be 1 or 2")
        if not isinstance(self.validation_failures, list) or any(
                type(item) is not str or not item for item in self.validation_failures):
            raise ValueError("validation_failures must be non-empty error strings")


@dataclass
class InvalidRevisionAttempt(Model):
    """Critic 格式错误或 Generator 完整修订校验耗尽，无有效新 rubric/Judge。"""

    attempt_number: int
    feedback: str | None
    validation_failures: list[str]
    analysis: str | None = None

    def __post_init__(self) -> None:
        if self.analysis is not None and (not isinstance(self.analysis, str) or not self.analysis.strip()):
            raise ValueError("revision analysis must be non-empty")
        if type(self.attempt_number) is not int or self.attempt_number not in (1, 2):
            raise ValueError("attempt_number must be 1 or 2")
        if not self.validation_failures or any(
                type(item) is not str or not item for item in self.validation_failures):
            raise ValueError("invalid revision must have validation errors")


@dataclass
class TrajectoryRecord(Model):
    """单个样本的阶段结果；记忆更新在相应阶段直接写入。

    sample_id/state: 样本 ID 与处理阶段；retrieved_experience_ids /
    selected_experience_ids: 召回和最终选用的经验 ID；
    initial_rubric/initial_judgment: 初始 rubric 及其判断，尚未生成时为空；
    revision_attempts: 最多两次同基线修订的操作及结果；
    reflected_candidates: 成功修订反思出的候选，后续直接补全绑定、信用和决策；
    operation_contributions: 成功操作 ID 到直接收益的映射；
    invalid_revision_attempts: 条数不合法、未进入 Judge 的修订尝试及校验记录。
    """

    sample_id: str
    state: str
    retrieved_experience_ids: list[str]
    selected_experience_ids: list[str]
    initial_rubric: Rubric | None
    initial_judgment: RubricJudgment | None
    revision_attempts: list[RevisionAttempt]
    reflected_candidates: list[CandidateExperience]
    operation_contributions: dict[str, float]
    invalid_revision_attempts: list[InvalidRevisionAttempt] = field(default_factory=list)
    judge_error: str | None = None
    sample_error: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _id(self.sample_id, "sample_id")
        if self.sample_error is not None:
            if (self.state != "sample_error" or set(self.sample_error) != {"stage", "type", "message"}
                    or any(not isinstance(v, str) or not v.strip() for v in self.sample_error.values())):
                raise ValueError("sample_error requires stage, type, message and sample_error state")
        if self.judge_error is not None and (not isinstance(self.judge_error, str)
                                             or not self.judge_error.strip()):
            raise ValueError("judge_error must be a non-empty message")
