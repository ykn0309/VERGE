"""单样本顺序 Runner：用模块契约串联训练和只读评估。"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Protocol, Sequence

from verge.calculator import (
    calculate_existing_experience_credit, calculate_operation_contributions,
    calculate_revision_budget,
)
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import (
    InvalidRevisionAttempt, PreferenceSample, RevisionAttempt, RevisionOperation, Rubric, RubricJudgment,
    TrajectoryRecord,
)
from verge.modules import (
    AdmissionChecker, Critic, FusionJudge, InitialRubricGenerator, Judge, JudgeOutputError,
    Reflector, Retriever, RevisionCreditAssigner, RevisionGenerator, Selector,
)
from verge.prompts import ExperienceView


class PipelineModules(Protocol):
    """Runner 只依赖模块接口；FakeModules 和其他实现可直接注入。"""

    retriever: Retriever
    selector: Selector
    initial_generator: InitialRubricGenerator
    judge: Judge
    critic: Critic
    revision_generator: RevisionGenerator
    reflector: Reflector
    revision_credit_assigner: RevisionCreditAssigner
    admission_checker: AdmissionChecker
    fusion_judge: FusionJudge


class InvalidRubricSizeError(ValueError):
    """修订输出的 rubric 标准条数不在允许范围内。"""


class RevisionExecutor:
    """将 Generator 的操作应用到 R0，并为 DELETE/MODIFY 补全原标准。"""

    def execute(
        self, initial: Rubric, operations: Sequence[RevisionOperation], attempt_number: int,
    ) -> tuple[Rubric, list[RevisionOperation]]:
        if initial.base_rubric_id is not None or attempt_number not in (1, 2):
            raise ValueError("revision executor requires R0 and attempt 1/2")
        old = {criterion.criterion_id: criterion for criterion in initial.criteria}
        revised = {key: replace(value, source_experiences=[]) for key, value in old.items()}
        touched: set[str] = set()
        operation_ids: set[str] = set()
        executed = []
        for operation in operations:
            key = (operation.new_or_updated_criterion.criterion_id
                   if operation.operation_type == "ADD" else operation.target_criterion_id)
            if (operation.operation_id in operation_ids or key in touched
                    or operation.before_criterion is not None):
                raise ValueError("conflicting or already executed revision operations")
            operation_ids.add(operation.operation_id)
            touched.add(key)
            if operation.operation_type == "ADD":
                if key in old:
                    raise ValueError("ADD requires a new criterion ID")
                revised[key] = replace(operation.new_or_updated_criterion, source_experiences=[])
                executed.append(operation)
            elif operation.operation_type == "DELETE":
                if key not in old:
                    raise ValueError("DELETE target must be in R0")
                del revised[key]
                executed.append(replace(operation, before_criterion=old[key]))
            elif operation.operation_type == "MODIFY":
                if key not in old:
                    raise ValueError("MODIFY target must be in R0")
                updated = operation.new_or_updated_criterion
                if (updated.question == old[key].question
                        and updated.importance_score == old[key].importance_score):
                    raise ValueError("no-op MODIFY does not change a criterion")
                revised[key] = replace(updated, source_experiences=[])
                executed.append(replace(operation, before_criterion=old[key]))
            else:
                raise ValueError("unknown revision operation type")
        if not 2 <= len(revised) <= 8:
            raise InvalidRubricSizeError("revised rubric must contain 2..8 criteria")
        rubric = Rubric(f"{initial.rubric_id}_revision_{attempt_number}", initial.rubric_id,
                        list(revised.values()))
        return rubric, executed


def _check_reused_judgments(
    initial: Rubric, initial_judgment: RubricJudgment,
    revised: Rubric, revised_judgment: RubricJudgment,
) -> None:
    """同一问题即使换了 ID，也不能因 Judge 重答而凭空取得修订收益。"""
    old_questions = {item.question for item in initial.criteria}
    new_questions = {item.question for item in revised.criteria}
    old_answers = {item.criterion_id: item for item in initial_judgment.criterion_judgments}
    new_answers = {item.criterion_id: item for item in revised_judgment.criterion_judgments}
    for question in old_questions & new_questions:
        before = {(old_answers[item.criterion_id].chosen, old_answers[item.criterion_id].rejected)
                  for item in initial.criteria if item.question == question}
        after = {(new_answers[item.criterion_id].chosen, new_answers[item.criterion_id].rejected)
                 for item in revised.criteria if item.question == question}
        if len(before) != 1 or before != after:
            raise ValueError("unchanged criterion question must reuse Boolean judgment")


class PipelineRunner:
    """顺序处理样本；Memory 在下游信用和 Fusion 两处立即写盘。"""

    def __init__(
        self, modules: PipelineModules, memory: ExperienceMemory, trajectories: TrajectoryLog,
        *, training: bool = True, evidence_weight: float = 1.0, max_attempts: int = 2,
        max_revision_output_retries: int = 0, executor: RevisionExecutor | None = None,
    ) -> None:
        if type(training) is not bool:
            raise ValueError("training must be Boolean")
        if type(max_attempts) is not int or max_attempts not in (1, 2):
            raise ValueError("max_attempts must be 1 or 2")
        if type(max_revision_output_retries) is not int or not 0 <= max_revision_output_retries <= 3:
            raise ValueError("max_revision_output_retries must be an integer in [0, 3]")
        if (type(evidence_weight) not in (int, float) or
                not 0 <= evidence_weight <= 1):
            raise ValueError("evidence_weight must be in [0, 1]")
        if training and memory.read_only:
            raise ValueError("training requires writable experience memory")
        self.modules = modules
        self.memory = memory
        self.trajectories = trajectories
        self.training = training
        self.evidence_weight = evidence_weight
        self.max_attempts = max_attempts
        self.max_revision_output_retries = max_revision_output_retries
        self.executor = executor if executor is not None else RevisionExecutor()
        self.current_stage = "not_started"

    def run(self, samples: Iterable[PreferenceSample]) -> list[TrajectoryRecord]:
        """按输入顺序处理，上一样本的 Memory 更新立即对下一样本可见。"""
        return [self.run_sample(sample) for sample in samples]

    def run_sample(self, sample: PreferenceSample) -> TrajectoryRecord:
        """处理一个样本：初始判断 → 必要时修订 → 候选经验入池 → 记录轨迹。"""
        if not isinstance(sample, PreferenceSample):
            raise ValueError("runner requires PreferenceSample")
        modules = self.modules
        # 召回、筛选和初始生成只看 task；preference pair 从 Judge 阶段才可见。
        self.current_stage = "retrieval"
        recalled = modules.retriever.retrieve(sample.task, self.memory.all())
        views = [ExperienceView.from_experience(item) for item in recalled]
        self.current_stage = "selection"
        selected = modules.selector.select(sample.task, views, training=self.training)
        recalled_ids = {item.experience_id for item in views}
        if any(item.experience_id not in recalled_ids for item in selected):
            raise ValueError("selected experiences must be retrieved experiences")
        self.current_stage = "initial_generator"
        initial = modules.initial_generator.generate(sample.task, selected)
        if initial.base_rubric_id is not None:
            raise ValueError("initial rubric must not have a revision base")
        self.current_stage = "initial_judge"
        try:
            initial_judgment = modules.judge.judge(initial, sample.chosen, sample.rejected)
        except JudgeOutputError as exc:
            record = TrajectoryRecord(sample.sample_id, "judge_error_initial",
                                      [item.experience_id for item in recalled],
                                      [item.experience_id for item in selected],
                                      initial, None, [], [], {}, judge_error=str(exc))
            self.current_stage = "trajectory"
            self.trajectories.append(record)
            return record
        # 已有经验的下游信用只由 R0 计算，立即写盘；后续修订失败也不回滚。
        if self.training:
            self.current_stage = "existing_credit"
            credits = calculate_existing_experience_credit(initial, initial_judgment)
            self.memory.apply_downstream_credits(sample.sample_id, credits,
                                                 evidence_weight=self.evidence_weight)
        attempts: list[RevisionAttempt] = []
        invalid_attempts: list[InvalidRevisionAttempt] = []
        candidates = []
        contributions: dict[str, float] = {}
        state = "initial_success" if initial_judgment.success else "final_failure"
        # 训练时仅对失败的 R0 修订；每次尝试都以 R0 为基线，而非接续上次结果。
        if self.training and not initial_judgment.success:
            for number in range(1, self.max_attempts + 1):
                self.current_stage = f"revision_{number}.critic"
                feedback = modules.critic.critique(
                    sample.task, sample.chosen, sample.rejected, initial, initial_judgment,
                    attempts[-1] if attempts else None,
                )
                repair_feedback = feedback
                revision_history = ({} if number == 1 else {
                    "chosen": sample.chosen, "rejected": sample.rejected,
                    "initial_judgment": initial_judgment,
                    "previous_attempt": attempts[-1] if attempts else None,
                    "previous_invalid_attempt": invalid_attempts[-1] if invalid_attempts else None,
                })
                validation_failures: list[str] = []
                for retry in range(self.max_revision_output_retries + 1):
                    try:
                        self.current_stage = f"revision_{number}.generator"
                        operations = modules.revision_generator.generate(
                            sample.task, initial, repair_feedback, number, **revision_history,
                        )
                        self.current_stage = f"revision_{number}.executor"
                        revised, executed = self.executor.execute(initial, operations, number)
                        break
                    except ValueError as exc:
                        validation_failures.append(str(exc)[:160] or type(exc).__name__)
                        if retry == self.max_revision_output_retries:
                            if self.current_stage == f"revision_{number}.executor":
                                invalid_attempts.append(InvalidRevisionAttempt(
                                    number, feedback, validation_failures,
                                ))
                                break
                            raise
                        # 只有合法 rubric 才交给 Judge；错误反馈不能暴露 chosen/rejected。
                        repair_feedback = (
                            f"{feedback}\n上一版 operations 未通过结构校验：{validation_failures[-1]}。"
                            f"R0 含 {len(initial.criteria)} 条标准；请从 R0 重新生成操作，"
                            "保证执行后仍有 2 到 8 条，ID 合法且不冲突。"
                        )
                if invalid_attempts and invalid_attempts[-1].attempt_number == number:
                    continue  # 本次修订失败；下一次仍基于 R0，不能凭空构造 Rk。
                self.current_stage = f"revision_{number}.judge"
                try:
                    revised_judgment = modules.judge.judge(
                        revised, sample.chosen, sample.rejected,
                        previous=initial, previous_judgment=initial_judgment,
                    )
                except JudgeOutputError as exc:
                    record = TrajectoryRecord(sample.sample_id, "judge_error_revision",
                                              [item.experience_id for item in recalled],
                                              [item.experience_id for item in selected],
                                              initial, initial_judgment, attempts, [], {},
                                              invalid_attempts, str(exc))
                    self.current_stage = "trajectory"
                    self.trajectories.append(record)
                    return record
                _check_reused_judgments(initial, initial_judgment, revised, revised_judgment)
                attempt = RevisionAttempt(number, feedback, executed, revised, revised_judgment,
                                          validation_failures)
                attempts.append(attempt)
                # 失败尝试仅留在轨迹；只对首次成功的操作反思、归因并生成候选。
                if revised_judgment.success:
                    state = f"revision_success_{number}"
                    self.current_stage = "reflection"
                    reflected = modules.reflector.reflect(sample.task, initial, initial_judgment,
                                                          attempt)
                    self.current_stage = "operation_contribution"
                    contributions = calculate_operation_contributions(
                        initial, initial_judgment, revised, revised_judgment, executed,
                    )
                    self.current_stage = "revision_budget"
                    budget = calculate_revision_budget(
                        initial, initial_judgment, revised, revised_judgment, self.evidence_weight,
                    )
                    self.current_stage = "revision_credit"
                    candidates = modules.revision_credit_assigner.assign(
                        reflected, executed, contributions, budget,
                    )
                    # 信用先分配，再逐条准入和融合；拒绝项的信用不转给其他候选。
                    for index, candidate in enumerate(candidates):
                        if candidate.credit is None or not candidate.linked_operation_ids:
                            continue  # CREDIT_FAIL：不得进入 Admission。
                        self.current_stage = "admission"
                        reviewed = modules.admission_checker.check(
                            sample.task, candidate, attempt, contributions,
                        )
                        if reviewed.admission == "PASS":
                            # 每条候选的召回必须观察此前 ADD/MERGE 的最新经验池。
                            self.current_stage = "fusion_retrieval"
                            pool = modules.retriever.retrieve(
                                f"{reviewed.content} {reviewed.trigger}", self.memory.all(),
                            )
                            self.current_stage = "fusion_judge"
                            match = modules.fusion_judge.find_match(
                                reviewed.content, reviewed.trigger,
                                [(item.content, item.trigger) for item in pool],
                            )
                            if match is not None and (type(match) is not int or
                                                      not 0 <= match < len(pool)):
                                raise ValueError("fusion match must be a recalled experience index")
                            self.current_stage = "fusion_write"
                            reviewed = self.memory.fuse(
                                reviewed, sample.sample_id,
                                pool[match].experience_id if match is not None else None,
                            )
                        candidates[index] = reviewed
                    break
        elif not self.training and not initial_judgment.success:
            state = "initial_failure"  # held-out evaluation 不 revision、不写入信用。
        # 汇总完整阶段结果；只有正常结束的样本才追加轨迹。
        self.current_stage = "trajectory"
        record = TrajectoryRecord(sample.sample_id, state,
                                  [item.experience_id for item in recalled],
                                  [item.experience_id for item in selected],
                                  initial, initial_judgment, attempts, candidates, contributions,
                                  invalid_attempts)
        self.trajectories.append(record)
        return record
