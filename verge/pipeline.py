"""单样本顺序 Runner：用模块契约串联训练和只读评估。"""

from __future__ import annotations

from typing import Iterable, Protocol

from verge.calculator import (
    calculate_existing_experience_credit, calculate_fusion_credit,
)
from verge.llm import LLMFormatError, LLMRefusalError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import (
    InvalidRevisionAttempt, PreferenceSample, RevisionAttempt, Rubric, RubricJudgment,
    TrajectoryRecord, FusionResult,
)
from verge.modules import (
    AdmissionChecker, Critic, FusionJudge, InitialRubricGenerator, Judge, JudgeOutputError,
    Reflector, Retriever, Selector,
)
from verge.prompts import ExperienceView


class PipelineModules(Protocol):
    """Runner 只依赖模块接口；FakeModules 和其他实现可直接注入。"""

    retriever: Retriever
    selector: Selector
    initial_generator: InitialRubricGenerator
    judge: Judge
    critic: Critic
    reflector: Reflector
    admission_checker: AdmissionChecker
    fusion_judge: FusionJudge


def _check_reused_judgments(
    initial: Rubric, initial_judgment: RubricJudgment,
    revised: Rubric, revised_judgment: RubricJudgment,
) -> None:
    """问题沿用时保持 R0 判断；R0 本身有相互矛盾的同名判断时不猜测答案。"""
    old_answers = {item.criterion_id: item for item in initial_judgment.criterion_judgments}
    new_answers = {item.criterion_id: item for item in revised_judgment.criterion_judgments}
    old_by_id = {item.criterion_id: item for item in initial.criteria}
    old_by_question: dict[str, set[tuple[bool, bool]]] = {}
    for item in initial.criteria:
        answer = old_answers[item.criterion_id]
        old_by_question.setdefault(item.question, set()).add((answer.chosen, answer.rejected))
    for item in revised.criteria:
        answer = new_answers[item.criterion_id]
        actual = (answer.chosen, answer.rejected)
        if (item.criterion_id in old_by_id
                and item.question == old_by_id[item.criterion_id].question):
            previous = old_answers[item.criterion_id]
            expected = (previous.chosen, previous.rejected)
        elif len(old_by_question.get(item.question, ())) == 1:
            expected = next(iter(old_by_question[item.question]))
        else:
            continue
        if actual != expected:
            raise ValueError("unchanged criterion question must reuse Boolean judgment")


class PipelineRunner:
    """顺序处理样本；Memory 在下游信用和 Fusion 两处立即写盘。"""

    def __init__(
        self, modules: PipelineModules, memory: ExperienceMemory, trajectories: TrajectoryLog,
        *, training: bool = True, evidence_weight: float = 1.0, max_attempts: int = 2,
        max_revision_output_retries: int = 2,
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
        self.current_stage = "not_started"

    def run(self, samples: Iterable[PreferenceSample]) -> list[TrajectoryRecord]:
        """按输入顺序处理，上一样本的 Memory 更新立即对下一样本可见。"""
        return [self.run_sample(sample) for sample in samples]

    def run_sample(self, sample: PreferenceSample) -> TrajectoryRecord:
        """处理一个样本：初始判断 → 必要时修订 → 候选经验入池 → 记录轨迹。"""
        if not isinstance(sample, PreferenceSample):
            raise ValueError("runner requires PreferenceSample")
        record = TrajectoryRecord(sample.sample_id, "in_progress", [], [], None, None, [], [], {})
        try:
            return self._run_sample(sample, record)
        except (ValueError, LLMRefusalError) as exc:
            # 仅隔离角色输出/样本契约错误；计算、存储、程序错误及全局 API 故障仍上抛。
            role_stages = {"selection", "initial_generator", "initial_judge", "reflection",
                           "admission", "fusion_judge"}
            if (self.current_stage not in role_stages and
                    not self.current_stage.endswith((".critic", ".generator", ".judge"))):
                raise
            record.state = "sample_error"
            record.sample_error = {"stage": self.current_stage, "type": type(exc).__name__,
                                   "message": str(exc)[:500] or type(exc).__name__}
            self.current_stage = "trajectory"
            self.trajectories.append(record)
            return record

    def _run_sample(self, sample: PreferenceSample, record: TrajectoryRecord) -> TrajectoryRecord:
        modules = self.modules
        # 召回、筛选和初始生成只看 task；preference pair 从 Judge 阶段才可见。
        self.current_stage = "retrieval"
        recalled = modules.retriever.retrieve(sample.task, self.memory.all())
        record.retrieved_experience_ids = [item.experience_id for item in recalled]
        views = [ExperienceView.from_experience(item) for item in recalled]
        self.current_stage = "selection"
        selected = modules.selector.select(sample.task, views, training=self.training)
        recalled_ids = {item.experience_id for item in views}
        if any(item.experience_id not in recalled_ids for item in selected):
            raise ValueError("selected experiences must be retrieved experiences")
        record.selected_experience_ids = [item.experience_id for item in selected]
        self.current_stage = "initial_generator"
        initial = modules.initial_generator.generate(sample.task, selected)
        if initial.base_rubric_id is not None:
            raise ValueError("initial rubric must not have a revision base")
        record.initial_rubric = initial
        self.current_stage = "initial_judge"
        try:
            initial_judgment = modules.judge.judge(initial, sample.chosen, sample.rejected, task=sample.task)
        except JudgeOutputError as exc:
            record = TrajectoryRecord(sample.sample_id, "judge_error_initial",
                                      [item.experience_id for item in recalled],
                                      [item.experience_id for item in selected],
                                      initial, None, [], [], {}, judge_error=str(exc))
            self.current_stage = "trajectory"
            self.trajectories.append(record)
            return record
        record.initial_judgment = initial_judgment
        # 已有经验的下游信用只由 R0 计算，立即写盘；后续修订失败也不回滚。
        if self.training:
            self.current_stage = "existing_credit"
            credits = calculate_existing_experience_credit(initial, initial_judgment)
            self.memory.apply_downstream_credits(sample.sample_id, credits,
                                                 evidence_weight=self.evidence_weight)
        attempts = record.revision_attempts
        invalid_attempts = record.invalid_revision_attempts
        candidates = []
        contributions: dict[str, float] = {}
        state = "initial_success" if initial_judgment.success else "final_failure"
        # Generator 会话限于单个样本；第二轮接续最近的有效 rubric。
        current, current_judgment = initial, initial_judgment
        if self.training and not initial_judgment.success:
            for number in range(1, self.max_attempts + 1):
                revision_history = ({} if number == 1 else {
                    "previous_attempt": attempts[-1] if attempts else None,
                    "previous_invalid_attempt": invalid_attempts[-1] if invalid_attempts else None,
                })
                self.current_stage = f"revision_{number}.critic"
                try:
                    proposal = modules.critic.critique(
                        sample.task, sample.chosen, sample.rejected, current, current_judgment,
                        number, **revision_history,
                    )
                except LLMFormatError as exc:
                    invalid_attempts.append(InvalidRevisionAttempt(number, None, [str(exc)], analysis=None))
                    continue
                validation_failures = []
                revised = None
                for retry in range(self.max_revision_output_retries + 1):
                    self.current_stage = f"revision_{number}.generator"
                    try:
                        revised = modules.initial_generator.revise(
                            current, proposal.improvement_advice, number,
                            validation_feedback=validation_failures[-1] if validation_failures else None,
                        )
                        if (revised.base_rubric_id != current.rubric_id
                                or not 2 <= len(revised.criteria) <= 8
                                or any(c.source_experiences or not c.question.strip() for c in revised.criteria)):
                            revised = None
                            raise ValueError("invalid full revised rubric or revision base")
                        break
                    except LLMFormatError as exc:
                        validation_failures.append(str(exc))
                        break  # 后端已耗尽格式纠错，不叠加语义重试。
                    except ValueError as exc:
                        revised = None
                        validation_failures.append(str(exc)[:160])
                if revised is None:
                    invalid_attempts.append(InvalidRevisionAttempt(
                        number, proposal.improvement_advice, validation_failures, analysis=proposal.analysis))
                    continue
                self.current_stage = f"revision_{number}.judge"
                try:
                    revised_judgment = modules.judge.judge(
                        revised, sample.chosen, sample.rejected, task=sample.task,
                        previous=current, previous_judgment=current_judgment,
                    )
                except JudgeOutputError as exc:
                    record.state = "judge_error_revision"
                    record.judge_error = str(exc)
                    self.current_stage = "trajectory"
                    self.trajectories.append(record)
                    return record
                _check_reused_judgments(current, current_judgment, revised, revised_judgment)
                attempt = RevisionAttempt(number, proposal.improvement_advice, [], revised, revised_judgment,
                                          validation_failures, analysis=proposal.analysis,
                                          previous_rubric=current, previous_judgment=current_judgment)
                attempts.append(attempt)
                current, current_judgment = revised, revised_judgment
                if revised_judgment.success:
                    state = f"revision_success_{number}"
                    self.current_stage = "reflection"
                    reflected = modules.reflector.reflect(sample.task, initial, initial_judgment,
                                                          attempt, chosen=sample.chosen,
                                                          rejected=sample.rejected)
                    record.reflected_candidates = reflected
                    candidates = reflected
                    record.reflected_candidates = candidates
                    for index, candidate in enumerate(candidates):
                        self.current_stage = "admission"
                        reviewed = modules.admission_checker.check(
                            sample.task, candidate, attempt,
                        )
                        candidates[index] = reviewed  # 准入已完成；后续 Fusion 错误仍保留此结果。
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
                            if not isinstance(match, FusionResult) or (match.match_index is not None
                                    and not 0 <= match.match_index < len(pool)):
                                raise ValueError("fusion match must be a valid recalled result")
                            self.current_stage = "fusion_credit"
                            fusion_credit = calculate_fusion_credit(
                                attempt.previous_rubric,
                                attempt.previous_judgment, revised, revised_judgment)
                            self.current_stage = "fusion_write"
                            reviewed = self.memory.fuse(
                                reviewed, sample.sample_id,
                                pool[match.match_index].experience_id if match.match_index is not None else None,
                                merged_content=match.content, merged_trigger=match.trigger,
                                fusion_credit=fusion_credit,
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
