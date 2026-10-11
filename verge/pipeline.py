"""Candidate-first experience learning with two application opportunities."""

from __future__ import annotations

from typing import Iterable, Protocol

from verge.calculator import calculate_experience_credits, validate_rubric_judgment
from verge.llm import LLMRefusalError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import (
    ApplicationAttempt,
    Credit,
    FusionResult,
    PreferenceSample,
    ProposalAttempt,
    TrajectoryRecord,
)
from verge.modules import (
    Critic,
    FusionJudge,
    InitialRubricGenerator,
    Judge,
    JudgeOutputError,
    Retriever,
    Selector,
)
from verge.prompts import ExperienceView


class PipelineModules(Protocol):
    retriever: Retriever
    selector: Selector
    initial_generator: InitialRubricGenerator
    judge: Judge
    critic: Critic
    fusion_judge: FusionJudge


class PipelineRunner:
    def __init__(
        self,
        modules: PipelineModules,
        memory: ExperienceMemory,
        trajectories: TrajectoryLog,
        *,
        training: bool = True,
        max_rounds: int = 2,
    ) -> None:
        if type(training) is not bool:
            raise ValueError("training must be Boolean")
        if type(max_rounds) is not int or max_rounds not in (1, 2):
            raise ValueError("max_rounds must be 1 or 2")
        if training and memory.read_only:
            raise ValueError("training requires writable memory")
        self.modules, self.memory, self.trajectories = modules, memory, trajectories
        self.training, self.max_rounds = training, max_rounds
        self.current_stage = "not_started"

    def run(self, samples: Iterable[PreferenceSample]) -> list[TrajectoryRecord]:
        return [self.run_sample(s) for s in samples]

    def run_sample(self, sample: PreferenceSample) -> TrajectoryRecord:
        if not isinstance(sample, PreferenceSample):
            raise ValueError("runner requires PreferenceSample")
        record = TrajectoryRecord(sample.sample_id)
        try:
            self._run_sample(sample, record)
        except JudgeOutputError as exc:
            record.state = (
                "judge_error_initial"
                if self.current_stage == "initial_judge"
                else "judge_error_application"
            )
            record.judge_error = str(exc)
            if record.application_attempts and self.current_stage.endswith(".judge"):
                record.application_attempts[-1].error = str(exc)
        except (ValueError, LLMRefusalError) as exc:
            role_stages = {
                "selection",
                "initial_generator",
                "initial_judge",
                "fusion_judge",
            }
            if (
                self.current_stage not in role_stages
                and not self.current_stage.endswith((".critic", ".generator", ".judge"))
            ):
                raise
            record.state = "sample_error"
            record.sample_error = {
                "stage": self.current_stage,
                "type": type(exc).__name__,
                "message": str(exc)[:500] or type(exc).__name__,
            }
        self.current_stage = "trajectory"
        self.trajectories.append(record)
        return record

    def _run_sample(self, sample, record):
        m = self.modules
        self.current_stage = "retrieval"
        recalled = m.retriever.retrieve(sample.task, self.memory.all())
        record.retrieved_experience_ids = [x.experience_id for x in recalled]
        self.current_stage = "selection"
        selected = m.selector.select(
            sample.task,
            [ExperienceView.from_experience(x) for x in recalled],
            training=self.training,
        )
        if len({x.experience_id for x in selected}) != len(selected) or set(
            x.experience_id for x in selected
        ) - set(record.retrieved_experience_ids):
            raise ValueError("selected IDs must be unique retrieved experiences")
        record.selected_experiences = [x.to_dict() for x in selected]
        self.current_stage = "initial_generator"
        initial = m.initial_generator.generate(sample.task, selected)
        allowed_e = {x.experience_id for x in selected}
        if any(set(c.source_experiences) - allowed_e for c in initial.criteria):
            raise ValueError("R0 sources must be selected experiences")
        record.initial_rubric = initial
        self.current_stage = "initial_judge"
        judgment = m.judge.judge(
            initial, sample.chosen, sample.rejected, task=sample.task
        )
        record.initial_judgment = judgment
        self.current_stage = "initial_calculator"
        validate_rubric_judgment(initial, judgment)
        if self.training:
            self.current_stage = "existing_credit"
            credits = calculate_experience_credits(initial, judgment)
            record.downstream_credits = credits
            self.memory.apply_downstream_credits(sample.sample_id, credits)
        record.state = "initial_success" if judgment.success else "initial_failure"
        if judgment.success or not self.training:
            return

        current, current_judgment = initial, judgment
        previous_candidate = None
        adopted = None
        invalid_error = None
        previous_rubric = previous_judgment = None
        candidate_ids = set()
        record.state = "validation_exhausted"
        for number in range(1, self.max_rounds + 1):
            attempt = ProposalAttempt(number)
            record.proposal_attempts.append(attempt)
            self.current_stage = f"application_{number}.critic"
            try:
                proposal = m.critic.propose(
                    sample.task,
                    sample.chosen,
                    sample.rejected,
                    current,
                    current_judgment,
                    number,
                    previous_candidate=previous_candidate,
                    adopted=adopted,
                    invalid_error=invalid_error,
                    previous_rubric=previous_rubric,
                    previous_judgment=previous_judgment,
                )
            except ValueError as exc:
                attempt.error = invalid_error = str(exc)
                continue  # invalid round; no invented Judge result or extra proposal.
            attempt.analysis, attempt.candidate = proposal.analysis, proposal.candidate
            candidate = proposal.candidate
            if candidate is None:
                record.state = "no_candidate"
                return
            if (
                candidate.candidate_id in allowed_e
                or candidate.candidate_id in candidate_ids
            ):
                raise ValueError("critic candidate IDs must be new and unique")
            candidate_ids.add(candidate.candidate_id)
            if candidate.fusion is not None:
                raise ValueError("critic must propose unfused candidates")
            application = ApplicationAttempt(number, candidate.candidate_id)
            record.application_attempts.append(application)
            self.current_stage = f"application_{number}.generator"
            try:
                rubric = m.initial_generator.apply(
                    sample.task, selected, candidate, number
                )
                if any(
                    set(c.source_experiences) - (allowed_e | {candidate.candidate_id})
                    for c in rubric.criteria
                ):
                    raise ValueError(
                        "application sources must refer to this call's E and C"
                    )
            except ValueError as exc:
                application.error = invalid_error = str(exc)
                previous_candidate, adopted = candidate, None
                continue
            application.rubric = rubric
            used = [
                c.criterion_id
                for c in rubric.criteria
                if candidate.candidate_id in c.source_experiences
            ]
            application.adopted = bool(used)
            application.adopted_criterion_ids = used
            self.current_stage = f"application_{number}.judge"
            result = m.judge.judge(
                rubric,
                sample.chosen,
                sample.rejected,
                task=sample.task,
                previous=current,
                previous_judgment=current_judgment,
            )
            application.judgment = result
            self.current_stage = "application_credit"
            application.credit = calculate_experience_credits(rubric, result).get(
                candidate.candidate_id, Credit()
            )
            previous_rubric, previous_judgment = current, current_judgment
            current, current_judgment = rubric, result
            previous_candidate, adopted, invalid_error = candidate, bool(used), None
            if not result.success:
                continue
            record.state = (
                "application_success" if used else "success_without_candidate"
            )
            if not used:
                return
            self.current_stage = "fusion_retrieval"
            pool = m.retriever.retrieve(
                f"{candidate.content} {candidate.trigger}", self.memory.all()
            )
            self.current_stage = "fusion_judge"
            match = m.fusion_judge.find_match(
                candidate.content,
                candidate.trigger,
                [(x.content, x.trigger) for x in pool],
            )
            if not isinstance(match, FusionResult) or (
                match.match_index is not None and match.match_index >= len(pool)
            ):
                raise ValueError("invalid fusion match")
            self.current_stage = "fusion_write"
            fused = self.memory.fuse(
                candidate,
                sample.sample_id,
                (
                    pool[match.match_index].experience_id
                    if match.match_index is not None
                    else None
                ),
                merged_content=match.content,
                merged_trigger=match.trigger,
                application_credit=application.credit,
            )
            attempt.candidate = fused
            return
        if (
            not record.application_attempts
            or record.application_attempts[-1].judgment is None
        ):
            record.state = "invalid_application_exhausted"
