"""Candidate-first experience learning with two application opportunities."""

from __future__ import annotations

from typing import Iterable, Protocol

from verge.calculator import calculate_experience_credits, validate_rubric_judgment, positive_experience_sources, calculate_experience_usage
from verge.llm import LLMFormatError, LLMSampleError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import (
    ApplicationAttempt,
    Credit,
    ExperienceCase,
    PreferenceSample,
    ProposalAttempt,
    TrajectoryRecord,
)
from verge.modules import (
    Critic,
    ExperienceUpdater,
    InitialRubricGenerator,
    Judge,
    JudgeOutputError,
    Retriever,
    Selector,
    CategoryOrganizer,
    retrieval_units,
)
from verge.prompts import ExperienceView


class PipelineModules(Protocol):
    retriever: Retriever
    selector: Selector
    initial_generator: InitialRubricGenerator
    judge: Judge
    critic: Critic
    experience_updater: ExperienceUpdater
    category_organizer: CategoryOrganizer


class PipelineRunner:
    def __init__(
        self,
        modules: PipelineModules,
        memory: ExperienceMemory,
        trajectories: TrajectoryLog,
        *,
        training: bool = True,
        max_rounds: int = 2,
        organization_batch_size: int = 5,
    ) -> None:
        if type(training) is not bool:
            raise ValueError("training must be Boolean")
        if type(max_rounds) is not int or max_rounds not in (1, 2):
            raise ValueError("max_rounds must be 1 or 2")
        if training and memory.read_only:
            raise ValueError("training requires writable memory")
        if type(organization_batch_size) is not int or organization_batch_size < 1:
            raise ValueError("organization_batch_size must be positive")
        self.modules, self.memory, self.trajectories = modules, memory, trajectories
        self.training, self.max_rounds = training, max_rounds
        self.organization_batch_size = organization_batch_size
        self.current_stage = "not_started"

    def run(self, samples: Iterable[PreferenceSample]) -> list[TrajectoryRecord]:
        records = [self.run_sample(s) for s in samples]
        if records:
            self.flush_organization(records[-1].sample_id)
        return records

    def run_sample(self, sample: PreferenceSample) -> TrajectoryRecord:
        if not isinstance(sample, PreferenceSample):
            raise ValueError("runner requires PreferenceSample")
        record = TrajectoryRecord(sample.sample_id)
        normal_end = False
        try:
            self._run_sample(sample, record)
            normal_end = True
        except JudgeOutputError as exc:
            record.state = (
                "judge_error_initial"
                if self.current_stage == "initial_judge"
                else "judge_error_application"
            )
            record.judge_error = str(exc)
            if record.application_attempts and self.current_stage.endswith(".judge"):
                record.application_attempts[-1].error = str(exc)
        except (ValueError, LLMSampleError) as exc:
            role_stages = {
                "selection",
                "initial_generator",
                "initial_judge",
            }
            if (
                self.current_stage not in role_stages
                and not self.current_stage.endswith((".critic", ".generator", ".judge"))
                and not (self.current_stage == "retrieval" and isinstance(exc, (LLMSampleError, LLMFormatError)))
            ):
                raise
            record.state = "sample_error"
            record.sample_error = {
                "stage": self.current_stage,
                "type": type(exc).__name__,
                "message": str(exc)[:500] or type(exc).__name__,
            }
            if record.application_attempts and self.current_stage.endswith((".generator", ".judge")):
                record.application_attempts[-1].error = record.sample_error["message"]
            if record.proposal_attempts and self.current_stage.endswith(".critic"):
                record.proposal_attempts[-1].error = record.sample_error["message"]
        if self.training and record.initial_judgment is not None:
            self._finish_sample(sample, record, admit=normal_end)
        if self.training:
            record.category_organization = self._organize(sample.sample_id)
        self.current_stage = "trajectory"
        self.trajectories.append(record)
        return record

    def _run_sample(self, sample, record):
        m = self.modules
        self.current_stage = "retrieval"
        categories = self.memory.categories()
        recalled = m.retriever.retrieve(sample.task, self.memory.all(), categories=categories)
        record.retrieved_experience_ids = [x.experience_id for x in recalled]
        record.retrieved_units = [{"unit_id": u.unit_id, "kind": "category" if u.category else "experience",
                                   "experience_ids": [e.experience_id for e in u.members]}
                                  for u in retrieval_units(recalled, categories)]
        by_category = {c.category_id: c for c in categories}
        self.current_stage = "selection"
        selected = m.selector.select(
            sample.task,
            [ExperienceView.from_experience(x, by_category.get(x.category_id)) for x in recalled],
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
            usage = calculate_experience_usage(initial, judgment, sample.sample_id,
                                               {v.experience_id: self.memory.get(v.experience_id).text_version for v in selected})
            self.memory.apply_downstream_credits(sample.sample_id, credits, usage=usage)
        record.state = "initial_success" if judgment.success else "initial_failure"
        if judgment.success or not self.training:
            return

        current, current_judgment = initial, judgment
        previous_candidate = None
        adopted = None
        invalid_error = None
        previous_rubric = previous_judgment = None
        candidate_ids = set()
        critic_conversation: list[dict[str, str]] = []  # sample-local; never reaches Generator
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
                    conversation=critic_conversation,
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
            if candidate.experience_id is not None:
                raise ValueError("critic must propose new candidates")
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
            application.eligible = bool(used) and candidate.candidate_id in positive_experience_sources(rubric, result)
            previous_rubric, previous_judgment = current, current_judgment
            current, current_judgment = rubric, result
            previous_candidate, adopted, invalid_error = candidate, bool(used), None
            if not result.success:
                continue
            record.state = (
                "application_success" if used else "success_without_candidate"
            )
            return
        if (
            not record.application_attempts
            or record.application_attempts[-1].judgment is None
        ):
            record.state = "invalid_application_exhausted"

    def _finish_sample(self, sample, record, *, admit):
        # Errors do not perform normal candidate admission; valid R0 evidence remains usable.
        if admit:
            eligible = [a for a in record.application_attempts if a.eligible]
            if eligible:
                app = eligible[-1]  # latest valid positive candidate, never compare margins across rubrics
                proposal = next(p for p in record.proposal_attempts if p.candidate and p.candidate.candidate_id == app.candidate_id)
                candidate = proposal.candidate
                case = ExperienceCase(
                    sample.sample_id, sample.task, sample.chosen, sample.rejected,
                    candidate.candidate_id, 1, candidate.content, candidate.trigger,
                    app.rubric, app.judgment, app.credit, proposal.analysis,
                )
                self.current_stage = "candidate_admission"
                proposal.candidate = self.memory.add(candidate, case)
                record.admitted_candidate_id = candidate.candidate_id
                record.admitted_experience_id = proposal.candidate.experience_id
        for view in record.selected_experiences:
            eid = view["experience_id"]
            if eid not in record.downstream_credits or not record.downstream_credits[eid].amount:
                continue
            experience = self.memory.get(eid)
            case = ExperienceCase(
                sample.sample_id, sample.task, sample.chosen, sample.rejected,
                eid, experience.text_version, view["content"], view["trigger"],
                record.initial_rubric, record.initial_judgment, record.downstream_credits[eid],
            )
            self.current_stage = "experience_case"
            if not self.memory.collect_case(eid, case):
                continue
            experience = self.memory.get(eid)
            if not experience.update_due:
                continue
            self.current_stage = "experience_updater"
            error = None
            try:
                update = self.modules.experience_updater.update(experience)
            except (ValueError, LLMSampleError) as exc:
                error = str(exc)[:500] or type(exc).__name__
            self.current_stage = "experience_text_write"
            record.experience_updates[eid] = (
                self.memory.update_text(eid, experience.content, experience.trigger, error=error)
                if error is not None else self.memory.apply_update(eid, update)
            )

    def _organize(self, sample_id: str, *, flush: bool = False):
        if not self.training or not self.memory.organization_due(self.organization_batch_size, flush=flush):
            return None
        self.current_stage = "category_organizer"
        try:
            plan = self.modules.category_organizer.organize(
                self.memory.all(), self.memory.categories(), dict(self.memory.pending_organization))
        except (ValueError, LLMSampleError) as exc:
            return self.memory.organize(sample_id, None, error=str(exc)[:500] or type(exc).__name__)
        self.current_stage = "category_organization_write"
        return self.memory.organize(sample_id, plan)

    def flush_organization(self, sample_id: str):
        """Training stop/finalization; event persists in memory history under the last task."""
        return self._organize(sample_id, flush=True)
