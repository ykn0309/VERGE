"""V2 learning: candidate-only retries, per-version evidence, isolated paired revision replay."""

from __future__ import annotations
from copy import deepcopy
from uuid import uuid4
from verge.models import (
    Experience,
    TrajectoryRecord,
    ProposalAttempt,
    RevisionRecord,
    ReplayResult,
)
from verge.calculator import (
    application,
    update_quality,
    aggregate_reports,
    replay_score,
    validate_rubric_judgment,
)
from verge.llm import LLMFormatError, LLMSampleError, APPLICATION_CONTEXT
from verge.prompts import NumberedIDCodec, experience_payload

ISOLATED = (LLMFormatError, LLMSampleError)


def revision_reasons(e, *, interval=3, window=6, errors=3, q_threshold=0.5):
    if e.status != "active" or len(e.downstream) - e.last_checked_count < interval:
        return None
    reasons = []
    if (
        e.q_posterior is not None
        and e.q_posterior < q_threshold
        and any(
            a.update_reason == "updated" for a in e.downstream[e.last_checked_count :]
        )
    ):
        reasons.append("quality_risk")
    recent = e.downstream[-window:]
    for stage in ("selector", "generator", "judge"):
        if (
            sum(
                a.sample.task_id not in e.consumed_error_tasks.get(stage, [])
                and any(r.stage == stage and r.status == "error" for r in a.reports)
                for a in recent
            )
            >= errors
        ):
            reasons.append(stage + "_errors")
    return reasons


def select_revision_cases(e, versions, *, limit=6, reasons=None, window=6):
    lineage = []
    current = e
    seen = set()
    while current:
        if current.experience_id in seen:
            raise ValueError("cyclic version lineage")
        seen.add(current.experience_id)
        lineage.append(current)
        current = versions.get(current.parent_id)
    all_cases = [a for ancestor in reversed(lineage) for a in ancestor.applications]
    # Most recent actual version record wins when a task has several historical uses.
    distinct = {a.sample.task_id: a for a in all_cases}
    cases = list(distinct.values())
    chosen = []
    kinds = {}
    chosen_tasks = set()

    def add(items, count, kind):
        for a in items:
            sid = a.sample.sample_id
            if a.sample.task_id not in chosen_tasks and count and len(chosen) < limit:
                chosen_tasks.add(a.sample.task_id)
                chosen.append(a)
                kinds[sid] = kind
                count -= 1

    problems = [
        a
        for a in reversed(e.downstream[-window:])
        if any(
            r.status == "error"
            and a.sample.task_id not in e.consumed_error_tasks.get(r.stage, [])
            and (reasons is None or r.stage + "_errors" in reasons)
            for r in a.reports
        )
        or (
            (reasons is None or "quality_risk" in reasons)
            and a.update_reason == "updated"
            and a.q_before is not None
            and a.q_after is not None
            and a.q_after < a.q_before
        )
    ]
    # Cover each triggered stage before taking the newest remaining problem records.
    stage_examples = []
    for stage in ("selector", "generator", "judge"):
        a = next(
            (
                a
                for a in problems
                if any(r.stage == stage and r.status == "error" for r in a.reports)
            ),
            None,
        )
        if a:
            stage_examples.append(a)
    add([*stage_examples, *problems], 3, "problem")
    add([a for a in cases if a.sample.sample_id == e.source_sample_id], 1, "source")
    add([a for a in reversed(cases) if a.feedback == "positive"], 1, "positive")
    add(
        [
            a
            for a in reversed(cases)
            if any(
                r.stage == "selector" and r.status in ("error", "unknown")
                for r in a.reports
            )
        ],
        1,
        "applicability_dispute",
    )
    add(reversed(cases), limit - len(chosen), "recent")
    return deepcopy(chosen), kinds


class PipelineRunner:
    def __init__(
        self,
        modules,
        memory,
        trajectories,
        *,
        training=True,
        max_rounds=2,
        interval=3,
        window=6,
        error_threshold=3,
        q_threshold=0.5,
        min_replay=3,
        revision_enabled=True,
    ):
        if training and memory.read_only:
            raise ValueError("training requires writable memory")
        if (
            max_rounds not in (1, 2)
            or interval < 1
            or window < 1
            or error_threshold < 1
            or min_replay < 1
        ):
            raise ValueError("invalid pipeline limits")
        self.modules = modules
        self.memory = memory
        self.trajectories = trajectories
        self.training = training
        self.max_rounds = max_rounds
        self.interval = interval
        self.window = window
        self.error_threshold = error_threshold
        self.q_threshold = q_threshold
        self.min_replay = min_replay
        self.revision_enabled = revision_enabled
        self.stage = "idle"

    def run(self, samples):
        return [self.run_sample(s) for s in samples]

    def evaluate(
        self, sample, pool, *, round_number=0, candidates=None, retriever=None
    ):
        m = self.modules
        if candidates is None:
            recalled = (retriever or m.retriever).retrieve(sample.task, pool)
            selected = m.selector.select(sample.task, recalled, training=self.training)
            if len({e.experience_id for e in selected}) != len(selected) or {
                e.experience_id for e in selected
            } - {e.experience_id for e in recalled}:
                raise ValueError("selector violated supplied candidate whitelist")
        else:
            recalled = selected = candidates
        rubric = m.initial_generator.generate(
            sample.task, selected, round_number=round_number
        )
        if any(
            set(c.source_experiences) - {e.experience_id for e in selected}
            for c in rubric.criteria
        ):
            raise ValueError("generator violated experience whitelist")
        judgment = m.judge.judge(
            rubric, sample.chosen, sample.rejected, task=sample.task
        )
        validate_rubric_judgment(rubric, judgment)
        return recalled, selected, rubric, judgment

    def run_sample(self, sample):
        rec = TrajectoryRecord(
            sample.sample_id, sample=sample, split="train" if self.training else "test"
        )
        try:
            self.stage = "initial"
            recalled, selected, rubric, judgment = self.evaluate(
                sample, self.memory.all(active_only=True)
            )
            rec.retrieved_experience_ids = [e.experience_id for e in recalled]
            rec.selected_experiences = [
                experience_payload(e, quality=True) for e in selected
            ]
            rec.retrieval_trace = deepcopy(self.modules.retriever.last_trace)
            rec.initial_rubric = rubric
            rec.initial_judgment = judgment
            if self.training:
                self.stage = "downstream"
                for e in selected:
                    a = application(sample, e, rubric, judgment)
                    a.recalled = a.selected = True
                    a.retrieval_rank = next(
                        (
                            i
                            for i, x in enumerate(recalled, 1)
                            if x.experience_id == e.experience_id
                        ),
                        None,
                    )
                    a.retrieval_trace = deepcopy(rec.retrieval_trace)
                    if not a.cited:
                        a.no_evidence_reason = "selected_not_cited"
                    if a.cited:
                        a.reports = self.modules.checkers.check(a)
                    a.observation = aggregate_reports(a.reports)
                    self.memory.record(a)
                    rec.applications.append(a)
            rec.state = "initial_success" if judgment.success else "initial_failure"
            if self.training and not judgment.success:
                self._correct(sample, rec, rubric, judgment)
        except ISOLATED as exc:
            rec.state = "sample_error"
            rec.sample_error = {
                "stage": self.stage,
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        if self.training and self.revision_enabled:
            for e in self.memory.all(active_only=True):
                reasons = revision_reasons(
                    e,
                    interval=self.interval,
                    window=self.window,
                    errors=self.error_threshold,
                    q_threshold=self.q_threshold,
                )
                if reasons is None:
                    continue
                self.memory.mark_checked(e.experience_id, reasons, self.window)
                if reasons:
                    rec.experience_updates.append(
                        self._revise(e, sample.sample_id, reasons)
                    )
        self.trajectories.append(rec)
        return rec

    def _correct(self, sample, rec, rubric, judgment):
        conversation = []
        codec = NumberedIDCodec()
        feedback = None
        for number in range(1, self.max_rounds + 1):
            attempt = ProposalAttempt(number)
            rec.proposal_attempts.append(attempt)
            self.stage = "critic"
            try:
                proposal = self.modules.critic.propose(
                    sample,
                    rubric,
                    judgment,
                    number,
                    conversation=conversation,
                    codec=codec,
                    feedback=feedback,
                )
                attempt.proposal = proposal
                if not proposal.candidates:
                    rec.state = "no_candidate"
                    return
                candidate = proposal.candidates[0]
                self.stage = "experience_checker"
                review = self.modules.experience_checker.check(
                    sample, rubric, judgment, candidate
                )
                attempt.review = review
            except ISOLATED as exc:
                attempt.error = str(exc)
                rec.state = "candidate_review_failed"
                return
            if not review.approved:
                feedback = {"experience_review": review.to_dict()}
                rec.state = "candidate_rejected"
                continue
            # Admission is independent of the subsequent Generator/Judge outcome.
            self.memory.add(candidate, sample, review)
            rec.admitted_experience_ids.append(candidate.experience_id)
            self.stage = "source_validation"
            try:
                _, _, new, j = self.evaluate(
                    sample, [], round_number=number, candidates=[candidate]
                )
                attempt.rubric, attempt.judgment = new, j
                a = application(sample, candidate, new, j, "source_validation")
                a.round_number = number
                a.recalled = a.selected = True
                if not a.cited:
                    a.no_evidence_reason = "selected_not_cited"
                else:
                    a.reports = self.modules.checkers.check(a)
                a.observation = aggregate_reports(a.reports)
                self.memory.record(a)
                attempt.applications.append(a)
                rec.state = (
                    "application_success" if j.success else "validation_exhausted"
                )
            except ISOLATED as exc:
                attempt.error = str(exc)
                rec.state = "source_validation_failed"
            return

    def _replay(self, sample, version, pool):
        token = APPLICATION_CONTEXT.set(
            {
                "sample_id": sample.sample_id,
                "phase": "replay",
                "actual_version_id": version.experience_id,
            }
        )
        try:
            return self._replay_arm(sample, version, pool)
        finally:
            APPLICATION_CONTEXT.reset(token)

    def _replay_arm(self, sample, version, pool):
        # Snapshot and retriever fork ensure rejected candidates never enter the live pool/index.
        arm = [
            deepcopy(e)
            for e in pool
            if e.experience_id != version.parent_id
            and e.experience_id != version.experience_id
        ]
        target = deepcopy(version)
        target.status = "active"
        arm.append(target)
        retriever = self.modules.retriever.fork()
        recalled, selected, r, j = self.evaluate(sample, arm, retriever=retriever)
        a = application(sample, target, r, j, "replay")
        a.recalled = target.experience_id in {e.experience_id for e in recalled}
        a.selected = target.experience_id in {e.experience_id for e in selected}
        a.retrieval_rank = next(
            (
                i
                for i, e in enumerate(recalled, 1)
                if e.experience_id == target.experience_id
            ),
            None,
        )
        a.retrieval_trace = deepcopy(retriever.last_trace)
        if not a.cited:
            a.no_evidence_reason = (
                "not_recalled"
                if not a.recalled
                else "not_selected" if not a.selected else "selected_not_cited"
            )
        if a.selected and a.cited:
            a.reports = self.modules.checkers.check(a)
        a.observation = aggregate_reports(a.reports)
        return a

    def _revise(self, e, sample_id, reasons):
        versions = {v.experience_id: v for v in self.memory.all()}
        cases, kinds = select_revision_cases(
            e, versions, reasons=reasons, window=self.window
        )
        event = RevisionRecord(
            "revision_" + uuid4().hex,
            e.experience_id,
            sample_id,
            reasons,
            [c.sample.sample_id for c in cases],
            kinds,
            case_refs=[
                {
                    "sample_id": c.sample.sample_id,
                    "task_id": c.sample.task_id,
                    "version_id": c.version_id,
                    "phase": c.phase,
                    "round_number": c.round_number,
                    "kind": kinds[c.sample.sample_id],
                }
                for c in cases
            ],
        )
        try:
            update = self.modules.experience_updater.update(
                e, cases, reasons, case_kinds=kinds
            )
            event.update = update
            event.action = update.action
            if update.action == "keep" or update.method == e.method:
                event.action = "keep"
                event.decision = "keep"
                event = self.memory.finish_revision(event)
                return event
            child = Experience(
                "exp_" + uuid4().hex,
                update.method,
                e.source_sample_id,
                source_task_id=e.source_task_id,
                parent_id=e.experience_id,
                status="candidate",
            )
            event.candidate = child
            parameters = self.memory.parameters
            if parameters and parameters.status == "usable":
                child.q_prior = child.q_posterior = parameters.q_prior
                child.parameter_version = parameters.version
                child.parameter_digest = parameters.fingerprint
            pool = self.memory.all(active_only=True)
            for case in cases:
                pair = ReplayResult(case.sample.sample_id, None, None)
                for name, version in (("old", e), ("new", child)):
                    try:
                        a = self._replay(case.sample, version, pool)
                        setattr(pair, name + "_application", a)
                        setattr(pair, name + "_score", replay_score(a))
                        if name == "new":
                            if a.sample.task_id == child.source_task_id:
                                a.q_before = a.q_after = child.q_posterior
                                a.update_reason = "duplicate_task"
                            else:
                                update_quality(child, a, parameters)
                            child.applications.append(a)
                        else:
                            a.q_before = a.q_after = e.q_posterior
                            a.update_reason = "parent_replay_no_update"
                    except ISOLATED as exc:
                        pair.errors[name] = str(exc)
                event.replays.append(pair)
            valid = [
                p
                for p in event.replays
                if p.old_score is not None and p.new_score is not None
            ]
            old = sum(p.old_score for p in valid)
            new = sum(p.new_score for p in valid)
            event.accepted = len(valid) >= self.min_replay and new > old
            event.decision = f"common_valid={len(valid)};old={old};new={new};minimum={self.min_replay}"
        except ISOLATED as exc:
            event.action = "error"
            event.error = str(exc)
            event.decision = "parent_kept"
        return self.memory.finish_revision(event)
