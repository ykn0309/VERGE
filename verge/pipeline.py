"""V2 learning: candidate-only retries, per-version evidence, isolated paired revision replay."""

from __future__ import annotations
from copy import deepcopy
from uuid import uuid4
import time, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
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
        margin_threshold=0.1,
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
        if not 0 < margin_threshold <= 2:
            raise ValueError("invalid margin_threshold")
        self.margin_threshold = margin_threshold
        self.stage = "idle"

    @property
    def stage(self):
        return getattr(self, "_stage", "idle")

    @stage.setter
    def stage(self, value):
        now = time.monotonic()
        record = getattr(self, "_active_record", None)
        if record is not None and hasattr(self, "_stage_started"):
            key = "stage:" + getattr(self, "_stage", "idle")
            record.timings[key] = record.timings.get(key, 0) + now - self._stage_started
        self._stage, self._stage_started = value, now

    def run(self, samples):
        return [self.run_sample(s) for s in samples]

    def evaluation_fingerprint(self, sample, pool):
        from pathlib import Path
        from verge.prompts import PROMPT_VERSION

        m = self.modules
        bs = [
            m.selector.backend,
            m.initial_generator.backend,
            m.judge.backend,
            *m.checkers.backends.values(),
        ]
        specs = [
            {
                "class": type(b).__name__,
                "model": getattr(b, "model", None),
                "thinking": getattr(b, "enable_thinking", None),
                "scenario": getattr(b, "scenario", None),
                "url": str(getattr(getattr(b, "_client", None), "base_url", "")),
            }
            for b in bs
        ]
        payload = {
            "task": sample.task,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "sample_id": sample.sample_id,
            "pool": [
                experience_payload(e, quality=m.selector.include_quality)
                for e in sorted(pool, key=lambda x: x.experience_id)
            ],
            "models": specs,
            "prompt": PROMPT_VERSION,
            "retriever": {k: getattr(m.retriever, k, None) for k in ("top_l", "rrf_k")},
            "embedding": {
                "class": type(getattr(m.retriever, "backend", None)).__name__,
                "url": str(
                    getattr(
                        getattr(getattr(m.retriever, "backend", None), "_client", None),
                        "base_url",
                        "",
                    )
                ),
                **{
                    k: getattr(getattr(m.retriever, "backend", None), k, None)
                    for k in ("model", "dimensions")
                },
            },
            "selection": {
                "max_selected": m.selector.max_selected,
                "include_quality": m.selector.include_quality,
            },
            "options": m.initial_generator.options,
            "training": self.training,
            "code": hashlib.sha256(
                b"".join(
                    p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
                )
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def evaluate(
        self, sample, pool, *, round_number=0, candidates=None, retriever=None
    ):
        m = self.modules
        self.last_evaluation_digest = (
            self.evaluation_fingerprint(sample, pool) if candidates is None else None
        )
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
        started = time.monotonic()
        rec = TrajectoryRecord(
            sample.sample_id, sample=sample, split="train" if self.training else "test"
        )
        self._active_record = rec
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

                def prepare(e):
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
                    a.evaluation_digest = self.last_evaluation_digest
                    a.selection_reason = getattr(
                        self.modules.selector, "last_reasons", {}
                    ).get(e.experience_id)
                    a.usage_reason = next(
                        (
                            u["reason"]
                            for u in rubric.experience_usage
                            if u["experience_id"] == e.experience_id
                        ),
                        None,
                    )
                    if a.cited:
                        a.reports = self.modules.checkers.check(a)
                    a.observation = aggregate_reports(a.reports)
                    return a

                if selected:
                    with ThreadPoolExecutor(max_workers=min(5, len(selected))) as pool:
                        fs = [
                            pool.submit(copy_context().run, prepare, e)
                            for e in selected
                        ]
                        prepared = [f.result() for f in fs]
                    for a in prepared:
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
        self.revise_pending(rec)
        self.stage = "idle"
        self._active_record = None
        rec.timings["sample_seconds"] = time.monotonic() - started
        self.trajectories.append(rec)
        return rec

    def revise_pending(self, rec):
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
                        self._revise(e, rec.sample_id, reasons)
                    )

    def _correct(self, sample, rec, rubric, judgment):
        conversation, codec, feedback = [], NumberedIDCodec(), None
        baseline = None
        for number in range(1, self.max_rounds + 1):
            attempt = ProposalAttempt(number)
            rec.proposal_attempts.append(attempt)
            try:
                self.stage = "critic_repair"
                if number == 1:
                    baseline, rec.baseline_assessment, repaired, changes = (
                        self.modules.critic.start(
                            sample,
                            rubric,
                            judgment,
                            conversation=conversation,
                            codec=codec,
                        )
                    )
                    rec.baseline_judgment = baseline
                    rec.baseline_diagnosis = deepcopy(
                        getattr(self.modules.critic, "last_diagnosis", {})
                    )
                else:
                    repaired, changes = self.modules.critic.repair(
                        sample,
                        rubric,
                        baseline,
                        number,
                        conversation=conversation,
                        codec=codec,
                        feedback={
                            "raw_judgment": feedback["raw_judgment"],
                            "final_judgment": feedback["final_judgment"],
                            "assessment": feedback["assessment"],
                            "margin_gain": feedback["margin_gain"],
                            "threshold": feedback["threshold"],
                            "previous_criteria": [
                                {"criterion_id": c.criterion_id, "question_index": i}
                                for i, c in enumerate(
                                    rec.proposal_attempts[-2].repair_rubric.criteria
                                )
                            ],
                        },
                    )
                attempt.repair_rubric, attempt.changes = repaired, changes
                attempt.diagnosis["repair"] = deepcopy(
                    getattr(self.modules.critic, "last_diagnosis", {})
                )
                if not changes:
                    rec.state = "no_rubric_revision"
                    return
                self.stage = "repair_judge"
                raw = self.modules.judge.judge(
                    repaired, sample.chosen, sample.rejected, task=sample.task
                )
                attempt.repair_judgment = raw
                final, assessment = self.modules.critic.assess(sample, repaired, raw)
                attempt.final_judgment, attempt.assessment = final, assessment
                attempt.diagnosis["judge_review"] = deepcopy(
                    getattr(self.modules.critic, "last_diagnosis", {})
                )
                attempt.margin_before, attempt.margin_after = (
                    baseline.margin,
                    final.margin,
                )
                attempt.margin_gain = final.margin - baseline.margin
                feedback = {
                    "initial_rubric": rubric.to_dict(),
                    "initial_raw_judgment": judgment.to_dict(),
                    "initial_final_judgment": baseline.to_dict(),
                    "rubric": repaired.to_dict(),
                    "raw_judgment": raw.to_dict(),
                    "final_judgment": final.to_dict(),
                    "assessment": assessment,
                    "diagnosis": deepcopy(attempt.diagnosis),
                    "margin_gain": attempt.margin_gain,
                    "threshold": self.margin_threshold,
                }
                # Decimal fractions avoid a boundary failure such as .3-.2 < .1.
                from fractions import Fraction

                def exact_margin(r, j):
                    from verge.calculator import _weight_ratios

                    w = _weight_ratios(r)
                    return sum(
                        w[x.criterion_id] * (int(x.chosen) - int(x.rejected))
                        for x in j.criterion_judgments
                    )

                gain = exact_margin(repaired, final) - exact_margin(rubric, baseline)
                if gain < Fraction(str(self.margin_threshold)):
                    rec.state = "repair_not_improved"
                    continue
                self.stage = "critic_extract"
                extracted = self.modules.critic.extract(
                    sample, repaired, final, changes, feedback
                )
                from verge.models import CriticProposal

                attempt.proposal = CriticProposal(assessment, [x[0] for x in extracted])
                admitted = []
                if not extracted:
                    rec.state = "no_admitted_experience"
                    return
                entries = {}
                for candidate, refs, reason in extracted:
                    entry = {
                        "experience_id": candidate.experience_id,
                        "change_indices": refs,
                        "reason": reason,
                        "reviews": [],
                        "methods": [candidate.method.to_dict()],
                        "admitted": False,
                        "error": None,
                    }
                    attempt.extractions.append(entry)
                    entries[candidate.experience_id] = entry
                feedback["candidate_support"] = [
                    {
                        "experience_id": c.experience_id,
                        "change_indices": refs,
                        "reason": reason,
                    }
                    for c, refs, reason in extracted
                ]
                active = [x[0] for x in extracted]
                self.stage = "experience_checker"
                reviews = self.modules.experience_checker.check_many(
                    sample, feedback, active
                )
                rejected = []
                for candidate in active:
                    entry = entries[candidate.experience_id]
                    review = reviews[candidate.experience_id]
                    if review is None:
                        entry["error"] = getattr(
                            self.modules.experience_checker, "last_errors", {}
                        ).get(candidate.experience_id, "invalid batch review item")
                        continue
                    entry["reviews"].append(review.to_dict())
                    if not review.approved:
                        rejected.append(candidate)
                # Retain first-pass approvals even if a rejected sibling fails its rewrite.
                if rejected:
                    try:
                        self.stage = "critic_rewrite"
                        methods = self.modules.critic.rewrite_many(
                            rejected,
                            reviews,
                            {
                                "task": sample.task,
                                "chosen": sample.chosen,
                                "rejected": sample.rejected,
                                "repair_evidence": feedback,
                            },
                        )
                        for c in rejected:
                            c.method = methods[c.experience_id]
                            entries[c.experience_id]["methods"].append(
                                c.method.to_dict()
                            )
                        self.stage = "experience_checker"
                        second = self.modules.experience_checker.check_many(
                            sample, feedback, rejected
                        )
                        for c in rejected:
                            result = second[c.experience_id]
                            if result is None:
                                entries[c.experience_id]["error"] = getattr(
                                    self.modules.experience_checker, "last_errors", {}
                                ).get(c.experience_id, "invalid batch review item")
                            else:
                                reviews[c.experience_id] = result
                                entries[c.experience_id]["reviews"].append(
                                    result.to_dict()
                                )
                    except ISOLATED as exc:
                        for c in rejected:
                            entries[c.experience_id]["error"] = str(exc)
                for candidate, refs, reason in extracted:
                    entry = entries[candidate.experience_id]
                    review = reviews[candidate.experience_id]
                    if entry["error"] or review is None or not review.approved:
                        continue
                    candidate.admission_evidence = {
                        **feedback,
                        "change_indices": refs,
                        "extraction_reason": reason,
                    }
                    self.memory.add(candidate, sample, review)
                    attempt.review = review
                    entry["admitted"] = True
                    rec.admitted_experience_ids.append(candidate.experience_id)
                    admitted.append(candidate)
                if not admitted:
                    rec.state = "no_admitted_experience"
                    return
                # Admission ends this task: the new method has not been applied.
                # Its Q remains at the configured prior until a later actual use.
                rec.state = "experience_admitted"
                return
            except ISOLATED as exc:
                attempt.error = str(exc)
                rec.state = "repair_failed"
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
        fingerprint = self.evaluation_fingerprint(sample, arm)
        for old in version.applications:
            if (
                old.evaluation_digest == fingerprint
                and old.sample.to_dict() == sample.to_dict()
            ):
                cached = deepcopy(old)
                cached.phase = "replay"
                cached.q_before = cached.q_after = None
                cached.update_reason = "not_evaluated"
                cached._cache_hit = True
                return cached
        recalled, selected, r, j = self.evaluate(sample, arm, retriever=retriever)
        a = application(sample, target, r, j, "replay")
        a.evaluation_digest = fingerprint
        a.selection_reason = getattr(self.modules.selector, "last_reasons", {}).get(
            target.experience_id
        )
        a.usage_reason = next(
            (
                u["reason"]
                for u in r.experience_usage
                if u["experience_id"] == target.experience_id
            ),
            None,
        )
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

    def _revise(self, e, sample_id, reasons, *, event_id=None):
        versions = {v.experience_id: v for v in self.memory.all()}
        cases, kinds = select_revision_cases(
            e, versions, reasons=reasons, window=self.window
        )
        event = RevisionRecord(
            event_id or "revision_" + uuid4().hex,
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

            def run_arm(case, version):
                factory = getattr(self, "replay_factory", None)
                if factory is None:
                    return self._replay(case.sample, version, pool)
                modules, cleanup = factory(case.sample)
                try:
                    runner = PipelineRunner(
                        modules,
                        self.memory,
                        self.trajectories,
                        training=self.training,
                        revision_enabled=False,
                    )
                    return runner._replay(case.sample, version, pool)
                finally:
                    cleanup()

            # Each concurrent arm owns its adapters; apply Q in fixed case order afterwards.
            tasks = {}
            with ThreadPoolExecutor(
                max_workers=4 if getattr(self, "replay_factory", None) else 1
            ) as executor:
                for i, case in enumerate(cases):
                    for name, version in (("old", e), ("new", child)):
                        tasks[i, name] = executor.submit(
                            copy_context().run, run_arm, case, deepcopy(version)
                        )
                for i, case in enumerate(cases):
                    pair = ReplayResult(case.sample.sample_id, None, None)
                    for name in ("old", "new"):
                        try:
                            a = tasks[i, name].result()
                            setattr(pair, name + "_application", a)
                            setattr(pair, name + "_score", replay_score(a))
                            if name == "old":
                                pair.old_cache_hit = getattr(a, "_cache_hit", False)
                                a.q_before = a.q_after = e.q_posterior
                                a.update_reason = "parent_replay_no_update"
                            else:
                                if a.sample.task_id == child.source_task_id:
                                    a.q_before = a.q_after = child.q_posterior
                                    a.update_reason = "duplicate_task"
                                else:
                                    update_quality(child, a, parameters)
                                child.applications.append(a)
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


def run_training_batch(
    samples, modules_factory, memory, trajectories, *, runner_options=None
):
    """Batch snapshot isolation; deterministic merge, then serial evolution.

    The factory supplies context-isolated modules and cleanup for each worker.
    Worker ledger changes are never copied over the live ledger wholesale.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from verge.memory import ExperienceMemory, TrajectoryLog

    options = dict(runner_options or {})
    snapshot = memory.path.read_bytes()
    initial_ids = {e.experience_id for e in memory.all()}

    def worker(sample):
        with TemporaryDirectory(prefix="verge-batch-worker-") as root:
            Path(root, "experience_memory.json").write_bytes(snapshot)
            local = ExperienceMemory(root, parameters=memory.parameters)
            modules, cleanup = modules_factory(sample)
            try:
                runner = PipelineRunner(
                    modules,
                    local,
                    TrajectoryLog(root),
                    **{**options, "revision_enabled": False},
                )
                record = runner.run_sample(sample)
                added = [e for e in local.all() if e.experience_id not in initial_ids]
                return record, added
            finally:
                cleanup()

    with ThreadPoolExecutor(max_workers=len(samples)) as executor:
        results = list(executor.map(worker, samples))
    # Stage the entire merge before publishing any batch evidence.
    with TemporaryDirectory(prefix="verge-batch-merge-") as root:
        Path(root, "experience_memory.json").write_bytes(snapshot)
        merged = ExperienceMemory(root, parameters=memory.parameters)
        for rec, added in results:
            for e in added:
                merged.add(e, e.source_sample, e.admission_review)
            for app in rec.applications:
                merged.record(app)
            for attempt in rec.proposal_attempts:
                for app in attempt.applications:
                    merged.record(app)
        memory._commit(merged._experiences, merged.revisions)
    records = [rec for rec, _ in results]
    # Batch end is the only point at which revision sees shared new evidence.
    if options.get("revision_enabled", True):
        modules, cleanup = modules_factory(samples[-1])
        try:
            runner = PipelineRunner(modules, memory, trajectories, **options)
            runner.revise_pending(records[-1])
        finally:
            cleanup()
    for rec in records:
        trajectories.append(rec)
    return records
