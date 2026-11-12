"""Bounded subset execution, fixed actual results, and source-only validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, dataclass
from itertools import combinations
import json

from verge.llm import LLMFormatError, LLMSampleError
from verge.models import ExecutionPath, TrajectoryRecord, CandidateAttempt, CriticRound
from verge.modules import CriticSession, experience_payload
from verge.retrieval import CandidateQuota, retrieve_candidates

RECOVERABLE = (LLMFormatError, LLMSampleError)


@dataclass
class SampleResult:
    record: TrajectoryRecord
    task_vector: tuple | None
    admitted: list


class PipelineRunner:
    def __init__(
        self,
        modules,
        embedder,
        *,
        task_group_limit=3,
        quota=CandidateQuota(),
        critic_rounds=2,
    ):
        self.modules = modules
        self.embedder = embedder
        self.task_group_limit = task_group_limit
        self.quota = quota
        self.critic_rounds = critic_rounds

    def execute(self, sample, experiences):
        path = ExecutionPath([e.experience_id for e in experiences])
        try:
            path.rubric = self.modules.generator.generate(sample.task, experiences)
            path.judgment = self.modules.judge.judge(
                sample.task, path.rubric, sample.chosen, sample.rejected
            )
        except RECOVERABLE as exc:
            path.error = f"{type(exc).__name__}: {exc}"
        return path

    def execute_subsets(self, sample, experiences, *, baseline=None):
        """Run each distinct subset once; reuse the task's existing baseline."""
        if len(experiences) > 2:
            raise ValueError("subset execution accepts at most two experiences")
        subsets = [
            list(c)
            for size in range(len(experiences) + 1)
            for c in combinations(experiences, size)
        ]
        paths = {}
        if baseline is not None:
            if baseline.provided_experience_ids:
                raise ValueError("baseline must provide no experience")
            paths[()] = baseline
        pending = [s for s in subsets if tuple(e.experience_id for e in s) not in paths]
        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = [
                    (
                        subset,
                        executor.submit(
                            copy_context().run, self.execute, sample, subset
                        ),
                    )
                    for subset in pending
                ]
                for subset, future in futures:
                    paths[tuple(e.experience_id for e in subset)] = future.result()
        return [paths[tuple(e.experience_id for e in subset)] for subset in subsets]

    def run_sample(self, sample, state, *, training=True):
        try:
            vector = self.embedder.embed(sample.task)
            routes = retrieve_candidates(
                vector,
                state.clusters,
                state.ledger,
                task_group_limit=self.task_group_limit,
                quota=self.quota,
            )
            selected, reasons = self.modules.selector.select(
                sample.task, routes, state.experiences
            )
        except RECOVERABLE as exc:
            record = TrajectoryRecord(
                sample, {}, {}, None, None, error=f"{type(exc).__name__}: {exc}"
            )
            return SampleResult(record, None, [])
        if len(selected) > 2:
            raise ValueError("Selector exceeded the two-experience budget")
        if training:
            paths = self.execute_subsets(sample, selected)
            baseline = paths[0]
            actual = paths[-1] if selected else None
        else:
            path = self.execute(sample, selected)
            baseline, actual = (None, path) if selected else (path, None)
            paths = [path]
        record = TrajectoryRecord(
            sample, asdict(routes), reasons, baseline, actual, subset_paths=paths
        )
        result = SampleResult(record, vector, [])
        if not training or record.used_path.correct is not False:
            return result
        self._repair(sample, state, routes, selected, record, result)
        return result

    def _repair(self, sample, state, routes, selected, record, result):
        session = CriticSession()
        parents = [c.experience_id for c in routes.candidates]
        payload = {
            "task": sample.task,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "initial_paths": [p.to_dict() for p in record.subset_paths],
            "actual_experience_ids": [e.experience_id for e in selected],
            "experiences": [
                experience_payload(state.experiences[eid]) for eid in parents
            ],
            "selection_reasons": [
                {"experience_id": eid, "reason": reason}
                for eid, reason in record.selection_reasons.items()
            ],
            "retrieval_candidates": [asdict(c) for c in routes.candidates],
        }
        for number in range(1, self.critic_rounds + 1):
            try:
                analysis, candidates = self.modules.critic.propose(
                    payload,
                    source_task_id=sample.task_id,
                    allowed_parent_ids=parents,
                    session=session,
                )
            except RECOVERABLE as exc:
                record.critic_rounds.append(
                    CriticRound(number, "", [], f"{type(exc).__name__}: {exc}")
                )
                return
            if len(candidates) > 2:
                raise ValueError("Critic exceeded the two-candidate budget")
            round_record = CriticRound(number, analysis, [])
            record.critic_rounds.append(round_record)
            if not candidates:
                return
            paths = self.execute_subsets(sample, candidates, baseline=record.baseline)
            round_record.subset_paths = paths
            singles = paths[1 : 1 + len(candidates)]
            passing = [i for i, path in enumerate(singles) if path.correct is True]
            joint = len(candidates) == 2 and not passing and paths[-1].correct is True
            admitted = set()
            embedding_errors = {}
            # Joint-only repair is published as a pair; embedding failure cannot
            # silently downgrade its evidence to a standalone admission.
            groups = [list(range(2))] if joint else [[i] for i in passing]
            for group in groups:
                embedded = []
                try:
                    for i in group:
                        e = candidates[i]
                        text = json.dumps(
                            e.method.to_dict(), ensure_ascii=False, sort_keys=True
                        )
                        embedded.append((e, self.embedder.embed(text)))
                except RECOVERABLE as exc:
                    for i in group:
                        embedding_errors[i] = f"embedding_failure: {exc}"
                else:
                    result.admitted.extend(embedded)
                    admitted.update(group)
            for i, (experience, path) in enumerate(zip(candidates, singles)):
                accepted = i in admitted
                reason = (
                    ("joint_correct" if joint else "standalone_correct")
                    if accepted
                    else (
                        embedding_errors.get(i)
                        or ("technical_failure" if path.error else "incorrect")
                    )
                )
                subset = (
                    (
                        [e.experience_id for e in candidates]
                        if joint
                        else [experience.experience_id]
                    )
                    if accepted
                    else []
                )
                round_record.attempts.append(
                    CandidateAttempt(experience, path, accepted, reason, subset)
                )
            if admitted:
                return
            # Format/transport/embedding failures are not semantic counterexamples.
            if not any(p.correct is False for p in singles) or embedding_errors:
                return
            payload = {
                "previous_attempts": [a.to_dict() for a in round_record.attempts],
                "validation_paths": [p.to_dict() for p in paths],
            }
