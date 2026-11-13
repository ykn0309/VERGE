"""Shared task-only draft, equal singleton trials, and source-only repair."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, dataclass
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


def preference_evidence(sample):
    """Preserve annotator disagreement and original response numbering.

    Never infer the original order from the normalized chosen/rejected strings.
    Missing metadata remains explicitly unknown.
    """
    chosen = sample.metadata.get("chosen_response")
    return {
        "original_response_mapping": {
            "response_1": (
                "chosen" if chosen == 1 else "rejected" if chosen == 2 else "unknown"
            ),
            "response_2": (
                "chosen" if chosen == 2 else "rejected" if chosen == 1 else "unknown"
            ),
        },
        "score_convention": "negative favors original response_1; positive favors original response_2; zero is a tie",
        "individual_preference": sample.metadata.get("individual_preference", []),
    }


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
        if not 0 <= critic_rounds <= 2:
            raise ValueError("Critic permits at most two rounds")
        self.modules = modules
        self.embedder = embedder
        self.task_group_limit = task_group_limit
        self.quota = quota
        self.critic_rounds = critic_rounds

    def execute(self, sample, draft, experiences):
        if len(experiences) > 1:
            raise ValueError("a path accepts at most one experience")
        path = ExecutionPath([e.experience_id for e in experiences])
        try:
            path.rubric = self.modules.generator.revise(sample.task, draft, experiences)
            path.judgment = self.modules.judge.judge(
                sample.task, path.rubric, sample.chosen, sample.rejected
            )
        except RECOVERABLE as exc:
            path.error = f"{type(exc).__name__}: {exc}"
        return path

    def run_sample(self, sample, state, *, training=True):
        record = TrajectoryRecord(
            sample, {}, {}, mode="train" if training else "inference"
        )
        vector = None
        try:
            vector = self.embedder.embed(sample.task)
            routes = retrieve_candidates(
                vector,
                state.clusters,
                state.ledger,
                task_group_limit=self.task_group_limit,
                quota=self.quota,
            )
            record.retrieval = asdict(routes)
            selected, record.selection_reasons = self.modules.selector.select(
                sample.task, routes, state.experiences, training=training
            )
            if len(selected) > (3 if training else 1):
                raise ValueError("Selector exceeded the mode-specific budget")
            record.draft = self.modules.generator.draft(sample.task)
        except RECOVERABLE as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            return SampleResult(record, vector, [])

        inputs = [[], *([e] for e in selected)] if training else [selected]
        with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
            futures = [
                executor.submit(
                    copy_context().run, self.execute, sample, record.draft, es
                )
                for es in inputs
            ]
            record.paths = [f.result() for f in futures]
        result = SampleResult(record, vector, [])
        if training and all(p.correct is False for p in record.paths):
            self._repair(sample, state, routes, record, result)
        return result

    def _repair(self, sample, state, routes, record, result):
        session = CriticSession()
        parents = [c.experience_id for c in routes.candidates]
        payload = {
            "task": sample.task,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "preference_evidence": preference_evidence(sample),
            "draft": record.draft.to_dict(),
            "initial_paths": [p.to_dict() for p in record.paths],
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
            if len(candidates) > 1:
                raise ValueError("Critic exceeded the single-candidate budget")
            round_record = CriticRound(number, analysis, [])
            record.critic_rounds.append(round_record)
            if not candidates:
                return
            experience = candidates[0]
            path = self.execute(sample, record.draft, [experience])
            accepted = False
            reason = "technical_failure" if path.correct is None else "incorrect"
            if path.correct is True:
                try:
                    text = json.dumps(
                        experience.method.to_dict(), ensure_ascii=False, sort_keys=True
                    )
                    embedded = self.embedder.embed(text)
                except RECOVERABLE as exc:
                    reason = f"embedding_failure: {exc}"
                else:
                    result.admitted.append((experience, embedded))
                    accepted, reason = True, "source_correct"
            round_record.attempts.append(
                CandidateAttempt(experience, path, accepted, reason)
            )
            if accepted or path.correct is not False:
                return
            # Same Critic conversation carries the original evidence. Only an
            # explicit content failure, not a technical error, permits round 2.
            payload = {"previous_attempt": round_record.attempts[0].to_dict()}
