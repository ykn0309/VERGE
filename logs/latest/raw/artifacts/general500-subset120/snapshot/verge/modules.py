"""LLM adapters: applicability screening, rubric generation, judging, proposing."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from uuid import uuid4
from fractions import Fraction
import json

from verge.models import Method, Experience, Rubric, Criterion, CriterionJudgment
from verge.calculator import calculate_rubric_judgment
from verge.llm import (
    LLMFormatError,
    LLMSampleError,
    ROLE_CALL_ID,
    ROLE_ID_MAPPING,
    JUDGE_RESPONSE_SIDE,
)
from verge.prompts import NumberedIDCodec, llm_messages
from verge.retrieval import CandidateRoutes


def validation_feedback(role, projected, error):
    """Explain the failed contract using only the current request's short IDs."""
    message = str(error)[:200] + "；请完整返回规定结构。"
    if role != "generator":
        return message
    return (
        message
        + "本次仅返回analysis和criteria，每条标准只含question、weight。"
        + "不要输出经验引用、experience_usage或使用声明。"
    )


def request(
    backend,
    role,
    payload,
    validate,
    *,
    retries=2,
    input_bytes=100000,
    history=None,
    codec=None,
):
    codec = codec or NumberedIDCodec()
    current = payload
    for attempt in range(retries + 1):
        projected = codec.project(current)
        messages = llm_messages(role, projected, history=history)
        if sum(len(m["content"].encode()) for m in messages) > input_bytes:
            raise LLMSampleError(f"{role}: input budget exceeded; no truncation")
        call_id = uuid4().hex
        token = ROLE_CALL_ID.set(call_id)
        mapping = ROLE_ID_MAPPING.set(
            {"forward": codec.forward, "reverse": codec.reverse}
        )
        try:
            raw = backend.generate(role, projected, conversation=history)
            try:
                value = validate(codec.restore(dict(raw)))
            except (ValueError, TypeError, KeyError) as exc:
                if getattr(backend, "trace_hook", None):
                    backend.trace_hook(
                        {
                            "event_type": "contract_validation",
                            "call_id": call_id,
                            "role": role,
                            "validation_status": "invalid",
                            "error": str(exc),
                            "contract_attempt": attempt + 1,
                        }
                    )
                if attempt == retries:
                    raise LLMFormatError(
                        f"{role}: invalid contract: {str(exc)[:180]}"
                    ) from exc
                current = {
                    **payload,
                    "validation_feedback": validation_feedback(role, projected, exc),
                }
                continue
            if history is not None:
                history.extend(
                    [
                        messages[-1],
                        {
                            "role": "assistant",
                            "content": json.dumps(raw, ensure_ascii=False),
                        },
                    ]
                )
            if getattr(backend, "trace_hook", None):
                backend.trace_hook(
                    {
                        "event_type": "contract_validation",
                        "call_id": call_id,
                        "role": role,
                        "validation_status": "valid",
                        "contract_attempt": attempt + 1,
                    }
                )
            return value
        finally:
            ROLE_ID_MAPPING.reset(mapping)
            ROLE_CALL_ID.reset(token)
    raise AssertionError("unreachable")


def experience_payload(e):
    return {"experience_id": e.experience_id, **e.method.to_dict()}


class LLMSelector:
    def __init__(self, backend, max_selected=2, **options):
        if type(max_selected) is not int or not 0 <= max_selected <= 2:
            raise ValueError("max_selected must be 0..2")
        self.backend, self.max_selected, self.options = backend, max_selected, options

    def select(self, task, routes: CandidateRoutes, pool: dict[str, Experience]):
        if not routes.candidates or not self.max_selected:
            return [], {}
        candidates = []
        for c in routes.candidates:
            if c.experience_id not in pool:
                raise ValueError("candidate missing from experience pool")
            candidates.append(
                {
                    **experience_payload(pool[c.experience_id]),
                    "retrieval_source": c.source,
                    "historical_reward": c.historical_score,
                    "evidence_tasks": next(
                        (
                            x.tasks
                            for x in routes.historical.scored
                            if x.experience_id == c.experience_id
                        ),
                        0,
                    ),
                }
            )
        allowed = {c.experience_id for c in routes.candidates}

        def validate(data):
            if set(data) != {"selected_experience_ids", "selection_reasons"}:
                raise ValueError("expected selected IDs and reasons")
            ids, reasons = data["selected_experience_ids"], data["selection_reasons"]
            if (
                not isinstance(ids, list)
                or any(not isinstance(x, str) for x in ids)
                or len(ids) > self.max_selected
                or len(set(ids)) != len(ids)
                or set(ids) - allowed
            ):
                raise ValueError("select distinct provided IDs within budget")
            if (
                not isinstance(reasons, list)
                or any(
                    not isinstance(r, dict)
                    or set(r) != {"experience_id", "reason"}
                    or not isinstance(r["reason"], str)
                    or not r["reason"].strip()
                    for r in reasons
                )
                or len(reasons) != len(ids)
                or {r["experience_id"] for r in reasons} != set(ids)
            ):
                raise ValueError("nonempty reasons must cover selected IDs once")
            return [pool[eid] for eid in ids], {
                r["experience_id"]: r["reason"] for r in reasons
            }

        return request(
            self.backend,
            "selector",
            {"task": task, "candidates": candidates, "max_selected": self.max_selected},
            validate,
            **self.options,
        )


class LLMGenerator:
    def __init__(self, backend, **options):
        self.backend, self.options = backend, options

    def generate(self, task, experiences):
        if len(experiences) > 2:
            raise ValueError("Generator accepts at most two experiences")

        def validate(data):
            if (
                set(data) != {"analysis", "criteria"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
            ):
                raise ValueError("expected analysis and criteria")
            rid = "rubric_" + uuid4().hex
            if not isinstance(data["criteria"], list) or any(
                not isinstance(c, dict) or set(c) != {"question", "weight"}
                for c in data["criteria"]
            ):
                raise ValueError("each criterion requires question and weight only")
            criteria = [
                Criterion.from_dict({"criterion_id": f"{rid}_{i}", **c})
                for i, c in enumerate(data["criteria"], 1)
            ]
            rubric = Rubric(rid, criteria, data["analysis"])
            total = sum((Fraction(str(c.weight)) for c in criteria), Fraction(0))
            for criterion in criteria:
                criterion.weight = float(Fraction(str(criterion.weight)) / total)
            return rubric

        return request(
            self.backend,
            "generator",
            {"task": task, "experiences": [experience_payload(e) for e in experiences]},
            validate,
            **self.options,
        )


class LLMJudge:
    def __init__(self, backend, **options):
        self.backend, self.options = backend, options

    def judge(self, task, rubric, chosen, rejected):
        criteria = [
            {"criterion_id": c.criterion_id, "question": c.question}
            for c in rubric.criteria
        ]
        allowed = {c.criterion_id for c in rubric.criteria}

        def side(response, label):
            token = JUDGE_RESPONSE_SIDE.set(label)
            try:

                def validate(data):
                    if set(data) != {"criterion_judgments"} or not isinstance(
                        data["criterion_judgments"], list
                    ):
                        raise ValueError("expected criterion_judgments")
                    rows = data["criterion_judgments"]
                    if any(
                        not isinstance(x, dict)
                        or set(x) != {"criterion_id", "satisfied", "reason"}
                        or type(x["satisfied"]) is not bool
                        or not isinstance(x["reason"], str)
                        or not x["reason"].strip()
                        for x in rows
                    ):
                        raise ValueError("Boolean and nonempty reason required")
                    if (
                        len(rows) != len(allowed)
                        or {x["criterion_id"] for x in rows} != allowed
                    ):
                        raise ValueError("each criterion must appear exactly once")
                    return {x["criterion_id"]: x for x in rows}

                return request(
                    self.backend,
                    "judge",
                    {"task": task, "criteria": criteria, "response": response},
                    validate,
                    **self.options,
                )
            finally:
                JUDGE_RESPONSE_SIDE.reset(token)

        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(copy_context().run, side, chosen, "chosen")
            b = executor.submit(copy_context().run, side, rejected, "rejected")
            left, right = a.result(), b.result()
        return calculate_rubric_judgment(
            rubric,
            [
                CriterionJudgment(
                    c.criterion_id,
                    left[c.criterion_id]["satisfied"],
                    right[c.criterion_id]["satisfied"],
                    left[c.criterion_id]["reason"],
                    right[c.criterion_id]["reason"],
                )
                for c in rubric.criteria
            ],
        )


@dataclass
class CriticSession:
    history: list = field(default_factory=list)
    codec: NumberedIDCodec = field(default_factory=NumberedIDCodec)


class LLMCritic:
    def __init__(self, backend, max_candidates=2, **options):
        self.backend, self.max_candidates, self.options = (
            backend,
            max_candidates,
            options,
        )

    def propose(self, payload, *, source_task_id, allowed_parent_ids, session):
        allowed = set(allowed_parent_ids)

        def validate(data):
            if (
                set(data) != {"analysis", "candidates"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
                or not isinstance(data["candidates"], list)
                or len(data["candidates"]) > self.max_candidates
            ):
                raise ValueError("expected analysis and bounded candidates")
            result = []
            for c in data["candidates"]:
                if not isinstance(c, dict) or set(c) != {"method", "parent_ids"}:
                    raise ValueError("candidate requires method and parent_ids")
                parents = c["parent_ids"]
                if (
                    not isinstance(parents, list)
                    or any(not isinstance(x, str) for x in parents)
                    or len(set(parents)) != len(parents)
                    or set(parents) - allowed
                ):
                    raise ValueError("parent IDs must reference provided experiences")
                result.append(
                    Experience(
                        "exp_" + uuid4().hex,
                        Method.from_dict(c["method"]),
                        source_task_id,
                        parents,
                    )
                )
            return data["analysis"], result

        return request(
            self.backend,
            "critic",
            {
                **payload,
                "max_candidates": self.max_candidates,
                "allowed_parent_ids": list(allowed_parent_ids),
            },
            validate,
            history=session.history,
            codec=session.codec,
            **self.options,
        )


@dataclass
class RoleModules:
    selector: LLMSelector
    generator: LLMGenerator
    judge: LLMJudge
    critic: LLMCritic
