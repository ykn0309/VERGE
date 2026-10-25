"""V2 role adapters and direct per-version hybrid retrieval."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from collections import Counter
from dataclasses import dataclass
from math import log, sqrt, isfinite
from uuid import uuid4
import hashlib, json, re
from verge.models import (
    Experience,
    Method,
    Parameter,
    Criterion,
    Rubric,
    CriterionJudgment,
    CheckerReport,
    CriticProposal,
    ExperienceUpdate,
)
from verge.llm import (
    LLMFormatError,
    LLMSampleError,
    ROLE_ID_MAPPING,
    JUDGE_RESPONSE_SIDE,
    FakeLLMBackend,
)
from verge.prompts import (
    NumberedIDCodec,
    llm_messages,
    experience_payload,
    evidence_payload,
)
from verge.calculator import calculate_rubric_judgment


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
        projected = codec.project({**current, "_message_budget": input_bytes})
        messages = llm_messages(role, projected, history=history)
        if sum(len(m["content"].encode()) for m in messages) > input_bytes:
            raise LLMSampleError(role + ": input budget exceeded; no truncation")
        token = ROLE_ID_MAPPING.set(
            {"forward": codec.forward, "reverse": codec.reverse}
        )
        try:
            raw = backend.generate(role, projected, conversation=history)
            try:
                return validate(codec.restore(dict(raw)))
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == retries:
                    raise LLMFormatError(
                        role + ": invalid contract: " + str(exc)[:180]
                    ) from exc
                current = {
                    **payload,
                    "validation_feedback": "上次结构无效："
                    + str(exc)[:160]
                    + "。请完整返回角色规定的结构。",
                }
        finally:
            ROLE_ID_MAPPING.reset(token)
    raise AssertionError("unreachable")


class FakeRetriever:
    def __init__(self, top_l=20):
        self.top_l = top_l
        self.last_trace = []

    def retrieve(self, task, experiences):
        return sorted(
            (e for e in experiences if e.status == "active"),
            key=lambda e: e.experience_id,
        )[: self.top_l]

    def fork(self):
        return FakeRetriever(self.top_l)


class HybridRetriever:
    TOKENIZER = "lowercase-words-cjk-bigrams-v2"

    def __init__(self, backend, top_l=20, *, rrf_k=60):
        if top_l < 1 or rrf_k < 1:
            raise ValueError("retrieval limits must be positive")
        self.backend = backend
        self.top_l = top_l
        self.rrf_k = rrf_k
        self.cache = {}
        self.last_trace = []

    def fork(self):
        result = HybridRetriever(self.backend, self.top_l, rrf_k=self.rrf_k)
        result.cache = dict(self.cache)
        return result

    @staticmethod
    def tokens(text):
        result = re.findall(r"[a-z0-9_]+", text.lower())
        for seq in re.findall(r"[\u3400-\u9fff]+", text):
            result.extend(
                seq if len(seq) == 1 else [seq[i : i + 2] for i in range(len(seq) - 1)]
            )
        return result

    def _vector(self, text):
        if text not in self.cache:
            v = tuple(self.backend.embed(text))
            norm = sqrt(sum(x * x for x in v))
            if not v or not all(isfinite(x) for x in v) or norm == 0:
                raise ValueError("invalid embedding")
            self.cache[text] = tuple(x / norm for x in v)
        return self.cache[text]

    def retrieve(self, task, experiences):
        items = sorted(
            (e for e in experiences if e.status == "active"),
            key=lambda e: e.experience_id,
        )
        if not items:
            self.last_trace = []
            return []
        docs = {
            e.experience_id: json.dumps(
                e.method.to_dict(), ensure_ascii=False, sort_keys=True
            )
            for e in items
        }
        q = self._vector(task)
        dense = {}
        for eid, text in docs.items():
            v = self._vector(text)
            if len(v) != len(q):
                raise ValueError("embedding dimensions changed")
            dense[eid] = sum(a * b for a, b in zip(q, v))
        terms = {eid: Counter(self.tokens(text)) for eid, text in docs.items()}
        query = Counter(self.tokens(task))
        avg = sum(sum(v.values()) for v in terms.values()) / len(items) or 1
        sparse = {}
        for eid, counts in terms.items():
            score = 0
            for t in query:
                tf = counts[t]
                if not tf:
                    continue
                df = sum(t in v for v in terms.values())
                idf = log(1 + (len(items) - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * (tf * 2.5)
                    / (tf + 1.5 * (0.25 + 0.75 * sum(counts.values()) / avg))
                )
            if score > 0:
                sparse[eid] = score
        ranked_dense = sorted(dense, key=lambda k: (-dense[k], k))
        ranked_sparse = sorted(sparse, key=lambda k: (-sparse[k], k))
        ranks = [
            {eid: i for i, eid in enumerate(xs, 1)}
            for xs in (ranked_dense, ranked_sparse)
        ]
        scores = {
            eid: sum(1 / (self.rrf_k + r[eid]) for r in ranks if eid in r)
            for eid in docs
        }
        order = sorted(docs, key=lambda k: (-scores[k], k))
        self.last_trace = [
            {
                "experience_id": eid,
                "vector_rank": ranks[0].get(eid),
                "keyword_rank": ranks[1].get(eid),
                "score": scores[eid],
            }
            for eid in order
        ]
        lookup = {e.experience_id: e for e in items}
        return [lookup[eid] for eid in order[: self.top_l]]


class LLMSelector:
    def __init__(self, backend, max_selected=5, **options):
        self.backend = backend
        self.max_selected = max_selected
        self.options = options
        if not 0 <= max_selected <= 5:
            raise ValueError("max_selected must be 0..5")

    def select(self, task, candidates, *, training=True):
        if not candidates or not self.max_selected:
            return []

        def validate(data):
            if set(data) != {"selected_experience_ids"}:
                raise ValueError("expected selected_experience_ids")
            ids = data["selected_experience_ids"]
            lookup = {e.experience_id: e for e in candidates}
            if (
                not isinstance(ids, list)
                or len(ids) > self.max_selected
                or len(ids) != len(set(ids))
                or set(ids) - lookup.keys()
            ):
                raise ValueError("select distinct supplied IDs only")
            return [lookup[n] for n in ids]

        return request(
            self.backend,
            "selector",
            {
                "task": task,
                "candidates": [experience_payload(e, quality=True) for e in candidates],
                "max_selected": self.max_selected,
            },
            validate,
            **self.options,
        )


class LLMInitialRubricGenerator:
    def __init__(self, backend, **options):
        self.backend = backend
        self.options = options

    def generate(self, task, selected, *, round_number=0):
        lookup = {e.experience_id: e for e in selected}

        def validate(data):
            if (
                set(data) != {"analysis", "criteria"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
            ):
                raise ValueError("analysis and criteria required")
            rid = "rubric_" + uuid4().hex
            if not isinstance(data["criteria"], list) or any(
                set(c)
                != {
                    "question",
                    "weight",
                    "source_experiences",
                    "bindings",
                    "binding_evidence",
                }
                for c in data["criteria"]
            ):
                raise ValueError(
                    "criteria require question, weight, source_experiences, bindings and binding_evidence only"
                )
            criteria = [
                Criterion.from_dict({"criterion_id": f"{rid}_{i}", **c})
                for i, c in enumerate(data["criteria"], 1)
            ]
            for c in criteria:
                if set(c.source_experiences) - lookup.keys():
                    raise ValueError("unknown experience source")
                if c.source_experiences:
                    method = lookup[c.source_experiences[0]].method
                    if c.bindings.keys() != method.parameters.keys():
                        raise ValueError("bind every defined parameter exactly once")
                    if any(v not in task for v in c.binding_evidence.values()):
                        raise ValueError(
                            "binding evidence must be a verbatim task fragment"
                        )
                    if method.template and c.question != method.template.format(
                        **c.bindings
                    ):
                        raise ValueError(
                            "question must render the supplied template exactly"
                        )
            return Rubric(rid, criteria, data["analysis"])

        payload = {
            "task": task,
            "experiences": [experience_payload(e) for e in selected],
        }
        if isinstance(self.backend, FakeLLMBackend):
            payload["_round"] = round_number
        return request(
            self.backend, "initial_generator", payload, validate, **self.options
        )

    def apply(self, task, candidates, round_number):
        return self.generate(task, candidates, round_number=round_number)


class JudgeOutputError(LLMFormatError):
    pass


class LLMJudge:
    def __init__(self, backend, **options):
        self.backend = backend
        self.options = options

    def judge(self, rubric, chosen, rejected, *, task):
        criteria = [
            {"criterion_id": c.criterion_id, "question": c.question}
            for c in rubric.criteria
        ]

        def side(response, label):
            token = JUDGE_RESPONSE_SIDE.set(label)

            def validate(data):
                if set(data) != {"criterion_judgments"}:
                    raise ValueError("unexpected judge fields")
                xs = data["criterion_judgments"]
                out = {}
                for x in xs:
                    if (
                        set(x) != {"criterion_id", "satisfied", "reason"}
                        or type(x["satisfied"]) is not bool
                        or not isinstance(x["reason"], str)
                        or not x["reason"].strip()
                    ):
                        raise ValueError("invalid Boolean or evidence")
                    if x["criterion_id"] in out:
                        raise ValueError("duplicate criterion")
                    out[x["criterion_id"]] = x
                if set(out) != {c.criterion_id for c in rubric.criteria}:
                    raise ValueError("incomplete criterion coverage")
                return out

            try:
                return request(
                    self.backend,
                    "judge",
                    {"task": task, "response": response, "criteria": criteria},
                    validate,
                    **self.options,
                )
            finally:
                JUDGE_RESPONSE_SIDE.reset(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(copy_context().run, side, r, l)
                for r, l in ((chosen, "chosen"), (rejected, "rejected"))
            ]
            results = []
            errors = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as exc:
                    results.append(None)
                    errors.append(exc)
            if errors:
                raise next(
                    (
                        e
                        for e in errors
                        if not isinstance(e, (LLMFormatError, LLMSampleError))
                    ),
                    errors[0],
                )
        a, b = results
        return calculate_rubric_judgment(
            rubric,
            [
                CriterionJudgment(
                    c.criterion_id,
                    a[c.criterion_id]["satisfied"],
                    b[c.criterion_id]["satisfied"],
                    a[c.criterion_id]["reason"],
                    b[c.criterion_id]["reason"],
                )
                for c in rubric.criteria
            ],
        )


class LLMCritic:
    def __init__(self, backend, max_candidates=5, **options):
        self.backend = backend
        self.max_candidates = max_candidates
        self.options = options

    def propose(
        self,
        sample,
        rubric,
        judgment,
        round_number,
        *,
        conversation,
        codec,
        feedback=None,
    ):
        def validate(data):
            if (
                set(data) != {"analysis", "candidates"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
            ):
                raise ValueError("critic analysis required")
            if (
                not isinstance(data["candidates"], list)
                or len(data["candidates"]) > self.max_candidates
            ):
                raise ValueError("candidate limit exceeded")
            methods = [Method.from_dict(x) for x in data["candidates"]]
            if len({m.to_json() for m in methods}) != len(methods):
                raise ValueError("duplicate candidate methods")
            return CriticProposal(
                data["analysis"],
                [
                    Experience(
                        "exp_" + uuid4().hex, m, sample.sample_id, status="candidate"
                    )
                    for m in methods
                ],
            )

        answers = {j.criterion_id: j for j in judgment.criterion_judgments}
        payload = {
            "task": sample.task,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "criteria": [
                {**c.to_dict(), **answers[c.criterion_id].to_dict()}
                for c in rubric.criteria
            ],
            "max_candidates": self.max_candidates,
            "round": round_number,
            "previous_feedback": feedback,
        }
        return request(
            self.backend,
            "critic",
            payload,
            validate,
            history=conversation,
            codec=codec,
            **self.options,
        )


class LLMExperienceUpdater:
    def __init__(self, backend, **options):
        self.backend = backend
        self.options = options

    def update(self, experience, cases, reasons, *, case_kinds=None):
        return request(
            self.backend,
            "experience_updater",
            {
                "experience": experience_payload(experience, quality=True),
                "reasons": reasons,
                "cases": [
                    {
                        **evidence_payload(c),
                        "case_kind": (case_kinds or {}).get(
                            c.sample.sample_id, "background"
                        ),
                    }
                    for c in cases
                ],
            },
            ExperienceUpdate.from_dict,
            **self.options,
        )


class LLMCheckers:
    def __init__(self, backends, **options):
        self.backends = backends
        self.options = options

    def check(self, app):
        related = [
            c for c in app.rubric.criteria if app.version_id in c.source_experiences
        ]
        if not related:
            return []
        method = {"experience_id": app.version_id, **app.method.to_dict()}
        basic = {"task": app.sample.task, "experience": method}
        criteria = [c.to_dict() for c in related]
        answers = {j.criterion_id: j for j in app.judgment.criterion_judgments}
        # Anonymous side assignment is deterministic per task, and never includes preference labels.
        swap = int(hashlib.sha256(app.sample.sample_id.encode()).hexdigest(), 16) % 2
        responses = [app.sample.chosen, app.sample.rejected]
        if swap:
            responses.reverse()
        judged = []
        for c in related:
            j = answers[c.criterion_id]
            sides = [(j.chosen, j.chosen_reason), (j.rejected, j.rejected_reason)]
            if swap:
                sides.reverse()
            judged.append(
                {
                    "criterion_id": c.criterion_id,
                    "question": c.question,
                    "a": {"satisfied": sides[0][0], "reason": sides[0][1]},
                    "b": {"satisfied": sides[1][0], "reason": sides[1][1]},
                }
            )
        payloads = {
            "selector": basic,
            "generator": {**basic, "criteria": criteria},
            "judge": {
                "task": app.sample.task,
                "response_a": responses[0],
                "response_b": responses[1],
                "criteria": judged,
            },
        }

        def run(stage):
            role = stage + "_checker"

            def validate(data):
                if set(data) != {"status", "evidence", "details"} or data[
                    "status"
                ] not in ("correct", "error", "unknown"):
                    raise ValueError(
                        "checker must report correct/error/unknown and evidence/details"
                    )
                report = CheckerReport.from_dict({"stage": stage, **data})
                if {d.criterion_id for d in report.details} - {
                    c.criterion_id for c in related
                } or (stage == "selector" and report.details):
                    raise ValueError(
                        "checker details must reference supplied criteria; selector is task-level"
                    )
                return report

            try:
                return request(
                    self.backends[stage],
                    role,
                    payloads[stage],
                    validate,
                    **self.options,
                )
            except (LLMFormatError, LLMSampleError) as exc:
                return CheckerReport(stage, "failed", str(exc))

        with ThreadPoolExecutor(max_workers=3) as pool:
            fs = [
                pool.submit(copy_context().run, run, s)
                for s in ("selector", "generator", "judge")
            ]
            return [f.result() for f in fs]


@dataclass
class LLMModules:
    backend: object
    retriever: object
    selector: object
    initial_generator: object
    judge: object
    critic: object
    experience_updater: object
    checkers: object


def make_llm_modules(
    backend,
    *,
    retriever=None,
    max_selected=5,
    max_candidates=5,
    retries=2,
    input_bytes=100000,
    role_backends=None,
):
    role_backends = role_backends or {}
    get = lambda r: role_backends.get(r, backend)
    opts = {"retries": retries, "input_bytes": input_bytes}
    return LLMModules(
        backend,
        retriever or FakeRetriever(),
        LLMSelector(get("selector"), max_selected, **opts),
        LLMInitialRubricGenerator(get("initial_generator"), **opts),
        LLMJudge(get("judge"), **opts),
        LLMCritic(get("critic"), max_candidates, **opts),
        LLMExperienceUpdater(get("experience_updater"), **opts),
        LLMCheckers(
            {s: get(s + "_checker") for s in ("selector", "generator", "judge")}, **opts
        ),
    )


def make_fake_modules(scenario="attempt_1_success", **options):
    return make_llm_modules(FakeLLMBackend(scenario), **options)
