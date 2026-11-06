"""V2 role adapters and direct per-version hybrid retrieval."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from contextvars import copy_context
from collections import Counter
from dataclasses import dataclass
from math import log, sqrt, isfinite
from uuid import uuid4
import hashlib, json, re
from verge.models import (
    Experience,
    Method,
    Criterion,
    Rubric,
    CriterionJudgment,
    CheckerReport,
    CriticProposal,
    ExperienceUpdate,
    ExperienceReview,
)
from verge.llm import (
    LLMFormatError,
    LLMSampleError,
    ROLE_ID_MAPPING,
    ROLE_CALL_ID,
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
        call_id = uuid4().hex
        call_token = ROLE_CALL_ID.set(call_id)

        def trace_validation(status, error=None):
            hook = getattr(backend, "trace_hook", None)
            if hook:
                hook(
                    {
                        "event_type": "contract_validation",
                        "role": role,
                        "call_id": call_id,
                        "contract_attempt": attempt + 1,
                        "validation_status": status,
                        "error": error,
                    }
                )

        try:
            raw = backend.generate(role, projected, conversation=history)
            try:
                value = validate(codec.restore(dict(raw)))
                trace_validation("valid")
                return value
            except (ValueError, TypeError, KeyError) as exc:
                trace_validation("invalid", str(exc))
                if attempt == retries:
                    raise LLMFormatError(
                        role + ": invalid contract: " + str(exc)[:180]
                    ) from exc
                current = {
                    **(
                        {k: payload[k] for k in ("round",) if k in payload}
                        if history
                        else payload
                    ),
                    "validation_feedback": "上次结构无效："
                    + str(exc)[:160]
                    + "。请完整返回角色规定的结构。",
                }
        finally:
            ROLE_CALL_ID.reset(call_token)
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
        self._cache_lock = Lock()
        self._pending = {}
        self.last_trace = []

    def fork(self):
        result = HybridRetriever(self.backend, self.top_l, rrf_k=self.rrf_k)
        result.share_cache(self)
        return result

    @staticmethod
    def tokens(text):
        result = re.findall(r"[a-z0-9_]+", text.lower())
        for seq in re.findall(r"[\u3400-\u9fff]+", text):
            result.extend(
                seq if len(seq) == 1 else [seq[i : i + 2] for i in range(len(seq) - 1)]
            )
        return result

    def share_cache(self, other):
        self.cache, self._cache_lock, self._pending = (
            other.cache,
            other._cache_lock,
            other._pending,
        )

    def _vector(self, text):
        with self._cache_lock:
            if text in self.cache:
                return self.cache[text]
            owner = text not in self._pending
            pending = self._pending.setdefault(text, Future())
        if not owner:
            return pending.result()
        try:
            v = tuple(self.backend.embed(text))
            norm = sqrt(sum(x * x for x in v))
            if not v or not all(isfinite(x) for x in v) or norm == 0:
                raise ValueError("invalid embedding")
            result = tuple(x / norm for x in v)
            with self._cache_lock:
                self.cache[text] = result
            pending.set_result(result)
            return result
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._cache_lock:
                self._pending.pop(text, None)

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
    def __init__(self, backend, max_selected=5, *, include_quality=True, **options):
        self.backend = backend
        self.max_selected = max_selected
        self.include_quality = include_quality
        self.options = options
        if not 0 <= max_selected <= 5:
            raise ValueError("max_selected must be 0..5")

    def select(self, task, candidates, *, training=True):
        self.last_reasons = {}
        if not candidates or not self.max_selected:
            return []

        def validate(data):
            if set(data) != {"selected_experience_ids", "selection_reasons"}:
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
            reasons = data["selection_reasons"]
            if (
                not isinstance(reasons, list)
                or len(reasons) != len(ids)
                or any(
                    set(r) != {"experience_id", "reason"}
                    or not isinstance(r["reason"], str)
                    or not r["reason"].strip()
                    for r in reasons
                )
                or {r["experience_id"] for r in reasons} != set(ids)
            ):
                raise ValueError("selection reasons must cover selected experiences")
            self.last_reasons = {r["experience_id"]: r["reason"] for r in reasons}
            return [lookup[n] for n in ids]

        return request(
            self.backend,
            "selector",
            {
                "task": task,
                "candidates": [
                    experience_payload(e, quality=self.include_quality)
                    for e in candidates
                ],
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
                set(data) != {"analysis", "criteria", "experience_usage"}
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
                }
                for c in data["criteria"]
            ):
                raise ValueError(
                    "criteria require question, weight and source_experiences only"
                )
            criteria = [
                Criterion.from_dict({"criterion_id": f"{rid}_{i}", **c})
                for i, c in enumerate(data["criteria"], 1)
            ]
            for c in criteria:
                if set(c.source_experiences) - lookup.keys():
                    raise ValueError("unknown experience source")
            usage = data["experience_usage"]
            cited = {x for c in criteria for x in c.source_experiences}
            if (
                not isinstance(usage, list)
                or len(usage) != len(lookup)
                or any(
                    set(u) != {"experience_id", "used", "reason"}
                    or type(u["used"]) is not bool
                    or not isinstance(u["reason"], str)
                    or not u["reason"].strip()
                    or u["used"] != (u["experience_id"] in cited)
                    for u in usage
                )
                or {u["experience_id"] for u in usage} != lookup.keys()
            ):
                raise ValueError(
                    "usage must cover every experience and match citations"
                )
            return Rubric(rid, criteria, data["analysis"], usage)

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
        self.cache = {}
        self.cache_lock = Lock()

    def share_cache(self, other):
        self.cache, self.cache_lock = other.cache, other.cache_lock

    def judge(self, rubric, chosen, rejected, *, task):
        from verge.prompts import PROMPT_VERSION

        b = self.backend
        key = json.dumps(
            {
                "task": task,
                "chosen": chosen,
                "rejected": rejected,
                "questions": [c.question for c in rubric.criteria],
                "model": getattr(b, "model", None),
                "backend": type(b).__name__,
                "thinking": getattr(b, "enable_thinking", None),
                "url": str(getattr(getattr(b, "_client", None), "base_url", "")),
                "scenario": getattr(b, "scenario", None),
                "prompt": PROMPT_VERSION,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        with self.cache_lock:
            cached = self.cache.get(key)
        if cached is not None:
            if getattr(b, "trace_hook", None):
                b.trace_hook(
                    {
                        "event_type": "judge_cache_hit",
                        "role": "judge",
                        "cache_digest": hashlib.sha256(key.encode()).hexdigest(),
                    }
                )
            return calculate_rubric_judgment(
                rubric,
                [
                    CriterionJudgment(c.criterion_id, *values)
                    for c, values in zip(rubric.criteria, cached)
                ],
            )
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
        result = calculate_rubric_judgment(
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

        with self.cache_lock:
            self.cache[key] = [
                (j.chosen, j.rejected, j.chosen_reason, j.rejected_reason)
                for j in result.criterion_judgments
            ]
        return result


def review_judge_evidence(rubric, judgment, data, *, additional_ids=()):
    """Apply only explicit, evidenced error corrections; leave all other Judge results intact."""
    from copy import deepcopy

    allowed = {c.criterion_id for c in rubric.criteria} | set(additional_ids)
    if not isinstance(data["issues"], list) or not isinstance(
        data["judge_corrections"], list
    ):
        raise ValueError("diagnosis issues/corrections must be arrays")
    for issue in data["issues"]:
        if (
            set(issue) != {"kind", "criterion_ids", "reason"}
            or issue["kind"] not in ("rubric_defect", "judge_error", "uncertain")
            or not isinstance(issue["criterion_ids"], list)
            or len(set(issue["criterion_ids"])) != len(issue["criterion_ids"])
            or set(issue["criterion_ids"]) - allowed
            or not isinstance(issue["reason"], str)
            or not issue["reason"].strip()
        ):
            raise ValueError("invalid diagnosis kind, criterion reference or reason")
    js = deepcopy(judgment.criterion_judgments)
    lookup = {j.criterion_id: j for j in js}
    seen = set()
    error_ids = {
        cid
        for issue in data["issues"]
        if issue["kind"] == "judge_error"
        for cid in issue["criterion_ids"]
    }
    for c in data["judge_corrections"]:
        if (
            set(c) != {"criterion_id", "side", "satisfied", "reason"}
            or c["criterion_id"] not in error_ids
            or c["side"] not in ("chosen", "rejected")
            or type(c["satisfied"]) is not bool
            or not isinstance(c["reason"], str)
            or not c["reason"].strip()
        ):
            raise ValueError(
                "correction requires a specific diagnosed Judge error and evidence"
            )
        key = (c["criterion_id"], c["side"])
        if key in seen:
            raise ValueError("duplicate Judge correction")
        seen.add(key)
        j = lookup[c["criterion_id"]]
        if getattr(j, c["side"]) == c["satisfied"]:
            raise ValueError(
                "only changed Judge errors belong in corrections; do not repeat unchanged judgments"
            )
        setattr(j, c["side"], c["satisfied"])
        setattr(j, c["side"] + "_reason", c["reason"])
    return calculate_rubric_judgment(rubric, js)


class LLMCritic:
    def __init__(self, backend, max_candidates=1, **options):
        self.backend = backend
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("invalid candidate setting")
        self.max_candidates = max_candidates
        self.options = options

    @staticmethod
    def _method(data):
        if set(data) != {
            "trigger",
            "principle",
            "application_guide",
            "exclude_when",
            "misuse",
        }:
            raise ValueError("complete five-field method required")
        return Method.from_dict(data)

    @staticmethod
    def evidence(sample, rubric, judgment):
        return {
            "task": sample.task,
            "chosen": sample.chosen,
            "rejected": sample.rejected,
            "rubric": rubric.to_dict(),
            "judgment": judgment.to_dict(),
        }

    def _repair(
        self, sample, rubric, judgment, number, *, conversation, codec, feedback=None
    ):
        def validate(data):
            if (
                set(data)
                != {"analysis", "issues", "judge_corrections", "changes", "criteria"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
            ):
                raise ValueError(
                    "diagnosis, sparse corrections, changes and criteria required"
                )
            # First attempt diagnoses the original Judge. Later attempts retain that baseline.
            if number != 1 and data["judge_corrections"]:
                raise ValueError(
                    "second modification must not change fixed initial Judge results"
                )
            previous_ids = {
                j["criterion_id"]
                for j in (feedback or {})
                .get("raw_judgment", {})
                .get("criterion_judgments", [])
            }
            baseline = review_judge_evidence(
                rubric, judgment, data, additional_ids=previous_ids
            )
            changes = data["changes"]
            if not isinstance(changes, list) or any(
                set(c) != {"change", "basis"}
                or any(not isinstance(v, str) or not v.strip() for v in c.values())
                for c in changes
            ):
                raise ValueError(
                    "each change requires a concrete change and task basis"
                )
            if not changes:
                if data["criteria"] != []:
                    raise ValueError("without a rubric change, criteria must be []")
                from copy import deepcopy

                result = deepcopy(rubric)
            else:
                rid = "repair_" + uuid4().hex
                if not isinstance(data["criteria"], list) or any(
                    set(c) != {"question", "weight"} for c in data["criteria"]
                ):
                    raise ValueError(
                        "repair criteria require only question and weight; source IDs are program-owned"
                    )
                result = Rubric(
                    rid,
                    [
                        Criterion.from_dict(
                            {
                                "criterion_id": f"{rid}_{i}",
                                **c,
                                "source_experiences": [],
                            }
                        )
                        for i, c in enumerate(data["criteria"])
                    ],
                    data["analysis"],
                )
            self.last_diagnosis = {
                "analysis": data["analysis"],
                "issues": data["issues"],
                "judge_corrections": data["judge_corrections"],
            }
            return result, changes, baseline, data["analysis"]

        payload = (
            {
                "task": sample.task,
                "chosen": sample.chosen,
                "rejected": sample.rejected,
                "initial_rubric": rubric.to_dict(),
                "initial_judgment": judgment.to_dict(),
                "round": number,
            }
            if not conversation
            else {"round": number, "previous_feedback": feedback}
        )
        return request(
            self.backend,
            "critic_repair",
            payload,
            validate,
            history=conversation,
            codec=codec,
            **self.options,
        )

    def start(self, sample, rubric, judgment, *, conversation, codec):
        repaired, changes, baseline, reason = self._repair(
            sample, rubric, judgment, 1, conversation=conversation, codec=codec
        )
        return baseline, reason, repaired, changes

    def repair(
        self, sample, rubric, judgment, number, *, conversation, codec, feedback=None
    ):
        repaired, changes, _, _ = self._repair(
            sample,
            rubric,
            judgment,
            number,
            conversation=conversation,
            codec=codec,
            feedback=feedback,
        )
        return repaired, changes

    def assess(self, sample, rubric, judgment, checker=None):
        def validate(data):
            if (
                set(data) != {"analysis", "issues", "judge_corrections"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
            ):
                raise ValueError("Judge result diagnosis required, not full re-judging")
            final = review_judge_evidence(rubric, judgment, data)
            self.last_diagnosis = data
            return final, data["analysis"]

        return request(
            self.backend,
            "critic_assess",
            self.evidence(sample, rubric, judgment),
            validate,
            **self.options,
        )

    def extract(self, sample, rubric, judgment, changes, evidence):
        def validate(data):
            if (
                set(data) != {"analysis", "candidates"}
                or not isinstance(data["analysis"], str)
                or not data["analysis"].strip()
                or not isinstance(data["candidates"], list)
                or len(data["candidates"]) > 2
            ):
                raise ValueError("extraction requires 0..2 candidates")
            out = []
            for item in data["candidates"]:
                if (
                    set(item) != {"method", "change_indices", "reason"}
                    or not isinstance(item["reason"], str)
                    or not item["reason"].strip()
                ):
                    raise ValueError("extraction requires method/change_indices/reason")
                refs = item["change_indices"]
                if (
                    not isinstance(refs, list)
                    or not refs
                    or any(
                        type(i) is not int or i < 0 or i >= len(changes) for i in refs
                    )
                    or len(set(refs)) != len(refs)
                ):
                    raise ValueError("invalid supported change indices")
                m = self._method(item["method"])
                out.append(
                    (
                        Experience(
                            "exp_" + uuid4().hex,
                            m,
                            sample.sample_id,
                            status="candidate",
                        ),
                        refs,
                        item["reason"],
                    )
                )
            if len({e.method.to_json() for e, _, _ in out}) != len(out):
                raise ValueError("duplicate extracted method")
            return out

        return request(
            self.backend,
            "critic_extract",
            {
                "task": sample.task,
                "chosen": sample.chosen,
                "rejected": sample.rejected,
                "repair_evidence": {
                    **evidence,
                    "rubric": rubric.to_dict(),
                    "final_judgment": judgment.to_dict(),
                    "changes": changes,
                },
            },
            validate,
            **self.options,
        )

    def rewrite(self, candidate, review, context):
        def validate(data):
            if (
                set(data) != {"method", "reason"}
                or not isinstance(data["reason"], str)
                or not data["reason"].strip()
            ):
                raise ValueError("rewrite method/reason required")
            return self._method(data["method"])

        return request(
            self.backend,
            "critic_rewrite",
            {
                "candidate": experience_payload(candidate),
                "review": review.to_dict(),
                "repair_evidence": context,
            },
            validate,
            **self.options,
        )

    def rewrite_many(self, candidates, reviews, context):
        lookup = {e.experience_id: e for e in candidates}

        def validate(data):
            if (
                set(data) != {"rewrites"}
                or not isinstance(data["rewrites"], list)
                or len(data["rewrites"]) != len(lookup)
            ):
                raise ValueError("rewrite must cover every rejected experience")
            out = {}
            for x in data["rewrites"]:
                if (
                    set(x) != {"experience_id", "method", "reason"}
                    or x["experience_id"] not in lookup
                    or x["experience_id"] in out
                    or not isinstance(x["reason"], str)
                    or not x["reason"].strip()
                ):
                    raise ValueError("invalid rewrite identity/reason")
                out[x["experience_id"]] = self._method(x["method"])
            return out

        return request(
            self.backend,
            "critic_rewrite_batch",
            {
                "repair_evidence": context,
                "candidates": [
                    {
                        "experience": experience_payload(e),
                        "review": reviews[e.experience_id].to_dict(),
                    }
                    for e in candidates
                ],
            },
            validate,
            **self.options,
        )

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


class LLMExperienceChecker:
    def __init__(self, backend, **options):
        self.backend = backend
        self.options = options

    def check_many(self, sample, evidence, candidates):
        lookup = {e.experience_id: e for e in candidates}
        self.last_errors = {}
        if not lookup or len(lookup) > 2 or len(lookup) != len(candidates):
            raise ValueError("review needs 1..2 distinct candidates")

        def validate(data):
            if set(data) != {"reviews"} or not isinstance(data["reviews"], list):
                raise ValueError("reviews must cover supplied candidates")
            rows = data["reviews"]
            if any(not isinstance(x, dict) or "experience_id" not in x for x in rows):
                raise ValueError("invalid review ID coverage")
            ids = [x["experience_id"] for x in rows]
            if (
                len(ids) != len(lookup)
                or len(set(ids)) != len(ids)
                or set(ids) != set(lookup)
            ):
                raise ValueError("invalid review ID coverage")
            out = {}
            errors = {}
            for x in rows:
                eid = x["experience_id"]
                try:
                    if set(x) != {"experience_id", "approved", "reason"}:
                        raise ValueError(
                            "review fields must be experience_id/approved/reason"
                        )
                    out[eid] = ExperienceReview.from_dict(
                        {k: x[k] for k in ("approved", "reason")}
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    out[eid] = None
                    errors[eid] = str(exc)
                    hook = getattr(self.backend, "trace_hook", None)
                    if hook:
                        hook(
                            {
                                "event_type": "batch_item_validation",
                                "role": "experience_checker_batch",
                                "experience_id": eid,
                                "validation_status": "invalid",
                                "error": str(exc),
                                "call_id": ROLE_CALL_ID.get(),
                            }
                        )
            self.last_errors = errors
            return out

        return request(
            self.backend,
            "experience_checker_batch",
            {
                "task": sample.task,
                "chosen": sample.chosen,
                "rejected": sample.rejected,
                "repair_evidence": evidence,
                "candidates": [experience_payload(e) for e in candidates],
            },
            validate,
            **self.options,
        )

    def check(self, sample, rubric, judgment, candidate, *, repair_evidence=None):
        answers = {j.criterion_id: j for j in judgment.criterion_judgments}
        return request(
            self.backend,
            "experience_checker",
            {
                "task": sample.task,
                "chosen": sample.chosen,
                "rejected": sample.rejected,
                "criteria": [
                    {**c.to_dict(), **answers[c.criterion_id].to_dict()}
                    for c in rubric.criteria
                ],
                "candidate": experience_payload(candidate),
                "repair_evidence": repair_evidence,
            },
            ExperienceReview.from_dict,
            **self.options,
        )


class LLMExperienceUpdater:
    def __init__(self, backend, **options):
        self.backend = backend
        self.options = options

    def update(self, experience, cases, reasons, *, case_kinds=None):
        def validate(data):
            update = ExperienceUpdate.from_dict(data)
            if update.action == "revise":
                LLMCritic._method(data["method"])
                actual = {
                    k
                    for k in experience.method.to_dict()
                    if experience.method.to_dict()[k] != update.method.to_dict()[k]
                }
                if actual != set(update.target_fields):
                    raise ValueError(
                        "target_fields must exactly match actual method changes"
                    )
            return update

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
            validate,
            **self.options,
        )


class LLMCheckers:
    def __init__(self, backends, **options):
        self.backends = backends
        self.options = options

    def check_judgment(self, sample, rubric, judgment):
        app = type("JudgeEvidence", (), {})()
        app.rubric, app.judgment, app.sample = rubric, judgment, sample
        return self._judge_only(app)

    def _judge_only(self, app):
        swap = int(hashlib.sha256(app.sample.sample_id.encode()).hexdigest(), 16) % 2
        responses = [app.sample.chosen, app.sample.rejected]
        if swap:
            responses.reverse()
        answers = {j.criterion_id: j for j in app.judgment.criterion_judgments}
        criteria = []
        for c in app.rubric.criteria:
            j = answers[c.criterion_id]
            sides = [(j.chosen, j.chosen_reason), (j.rejected, j.rejected_reason)]
            if swap:
                sides.reverse()
            criteria.append(
                {
                    "criterion_id": c.criterion_id,
                    "question": c.question,
                    "a": {"satisfied": sides[0][0], "reason": sides[0][1]},
                    "b": {"satisfied": sides[1][0], "reason": sides[1][1]},
                }
            )

        def validate(data):
            if set(data) != {"status", "evidence", "details"} or data["status"] not in (
                "correct",
                "error",
                "unknown",
            ):
                raise ValueError("invalid judge checker report")
            r = CheckerReport.from_dict({"stage": "judge", **data})
            if {d.criterion_id for d in r.details} - {
                c.criterion_id for c in app.rubric.criteria
            }:
                raise ValueError("unknown checker criterion")
            return r

        try:
            return request(
                self.backends["judge"],
                "judge_checker",
                {
                    "task": app.sample.task,
                    "response_a": responses[0],
                    "response_b": responses[1],
                    "criteria": criteria,
                },
                validate,
                **self.options,
            )
        except (LLMFormatError, LLMSampleError) as exc:
            return CheckerReport("judge", "failed", str(exc))

    def check(self, app):
        related = [
            c for c in app.rubric.criteria if app.version_id in c.source_experiences
        ]
        if not related:
            return []
        method = {"experience_id": app.version_id, **app.method.to_dict()}
        basic = {
            "task": app.sample.task,
            "experience": method,
            "selection_reason": app.selection_reason,
            "usage_reason": app.usage_reason,
        }
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
    experience_checker: object


def make_llm_modules(
    backend,
    *,
    retriever=None,
    max_selected=5,
    max_candidates=1,
    retries=2,
    input_bytes=100000,
    role_backends=None,
    selector_include_quality=True,
):
    role_backends = role_backends or {}
    get = lambda r: role_backends.get(r, backend)
    opts = {"retries": retries, "input_bytes": input_bytes}
    return LLMModules(
        backend,
        retriever or FakeRetriever(),
        LLMSelector(
            get("selector"),
            max_selected,
            include_quality=selector_include_quality,
            **opts,
        ),
        LLMInitialRubricGenerator(get("initial_generator"), **opts),
        LLMJudge(get("judge"), **opts),
        LLMCritic(get("critic"), max_candidates, **opts),
        LLMExperienceUpdater(get("experience_updater"), **opts),
        LLMCheckers(
            {s: get(s + "_checker") for s in ("selector", "generator", "judge")}, **opts
        ),
        LLMExperienceChecker(get("experience_checker"), **opts),
    )


def make_fake_modules(scenario="attempt_1_success", **options):
    return make_llm_modules(FakeLLMBackend(scenario), **options)
