"""Deterministic feedback and exact Q/C Bayesian-network inference (no LLM calls)."""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from math import exp, isclose, isfinite, log, log1p
from pathlib import Path
import json
from typing import Any
from verge.models import (
    Model,
    Rubric,
    RubricJudgment,
    CriterionJudgment,
    ApplicationRecord,
    CheckerReport,
    PreferenceSample,
    Experience,
)


def _weight_ratios(rubric):
    weights = {c.criterion_id: Fraction(str(c.weight)) for c in rubric.criteria}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("positive weight budget required")
    return {k: v / total for k, v in weights.items()}


def normalize_weights(rubric):
    return {k: float(v) for k, v in _weight_ratios(rubric).items()}


def calculate_rubric_judgment(rubric, criterion_judgments):
    weights = _weight_ratios(rubric)
    answers = {j.criterion_id: j for j in criterion_judgments}
    if len(answers) != len(criterion_judgments) or answers.keys() != weights.keys():
        raise ValueError("judgments must cover exactly the rubric criteria")
    a = sum(w * answers[k].chosen for k, w in weights.items())
    b = sum(w * answers[k].rejected for k, w in weights.items())
    return RubricJudgment(
        rubric.rubric_id,
        {k: float(v) for k, v in weights.items()},
        list(criterion_judgments),
        float(a),
        float(b),
        float(a - b),
        a > b,
    )


def validate_rubric_judgment(rubric, judgment):
    expected = calculate_rubric_judgment(rubric, judgment.criterion_judgments)
    if (
        expected.rubric_id != judgment.rubric_id
        or expected.normalized_weights != judgment.normalized_weights
        or expected.success != judgment.success
    ):
        raise ValueError("judgment does not match its rubric")
    for name in ("chosen_score", "rejected_score", "margin"):
        if not isclose(
            getattr(expected, name), getattr(judgment, name), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("judgment scores do not match Boolean evidence")


def application(sample, experience, rubric, judgment, phase="downstream"):
    validate_rubric_judgment(rubric, judgment)
    weights = _weight_ratios(rubric)
    answers = {j.criterion_id: j for j in judgment.criterion_judgments}
    pos = Fraction(0)
    neg = Fraction(0)
    counts = dict(positive=0, reverse=0, both_true=0, both_false=0)
    used = False
    for c in rubric.criteria:
        if experience.experience_id not in c.source_experiences:
            continue
        j = answers[c.criterion_id]
        direction = (
            "positive"
            if j.chosen and not j.rejected
            else (
                "reverse"
                if j.rejected and not j.chosen
                else "both_true" if j.chosen else "both_false"
            )
        )
        counts[direction] += 1
        used |= weights[c.criterion_id] > 0
        if direction == "positive":
            pos += weights[c.criterion_id]
        if direction == "reverse":
            neg += weights[c.criterion_id]
    f = (
        "unused"
        if not used
        else ("positive" if pos > neg else "reverse" if neg > pos else "tie")
    )
    record = ApplicationRecord(
        sample,
        experience.experience_id,
        experience.method,
        phase,
        rubric,
        judgment,
        f,
        float(pos),
        float(neg),
        counts,
        recalled=sum(counts.values()) > 0,
        selected=sum(counts.values()) > 0,
        cited=sum(counts.values()) > 0,
    )
    record.cited = sum(counts.values()) > 0
    record.selected = record.recalled = record.cited
    record.no_evidence_reason = (
        ("zero_weight_only" if record.cited else "not_cited") if f == "unused" else None
    )
    return record


def aggregate_reports(reports):
    stages = {r.stage: r.status for r in reports}
    if "error" in stages.values():
        return "error"
    if set(stages) != {"selector", "generator", "judge"} or "failed" in stages.values():
        return "unavailable"
    return "correct" if all(v == "correct" for v in stages.values()) else "unknown"


class ProbabilityError(ValueError):
    """Global parameter/evidence inconsistency; must fail fast, never a negative label."""


@dataclass
class ProbabilityModel(Model):
    version: str
    q_prior: float
    execution_prior: float
    feedback: dict[str, dict[str, float]]
    observation: dict[str, dict[str, float]]
    metadata: dict[str, Any]
    status: str = "experimental"

    def __post_init__(self):
        if not self.version or self.status not in ("experimental", "usable"):
            raise ProbabilityError("invalid parameter identity/status")
        for v in (self.q_prior, self.execution_prior):
            if type(v) not in (int, float) or not isfinite(v) or not 0 < v < 1:
                raise ProbabilityError("priors must be strictly between 0 and 1")
        if set(self.feedback) != {"0:0", "0:1", "1:0", "1:1"} or set(
            self.observation
        ) != {"0", "1"}:
            raise ProbabilityError("complete Q/C tables required")
        for table, states in (
            (self.feedback, {"positive", "reverse", "tie"}),
            (self.observation, {"correct", "error", "unknown"}),
        ):
            for row in table.values():
                if (
                    set(row) != states
                    or any(
                        type(x) not in (int, float)
                        or not isfinite(x)
                        or not 0 <= x <= 1
                        for x in row.values()
                    )
                    or not isclose(sum(row.values()), 1, rel_tol=0, abs_tol=1e-9)
                ):
                    raise ProbabilityError("probability rows must sum to one")
        if self.metadata.get("scope") != "available_reports" or not self.metadata.get(
            "estimation_source"
        ):
            raise ProbabilityError(
                "record a consistent available-report estimation scope and source"
            )

        if self.metadata.get("feedback_protocol") != "net_direction_tie_v1":
            raise ProbabilityError(
                "parameters require feedback_protocol=net_direction_tie_v1 (tie includes balanced directions)"
            )

        if self.status == "usable":
            for q in (0, 1):
                for f in ("positive", "reverse", "tie"):
                    for o in ("correct", "error"):
                        if self.likelihood(q, f, o) == float("-inf"):
                            raise ProbabilityError(
                                "usable tables require positive marginal likelihoods; estimate/smooth before activation"
                            )

    @property
    def fingerprint(self):
        import hashlib, json

        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    @classmethod
    def load(cls, path):
        return cls.from_json(Path(path).read_text())

    @staticmethod
    def _logsum(values):
        values = list(values)
        if not values:
            return float("-inf")
        m = max(values)
        return m + log(sum(exp(v - m) for v in values))

    def likelihood(self, q, f, o):
        terms = []
        for c in (0, 1):
            pc = self.execution_prior if c else 1 - self.execution_prior
            pf = self.feedback[f"{q}:{c}"][f]
            po = self.observation[str(c)][o]
            if pf and po:
                terms.append(log(pc) + log(pf) + log(po))
        return self._logsum(terms)

    def infer(self, prior, f, o):
        if f not in ("positive", "reverse", "tie") or o not in ("correct", "error"):
            raise ProbabilityError("inference requires a valid F and explicit O")
        logs = {str(q): self.likelihood(q, f, o) for q in (0, 1)}
        a = log(prior) + logs["1"] if prior > 0 else float("-inf")
        b = log1p(-prior) + logs["0"] if prior < 1 else float("-inf")
        denom = self._logsum(v for v in (a, b) if v != float("-inf"))
        if denom == float("-inf"):
            raise ProbabilityError(
                "observed evidence has zero probability under parameters"
            )
        return exp(a - denom), logs


def update_quality(experience, record, parameters):
    record.q_before = record.q_after = experience.q_posterior
    record.observation = aggregate_reports(record.reports)
    if record.phase == "admission":
        record.update_reason = "admission_prior_only"
        return
    if record.feedback not in ("positive", "reverse", "tie"):
        record.update_reason = record.feedback
        return
    if record.observation not in ("correct", "error"):
        record.update_reason = "observation_" + record.observation
        return
    if parameters is None or parameters.status != "usable":
        record.update_reason = "parameters_unavailable"
        return
    if experience.parameter_digest not in (None, parameters.fingerprint):
        raise ProbabilityError(
            "parameter change requires rebuilding the posterior from the ledger"
        )
    prior = (
        experience.q_posterior
        if experience.q_posterior is not None
        else parameters.q_prior
    )
    record.q_before = prior
    # Retain finite log odds so rounded display probabilities 0/1 do not become absorbing.
    logs = {
        str(q): parameters.likelihood(q, record.feedback, record.observation)
        for q in (0, 1)
    }
    if any(v == float("-inf") for v in logs.values()):
        raise ProbabilityError(
            "zero likelihood for a quality state; smooth/validate estimated tables before use"
        )
    before = (
        experience.q_log_odds
        if experience.q_log_odds is not None
        else log(prior) - log1p(-prior)
    )
    after = before + logs["1"] - logs["0"]
    posterior = 1 / (1 + exp(-after)) if after >= 0 else exp(after) / (1 + exp(after))
    record.q_log_odds_before = before
    record.q_log_odds_after = after
    experience.q_log_odds = after
    # Save finite, scaled log likelihoods: exp(logL-maxlogL) preserves the ratio.
    finite = [v for v in logs.values() if v != float("-inf")]
    scale = max(finite)
    record.likelihoods = {q: exp(v - scale) for q, v in logs.items()}
    record.parameter_digest = parameters.fingerprint
    experience.parameter_digest = parameters.fingerprint
    record.parameter_version = parameters.version
    record.q_after = posterior
    record.update_reason = "updated"
    experience.q_prior = parameters.q_prior
    experience.q_posterior = posterior
    experience.parameter_version = parameters.version


def rebuild_quality(experience, parameters):
    usable = parameters is not None and parameters.status == "usable"
    experience.q_prior = experience.q_posterior = parameters.q_prior if usable else None
    experience.parameter_version = parameters.version if usable else None
    experience.parameter_digest = parameters.fingerprint if parameters else None
    experience.q_log_odds = None
    seen = {experience.source_task_id} if experience.source_task_id else set()
    seen.update(
        a.sample.task_id
        for a in experience.applications
        if a.phase == "admission" or a.sample.sample_id == experience.source_sample_id
    )
    for a in experience.applications:
        skipped = a.update_reason in ("duplicate", "duplicate_task")
        for name in (
            "q_before",
            "q_after",
            "q_log_odds_before",
            "q_log_odds_after",
            "parameter_version",
            "parameter_digest",
        ):
            setattr(a, name, None)
        a.likelihoods = {}
        a.observation = aggregate_reports(a.reports)
        if a.phase == "admission":
            a.update_reason = "admission_prior_only"
            continue
        if (
            skipped
            or a.sample.task_id in seen
            or a.sample.sample_id == experience.source_sample_id
        ):
            a.q_before = a.q_after = experience.q_posterior
            a.update_reason = "duplicate_task"
            continue
        seen.add(a.sample.task_id)
        update_quality(experience, a, parameters)
    return experience


def replay_score(a):
    if a is None:
        return None
    return {"positive": 1, "reverse": -1}.get(a.feedback, 0)
