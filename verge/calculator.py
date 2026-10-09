"""不依赖 LLM、存储或外部状态的 VERGE 确定性计算。"""

from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite

from verge.models import (
    CriterionJudgment,
    Credit,
    EvidenceRecord,
    Experience,
    Rubric,
    RubricJudgment,
)


_TOL = 1e-9


def _unit(value: float, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


def normalize_weights(rubric: Rubric) -> dict[str, float]:
    """仅在当前完整 rubric R 内归一化：ω_i(R) = s_i / Σ_{j∈R} s_j。"""
    if not rubric.criteria:
        raise ValueError("cannot normalize an empty rubric")
    total = sum(criterion.importance_score for criterion in rubric.criteria)
    if total <= 0:
        raise ValueError("importance score total must be positive")
    return {criterion.criterion_id: criterion.importance_score / total
            for criterion in rubric.criteria}


def calculate_rubric_judgment(
    rubric: Rubric, criterion_judgments: list[CriterionJudgment],
) -> RubricJudgment:
    """S_R(y) = Σ_i ω_i J(y,c_i)；M(R) = S_R(y+) - S_R(y-)。

    d_i = J(y+,c_i) - J(y-,c_i) ∈ {-1,0,1}；M(R)>0 才成功，tie 算失败。
    """
    weights = normalize_weights(rubric)
    answers = {item.criterion_id: item for item in criterion_judgments}
    if len(answers) != len(criterion_judgments) or answers.keys() != weights.keys():
        raise ValueError("criterion judgments must match the rubric exactly")
    chosen = sum(weight * answers[key].chosen for key, weight in weights.items())
    rejected = sum(weight * answers[key].rejected for key, weight in weights.items())
    # 用整数 score 的差值判定符号，避免 separately rounded scores 把精确 tie 判为成功。
    total = sum(item.importance_score for item in rubric.criteria)
    numerator = sum(item.importance_score * (answers[item.criterion_id].chosen
                                             - answers[item.criterion_id].rejected)
                    for item in rubric.criteria)
    margin = numerator / total
    return RubricJudgment(rubric.rubric_id, weights, list(criterion_judgments),
                          chosen, rejected, margin, numerator > 0)


def _contributions(rubric: Rubric, judgment: RubricJudgment) -> dict[str, float]:
    """校验 judgment，再计算单条标准的 q_i(R) = ω_i(R) d_i。"""
    weights = normalize_weights(rubric)
    answers = {item.criterion_id: item for item in judgment.criterion_judgments}
    numerator = sum(item.importance_score * (answers[item.criterion_id].chosen
                                             - answers[item.criterion_id].rejected)
                    for item in rubric.criteria) if answers.keys() == weights.keys() else None
    if (judgment.rubric_id != rubric.rubric_id
            or len(answers) != len(judgment.criterion_judgments)
            or answers.keys() != weights.keys()
            or judgment.normalized_weights.keys() != weights.keys()
            or any(not isclose(judgment.normalized_weights[key], weight, abs_tol=_TOL)
                   for key, weight in weights.items())
            or not isclose(judgment.chosen_score,
                           sum(weight * answers[key].chosen for key, weight in weights.items()),
                           abs_tol=_TOL)
            or not isclose(judgment.rejected_score,
                           sum(weight * answers[key].rejected for key, weight in weights.items()),
                           abs_tol=_TOL)
            or not isclose(judgment.margin, judgment.chosen_score - judgment.rejected_score,
                           abs_tol=_TOL)
            or judgment.success != (numerator > 0)
            or (numerator == 0 and judgment.margin != 0)
            or not isclose(judgment.margin,
                           numerator / sum(item.importance_score for item in rubric.criteria),
                           abs_tol=_TOL)):
        raise ValueError("judgment does not match the rubric")
    return {key: weight * (answers[key].chosen - answers[key].rejected)
            for key, weight in weights.items()}


def calculate_experience_credits(
    rubric: Rubric, judgment: RubricJudgment
) -> dict[str, Credit]:
    """P/N separately; multiple declared sources split each criterion's weight."""
    contributions = _contributions(rubric, judgment)
    totals: dict[str, list[float]] = {}
    for criterion in rubric.criteria:
        if not criterion.source_experiences:
            continue
        share = contributions[criterion.criterion_id] / len(
            criterion.source_experiences
        )
        for source in criterion.source_experiences:
            pair = totals.setdefault(source, [0.0, 0.0])
            pair[0] += max(share, 0)
            pair[1] += max(-share, 0)
    return {key: Credit(*pair) for key, pair in totals.items() if sum(pair) > 0}


def apply_beta_evidence(experience: Experience, evidence: EvidenceRecord) -> Experience:
    return replace(
        experience,
        alpha=experience.alpha + evidence.positive_credit,
        beta=experience.beta + evidence.negative_credit,
        evidence_history=[*experience.evidence_history, evidence],
    )


def validate_rubric_judgment(rubric: Rubric, judgment: RubricJudgment) -> None:
    """Verify a role result without allocating or persisting credit."""
    _contributions(rubric, judgment)
