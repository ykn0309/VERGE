"""Exact positive-weight normalization and Boolean rubric scoring."""

from fractions import Fraction
from verge.models import CriterionJudgment, Rubric, RubricJudgment


def calculate_rubric_judgment(
    rubric: Rubric, judgments: list[CriterionJudgment]
) -> RubricJudgment:
    lookup = {j.criterion_id: j for j in judgments}
    if len(lookup) != len(judgments) or set(lookup) != {
        c.criterion_id for c in rubric.criteria
    }:
        raise ValueError("judgments must cover every criterion exactly once")
    raw = {c.criterion_id: Fraction(str(c.weight)) for c in rubric.criteria}
    total = sum(raw.values(), Fraction(0))
    if total <= 0:
        raise ValueError("positive rubric weight required")
    weights = {cid: w / total for cid, w in raw.items()}
    chosen = sum((w * lookup[cid].chosen for cid, w in weights.items()), Fraction(0))
    rejected = sum(
        (w * lookup[cid].rejected for cid, w in weights.items()), Fraction(0)
    )
    return RubricJudgment(
        rubric.rubric_id,
        {cid: float(w) for cid, w in weights.items()},
        judgments,
        float(chosen),
        float(rejected),
        float(chosen - rejected),
        chosen > rejected,
    )
