"""不依赖 LLM、存储或外部状态的 VERGE V1 确定性计算。"""

from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite

from verge.models import (
    CandidateExperience,
    Criterion,
    CriterionJudgment,
    EvidenceRecord,
    Experience,
    RevisionOperation,
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


def calculate_existing_experience_credit(
    initial_rubric: Rubric, initial_judgment: RubricJudgment,
) -> dict[str, float]:
    """只用 R0：U_t(e) = Σ_{i:e∈S_i} [ω_i(R0) d_i / |S_i|]。

    S_i 是该 criterion 的 source_experiences；S_i 为空或 U_t(e)=0 不更新。
    同一 criterion 的多个来源均分，因此 Σ_e U_t(e) 等于有来源标准的贡献和。
    """
    contributions = _contributions(initial_rubric, initial_judgment)
    credits: dict[str, float] = {}
    for criterion in initial_rubric.criteria:
        sources = criterion.source_experiences
        if len(sources) != len(set(sources)):
            raise ValueError("duplicate source experience on a criterion")
        if not sources:
            continue
        share = contributions[criterion.criterion_id] / len(sources)
        for experience_id in sources:
            credits[experience_id] = credits.get(experience_id, 0.0) + share
    result = {key: value for key, value in credits.items() if not isclose(value, 0, abs_tol=_TOL)}
    assert all(-1 - _TOL <= credit <= 1 + _TOL for credit in result.values())
    return result


def _same_revision_criterion(left: Criterion, right: Criterion) -> bool:
    """修订时只比较标准 ID、问题和重要性分数，不比较来源经验。"""
    return (left.criterion_id == right.criterion_id
            and left.question == right.question
            and left.importance_score == right.importance_score)


def calculate_operation_contributions(
    initial_rubric: Rubric,
    initial_judgment: RubricJudgment,
    revised_rubric: Rubric,
    revised_judgment: RubricJudgment,
    operations: list[RevisionOperation],
) -> dict[str, float]:
    """令 q_i(R) = ω_i(R) d_i，计算 operation 的直接收益 δ_o：

    ADD: q_i(Rk)；DELETE: -q_i(R0)；MODIFY: q_i(Rk) - q_i(R0)。
    未修改标准因重新归一化而产生的间接差异不归因给任何 operation；
    source_experiences 不构成修订，Rk 的来源字段不参与任何操作校验或收益计算。
    因此 Σ_o δ_o 不一定等于 M(Rk) - M(R0)。
    """
    before = _contributions(initial_rubric, initial_judgment)
    after = _contributions(revised_rubric, revised_judgment)
    if revised_rubric.base_rubric_id != initial_rubric.rubric_id:
        raise ValueError("revised rubric must use the initial rubric as base")
    old_criteria = {item.criterion_id: item for item in initial_rubric.criteria}
    new_criteria = {item.criterion_id: item for item in revised_rubric.criteria}
    result: dict[str, float] = {}
    affected: set[str] = set()
    for op in operations:
        if op.operation_id in result:
            raise ValueError("duplicate operation ID")
        key = (op.new_or_updated_criterion.criterion_id if op.operation_type == "ADD"
               else op.target_criterion_id)
        if key in affected:
            raise ValueError("conflicting operations on a criterion")
        affected.add(key)
        if op.operation_type == "ADD":
            if (key in old_criteria or key not in new_criteria
                    or op.new_or_updated_criterion is None
                    or not _same_revision_criterion(new_criteria[key], op.new_or_updated_criterion)):
                raise ValueError("ADD must introduce its criterion in the revised rubric")
            result[op.operation_id] = after[key]
        elif op.operation_type == "DELETE":
            if (key not in old_criteria or op.before_criterion is None
                    or not _same_revision_criterion(old_criteria[key], op.before_criterion)
                    or key in new_criteria):
                raise ValueError("DELETE must remove its recorded before_criterion")
            result[op.operation_id] = -before[key]
        else:
            if (key not in old_criteria or key not in new_criteria
                    or op.before_criterion is None or op.new_or_updated_criterion is None
                    or not _same_revision_criterion(old_criteria[key], op.before_criterion)
                    or not _same_revision_criterion(new_criteria[key], op.new_or_updated_criterion)):
                raise ValueError("MODIFY must record its before/after criteria")
            if _same_revision_criterion(op.before_criterion, op.new_or_updated_criterion):
                raise ValueError("no-op MODIFY does not change a criterion")
            result[op.operation_id] = after[key] - before[key]
    if (set(old_criteria) ^ set(new_criteria)) - affected:
        raise ValueError("unrecorded criterion addition or deletion")
    unchanged = (old_criteria.keys() & new_criteria.keys()) - affected
    if any(not _same_revision_criterion(old_criteria[key], new_criteria[key])
           for key in unchanged):
        raise ValueError("unrecorded criterion modification")
    before_answers = {item.criterion_id: item for item in initial_judgment.criterion_judgments}
    after_answers = {item.criterion_id: item for item in revised_judgment.criterion_judgments}
    same_question = {key for key in old_criteria.keys() & new_criteria.keys()
                     if old_criteria[key].question == new_criteria[key].question}
    if any(before_answers[key] != after_answers[key] for key in same_question):
        raise ValueError("criteria with unchanged questions must reuse Boolean judgments")
    return result


def calculate_revision_budget(
    initial_rubric: Rubric,
    initial_judgment: RubricJudgment,
    revised_rubric: Rubric,
    revised_judgment: RubricJudgment,
    evidence_weight: float = 1.0,
) -> float:
    """仅初始失败、修订成功时：ΔM = M(Rk)-M(R0)>0；B_t = w_t min(1,ΔM)。

    默认 evidence_weight=w_t=1；有 noisy-gold weight 时仅在此处乘一次。
    """
    _unit(evidence_weight, "evidence_weight")
    _contributions(initial_rubric, initial_judgment)
    _contributions(revised_rubric, revised_judgment)
    if revised_rubric.base_rubric_id != initial_rubric.rubric_id:
        raise ValueError("revised rubric must use the initial rubric as base")
    if initial_judgment.success or not revised_judgment.success:
        raise ValueError("revision budget requires initial failure and revision success")
    improvement = revised_judgment.margin - initial_judgment.margin
    if not isfinite(improvement) or improvement <= 0:
        raise ValueError("revision margin improvement must be positive")
    return evidence_weight * min(1.0, improvement)


def allocate_revision_credit(
    candidates: list[CandidateExperience],
    operation_contributions: dict[str, float],
    budget: float,
) -> list[CandidateExperience]:
    """S_o 为绑定 operation o 的候选集，r_t(e)=Σ_{o:e∈S_o} δ_o/|S_o|。

    Z_t=Σ_o|δ_o|；候选暂定信用 I_t(e)=B_t r_t(e)/Z_t，且 Σ_e|I_t(e)|≤B_t。
    无绑定或 Z_t=0 的候选保持 credit=None（CREDIT_FAIL）；零收益但 Z_t>0
    的已绑定候选得到数值 0。Admission FAIL 后不重分配，输入对象不修改。
    """
    _unit(budget, "budget")
    if any(type(value) not in (int, float) or not isfinite(value)
           for value in operation_contributions.values()):
        raise ValueError("operation contributions must be finite")
    ids = [item.candidate_id for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ID")
    if any(item.credit is not None or item.admission is not None or item.fusion is not None
           for item in candidates):
        raise ValueError("candidates must not already have credit or decisions")
    if any(op_id not in operation_contributions
           for item in candidates for op_id in item.linked_operation_ids):
        raise ValueError("binding references an unknown operation")
    counts: dict[str, int] = {}
    for item in candidates:
        for op_id in item.linked_operation_ids:
            counts[op_id] = counts.get(op_id, 0) + 1
    z = sum(abs(value) for value in operation_contributions.values())
    if z == 0:
        return list(candidates)
    result = [replace(item, credit=budget * sum(
        operation_contributions[op_id] / counts[op_id] for op_id in item.linked_operation_ids
    ) / z) if item.linked_operation_ids else item for item in candidates]
    assert sum(abs(item.credit or 0) for item in result) <= budget + _TOL
    return result


def apply_beta_evidence(experience: Experience, evidence: EvidenceRecord) -> Experience:
    """正信用：α' = α + w_t credit；负信用：β' = β + w_t |credit|。

    零信用或零权重不更新；confidence = α/(α+β) 由 Experience 派生。
    Revision credit 已在预算中乘 noisy-gold weight，写入时须令此处 w_t=1，
    避免重复加权。返回新 Experience，输入对象不修改。
    """
    _unit(evidence.evidence_weight, "evidence_weight")
    if evidence.evidence_type == "revision_credit" and evidence.evidence_weight != 1:
        raise ValueError("revision credit is already weighted in the budget; evidence_weight must be 1")
    if abs(evidence.credit) > 1 + _TOL:
        raise ValueError("credit must be in [-1, 1]")
    if evidence.credit == 0 or evidence.evidence_weight == 0:
        return experience
    change = abs(evidence.credit) * evidence.evidence_weight
    return replace(
        experience,
        alpha=experience.alpha + (change if evidence.credit > 0 else 0),
        beta=experience.beta + (change if evidence.credit < 0 else 0),
        evidence_history=[*experience.evidence_history, evidence],
    )
