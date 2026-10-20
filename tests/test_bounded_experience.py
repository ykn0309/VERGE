"""End-to-end local eligibility, delayed selection, bounded updates and error boundaries."""
from experience_helpers import keep_update
from copy import deepcopy
from dataclasses import replace
import json
import httpx
import pytest
from experience_helpers import seed
from test_pipeline import runner, sample
from test_real_llm import backend, result
from verge.calculator import calculate_rubric_judgment, positive_experience_sources
from verge.llm import LLMFormatError, LLMRequestError, LLMSampleError
from verge.models import CandidateExperience, Criterion, CriterionJudgment, Rubric, TrajectoryRecord
from verge.modules import JudgeOutputError, LLMExperienceUpdater


def scripted_candidate_runner(tmp_path, first='positive', second='positive'):
    r = runner(tmp_path, 'final_failure')
    seen_memory = []
    original_judge = r.modules.judge.judge
    original_propose = r.modules.critic.propose

    def propose(*args, **kwargs):
        if args[5] == 2:
            assert not r.memory.all()  # C1 is only staged, never immediately admitted
            if second == 'empty':
                return replace(original_propose(*args, **kwargs), candidate=None)
            if second == 'critic_error':
                raise LLMFormatError('critic output exhausted')
        return original_propose(*args, **kwargs)

    def apply(task, selected, candidates, number):
        c = candidates[0]
        mode = first if number == 1 else second
        seen_memory.append(deepcopy(r.memory.all()))
        if mode == 'generator_error':
            raise ValueError('invalid application')
        weight = .2 if number == 1 else .3
        return Rubric(f'r{number}', [
            Criterion('local', mode, weight, [] if mode == 'unused' else [c.candidate_id]),
            Criterion('other', 'unrelated negative criterion', 1-weight, []),
        ])

    def judge(rubric, *args, **kwargs):
        if rubric.rubric_id == 'r2' and second == 'judge_error':
            raise JudgeOutputError('judge exhausted')
        if rubric.rubric_id in ('r1', 'r2'):
            mode = rubric.criteria[0].question
            return calculate_rubric_judgment(rubric, [
                CriterionJudgment('local', mode in ('positive', 'unused'), mode=='negative', '甲依据', '乙依据'),
                CriterionJudgment('other', False, True, '甲依据', '乙依据'),
            ])
        return original_judge(rubric, *args, **kwargs)

    r.modules.critic.propose, r.modules.initial_generator.apply, r.modules.judge.judge = propose, apply, judge
    return r, seen_memory


@pytest.mark.parametrize('second,selected', [
    ('positive', 2), ('negative', 1), ('neutral', 1), ('unused', 1),
    ('empty', 1), ('critic_error', 1), ('generator_error', 1), ('judge_error', None),
])
def test_delayed_single_admission_c2_or_c1_fallback(tmp_path, second, selected):
    r, seen = scripted_candidate_runner(tmp_path, second=second)
    record = r.run_sample(sample())
    assert all(not memory for memory in seen)
    assert not any(role=='fusion_judge' for role, _ in r.modules.backend.requests)
    assert not hasattr(r.modules, 'fusion_judge')
    if selected is None:
        assert record.state == 'judge_error_application'
        assert not r.memory.all() and not record.admitted_candidate_ids
    else:
        app = next(a for a in record.application_attempts if a.round_number == selected)
        assert app.eligible and not app.judgment.success
        assert len(r.memory.all()) == 1 and record.admitted_candidate_ids == [app.candidate_id]
        e = r.memory.all()[0]
        assert e.cases[0].rubric == app.rubric and e.cases[0].credit == app.credit
        assert e.evidence_history[0].candidate_id == app.candidate_id
        assert len(e.evidence_history) == 1 and e.alpha == pytest.approx(1+app.credit.positive_credit)
        assert e.content == next(c.content for p in record.proposal_attempts for c in p.atomic_candidates if c.candidate_id==app.candidate_id)
    assert TrajectoryRecord.from_json(record.to_json()) == record


def test_both_candidates_nonpositive_do_not_admit(tmp_path):
    r, _ = scripted_candidate_runner(tmp_path, first='negative', second='neutral')
    rec = r.run_sample(sample())
    assert not rec.admitted_candidate_ids and not r.memory.all()


@pytest.mark.parametrize('weights,expected', [([.1, .2, .3], False), ([.1, .2, .29999999999999993], True)])
def test_exact_local_net_eligibility_not_float_noise(weights, expected):
    r = Rubric('r', [Criterion(str(i), '要求？', w, ['e', 'other']) for i, w in enumerate(weights)])
    j = calculate_rubric_judgment(r, [CriterionJudgment(str(i), i<2, i==2, '甲', '乙') for i in range(3)])
    assert ('e' in positive_experience_sources(r, j)) is expected


def existing_runner(tmp_path, *, overall_success=True):
    r = runner(tmp_path, 'final_failure')
    eid = seed(r.memory, CandidateExperience('seed', '原方法', '*')).experience_id
    r.modules.initial_generator.generate = lambda task, views: Rubric('existing_r0', [
        Criterion('p', '正向设计要求', .3, [eid]),
        Criterion('n', '另一要求', .1, [eid]),
        Criterion('task', '独立任务要求', .6, []),
    ])
    original = r.modules.judge.judge

    def judge(rubric, *args, **kwargs):
        if rubric.rubric_id == 'existing_r0':
            return calculate_rubric_judgment(rubric, [
                CriterionJudgment('p', True, False, '正向甲', '正向乙'),
                CriterionJudgment('n', False, True, '负向甲', '负向乙'),
                CriterionJudgment('task', overall_success, not overall_success, '任务甲', '任务乙'),
            ])
        return original(rubric, *args, **kwargs)
    r.modules.judge.judge = judge
    return r, eid
