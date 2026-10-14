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
from verge.llm import LLMFormatError, LLMRequestError
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

    def apply(task, selected, c, number):
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
        assert not r.memory.all() and record.admitted_candidate_id is None
    else:
        app = next(a for a in record.application_attempts if a.round_number == selected)
        assert app.eligible and not app.judgment.success
        assert len(r.memory.all()) == 1 and record.admitted_candidate_id == app.candidate_id
        e = r.memory.all()[0]
        assert e.cases[0].rubric == app.rubric and e.cases[0].credit == app.credit
        assert e.evidence_history[0].candidate_id == app.candidate_id
        assert len(e.evidence_history) == 1 and e.alpha == pytest.approx(1+app.credit.positive_credit)
        assert e.content == next(p.candidate.content for p in record.proposal_attempts if p.candidate and p.candidate.candidate_id==app.candidate_id)
    assert TrajectoryRecord.from_json(record.to_json()) == record


def test_both_candidates_nonpositive_do_not_admit(tmp_path):
    r, _ = scripted_candidate_runner(tmp_path, first='negative', second='neutral')
    rec = r.run_sample(sample())
    assert not rec.admitted_candidate_id and not r.memory.all()


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


def warm_window(r):
    for sid in ('warm1', 'warm2'):
        rec = r.run_sample(sample(sid))
        assert not rec.experience_updates


@pytest.mark.parametrize('overall_success', [True, False])
def test_positive_r0_updates_after_all_applications_with_fixed_e(tmp_path, overall_success):
    r, eid = existing_runner(tmp_path, overall_success=overall_success)
    warm_window(r)
    calls = []

    def update(e):
        calls.append(e)
        if not overall_success:
            assert len(list(r.modules.backend.requests)) > 1
        return keep_update(e, '跨任务共同方法', '*')

    r.modules.experience_updater.update = update
    rec = r.run_sample(sample('next'))
    assert len(calls)==1 and len(calls[0].cases)==4
    assert calls[0].cases[-1].sample_id=='next'
    assert len(rec.experience_updates)==1
    e = r.memory.get(eid)
    assert e.text_version==2 and e.content=='跨任务共同方法'
    assert (e.alpha,e.beta)==pytest.approx((2.1,1.3))
    assert [ev.text_version for ev in e.evidence_history]==[1,1,1,1]
    for role, payload in r.modules.backend.requests:
        if role=='initial_generator':
            assert payload['selected_experiences'][0]['content']=='原方法'
            assert 'cases' not in payload['selected_experiences'][0]
    assert TrajectoryRecord.from_json(rec.to_json())==rec


@pytest.mark.parametrize('error', [ValueError, LLMFormatError])
def test_updater_failure_preserves_text_credit_case_and_records_error(tmp_path, error):
    r, eid = existing_runner(tmp_path)
    warm_window(r)
    r.modules.experience_updater.update = lambda e: (_ for _ in ()).throw(error('invalid update'))
    rec = r.run_sample(sample('next'))
    e = r.memory.get(eid)
    assert e.content=='原方法' and e.text_version==1 and len(e.cases)==4
    assert (e.alpha,e.beta)==pytest.approx((2.1,1.3))
    assert rec.state=='initial_success' and rec.experience_updates[eid].error=='invalid update'


def test_updater_network_failure_is_fail_fast(tmp_path):
    r, _ = existing_runner(tmp_path)
    warm_window(r)
    r.modules.experience_updater.update = lambda e: (_ for _ in ()).throw(LLMRequestError('network'))
    with pytest.raises(LLMRequestError):
        r.run_sample(sample('next'))
    assert len(list(r.trajectories.all())) == 2


def test_tenth_case_last_call_eleventh_case_only_credit(tmp_path):
    r, eid = existing_runner(tmp_path)
    calls=[]
    r.modules.experience_updater.update=lambda e: (calls.append(len(e.cases)) or keep_update(e))
    for i in range(1,12):
        r.run_sample(sample(f'next{i}'))
    e=r.memory.get(eid)
    assert calls==[4,7,10]
    assert len(e.cases)==10 and e.text_frozen and len(e.evidence_history)==12
    assert (e.alpha,e.beta)==pytest.approx((1.2+11*.3,1+11*.1))


def test_http_updater_reads_all_cases_and_can_keep_original(monkeypatch,tmp_path):
    r,eid=existing_runner(tmp_path)
    r.run_sample(sample('next'))
    e=r.memory.get(eid)
    requests=[]
    def handler(request):
        body=json.loads(request.content)
        requests.append(body)
        return result({'content':'可复用方法','trigger':'明确任务条件'})
    updater=LLMExperienceUpdater(backend(monkeypatch,handler,max_retries=0))
    assert updater.update(e)==keep_update(e,'可复用方法','明确任务条件')
    assert len(requests)==1 and len(requests[0]['messages'])==2
    user=requests[0]['messages'][1]['content']
    assert '任务原文' in user and sample().task in user
    assert '正向甲' in user and '负向乙' in user
    assert 'metadata' not in user and 'overall_preference' not in user


def test_compact_update_cases_filter_unrelated_evidence_and_preserve_full_answers(tmp_path):
    from verge.prompts import experience_update_payload
    r, eid = existing_runner(tmp_path)
    r.run_sample(sample('next'))
    e = r.memory.get(eid)
    original = e.to_dict()
    payload = experience_update_payload(e)
    assert set(payload['current_experience']) == {'content', 'trigger'}
    for case, card in zip(e.cases, payload['cases']):
        assert set(card) == {'sample_id', 'task', 'chosen', 'rejected', 'criteria'}
        assert (card['task'], card['chosen'], card['rejected']) == (case.task, case.chosen, case.rejected)
        related = [c for c in case.rubric.criteria if case.source_id in c.source_experiences]
        assert [c['question'] for c in card['criteria']] == [c.question for c in related]
        for criterion, projected in zip(related, card['criteria']):
            verdict = next(j for j in case.judgment.criterion_judgments if j.criterion_id == criterion.criterion_id)
            assert projected == dict(question=criterion.question, feedback="positive" if verdict.chosen > verdict.rejected else "negative" if verdict.chosen < verdict.rejected else "neutral", chosen=verdict.chosen,
                rejected=verdict.rejected, chosen_reason=verdict.chosen_reason,
                rejected_reason=verdict.rejected_reason)
    assert len(payload['cases'][-1]['criteria']) == 2  # Third criterion is unrelated.
    changed = experience_update_payload(replace(e, content='新版方法'))
    assert all(card['applied_experience'] == {'content': case.content, 'trigger': case.trigger}
               for case, card in zip(e.cases, changed['cases']))
    assert e.to_dict() == original  # Persistent audit evidence stays complete.


def test_context_budget_skips_without_truncating_or_calling_backend(tmp_path):
    r,eid=existing_runner(tmp_path)
    r.modules.experience_updater.update=LLMExperienceUpdater(r.modules.backend).update
    warm_window(r)
    rec=r.run_sample(replace(sample('huge'),task='原文'*60000))
    assert 'context budget' in rec.experience_updates[eid].error
    assert r.memory.get(eid).content=='原方法'
    assert r.memory.get(eid).cases[-1].task=='原文'*60000
    assert not any(role=='experience_updater' for role,_ in r.modules.backend.requests)


def test_full_http_existing_update_after_two_applications(monkeypatch, tmp_path, wire_payload):
    """Production adapters preserve E, record local failure and send all cases to the updater."""
    from verge.memory import ExperienceMemory, TrajectoryLog
    from verge.modules import make_llm_modules
    from verge.pipeline import PipelineRunner
    calls, traces = [], []

    def handler(request):
        body = json.loads(request.content)
        payload = wire_payload(body)
        calls.append(payload)
        if 'max_selected' in payload:
            return result({'selected_experience_ids': ['1']})
        if 'selected_experiences' in payload:
            assert payload['selected_experiences'][0]['content'] == '来源方法'
            if 'candidate_experience' not in payload:
                questions = [('r0positive', .3, ['1']), ('r0negative', .1, ['1']), ('r0task', .6, [])]
            else:
                questions = [('appnegative', .2, ['2']), ('apptask', .8, [])]
            return result({'analysis':'任务设计', 'criteria':[
                {'question':q,'weight':w,'source_experiences':sources} for q,w,sources in questions
            ]})
        if 'current_rubric' in payload:
            return result({'analysis':'诊断事实', 'candidates':[{'content':'当前候选','trigger':'任务条件'}]})
        if 'response' in payload:
            chosen = payload['response'] == 'answer chosen private'
            return result({'criterion_judgments':[
                {'criterion_id':c['criterion_id'], 'satisfied': chosen if c['question']=='r0positive' else not chosen,
                 'reason':'对应原回答依据'} for c in payload['criteria']
            ]})
        assert set(payload)=={'current_experience','cases'}
        assert len(payload['cases'])==4
        assert payload['cases'][0]['sample_id']=='1'
        assert payload['cases'][1]['sample_id']=='2'
        assert all('credit' not in case for case in payload['cases'])
        assert all(set(c) == {'question', 'feedback', 'chosen', 'rejected', 'chosen_reason', 'rejected_reason'}
                   for case in payload['cases'] for c in case['criteria'])
        assert all('metadata' not in case for case in payload['cases'])
        return result({'content':'跨案例方法','trigger':'跨案例条件'})

    llm = backend(monkeypatch, handler, max_retries=0, trace_hook=traces.append)
    m = ExperienceMemory(tmp_path)
    origin_id, next_id = 'helpsteer3:train:' + 'a' * 64, 'helpsteer3:train:' + 'b' * 64
    eid = seed(m,CandidateExperience('origin_c','来源方法','*'),origin_id).experience_id
    r = PipelineRunner(make_llm_modules(llm),m,TrajectoryLog(tmp_path))
    r.run_sample(sample('warm1'))
    r.run_sample(sample('warm2'))
    rec = r.run_sample(sample(next_id))
    assert rec.state=='validation_exhausted' and len(rec.application_attempts)==2
    assert not rec.admitted_candidate_id and len(m.all())==1
    assert m.get(eid).content=='跨案例方法' and m.get(eid).text_version==2
    assert calls[-1]['current_experience']['content']=='来源方法'
    assert traces[-1]['role']=='experience_updater' and len(traces[-1]['messages'])==2
    assert len([t for t in traces if t['role']=='critic'])==6
    assert not any(t['role']=='fusion_judge' for t in traces)
    assert TrajectoryRecord.from_json(rec.to_json())==rec
    internal_ids = {eid, origin_id, next_id}
    internal_ids.update(p.candidate.candidate_id for p in rec.proposal_attempts)
    for rubric in [rec.initial_rubric, *(a.rubric for a in rec.application_attempts)]:
        internal_ids.add(rubric.rubric_id)
        internal_ids.update(c.criterion_id for c in rubric.criteria)
    assert {t['role'] for t in traces} == {'selector', 'initial_generator', 'judge', 'critic', 'experience_updater'}
    assert all(real_id not in json.dumps(t['messages'], ensure_ascii=False)
               for t in traces for real_id in internal_ids)
    assert [c.sample_id for c in m.get(eid).cases] == [origin_id, "warm1", "warm2", next_id]
    assert m.get(eid).cases[-1].credit.positive_credit == pytest.approx(.3)
    assert m.get(eid).cases[-1].credit.negative_credit == pytest.approx(.1)
