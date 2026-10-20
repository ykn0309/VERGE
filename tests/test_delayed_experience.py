"""Real credit directions, three-task windows, failures, saturation and no splitting."""
from dataclasses import replace
import json
import pytest
from experience_helpers import seed, case_for, keep_update
from test_bounded_experience import existing_runner
from test_pipeline import sample
from test_real_llm import backend, result
from verge.calculator import calculate_experience_credits, calculate_rubric_judgment, positive_experience_sources
from verge.memory import ExperienceMemory
from verge.models import CandidateExperience, Criterion, CriterionJudgment, ExperienceUpdate, Rubric
from verge.modules import LLMExperienceUpdater
from verge.prompts import experience_update_payload


def directional_rubric(eid, direction):
    r = Rubric('direction', [Criterion('method', '方法核验？', .4, [eid]),
                            Criterion('task', '其他要求？', .6, [])])
    pairs = {'positive':(True, False), 'negative':(False, True),
             'both_true':(True, True), 'both_false':(False, False)}
    a,b = pairs[direction]
    j = calculate_rubric_judgment(r, [CriterionJudgment('method',a,b,'原文甲依据','原文乙依据'),
                                     CriterionJudgment('task',True,False,'任务甲','任务乙')])
    return r,j


@pytest.mark.parametrize('direction,p,n', [('positive',.4,0),('negative',0,.4),
                                         ('both_true',0,0),('both_false',0,0)])
def test_credit_matches_each_verdict_and_shared_mass(direction,p,n):
    r,j = directional_rubric('e',direction)
    credit = calculate_experience_credits(r,j)['e']
    assert (credit.positive_credit,credit.negative_credit)==pytest.approx((p,n))
    assert ('e' in positive_experience_sources(r,j)) == (p>n)
    shared = replace(r,criteria=[replace(r.criteria[0],source_experiences=['e','f']),r.criteria[1]])
    js = calculate_rubric_judgment(shared,j.criterion_judgments)
    credits = calculate_experience_credits(shared,js)
    assert sum(c.amount for c in credits.values())==pytest.approx(p+n)
    assert credits['e']==credits['f']


def test_neutral_does_not_penalize_exact_candidate_admission():
    r=Rubric('r',[Criterion('p','正向？',.2,['e']),Criterion('tie','同判？',.8,['e'])])
    j=calculate_rubric_judgment(r,[CriterionJudgment('p',True,False,'甲','乙'),
                                 CriterionJudgment('tie',True,True,'甲','乙')])
    assert positive_experience_sources(r,j) == {'e'}


def test_positive_negative_cases_and_neutral_exclusion(tmp_path):
    r,eid=existing_runner(tmp_path)
    active={'direction':'positive'}
    r.modules.initial_generator.generate=lambda task,views:directional_rubric(eid,active['direction'])[0]
    original=r.modules.judge.judge
    r.modules.judge.judge=lambda rubric,*args,**kw:directional_rubric(eid,active['direction'])[1] if rubric.rubric_id=='direction' else original(rubric,*args,**kw)
    calls=[]
    r.modules.experience_updater.update=lambda e:(calls.append(e) or ExperienceUpdate('keep'))
    for i,d in enumerate(['positive','negative','both_true','both_false','positive','negative','positive']):
        active['direction']=d
        rec=r.run_sample(sample(f'next{i}'))
        assert bool(rec.experience_updates)==(i==6)
    e=r.memory.get(eid)
    assert len(calls)==1 and len(e.downstream_cases)==5
    assert e.pending_feedback_count==0 and len(e.working_cases)==5
    payload=experience_update_payload(calls[0])
    assert len(payload['cases'])==5
    assert {c['criteria'][0]['feedback'] for c in payload['cases']}=={'positive','negative'}
    assert not any(c['sample_id'] in ('next2','next3') for c in payload['cases'])


def test_updater_http_keep_and_revise_length_prompt_only(monkeypatch,tmp_path):
    from test_memory import pool,feedback
    m,eid=pool(tmp_path)
    for i in range(5):feedback(m,eid,f't{i}')
    e=m.get(eid)
    seen=[]
    def handler(req):
        seen.append(json.loads(req.content))
        return result({'action':'revise','draft':{'content':'长'*200,'trigger':'条件'*40}})
    u=LLMExperienceUpdater(backend(monkeypatch,handler,max_retries=0)).update(e)
    assert len(u.draft.content)==200 and len(seen)==1
    assert len(experience_update_payload(e)['cases'])==5
    assert 't0' not in seen[0]['messages'][1]['content'] # numbered sample IDs
