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
                                         ('both_true',0,.1),('both_false',0,.1)])
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


def test_neutral_penalty_affects_exact_candidate_admission():
    r=Rubric('r',[Criterion('p','正向？',.2,['e']),Criterion('tie','同判？',.8,['e'])])
    j=calculate_rubric_judgment(r,[CriterionJudgment('p',True,False,'甲','乙'),
                                 CriterionJudgment('tie',True,True,'甲','乙')])
    assert not positive_experience_sources(r,j)  # .2 positive == .8/4 negative


def test_mixed_three_task_window_includes_negative_and_neutral_then_resets(tmp_path):
    r,eid=existing_runner(tmp_path)
    r.modules.retriever.retrieve=lambda task,pool: pool
    r.modules.selector.select=lambda task,views,**kw: views
    active={'direction':'positive'}
    r.modules.initial_generator.generate=lambda task, views: directional_rubric(eid,active['direction'])[0]
    r.modules.judge.judge=lambda rubric,*args,**kw: directional_rubric(eid,active['direction'])[1]
    calls=[]
    r.modules.experience_updater.update=lambda e: (calls.append(e) or keep_update(e,'修正后的方法','较窄条件'))
    for i,d in enumerate(['positive','negative','both_true','both_false','positive','negative'],1):
        active['direction']=d
        rec=r.run_sample(sample(f'next{i}'))
        assert bool(rec.experience_updates)==(i in (3,6))
        assert len(r.memory.all())==1
    e=r.memory.get(eid)
    assert [len(c.cases) for c in calls]==[4,7]
    assert e.pending_feedback_count==0 and e.text_version==2
    assert (e.alpha,e.beta)==pytest.approx((2,2))  # origin .2 + two .4; N=.4+.1+.1+.4
    payload=experience_update_payload(calls[0])
    assert [c['criteria'][0]['feedback'] for c in payload['cases'][1:]]==['positive','negative','neutral']
    assert all(c['chosen']==case.chosen and c['rejected']==case.rejected
               for c,case in zip(payload['cases'],calls[0].cases))
    assert ExperienceMemory(tmp_path).get(eid)==e
    assert not any(hasattr(e,key) for key in ('parent_id','child_ids','inherited_credit','retired'))


def test_failed_window_keeps_pending_feedback_and_final_tenth_forces_attempt(tmp_path):
    r,eid=existing_runner(tmp_path)
    calls=[]
    def update(e):
        calls.append(len(e.cases))
        if len(e.cases)==4: raise ValueError('invalid bounded output')
        return keep_update(e)
    r.modules.experience_updater.update=update
    for i in range(1,11): r.run_sample(sample(str(i)))
    e=r.memory.get(eid)
    assert calls==[4,5,8,10]  # failure at 4 does not reset; final call after only two new cases
    assert len(e.cases)==10 and e.text_frozen
    assert e.update_history[0].error and all(u.error is None for u in e.update_history[1:])
    assert len(e.evidence_history)==11 and e.pending_feedback_count==0


def test_window_requires_new_distinct_tasks_and_prevents_early_direct_write(tmp_path):
    m=ExperienceMemory(tmp_path);eid=seed(m,CandidateExperience('c','方法','*')).experience_id
    for i in range(1,4):
        e=m.get(eid);case=case_for(eid,str(i),e.content,e.trigger,p=0,n=.2)
        m.apply_downstream_credits(str(i),{eid:case.credit})
        assert m.collect_case(eid,case)
        assert not m.collect_case(eid,case)
        assert m.get(eid).pending_feedback_count==i
        if i<3:
            before=m.path.read_bytes()
            with pytest.raises(ValueError,match='pending'):m.apply_update(eid,ExperienceUpdate('不可提前改写','*'))
            assert m.path.read_bytes()==before
    m.apply_update(eid,keep_update(m.get(eid)))
    with pytest.raises(ValueError,match='pending'):m.apply_update(eid,keep_update(m.get(eid)))
    assert m.get(eid).pending_feedback_count==0


def test_updater_protocol_rejects_split_and_keeps_length_prompt_only(monkeypatch,tmp_path):
    r,eid=existing_runner(tmp_path);r.run_sample(sample('next'));e=r.memory.get(eid)
    before=r.memory.path.read_bytes()
    old={'experiences':[{'content':'A','trigger':'*','sample_ids':['1']},
                        {'content':'B','trigger':'*','sample_ids':['2']}]}
    with pytest.raises(ValueError):
        LLMExperienceUpdater(backend(monkeypatch,lambda req:result(old),max_retries=0),invalid_output_retries=0).update(e)
    assert r.memory.path.read_bytes()==before
    calls=[]
    def handler(req):
        body=json.loads(req.content);calls.append(body)
        assert 'sample_ids' not in body['messages'][0]['content']
        return result({'content':'长'*200,'trigger':'条件'*40})
    updated=LLMExperienceUpdater(backend(monkeypatch,handler,max_retries=0)).update(e)
    assert len(updated.content)==200 and len(calls)==1
