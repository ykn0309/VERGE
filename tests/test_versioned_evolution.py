"""Plan67 multi-candidate, revision, transport and isolation contracts."""
from dataclasses import replace
import json
import pytest
from test_pipeline import runner,sample
from test_memory import pool,feedback
from verge.models import AtomicMethod,AtomicResult,ExperienceUpdate,TrajectoryRecord,CategoryAssignment
from verge.modules import LLMAtomizer,LLMCategoryOrganizer,LLMCategoryDescriber,HybridRetriever,make_fake_modules
from verge.pipeline import PipelineRunner
from verge.memory import TrajectoryLog,ExperienceMemory
from verge.llm import LLMSampleError,LLMRequestError
from test_real_llm import backend,result

@pytest.mark.parametrize('scenario,expected', [('attempt_1_success',1),('attempt_2_success',2),('final_failure',None)])
def test_multiple_candidates_one_generator_per_round_and_shared_credit(tmp_path,scenario,expected):
    r=runner(tmp_path,scenario)
    r.modules.atomizer.atomize=lambda draft,**kw:AtomicResult([AtomicMethod('方法A','条件'),AtomicMethod('方法B','条件')])
    rec=r.run_sample(sample())
    calls=[p for role,p in r.modules.backend.requests if role=='initial_generator']
    assert len(calls)==1+len(rec.proposal_attempts)
    assert all(len(p.atomic_candidates)==2 for p in rec.proposal_attempts)
    if expected:
        assert len(rec.admitted_experience_ids)==2
        assert all(e.alpha==pytest.approx(1.4) for e in r.memory.all())
        assert {a.round_number for a in rec.application_attempts if a.candidate_id in rec.admitted_candidate_ids}=={expected}
        assert len(calls[-1]['candidate_experiences'])==2
    else:assert not rec.admitted_experience_ids
    assert TrajectoryRecord.from_json(rec.to_json())==rec

@pytest.mark.parametrize('error',[ValueError('bad'),LLMSampleError('timeout')])
def test_evolution_atomizer_error_retains_parent_credit_and_next_task_runs(tmp_path,error):
    from test_bounded_experience import existing_runner
    r,eid=existing_runner(tmp_path)
    r.modules.experience_updater.update=lambda e:ExperienceUpdate('revise',AtomicMethod('新方法','*'))
    r.modules.atomizer.atomize=lambda *a,**k:(_ for _ in ()).throw(error)
    for i in range(5):rec=r.run_sample(sample(str(i)))
    assert rec.experience_updates[eid].result=='error'
    e=r.memory.get(eid)
    assert e.status=='active' and e.content=='原方法' and len(e.evidence_history)==6
    assert not e.update_due and e.pending_feedback_count==5
    following=r.run_sample(sample('next'))
    assert following.experience_updates[eid].error
    assert len(r.memory.all())==1

@pytest.mark.parametrize('method',['content','trigger'])
def test_revision_new_id_and_child_window_has_no_origin(tmp_path,method):
    m,eid=pool(tmp_path)
    for i in range(5):feedback(m,eid,str(i))
    old=m.get(eid)
    draft=AtomicMethod('新方法' if method=='content' else old.content,'新条件' if method=='trigger' else old.trigger)
    ev=m.evolve(eid,ExperienceUpdate('revise',draft),AtomicResult([draft]))
    child=m.get(ev.child_ids[0])
    assert child.origin_kind=='evolution' and not child.cases
    for i in range(4):feedback(m,child.experience_id,'c'+str(i),p=0,n=.2)
    assert not m.get(child.experience_id).update_due
    feedback(m,child.experience_id,'c4')
    assert m.get(child.experience_id).update_due

@pytest.mark.parametrize('payload',[{'experiences':[]},{'experiences':[{'content':'x','trigger':'*'}]*2}, {'experiences':[{'content':'x','trigger':'*'}],'analysis':'not allowed'}])
def test_atomizer_invalid_structure_is_bounded(monkeypatch,payload):
    requests=[]
    def h(req):requests.append(req);return result(payload)
    with pytest.raises(ValueError):LLMAtomizer(backend(monkeypatch,h,max_retries=0),invalid_output_retries=0).atomize(AtomicMethod('原方法','*'))
    assert len(requests)==1

@pytest.mark.parametrize('role',['atomizer','category_organizer','category_describer'])
def test_new_role_http_strict_fields_and_short_ids(monkeypatch,role):
    from verge.models import Experience,ExperienceCategory
    seen=[]
    outputs={'atomizer':{'experiences':[{'content':'方法A','trigger':'条件A'},{'content':'方法B','trigger':'条件B'}]},
             'category_organizer':{'action':'assign','category_id':1,'name':None,'description':None},
             'category_describer':{'name':'计算核验','description':'数量关系及单位'}}
    def h(req):seen.append(json.loads(req.content));return result(outputs[role])
    b=backend(monkeypatch,h,max_retries=0)
    if role=='atomizer':out=LLMAtomizer(b).atomize(AtomicMethod('方法A和B','条件'))
    elif role=='category_organizer':
        out=LLMCategoryOrganizer(b).organize(Experience('exp_long_id','方法','*',1,1,[],[]),[{'category_id':'cat_long_id','name':'计算','description':'数量核验','members':[]}])
        assert out.category_id=='cat_long_id'
    else:out=LLMCategoryDescriber(b).describe(ExperienceCategory('cat_long_id','计算','数量核验'),[{'experience_id':'exp_long_id','action':'added','content':'方法','trigger':'条件'}],1)
    assert len(seen)==1
    user=seen[0]['messages'][1]['content']
    assert 'exp_long_id' not in user and 'cat_long_id' not in user

@pytest.mark.parametrize('kind',['pure_english','chinese','empty_terms'])
def test_bm25_zero_no_fake_rank_and_max_pool(tmp_path,kind):
    from test_atomic_categories import setup_pool,Embed
    m,ids=setup_pool(tmp_path,2)
    r=HybridRetriever(Embed())
    query={'pure_english':'dimension','chinese':'核验','empty_terms':'!!!'}[kind]
    r.retrieve(query,m.all(),categories=m.categories())
    if kind=='empty_terms':assert all(x['bm25_rank'] is None for x in r.last_trace)
    if kind=='chinese':assert all(x['bm25_rank'] is not None for x in r.last_trace)
    if kind=='pure_english':assert all(x['bm25_rank'] is None for x in r.last_trace)


def test_atomizer_one_result_cannot_rewrite_or_resurrect_previous_text(monkeypatch):
    seen=[]
    def h(req):
        seen.append(json.loads(req.content))
        return result({'experiences':[{'content':'凭空增加的新限制','trigger':'被扩大范围'}]})
    draft=AtomicMethod('当前正确方法','任务指定条件')
    out=LLMAtomizer(backend(monkeypatch,h,max_retries=0)).atomize(draft,previous=AtomicMethod('旧规则不应出现','旧范围'))
    assert out.experiences==[draft]
    assert '旧规则不应出现' not in str(seen) and '旧范围' not in str(seen)


def test_structured_output_budget_is_not_per_field_length_validation():
    from verge.llm import FakeLLMBackend
    b=FakeLLMBackend()
    b.generate=lambda *a,**k:{'experiences':[{'content':'x'*100001,'trigger':'*'}]}
    with pytest.raises(ValueError,match='output budget'):
        LLMAtomizer(b,invalid_output_retries=0).atomize(AtomicMethod('方法','*'))


def test_category_top10_after_member_dedup_and_retired_excluded():
    from verge.models import Experience,ExperienceCategory
    from test_atomic_categories import Embed
    cats=[ExperienceCategory(f'cat{i:02}','类别'+str(i),'方法') for i in range(12)]
    ex=[Experience(f'e{i:02}-{j:02}','方法','条件',1,1,[],[],category_id=c.category_id)
        for i,c in enumerate(cats) for j in range(5)]
    query=replace(ex[0],experience_id='query',category_id=None)
    ex.append(Experience('retired','方法','条件',1,1,[],[],status='retired',retired_by_event_id='event'))
    r=HybridRetriever(Embed())
    found=r.category_candidates(query,ex,cats)
    assert len(found)==10 and len({c['category_id'] for c in found})==10
    assert all(len(c['members'])==3 and c['member_count']==5 for c in found)
    assert all(x['document_id']!='retired' for x in r.last_trace)
