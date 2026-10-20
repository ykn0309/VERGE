"""Local category permissions, separate indexes and hybrid rankings (plan67)."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
from experience_helpers import seed, case_for
from verge.models import CandidateExperience, ExperienceCategory, CategoryAssignment, CategoryDescription, AtomicMethod, AtomicResult, ExperienceUpdate
from verge.memory import ExperienceMemory
from verge.modules import HybridRetriever, LLMCategoryOrganizer, make_fake_modules
from verge.pipeline import PipelineRunner
from verge.memory import TrajectoryLog

class Embed:
    def __init__(self): self.calls=[]
    def embed(self,text):
        self.calls.append(text)
        return (1.,.2)

def setup_pool(tmp_path,n=3):
    m=ExperienceMemory(tmp_path)
    ids=[]
    for i in range(n):
        ids.append(seed(m,CandidateExperience('c'+str(i),'核验单位'+str(i),'计算'),'s'+str(i)).experience_id)
    return m,ids

def test_local_assignment_and_description_permissions(tmp_path):
    m,ids=setup_pool(tmp_path)
    event=m.assign('s0',ids[0],CategoryAssignment('create',name='计算',description='核验数量关系'),'v1')
    cid=event.category_id
    m.assign('s1',ids[1],CategoryAssignment('assign',cid),'v2')
    assert m.get(ids[2]).category_id is None
    assert len(m.dirty_categories[cid])==2
    m.describe('s1',cid,None,error='timeout')
    assert m.get(ids[1]).category_id==cid and m.dirty_categories
    m.describe('s1',cid,CategoryDescription('数量计算','核验计算中的关系和单位'))
    assert not m.dirty_categories and m.categories()[0].name=='数量计算'
    assert ExperienceMemory(tmp_path).get(ids[1]).category_id==cid
    with pytest.raises(ValueError): CategoryDescription.from_dict({'name':'x','description':'x','experience_ids':[]})
    with pytest.raises(ValueError): CategoryAssignment('assign',cid,name='not allowed')

def test_hybrid_member_hits_and_no_credit_reembedding(tmp_path):
    m,ids=setup_pool(tmp_path)
    cid=m.assign('s0',ids[0],CategoryAssignment('create',name='公式',description='数值核验'),'v').category_id
    m.assign('s1',ids[1],CategoryAssignment('assign',cid),'v')
    backend=Embed(); r=HybridRetriever(backend)
    recalled=r.retrieve('核验单位1',m.all(),categories=m.categories())
    assert set(x.experience_id for x in recalled)==set(ids)
    assert len([d for d in r.last_trace if d['unit_id']==cid])==3
    assert len(backend.calls)==5 # query + category + 3 member documents
    before=len(backend.calls)
    r.retrieve('核验单位1',m.all(),categories=m.categories())
    assert len(backend.calls)==before+1
    units,_=r.rank('核验单位1',m.all(),m.categories())
    assert len(units)==2
    assert any(d['bm25_rank'] for d in r.last_trace)
    assert len(r.category_candidates(m.get(ids[0]),m.all(),m.categories()))==1
    assert all(d['document_id']!=ids[0] for d in r.last_trace)
    assert r.fork()._vectors==r._vectors

def test_five_batch_local_classification_and_description_failure(tmp_path):
    m,ids=setup_pool(tmp_path,5)
    modules=make_fake_modules()
    modules.category_organizer.organize=lambda e,cs: (CategoryAssignment('assign',cs[0]['category_id']) if cs else CategoryAssignment('create',name='计算',description='计算方法'))
    count=[]
    def desc(*args):
        count.append(args)
        raise ValueError('invalid')
    modules.category_describer.describe=desc
    runner=PipelineRunner(modules,m,TrajectoryLog(tmp_path))
    events=runner._organize('s4')
    assert len(count)==1 and len(m.categories())==1
    assert not m.pending_organization and m.dirty_categories
    assert all(e.category_id==m.categories()[0].category_id for e in m.all())
    assert runner._organize('s4')==[] # same dirty version not retried
    assert len(events)==6

def test_retirement_removes_member_and_empty_category(tmp_path):
    m,ids=setup_pool(tmp_path,1); eid=ids[0]
    m.assign('s0',eid,CategoryAssignment('create',name='计算',description='核验'),'v')
    from test_memory import feedback
    for i in range(5): feedback(m,eid,'t'+str(i))
    draft=AtomicMethod('检查量纲','定量任务')
    ev=m.evolve(eid,ExperienceUpdate('revise',draft),AtomicResult([draft]))
    assert not m.categories() and not m.dirty_categories
    assert m.get(eid).status=='retired'
    child=m.get(ev.child_ids[0])
    assert child.category_id is None and child.alpha==pytest.approx(2.2) and child.beta==pytest.approx(1.2)
    assert m.pending_organization=={child.experience_id:1}
