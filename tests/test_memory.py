"""Immutable version credit, windows and atomic replacement."""
import json
import pytest
from experience_helpers import seed,case_for
from verge.memory import ExperienceMemory,TrajectoryLog
from verge.models import CandidateExperience, Credit, AtomicMethod, AtomicResult, ExperienceUpdate, TrajectoryRecord

def pool(tmp_path):
    m=ExperienceMemory(tmp_path)
    eid=seed(m,CandidateExperience('c','方法','*'),'origin').experience_id
    return m,eid

def feedback(m,eid,sid,p=.2,n=0):
    e=m.get(eid);c=case_for(eid,sid,e.content,e.trigger,p=p,n=n)
    m.apply_downstream_credits(sid,{eid:c.credit})
    assert m.collect_case(eid,c)
    return c

def test_keep_consumes_five_and_continues_beyond_ten(tmp_path):
    m,eid=pool(tmp_path)
    for i in range(15):
        feedback(m,eid,f't{i}')
        e=m.get(eid)
        assert e.update_due==((i+1)%5==0)
        if e.update_due: m.evolve(eid,ExperienceUpdate('keep'),None)
    e=ExperienceMemory(tmp_path).get(eid)
    assert e.pending_feedback_count==0 and len(e.cases)==16 and len(e.working_cases)==5
    assert e.status=="active" and len(e.update_history)==3
    assert e.alpha==pytest.approx(4.2)

def test_split_retains_parent_evidence_and_shares_credit(tmp_path):
    m,eid=pool(tmp_path)
    for i in range(5): feedback(m,eid,f't{i}',p=.2,n=.1)
    before=m.get(eid)
    result=AtomicResult([AtomicMethod('动作A','条件A'),AtomicMethod('动作B','条件B')])
    ev=m.evolve(eid,ExperienceUpdate('revise',AtomicMethod('动作A和B','条件')),result)
    parent=m.get(eid)
    assert parent.status=='retired' and parent.evidence_history==before.evidence_history
    assert parent.alpha==before.alpha and parent.beta==before.beta
    assert len(m.all(active_only=True))==2
    for key in ev.child_ids:
        child=m.get(key)
        assert child.alpha==pytest.approx(1.6) and child.beta==pytest.approx(1.4)
        assert not child.cases and not child.usage_history and not child.evidence_history
        assert child.origin_refs==[eid] and child.created_by_event_id==ev.event_id
    assert ExperienceMemory(tmp_path).get(eid)==parent
    with pytest.raises(ValueError):m.apply_downstream_credits('later',{eid:Credit(.1,0)})

def test_failed_window_retries_only_on_new_case(tmp_path):
    m,eid=pool(tmp_path)
    for i in range(5):feedback(m,eid,f't{i}')
    m.evolve(eid,None,None,error='timeout')
    assert not m.get(eid).update_due and m.get(eid).pending_feedback_count==5
    feedback(m,eid,'t5',p=0,n=.2)
    e=m.get(eid)
    assert e.update_due and [c.sample_id for c in e.working_cases]==[f't{i}' for i in range(1,6)]
    m.evolve(eid,ExperienceUpdate('keep'),None)
    assert m.get(eid).pending_feedback_count==0 and len(m.get(eid).cases)==7

def test_unchanged_revision_keep_and_duplicate_feedback_rejected(tmp_path):
    m,eid=pool(tmp_path)
    for i in range(5):feedback(m,eid,f't{i}')
    e=m.get(eid);draft=AtomicMethod(e.content,e.trigger)
    ev=m.evolve(eid,ExperienceUpdate('revise',draft),AtomicResult([draft]))
    assert ev.result=='keep' and not ev.child_ids and m.get(eid).status=='active'
    before=m.path.read_bytes()
    with pytest.raises(ValueError): m.apply_downstream_credits('t4',{eid:Credit(.2,0)})
    assert m.path.read_bytes()==before
    with pytest.raises(ValueError):m.evolve(eid,ExperienceUpdate('keep'),None)

def test_readonly_and_old_protocol(tmp_path):
    m,eid=pool(tmp_path);before=m.path.read_bytes();f=ExperienceMemory(tmp_path,read_only=True)
    for action in (lambda:f.evolve(eid,None,None,error='x'),lambda:f.apply_downstream_credits('x',{eid:Credit(.1,0)}),
                   lambda:seed(f,CandidateExperience('new','方法','*'),'new')):
        with pytest.raises(ValueError):action()
    assert m.path.read_bytes()==before
    data=json.loads(before);data['schema_version']=6;m.path.write_text(json.dumps(data))
    with pytest.raises(ValueError):ExperienceMemory(tmp_path)
    t=TrajectoryRecord('s','initial_failure');log=TrajectoryLog(tmp_path);log.append(t)
    assert log.get('s')==t
