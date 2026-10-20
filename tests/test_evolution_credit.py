"""Credit inheritance is conserved, bounded by the actual window, and auditable."""
import json
from math import fsum

import pytest

from test_memory import pool, feedback
from verge.memory import ExperienceMemory
from verge.models import AtomicMethod, AtomicResult, ExperienceUpdate
from verge.web import _evolution


def revise(memory, eid, count=1):
    methods = [AtomicMethod(f"独立方法{i}", f"独立条件{i}") for i in range(count)]
    return memory.evolve(eid, ExperienceUpdate("revise", AtomicMethod("修订草稿", "范围")), AtomicResult(methods))


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("p,n", [(0.2, 0), (0, 0.3), (0.1, 0.1), (0.15, 0.05)])
def test_inheritance_conserves_credit_and_one_penalty(tmp_path, count, p, n):
    m, eid = pool(tmp_path)
    for i in range(5):
        feedback(m, eid, f"t{i}", p=p, n=n)
    before = m.get(eid)
    event = revise(m, eid, count)
    children = [m.get(key) for key in event.child_ids]
    penalty = .2 * 5 * (p+n)
    assert event.window_credit_amount == pytest.approx(5*(p+n))
    assert event.penalty_credit == pytest.approx(penalty)
    assert event.penalty_lambda == .2
    assert fsum(c.positive_credit for c in children) == pytest.approx(before.positive_credit)
    assert fsum(c.negative_credit for c in children) == pytest.approx(before.negative_credit)
    assert fsum(c.penalty_credit for c in children) == pytest.approx(penalty)
    for c in children:
        assert c.alpha == pytest.approx(1 + before.positive_credit/count)
        assert c.beta == pytest.approx(1 + (before.negative_credit + penalty)/count)
        assert not c.evidence_history and not c.cases and not c.usage_history
        assert c.evidence_amount == pytest.approx(before.evidence_amount/count)
        assert c.pending_feedback_count == 0 and not c.update_due
    assert m.get(eid).evidence_history == before.evidence_history
    assert (m.get(eid).alpha, m.get(eid).beta) == (before.alpha, before.beta)
    restored = ExperienceMemory(tmp_path)
    assert [restored.get(c.experience_id) for c in children] == children


def test_repeated_evolution_inherits_old_penalty_without_double_counting(tmp_path):
    m, eid = pool(tmp_path)
    for i in range(5): feedback(m, eid, f"a{i}")
    first = revise(m, eid)
    child = first.child_ids[0]
    for i in range(5): feedback(m, child, f"b{i}", p=.1, n=.2)
    before = m.get(child)
    second = revise(m, child, 3)
    grandchildren = [m.get(k) for k in second.child_ids]
    for e in grandchildren:
        assert e.inherited_credit.inherited_penalty == pytest.approx(.2/3)
        assert e.inherited_credit.evolution_penalty == pytest.approx(.3/3)
        assert e.inherited_credit.positive_credit == pytest.approx(1.7/3)
        assert e.inherited_credit.negative_credit == pytest.approx(1/3)
    assert sum(e.penalty_credit for e in grandchildren) == pytest.approx(.5)
    assert m.get(child).penalty_credit == before.penalty_credit
    # Own feedback affects the leaf exactly once; inherited evidence is not a task.
    leaf = second.child_ids[0]
    old = m.get(leaf)
    feedback(m, leaf, 'leaf', p=0, n=.1)
    new = ExperienceMemory(tmp_path).get(leaf)
    assert new.beta == pytest.approx(old.beta+.1)
    assert new.penalty_credit == old.penalty_credit and len(new.downstream_cases) == 1


def test_penalty_uses_latest_window_not_full_history_or_inherited_evidence(tmp_path):
    m, eid = pool(tmp_path)
    for i in range(5): feedback(m,eid,f"a{i}",p=.8)
    keep = m.evolve(eid, ExperienceUpdate('keep'), None)
    assert keep.penalty_credit == 0
    for i in range(5): feedback(m,eid,f"b{i}",p=0,n=.1)
    event = revise(m,eid)
    assert event.window_credit_amount == pytest.approx(.5)
    assert event.penalty_credit == pytest.approx(.1)
    assert m.get(event.child_ids[0]).positive_credit == pytest.approx(4.2)


@pytest.mark.parametrize('mode', ['keep', 'same', 'error'])
def test_nonreplacement_never_penalizes(tmp_path, mode):
    m,eid = pool(tmp_path)
    for i in range(5): feedback(m,eid,f"t{i}")
    before=m.get(eid)
    if mode=='same':
        draft=AtomicMethod(before.content,before.trigger)
        event=m.evolve(eid,ExperienceUpdate('revise',draft),AtomicResult([draft]))
    elif mode=='error': event=m.evolve(eid,None,None,error='exhausted')
    else: event=m.evolve(eid,ExperienceUpdate('keep'),None)
    after=m.get(eid)
    assert event.penalty_credit==0 and not event.child_ids
    assert (after.alpha,after.beta,after.inherited_credit)==(before.alpha,before.beta,None)


@pytest.mark.parametrize('field', ['positive_credit', 'negative_credit', 'inherited_penalty', 'evolution_penalty', 'event_id'])
def test_reader_rejects_forged_inheritance_even_when_beta_matches(tmp_path, field):
    m,eid=pool(tmp_path)
    for i in range(5): feedback(m,eid,f"t{i}")
    event=revise(m,eid,2)
    data=json.loads(m.path.read_text())
    child=next(e for e in data['experiences'] if e['experience_id']==event.child_ids[0])
    if field=='event_id': child['inherited_credit'][field]='forged_event'
    else:
        child['inherited_credit'][field]+=.1
        child['alpha' if field=='positive_credit' else 'beta']+=.1
        child['confidence']=child['alpha']/(child['alpha']+child['beta'])
    m.path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='inherited credit'):
        ExperienceMemory(tmp_path)


def test_web_timeline_accounts_for_birth_allocation_without_fake_feedback(tmp_path):
    m,eid=pool(tmp_path)
    for i in range(5): feedback(m,eid,f"t{i}")
    event=revise(m,eid)
    child=m.get(event.child_ids[0])
    warnings=[]
    timeline=_evolution(child.to_dict(),[{'experience_updates':{eid:event.to_dict()}}],{'t4':0},warnings)
    assert not warnings and [e['kind'] for e in timeline]==['prior','credit_inheritance']
    assert timeline[-1]['alpha']==child.alpha and timeline[-1]['beta']==child.beta
    assert timeline[-1]['sample_id']=='t4' and not child.evidence_history


def test_readonly_and_duplicate_evolution_cannot_charge_twice(tmp_path):
    m,eid=pool(tmp_path)
    for i in range(5): feedback(m,eid,f"t{i}")
    revise(m,eid)
    before=m.path.read_bytes()
    with pytest.raises(ValueError): revise(m,eid)
    with pytest.raises(ValueError): revise(ExperienceMemory(tmp_path,read_only=True),eid)
    assert before==m.path.read_bytes()


def test_real_web_projection_and_frontend_distinguish_credit_sources(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from test_delayed_run_audit import fixture_run
    from verge.modules import LLMExperienceUpdater
    from verge.web import load_run
    monkeypatch.setattr(LLMExperienceUpdater,'update',lambda self,e:
        ExperienceUpdate('revise',AtomicMethod(e.content+'修订',e.trigger)))
    directory=fixture_run(tmp_path)
    data=load_run(directory)
    assert not data['warnings']
    child=next(e for e in data['experiences'] if e['inherited_credit'] and not e['evidence_history'])
    credit=child['credit_breakdown']
    assert credit['inherited_positive']>0 and credit['evolution_penalty']>0
    assert credit['observed_positive']==credit['observed_negative']==0
    assert child['downstream_task_count']==0 and child['pending_feedback_count']==0
    assert child['evidence_amount']==pytest.approx(child['alpha']+child['beta']-2-credit['evolution_penalty']-credit['inherited_penalty'])
    assert child['evolution'][-1]['confidence']==pytest.approx(child['confidence'])
    source=tmp_path/'render.js'
    source.write_text(Path('verge/web_assets/app.js').read_text().split('const experiencePreview =')[0])
    payload=tmp_path/'web_data.json';payload.write_text(json.dumps(data))
    program="""const vm=require('node:vm'),fs=require('node:fs');
const ctx={document:{getElementById:()=>({})},URLSearchParams,location:{search:''},setTimeout,clearTimeout};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
vm.runInContext('data='+fs.readFileSync(process.argv[2],'utf8'),ctx);
console.log(vm.runInContext('experienceDetail(data.experiences.find(e=>e.inherited_credit&&!e.evidence_history.length))',ctx));"""
    rendered=subprocess.run(['node','-e',program,str(source),str(payload)],capture_output=True,text=True)
    assert rendered.returncode==0,rendered.stderr
    assert '尚无本版本数值反馈' in rendered.stdout and '继承信用' in rendered.stdout
    assert '历史惩罚' in rendered.stdout and '本次演化惩罚' in rendered.stdout
    assert '从Beta(1,1)重新积累' not in rendered.stdout
