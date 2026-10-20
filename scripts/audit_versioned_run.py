"""Read-only reconstruction of version-specific credit and evolution (protocol10)."""
from __future__ import annotations
import json,hashlib,argparse
from pathlib import Path
from collections import Counter
from math import isclose
from verge.models import TrajectoryRecord
from verge.calculator import calculate_experience_credits,calculate_experience_usage

def require(condition,message):
    if not condition:raise ValueError(message)

def audit(directory,expected=None):
    directory=Path(directory)
    m=json.loads((directory/'run_manifest.json').read_text())
    require(m['format_version']==10,'wrong manifest protocol')
    root=directory/m['config']['store']['config']['root_dir']
    paths=[directory/'run_manifest.json',*root.glob('*.json*')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    raw=json.loads((root/'experience_memory.json').read_text())
    require(raw['schema_version']==8,'wrong memory protocol')
    final={e['experience_id']:e for e in raw['experiences']}
    records=[TrajectoryRecord.from_json(x) for x in (root/'trajectories.jsonl').read_text().splitlines()]
    if expected is not None:require(len(records)==expected,'unexpected sample count')
    train=Path(m['config']['dataset']['config']['paths']['train'])
    data=[json.loads(x) for x in train.read_text().splitlines()]
    require(hashlib.sha256(train.read_bytes()).hexdigest()==m['dataset_sha256']['train'],'dataset changed')
    require([r.sample_id for r in records]==[x['sample_id'] for x in data[:len(records)]],'unordered prefix')
    state={};updates=0;windows=[]
    def birth(eid,content,trigger,p=0,n=0,penalty=0,inheritance=None):
        require(eid not in state,'duplicate birth')
        state[eid]={'content':content,'trigger':trigger,'p':p,'n':n,'penalty':penalty,'inheritance':inheritance,'window_amounts':{},'active':True,'downstream':[], 'consumed':0,'attempted':0,'usage':Counter()}
    for r in records:
        for v in r.selected_experiences:
            e=state[v['experience_id']]
            require(e['active'] and (e['content'],e['trigger'])==(v['content'],v['trigger']),'selected retired/mismatched version')
            require(isclose(v['confidence'],(1+e['p'])/(2+e['p']+e['n']+e['penalty']),abs_tol=1e-9),'snapshot credit mismatch')
        if r.initial_judgment:
            credits=calculate_experience_credits(r.initial_rubric,r.initial_judgment)
            require(credits==r.downstream_credits,'R0 credit mismatch')
            usages=calculate_experience_usage(r.initial_rubric,r.initial_judgment,r.sample_id,{k:1 for k in state})
            for eid,c in credits.items():
                e=state[eid];e['p']+=c.positive_credit;e['n']+=c.negative_credit
                if c.amount:
                    e['downstream'].append(r.sample_id);e['window_amounts'][r.sample_id]=c.amount
            for eid,u in usages.items():
                state[eid]['usage'].update({k:getattr(u,k) for k in ('positive','negative','both_true','both_false')})
        for p in r.proposal_attempts:
            for c in p.atomic_candidates:
                if c.experience_id:
                    app=next(a for a in r.application_attempts if a.candidate_id==c.candidate_id)
                    birth(c.experience_id,c.content,c.trigger,app.credit.positive_credit,app.credit.negative_credit)
                    u=calculate_experience_usage(app.rubric,app.judgment,r.sample_id,{x:1 for criterion in app.rubric.criteria for x in criterion.source_experiences})[c.candidate_id]
                    state[c.experience_id]['usage'].update({k:getattr(u,k) for k in ('positive','negative','both_true','both_false')})
        due={eid for eid,e in state.items() if e['active'] and len(e['downstream'])-e['consumed']>=5 and len(e['downstream'])>e['attempted']}
        require(set(r.experience_updates)==due,'evolution invocation mismatch')
        for eid,ev in r.experience_updates.items():
            e=state[eid];updates+=1
            require(ev.case_ids==e['downstream'][-5:] and ev.sample_id==r.sample_id,'evolution window mismatch')
            amount=sum(e['window_amounts'][sid] for sid in ev.case_ids)
            require(isclose(ev.window_credit_amount,amount,abs_tol=1e-9),'window credit amount mismatch')
            require(ev.penalty_lambda==0.2,'wrong penalty lambda')
            require(isclose(ev.penalty_credit,0.2*amount if ev.child_ids else 0,abs_tol=1e-9),'penalty mismatch')
            windows.append(len(ev.case_ids));e['attempted']=len(e['downstream'])
            if not ev.error:e['consumed']=e['attempted']
            if ev.child_ids:
                e['active']=False
                require(ev.atomic_result is not None,'missing atomic output')
                k=len(ev.child_ids)
                for child,method in zip(ev.child_ids,ev.atomic_result.experiences):
                    inheritance={'parent_id':eid,'event_id':ev.event_id,'positive_credit':e['p']/k,
                        'negative_credit':e['n']/k,'inherited_penalty':e['penalty']/k,'evolution_penalty':ev.penalty_credit/k}
                    birth(child,method.content,method.trigger,e['p']/k,e['n']/k,
                          (e['penalty']+ev.penalty_credit)/k,inheritance)
    require(set(state)==set(final),'pool IDs mismatch')
    for eid,e in state.items():
        f=final[eid]
        actual_inheritance=f['inherited_credit'];expected_inheritance=e['inheritance']
        require((actual_inheritance is None)==(expected_inheritance is None),'birth credit allocation mismatch')
        if expected_inheritance:
            require(actual_inheritance['parent_id']==expected_inheritance['parent_id'] and
                    actual_inheritance['event_id']==expected_inheritance['event_id'] and
                    all(isclose(actual_inheritance[k],expected_inheritance[k],abs_tol=1e-9)
                        for k in ('positive_credit','negative_credit','inherited_penalty','evolution_penalty')),
                    'birth credit allocation mismatch')
        inherited=e['inheritance'] or {}
        require(isclose(sum(x['positive_credit'] for x in f['evidence_history']),e['p']-inherited.get('positive_credit',0),abs_tol=1e-9), 'own positive ledger mismatch')
        require(isclose(sum(x['negative_credit'] for x in f['evidence_history']),e['n']-inherited.get('negative_credit',0),abs_tol=1e-9), 'own negative ledger mismatch')
        require(isclose(f['alpha'],1+e['p'],abs_tol=1e-9) and isclose(f['beta'],1+e['n']+e['penalty'],abs_tol=1e-9),'final Beta mismatch')
        require((f['status']=='active')==e['active'],'final lifecycle mismatch')
        require((f['content'],f['trigger'])==(e['content'],e['trigger']),'immutable text changed')
        actual=Counter({k:sum(u[k] for u in f['usage_history']) for k in ('positive','negative','both_true','both_false')})
        require(actual==e['usage'],'final usage mismatch')
        require(f['last_consumed_feedback_seq']==e['consumed'] and f['last_attempted_feedback_seq']==e['attempted'],'feedback sequence mismatch')
        require([c['sample_id'] for c in f['cases'] if c['source_id']==eid]==e['downstream'],'case history mismatch')
    require(before=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before},'source changed')
    return {'samples':len(records),'experiences':len(final),'active_experiences':sum(e['active'] for e in state.values()),
            'retired_experiences':sum(not e['active'] for e in state.values()),'updates':updates,'window_sizes':windows,
            'final_successes':sum(r.state in ('initial_success','application_success','success_without_candidate') for r in records),
            'source_sha256':before}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path);parser.add_argument('--expected',type=int)
    args=parser.parse_args();print(json.dumps(audit(args.run_dir,args.expected),ensure_ascii=False,indent=2))
