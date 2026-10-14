"""Read-only reconstruction of the current single-ID delayed-experience training protocol."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from math import isclose

from verge.calculator import calculate_experience_credits
from verge.memory import ExperienceMemory,TrajectoryLog
from verge.models import EvidenceRecord,ExperienceCase


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)

def audit(directory,expected=None):
    manifest_file=directory/'run_manifest.json';m=json.loads(manifest_file.read_text())
    require(m['format_version']==7,'wrong manifest protocol')
    root=directory/m['config']['store']['config']['root_dir']
    source=Path(m['config']['dataset']['config']['paths']['train'])
    require(digest(source)==m['dataset_sha256']['train'],'dataset hash mismatch')
    protected=[manifest_file,source,root/'trajectories.jsonl',root/'experience_memory.json',root/'role_io.jsonl',root/'llm_requests.jsonl']
    before={str(p):digest(p) for p in protected if p.exists()}
    dataset=[json.loads(x) for x in source.read_text().splitlines()]
    records=list(TrajectoryLog(root).all());data={s['sample_id']:s for s in dataset}
    require([r.sample_id for r in records]==[s['sample_id'] for s in dataset[:len(records)]],'not an ordered training prefix')
    if expected is not None:require(len(records)==expected,'unexpected sample count')
    final={e.experience_id:e for e in ExperienceMemory(root,read_only=True).all()}
    state={};counts=Counter();window_sizes=[]
    for rec in records:
        raw=data[rec.sample_id];due=set()
        for view in rec.selected_experiences:
            eid=view['experience_id'];e=state[eid]
            require((view['content'],view['trigger'])==(e.content,e.trigger),'selected text mismatch')
            require(isclose(view['confidence'],e.confidence,abs_tol=1e-9),'selected confidence mismatch')
        if rec.initial_judgment is not None:
            credits=calculate_experience_credits(rec.initial_rubric,rec.initial_judgment)
            require(credits==rec.downstream_credits,'R0 credit mismatch')
            for eid,c in credits.items():
                e=state[eid];counts['downstream_events']+=1
                counts['net_positive' if c.net>0 else 'net_negative' if c.net<0 else 'net_tie']+=1
                state[eid]=replace(e,alpha=e.alpha+c.positive_credit,beta=e.beta+c.negative_credit,
                    evidence_history=[*e.evidence_history,EvidenceRecord(rec.sample_id,'downstream_credit',c.positive_credit,c.negative_credit,text_version=e.text_version)])
            for view in rec.selected_experiences:
                eid=view['experience_id']
                if eid not in credits or len(state[eid].cases)>=10:continue
                e=state[eid];c=ExperienceCase(rec.sample_id,raw['task'],raw['chosen'],raw['rejected'],
                    eid,e.text_version,view['content'],view['trigger'],rec.initial_rubric,rec.initial_judgment,credits[eid])
                state[eid]=replace(e,cases=[*e.cases,c],source_trajectories=[*e.source_trajectories,rec.sample_id])
                if state[eid].update_due:due.add(eid)
        if rec.admitted_experience_id:
            eligible=[a for a in rec.application_attempts if a.eligible]
            require(eligible and eligible[-1].candidate_id==rec.admitted_candidate_id,'latest eligible candidate mismatch')
            app=eligible[-1];candidate=next(p.candidate for p in rec.proposal_attempts if p.candidate and p.candidate.candidate_id==app.candidate_id)
            e=final[rec.admitted_experience_id]
            origin=ExperienceCase(rec.sample_id,raw['task'],raw['chosen'],raw['rejected'],candidate.candidate_id,1,
                candidate.content,candidate.trigger,app.rubric,app.judgment,app.credit,
                next(p.analysis for p in rec.proposal_attempts if p.candidate and p.candidate.candidate_id==candidate.candidate_id))
            state[e.experience_id]=replace(e,content=candidate.content,trigger=candidate.trigger,
                alpha=1+app.credit.positive_credit,beta=1+app.credit.negative_credit,
                cases=[origin],source_trajectories=[rec.sample_id],text_version=1,update_history=[],
                evidence_history=[EvidenceRecord(rec.sample_id,'application_credit',app.credit.positive_credit,app.credit.negative_credit,candidate.candidate_id,text_version=1)])
        require(set(rec.experience_updates)==due,'delayed update invocation mismatch')
        for eid,event in rec.experience_updates.items():
            e=state[eid];window_sizes.append(e.pending_feedback_count)
            require((event.previous_content,event.previous_trigger,event.previous_version)==(e.content,e.trigger,e.text_version),'update prior text mismatch')
            require(event.sample_id==rec.sample_id and event.case_ids==[c.sample_id for c in e.cases],'update case mismatch')
            state[eid]=replace(e,content=event.content,trigger=event.trigger,text_version=event.text_version,
                               update_history=[*e.update_history,event])
    require(state.keys()==final.keys(),'pool IDs mismatch')
    for eid,e in state.items():
        actual=final[eid]
        require(isclose(e.alpha,actual.alpha,abs_tol=1e-9) and isclose(e.beta,actual.beta,abs_tol=1e-9),'final Beta mismatch')
        require(replace(e,alpha=actual.alpha,beta=actual.beta)==actual,'final text/cases/evidence/history mismatch')
    roles=[json.loads(x) for x in (root/'role_io.jsonl').read_text().splitlines()]
    require(all(x['split']=='train' for x in roles),'unexpected held-out calls')
    llm=[x for x in roles if x['role'] in {'selector','initial_generator','judge','critic','experience_updater'}]
    require(all(x['model']=='qwen3.5-flash' for x in llm),'model changed')
    updates=[u for e in final.values() for u in e.update_history]
    require(before=={p:digest(Path(p)) for p in before},'audit changed source')
    return {'samples':len(records),'states':dict(Counter(r.state for r in records)),
        'initial_successes':sum(bool(r.initial_judgment and r.initial_judgment.success) for r in records),
        'final_successes':sum(r.state in {'initial_success','application_success','success_without_candidate'} for r in records),
        'experiences':len(final),'downstream_feedback':dict(counts),'updates':len(updates),
        'update_errors':sum(u.error is not None for u in updates),'window_sizes':window_sizes,
        'case_counts':dict(Counter(len(e.cases) for e in final.values())),
        'frozen_experiences':sum(e.text_frozen for e in final.values()),
        'confidence_range':[min((e.confidence for e in final.values()),default=None),max((e.confidence for e in final.values()),default=None)],
        'llm_role_calls':dict(Counter(x['role'] for x in roles)),
        'checks':'ordered prefix, R0 credit, latest candidate, delayed windows, final Beta/text/cases/versions/history, Qwen model, train only, original hashes unchanged',
        'source_sha256':before}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('--expected',type=int);p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.run_dir,a.expected);text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:
        require(a.output.resolve() not in {Path(p).resolve() for p in result['source_sha256']},'output would overwrite source')
        a.output.write_text(text+'\n')
    print(text)
