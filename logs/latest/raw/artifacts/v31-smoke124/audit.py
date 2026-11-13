"""Read-only checks against saved v3.1 trajectories and actual request payloads."""
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib,json
from pathlib import Path
from verge.memory import ExperienceMemory
from verge.prompts import NumberedIDCodec

ROOT=Path('runs/v31-smoke124')
OUT=Path('artifacts/v31-smoke124')

def audit():
    raw=json.loads((ROOT/'checkpoint.json').read_text())
    state=ExperienceMemory.load(ROOT/'checkpoint.json')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    events=[]
    for line in (ROOT/'role_io.jsonl').read_text().splitlines():
        try: events.append(json.loads(line))
        except ValueError: pass
    valid={r['call_id'] for r in events if r.get('event_type')=='contract_validation' and r.get('validation_status')=='valid'}
    calls=[r for r in events if r.get('messages')]
    bysample=defaultdict(list)
    for call in calls: bysample[call['sample_id']].append(call)
    violations=[]
    summaries=[]
    for record in state.trajectories.values():
        sid=record.sample.sample_id
        observed=[r for r in bysample[sid] if r.get('call_id') in valid and not r.get('error')]
        inputs=[]
        for call in observed:
            codec=NumberedIDCodec();codec.forward=call.get('id_mapping',{}).get('forward',{});codec.reverse=call.get('id_mapping',{}).get('reverse',{})
            inputs.append((call['role'],codec.restore(call['input'])))
        drafts=[p for role,p in inputs if role=='generator' and p.get('stage')=='draft']
        if len(drafts)!=int(record.draft is not None):violations.append([sid,'successful draft count',len(drafts)])
        revisions=[p for role,p in inputs if role=='generator' and p.get('stage')=='revise']
        attempts=[a for r in record.critic_rounds for a in r.attempts]
        expected=[p for p in [*record.paths,*[a.path for a in attempts]] if p.rubric is not None]
        if len(revisions)!=len(expected):violations.append([sid,'successful revision count',len(revisions),len(expected)])
        for p in revisions:
            if p['draft']!=record.draft.to_dict():violations.append([sid,'draft replaced'])
            if len(p.get('experiences',[]))>1:violations.append([sid,'combination input'])
        if Counter(tuple(e['experience_id'] for e in p['experiences']) for p in revisions)!=Counter(tuple(p.provided_experience_ids) for p in expected):violations.append([sid,'revision provision differs'])
        for role,p in inputs:
            if role=='generator' and set(p)-{'task','stage','draft','experiences','validation_feedback'}:violations.append([sid,'generator information boundary'])
            if role=='judge' and set(p)-{'task','criteria','response','validation_feedback'}:violations.append([sid,'judge information boundary'])
            if role=='selector' and set(p)-{'task','candidates','max_selected','validation_feedback'}:violations.append([sid,'selector information boundary'])
            if role=='critic' and 'task' in p:
                if p['preference_evidence']['individual_preference']!=record.sample.metadata.get('individual_preference',[]):violations.append([sid,'human evidence mismatch'])
        summaries.append({'sample_id':sid,'task':record.sample.task,'selected':list(record.selection_reasons),'paths':[{'ids':p.provided_experience_ids,'correct':p.correct,'margin':p.judgment.margin if p.judgment else None,'error':p.error,'criteria':[c.question for c in p.rubric.criteria] if p.rubric else []} for p in record.paths],
                          'critic':[{'round':r.round_number,'analysis':r.analysis,'error':r.error,'attempts':[{'id':a.experience.experience_id,'method':a.experience.method.to_dict(),'admitted':a.admitted,'result':a.path.correct,'error':a.path.error} for a in r.attempts]} for r in record.critic_rounds],
                          'task_groups':record.retrieval.get('task_groups',[]),'historical_selected':[c['experience_id'] for c in record.retrieval.get('candidates',[]) if c['historical_score'] is not None and c['experience_id'] in record.selection_reasons]})
    requests=[]
    if (ROOT/'llm_requests.jsonl').exists():
        for line in (ROOT/'llm_requests.jsonl').read_text().splitlines():
            try: requests.append(json.loads(line))
            except ValueError: pass
    token=Counter()
    for r in requests:
        usage=r.get('usage') or {}
        token['input']+=usage.get('prompt_tokens',usage.get('input_tokens',0)) or 0
        token['output']+=usage.get('completion_tokens',usage.get('output_tokens',0)) or 0
    records=list(state.trajectories.values());paths=[p for r in records for p in r.paths];attempts=[a for r in records for c in r.critic_rounds for a in c.attempts]
    outcome=Counter(r.outcome or 'unknown' for r in state.ledger.records)
    summary={'status':manifest['status'],'completed':len(records),'pool':len(state.experiences),'initial_paths':len(paths),'baseline':dict(Counter(str(r.baseline.correct) if r.baseline else 'None' for r in records)),
             'selected_counts':dict(Counter(len(r.selection_reasons) for r in records)), 'critic_tasks':sum(bool(r.critic_rounds) for r in records),'critic_rounds':sum(len(r.critic_rounds) for r in records),'attempts':len(attempts),'admitted':sum(a.admitted for a in attempts),
             'pairs':dict(outcome),'initial_errors':sum(p.error is not None for p in paths),'candidate_errors':sum(a.path.error is not None for a in attempts),'task_errors':sum(r.error is not None for r in records),'critic_errors':sum(c.error is not None for r in records for c in r.critic_rounds),
             'task_groups':len(state.clusters.groups('task')),'experience_groups':len(state.clusters.groups('experience')),'associations':len(raw['clustering']['citations']), 'cluster_status':state.clusters.last_update.status if state.clusters.last_update else None,
             'requests':dict(Counter(r.get('role') for r in requests)),'tokens':dict(token),'invalid_contracts':sum(r.get('validation_status')=='invalid' for r in events), 'violations':violations}
    (OUT/'live-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'sample-review.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':audit()
