"""Read-only observations; never stop training or change its checkpoint."""
import json,collections,time,hashlib
from pathlib import Path
from verge.memory import ExperienceMemory
from verge.models import task_identity
root=Path('runs/history-learning-general500-subset120')
out=Path('artifacts/general500-subset120')
manifest=json.loads((root/'manifest.json').read_text())
raw=(root/'checkpoint.json').read_bytes(); c=json.loads(raw)
def correct(p):
 if not p or p.get('error') or not p.get('judgment'):return None
 return p['judgment']['success']
counts=collections.Counter(); paired=collections.Counter(); selection=collections.Counter(); roles=collections.defaultdict(lambda:[0,0,0]); technical=collections.Counter(); rounds=collections.Counter(); review=[]
initial_paths=0; group_hits=historical_hits=0; full4=0; joint_admits=0
for r in c['trajectories']:
 actual=r['actual'] or r['baseline']; a=correct(actual); b=correct(r['baseline']); ids=(r.get('actual') or {}).get('provided_experience_ids',[])
 counts[str(a)]+=1; selection[str(len(ids))]+=1; initial_paths+=len(r.get('subset_paths',[]))
 if ids:
  if a is None or b is None: label='unknown'
  elif a and not b:label='improved'
  elif b and not a:label='regressed'
  elif a:label='both_correct'
  else:label='both_wrong'
  paired[label]+=1
 else:label='no_experience'
 if b is not None:counts['baseline_'+str(b)]+=1
 else:counts['baseline_unknown']+=1
 for p in r['subset_paths']:
  if p.get('error'):technical['initial_path']+=1
 if r.get('error'):technical['preselection']+=1
 group_hits+=bool(r.get('retrieval',{}).get('task_groups'))
 historical_hits+=bool(r.get('retrieval',{}).get('historical',{}).get('scored'))
 if len(ids)==2 and all(correct(p) is not None for p in r['subset_paths']):full4+=1
 for cr in r['critic_rounds']:
  rounds['rounds']+=1
  if cr.get('error'):technical['critic']+=1
  rounds['candidates']+=len(cr['attempts'])
  for at in cr['attempts']:
   rounds['admitted']+=bool(at['admitted'])
   joint_admits+=at.get('reason')=='joint_correct'
  for p in cr.get('subset_paths',[]):
   if p.get('provided_experience_ids') and p.get('error'):technical['candidate_path']+=1
 if r['critic_rounds']:rounds['triggered_tasks']+=1
 if any(at['admitted'] for cr in r['critic_rounds'] for at in cr['attempts']):rounds['source_repaired']+=1
 review.append({'id':r['sample']['sample_id'],'task':r['sample']['task'][:110],'selected':len(ids),'actual':a,'baseline':b,'pair':label,'rounds':len(r['critic_rounds']),'admitted':sum(at['admitted'] for cr in r['critic_rounds'] for at in cr['attempts'])})
for line in (root/'llm_requests.jsonl').open():
 try:r=json.loads(line)
 except ValueError:continue
 u=r.get('usage') or {};v=roles[r.get('role','?')];v[0]+=1;v[1]+=u.get('prompt_tokens',0);v[2]+=u.get('completion_tokens',0)
violations=[]; sources={e['experience_id']:e['source_task_id'] for e in c['experiences']}
for f in c['feedback']:
 for eid in f['experience_ids']:
  if sources.get(eid)==f['task_id']:violations.append('source_credit:'+eid)
for task,eid in c['clustering'].get('citations',[]):
 if sources.get(eid)==task:violations.append('source_edge:'+eid)
# Full loader also checks complete subset protocol and all state invariants.
try:ExperienceMemory.load(root/'checkpoint.json',protocol_id=manifest['protocol_id'],read_only=True)
except Exception as exc:violations.append('load:'+str(exc))
cl=c['clustering']; groups=collections.defaultdict(set)
for kind,items in cl.get('assignments',{}).items():
 if isinstance(items,dict):groups[kind].update(items.values())
summary={'status':manifest['status'],'completed':len(c['trajectories']),'elapsed_minutes':round((time.time()-manifest['started'])/60,2),'experiences':len(c['experiences']),'actual_counts':dict(counts),'selected_count':dict(selection),'paired':dict(paired),'initial_paths':initial_paths,'full_four_path_tasks':full4,'critic':dict(rounds),'joint_admitted_count':joint_admits,'technical':dict(technical),'task_group_hits':group_hits,'historical_candidate_hits':historical_hits,'feedback_records':len(c['feedback']),'groups':{k:len(v) for k,v in groups.items()},'requests_by_role':dict(roles),'tokens':{'input':sum(v[1] for v in roles.values()),'output':sum(v[2] for v in roles.values())},'associations':len(c['clustering'].get('citations',[])),'protocol_violations':violations,'checkpoint_sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(summary,ensure_ascii=False,indent=2));print('RECENT',json.dumps(review[-12:],ensure_ascii=False))
(out/'live-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(out/'sample-index.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
