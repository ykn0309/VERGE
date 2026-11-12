"""Descriptive analysis of the saved training run; no model calls or writes to it."""
from pathlib import Path
import json,collections,hashlib,time
from verge.models import task_identity
from verge.feedback import CreditFeedback
rdir=Path('runs/history-learning-general500-subset120'); out=Path('artifacts/general500-subset120')
c=json.loads((rdir/'checkpoint.json').read_text()); data=[json.loads(x) for x in (out/'train.jsonl').open()]; order={x['sample_id']:i for i,x in enumerate(data)}
def result(p):
 return None if not p or p.get('error') or not p.get('judgment') else p['judgment']['success']
margins=collections.defaultdict(collections.Counter); patterns=collections.Counter(); cohorts=collections.defaultdict(collections.Counter); credits=collections.defaultdict(list); examples=collections.defaultdict(list)
for rec in c['trajectories']:
 ids=(rec.get('actual') or {}).get('provided_experience_ids',[]); a=result(rec['actual'] or rec['baseline']); b=result(rec['baseline']); bucket=str(order[rec['sample']['sample_id']]//100+1)

 for label,path in [('actual',rec['actual'] or rec['baseline']),('baseline',rec['baseline'])]:
  if result(path) is None:margins[label]['unknown']+=1
  else:
   margin=path['judgment']['margin'];margins[label]['positive' if margin>0 else 'tie' if margin==0 else 'negative']+=1
 co=cohorts[bucket];co['n']+=1;co['actual_correct']+=a is True;co['baseline_correct']+=b is True;co['selected']+=bool(ids);co['task_group_hit']+=bool(rec['retrieval'].get('task_groups'));co['historical_scored_hit']+=bool(rec['retrieval'].get('historical',{}).get('scored'))
 if len(ids)==2:
  vals=tuple(result(p) for p in rec['subset_paths']);patterns[str(vals)]+=1;examples['four_path'].append(rec['sample']['sample_id'])
 if a is not None and b is not None and ids:
  label='improve' if a and not b else 'regress' if b and not a else 'same_correct' if a else 'same_wrong';co[label]+=1;examples[label].append(rec['sample']['sample_id'])
for row in c['feedback']:
 f=CreditFeedback(**row)
 if f.credits:
  for eid,val in f.credits.items():credits[eid].append(val)
ass=c['clustering']['assignments']; distributions={kind:collections.Counter(objects.values()) for kind,objects in ass.items()}
experience_stats=[]
for e in c['experiences']:
 vals=credits[e['experience_id']];edges=[task for task,eid in c['clustering']['citations'] if eid==e['experience_id']]
 experience_stats.append({'id':e['experience_id'],'trigger':e['method']['trigger'],'observations':len(vals),'credit_sum':sum(vals),'credit_mean':sum(vals)/len(vals) if vals else None,'positive':sum(v>0 for v in vals),'negative':sum(v<0 for v in vals),'zero':sum(v==0 for v in vals),'downstream_tasks':len(edges)})
summary={'completed':len(c['trajectories']),'margin_counts':{k:dict(v) for k,v in margins.items()},'input_order_cohorts':{k:dict(v) for k,v in sorted(cohorts.items())},'four_path_outcome_patterns':dict(patterns),'group_distributions':{k:{'groups':len(v),'singleton_groups':sum(n==1 for n in v.values()),'largest_group':max(v.values(),default=0),'objects':sum(v.values())} for k,v in distributions.items()},'experience_downstream_coverage':{'no_downstream_tasks':sum(e['downstream_tasks']==0 for e in experience_stats),'no_valid_credit':sum(e['observations']==0 for e in experience_stats),'at_least_3_valid_tasks':sum(e['observations']>=3 for e in experience_stats),'at_least_5_valid_tasks':sum(e['observations']>=5 for e in experience_stats)},'experience_stats':sorted(experience_stats,key=lambda e:(-e['observations'],e['id'])),'example_ids':dict(examples)}
(out/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('experience_stats','example_ids')},ensure_ascii=False,indent=2))
