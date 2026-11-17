from pathlib import Path
from collections import Counter
import json,time
p=Path('runs/reference500-128');m=json.loads((p/'manifest.json').read_text());c=json.loads((p/'checkpoint.json').read_text());rs=c['trajectories'];counts=Counter()
for f in c['feedback']:
 a,b=f['experience_correct'],f['baseline_correct'];counts['unknown' if a is None or b is None else 'improved' if a and not b else 'regressed' if b and not a else 'both_correct' if a else 'both_wrong']+=1
row={'time':time.time(),'status':m['status'],'completed':len(rs),'experiences':len(c['experiences']),'minutes':round((time.time()-m['started'])/60,1),'initial_errors':sum(bool(p['error']) for r in rs for p in r['paths']),'critic_errors':sum(bool(z['error']) for r in rs for z in r['critic_rounds']),'pairs':dict(counts),'task_groups':len(set(c['clustering']['assignments']['task'].values())),'experience_groups':len(set(c['clustering']['assignments']['experience'].values())),'cluster_status':(c['clustering'].get('last_update') or {}).get('status'),'test':m.get('test')}
f=p/'test_controls/paired.jsonl';row['test_controls']=sum(1 for _ in f.open()) if f.exists() else 0
with Path('artifacts/reference500-128/monitor.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
print(json.dumps(row))
