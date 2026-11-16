from pathlib import Path
from collections import Counter
import json,hashlib,math
from verge.memory import ExperienceMemory
r=Path('runs/v31-general500-125');a=Path('artifacts/v31-general500-125');s=ExperienceMemory.load(r/'checkpoint.json');records=list(s.trajectories.values());tests=[json.loads(x) for x in (r/'test.jsonl').read_text().splitlines()];pairs=[json.loads(x) for x in (r/'test_controls/paired.jsonl').read_text().splitlines()]
assert len(records)==500 and len(tests)==len(pairs)==200
assert len({x['sample_id'] for x in pairs})==200
protocol=json.loads((a/'protocol.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in protocol['code_hashes'].items())
result=json.loads((a/'result.json').read_text());assert hashlib.sha256((r/'checkpoint.json').read_bytes()).hexdigest()==result['frozen_sha256']
def outcomes(rows):
 c=Counter()
 for x in rows:
  q,b=x['actual_correct'],x['baseline_correct'];c['unknown' if q is None or b is None else 'improved' if q and not b else 'regressed' if b and not q else 'both_correct' if q else 'both_wrong']+=1
 return dict(c)
selected={x['sample']['sample_id'] for x in tests if x['paths'] and x['paths'][0]['provided_experience_ids']}
credits=Counter(f.outcome or 'unknown' for f in s.ledger.records)
usage=Counter();requests=Counter();invalid=Counter()
for path in [r/'llm_requests.jsonl',r/'test_controls/llm_requests.jsonl']:
 for line in path.open():
  x=json.loads(line);requests[x['role']]+=1;u=x.get('usage') or {};usage['input']+=u.get('prompt_tokens',u.get('input_tokens',0)) or 0;usage['output']+=u.get('completion_tokens',u.get('output_tokens',0)) or 0
for path in [r/'role_io.jsonl',r/'test_controls/role_io.jsonl']:
 for line in path.open():
  x=json.loads(line)
  if x.get('validation_status')=='invalid':invalid[x['role']]+=1
m=json.loads((r/'manifest.json').read_text());valid=[p for p in pairs if p['actual_correct'] is not None and p['baseline_correct'] is not None]
result.update(train_baseline=dict(Counter(str(x.baseline.correct) if x.baseline else 'None' for x in records)),train_selection=dict(Counter(len(x.selection_reasons) for x in records)),train_credit=dict(credits),test_selected=len(selected),selected_test_pairs=outcomes([x for x in pairs if x['sample_id'] in selected]),baseline_correct=sum(x['baseline_correct'] is True for x in pairs),baseline_unknown=sum(x['baseline_correct'] is None for x in pairs),valid_actual_correct=sum(x['actual_correct'] for x in valid),valid_baseline_correct=sum(x['baseline_correct'] for x in valid),task_group_sizes=dict(Counter(len(v) for v in s.clusters.groups('task').values())),experience_group_sizes=dict(Counter(len(v) for v in s.clusters.groups('experience').values())),critic_tasks=sum(bool(x.critic_rounds) for x in records),critic_rounds=sum(len(x.critic_rounds) for x in records),historical_candidate_tasks=sum(any(c['historical_score'] is not None for c in x.retrieval.get('candidates',[])) for x in records),historical_selected_tasks=sum(any(c['historical_score'] is not None and c['experience_id'] in x.selection_reasons for c in x.retrieval.get('candidates',[])) for x in records),usage=dict(usage),requests=dict(requests),invalid_contracts=dict(invalid),elapsed_minutes=(result['finished']-m['started'])/60,code_unchanged=True,checkpoint_verified=True)
result['mcnemar_exact_two_sided']=min(1,2*sum(math.comb(24,i) for i in range(11))/2**24)
(a/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
