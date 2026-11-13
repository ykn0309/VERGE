from pathlib import Path
from collections import Counter
import json,hashlib
from verge.prompts import NumberedIDCodec
from verge.memory import ExperienceMemory
from audit import audit

ROOT=Path('runs/v31-smoke124');OUT=Path('artifacts/v31-smoke124')
audit()
summary=json.loads((OUT/'live-summary.json').read_text())
manifest=json.loads((ROOT/'manifest.json').read_text())
state=ExperienceMemory.load(ROOT/'checkpoint.json')
tests=[json.loads(x) for x in (ROOT/'test.jsonl').read_text().splitlines()]
events=[json.loads(x) for x in (ROOT/'role_io.jsonl').read_text().splitlines()]
valid={e['call_id'] for e in events if e.get('validation_status')=='valid'}
requests=[e for e in events if e.get('messages') and e.get('call_id') in valid and not e.get('error')]
violations=summary['violations']
for record in tests:
    sid=record['sample']['sample_id']
    calls=[r for r in requests if r['sample_id']==sid]
    assert record['mode']=='inference' and len(record['paths'])==1 and not record['critic_rounds']
    assert len(record['paths'][0]['provided_experience_ids'])<=1
    for call in calls:
        codec=NumberedIDCodec();codec.forward=call['id_mapping']['forward'];codec.reverse=call['id_mapping']['reverse'];p=codec.restore(call['input'])
        assert call['role']!='critic'
        if call['role']=='generator' and p.get('stage')=='revise':
            assert p['draft']==record['draft'] and len(p['experiences'])<=1
        if call['role']=='selector':assert p['max_selected']==1
assert hashlib.sha256((ROOT/'checkpoint.json').read_bytes()).hexdigest()==manifest['test']['frozen_sha256']
protocol=json.loads((OUT/'protocol.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in protocol['code_hashes'].items())
training=list(state.trajectories.values())
summary.update(elapsed_minutes=(manifest['finished']-manifest['started'])/60,
    test={'tasks':len(tests),'correct':sum(p['paths'][0]['judgment'] is not None and not p['paths'][0]['error'] and p['paths'][0]['judgment']['success'] for p in tests),
          'unknown':sum(p['paths'][0]['error'] is not None for p in tests),'selected_counts':dict(Counter(len(r['paths'][0]['provided_experience_ids']) for r in tests)),'frozen_hash_unchanged':True},
    task_group_sizes=dict(Counter(len(v) for v in state.clusters.groups('task').values())),
    experience_group_sizes=dict(Counter(len(v) for v in state.clusters.groups('experience').values())),
    historical_candidate_tasks=sum(any(c['historical_score'] is not None for c in r.retrieval.get('candidates',[])) for r in training),
    historical_selected_tasks=sum(any(c['historical_score'] is not None and c['experience_id'] in r.selection_reasons for c in r.retrieval.get('candidates',[])) for r in training),
    code_unchanged=True)
(OUT/'final-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False))
