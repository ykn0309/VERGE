"""Summarize real diagnostic runs without treating training or reviewer flags as test accuracy."""
import sys,json,hashlib,statistics
from pathlib import Path
from collections import Counter
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verge.models import Experience
from verge.calculator import ProbabilityModel,rebuild_quality

ROOT=Path(__file__).resolve().parents[1]

def analyze(name):
    p=ROOT/'runs'/name;root=p/'artifacts'
    samples=[json.loads(l) for l in (root/'trajectories.jsonl').open()]
    req=[json.loads(l) for l in (root/'llm_requests.jsonl').open()]
    memory=json.loads((root/'experience_memory.json').read_text())
    records=[json.loads(l) for l in (root/'role_io.jsonl').open()]
    apps=[a for r in samples for a in r['applications']]+[a for r in samples for t in r['proposal_attempts'] for a in t['applications']]
    result={'summary':json.loads((p/'summary.json').read_text()),'states':dict(Counter(r['state'] for r in samples)),
      'requests':len(req),'requests_by_role':dict(Counter(r.get('role') for r in req)),
      'tokens':{k:sum(r.get('usage',{}).get(k,0) for r in req) for k in ['prompt_tokens','completion_tokens','total_tokens']},
      'source_feedback':dict(Counter(a['feedback'] for a in apps if a['phase']=='source_validation')),
      'downstream_feedback':dict(Counter(a['feedback'] for a in apps if a['phase']=='downstream')),
      'invalid_contracts':dict(Counter(r.get('role') for r in records if r.get('validation_status')=='invalid')),
      'judge_cache_hits':sum(r.get('event_type')=='judge_cache_hit' for r in records),
      'revisions':[{'parent':r['parent_id'],'accepted':r['accepted'],'action':r['action'],'pairs':len(r['replays']),'old_cache_hits':sum(x.get('old_cache_hit',False) for x in r['replays']),'decision':r['decision'],'error':r['error']} for r in memory['revisions']],
      'versions':len(memory['experiences']),'active':sum(e['status']=='active' for e in memory['experiences']),
      'sample_ids':[r['sample_id'] for r in samples]}
    if memory['schema_version']==15:
        parameters=ProbabilityModel.load(p/'probability_model.json');count=0
        for raw in memory['experiences']:
            e=Experience.from_dict(raw);expected=deepcopy(e);rebuild_quality(expected,parameters)
            assert abs((e.q_posterior or 0)-(expected.q_posterior or 0))<1e-12
            if e.q_log_odds is not None:assert abs(e.q_log_odds-expected.q_log_odds)<1e-12
            assert len([(a.sample.task_id) for a in e.applications if a.update_reason=='updated'])==len({a.sample.task_id for a in e.applications if a.update_reason=='updated'})
            count+=1
        result['quality_rebuild_verified_versions']=count
        attempts=[a for r in samples for a in r['proposal_attempts']]
        result['repairs']={'attempts':len(attempts),'effective_tasks':sum(any(a.get('margin_gain') is not None and a['margin_gain']>=.1-1e-12 for a in r['proposal_attempts']) for r in samples),
            'raw_judge_successful_attempts':sum(bool(a.get('repair_judgment') and a['repair_judgment']['success']) for a in attempts),
            'critic_final_successful_attempts':sum(bool(a.get('final_judgment') and a['final_judgment']['success']) for a in attempts),
            'extracted':sum(len(a.get('extractions',[])) for a in attempts),
            'review_calls':sum(len(e['reviews']) for a in attempts for e in a.get('extractions',[]))}
        assert all(r['snapshot_digest'] for r in samples)
        assert [r['completion_sequence'] for r in samples]==list(range(1,len(samples)+1))
    return result

if __name__=='__main__':
    names=['protocol100-baseline20-batch3','protocol100-role-guides20-stream5','protocol100-role-guides20-stream5-v2'];out=ROOT/'artifacts/protocol100-analysis';out.mkdir(exist_ok=True)
    results={n:analyze(n) for n in names}
    assert len({tuple(sorted(r['sample_ids'])) for r in results.values()})==1
    for n,file in zip(names,['baseline-timing.json','new-timing.json','new-v2-timing.json']):results[n]['timing']=json.loads((ROOT/'artifacts/protocol100-flow-data'/file).read_text())
    protected={}
    for file,h in json.loads((ROOT/'artifacts/protocol100-module-replay-v1/protected_hashes.json').read_text()).items():
        protected[file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h
    assert all(protected.values())
    artifact={'scope':'Same20 reused diagnostic training tasks; NOT held-out test accuracy; assumed tables uncalibrated; combination changes and schedule exposure confounded','runs':results,'original_hashes_unchanged':protected}
    (out/'summary.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    for n,r in results.items():print(n,round(r['timing']['seconds'],1),'sec',r['requests'],'requests',r['summary'],'revisions',len(r['revisions']))
