"""Fixed v3.1 run plus a read-only, same-draft no-experience test control."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import hashlib,json,time,traceback
from verge.cli import run,Components,JSONLogs,CURRENT_SAMPLE
from verge.config import load_config
from verge.memory import ExperienceMemory,atomic_json
from verge.models import TrajectoryRecord,ExecutionPath

CONFIG='configs/reference127.yaml'
ROOT=Path('runs/reference127')
ART=Path('artifacts/reference127')

def main():
    manifest=run(CONFIG,resume=(ROOT/"manifest.json").exists())
    state=ExperienceMemory.load(ROOT/'checkpoint.json')
    before=hashlib.sha256((ROOT/'checkpoint.json').read_bytes()).hexdigest()
    rows=[TrajectoryRecord.from_dict({k:v for k,v in json.loads(line).items() if k!="sample_id"}) for line in (ROOT/'test.jsonl').read_text().splitlines()]
    controls=ROOT/'test_controls';controls.mkdir()
    logs=JSONLogs(controls);components=Components(load_config(CONFIG),logs)
    atomic_json(controls/'manifest.json',{'status':'running','frozen_sha256':before,'tasks':len(rows)})
    def evaluate(record):
        CURRENT_SAMPLE.set(record.sample.sample_id)
        if record.draft is None:
            baseline=ExecutionPath([],error=record.error or 'missing draft')
        elif record.paths and not record.paths[0].provided_experience_ids:
            baseline=record.paths[0]
        else:
            baseline=components.runner(record.sample).execute(record.sample,record.draft,[])
        actual=record.prediction_path
        result={'sample_id':record.sample.sample_id,'baseline':baseline.to_dict(),
                'actual_correct':actual.correct if actual else None,'baseline_correct':baseline.correct,
                'reused_actual':bool(actual and not actual.provided_experience_ids)}
        logs.append('paired.jsonl',result)
        return result
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(copy_context().run,evaluate,r) for r in rows]
            pairs=[f.result() for f in futures]
        assert hashlib.sha256((ROOT/'checkpoint.json').read_bytes()).hexdigest()==before
        outcomes={'improved':0,'regressed':0,'both_correct':0,'both_wrong':0,'unknown':0}
        for p in pairs:
            a,b=p['actual_correct'],p['baseline_correct']
            name='unknown' if a is None or b is None else 'both_correct' if a and b else 'improved' if a else 'regressed' if b else 'both_wrong'
            outcomes[name]+=1
        summary={'status':'completed','training_tasks':len(state.trajectories),'pool':len(state.experiences),'test':manifest['test'],'test_pairs':outcomes,'frozen_sha256':before,'finished':time.time()}
        atomic_json(controls/'manifest.json',summary);atomic_json(ART/'result.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:components.close()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(ART/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise
