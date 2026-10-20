"""Twenty frozen historical Selector inputs, only the credit explanation changed."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from threading import Lock
from time import monotonic

from verge.llm import OpenAICompatibleBackend, LLMSampleError
from verge.models import ExperienceCategory
from verge.modules import LLMSelector
from verge.prompts import PROMPT_VERSION, ExperienceView


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    p.chmod(0o600)


def replay(source: Path, destination: Path):
    if destination.exists(): raise ValueError('destination already exists')
    rows=[r for r in json.loads((source/'inputs.json').read_text()) if r['role']=='selector']
    if len(rows)!=20 or len({r['input']['task'] for r in rows})!=20:
        raise ValueError('expected twenty task-deduplicated historical inputs')
    protected=[source/'inputs.json',source/'manifest.json',*(source/'results').glob('selector-*.json')]
    original_manifest=json.loads((source/'manifest.json').read_text())
    protected+=list(map(Path,original_manifest['source_sha256']))
    hashes={str(p):sha(p) for p in protected}
    for row in rows:
        row['previous_replay']=json.loads((source/'results'/(row['key']+'.json')).read_text())['output']
    write(destination/'inputs.json',rows)
    code=[*Path('verge').glob('*.py'),Path(__file__).resolve().relative_to(Path.cwd().resolve())]
    for p in code:
        target=destination/'code_snapshot'/p
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    spec={'model':'qwen3.5-flash','base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1',
          'api_key_env':'DASHSCOPE_API_KEY_2','enable_thinking':False,'timeout_seconds':120,'max_retries':2}
    write(destination/'manifest.json',{'source_sha256':hashes,'input_sha256':sha(destination/'inputs.json'),
        'code_sha256':{str(p):sha(p) for p in code},'model':spec,'prompt_version':PROMPT_VERSION,
        'target_calls':20,'pipeline_run':False,
        'limitations':['Candidates and reliability values are unchanged historical inputs, not newly evolved child versions.',
                       'Previous outputs are saved references, not a paired repeated-run causal comparison.']})
    lock=Lock()
    def one(row):
        def append(name,event):
            with lock:
                p=destination/name
                with p.open('a') as f:
                    p.chmod(0o600);f.write(json.dumps({'case_key':row['key'],**event},ensure_ascii=False)+'\n')
        backend=OpenAICompatibleBackend(**spec,trace_hook=lambda e:append('role_io.jsonl',e),
                                        metadata_hook=lambda e:append('llm_requests.jsonl',e))
        p=row['input']; views=[];start=monotonic();output=None;error=None
        try:
            for unit in p['candidates']:
                category=ExperienceCategory(unit['category_id'],unit['name'],unit['description']) if 'members' in unit else None
                views.extend(ExperienceView(x['experience_id'],x['content'],x['trigger'],x['confidence'],x['evidence_amount'],category)
                             for x in unit.get('members',[unit]))
            selected=LLMSelector(backend,max_selected=p['max_selected']).select(p['task'],views,training=p.get('training',True))
            output={'selected_experience_ids':[v.experience_id for v in selected]}
        except (ValueError,LLMSampleError) as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:backend.close()
        result={**row,'output':output,'error':error,'elapsed_seconds':round(monotonic()-start,2)}
        write(destination/'results'/(row['key']+'.json'),result)
        print(row['key']+(' OK' if error is None else ' INVALID'),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as executor: results=list(executor.map(one,rows))
    if hashes!={p:sha(Path(p)) for p in hashes}:raise ValueError('source artifacts changed')
    metadata=[json.loads(line) for line in (destination/'llm_requests.jsonl').read_text().splitlines()]
    summary={'cases':len(results),'valid':sum(r['error'] is None for r in results),
             'changed_selection_sets':sum(r['error'] is None and set(r['output']['selected_experience_ids'])!=set(r['previous_replay']['selected_experience_ids']) for r in results),
             'http_requests':len(metadata),'source_unchanged':True,'pipeline_run':False}
    write(destination/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('runs/versioned-evolution-module-replay-67'))
    parser.add_argument('--destination',type=Path,default=Path('runs/inherited-credit-selector-replay-68'))
    args=parser.parse_args();print(json.dumps(replay(args.source,args.destination),ensure_ascii=False,indent=2))
