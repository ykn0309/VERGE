"""Isolated process workers for frozen module/model replays; no pipeline state."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch
import httpx
from scripts.replay_experience_splits import digest,save
from scripts.replay_role_few_shot import validate
from verge.llm import LLMFormatError,LLMRefusalError,OpenAICompatibleBackend

ROOT=Path('runs/module-model-replay-20')

class RecordedClient(httpx.Client):
    def __init__(self,spec,append):
        super().__init__(base_url=spec['base_url'].rstrip('/')+'/',timeout=spec['timeout_seconds'],follow_redirects=False)
        self.extras=deepcopy(spec.get('request_extras',{}));self.record=append
    def post(self,url,*,headers,json):
        body={**deepcopy(json),**deepcopy(self.extras)}
        # No authentication headers or key values enter artifacts.
        self.record('http_io.jsonl',{'request_body':body})
        response=super().post(url,headers=headers,json=body)
        self.record('http_io.jsonl',{'status':response.status_code,'response_body':response.text})
        return response


def run_group(job):
    role,model_key,variant=job
    manifest=json.loads((ROOT/'replay_manifest.json').read_text())
    spec=manifest['models'][model_key]
    prepared=[r for r in json.loads((ROOT/'prepared_inputs.json').read_text()) if r['role']==role and r['variant']==variant]
    output=ROOT/role/model_key/variant;output.mkdir(parents=True,exist_ok=True);output.chmod(0o700)
    context={'role':role,'model_key':model_key,'variant':variant}
    def append(name,event):
        path=output/name
        with path.open('a',encoding='utf-8') as f:
            path.chmod(0o600);f.write(json.dumps({**context,**event},ensure_ascii=False,allow_nan=False)+'\n')
    client=RecordedClient(spec,append)
    backend=OpenAICompatibleBackend(spec['model'],**{k:spec[k] for k in
        ('base_url','api_key_env','timeout_seconds','max_retries','enable_thinking')},client=client,
        trace_hook=lambda e:append('role_io.jsonl',e),metadata_hook=lambda e:append('llm_requests.jsonl',e))
    results=[]
    try:
        for case in prepared:
            context['case_key']=case['case_key']
            path=output/(case['case_key']+'.json')
            if path.exists():
                row=json.loads(path.read_text())
                if row['messages']!=case['messages'] or row['input']!=case['input'] or row['model_config']!=spec:
                    raise ValueError('resume input/model mismatch')
                results.append(row);continue
            messages=deepcopy(case['messages']);started=time.monotonic();result,error=None,None
            for structure_attempt in range(3):
                try:
                    # Each worker is a separate process; this adapter cannot alter other workers or production.
                    with patch('verge.llm.llm_messages',lambda *a,**kw:deepcopy(messages)):
                        raw=backend.generate(role,case['input'])
                    validate(role,case['input'],raw)
                    if variant=='facts_only' and raw['candidates']:
                        raise ValueError('facts-only diagnostic requires empty candidates')
                    result=raw;error=None;break
                except (LLMFormatError,LLMRefusalError) as exc:
                    error=type(exc).__name__+': '+str(exc);break
                except ValueError as exc:
                    error=str(exc)
                    if structure_attempt==2:break
                    messages.extend([{'role':'assistant','content':json.dumps(raw,ensure_ascii=False)},
                        {'role':'user','content':'输出结构校验失败：'+error+'。请返回完整合法JSON结果。'}])
            row={**case,'model_key':model_key,'model_config':spec,'new_output':result,'error':error,
                 'structure_retries':structure_attempt,'elapsed_seconds':round(time.monotonic()-started,2)}
            save(path,row);results.append(row)
            save(output/'progress.json',{'completed':len(results),'valid':sum(x['new_output'] is not None for x in results)})
            print(f'{role}/{model_key}/{variant}: {len(results)}/20',flush=True)
    finally:
        backend.close();client.close()
    summary={'role':role,'model_key':model_key,'variant':variant,'completed':len(results),
             'valid':sum(x['new_output'] is not None for x in results),'structure_retries':sum(x['structure_retries'] for x in results)}
    save(output/'summary.json',summary)
    return summary


def verify_frozen():
    m=json.loads((ROOT/'replay_manifest.json').read_text())
    if digest(ROOT/'selection.json')!=m['selection_sha256'] or digest(ROOT/'prepared_inputs.json')!=m['input_sha256']:
        raise ValueError('frozen selection/input changed')
    if any(digest(Path(p))!=h for p,h in m['source_sha256'].items()):
        raise ValueError('protected source/code changed')
    return m


def main():
    manifest=verify_frozen()
    jobs=[(role,model,variant) for role in manifest['roles'] for model in manifest['models']
          for variant in (('production','facts_only') if role=='critic' else ('production',))]
    completed=[]
    pool=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))
    try:
        futures={pool.submit(run_group,j):j for j in jobs}
        for future in as_completed(futures):
            completed.append(future.result())
            save(ROOT/'progress.json',{'completed_groups':len(completed),'planned_groups':len(jobs),'groups':completed})
    except BaseException:
        for future in futures:future.cancel()
        # Stop other API work when any infrastructure/program failure aborts the experiment.
        for process in list(pool._processes.values()):process.terminate()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        verify_frozen()
    save(ROOT/'summary.json',{'groups':completed,'completed_outputs':sum(x['completed'] for x in completed),
          'valid_outputs':sum(x['valid'] for x in completed),'source_unchanged':True,'planned_outputs':manifest['planned_outputs']})
    print('all240 planned module/model outputs completed',flush=True)

if __name__=='__main__':main()
