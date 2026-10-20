"""Fixed-input confirmation of the constrained Atomizer, no downstream feedback."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,time,shutil
from threading import Lock
from verge.models import AtomicMethod
from verge.modules import LLMAtomizer
from verge.llm import OpenAICompatibleBackend,LLMSampleError
from scripts.replay_versioned_evolution import SOURCE,DEST,write,sha

def main():
    root=DEST/'atomizer-confirmation'
    if root.exists():raise ValueError('confirmation already exists')
    root.mkdir();root.chmod(0o700)
    original=json.loads((DEST/'manifest.json').read_text())
    for p,h in original['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('source changed')
    inputs=[x for x in json.loads((DEST/'inputs.json').read_text()) if x['role']=='atomizer']
    manifest={'input_sha256':sha(DEST/'inputs.json'),'source_sha256':original['source_sha256'],
        'model':'qwen3.5-flash','enable_thinking':False,'cases':len(inputs),
        'code_sha256':{str(p):sha(p) for p in Path('verge').glob('*.py')},
        'changes':['Do not send prior text to Atomizer; origin flag only.',
            'One-item output preserves draft bytes programmatically; only multiple items can rewrite.',
            'Two original non-dataset examples demonstrate independent checks and inseparable condition/action.'],
        'budget_revision':'Initial160 + fixed40 confirmation = 200 effective target calls; previous 180 proposal revised after observed old-rule leakage. No full pipeline.'}
    write(root/'manifest.json',manifest)
    for source in manifest['code_sha256']:
        target=root/'code_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    lock=Lock()
    def run(case):
        def append(event):
            with lock:
                p=root/'role_io.jsonl'
                with p.open('a') as f:p.chmod(0o600);f.write(json.dumps({'case_key':case['key'],**event},ensure_ascii=False)+'\n')
        b=OpenAICompatibleBackend('qwen3.5-flash',base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key_env='DASHSCOPE_API_KEY_2',enable_thinking=False,timeout_seconds=120,max_retries=2,trace_hook=append)
        start=time.monotonic();out=error=None
        try:
            p=case['input'];out=LLMAtomizer(b).atomize(AtomicMethod.from_dict(p['draft']),previous=AtomicMethod.from_dict(p['previous']) if p.get('previous') else None).to_dict()
        except (ValueError,LLMSampleError) as e:error=type(e).__name__+': '+str(e)
        finally:b.close()
        row={**case,'output':out,'error':error,'elapsed_seconds':time.monotonic()-start}
        write(root/'results'/(case['key']+'.json'),row)
        print(case['key']+(' OK' if error is None else ' INVALID'),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as ex:rows=[f.result() for f in as_completed([ex.submit(run,c) for c in inputs])]
    for p,h in original['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('source changed')
    write(root/'summary.json',{'cases':len(rows),'valid':sum(r['error'] is None for r in rows),
         'single':sum(r['output'] is not None and len(r['output']['experiences'])==1 for r in rows),
         'split':sum(r['output'] is not None and len(r['output']['experiences'])>1 for r in rows),'source_unchanged':True})

if __name__=='__main__':main()
