"""Frozen, paired first-pass evaluation; no training or model/quality updates."""
from __future__ import annotations
import argparse, hashlib, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.cli import components, code_version
from verge.config import load_config
from verge.dataset import JsonlDataset
from verge.memory import ExperienceMemory, atomic_json
from verge.pipeline import PipelineRunner
from verge.prompts import PROMPT_VERSION

CONTEXT = ContextVar('paired_evaluation_context', default={})

def evaluate(config_path, test_path, train_dir, output, workers=8):
    config=load_config(config_path)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=Path(train_dir)/'artifacts'/'experience_memory.json'
    original=source.read_bytes(); sha=hashlib.sha256(original).hexdigest()
    (out/'frozen_experience_memory.json').write_bytes(original)
    memory=ExperienceMemory(source.parent,read_only=True)
    pool=memory.all(active_only=True)
    samples=list(JsonlDataset({'test':test_path}).load('test'))
    train=list(JsonlDataset(config['dataset']['config']['paths']).load('train'))
    assert not {s.task_id for s in train}&{s.task_id for s in samples}
    lock=threading.Lock();local=threading.local();allmods=[]
    def append(name,event):
        with lock:
            with (out/name).open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    def hook(name):
        def save(event):append(name,{**CONTEXT.get(),**event})
        return save
    def attach(m):
        bs={id(x.backend):x.backend for x in (m.selector,m.initial_generator,m.judge,m.critic,m.experience_updater,m.experience_checker)}
        bs.update({id(b):b for b in m.checkers.backends.values()})
        bs[id(m.retriever.backend)]=m.retriever.backend
        for b in bs.values():b.trace_hook=hook('role_io.jsonl');b.metadata_hook=hook('llm_requests.jsonl')
    # Cache method vectors once; each worker has its own clients and mutable trace/cache.
    base=components(config);attach(base)
    for e in pool:base.retriever._vector(json.dumps(e.method.to_dict(),ensure_ascii=False,sort_keys=True))
    vectors=dict(base.retriever.cache);allmods.append(base)
    def one(sample,arm):
        if not hasattr(local,'mods'):
            local.mods=components(config);attach(local.mods)
            local.mods.retriever.cache.update(vectors)
            with lock:allmods.append(local.mods)
        m=local.mods; token=CONTEXT.set({'sample_id':sample.sample_id,'arm':arm,'split':'test'})
        start=time.time(); row={'sample_id':sample.sample_id,'task_id':sample.task_id,'arm':arm}
        try:
            runner=PipelineRunner(m,memory,None,training=False,revision_enabled=False)
            recalled,selected,rubric,judgment=runner.evaluate(sample,pool if arm=='experience' else [])
            row.update(valid=True,correct=judgment.success,rubric=rubric.to_dict(),judgment=judgment.to_dict(),recalled=[e.experience_id for e in recalled],selected=[e.experience_id for e in selected],retrieval_trace=m.retriever.last_trace)
        except Exception as exc:
            row.update(valid=False,correct=False,error={'type':type(exc).__name__,'message':str(exc)})
        finally:CONTEXT.reset(token)
        row['seconds']=time.time()-start
        append('results.jsonl',row)
        return row
    protocol={'state':'running','train_count':len(train),'test_count':len(samples),'active_experiences':len(pool),'memory_sha256':sha,'code_version':code_version(),'prompt_version':PROMPT_VERSION,'workers':workers,'thinking':False,'parameters':config['modules']['reliability']['config'],'assumed_parameters':True,'scope':'task-similarity-selected held-out General subset of original train; not official test or representative overall evaluation','criterion':'same tasks; first-pass accuracy; errors counted incorrect; no Critic/Checker/pool updates','test_sha256':hashlib.sha256(Path(test_path).read_bytes()).hexdigest()}
    atomic_json(out/'protocol.json',protocol)
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(one,s,arm) for s in samples for arm in ('baseline','experience')]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%20==0:print(f'evaluated {len(rows)}/{2*len(samples)}',flush=True)
            if len(rows)%10==0:atomic_json(out/'progress.json',{'completed':len(rows),'total':2*len(samples)})
    assert source.read_bytes()==original, 'frozen source changed'
    paired={s.sample_id:{} for s in samples}
    for row in rows:paired[row['sample_id']][row['arm']]=row
    n=len(samples)
    summary={'n':n,'active_experiences':len(pool)}
    for arm in ('baseline','experience'):
        rr=[r for r in rows if r['arm']==arm]
        summary[arm]={'correct':sum(r['correct'] for r in rr),'accuracy':sum(r['correct'] for r in rr)/n,'errors':sum(not r['valid'] for r in rr)}
    summary['improved']=sum(not p['baseline']['correct'] and p['experience']['correct'] for p in paired.values())
    summary['regressed']=sum(p['baseline']['correct'] and not p['experience']['correct'] for p in paired.values())
    summary['gain_pp']=100*(summary['experience']['accuracy']-summary['baseline']['accuracy'])
    summary['source_unchanged']=True
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'protocol.json',{**protocol,'state':'completed'})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    for m in allmods:
        bs={id(x.backend):x.backend for x in (m.selector,m.initial_generator,m.judge,m.critic,m.experience_updater,m.experience_checker)}
        bs.update({id(b):b for b in m.checkers.backends.values()});bs[id(m.retriever.backend)]=m.retriever.backend
        for b in bs.values():b.close()
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--test',required=True);p.add_argument('--train-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();evaluate(a.config,a.test,a.train_dir,a.output,a.workers)
