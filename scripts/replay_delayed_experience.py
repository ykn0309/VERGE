"""Fixed historical Updater inputs with signed evidence; never run or mutate the pipeline.

Historical split output is parsed only by the baseline experiment adapter, not by production.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_module_model_replay import RecordedClient
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMRefusalError, _valid_shape
from verge.models import ExperienceUpdate
from verge.prompts import NumberedIDCodec, llm_messages, PROMPT_VERSION

SOURCE=Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3')
AUDIT=Path('runs/experience-confidence-audit-100')
OUT=Path('runs/delayed-experience-signed-replay-20')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    path.chmod(0o600)

def criterion_card(c):
    return {**{k:c[k] for k in ('question','chosen','rejected','chosen_reason','rejected_reason')},
            'feedback':'positive' if c['chosen']>c['rejected'] else 'negative' if c['chosen']<c['rejected'] else 'neutral'}

def from_case(case):
    verdicts={j['criterion_id']:j for j in case['judgment']['criterion_judgments']}
    return {'sample_id':case['sample_id'],**{k:case[k] for k in ('task','chosen','rejected','content','trigger')},
            'criteria':[criterion_card({**c,**verdicts[c['criterion_id']]}) for c in case['rubric']['criteria']
                        if case['source_id'] in c['source_experiences']]}

def make_payload(row,experience,order):
    """Use only cases available before this event, then add its actual signed R0 evidence."""
    prior=[from_case(c) for c in experience['cases'] if order[c['sample_id']]<row['sample_index']]
    current={'sample_id':row['sample_id'],**{k:row[k] for k in ('task','chosen','rejected','content','trigger')},
             'criteria':[criterion_card(c) for c in row['criteria']]}
    cards=prior+[current]
    if not 2<=len(cards)<=10: raise ValueError('replay requires actual origin and bounded history')
    for card in cards:
        content,trigger=card.pop('content'),card.pop('trigger')
        if (content,trigger)!=(row['content'],row['trigger']):
            card['applied_experience']={'content':content,'trigger':trigger}
    payload={'current_experience':{'content':row['content'],'trigger':row['trigger']},'cases':cards}
    return NumberedIDCodec('experience_updater',payload).project(payload)

def prepare(root):
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    trajectories=[json.loads(x) for x in (SOURCE/'artifacts/trajectories.jsonl').read_text().splitlines()]
    order={r['sample_id']:i+1 for i,r in enumerate(trajectories)}
    memory=json.loads((SOURCE/'artifacts/experience_memory.json').read_text())
    experiences={e['experience_id']:e for e in memory['experiences']}
    numeric=[e for e in json.loads((AUDIT/'numeric_events.json').read_text()) if e['evidence_type']=='downstream_credit']
    exposure=json.loads((AUDIT/'exposures.json').read_text())
    # Full responses for neutral uses come from the original prepared dataset, not generated summaries.
    manifest=json.loads((SOURCE/'run_manifest.json').read_text())
    data_path=Path(manifest['config']['dataset']['config']['paths']['train'])
    dataset={r['sample_id']:r for r in [json.loads(x) for x in data_path.read_text().splitlines()]}
    negative=[e for e in numeric if e['net_direction']=='negative' and e['sample_index'] in (29,52,61,64,71,72,81,92)]
    positive=[e for e in numeric if e['sample_index'] in (33,42,50,53,66,74,78,82)]
    neutral=[]
    for e in exposure:
        if e['neutral']:
            raw=dataset[e['sample_id']]
            neutral.append({**e,'chosen':raw['chosen'],'rejected':raw['rejected']})
    selected=[];seen=set()
    for group,quota,pool in [('negative',8,negative),('positive',8,positive),('neutral',4,neutral)]:
        count=0
        for row in pool:
            if row['task'] in seen:continue
            payload=make_payload(row,experiences[row['experience_id']],order)
            seen.add(row['task']);selected.append({'case_key':f'case_{len(selected)+1:02d}',
                'stratum':group,'source_event':{'sample_id':row['sample_id'],'sample_index':row['sample_index'],
                'experience_id':row['experience_id']},'input':payload})
            count+=1
            if count==quota:break
        if count!=quota:raise ValueError(f'insufficient unique tasks for {group}: {count}')
    baseline=json.loads(Path('/tmp/verge-plan61-baseline-system.json').read_text())
    spec=json.loads(Path('runs/module-prompt-replay-20/replay_manifest.json').read_text())['models']['qwen']
    paths=[SOURCE/'run_manifest.json',*(SOURCE/'artifacts'/n for n in ('trajectories.jsonl','experience_memory.json','role_io.jsonl')),
           AUDIT/'numeric_events.json',AUDIT/'exposures.json',data_path]
    save(root/'prepared_inputs.json',selected)
    save(root/'replay_manifest.json',{'model':spec,'prompt_version':PROMPT_VERSION,'baseline_system':baseline,
        'updated_system':llm_messages('experience_updater',{})[0],
        'source_sha256':{str(p):digest(p) for p in paths},'input_sha256':digest(root/'prepared_inputs.json'),
        'code_sha256':{str(p):digest(p) for p in [*Path('verge').glob('*.py'),Path(__file__)]},
        'planned_outputs':40,'unique_target_tasks':20,'strata':{'negative':8,'positive':8,'neutral':4},
        'conversion':'Historical finite cases before each target event plus actual R0 target; add feedback direction, drop numerical credit/weights, project short case IDs. Both variants receive identical complete inputs. Historical split baseline output adapter only in this script.',
        'limitations':'Historical per-experience windows, not the new three-task execution or saturation; no chained updates, no downstream effect or accuracy claim.'})

def verify(root):
    m=json.loads((root/'replay_manifest.json').read_text())
    assert digest(root/'prepared_inputs.json')==m['input_sha256']
    for name in ('source_sha256','code_sha256'):
        for path,value in m[name].items():
            if digest(Path(path))!=value:raise ValueError(f'protected file changed: {path}')
    return m

def validate_output(raw,payload,variant):
    if variant=='updated':return ExperienceUpdate.from_dict(raw).to_dict()
    if not isinstance(raw,dict) or set(raw)!={'experiences'} or len(raw['experiences']) not in (1,2):
        raise ValueError('historical baseline requires one or two experiences')
    ids=[]
    for draft in raw['experiences']:
        if set(draft)!={'content','trigger','sample_ids'} or not draft['sample_ids']:raise ValueError('invalid historical draft')
        ExperienceUpdate(draft['content'],draft['trigger'])
        ids.extend(str(v) if type(v) is int else v for v in draft['sample_ids'])
    expected={c['sample_id'] for c in payload['cases']}
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError('invalid historical partition')
    return raw

def historical_shape(role,value):
    if role!='experience_updater':return _valid_shape(role,value)
    try:
        return isinstance(value,dict) and set(value)=={'experiences'} and isinstance(value['experiences'],list)
    except (KeyError,TypeError):return False

def run_variant(root,variant):
    m=verify(root);spec=m['model'];directory=root/variant;directory.mkdir(exist_ok=True)
    directory.chmod(0o700);context={}
    def append(name,event):
        p=directory/name
        with p.open('a') as f:
            p.chmod(0o600);f.write(json.dumps({**context,**event},ensure_ascii=False,allow_nan=False)+'\n')
    client=RecordedClient(spec,append)
    backend=OpenAICompatibleBackend(spec['model'],**{k:spec[k] for k in ('base_url','api_key_env','timeout_seconds','max_retries','enable_thinking')},
        client=client,trace_hook=lambda e:append('role_io.jsonl',e),metadata_hook=lambda e:append('llm_requests.jsonl',e))
    output=[]
    try:
        for case in json.loads((root/'prepared_inputs.json').read_text()):
            path=directory/(case['case_key']+'.json')
            if path.exists():output.append(json.loads(path.read_text()));continue
            context.update(case_key=case['case_key'],variant=variant)
            messages=llm_messages('experience_updater',case['input']);messages[0]=m[variant+'_system']
            raw=None;error=None
            for attempt in range(3):
                try:
                    # A worker process isolates historical output parsing from current production.
                    with patch('verge.llm.llm_messages',lambda *args,**kw:deepcopy(messages)):
                        if variant=='baseline':
                            with patch('verge.llm._valid_shape',historical_shape):raw=backend.generate('experience_updater',case['input'])
                        else:raw=backend.generate('experience_updater',case['input'])
                    raw=validate_output(raw,case['input'],variant);error=None;break
                except (LLMFormatError,LLMRefusalError) as exc:error=str(exc);break
                except ValueError as exc:
                    error=str(exc)
                    if attempt<2:messages.extend([{'role':'assistant','content':json.dumps(raw,ensure_ascii=False)},
                        {'role':'user','content':'输出结构无效：'+error+'。请返回完整合法JSON。'}])
            row={**case,'messages':messages,'new_output':raw if error is None else None,'error':error,'structure_retries':attempt}
            save(path,row);output.append(row)
            print(f'{variant}: {len(output)}/20',flush=True)
    finally:backend.close();client.close()
    verify(root)
    return {'variant':variant,'completed':len(output),'valid':sum(r['new_output'] is not None for r in output)}

def run(root):
    # Processes isolate global prompt/shape adapters. Never patch concurrent threads.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    pool=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'))
    try:
        results=list(pool.map(run_variant,[root,root],['baseline','updated']))
    except BaseException:
        for process in list(pool._processes.values()):process.terminate()
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    verify(root);save(root/'summary.json',{'groups':results,'source_unchanged':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--root',type=Path,default=OUT)
    args=parser.parse_args();prepare(args.root) if args.mode=='prepare' else run(args.root)
