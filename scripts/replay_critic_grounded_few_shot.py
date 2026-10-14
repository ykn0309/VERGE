"""Twelve fixed Critic outputs; original examples vs exact production messages."""
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.replay_experience_splits import digest, fingerprint, save
from scripts.replay_role_few_shot import example_suffix, validate
from verge import prompts
from verge.llm import LLMFormatError, LLMRefusalError, OpenAICompatibleBackend


def paired_messages(event, first, suffix):
    messages = prompts.llm_messages('critic', first['input'])
    if len(event['messages']) == 4:
        messages.append(deepcopy(event['messages'][2]))
        messages = prompts.llm_messages('critic', event['input'], history=messages)
    elif len(event['messages']) != 2:
        raise ValueError('requires first or actual second business turn')
    variant = deepcopy(messages)
    variant[0]['content'] += '\n\n' + suffix
    return messages, variant


def main():
    source = Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3').resolve()
    output = Path('runs/critic-grounded-few-shot/results').resolve()
    examples_path = output.parent / 'examples.json'
    examples = json.loads(examples_path.read_text())
    if len(examples)!=2: raise ValueError('exactly two original examples')
    suffix = example_suffix('critic',examples)
    manifest_path = source / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    root = source / manifest['config']['store']['config']['root_dir']
    traces = root / 'role_io.jsonl'
    events = [json.loads(x) for x in traces.read_text().splitlines()]
    spec = manifest['config']['models']['default']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('fixed Flash non-thinking only')
    protected = [manifest_path,traces,root/'experience_memory.json',root/'trajectories.jsonl',
                 examples_path,Path(prompts.__file__),Path(__file__)]
    hashes = {str(p):digest(p) for p in protected}
    prepared = []
    for line,key in [(163,'first-content'),(167,'second-history'),(724,'video'),
                     (4,'feelings'),(390,'story'),(281,'tent-control')]:
        event = events[line-1]
        if event['role']!='critic' or event['error'] is not None or event['output'] is None:
            raise ValueError('requires valid historical Critic input')
        first=event
        if len(event['messages'])==4:
            matches=[e for e in events[:line-1] if e['role']=='critic' and e['sample_id']==event['sample_id']
                     and e['messages']==event['messages'][:2]]
            if len(matches)!=1: raise ValueError('exact first-turn history ambiguous')
            first=matches[0]
        baseline,few_shot=paired_messages(event,first,suffix)
        assert baseline[1:]==few_shot[1:]
        for variant,messages in [('baseline',baseline),('few_shot',few_shot)]:
            prepared.append({'case_key':key,'trace_line':line,'sample_id':event['sample_id'],
                'variant':variant,'input':event['input'],'historical_output':event['output'],'messages':messages})
    if len(prepared)!=12 or any(sum(len(m['content'].encode()) for m in r['messages'])>100_000 for r in prepared):
        raise ValueError('bounded selection/context limit; no truncation')
    output.mkdir(parents=True,exist_ok=False)
    output.chmod(0o700)
    save(output/'replay_manifest.json',{'model_config':spec,'source_sha256':hashes,'planned_outputs':12,
        'prompt_version':prompts.PROMPT_VERSION,'examples':examples,'message_hashes':[fingerprint(r) for r in prepared],
        'scope':'Critic only; 6 inputs/5 tasks; two versions, existing backend up to 3 finite attempts/output',
        'conversion':'render recorded payload with exact current production layout, no experimental legend; actual first assistant verbatim; only system few-shot suffix differs',
        'evaluation':'task-grounded constraints, truthful diagnosis, reusable public method and advance trigger, normal behavior retained; empty candidate is not automatically correct',
        'limitations':'one output per version/input; second turn remains original actual assistant, not replay first output; no overall accuracy or causal claim'})
    save(output/'prepared_inputs.json',prepared)
    context={}
    def append(name,event):
        path=output/name
        with path.open('a',encoding='utf-8') as f:
            path.chmod(0o600)
            f.write(json.dumps({**context,**event},ensure_ascii=False,allow_nan=False)+'\n')
    backend=OpenAICompatibleBackend(spec['model'],**{k:spec[k] for k in
        ('base_url','api_key_env','timeout_seconds','max_retries','enable_thinking') if k in spec},
        trace_hook=lambda e:append('role_io.jsonl',e),metadata_hook=lambda e:append('llm_requests.jsonl',e))
    completed=[]
    try:
        for case in prepared:
            context.update(case_key=case['case_key'],variant=case['variant'])
            started=time.monotonic()
            result,error=None,None
            try:
                with patch('verge.llm.llm_messages',lambda *a,**kw:deepcopy(case['messages'])):
                    result=backend.generate('critic',case['input'])
                validate('critic',case['input'],result)
            except (LLMFormatError,LLMRefusalError) as exc:
                result,error=None,type(exc).__name__+': '+str(exc)
            row={**case,'new_output':result,'error':error,'elapsed_seconds':round(time.monotonic()-started,2)}
            save(output/f"{case['case_key']}_{case['variant']}.json",row)
            completed.append(row)
            print(f"{len(completed)}/12 {case['case_key']} {case['variant']}: {'valid' if result else 'invalid'}",flush=True)
    finally:
        backend.close()
        unchanged=all(digest(Path(p))==h for p,h in hashes.items())
        save(output/'progress.json',{'completed':len(completed),'protected_files_unchanged':unchanged})
        if not unchanged: raise ValueError('protected source/code/examples changed')
    save(output/'summary.json',{'completed':len(completed),'valid':sum(r['new_output'] is not None for r in completed),
                              'protected_files_unchanged':True})


if __name__=='__main__':main()
