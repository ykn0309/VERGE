"""Fixed Critic evidence: reason placement and independent 2x2 label factors."""
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.replay_experience_splits import digest, fingerprint, save
from scripts.replay_role_few_shot import validate
from verge import prompts
from verge.llm import LLMFormatError, LLMRefusalError, OpenAICompatibleBackend

VARIANTS = {
    'words_inline': (False, False, 'inline'),
    'words_separate': (False, False, 'separate'),
    'words_absent': (False, False, 'absent'),
    'ab_words': (True, False, 'inline'),
    'preferred_boolean': (False, True, 'inline'),
    'ab_boolean': (True, True, 'inline'),
}
# Identical legend across all variants controls its presence, including second-turn history.
LEGEND = ('编号说明：回答A=优选回答（chosen），回答B=非优选回答（rejected）；'
          'Judge判定true=满足本条标准，false=未满足本条标准。偏好身份与逐项是否满足是不同信息。')


def messages_for(event, first, variant):
    labels, boolean, reasons = VARIANTS[variant]
    renderer = prompts._natural_input
    def display(role, payload):
        return renderer(role, payload, critic_labels=labels, critic_boolean=boolean,
                        critic_reasons=reasons)
    with patch.object(prompts, '_natural_input', display):
        messages = prompts.llm_messages('critic', first['input'])
        if len(event['messages']) == 4:
            messages.append(deepcopy(event['messages'][2]))
            messages = prompts.llm_messages('critic', event['input'], history=messages)
        elif len(event['messages']) != 2:
            raise ValueError('only actual first/second turns supported')
    messages[0]['content'] += '\n\n' + LEGEND
    return messages


def main():
    source = Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3').resolve()
    output = Path('runs/critic-readable-input-replay').resolve()
    manifest_path = source / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    root = source / manifest['config']['store']['config']['root_dir']
    traces = root / 'role_io.jsonl'
    events = [json.loads(x) for x in traces.read_text().splitlines()]
    spec = manifest['config']['models']['default']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('fixed Flash non-thinking only')
    hashes = {str(p): digest(p) for p in [manifest_path, traces, root/'experience_memory.json', root/'trajectories.jsonl']}
    prepared = []
    for line, key in [(163,'first-content'),(167,'second-history'),(724,'video'),
                      (4,'feelings'),(390,'story'),(281,'tent-control')]:
        event = events[line-1]
        if event['role'] != 'critic' or event['error'] is not None or event['output'] is None:
            raise ValueError('requires valid historical Critic call')
        first = event
        if len(event['messages']) == 4:
            matches = [e for e in events[:line-1] if e['role']=='critic' and e['sample_id']==event['sample_id']
                       and e['messages']==event['messages'][:2]]
            if len(matches)!=1: raise ValueError('first-turn exact history source ambiguous')
            first=matches[0]
        for variant in VARIANTS:
            prepared.append({'case_key':key,'trace_line':line,'sample_id':event['sample_id'],
                'variant':variant,'input':event['input'],'historical_output':event['output'],
                'messages':messages_for(event,first,variant)})
    assert len(prepared)==36
    if any(sum(len(m['content'].encode()) for m in r['messages'])>100_000 for r in prepared):
        raise ValueError('context limit; no truncation')
    output.mkdir(parents=True,exist_ok=False)
    output.chmod(0o700)
    save(output/'replay_manifest.json',{'model_config':spec,'source_sha256':hashes,'planned_outputs':36,
        'prompt_version':prompts.PROMPT_VERSION,'script_sha256':digest(Path(__file__)),
        'production_prompts_sha256':digest(Path(prompts.__file__)),'variants':VARIANTS,'legend':LEGEND,
        'message_hashes':[fingerprint(r) for r in prepared],
        'scope':'Critic only, 5 tasks/6 inputs; at most 3 backend format/transport attempts per output',
        'conversion':'regenerate evidence presentation from full recorded input; keep actual assistant verbatim; common legend; reasons removed only in absent diagnostic; IDs unchanged',
        'limitations':'one output per input/variant; second-turn retained assistant uses original labels; reason omission changes available evidence; common legend differs from production'})
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
            print(f"{len(completed)}/36 {case['case_key']} {case['variant']}: {'valid' if result else 'invalid'}",flush=True)
    finally:
        backend.close()
        unchanged=all(digest(Path(p))==h for p,h in hashes.items())
        save(output/'progress.json',{'completed':len(completed),'source_unchanged':unchanged})
        if not unchanged:raise ValueError('source changed')
    save(output/'summary.json',{'completed':len(completed),'valid':sum(r['new_output'] is not None for r in completed),
                              'source_unchanged':True})


if __name__=='__main__':main()
