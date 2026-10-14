"""Eight diagnostic Critic calls; isolate preference, Judge reasons and prior analysis."""
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.replay_experience_splits import digest, fingerprint, save
from scripts.replay_role_few_shot import validate
from scripts.replay_role_prompt_ablation import LABELS
from verge import prompts
from verge.llm import LLMFormatError, LLMRefusalError, OpenAICompatibleBackend


def without_reasons(value):
    if isinstance(value, dict):
        return {k: without_reasons(v) for k, v in value.items() if k not in ('chosen_reason', 'rejected_reason')}
    if isinstance(value, list):
        return [without_reasons(x) for x in value]
    return deepcopy(value)


def preference_messages(event, visible):
    payload = deepcopy(event['input'])
    for key in ('chosen_score', 'rejected_score', 'margin', 'success'):
        payload['current_judgment'].pop(key, None)
    with patch.dict(prompts._INPUT_TITLES, LABELS):
        messages = prompts.llm_messages('critic', payload)
    if visible:
        messages[0]['content'] += '\n\n回答A是偏好较高的chosen，回答B是偏好较低的rejected。'
    return payload, messages


def main():
    source = Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3').resolve()
    output = Path('runs/critic-evidence-ablation').resolve()
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
    for line, key in [(163, 'first-turn-reasons'), (724, 'video-reasons')]:
        event = events[line-1]
        for variant in ('reasons_present', 'reasons_absent'):
            payload = deepcopy(event['input']) if variant == 'reasons_present' else without_reasons(event['input'])
            prepared.append({'case_key':key, 'trace_line':line, 'variant':variant, 'input':payload,
                             'messages':prompts.llm_messages('critic',payload), 'historical_output':event['output']})
    event = events[166]
    for variant in ('prior_analysis_present', 'prior_analysis_masked'):
        messages = deepcopy(event['messages'])
        messages[0] = prompts.llm_messages('critic', event['input'])[0]
        if variant == 'prior_analysis_masked':
            old = json.loads(messages[2]['content'])
            old['analysis'] = '本次诊断对照不提供首轮analysis，首轮候选及实际应用反馈保留。'
            messages[2]['content'] = json.dumps(old, ensure_ascii=False)
        prepared.append({'case_key':'second-turn-history','trace_line':167,'variant':variant,'input':event['input'],
                         'messages':messages,'historical_output':event['output']})
    event = events[3]
    for visible in (True, False):
        payload,messages = preference_messages(event, visible)
        prepared.append({'case_key':'feelings-preference','trace_line':4,'variant':'preference_visible' if visible else 'preference_hidden',
                         'input':payload,'messages':messages,'historical_output':event['output']})
    assert len(prepared)==8
    if any(sum(len(m['content'].encode()) for m in r['messages'])>100_000 for r in prepared):
        raise ValueError('context limit; no truncation')
    output.mkdir(parents=True,exist_ok=False)
    output.chmod(0o700)
    save(output/'replay_manifest.json',{'model_config':spec,'source_sha256':hashes,'planned_calls':8,
        'prompt_version':prompts.PROMPT_VERSION,'script_sha256':digest(Path(__file__)),
        'production_prompts_sha256':digest(Path(prompts.__file__)), 'message_hashes':[fingerprint(r) for r in prepared],
        'scope':'fixed 4 paired diagnostics; existing bounded backend attempts; no pipeline or deployment',
        'conversion':'reason pair removes reasons only; history pair masks actual assistant analysis only; preference pair both remove summary scores and use A/B, then only one adds preference mapping',
        'limitations':'single task for preference/history; removing evidence changes information budget; cannot infer general prevalence or unique internal cause'})
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
            except (LLMFormatError,LLMRefusalError,ValueError) as exc:
                result,error=None,type(exc).__name__+': '+str(exc)
            row={**case,'new_output':result,'error':error,'elapsed_seconds':round(time.monotonic()-started,2)}
            save(output/f"{case['case_key']}_{case['variant']}.json",row)
            completed.append(row)
            print(f"{case['case_key']} {case['variant']}: {'valid' if result else 'invalid'}",flush=True)
    finally:
        backend.close()
        unchanged=all(digest(Path(p))==h for p,h in hashes.items())
        save(output/'progress.json',{'completed':len(completed),'source_unchanged':unchanged})
        if not unchanged:raise ValueError('source changed')
    save(output/'summary.json',{'completed':len(completed),'valid':sum(r['new_output'] is not None for r in completed),
                              'source_unchanged':True,'production_unchanged':digest(Path(prompts.__file__))==json.loads((output/'replay_manifest.json').read_text())['production_prompts_sha256']})


if __name__=='__main__':main()
