"""Audit frozen prompt comparisons, raw HTTP settings, and limited Judge references."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from scripts.replay_experience_splits import save
from scripts.replay_role_few_shot import validate
from scripts.run_module_prompt_replay import verify

def audit(root):
    manifest=verify(root)
    prepared=json.loads((root/'prepared_inputs.json').read_text())
    metrics=[];request_total=0
    secrets=[v for k in ('DASHSCOPE_API_KEY_2','DEEPSEEK_API_KEY') if (v:=os.environ.get(k))]
    reference_path=Path('runs/module-model-replay-20/judge_observable_reference.json') if manifest['stage']=='screening' else Path('runs/module-prompt-replay-20/confirmation_judge_reference.json')
    reference=json.loads(reference_path.read_text())['reference']
    reference_count=sum(len(v) for v in reference.values())
    for role in manifest['roles']:
        for variant in manifest['variants'][role]:
            directory=root/role/'qwen'/variant
            inputs=[r for r in prepared if r['role']==role and r['variant']==variant]
            results=[json.loads((directory/(r['case_key']+'.json')).read_text()) for r in inputs]
            traces=[json.loads(s) for s in (directory/'role_io.jsonl').read_text().splitlines()]
            http=[json.loads(s) for s in (directory/'http_io.jsonl').read_text().splitlines()]
            requests=[r for r in http if 'request_body' in r];request_total+=len(requests)
            for source,result in zip(inputs,results):
                assert result['messages']==source['messages'] and result['input']==source['input']
                assert result['model_config']==manifest['models']['qwen']
                calls=[r for r in requests if r['case_key']==source['case_key']]
                assert calls and calls[0]['request_body']['messages']==source['messages']
                if result['new_output'] is not None: validate(role,source['input'],result['new_output'])
            for request in requests:
                body=request['request_body']
                assert body['model']=='qwen3.5-flash' and body['enable_thinking'] is False
                assert body['max_tokens']==8192 and body['response_format']=={'type':'json_object'}
            for response in http:
                if 'response_body' in response:
                    value=json.loads(response['response_body'])
                    if value.get('choices'): assert not value['choices'][0]['message'].get('reasoning_content')
            for filename in ('http_io.jsonl','role_io.jsonl','llm_requests.jsonl'):
                content=(directory/filename).read_text()
                assert not any(secret in content for secret in secrets)
            failed={r['case_key'] for r in traces if r['attempt']==1 and r['error']}
            row={'role':role,'variant':variant,'inputs':len(inputs),'valid':sum(r['new_output'] is not None for r in results),
                 'requests':len(requests),'first_format_valid':len(inputs)-len(failed),
                 'structure_retries':sum(r['structure_retries'] for r in results),
                 'errors':dict(Counter(r['error'] for r in traces if r['error']))}
            if role=='experience_updater':
                row['splits']=sum(len(r['new_output']['experiences'])==2 for r in results if r['new_output'])
                row['identical_children']=sum(len(r['new_output']['experiences'])==2 and
                    len({(e['content'],e['trigger']) for e in r['new_output']['experiences']})==1 for r in results if r['new_output'])
            if role=='critic': row['empty_candidates']=sum(not r['new_output']['candidates'] for r in results if r['new_output'])
            if role=='judge':
                matches=0;mismatches=[]
                for result in results:
                    if not result['new_output']: continue
                    values={r['criterion_id']:r['satisfied'] for r in result['new_output']['criterion_judgments']}
                    for key,expected in reference[result['case_key']].items():
                        if values[key]==expected:matches+=1
                        else:mismatches.append({'case_key':result['case_key'],'criterion_id':key,'expected':expected,'actual':values[key]})
                row.update(reference_matches=matches,reference_items=reference_count,mismatches=mismatches)
            metrics.append(row)
    # Each case sees identical full evidence in every version; only its system instruction differs.
    for role in manifest['roles']:
        baseline=[r for r in prepared if r['role']==role and r['variant']=='baseline']
        for original in baseline:
            other=[r for r in prepared if r['role']==role and r['case_key']==original['case_key']]
            assert all(r['input']==original['input'] and r['messages'][1:]==original['messages'][1:] for r in other)
    result={'planned_outputs':manifest['planned_outputs'],'completed_outputs':sum(r['inputs'] for r in metrics),
            'valid_outputs':sum(r['valid'] for r in metrics),'actual_requests':request_total,'metrics':metrics,
            'frozen_source_code_and_inputs_unchanged':True,'system_only_prompt_difference':True,
            'actual_non_thinking_verified':True,'reasoning_content_empty':True,'credentials_absent':True,
            'limitations':'Not end-to-end or universal optimum; finite correction feedback differs by output; partial Judge references only: screening46 previously audited, confirmation40 labeled before calls, not full accuracy.'}
    save(root/'audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('runs/module-prompt-replay-20'))
    audit(parser.parse_args().root)
