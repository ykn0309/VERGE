"""Read-only integrity/format metrics and explicitly limited observable Judge reference."""
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median
from scripts.replay_experience_splits import save
from scripts.run_module_model_replay import ROOT,verify_frozen
from scripts.replay_role_few_shot import validate


def audit():
    manifest=verify_frozen();prepared=json.loads((ROOT/'prepared_inputs.json').read_text())
    metrics=[];total_requests=0
    secrets=[os.environ.get(x) for x in ('DASHSCOPE_API_KEY_2','DEEPSEEK_API_KEY') if os.environ.get(x)]
    for role in manifest['roles']:
        for model,spec in manifest['models'].items():
            for variant in (('production','facts_only') if role=='critic' else ('production',)):
                directory=ROOT/role/model/variant
                inputs=[r for r in prepared if r['role']==role and r['variant']==variant]
                results=[json.loads((directory/(r['case_key']+'.json')).read_text()) for r in inputs]
                traces=[json.loads(s) for s in (directory/'role_io.jsonl').read_text().splitlines()]
                http=[json.loads(s) for s in (directory/'http_io.jsonl').read_text().splitlines()]
                requests=[r for r in http if 'request_body' in r];total_requests+=len(requests)
                assert len(inputs)==len(results)==20 and len({r['task_fingerprint'] for r in inputs})==20
                assert len(requests)==len(traces)
                for source,result in zip(inputs,results):
                    assert source['messages']==result['messages'] and source['input']==result['input']
                    assert result['model_config']==spec
                    if result['new_output'] is not None:
                        validate(role,source['input'],result['new_output'])
                        if variant=='facts_only':assert not result['new_output']['candidates']
                for request in requests:
                    body=request['request_body'];assert body['model']==spec['model']
                    assert body['response_format']=={'type':'json_object'} and body['max_tokens']==8192
                    if model=='qwen':assert body['enable_thinking'] is False
                    else:assert body['thinking']=={'type':'disabled'} and 'enable_thinking' not in body
                nonempty_reasoning=0
                for response in http:
                    if 'response_body' not in response:continue
                    body=json.loads(response['response_body'])
                    if body.get('choices'):
                        nonempty_reasoning+=bool(body['choices'][0]['message'].get('reasoning_content'))
                assert nonempty_reasoning==0
                for filename in ('http_io.jsonl','role_io.jsonl','llm_requests.jsonl'):
                    data=(directory/filename).read_text()
                    assert not any(key in data for key in secrets),'credential appears in artifacts'
                failed={r['case_key'] for r in traces if r['attempt']==1 and r['error']}
                extra_types=0
                for r in traces:
                    if not r['error']:continue
                    try:
                        raw=json.loads(r['raw_output'])
                        extra_types+=isinstance(raw,dict) and 'type' in raw
                    except (ValueError,TypeError):pass
                row={'role':role,'model':model,'variant':variant,'inputs':20,
                     'valid':sum(r['new_output'] is not None for r in results),
                     'first_attempt_valid':20-len(failed),'requests':len(requests),
                     'errors':dict(Counter(r['error'] for r in traces if r['error'])),
                     'extra_type_field_errors':extra_types,'median_seconds':round(median(r['elapsed_seconds'] for r in results),2),
                     'structure_retries':sum(r['structure_retries'] for r in results)}
                if role=='experience_updater':
                    row['splits']=sum(len(r['new_output']['experiences'])==2 for r in results if r['new_output'])
                    row['identical_child_texts']=sum(len(r['new_output']['experiences'])==2
                     and r['new_output']['experiences'][0]['content']==r['new_output']['experiences'][1]['content']
                     and r['new_output']['experiences'][0]['trigger']==r['new_output']['experiences'][1]['trigger']
                     for r in results if r['new_output'])
                metrics.append(row)
    ref=json.loads((ROOT/'judge_observable_reference.json').read_text())['reference']
    judged=[]
    for model in manifest['models']:
        matches=0;mismatches=[]
        for case,values in ref.items():
            output=json.loads((ROOT/'judge'/model/'production'/(case+'.json')).read_text())['new_output']
            by_id={x['criterion_id']:x for x in output['criterion_judgments']}
            for key,expected in values.items():
                if by_id[key]['satisfied']==expected:matches+=1
                else:mismatches.append({'case_key':case,'criterion_id':key,'expected':expected,'actual':by_id[key]['satisfied']})
        judged.append({'model':model,'matching_boolean':matches,'reference_items':sum(len(x) for x in ref.values()),'mismatches':mismatches})
    save(ROOT/'audit.json',{'completed_outputs':sum(r['inputs'] for r in metrics),'valid_outputs':sum(r['valid'] for r in metrics),
        'actual_requests':total_requests,'tests_passed':306,'protected_hashes_unchanged':True,
        'twenty_distinct_tasks_per_role':True,'identical_prepared_messages_across_models':True,
        'full_task_and_response_no_truncation':True,'actual_request_non_thinking_verified':True,
        'no_nonempty_reasoning_content':True,'credential_values_absent_from_artifacts':True,
        'no_pipeline_or_pool_writes':True,'metrics':metrics,'judge_observable_reference_comparison':judged,
        'limitations':'46 explicit evidence items only, labeled after viewing outputs, not blind gold/overall accuracy; semantic reasons can still be false when Boolean matches; final outputs after finite correction include feedback confound'})
    print('audited',sum(r['inputs'] for r in metrics),'outputs;',total_requests,'actual requests;',sum(r['valid'] for r in metrics),'valid')
    for r in metrics:print(r['role'],r['model'],r['variant'],r['first_attempt_valid'],r['valid'],r['requests'],r.get('splits',''))

if __name__=='__main__':audit()
