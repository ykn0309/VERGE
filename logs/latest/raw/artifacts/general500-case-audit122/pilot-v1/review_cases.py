"""Read-only auxiliary content audit. Original run and experience pool are immutable."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, json, os, threading, time
import httpx

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT/'cases.json').read_text())
PROMPT = (ROOT/'audit_prompt.txt').read_text()
TAGS = set('scope_mismatch extra_requirement missing_core_requirement premise_as_fact strategy_as_mandatory polarity_or_conditional task_faithful_check judge_answer_misread judge_rule_drift unverified_facts weight_or_granularity baseline_defect combination_interference uncertain'.split())
LOCK = threading.Lock()

def payload(case):
    refs={'task':case['task'],'chosen':case['chosen_answer'],'rejected':case['rejected_answer']}
    aliases={x['experience_id']:f'e{i+1}' for i,x in enumerate(case['selected_experiences'])}
    for e in case['selected_experiences']:
        a=aliases[e['experience_id']]
        for k,v in e['method'].items():refs[f'{a}.{k}']=v
        refs[f'select.{a}']=e['selection_reason'] or ''
    paths={}
    candidates=[('b',case['baseline']),('a',case['actual'])]+[(f's{i+1}',p) for i,p in enumerate(case['auxiliary_subsets'])]
    for label,p in candidates:
        data={'provided':[aliases[x] for x in p['provided_experience_ids']], 'error':p['error'],'criteria':[]}
        if p['rubric']:
            refs[f'{label}.analysis']=p['rubric']['analysis']
            judges={j['criterion_id']:j for j in (p.get('judgment') or {}).get('criterion_judgments',[])}
            for i,c in enumerate(p['rubric']['criteria']):
                ref=f'{label}{i+1}';refs[ref]=c['question'];j=judges.get(c['criterion_id']);entry={'ref':ref,'weight':c['weight']}
                if j:
                    entry.update({k:j[k] for k in ('chosen','rejected','weighted_preference_contribution')})
                    refs[f'{ref}.chosen_reason']=j['chosen_reason'];refs[f'{ref}.rejected_reason']=j['rejected_reason']
                data['criteria'].append(entry)
        if p['judgment']:data.update({k:p['judgment'][k] for k in ('chosen_score','rejected_score','margin','success')})
        paths[label]=data
    return {'sample_id':case['sample_id'],'input_index':case['input_index'],'observed_change':case['outcome'],'experience_aliases':aliases,'paths':paths,'refs':refs}


def validate(value, packet):
    assert isinstance(value,dict) and value.get('sample_id')==packet['sample_id'],'sample_id mismatch'
    assert set(value)=={'sample_id','summary','selection','rubric_comparison','findings','interpretation','cannot_determine'},'expected complete fields'
    assert all(isinstance(value[k],str) and value[k].strip() for k in ('summary','cannot_determine')),'summary/limit text missing'
    assert value['rubric_comparison'] in {'more_task_faithful','less_task_faithful','mixed','no_clear_difference','uncertain'},'rubric comparison invalid'
    assert value['interpretation'] in {'plausible_rubric_improvement','plausible_rubric_degradation','mixed','unresolved'},'interpretation invalid'
    sel=value['selection'];assert isinstance(sel,list) and sorted(x.get('id','') for x in sel)==sorted(packet['experience_aliases'].values()),'experience selection entries invalid'
    for x in sel:assert x.get('status') in {'supported','mixed','unsupported','uncertain'} and x.get('reason'),'selection status/reason invalid'
    fs=value['findings'];assert isinstance(fs,list) and len(fs)<=6,'findings invalid'
    for f in fs:
        assert f.get('tag') in TAGS and f.get('note'),'finding tag/note invalid'
        ev=f.get('evidence');assert isinstance(ev,list) and 1<=len(ev)<=3,'evidence required'
        for e in ev:
            assert e.get('ref') in packet['refs'],'unknown evidence ref'
            assert isinstance(e.get('quote'),str) and e['quote'].strip() and e['quote'] in packet['refs'][e['ref']],f"nonliteral quote at {e.get('ref')}"
    return value


def review(case):
    name=f"{case['input_index']:03d}-{case['sample_id'].split(':')[-1][:12]}"
    dest=ROOT/'reviews'/f'{name}.json'
    if dest.exists():
        v=json.loads(dest.read_text())
        if v.get('status')=='valid':return v
    packet=payload(case);(ROOT/'packets'/f'{name}.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    messages=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'))}]
    errors=[]
    with httpx.Client(timeout=httpx.Timeout(120),follow_redirects=False) as client:
        for attempt in range(1,4):
            body={'model':'qwen3.5-flash','messages':messages,'enable_thinking':False,'temperature':0,'max_tokens':4000,'response_format':{'type':'json_object'}}
            request_path=ROOT/'requests'/f'{name}-{attempt}.json';request_path.write_text(json.dumps(body,ensure_ascii=False)+'\n')
            start=time.time();log={'sample_id':case['sample_id'],'index':case['input_index'],'attempt':attempt,'started':start}
            try:
                response=client.post('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY_2']},json=body)
                log['http_status']=response.status_code
                raw=response.json();(ROOT/'responses'/f'{name}-{attempt}.json').write_text(json.dumps(raw,ensure_ascii=False)+'\n')
                log['usage']=raw.get('usage');log['actual_model']=raw.get('model')
                response.raise_for_status()
                content=raw['choices'][0]['message']['content'];value=validate(json.loads(content),packet)
                log['status']='valid';result={'status':'valid','input_index':case['input_index'],'outcome':case['outcome'],'attempt':attempt,'review':value}
                dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            except Exception as exc:
                log['status']='failed';log['error']=f'{type(exc).__name__}: {str(exc)[:250]}'
                # Raw bodies are preserved in files. Never print provider headers or keys.
                errors.append(log['error']);result=None
            log['seconds']=time.time()-start
            with LOCK:
                with (ROOT/'requests.jsonl').open('a') as f:f.write(json.dumps(log,ensure_ascii=False)+'\n')
            if result:return result
            if attempt<3:
                if log.get('http_status')==200:messages=messages[:2]+[{'role':'user','content':'上次输出合同校验失败：'+errors[-1]+'。请根据相同输入重新输出完整JSON，证据必须是对应refs中的逐字片段。'}]
                else:time.sleep(attempt)
    result={'status':'failed','input_index':case['input_index'],'outcome':case['outcome'],'sample_id':case['sample_id'],'errors':errors};dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    for folder in ('reviews','packets','requests','responses'):(ROOT/folder).mkdir(exist_ok=True)
    cases=CASES
    if args.pilot:
        cases = []
        for outcome in ('improved', 'regressed'):
            cases.extend([case for case in CASES if case['outcome'] == outcome][:3])
        assert len(cases) == 6
    original=json.loads((ROOT/'protocol.json').read_text());assert hashlib.sha256(Path(original['checkpoint']).read_bytes()).hexdigest()==original['checkpoint_sha256']
    with ThreadPoolExecutor(max_workers=5) as pool:
        for future in as_completed([pool.submit(review,c) for c in cases]):
            r=future.result();print(json.dumps({'index':r['input_index'],'outcome':r['outcome'],'status':r['status']},ensure_ascii=False),flush=True)
    assert hashlib.sha256(Path(original['checkpoint']).read_bytes()).hexdigest()==original['checkpoint_sha256']

if __name__=='__main__':main()
