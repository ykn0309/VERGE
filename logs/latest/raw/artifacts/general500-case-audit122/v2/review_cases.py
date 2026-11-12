"""Read-only auxiliary content audit. Original run and experience pool are immutable."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, json, os, threading, time
import httpx

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT/'cases.json').read_text())
PROMPT = (ROOT/'audit_prompt.txt').read_text()
KINDS=set('scope_issue added_obligation lost_requirement useful_check judge_reading_issue judge_rule_drift requires_external_fact uncertain'.split())
LOCK = threading.Lock()

def payload(case):
    swap=case['input_index'] % 2 == 0
    x_side,y_side=('rejected','chosen') if swap else ('chosen','rejected')
    refs={'TASK':case['task'],'X':case[x_side+'_answer'],'Y':case[y_side+'_answer']}
    aliases={e['experience_id']:f'E{i+1}' for i,e in enumerate(case['selected_experiences'])}
    for e in case['selected_experiences']:
        alias=aliases[e['experience_id']]
        for k,v in e['method'].items():refs[f'{alias}.{k}']=v
        refs[f'Select.{alias}']=e['selection_reason'] or ''
    paths={}
    for label,path in [('N',case['baseline']),('W',case['actual'])]:
        data={'provided':[aliases[x] for x in path['provided_experience_ids']],'criteria':[]}
        refs[f'{label}.analysis']=path['rubric']['analysis']
        judgments={x['criterion_id']:x for x in path['judgment']['criterion_judgments']}
        for i,c in enumerate(path['rubric']['criteria']):
            key=f'{label}{i+1}';refs[key]=c['question'];j=judgments[c['criterion_id']]
            for anon,side in [('X',x_side),('Y',y_side)]:refs[f'{key}.{anon}.reason']=j[side+'_reason']
            data['criteria'].append({'ref':key,'weight':path['judgment']['normalized_weights'][c['criterion_id']],'X_satisfied':j[x_side],'Y_satisfied':j[y_side]})
        paths[label]=data
    return {'case_index':case['input_index'],'experiences':list(aliases.values()),'paths':paths,'refs':refs}


def validate(value, packet):
    assert isinstance(value,dict) and set(value)=={'case_index','selection','observations','limit'},'expected complete fields'
    assert value['case_index']==packet['case_index'],'case_index mismatch'
    assert isinstance(value['limit'],str) and value['limit'].strip(),'limit missing'
    selection=value['selection'];assert isinstance(selection,list) and sorted(x.get('id','') for x in selection)==sorted(packet['experiences']),'selection entries invalid'
    for x in selection:
        assert x.get('fit') in {'supported','mixed','unsupported','uncertain'} and x.get('reason'),'selection status/reason invalid'
    observations=value['observations'];assert isinstance(observations,list) and len(observations)<=5,'observations invalid'
    for x in observations:assert x.get('kind') in KINDS and x.get('note'),'observation kind/note invalid'
    for item in selection+observations:
        refs=item.get('refs');assert isinstance(refs,list) and 1<=len(refs)<=3,'1..3 refs required'
        assert all(r in packet['refs'] for r in refs),'unknown ref; choose existing keys only'
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
            body={'model':'qwen3.5-flash','messages':messages,'enable_thinking':False,'temperature':0,'max_tokens':2500,'response_format':{'type':'json_object'}}
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
                if log.get('http_status')==200:messages=messages[:2]+[{'role':'user','content':'上次JSON合同校验失败：'+errors[-1]+'。请基于同一输入重新输出完整JSON，refs只选择输入已有的键。'}]
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
