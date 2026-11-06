"""Blinded semantic screening of isolated outputs; model reports are not truth."""
from __future__ import annotations
import json,sys,random,threading,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verge.cli import backend
from verge.config import load_config
import verge.llm as llm
ROOT=Path('artifacts/generator-application-94')
GUIDANCE='''你是评分标准内容审阅员。只看task、候选方法和匿名rubric，不看偏好回答或组别。对每份rubric分别审查，不进行组间排名。task中的用户要求是依据，必要的隐含正确性也可以核验，不能因为task没逐字写就标错。不要把风格偏好当缺陷，不要求强制引用经验。检查是否新增无依据义务(extra_obligation)、遗漏核心任务(core_omission)、误解对象/背景/事实(task_error)、true方向反转(polarity_error)、存在正确且有价值的方法应用(useful_use)。每项只能yes/no/unknown。无法核实的事实、数量和语义保留unknown，不能臆测。只返回JSON：{"reviews":[{"id":"给定匿名ID","extra_obligation":"no","core_omission":"no","task_error":"unknown","polarity_error":"no","useful_use":"yes","evidence":["具体标准与task依据，含不能确定处"]}]}。这些示意值须替换。每个输入ID恰好一条。'''
def messages(role,payload,**kwargs):return [{'role':'system','content':GUIDANCE},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
llm.llm_messages=messages

def main():
 cases=[c for c in json.loads(Path('artifacts/experience-utilization-audit/fixed_inputs.json').read_text()) if c['split']=='development' and c['role']=='initial_generator']
 rows=[json.loads(l) for l in (ROOT/'development/results.jsonl').read_text().splitlines()]
 if len(rows)!=120:raise RuntimeError('generation incomplete')
 out=ROOT/'semantic-screen';out.mkdir(exist_ok=False);lock=threading.Lock()
 def append(name,row):
  with lock:
   with (out/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 mappings=[];jobs=[]
 for c in cases:
  subset=[r for r in rows if r['sample_id']==c['sample_id'] and r['valid']];random.Random(c['sample_id']+'94').shuffle(subset)
  rubrics=[]
  for i,r in enumerate(subset):
   label=f'R{i+1}';mappings.append({'sample_id':c['sample_id'],'id':label,'variant':r['variant'],'repetition':r['repetition']});rubrics.append({'id':label,'criteria':r['output']['criteria']})
  jobs.append((c['sample_id'],{'task':c['input']['task'],'experiences':c['input']['experiences'],'rubrics':rubrics}))
 (out/'mapping.json').write_text(json.dumps(mappings,indent=2))
 spec=load_config('configs/midterm-general-v2-assumed-train-300-batch3.yaml')['models']['default']
 (out/'protocol.json').write_text(json.dumps({'guidance':GUIDANCE,'model':spec,'blinded':'arm, repetition, preference labels hidden; seeded anonymous ordering','limitation':'single Qwen same family, not independent truth; root reviews findings','input_results_sha256':hashlib.sha256((ROOT/'development/results.jsonl').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
 def one(job):
  sid,payload=job;b=backend(spec);b.trace_hook=lambda e:append('role_io.jsonl',{'sample_id':sid,**e});b.metadata_hook=lambda e:append('llm_requests.jsonl',{'sample_id':sid,**e})
  try:
   r=b.generate('critic',payload)
   assert {x['id'] for x in r['reviews']}=={x['id'] for x in payload['rubrics']}
   assert len(r['reviews'])==len(payload['rubrics'])
   for x in r['reviews']:
    for k in ['extra_obligation','core_omission','task_error','polarity_error','useful_use']:assert x[k] in ['yes','no','unknown']
   append('results.jsonl',{'sample_id':sid,'valid':True,**r})
  except Exception as e:append('results.jsonl',{'sample_id':sid,'valid':False,'error':str(e)})
  finally:b.close()
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,f in enumerate(as_completed([pool.submit(one,j) for j in jobs]),1):f.result();print(i,flush=True)
if __name__=='__main__':main()
