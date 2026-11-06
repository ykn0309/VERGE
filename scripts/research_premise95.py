"""Isolated research: task-blind method compilation and task-first requirement slots."""
from __future__ import annotations
import sys,json,threading,hashlib,argparse,re
from pathlib import Path
from copy import deepcopy
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verge.cli import backend
from verge.config import load_config
from verge.prompts import llm_messages as normal_messages
import verge.llm as llm
FIXTURES=Path('artifacts/experience-utilization-audit/fixed_inputs.json')
KNOWN_ARMS={'baseline','compiled','slots','repair','repair_strict','focus','normalized_slots_repair','normalized_reasoned','plus_fixed','normalized_plus','plus_reasoned_fixed','normalized_plus_reasoned','counterexample_fixed','counterexample_raw','guarded_full','direct_plus','taskonly_plus','rebuilt','gated','taskonly_flash','necessity_gated'}
ROOT=Path('artifacts/premise-research-95');STAGE=ContextVar('stage95',default='baseline')
COMPILE='你是核验知识编辑，输入只有历史方法，没有当前应用题。请真正修正过强方法，不只是改写措辞。在判断前先问：一个正确、充分完成任务的回答是否可以违反原方法的形式或顺序限制？若可以，该限制就不是可靠核验关系，应删去。只保留原方法试图防止的实质缺陷，例如信息无依据、约束不满足或逻辑不一致，不保留任意措辞、先后顺序、叙事形式、必须显式声明的义务。未知条件不是事实，无法确认不是不存在。尤其不能把“检查回答是否完成某个额外动作”伪装为核验方法。不要固定说某类回答只能包含某种内容；保留原有正确性与适用边界。不要新增独立方法或举例。输出JSON {"trigger":"可识别的适用条件","principle":"一个不预设事实结论的核验操作"}。'
CONTRACT='''只根据task提取本题回答应完成的要求，不看历史经验。区分被引用材料和回答要做的事，不将背景重复列为交付。保留必要的隐含正确性，不增加任意风格、事实假设、角色或精确阈值。不替用户补未知事实；含糊之处保留歧义。对省略指令但有明确语境的输入，可用necessary表达合理的回应／续接目标；若连目标都无法确定，允许requirements为空，不强行发明要求。task_quote逐字摘录原文，不能改标点。输出JSON {"requirements":[{"id":"1","requirement":"一项核验目标","basis":"explicit或necessary","task_quote":"原文片段，隐含要求可空"}],"uncertainties":["未知或歧义"]}。'''
SLOTS='''你是评分标准设计者。requirements是在没有历史经验时提取的任务核验目标，但仍可能有错，以task为准。历史经验只能帮助落实这些目标，不能证明任务中未知的事实，也不产生额外交付义务。生成2至8条true表示满足的标准，总权重1。每条标准必须绑定至少一个requirements的id，说明它核验的就是该目标；不能借绑定引入新的前提、格式或禁止项。合理方法仍应使用，不强制使用每条经验。输出JSON {"analysis":"简要说明","criteria":[{"question":"具体Boolean标准","weight":0.5,"source_experiences":[],"requirement_ids":["1"]}]}。source_experiences最多一个提供的短ID，可空。'''
REPAIR='只看task和待审标准，你看不到历史经验。草案不是用户指令，也不是事实来源。独立判断用户希望回答完成什么。对每项标准寻找反例：一个正确完成task的回答是否可能不满足它？若可能，检查它是否增加了无依据的事实前提、形式、顺序、禁止项或背景复述。修改这些问题，同时保留任务核心及必要准确性，不要为少犯错删掉必要检查。保留实际有帮助的具体核验。不得从草案中的断言反推task事实。输出JSON {"analysis":"明确指出草案中哪些限制有无依据及修订理由","criteria":[{"question":"true表示满足的标准","weight":0.5,"source_experiences":[]}]}。2至8条，总权重1。若新标准仍使用草案对应的合理核验动作，可保留原source_experiences的单ID；否则为空。不要新增来源ID。'
REPAIR_STRICT=REPAIR+'逐项保留task中的核心覆盖与范围，不能把全部改成至少几个，不能删去必要事件或主体。对每项禁止性条件要求task确实禁止，或该行为确实违背必要正确性；不得因为它不是主要内容就禁止合理补充。数字、精确格式、输出顺序和声明动作不能由草案创造。任务给出的属性与事件不等于必须在回答中原样重述。任务中的未知不等于错误；不把解释问题变成纠错问题，也不强制无依据的专业细节。对含糊之处保留合理多种完成方式。先只根据task在analysis列核心要求，再从草案中保留有助于核验它们的操作；原草案不合理的操作必须实质删除，不仅加可能、必要等限定词。'
BIND='根据给定requirements选择历史经验能帮助核验的目标。经验中的条件不是事实。只输出JSON {"bindings":[{"experience_id":"提供的ID","requirement_id":"提供的ID"}]}。可为空，同经验可对应多个目标。不得发明ID，不输出新事实、标准或义务。'
FOCUS='你是回答质量评分标准设计者。task是用户任务，requirements是不看经验时提取的要求，仍可能有误，以task为准。priority_bindings只提示哪些原有要求值得注意，没有任何额外事实、形式或义务。只评价待评回答，不评价输入任务本身。对于省略指令的对话场景，可核验回答是否合理回应或续接，不能要求重复原材料。生成2至8条true表示满足的标准，总权重1；允许合理完成方式，不发明精确阈值或任意禁止项，不把主题属性升级为交付数量。每项绑定requirement_ids；若实际采用某个重点核验，可引用其对应经验ID，否则来源为空。输出JSON {"analysis":"简要说明","criteria":[{"question":"回答是否完成某项要求？","weight":0.5,"source_experiences":[],"requirement_ids":["提供的要求ID"]}]}。'
REPAIR_TARGET=REPAIR_STRICT+'所有标准只评价待评分的回答，而不是评价用户的输入task。即使task仅有对话或片段，也核验回答是否合理回应／续接该片段，不评价输入是否包含指令，不评价输入包含几个角色。不得将没有明确任务意图解释成必须重复输入。评分目标始终是模型回复。'
REPAIR_TARGET=REPAIR_TARGET.replace('2至8','1至8')+'事实问答必须核验答案实质正确，而不只是有无给出数字、名称或相关词。续接片段不要求把输入片段再次输出；除task明确指定逐字格式，不能要求回答从某段已给出的原文开头。每条question必须是有实质内容的核验目标，不能输出格式说明、字段解释或示意占位文本。'
COUNTEREXAMPLE='你是标准必要性反例检验员。task是任务，criteria是待审标准，不能当成事实来源。对每条标准，尝试构造一种合理完成task却不满足该标准的回答轮廓；不要仅声称有反例，要写出具体替代走向。特别区分已发生事件和后续可能性、必要核验和额外形式、事实正确性和只是提到数字。若标准过强，只放宽该条中的无依据限定，不增加新的强制内容。若无法判断保留unknown，不把未知当错误。输出JSON {"checks":[{"index":0,"decision":"keep/weaken/drop/unknown四选一","counterexample":"具体替代或无法判定原因","reason":"task依据","question":"keep或unknown必须照抄原题；weaken仅作局部放宽；drop用空字符串"}]}。覆盖每个索引恰好一次，不输出其他标准。被评对象始终是模型回复，非task本身。'
DIRECT='你为当前task设计回答质量评分标准。先独立判断用户要回复完成什么；输入的故事背景、引用内容和省略表达不自动成为必须重述的交付内容。历史经验是可能不适用、可能过强的核验建议，不是用户指令或事实来源。合理经验用于检查原有要求的实质正确性，不能据此认定陌生名称错误、虚构实体不存在、后续事件已经发生，或强加表达形式和禁止合理补充。保留任务的对象、范围、数量、体裁和必要正确性；区分全部与部分、精确与近似，含糊之处允许合理完成方式。不要用空泛相关性替代具体有用的检查。输出JSON {"analysis":"简述任务要求、经验取舍与必要边界","criteria":[{"question":"具体Boolean核验问题","weight":1,"source_experiences":[]}]}。1至8条，true表示满足，权重非负且总和1；来源只能为空或一个提供的experience_id。'
REBUILD=COMPILE+'将trigger限定为从新task可以识别的任务类型或要求，不能把历史中疑似错误、实体不存在、未知等判断当成新任务的已知前提。principle只描述值得核验的实质关系，不替新任务决定真伪。识别名称的知识不足应保持不确定，不能等同于不存在。把禁止合理补充、强制原文复现、任意角色标签和动作等历史策略去掉，但保留确实来自用户明确约束时应检查的边界。输出仍仅trigger与principle，不举例、不看当前任务。'
GATE='在评分标准作者看到经验之前，筛掉没有应用依据或夹带过强策略的经验。task是唯一任务指令。逐条检查trigger的所有关键前提是否在task或可靠常识中成立，未验证的不存在/拼写错误/数据不可得等不能因经验写了就成立。再检查principle的具体核验操作是否有助于本题原有要求，不能要求任务外的形式、内容、重复背景或禁止合理补充。若只有泛泛主题相似、前提未知，或必须大改方法才合理，use=false；若方法在本题确实成立且有帮助则use=true。允许不选，也保留合理的事实依据、明确数量等方法；不能为减少错误全部拒绝。你只筛选，不改写经验或任务。输出JSON {"decisions":[{"experience_id":"每个提供ID恰好一次","use":true,"reason":"关键前提与具体方法的依据；use只能true或false"}]}。'
NECESSITY_GATE='你在评分标准作者接触历史经验前检查其能否原样应用。先只按task判断任务目标和可支持前提，再逐条看经验。经验的主题相关不代表trigger已成立，陌生、大小写不同或你不知道不代表名称错了。必须检查principle的每个强制动作，不能只因其中一部分有用就放行整条。为每条经验尝试构造一个充分完成task、却合理违反principle的具体回答轮廓。若可以，counterexample_found=true，写出该轮廓；否则为false并解释原因。区分合理补充与替代主题、角色反应内容与任意排版、复现背景与保持一致。别为保持原经验而把这些可选策略说成必要。premises_supported只在所有关键前提已由task或可靠常识支持时为true；仅有可能、未核实或任务目标含糊则为false。不重写经验，不设计标准。JSON {"decisions":[{"experience_id":"每个提供ID恰好一次","premises_supported":true,"counterexample_found":false,"counterexample":"具体合理完成的回答轮廓或无法构造的理由","reason":"前提和方法整体的判断依据"}]}。两个布尔字段均只能true或false。程序只在前提成立且未发现反例时让原经验通过。'
GUIDES={'necessity_gate':NECESSITY_GATE,'gate':GATE,'rebuild':REBUILD,'direct_plus':DIRECT,'compile':COMPILE,'contract':CONTRACT,'slots':SLOTS,'repair':REPAIR,'repair_strict':REPAIR_STRICT,'bind':BIND,'focus':FOCUS,'repair_target':REPAIR_TARGET,'repair_reasoned':REPAIR_TARGET,'repair_plus':REPAIR_TARGET,'repair_plus_reasoned':REPAIR_TARGET,'counterexample':COUNTEREXAMPLE}
def messages(role,payload,**kwargs):
 stage=STAGE.get()
 if stage=='baseline':return normal_messages(role,payload,**kwargs)
 return [{'role':'system','content':GUIDES[stage]},{'role':'user','content':json.dumps({k:v for k,v in payload.items() if not k.startswith('_')},ensure_ascii=False)}]
llm.llm_messages=messages

def valid(result,payload,stage):
 if stage=='necessity_gate':
  assert len(result['decisions'])==len(payload['experiences']) and {x['experience_id'] for x in result['decisions']}=={e['experience_id'] for e in payload['experiences']}
  assert all(type(x['premises_supported']) is bool and type(x['counterexample_found']) is bool and isinstance(x['counterexample'],str) and isinstance(x['reason'],str) for x in result['decisions']);return
 if stage=='gate':
  assert len(result['decisions'])==len(payload['experiences']) and {x['experience_id'] for x in result['decisions']}=={e['experience_id'] for e in payload['experiences']}
  assert all(type(x['use']) is bool and isinstance(x['reason'],str) for x in result['decisions']);return
 if stage in ('compile','rebuild'):assert set(result)=={'trigger','principle'} and all(isinstance(v,str) and v.strip() for v in result.values());return
 if stage=='bind':
  assert isinstance(result['bindings'],list)
  for x in result['bindings']:
   assert x['experience_id'] in {e['experience_id'] for e in payload['experiences']} and x['requirement_id'] in {r['id'] for r in payload['requirements']}
  return
 if stage=='counterexample':
  assert {x['index'] for x in result['checks']}==set(range(len(payload['criteria']))) and len(result['checks'])==len(payload['criteria'])
  for x in result['checks']:
   assert x['decision'] in ('keep','weaken','drop','unknown')
   if x['decision'] in ('keep','unknown'):assert x['question']==payload['criteria'][x['index']]['question']
   if x['decision']=='weaken':assert x['question']
  return
 if stage=='contract':
  req=result['requirements'];assert isinstance(req,list) and len({x['id'] for x in req})==len(req)
  for r in req:
   assert r['basis'] in ('explicit','necessary') and r['requirement']
   assert isinstance(r['task_quote'],str)  # quote presence is auditable, not a hard semantic gate
  return
 assert isinstance(result['analysis'],str) and (1 if stage in ('repair_target','repair_reasoned','repair_plus','repair_plus_reasoned','direct_plus') else 2)<=len(result['criteria'])<=8
 allowed={e for c in payload.get('draft_criteria',[]) for e in c['source_experiences']} if stage in ('repair','repair_strict','repair_target','repair_reasoned','repair_plus','repair_plus_reasoned') else {e['experience_id'] for e in payload['experiences']}
 for c in result['criteria']:
  if c.get('question','').strip() in {'true表示满足的标准','具体Boolean标准','具体标准','评价标准','回答是否完成某项要求？'}:raise ValueError('question contains a schema placeholder, replace it with a task-specific check')
  assert isinstance(c['question'],str) and c['question'] and type(c['weight']) in (int,float) and 0<=c['weight']<=1
  assert isinstance(c['source_experiences'],list) and len(c['source_experiences'])<=1 and set(c['source_experiences'])<=allowed
  if stage in ('slots','focus') and payload.get('requirements'):assert c['requirement_ids'] and set(c['requirement_ids'])<={r['id'] for r in payload['requirements']}
 assert sum(c['weight'] for c in result['criteria'])>0

def apply_counterexamples(original, audit, task, *, protect_numbers=False):
    """Protect literal numeric requirements conservatively; not semantic proof."""
    criteria=[];decisions=[]
    task_numbers=set(re.findall(r'\d+(?:\.\d+)?',task))
    for check in sorted(audit['checks'],key=lambda x:x['index']):
        old=original['criteria'][check['index']]
        protected=protect_numbers and bool(task_numbers & set(re.findall(r'\d+(?:\.\d+)?',old['question'])))
        if protected:
            criteria.append(deepcopy(old));decisions.append({'index':check['index'],'reason':'literal_numeric_constraint_protected','proposed':check['decision']})
        elif check['decision']!='drop':criteria.append({**old,'question':check['question']})
    if not criteria:raise ValueError('all checks dropped; no content success')
    total=sum(x['weight'] for x in criteria)
    if total<=0:raise ValueError('no positive evaluation budget')
    result={'analysis':original['analysis'],'criteria':[{**x,'weight':x['weight']/total} for x in criteria]}
    valid(result,{'draft_criteria':original['criteria']},'repair_plus')
    return result, decisions

def run(round_name,arms,repeats=1,split='development'):
 if not arms or set(arms)-KNOWN_ARMS or repeats<1:raise ValueError('invalid experiment arm or repeats')
 out=ROOT/round_name;out.mkdir(parents=True,exist_ok=False);(out/'research_premise95.py').write_bytes(Path(__file__).read_bytes());lock=threading.Lock()
 def append(name,row):
  with lock:
   with (out/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 spec=load_config('configs/midterm-general-v2-assumed-train-300-batch3.yaml')['models']['default']
 cases=[c for c in json.loads(FIXTURES.read_text()) if c['split']==split and c['role']=='initial_generator']
 compile_stage='rebuild' if 'rebuilt' in arms else 'compile'
 cache=ROOT/(compile_stage+'_'+hashlib.sha256(GUIDES[compile_stage].encode()).hexdigest()[:12]+'.json');methods={}
 if cache.exists():methods=json.loads(cache.read_text())
 def request(stage,payload,ctx):
  overrides={'repair_reasoned':{'enable_thinking':True,'timeout_seconds':240},'repair_plus':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':False},'repair_plus_reasoned':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'counterexample':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'direct_plus':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'rebuild':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'gate':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':False,'timeout_seconds':120},'necessity_gate':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240}}
  stage_spec={**spec,**overrides.get(stage,{})}
  b=backend(stage_spec);b.trace_hook=lambda e:append('role_io.jsonl',{**ctx,'stage':stage,**e});b.metadata_hook=lambda e:append('llm_requests.jsonl',{**ctx,'stage':stage,**e});token=STAGE.set(stage);base=deepcopy(payload)
  try:
   last_error=''
   for attempt in range(3):
    data=deepcopy(base)
    if attempt:data['validation_feedback']='前次输出无效：'+last_error+'。请严格返回所要求的JSON，依据原输入，不添加字段。'
    r=b.generate('initial_generator' if stage=='baseline' else 'critic',data)
    try:valid(r,data,stage);return r
    except (AssertionError,KeyError,TypeError,ValueError) as exc:
     last_error=str(exc) or '字段结构、ID或必需绑定错误'
     if attempt==2:raise ValueError('invalid stage contract')
  finally:STAGE.reset(token);b.close()
 def key(e):return hashlib.sha256(json.dumps({k:e[k] for k in ('trigger','principle')},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 if any(arm not in ('baseline','repair','repair_strict','normalized_slots_repair','normalized_reasoned','plus_fixed','normalized_plus','plus_reasoned_fixed','normalized_plus_reasoned','counterexample_fixed','counterexample_raw','guarded_full','direct_plus','taskonly_plus','gated','taskonly_flash','necessity_gated') for arm in arms):
  unique={key(e):{k:e[k] for k in ('trigger','principle')} for c in cases for e in c['input']['experiences']}
  def compile_one(item):
   k,e=item;return k,request(compile_stage,e,{'method_key':k})
  with ThreadPoolExecutor(max_workers=8) as cp:
   for k,value in cp.map(compile_one,[(k,e) for k,e in unique.items() if k not in methods]):methods[k]=value
  cache.write_text(json.dumps(methods,ensure_ascii=False,indent=2))
 def one(case,arm,rep):
  ctx={'sample_id':case['sample_id'],'arm':arm,'repetition':rep,'group':case['group']};payload=deepcopy(case['input'])
  try:
   if arm not in ('baseline','repair','repair_strict','normalized_slots_repair','normalized_reasoned','plus_fixed','normalized_plus','plus_reasoned_fixed','normalized_plus_reasoned','counterexample_fixed','counterexample_raw','guarded_full','direct_plus','taskonly_plus','gated','taskonly_flash','necessity_gated'):payload['experiences']=[{'experience_id':e['experience_id'],**methods[key(e)]} for e in payload['experiences']]
   if arm in ('gated','taskonly_flash','necessity_gated'):
    if arm in ('gated','necessity_gated'):
     gate=request('necessity_gate' if arm=='necessity_gated' else 'gate',payload,ctx);append('gates.jsonl',{**ctx,**gate});kept={x['experience_id'] for x in gate['decisions'] if (x['premises_supported'] and not x['counterexample_found'])} if arm=='necessity_gated' else {x['experience_id'] for x in gate['decisions'] if x['use']};payload['experiences']=[e for e in payload['experiences'] if e['experience_id'] in kept]
    else:payload['experiences']=[]
    append('filtered_inputs.jsonl',{**ctx,'input':payload});result=request('baseline',payload,ctx)
   elif arm in ('direct_plus','taskonly_plus'):
    if arm=='taskonly_plus':payload['experiences']=[]
    result=request('direct_plus',payload,ctx)
   elif arm in ('counterexample_fixed','counterexample_raw'):
    if arm=='counterexample_raw':
     previous=[json.loads(l) for l in Path('artifacts/generator-application-94/development/results.jsonl').read_text().splitlines()]
     prior=next(x for x in previous if x['sample_id']==case['sample_id'] and x['repetition']==rep and x['variant']=='baseline')
    else:
     previous=[json.loads(l) for l in (ROOT/'round11/results.jsonl').read_text().splitlines()]
     prior=next(x for x in previous if x['sample_id']==case['sample_id'] and x['repetition']==rep)
    if not prior['valid']:raise ValueError('upstream failed; do not replace missing evidence')
    original=prior['output'];audit=request('counterexample',{'task':payload['task'],'criteria':original['criteria']},ctx);append('counterexamples.jsonl',{**ctx,**audit})
    criteria=[]
    for check in sorted(audit['checks'],key=lambda x:x['index']):
     if check['decision']!='drop':criteria.append({**original['criteria'][check['index']],'question':check['question']})
    if not criteria:raise ValueError('all checks dropped; no content success')
    total=sum(x['weight'] for x in criteria)
    if total<=0:raise ValueError('no positive evaluation budget')
    result={'analysis':original['analysis'],'criteria':[{**x,'weight':x['weight']/total} for x in criteria]}
    valid(result,{'draft_criteria':original['criteria']},'repair_plus')
   elif arm in ('plus_fixed','plus_reasoned_fixed'):
    previous=[json.loads(l) for l in (ROOT/'round6/intermediates.jsonl').read_text().splitlines()]
    draft=next(x['output'] for x in previous if x['sample_id']==case['sample_id'] and x['repetition']==rep)
    result=request('repair_plus_reasoned' if arm=='plus_reasoned_fixed' else 'repair_plus',{'task':payload['task'],'draft_criteria':draft['criteria']},ctx)
   elif arm in ('normalized_slots_repair','normalized_reasoned','normalized_plus','normalized_plus_reasoned','guarded_full'):
    normalized={x['experience_id']:x['method'] for x in json.loads(Path('artifacts/experience-utilization-audit/normalized_methods.json').read_text())}
    mapping=case['id_mapping']['reverse']['experience']
    payload['experiences']=[{'experience_id':e['experience_id'],**normalized[mapping[e['experience_id']]]} for e in payload['experiences']]
    contract=request('contract',{'task':payload['task']},ctx);append('contracts.jsonl',{**ctx,**contract})
    if contract['requirements']:
     intermediate=request('slots',{**payload,**contract},ctx)
    else:
     intermediate=request('baseline',{'task':payload['task'],'experiences':[]},ctx);ctx['fallback']='ambiguous_task_no_experience'
    append('intermediates.jsonl',{**ctx,'output':intermediate})
    result=request('repair_plus_reasoned' if arm in ('normalized_plus_reasoned','guarded_full') else 'repair_reasoned' if arm=='normalized_reasoned' else 'repair_plus' if arm=='normalized_plus' else 'repair_target',{'task':payload['task'],'draft_criteria':intermediate['criteria']},ctx)
   elif arm=='focus':
    contract=request('contract',{'task':payload['task']},ctx);append('contracts.jsonl',{**ctx,**contract})
    bound=request('bind',{'requirements':contract['requirements'],'experiences':payload['experiences']},ctx) if contract['requirements'] else {'bindings':[]}
    append('bindings.jsonl',{**ctx,**bound})
    # The final author cannot see any historical method text.
    public={'task':payload['task'],**contract,'priority_bindings':bound['bindings'],'experiences':[{'experience_id':e['experience_id']} for e in payload['experiences']]}
    result=request('focus',public,ctx)
   elif arm in ('repair','repair_strict'):
    prior=[json.loads(l) for l in (ROOT/'round2/results.jsonl').read_text().splitlines()]
    draft=next((x['output'] for x in prior if x['sample_id']==case['sample_id'] and x['arm']=='slots' and x['repetition']==rep and x['valid']),None)
    if draft is None:raise ValueError('upstream slots invalid; preserve missing output')
    public={'task':payload['task'],'draft_criteria':draft['criteria']}
    result=request(arm,{**public,'_message_budget':100000},ctx)
   elif arm=='slots':
    contract=request('contract',{'task':payload['task']},ctx);append('contracts.jsonl',{**ctx,**contract});payload={**payload,**contract}
    if not contract['requirements']:
     payload['experiences']=[];result=request('baseline',payload,ctx);ctx['fallback']='ambiguous_task_no_experience'
    else:result=request('slots',payload,ctx)
   else:result=request('baseline',payload,ctx)
   if arm=='guarded_full':
    append('pre_audit.jsonl',{**ctx,'output':result})
    audit=request('counterexample',{'task':payload['task'],'criteria':result['criteria']},ctx);append('counterexamples.jsonl',{**ctx,**audit})
    result,protected=apply_counterexamples(result,audit,payload['task'],protect_numbers=True);append('protection.jsonl',{**ctx,'protected':protected})
   append('results.jsonl',{**ctx,'valid':True,'output':result})
  except Exception as e:append('results.jsonl',{**ctx,'valid':False,'error':str(e)})
 (out/'protocol.json').write_text(json.dumps({'arms':arms,'repeats':repeats,'split':split,'cases':len(cases),'model':spec,'stage_overrides':{'repair_reasoned':{'enable_thinking':True,'timeout_seconds':240},'repair_plus':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':False},'repair_plus_reasoned':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'counterexample':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'direct_plus':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'rebuild':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240},'gate':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':False,'timeout_seconds':120},'necessity_gate':{'model':'qwen3.5-plus-2026-04-20','enable_thinking':True,'timeout_seconds':240}},'guides':GUIDES,'compiled_cache':str(cache),'quote_policy':'record quotes; mismatch does not prove semantic failure','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'isolated generator variants; compiled changes text; slots combines compilation and task-only contract','raw_input_sha256':hashlib.sha256(FIXTURES.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,f in enumerate(as_completed([pool.submit(one,c,a,r) for c in cases for a in arms for r in range(repeats)]),1):f.result();print(i,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('round');p.add_argument('--arms',nargs='+',default=['baseline','compiled','slots']);p.add_argument('--repeats',type=int,default=1);p.add_argument('--split',default='development');p.add_argument('--fixtures',type=Path,default=FIXTURES);a=p.parse_args();FIXTURES=a.fixtures;run(a.round,a.arms,a.repeats,a.split)
