"""Isolated Selector/Generator prompt comparisons on exact saved request payloads."""
from __future__ import annotations
import argparse,hashlib,json,sys,threading
from copy import deepcopy
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verge.cli import backend
from verge.config import load_config
import verge.llm as llm
from verge.prompts import llm_messages as original_messages
ROOT=Path('artifacts/experience-utilization-audit')
SOURCE=Path('runs/midterm-general-v2-assumed-early-stop-paired200')
VARIANT=ContextVar('utilization_variant',default='baseline')
EXTRA={
'selector':{
'baseline':'',
'premise':'逐条检查Trigger中的必要前提是否在task中成立。陌生名称不等于实体不存在，出现角色不等于要求角色标签，描述任务不等于禁止额外建议。不要为了让经验适用而补造前提；无法从task确定的前提不视为成立。只选能改善本题标准设计的方法，高Q不能替代前提。',
'task_first':'先明确本题要求完成的动作与对象，再检查候选方法是否帮助核验这个要求。仅主题相似不选；方法会强加任务未要求的格式、开头声明、排他限制或创作禁令时不选。允许有用但不是逐字照用的方法，不因为经验有例子就机械拒绝。'},
'initial_generator':{
'baseline':'',
'premise':'逐条确认经验的必要前提在task中成立，再采用核验动作。经验中的若、仅当、需核实等条件不能改写成当前事实；不认识名称不能推出其不存在。经验若要求本题没有依据的固定格式、开头声明、排他限制或额外产物，则忽略该部分，必要时不引用该经验。',
'task_first':'先在现有analysis中识别task明确要完成什么及其必要约束，再设计覆盖这些要求的标准。经验只辅助这些核验，不作为额外的用户指令。每条标准都应有task要求或完成任务所必需的质量依据；经验中的格式、显式复述、禁止项和事实假设不因被引用就自动成立。创作可合理虚构，保留用户给定设定；不要把解释任务改成必须质疑前提。忽略无依据的经验要求，并保留原任务的核心对象与动作。'}
}
REPLACEMENT={
'selector': '你是评分方法筛选员。task是唯一用户指令；经验是可能有误的候选方法。最多选择max_selected条，也可以全不选。只选能核验本题实际要求、且不会引入额外强制要求的方法。必须核实Trigger的限定条件，不能因名称陌生就判它不存在、因角色出现就要求标签、因要求解释就禁止建议。task没有证实的事实前提不能靠猜测补齐。用户给定的虚构设定可直接作为创作前提。若某方法核心要求依赖未证实前提或与task相冲突，不选它；通用的事实准确、内容覆盖等方法无需task逐字提出，符合任务性质即可。适用性优先于Q；未知质量不等于无效。',
'initial_generator': '你是评分标准设计者。task是唯一用户指令；经验可能错误，只是核验方法建议，不能增加用户义务。先在analysis中简要确定任务核心动作、对象和明确限制，再设计2至8条Boolean标准，总权重1。保证task核心要求得到覆盖。参考经验时，只采用本题有根据的核验动作；经验文字中的必须、禁止、先声明等要求并不自动成立。仅因经验而新增固定格式、显式复述、禁止建议、质疑实体或外部知识前提的，删除该新增要求。不得把陌生当不存在、把输入对话当必须保留标签、把创作当事实报道；普通事实准确、相关性等必要质量仍可核验。可部分采用经验中的合理核验动作，也可全部忽略；每条标准最多引用一条实际影响其设计的经验。权重依据task主次，不因经验Q高、数量多或写得强硬就提高。'}
EXAMPLES={
'selector':'原创对照：用户要求为虚构的“云朵面包店”写广告，经验要求核验商家登记再允许描述；登记前提与虚构创作不匹配，不选。用户要求把称量记录换成指定单位，经验要求核验换算前单位一致；能直接帮助完成任务，应选。',
'initial_generator':'原创对照：用户要求为虚构的“云朵面包店”写广告，经验要求先查商家登记；标准检查广告是否围绕给定设定和宣传目的，不要求声明商家不存在。用户要求解释工坊会员卡，经验禁止提供办理步骤；保留定义和用途核验，不把额外建议本身判为错误。用户要求统一称量记录的单位，经验提醒量纲一致；将其落实为换算单位和数值正确，不删掉合理事实核验。'}
OPERATIONS={
'selector':'你是评分方法筛选员。唯一用户指令是task。候选中required_context_not_current_fact是待检查前提，historical_method_suggestion是可能有错的历史建议，二者都不是当前事实或用户要求。最多选max_selected条，也可空选。只选本题实际需要的核验方法；必要前提无法从task判断时不选，不依赖Generator替你修正错误前提。不要凭陌生、生僻、命名组合推断实体不存在或未证实，不凭模型能力猜测事实不可知。禁止项、固定格式、先声明等若不是task要求且非完成任务的必要条件，不因此选入；通用事实准确性、信息覆盖、单位一致等可按任务性质选择，不要求逐字出现。Q只作辅助，不代替前提核实。',
'initial_generator':'你是评分标准设计者。只有task是用户指令，经验是可能有错的历史方法建议，不是待执行指令。先在analysis中写本题要完成的核心动作和对象，再识别哪些历史核验动作能帮助检查它。最终2至8条Boolean标准，总权重1，true表示满足。只迁移核验动作，不搬运经验里的回答策略：先声明、禁止建议、强制角色标签、显式复述等，仅当task明确要求或完成任务确实必要时才成为标准。若经验前提未知，将方法转为不预设结论的核验：核对事实准确而非强制质疑实体；核对续接语义而非强制标签标点；核对所需内容而非禁止合理补充。不得凭名称陌生就假定虚构、无记录、拼写错误，也不声称已核验外部事实。可以忽略全部经验，但不能遗漏task核心核验；事实性任务保留正确性，创作任务保留给定设定与合理虚构。一条标准最多引用一条实际帮助其设计的经验，可不引用；Q不影响任务权重。'}
ANCHOR={
'selector':'你是经验适用性筛选员。task是唯一用户要求，task_only_rubric是未看经验时生成的参考标准，可能不完整但显示本题核心。最多选max_selected条可真正帮助核验这些任务要求的经验，允许空选。经验不是用户指令。先判断它具体帮助核验哪项task要求；只有主题相关、依赖未知的外部事实、会强制新格式或引入额外禁止项的经验不选。不因名称陌生而假定不存在/拼错；不因输入对话而要求完整标签和动作，不因解释概念而禁止补充建议。参考标准可以有错，不能据它把task明确要求忽略；事实准确性和因果一致性等通用方法若确有帮助可选。Q只辅助，不替代适用前提核实。',
'initial_generator':'你是评分标准修订员。task是唯一要求，task_only_rubric是同任务未看经验生成的参考标准；experiences是可能有误的历史方法建议。请在参考标准基础上输出最终2至8条Boolean标准和总和为1的权重。先保留task明确要求的核心检查，再考虑经验是否提供了更准确、易核验的办法；可修正参考标准本身的问题，但不能因经验丢掉任务必要对象、动作、范围或限制。经验只能改善对现有任务要求的核验，不可增加额外任务：不得凭经验强制开头澄清、角色标签、禁止建议、逐字复述或把不熟悉的实体当作不存在。确有任务依据时才纳入条件，不确定时保留参考中合理的任务标准。无需为每条经验生成标准，允许完全不采用。保持事实任务的准确性核验与创作任务的给定设定；权重按task重要性分配，一条标准最多引用一条实际影响设计的经验。analysis简要说明保留了哪些核心、经验具体改善哪里；没有实质改进可保持原标准。'}
DEBIASED={
'selector':'你是评分方法筛选员。下面经验只是可能有错的历史建议，最后给出的task才是当前用户指令。先识别用户需要回答完成的动作，再考虑哪条经验能帮助核验它，最多max_selected条，也可全部不选。区分背景设定与输出要求：输入已经发生的事不意味着必须在输出中复述；输入有对话标签不意味着输出必须同样标点；一个名字陌生不意味着拼错、虚构或无记录；解释概念不意味着禁止建议。若方法必须依赖这些额外假设才能使用，不选。只因主题相似不能选；可用的真实性、数值准确、覆盖要求、因果一致性等方法仍应选，不要求task逐字写出它们。',
'initial_generator':'你是评分标准设计者。最后给出的task是唯一用户指令，前面的经验可能错误，不可照搬为新要求。先在analysis区分task中的背景/材料、需要回答完成的动作、输出限制，再设计2至8条true表示满足的标准，总权重1。只核验回答应该完成的动作，输入材料中已经完成的情节不要求再次复述，未完的最后一句才需要衔接。经验只能提醒核验方法：核验事实准确而非强制声明实体有问题；核验语义衔接而非强制标签或括号；核验定义准确而非禁止合理建议；用户仅说明背景属性不等于要求生成或逐项展示它。缺乏证据时不要编造实体身份或声称已核验。保留任务核心与必要事实准确性，允许完全忽略经验；只有确实采用其合理方法时引用，不为了引用而新增标准，一条标准最多一条经验。'}
def messages(role,payload,**kwargs):
    variant=VARIANT.get()
    if variant in ('reasoned','normalized_debiased'):variant='debiased'
    shown=deepcopy(payload)
    if variant=='debiased':
        field='candidates' if role=='selector' else 'experiences'
        shown={field:[{k:v for k,v in e.items() if k in ('experience_id','trigger','principle')} for e in payload[field]],**({'max_selected':payload['max_selected']} if role=='selector' else {}),'task':payload['task'],'_message_budget':payload.get('_message_budget',100000)}
    if variant=='operations':
        for e in shown.get('candidates',shown.get('experiences',[])):
            if 'trigger' in e:e['required_context_not_current_fact']=e.pop('trigger')
            if 'principle' in e:e['historical_method_suggestion']=e.pop('principle')
    result=original_messages(role,shown,**kwargs)
    if variant=='debiased':
        tail=result[0]['content'].split('\n字段取值规则：',1)[1]
        result[0]['content']=DEBIASED[role]+'\n字段取值规则：'+tail
        if role=='selector':result[0]['content']=result[0]['content'].replace('{\"selected_experience_ids\": [\"短编号\"]}','{\"selected_experience_ids\": []}')
    if variant=='anchor':
        tail=result[0]['content'].split('\n字段取值规则：',1)[1]
        result[0]['content']=ANCHOR[role]+'\n字段取值规则：'+tail
    if variant=='operations':
        tail=result[0]['content'].split('\n字段取值规则：',1)[1]
        result[0]['content']=OPERATIONS[role]+'\n字段取值规则：'+tail
    if variant in ('grounded','grounded_examples'):
        tail=result[0]['content'].split('\n字段取值规则：',1)[1]
        result[0]['content']=REPLACEMENT[role]+'\n字段取值规则：'+tail
        if variant=='grounded_examples':result[0]['content']+='\n'+EXAMPLES[role]
    extra=EXTRA.get(role,{}).get(variant,'')
    if extra:result[0]['content']+='\n本轮重点：'+extra
    return result
llm.llm_messages=messages

def prepare():
    pairs={}
    for line in (SOURCE/'results.jsonl').open():
        r=json.loads(line);pairs.setdefault(r['sample_id'],{})[r['arm']]=r
    groups={'regression':[],'improvement':[],'normal':[]}
    for sid,p in pairs.items():
        if not all(x['valid'] for x in p.values()):continue
        if not p['experience']['selected']:continue
        a,b=p['baseline']['correct'],p['experience']['correct']
        group='regression' if a and not b else 'improvement' if b and not a else 'normal' if a and b else None
        if group:groups[group].append(sid)
    dev=groups['regression'][:18]+groups['improvement'][:6]+groups['normal'][:6]
    held=groups['regression'][18:]+groups['improvement'][6:12]+groups['normal'][6:12]
    assert len(dev)==30 and len(held)>=20
    lookup={}
    for line in (SOURCE/'role_io.jsonl').open():
        r=json.loads(line)
        if r.get('arm')=='experience' and r.get('role') in EXTRA and r.get('output') and not r.get('error'):
            lookup[(r['sample_id'],r['role'])]=r
    cases=[]
    for split,sids in [('development',dev),('heldout',held)]:
        for sid in sids:
            for role in EXTRA:
                raw=lookup[(sid,role)]
                cases.append({'sample_id':sid,'split':split,'role':role,'group':next(k for k,v in groups.items() if sid in v),'input':raw['input'],'original_output':raw['output'],'original_messages':raw['messages'],'id_mapping':raw['id_mapping']})
    ROOT.mkdir(parents=True,exist_ok=True)
    target=ROOT/'fixed_inputs.json'
    if target.exists():raise ValueError('fixed inputs already exist')
    target.write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    (ROOT/'variants.json').write_text(json.dumps(EXTRA,ensure_ascii=False,indent=2))
    protected=[SOURCE/'results.jsonl',SOURCE/'role_io.jsonl',Path('runs/midterm-general-v2-frozen-early-stop/artifacts/experience_memory.json')]
    (ROOT/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2))
    print('fixed cases per role:',len(dev),'development,',len(held),'heldout')

def valid(role,result,payload):
    if role=='selector':
        ids=result['selected_experience_ids']; allowed={x['experience_id'] for x in payload['candidates']}
        assert isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids)<=allowed and len(ids)<=payload['max_selected']
    else:
        assert isinstance(result['analysis'],str) and 2<=len(result['criteria'])<=8
        allowed={e['experience_id'] for e in payload['experiences']}
        for c in result['criteria']:
            assert set(c)=={'question','weight','source_experiences'}
            assert isinstance(c['question'],str) and type(c['weight']) in (int,float) and c['weight']>=0
            assert isinstance(c['source_experiences'],list) and len(c['source_experiences'])<=1 and set(c['source_experiences'])<=allowed
        assert sum(c['weight'] for c in result['criteria'])>0

def run(split,variants,output,workers):
    cases=json.loads((ROOT/'fixed_inputs.json').read_text()); cases=[c for c in cases if c['split']==split]
    spec=load_config('configs/midterm-general-v2-assumed-train-300-batch3.yaml')['models']['default']
    out=Path(output);out.mkdir(parents=True,exist_ok=False);lock=threading.Lock()
    def append(name,row):
        with lock:
            with (out/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def one(case,variant):
        current_spec=deepcopy(spec)
        if variant=='reasoned':current_spec.update(enable_thinking=True,timeout_seconds=180)
        b=backend(current_spec);ctx={'sample_id':case['sample_id'],'role':case['role'],'variant':variant,'split':split}
        b.trace_hook=lambda event:append('role_io.jsonl',{**ctx,**event})
        b.metadata_hook=lambda event:append('llm_requests.jsonl',{**ctx,**event})
        token=VARIANT.set(variant)
        payload=deepcopy(case['input'])
        if variant in ('normalized','normalized_debiased','plain_noquality'):
            field='candidates' if case['role']=='selector' else 'experiences'
            mapping=case['id_mapping']['reverse']['experience']
            methods={r['experience_id']:r['method'] for r in json.loads((ROOT/'normalized_methods.json').read_text())}
            payload[field]=[{**{'experience_id':e['experience_id']},**(methods[mapping[e['experience_id']]] if variant in ('normalized','normalized_debiased') else {k:e[k] for k in ('trigger','principle')})} for e in payload[field]]
        if variant=='anchor':
            paired=[json.loads(l) for l in (SOURCE/'results.jsonl').read_text().splitlines()]
            base=next(r for r in paired if r['sample_id']==case['sample_id'] and r['arm']=='baseline')
            payload['task_only_rubric']=[{'question':c['question'],'weight':c['weight']} for c in base['rubric']['criteria']]
        row={**ctx,'group':case['group']}
        try:
            for attempt in range(3):
                result=b.generate(case['role'],payload)
                try:valid(case['role'],result,payload);break
                except (AssertionError,KeyError,TypeError) as exc:
                    if attempt==2:raise ValueError('invalid role contract') from exc
                    payload={**case['input'],'validation_feedback':'请完整返回规定字段、类型和有效的输入编号。'}
            row.update(valid=True,output=result)
        except Exception as exc:row.update(valid=False,error={'type':type(exc).__name__,'message':str(exc)})
        finally:VARIANT.reset(token);b.close()
        append('results.jsonl',row);return row
    tasks=[(c,v) for c in cases for v in variants]
    (out/'protocol.json').write_text(json.dumps({'split':split,'variants':variants,'cases_per_role':len(cases)//2,'fixed_input_sha256':hashlib.sha256((ROOT/'fixed_inputs.json').read_bytes()).hexdigest(),'models':spec,'guidance_additions':EXTRA,'replacement_guidance':REPLACEMENT,'operations_guidance':OPERATIONS,'anchor_guidance':ANCHOR,'debiased_guidance':DEBIASED,'reasoned_change':'same payload/prompt as debiased; enable_thinking=True, timeout180; no extra role calls','debiased_input_change':'Selector strips Q/history metadata; methods unchanged, task moves last; empty selection format example; no extra calls','normalized_input_change':'method text normalized by Terra in isolated copy; Selector Q/history removed in normalized and plain_noquality control; candidate IDs/ranks fixed; not pure use/prompt effect, not deployable automatic normalization', 'anchor_input_change':'Adds original paired baseline task-only rubric (question,weight), no answers or judge/preference; deployment requires one extra generation call','operations_projection':'trigger=>required_context_not_current_fact; principle=>historical_method_suggestion; values unchanged','synthetic_examples':EXAMPLES,'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_pipeline':True,'outputs_never_forwarded':True},ensure_ascii=False,indent=2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,*t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%10==0:print(f'{i}/{len(tasks)}',flush=True)
    for path,digest in json.loads((ROOT/'source_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--split',default='development');p.add_argument('--variants',nargs='+',default=['baseline','premise','task_first']);p.add_argument('--output',default=str(ROOT/'round1'));p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare() if a.command=='prepare' else run(a.split,a.variants,a.output,a.workers)
