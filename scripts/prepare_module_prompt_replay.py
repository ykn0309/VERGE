"""Freeze three prompt variants and independent historical confirmation inputs."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import random

from scripts.prepare_module_model_replay import SOURCE, project
from scripts.replay_critic_grounded_few_shot import paired_messages
from scripts.replay_experience_splits import digest, fingerprint, save
from verge import prompts

PRIOR = Path('runs/module-model-replay-20')
SCREEN = Path('runs/module-prompt-replay-20')
CONFIRM = Path('runs/module-prompt-confirmation-20')

# All teaching tasks below are independently invented, not drawn from the dataset.
GUIDANCE = {
 'selector': '''针对当前task选择有实质标准设计帮助的经验，允许不选。
先辨认当前要求执行的操作，材料主题不等于任务操作。逐条判断：经验的核验方法是否适用、会否添加无依据要求或放宽明确约束。置信度仅在方法适用后辅助取舍。经验若只与主题相关，或只有看过回答/标准才能判断其trigger，不选。不得为凑数量选择经验；也不得因措辞不完全相同排除真正适用的方法。''',
 'initial_generator': '''根据task及可选经验设计2–8条正向是/否标准，并分配总权重预算1。
先确定本次要求回答者做什么，区别材料内容、未来步骤和当前指令。明确数量、范围、格式和禁止事项按task保留；不从经验补造硬门槛或替代任务目标。
经验仅作方法参考，可采用、组合或忽略；来源只记录实质改变问题或权重的方法，任务本身足以提出且未受经验改变的条目用[]。
每项“是”必须表示表现良好。以核验动作表达问题，重复维度共享预算，核心要求优先于辅助表达。analysis简述取舍，最终标准须落实取舍，不新增总分否决、部分分或跳过规则。''',
 'judge': '''仅核对当前task、标准和这份回答。逐项先定位对应原文，再输出Boolean及简短依据。
材料没有提供的内容不能当作已知原文；task、标准中提到的文字不等于回答已经写出。按标准的语义要求核验，除非明确要求逐字匹配，不因换词否认同一含义。范围、否定和数量分别核对，不凭大致印象补造计数或引用。
reason说明这份回答真正写出或缺失的依据，satisfied必须与reason的结论一致。不能用其他项通过或回答总体流畅替代本项判断。不扩展标准或虚构外部核验。''',
 'critic': '''从输入证据提出供未来首次设计评分标准使用的方法，返回简洁analysis和0–1条候选。
先以task和两回答原文核对事实归属，再核对标准及两侧Judge。Judge理由与历史analysis都是待核对说法，不能将其中的引用、错误或推测移入回答原文。
偏好身份、整体失败或两答同分都不构成新增要求的依据。只有可确认的标准设计缺陷才提方法；若只有Judge错误、偏好无法从任务说明解释或无可信设计动作，返回空候选。真实遗漏仍须识别，不为了减少候选忽略遗漏。
content写标准设计动作，trigger从生成前task/输入/领域条件识别；将实例替换为语义角色或task参数。不要加入无依据的数量、章节、句式、逐字要求，不放宽task，不写给Judge或回答者的指令，不改变加权Boolean机制。第二轮按真实反馈重新核对，不沿用历史分析的事实错误。''',
 'experience_updater': '''根据当前经验及全部案例，输出一条共同方法或两条不同方法，并分配全部案例；不输出analysis。
先识别task实际要求的核验动作，原回答是事实依据，引用标准和Judge解释只是线索。明显误判或task不支持的规则不进入经验。当前文本也可能有误，证据不足时保持合理的原方法。
同一动作在不同对象、场景或参数上的应用保持一条。合在一起必须保留各组案例实际需要的动作和适用范围；不能只找到共同主题而丢掉方法。只有动作/证据规则实质不同、合成一条会丢边界或变成方法清单时拆成两条；不确定时保持一条。两条若只是近义改写，应保留一条。
content写可复用的标准设计方法，trigger从生成前条件识别，不依赖回答或已有标准。去除案例专名和固定情节，保留必要专业限定，不降低task要求，不增加否决/部分分/自动跳过。每条content≤150字符、trigger≤60字符；每个短案例编号恰好分配一次，各组非空可为一个案例。'''
}

EXAMPLES = {
 'selector': [
  {'task':'把设备维修记录按日期提取为表格；记录包含一段旅行轶事。',
   'experiences':{'1':'设计旅行故事的情节张力标准','2':'提取任务逐项核对记录字段与原文归属'},
   'result':{'selected_experience_ids':['2']},'why':'执行的是信息提取，旅行仅是材料主题。'},
  {'task':'推荐6个室内盆栽，只列名称。','experiences':{'1':'数量太多时允许改为2个精选','2':'看到回答后核验配图是否美观'},
   'result':{'selected_experience_ids':[]},'why':'一个放宽明确数量，另一个依赖生成后回答，均不适用。'}],
 'initial_generator': [
  {'task':'下面介绍将来要写的研究摘要要求。现在只回复“准备好了”，等待材料。',
   'experience':'摘要应核验来源忠实性。',
   'result':{'analysis':'当前仅确认，尚不执行摘要任务；摘要方法不适用。','criteria':[
    {'question':'是否给出要求的确认语“准备好了”？','weight':0.8,'source_experiences':[]},
    {'question':'是否保持仅回复确认语的范围？','weight':0.2,'source_experiences':[]}]},'why':'评价当前行为，不能把未来摘要要求提前变成当前标准。'},
  {'task':'列出4个节水措施，不比较措施之间的优劣。','experience_id':'1',
   'experience':'建议类标准通常只需2个精选，并可要求比较优劣。',
   'result':{'analysis':'经验与数量及禁止比较的明确要求冲突，忽略它。','criteria':[
    {'question':'是否列出4个与节水有关的措施？','weight':0.6,'source_experiences':[]},
    {'question':'是否遵守不比较措施优劣的限制？','weight':0.4,'source_experiences':[]}]},'why':'不把坏方法变成硬约束；“是”方向始终表示合格。'}],
 'judge': [
  {'task':'概括所附报告，但本次消息未附报告。','response':'报告显示所有地区的用水量均下降。',
   'criteria':{'1':'是否有所附报告中的依据支撑结论？'},
   'result':{'criterion_judgments':[{'criterion_id':'1','satisfied':False,'reason':'输入未附报告，不能确认回答结论来自报告。'}]}},
  {'task':'列出不含比较的建议。','response':'先关灯；拔掉闲置充电器；减少空调使用。',
   'criteria':{'1':'是否列出3条建议？','2':'是否遵守不比较建议优劣的要求？'},
   'result':{'criterion_judgments':[{'criterion_id':'1','satisfied':True,'reason':'共列出关灯、拔充电器、减少空调3条。'},
    {'criterion_id':'2','satisfied':True,'reason':'没有比较各建议的优劣。'}]}},
  {'task':'说明是否能完成。','response':'目前无法完成。',
   'criteria':{'1':'是否明确说明现在不能完成？','2':'是否说明不能完成的原因？'},
   'result':{'criterion_judgments':[{'criterion_id':'1','satisfied':True,'reason':'“目前无法完成”表达现在不能完成，语义满足。'},
    {'criterion_id':'2','satisfied':False,'reason':'仅说明无法完成，没有给出原因。'}]}}],
 'critic': [
  {'task':'简短说明如何整理书桌，不规定格式。','answer_A':'关掉设备，再把线缆分类。','answer_B':'1. 关掉设备。2. 将线缆分类。',
   'criteria':'是否给出整理方法？两侧Judge均判满足。',
   'result':{'analysis':'两回答都有方法；列表与短句都是允许形式。偏好差异不能推出禁止列表或必须单段，未发现可信设计缺陷。','candidates':[]}},
  {'task':'写操作提示，必须提醒先断电。','answer_A':'先断电，再清洁外壳。','answer_B':'清洁外壳。',
   'criteria':'只有是否写清清洁动作；两側Judge均判满足。',
   'result':{'analysis':'Judge对现有条目均满足有依据，但标准遗漏task的断电前置要求，可补充该要求的核验方法。','candidates':[
    {'content':'涉及明确操作前置要求时，将前置条件与后续动作分别设为正向核验项，避免只检查结果动作。','trigger':'任务明确给出操作前置条件'}]}},
  {'task':'简短建议，不限制列表。','answer_A':'1. 打开窗户。','answer_B':'打开窗户。',
   'Judge_reason':'声称回答A写了“我拒绝开窗”，实际原文没有。',
   'result':{'analysis':'错误引用来自Judge解释，不能归给A。任务允许列表，没有可信标准设计缺陷，不据此创造格式规则。','candidates':[]}}],
 'experience_updater': [
  {'current':'核验记录中事件与执行者的归属。','cases':[
   {'sample_id':'1','task':'核对配送记录的执行者','supported_method':'对照记录确认动作属于哪位执行者'},
   {'sample_id':'2','task':'核对维修记录的执行者','supported_method':'对照记录确认动作属于哪位执行者'}],
   'result':{'experiences':[{'content':'核对事件与执行者的对应关系，按原始记录确认动作归属，避免把一方行为写到另一方。','trigger':'任务需要从材料识别事件及执行者','sample_ids':['1','2']}]},'why':'对象不同而核验动作相同，不产生两条近义经验。'},
  {'current':'检查数据处理是否正确。','cases':[
   {'sample_id':'1','task':'逐项转录表格','supported_method':'核对每个字段与原表一致'},
   {'sample_id':'2','task':'根据表格计算比例','supported_method':'核对分子分母定义及计算'}],
   'result':{'experiences':[
    {'content':'转录材料时逐字段对照原始值与归属，避免漏项或错配。','trigger':'任务要求逐项转录材料','sample_ids':['1']},
    {'content':'计算比例时把分子分母定义与计算结果设为独立核验项，避免只看结果格式。','trigger':'任务要求基于材料计算比例','sample_ids':['2']}]},'why':'共同主题“数据正确”不足以保留两类不同核验动作，故拆分。'},
  {'current':'建议类任务核验建议是否回应用户目标。','cases':[
   {'sample_id':'1','task':'给简短建议，不限格式','supported_method':'建议回应目标'},
   {'sample_id':'2','task':'给简短建议，不限格式','Judge_reason':'声称短建议必须无列表，此要求无task依据'}],
   'result':{'experiences':[{'content':'建议类任务核验建议是否回应用户目标。','trigger':'任务要求提供建议','sample_ids':['1','2']}]},'why':'不把错误Judge规则追加进经验，证据不足案例仍保留归属。'}]
}

def variant_messages(row, variant):
    result = deepcopy(row['messages'])
    if variant != 'baseline':
        role = row['role']
        result[0]['content'] = GUIDANCE[role] + '\n\n' + prompts._ROLE_INSTRUCTIONS[role]
        if variant == 'few_shot':
            result[0]['content'] += ('\n\n以下是独立构造的教学示例，不是当前任务或证据。'
              '编号只在示例内有效。只学习核验方法，不把示例要求复制到当前任务。\n' +
              json.dumps(EXAMPLES[role], ensure_ascii=False))
        elif variant != 'concise':
            raise ValueError('unknown prompt variant')
    if sum(len(m['content'].encode()) for m in result) > 100_000:
        raise ValueError('context limit; never truncate')
    assert result[1:] == row['messages'][1:]
    return result

def confirmation_pool(base):
    manifest = json.loads((SOURCE/'run_manifest.json').read_text())
    events = [json.loads(s) for s in (SOURCE/manifest['config']['store']['config']['root_dir']/'role_io.jsonl').read_text().splitlines()]
    rows = []
    rng = random.Random(59)
    for role in GUIDANCE:
        excluded = {r['task_fingerprint'] for r in base if r['role']==role}
        options = {}
        for line,event in enumerate(events,1):
            if event['role'] != role or event['output'] is None or event['error'] is not None: continue
            if role=='critic' and len(event['messages']) not in (2,4): continue
            payload = project(role,event['input'])
            task = payload.get('task',payload.get('cases',[{}])[-1].get('task',''))
            key = fingerprint(task)
            if key in excluded: continue
            options.setdefault(key,[]).append((line,event,payload))
        target=8 if role=='experience_updater' else 20
        if len(options)<target: raise ValueError(f'insufficient new task inputs for {role}')
        buckets = {k:[] for k in ('short','medium','long')}
        for key, values in sorted(options.items()):
            task=values[0][2].get('task',values[0][2].get('cases',[{}])[-1].get('task',''))
            buckets['short' if len(task)<250 else 'long' if len(task)>4000 else 'medium'].append((key,values))
        for values in buckets.values(): rng.shuffle(values)
        selected=[]
        while len(selected)<target:
            for bucket,values in buckets.items():
                if values and len(selected)<target:
                    key,options_for_task=values.pop()
                    # Preserve real later business turns when available, without repeating a task.
                    line,event,payload=options_for_task[-1 if len(selected)%2 else 0]
                    selected.append((key,line,event,payload,bucket))
        for index,(key,line,event,payload,bucket) in enumerate(selected,1):
            if role=='critic':
                first=event
                if len(event['messages'])==4:
                    matches=[e for e in events[:line-1] if e['role']==role and e['sample_id']==event['sample_id']
                              and e['messages']==event['messages'][:2]]
                    if len(matches)!=1: raise ValueError('ambiguous Critic history')
                    first=matches[0]
                messages=paired_messages(event,first,'unused')[0]
            else: messages=prompts.llm_messages(role,payload)
            rows.append({'role':role,'case_key':f'{index:02d}','trace_line':line,'sample_id':event['sample_id'],
                'task_fingerprint':key,'stratum':bucket,'historical_input':event['input'],
                'input':payload,'historical_output':event['output'],'messages':messages,'variant':'baseline'})
    return rows

def prepare(stage, choices_path=None):
    if stage=='screening':
        output=SCREEN
        base=[deepcopy(r) for r in json.loads((PRIOR/'prepared_inputs.json').read_text()) if r['variant']=='production']
        variants={role:['baseline','concise','few_shot'] for role in GUIDANCE}
        pool=confirmation_pool(base)
        protected=json.loads((PRIOR/'replay_manifest.json').read_text())['source_sha256']
        # Historical experiment remains read-only; current files are frozen afresh for this experiment.
        files=[Path(p) for p in protected]+[PRIOR/'prepared_inputs.json',PRIOR/'judge_observable_reference.json']
    else:
        from scripts.run_module_prompt_replay import verify
        verify(SCREEN)
        output=CONFIRM
        base=json.loads((SCREEN/'confirmation_pool.json').read_text())
        choices=json.loads(choices_path.read_text())
        if set(choices)!=set(GUIDANCE) or any(v not in ('concise','few_shot') for v in choices.values()):
            raise ValueError('one preselected challenger per module required')
        variants={role:['baseline',choices[role]] for role in GUIDANCE}
        pool=[]
        files=[Path(p) for p in json.loads((SCREEN/'replay_manifest.json').read_text())['source_sha256']]
        files += [SCREEN/'prepared_inputs.json',SCREEN/'confirmation_pool.json',choices_path]
    rows=[]
    for row in base:
        for variant in variants[row['role']]:
            value=deepcopy(row);value['variant']=variant;value['messages']=variant_messages(row,variant);rows.append(value)
    for role in GUIDANCE:
        originals=[r for r in base if r['role']==role]
        expected=8 if stage=='confirmation' and role=='experience_updater' else 20
        if len(originals)!=expected or len({r['task_fingerprint'] for r in originals})!=expected:
            raise ValueError('incorrect task-distinct input coverage')
    output.mkdir(parents=True,exist_ok=False);output.chmod(0o700)
    save(output/'prepared_inputs.json',rows)
    if pool: save(output/'confirmation_pool.json',pool)
    save(output/'prompt_variants.json',{'guidance':GUIDANCE,'original_examples':EXAMPLES})
    files += [Path(__file__),Path('scripts/run_module_prompt_replay.py'),Path('scripts/run_module_model_replay.py'),
              output/'prompt_variants.json']
    if pool: files.append(output/'confirmation_pool.json')
    save(output/'replay_manifest.json',{'source_sha256':{str(p.resolve()):digest(p) for p in files},
        'input_sha256':digest(output/'prepared_inputs.json'),'roles':list(GUIDANCE),'variants':variants,
        'models':{'qwen':json.loads((PRIOR/'replay_manifest.json').read_text())['models']['qwen']},
        'stage':stage,'planned_outputs':len(rows),'unique_tasks_per_role':{
          role:8 if stage=='confirmation' and role=='experience_updater' else 20 for role in GUIDANCE},'production_unchanged':True,
        'scope':'Independent role calls, complete evidence and real original Critic history; only system differs; no pipeline/pool writes.',
        'evaluation':'Task-grounded semantic wins/losses, normal behavior and necessary actions retained; no universal optimum claim.',
        'budget':'4 process workers; up to3 backend attempts times3 structure attempts per output; fail-fast infrastructure/program errors.',
        'confirmation_selection':'Frozen before screening outputs;20 further unique tasks per role except Updater only8 available; stratified by task length seed59; Updater packs may overlap source cases.',
        'examples_provenance':'Independently authored fictional tasks; not dataset extracts or rewrites.'})
    print(f'prepared {stage}: {len(rows)} outputs')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['screening','confirmation'],default='screening')
    parser.add_argument('--choices',type=Path);args=parser.parse_args()
    if args.stage=='confirmation' and not args.choices: parser.error('--choices required')
    prepare(args.stage,args.choices)
