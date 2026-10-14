"""Freeze 20 task-distinct historical inputs per module before any comparison calls."""
from copy import deepcopy
import json
from pathlib import Path
from scripts.replay_experience_splits import digest, fingerprint, save
from scripts.replay_critic_grounded_few_shot import paired_messages
from scripts.replay_simplified_inputs import simplified_payload
from verge import prompts

SOURCE=Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3')
OUTPUT=Path('runs/module-model-replay-20')
# Explicitly selected prior to seeing new model outputs. Comments describe coverage, not gold labels.
SELECTION={
 'selector':[(24,'boundary','ASCII任务与故事经验的主题/操作区分'),(48,'control','SEO元数据且可不选'),
 (68,'boundary','品牌咨询与资料缺失'),(112,'problem','故事动作/状态方法适用性'),
 (154,'control','艺术家简介及写作风格'),(159,'problem','角色短简介与场景要求'),
 (184,'control','短文案和少选'),(199,'boundary','文学词义而非故事生成'),
 (211,'problem','未填写占位符'),(234,'problem','吉他推荐与榜单边界'),
 (277,'control','明确故事延续'),(305,'boundary','指令分析与材料操作区别'),
 (324,'problem','角色存在/知识边界'),(386,'problem','物种范围与叙事任务'),
 (428,'boundary','时效信息与口径'),(457,'problem','一词品牌/符号格式'),
 (517,'problem','信息提取与材料主题'),(537,'boundary','长材料实体提取'),
 (619,'problem','词表/篇幅与经验适用'),(720,'problem','具体读听追问与泛化榜单经验')],
 'initial_generator':[(1,'control','无经验的自我/他人建议'),(25,'boundary','无经验ASCII图形'),
 (49,'control','无经验SEO输出'),(69,'boundary','无经验品牌信息咨询'),
 (113,'problem','已有经验落实故事动作'),(160,'problem','短简介/属性约束与经验关系'),
 (212,'problem','占位符不等于可任意补造'),(235,'problem','推荐目标与信息边界'),
 (256,'boundary','任务给定数值的战争叙事'),(306,'problem','只分析材料时不要求完成写作'),
 (325,'problem','已有方法条件与角色事实'),(387,'problem','内部物种范围和故事外观察者'),
 (429,'boundary','人口统计时效/口径'),(518,'problem','提取任务不引入材料主题要求'),
 (620,'boundary','给定词表/大约篇幅'),(190,'problem','当前候选应用：短文案'),
 (204,'boundary','当前候选应用：词义分析'),(485,'control','当前候选应用：三天训练安排'),
 (510,'control','当前候选应用：原创隐喻'),(695,'problem','当前候选应用：100项排名')],
 'judge':[(2,'control','自我/他人建议，chosen'),(26,'boundary','ASCII图形，rejected'),
 (70,'boundary','专业品牌举例，chosen'),(161,'problem','人物细节/动态，rejected'),
 (213,'problem','占位符咨询，chosen'),(236,'problem','吉他推荐，rejected'),
 (257,'boundary','明确数值设定，rejected'),(279,'control','暴雨帐篷故事，chosen'),
 (307,'problem','材料分析指令，chosen'),(326,'boundary','角色对比/领域事实，chosen'),
 (366,'control','简单作曲家事实，rejected'),(389,'problem','世界内外物种/发现过程，rejected'),
 (431,'boundary','人口时效及口径，rejected'),(466,'problem','品牌数量/词数/符号，chosen'),
 (520,'problem','逐字提取边界，rejected'),(541,'problem','长材料多个地点实体，chosen'),
 (597,'control','算术及游戏语境，rejected'),(622,'boundary','词表/大约150词，chosen'),
 (641,'control','两段故事，rejected'),(723,'problem','读听追问是否被直接讨论，chosen')],
 'critic':[(4,'control','两答均有方法，不因章节差异强造要求'),(32,'problem','ASCII二轮真实历史'),
 (77,'boundary','专业品牌二轮真实历史'),(167,'problem','人物内容归属二轮真实历史'),
 (189,'problem','短文案与方向假设'),(203,'boundary','文学代词分析'),
 (215,'control','占位符与无可信经验'),(243,'boundary','五种语言/排除范围'),
 (281,'control','两答核心故事均满足'),(314,'problem','材料分析二轮真实历史'),
 (382,'boundary','极短模糊指令'),(390,'problem','内外世界范围及无依据原创限制'),
 (437,'boundary','旧新剧版本歧义'),(476,'boundary','两个商业情形和例子'),
 (484,'problem','训练方案质量与硬门槛'),(509,'problem','原创隐喻不等于固定句式'),
 (521,'problem','提取与逐字复述边界'),(623,'boundary','词表/近似篇幅'),
 (694,'problem','排名完整性/数目'),(724,'problem','读听子问题与Judge误判')],
 'experience_updater':[(128,'problem','故事动作共同方法'),(145,'problem','场景和角色不同不自动拆分'),
 (238,'control','推荐方法保持一条'),(269,'boundary','数学解释与相关方法边界'),
 (286,'problem','叙事完整性/状态方法'),(298,'problem','人格分析与文学方法区分'),
 (323,'problem','创作与材料归纳不同动作'),(329,'problem','实体知识和论证方法'),
 (348,'boundary','多个游戏人物共同方法'),(377,'control','短宣传稿保持一条'),
 (427,'problem','电影解释不同证据动作'),(467,'problem','数量与格式的实质差异'),
 (542,'problem','实体提取与叙事归纳'),(553,'control','四案例保持一条'),
 (585,'problem','五案例摘要与创作边界'),(592,'problem','业务请求与信息缺失'),
 (635,'control','论证比较保持一条'),(642,'boundary','明确两段参数'),
 (702,'problem','五案例吉他与其他方法'),(730,'problem','五案例题面约束与叙事方法')]
}
FACT_GUIDANCE=('你是Critic模块的事实核对诊断员。本次只核对输入中的task要求、标准文本、两回答分别写了什么、'
 '两侧Judge的实际判定和理由，以及已有历史分析的关键事实是否准确。Judge理由和历史分析可能错误，须与原文对照。'
 '在现有analysis中简洁报告可核对的事实及发现的不一致；不设计新标准，不讨论权重调整，不提炼经验，不依据偏好身份补造事实。'
 '以JSON返回现有结构：{"analysis":"简洁事实核对","candidates":[]}，candidates必须为空。不增加字段。')
FACT_INSTRUCTION='本次改为独立事实核对诊断：只核对上述证据及历史分析中的关键事实，不提出经验或标准修订；返回analysis及空candidates。'


def project(role,payload):
    value=deepcopy(payload)
    if role in ('experience_updater','initial_generator'):
        value=simplified_payload(role,value)
    if role=='experience_updater':
        value=prompts.NumberedIDCodec(role,value).project(value)
    return value


def prepare():
    manifest_path=SOURCE/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    root=SOURCE/manifest['config']['store']['config']['root_dir']
    traces=root/'role_io.jsonl'
    events=[json.loads(x) for x in traces.read_text().splitlines()]
    rows=[];selection=[]
    for role,cases in SELECTION.items():
        task_ids=[]
        for index,(line,stratum,note) in enumerate(cases,1):
            event=events[line-1]
            if event['role']!=role or event['output'] is None or event['error'] is not None:
                raise ValueError(f'{role}:{line} must be valid historical event')
            payload=project(role,event['input'])
            if role=='critic':
                first=event
                if len(event['messages'])==4:
                    matches=[e for e in events[:line-1] if e['role']==role and e['sample_id']==event['sample_id']
                             and e['messages']==event['messages'][:2]]
                    if len(matches)!=1:raise ValueError('first history source ambiguous')
                    first=matches[0]
                messages=paired_messages(event,first,'unused')[0]
            else:messages=prompts.llm_messages(role,payload)
            task=payload.get('task',payload.get('cases',[{}])[-1].get('task',''))
            task_ids.append(fingerprint(task))
            base={'role':role,'case_key':f'{index:02d}','trace_line':line,'sample_id':event['sample_id'],
                  'task_fingerprint':task_ids[-1],'stratum':stratum,'selection_reason':note,
                  'historical_input':event['input'],'input':payload,'historical_output':event['output'],
                  'historical_response_side':event.get('response_side'),'messages':messages,'variant':'production'}
            if role=='experience_updater':base['case_count']=len(payload['cases'])
            selection.append({k:v for k,v in base.items() if k not in ('historical_input','input','historical_output','messages')})
            rows.append(base)
            if role=='critic':
                diagnostic=deepcopy(base);diagnostic['variant']='facts_only'
                diagnostic['messages'][0]['content']=FACT_GUIDANCE
                diagnostic['messages'][-1]['content']+='\n\n'+FACT_INSTRUCTION
                rows.append(diagnostic)
        if len(cases)!=20 or len(set(task_ids))!=20:raise ValueError(f'{role} requires20 distinct task inputs')
    if len(rows)!=120:raise ValueError('requires100 production +20 diagnostic rows')
    if any(sum(len(x['content'].encode()) for x in r['messages'])>100_000 for r in rows):
        raise ValueError('context budget exceeded; never truncate')
    OUTPUT.mkdir(parents=True,exist_ok=False);OUTPUT.chmod(0o700)
    save(OUTPUT/'selection.json',selection)
    save(OUTPUT/'prepared_inputs.json',rows)
    protected=[manifest_path,traces,root/'experience_memory.json',root/'trajectories.jsonl',Path(prompts.__file__),Path(__file__),Path('verge/llm.py'),Path('scripts/run_module_model_replay.py')]
    save(OUTPUT/'replay_manifest.json',{'source_run':str(SOURCE.resolve()),'source_sha256':{str(p.resolve()):digest(p) for p in protected},
      'selection_sha256':digest(OUTPUT/'selection.json'),'input_sha256':digest(OUTPUT/'prepared_inputs.json'),
      'prompt_version':prompts.PROMPT_VERSION,'planned_outputs':240,'roles':list(SELECTION),'unique_tasks_per_role':20,
      'models':{
       'qwen':{'model':'qwen3.5-flash','base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key_env':'DASHSCOPE_API_KEY_2',
        'timeout_seconds':120,'max_retries':2,'enable_thinking':False,'request_extras':{'max_tokens':8192}},
       'deepseek':{'model':'deepseek-flash','base_url':'https://api.deepseek.com','api_key_env':'DEEPSEEK_API_KEY',
        'timeout_seconds':120,'max_retries':2,'enable_thinking':None,'request_extras':{'thinking':{'type':'disabled'},'max_tokens':8192}}},
      'scope':'200 production outputs +40 Critic facts-only diagnostics; no pipeline or pool writes; user explicitly authorizes DeepSeek over prior Qwen-only convention',
      'conversions':'Generator removes historical reliability; Updater removes credit/weight/source_count and uses short sample IDs; Critic uses current readable layout and actual original assistant; other roles full evidence; no truncation',
      'budget':'each output at most3 backend attempts per structure call, at most3 structure calls; max4 worker processes; infra/program errors fail-fast after existing bounded backend retries',
      'limitations':'curated stress/control suite from one historical100-task run, not population frequency or end-to-end accuracy; Updater packs overlap cases and cover2-5 cases, not saturation; Critic second turns inherit Qwen historical assistant for both models',
      'official_api_sources':['https://api-docs.deepseek.com/','https://api-docs.deepseek.com/guides/thinking_mode/']})
    print('prepared100 production inputs +20 fact-only diagnostics;240 outputs;20 distinct tasks/role')

if __name__=='__main__':prepare()
