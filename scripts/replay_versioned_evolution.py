"""Isolated plan67 replays. No PipelineRunner, no source-memory writes.

Historical protocol8 JSON is deliberately projected for testing only; production
readers continue to reject it. Missing five-case windows are reported explicitly.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
from types import SimpleNamespace
from verge.models import (AtomicMethod,AtomicResult,CategoryAssignment,CategoryDescription,
                          ExperienceUpdate,Experience,ExperienceCategory,CandidateExperience,Criterion,Rubric,RubricJudgment)
from verge.modules import (LLMAtomizer,LLMExperienceUpdater,LLMCategoryOrganizer,LLMCategoryDescriber,
                           LLMSelector,LLMInitialRubricGenerator,LLMCritic,HybridRetriever)
from verge.prompts import (PROMPT_VERSION,experience_update_payload,ExperienceView,NumberedIDCodec)
from verge.llm import OpenAICompatibleBackend,OpenAICompatibleEmbedding,LLMSampleError

SOURCE=Path('runs/helpsteer3-general-atomic-experience-qwen-flash-200')
DEST=Path('runs/versioned-evolution-module-replay-67')
ROLES=['atomizer','experience_updater','category_organizer','category_describer','selector','initial_generator','critic']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');p.chmod(0o600)
def readrows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def unique(rows,key):
    seen=set();out=[]
    for r in rows:
        k=key(r)
        if k not in seen:out.append(r);seen.add(k)
    return out

def prepare():
    if DEST.exists():raise ValueError('replay directory already exists; do not overwrite')
    root=SOURCE/'artifacts'
    memory=json.loads((root/'experience_memory.json').read_text());tr=readrows(root/'trajectories.jsonl')
    roles=readrows(root/'role_io.jsonl');samples=readrows(Path('runs/helpsteer3-general-consensus-data/train.jsonl'))
    tasks={r['sample_id']:r for r in samples};order={r['sample_id']:i for i,r in enumerate(tr)}
    old={e['experience_id']:e for e in memory['experiences']}
    cases=[]
    def add(role,sid,payload,baseline=None,scope='target_contract',note=''):
        cases.append({'key':f'{role}-{sum(c["role"]==role for c in cases)+1:02}', 'role':role,'sample_id':sid,
                      'input':payload,'baseline':baseline,'scope':scope,'conversion':note})
    # Candidate atomization: unique real tasks, long/multi-action and normal cases mixed.
    critics=unique([r for r in roles if r['role']=='critic' and (r.get('output') or {}).get('candidates')],lambda r:tasks[r['sample_id']]['task'])
    priority=sorted(critics,key=lambda r:-len(r['output']['candidates'][0]['content']))
    selected=priority[:10]+[r for r in critics if r not in priority[:10]][:10]
    for r in selected:
        add('atomizer',r['sample_id'],{'draft':r['output']['candidates'][0]},r['output']['candidates'][0],note='Actual Critic output extracted; no task text or credit sent to Atomizer.')
    # Revision source includes actual updates; fill from historical versions as clearly labeled identity controls.
    revisions=unique([r for r in roles if r['role']=='experience_updater' and r.get('output')],lambda r:tasks[r['sample_id']]['task'])
    seen={r['sample_id'] for r in revisions}
    revision_rows=[(r['sample_id'],r['output'],r['input']['current_experience'],'historical_update') for r in revisions]
    for e in memory['experiences']:
        sid=e['cases'][0]['sample_id']
        if sid not in seen:
            revision_rows.append((sid,{'content':e['cases'][0]['content'],'trigger':e['cases'][0]['trigger']},None,'identity_control'))
            seen.add(sid)
    for sid,draft,previous,scope in revision_rows[:20]:add('atomizer',sid,{'draft':draft,'previous':previous},draft,scope=scope,note='Update outputs when available; remaining are actual source texts used as identity controls, not fabricated Updater outputs.')
    # Updater: three complete same-text five-case windows, then real short windows as diagnostics.
    groups=defaultdict(list)
    for r in tr:
        for v in r['selected_experiences']:
            c=r['downstream_credits'].get(v['experience_id'],{})
            if c.get('positive_credit',0)+c.get('negative_credit',0)>0:
                key=(v['experience_id'],v['content'],v['trigger'])
                groups[key].append((r,v,c))
    target=[];diagnostic=[]
    for (eid,content,trigger),events in groups.items():
        for n in range(1,len(events)+1):
            if n<2:continue
            chosen=events[max(0,n-5):n]
            cards=[]
            for r,v,credit in chosen:
                raw=tasks[r['sample_id']]
                cards.append(SimpleNamespace(sample_id=r['sample_id'],task=raw['task'],chosen=raw['chosen'],rejected=raw['rejected'],
                   source_id=eid,content=content,trigger=trigger,rubric=Rubric.from_dict(r['initial_rubric']),judgment=RubricJudgment.from_dict(r['initial_judgment'])))
            payload=experience_update_payload(SimpleNamespace(content=content,trigger=trigger,working_cases=cards))
            item=(events[n-1][0]['sample_id'],payload)
            (target if n>=5 else diagnostic).append(item)
    chosen=unique(target,lambda x:tasks[x[0]]['task'])
    seen={sid for sid,_ in chosen}
    for sid,payload in diagnostic:
        if sid not in seen and len(chosen)<20:chosen.append((sid,payload));seen.add(sid)
    for sid,payload in chosen[:20]:add('experience_updater',sid,payload,scope='target_contract' if len(payload['cases'])==5 else 'short_window_diagnostic',note='Reconstructed actual same-ID AND identical trigger/content history. Fewer than five is diagnostic only; production adapter requires five.')
    def snapshot(index):
        ex=[]
        for e in memory['experiences']:
            if order[e['cases'][0]['sample_id']]>index:continue
            content,trigger=e['cases'][0]['content'],e['cases'][0]['trigger']
            for u in e['update_history']:
                if order[u['sample_id']]<=index and not u['error']:content,trigger=u['content'],u['trigger']
            ex.append(Experience(e['experience_id'],content,trigger,1,1,[],[]))
        past=[h for h in memory['organization_history'] if order[h['sample_id']]<=index and not h['error']]
        cats=[]
        if past:
            h=past[-1];cats=[ExperienceCategory.from_dict(c) for c in h['categories']]
            for e in ex:e.category_id=h['assignments'].get(e.experience_id)
        return ex,cats
    # Classification: 20 independent origin tasks, frozen historical pools (no outputs fed to other cases).
    for e in memory['experiences'][:20]:
        sid=e['cases'][0]['sample_id'];ex,cats=snapshot(order[sid]);target=next(x for x in ex if x.experience_id==e['experience_id'])
        add('category_organizer',sid,{'target':target.to_dict(),'pool':[x.to_dict() for x in ex],'categories':[c.to_dict() for c in cats]},scope='retrieval_input_pending',note='Snapshot reconstructed from preexisting birth/update/classification events up to origin task, excludes future metadata. Retrieval is frozen independently per case.')
    # Description changes: actual organizer membership changes broken into per-category delta inputs.
    previous={};previous_cats={};changes=[]
    for h in memory['organization_history']:
        if h['error']:continue
        cats={c['category_id']:c for c in h['categories']};assign=h['assignments'];ex,_=snapshot(order[h['sample_id']]);by={e.experience_id:e for e in ex}
        for cid,c in cats.items():
            delta=[]
            for eid,new in assign.items():
                oldcid=previous.get(eid)
                if oldcid==new:continue
                if cid==new or cid==oldcid:
                    e=by[eid];delta.append({'action':'added' if cid==new else 'removed','experience_id':eid,'content':e.content,'trigger':e.trigger})
            if delta:changes.append((h['sample_id'],{'category':previous_cats.get(cid,c),'changes':delta,'member_count':sum(x==cid for x in assign.values())}))
        previous,previous_cats=assign,cats
    # At most 7 organizer source tasks: cannot claim 20 task-dedup genuine description events.
    chosen_changes=unique(changes,lambda x:tasks[x[0]]['task'])
    seen_tasks={tasks[sid]['task'] for sid,_ in chosen_changes}
    for e in memory['experiences']:
        sid=e['cases'][0]['sample_id']
        if tasks[sid]['task'] in seen_tasks or not e['category_id']:continue
        category=next(c for c in memory['categories'] if c['category_id']==e['category_id'])
        chosen_changes.append((sid,{'category':category,'changes':[{'action':'added','experience_id':e['experience_id'],'content':e['content'],'trigger':e['trigger']}], 'member_count':sum(x['category_id']==e['category_id'] for x in memory['experiences']), 'projection':'single_member_delta_control'}))
        seen_tasks.add(tasks[sid]['task'])
        if len(chosen_changes)>=20:break
    for sid,payload in chosen_changes[:20]:
        scope='single_member_delta_control' if payload.get('projection') else 'real_batch_delta'
        add('category_describer',sid,{k:v for k,v in payload.items() if k!='projection'},scope=scope,note='Historical deltas deduped by source task; supplemental single-member controls use actual member/final category metadata, not historical Describer calls. Explicit context reconstruction.')
    for role in ['selector','initial_generator','critic']:
        selected=unique([r for r in roles if r['role']==role and r.get('status')==200 and r.get('output') and (role!='critic' or r['input']['round_number']==1)],lambda r:tasks[r['sample_id']]['task'])
        # Fixed strided coverage across the run, rather than just the first twenty tasks.
        picked=[selected[i*(len(selected)-1)//19] for i in range(20)]
        for r in picked:
            payload=r['input']
            if role=='initial_generator' and 'candidate_experience' in payload:
                payload={**payload,'candidate_experiences':[payload['candidate_experience']]};del payload['candidate_experience']
            add(role,r['sample_id'],payload,r['output'],note='Actual historical input; Generator single candidate explicitly converted to one-item collection. No model outputs propagate between cases.')
    protected=[SOURCE/'run_manifest.json',root/'experience_memory.json',root/'trajectories.jsonl',root/'role_io.jsonl',root/'llm_requests.jsonl',Path('runs/helpsteer3-general-consensus-data/train.jsonl')]
    DEST.mkdir(parents=True);DEST.chmod(0o700)
    write(DEST/'inputs.json',cases)
    write(DEST/'manifest.json',{'source':str(SOURCE),'source_sha256':{str(p):sha(p) for p in protected},'input_sha256':sha(DEST/'inputs.json'),
         'code_version':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'code_sha256':{str(p):sha(p) for p in Path('verge').glob('*.py')},'prompt_version':PROMPT_VERSION,
         'model':'qwen3.5-flash','enable_thinking':False,'scope_counts':dict(Counter((c['role']+':'+c['scope']) for c in cases)),
         'limitations':['Only three complete five-case same-text windows exist. Short windows are diagnostics, not target validation.',
           'Only seven genuine category-organization tasks exist; description coverage includes explicitly derived single-member delta controls.',
           'Current historical pools have fewer than twenty retrieval units; Top20 cannot establish retrieval improvement.'],
         'max_valid_target_calls':180,'pipeline_run':False})
    print(json.dumps({'prepared':len(cases),'counts':dict(Counter(c['role'] for c in cases)),'target_updater_windows':sum(c['role']=='experience_updater' and c['scope']=='target_contract' for c in cases)},ensure_ascii=False))

def run():
    manifest=json.loads((DEST/'manifest.json').read_text());cases=json.loads((DEST/'inputs.json').read_text())
    for p,h in manifest['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('source changed')
    if sha(DEST/'inputs.json')!=manifest['input_sha256']:raise ValueError('inputs changed')
    if any(sha(Path(p))!=h for p,h in manifest['code_sha256'].items()):raise ValueError('production code changed since frozen inputs')
    lock=Lock()
    def one(case):
        path=DEST/'results'/(case['key']+'.json')
        if path.exists():return json.loads(path.read_text())
        def append(name,event):
            with lock:
                target=DEST/name
                with target.open('a') as f:target.chmod(0o600);f.write(json.dumps({'case_key':case['key'],**event},ensure_ascii=False)+'\n')
        spec={'model':'qwen3.5-flash','base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key_env':'DASHSCOPE_API_KEY_2',
              'enable_thinking':False,'timeout_seconds':120,'max_retries':2}
        b=OpenAICompatibleBackend(**spec,trace_hook=lambda e:append('role_io.jsonl',e),metadata_hook=lambda e:append('llm_requests.jsonl',e))
        embed=None;role=case['role'];p=case['input'];start=time.monotonic();out=None;err=None;projected=p
        try:
            if role=='atomizer':out=LLMAtomizer(b).atomize(AtomicMethod.from_dict(p['draft']),previous=AtomicMethod.from_dict(p['previous']) if p.get('previous') else None).to_dict()
            elif role=='experience_updater':
                # Same production call/parser, but diagnostic input lengths deliberately bypass adapter precondition.
                out=LLMExperienceUpdater(b).call(role,p,ExperienceUpdate).to_dict()
            elif role=='category_organizer':
                embed=OpenAICompatibleEmbedding('text-embedding-v4',base_url=spec['base_url'],api_key_env=spec['api_key_env'],dimensions=1024,timeout_seconds=120,max_retries=2)
                ex=[Experience.from_dict(x) for x in p['pool']];cats=[ExperienceCategory.from_dict(x) for x in p['categories']];e=Experience.from_dict(p['target'])
                r=HybridRetriever(embed);candidates=r.category_candidates(e,ex,cats);projected={'target':p['target'],'candidates':candidates}
                out=LLMCategoryOrganizer(b).organize(e,candidates).to_dict();append('retrieval.jsonl',{'input':projected,'trace':r.last_trace})
            elif role=='category_describer':out=LLMCategoryDescriber(b).describe(ExperienceCategory.from_dict(p['category']),p['changes'],p['member_count']).to_dict()
            elif role=='selector':
                views=[]
                for unit in p['candidates']:
                    category=ExperienceCategory(unit['category_id'],unit['name'],unit['description']) if 'members' in unit else None
                    for x in unit.get('members',[unit]):views.append(ExperienceView(x['experience_id'],x['content'],x['trigger'],x['confidence'],x['evidence_amount'],category))
                out={'selected_experience_ids':[v.experience_id for v in LLMSelector(b,max_selected=p['max_selected']).select(p['task'],views,training=p.get('training',True))]}
                validate_selection(p,out)
            elif role=='initial_generator':
                views=[ExperienceView(x['experience_id'],x['content'],x['trigger'],.5,0) for x in p['selected_experiences']]
                cs=[CandidateExperience(x['experience_id'],x['content'],x['trigger']) for x in p.get('candidate_experiences',[])]
                gen=LLMInitialRubricGenerator(b)
                out=(gen.apply(p['task'],views,cs,1) if cs else gen.generate(p['task'],views)).to_dict()
            elif role=='critic':
                out=LLMCritic(b).propose(p['task'],p['chosen'],p['rejected'],Rubric.from_dict(p['current_rubric']),RubricJudgment.from_dict(p['current_judgment']),p['round_number'],
                     previous_candidate=CandidateExperience.from_dict(p['previous_candidate']) if p.get('previous_candidate') else None,adopted=p.get('adopted'),invalid_error=p.get('invalid_error')).to_dict()
        except (ValueError,LLMSampleError) as exc:err=type(exc).__name__+': '+str(exc)
        finally:
            b.close()
            if embed:embed.close()
        row={**case,'actual_input':projected,'output':out,'error':err,'elapsed_seconds':round(time.monotonic()-start,2)}
        write(path,row);print(case['key']+(' OK' if not err else ' INVALID'),flush=True);return row
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows=[f.result() for f in as_completed([executor.submit(one,c) for c in cases])]
    for p,h in manifest['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('source changed')
    summary={'cases':len(rows),'valid':sum(not r['error'] for r in rows),'roles':{role:{'cases':sum(r['role']==role for r in rows),'valid':sum(r['role']==role and not r['error'] for r in rows)} for role in ROLES},'source_unchanged':True,'limitations':manifest['limitations']}
    write(DEST/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

# Keep role output validation independent of full-pipeline objects.
from dataclasses import dataclass
from verge.models import Model
@dataclass
class SelectionOutput(Model):
    selected_experience_ids:list[str]

def validate_selection(payload,out):
    valid={m['experience_id'] for unit in payload['candidates'] for m in unit.get('members',[unit])}
    ids=out['selected_experience_ids']
    if len(ids)>payload['max_selected'] or len(ids)!=len(set(ids)) or set(ids)-valid:raise ValueError('selection outside candidates')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);a=parser.parse_args()
    prepare() if a.mode=='prepare' else run()
