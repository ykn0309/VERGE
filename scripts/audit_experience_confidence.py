"""Read-only reconstruction of observed experience use, Beta and lineage."""
from collections import Counter, defaultdict
from copy import deepcopy
from math import fsum, isclose
import json
from pathlib import Path

from scripts.replay_experience_splits import digest, save
from verge.calculator import calculate_experience_credits, calculate_rubric_judgment, positive_experience_sources
from verge.memory import ExperienceMemory
from verge.models import Credit, Experience, Rubric, RubricJudgment

RUN=Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3')
OUTPUT=Path('runs/experience-confidence-audit-100')

def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def close(a,b):return isclose(a,b,rel_tol=0,abs_tol=1e-9)
def require(condition,reason):
    if not condition:raise AssertionError(reason)
def confidence(e):return e['alpha']/(e['alpha']+e['beta'])
def root_id(key,final):
    while final[key]['parent_id']:key=final[key]['parent_id']
    return key

def credited_criteria(rubric,judgment,source):
    answers={x['criterion_id']:x for x in judgment['criterion_judgments']}
    total=fsum(c['weight'] for c in rubric['criteria'])
    return [{**c,**answers[c['criterion_id']],
             'weight_share':c['weight']/total/len(c['source_experiences'])}
            for c in rubric['criteria'] if source in c['source_experiences']]

def main():
    manifest=json.loads((RUN/'run_manifest.json').read_text())
    source=RUN/manifest['config']['store']['config']['root_dir']
    data_path=Path(manifest['config']['dataset']['config']['paths']['train'])
    protected=[RUN/'run_manifest.json',source/'trajectories.jsonl',source/'experience_memory.json',
               source/'role_io.jsonl',source/'llm_requests.jsonl',data_path,Path(__file__)]
    protected += list(Path('verge').glob('*.py'))+[Path('AGENTS.md')]
    hashes={str(p.resolve()):digest(p) for p in protected}
    trajectories=rows(source/'trajectories.jsonl');data=rows(data_path)
    require(len(trajectories)==100,'expected observed100 prefix')
    require([r['sample_id'] for r in trajectories]==[r['sample_id'] for r in data[:100]],'sample order')
    dataset={r['sample_id']:r for r in data[:100]}
    memory=ExperienceMemory(source,read_only=True)
    final={e.experience_id:e.to_dict() for e in memory.all(include_retired=True)}
    index={r['sample_id']:i for i,r in enumerate(trajectories,1)}
    state={};stats={};events=[];updates=[];exposures=[];snapshots=[]
    expected_evidence=defaultdict(list);birth_text={}

    def register(e,born):
        key=e['experience_id'];require(key not in state,'duplicate birth')
        state[key]=deepcopy(e);birth_text[key]=(e['content'],e['trigger'])
        stats[key]={'experience_id':key,'root_id':root_id(key,final),'parent_id':e['parent_id'],
          'birth_sample':born,'birth_confidence':confidence(e),'birth_alpha':e['alpha'],'birth_beta':e['beta'],
          'retrieved':0,'selected':0,'cited':0,'neutral_uses':0,'positive_events':0,'negative_events':0,
          'retired_sample':None,'events':[],'versions':[{ 'sample_index':born,'text_version':e['text_version'],
            'content':e['content'],'trigger':e['trigger'],'confidence':confidence(e),'cause':'birth'}]}

    def evidence_event(key,sid,kind,p,n,rubric,judgment,source_id,sample_index,round_number=None):
        e=state[key];before=confidence(e)
        e['alpha']+=p;e['beta']+=n
        after=confidence(e)
        original=dataset[sid]
        event={'event_key':f'{sample_index:03d}:{kind}:{key}','sample_index':sample_index,
          'sample_id':sid,'experience_id':key,'root_id':root_id(key,final),'evidence_type':kind,
          'round_number':round_number,'text_version':e['text_version'],'content':e['content'],'trigger':e['trigger'],
          'positive_credit':p,'negative_credit':n,'confidence_before':before,'confidence_after':after,
          'confidence_delta':after-before,'net_direction':'positive' if p>n else 'negative' if n>p else 'tie',
          'confidence_direction':'up' if after>before+1e-12 else 'down' if after<before-1e-12 else 'equal',
          'task':original['task'],'chosen':original['chosen'],'rejected':original['rejected'],
          'criteria':credited_criteria(rubric,judgment,source_id),'overall_success':judgment['success']}
        events.append(event);stats[key]['events'].append(event['event_key'])
        if kind=='downstream_credit':
            stats[key]['positive_events']+=p>n;stats[key]['negative_events']+=n>p
        expected_evidence[key].append((sid,kind,p,n,e['text_version']))

    for sample_index,record in enumerate(trajectories,1):
        sid=record['sample_id'];original=dataset[sid]
        for key in record['retrieved_experience_ids']:
            require(key in state and not state[key]['child_ids'],'retrieval of absent/retired experience')
            stats[key]['retrieved']+=1
        selected={v['experience_id']:v for v in record['selected_experiences']}
        for key,view in selected.items():
            e=state[key];stats[key]['selected']+=1
            require((view['content'],view['trigger'])==(e['content'],e['trigger']),'selected text mismatch')
            require(close(view['confidence'],confidence(e)),'selected confidence mismatch')
        if record['initial_judgment']:
            r=Rubric.from_dict(record['initial_rubric']);j=RubricJudgment.from_dict(record['initial_judgment'])
            require(calculate_rubric_judgment(r,j.criterion_judgments)==j,'R0 judgment recomputation')
            credits=calculate_experience_credits(r,j)
            require({key:c.to_dict() for key,c in credits.items()}==record['downstream_credits'],'R0 credit recomputation')
            for key,view in selected.items():
                used=credited_criteria(record['initial_rubric'],record['initial_judgment'],key)
                stats[key]['cited']+=bool(used)
                neutral=bool(used) and key not in credits;stats[key]['neutral_uses']+=neutral
                exposures.append({'sample_index':sample_index,'sample_id':sid,'experience_id':key,
                  'root_id':root_id(key,final),'text_version':state[key]['text_version'],'task':original['task'],
                  'selected':True,'cited':bool(used),'numerical_evidence':key in credits,'neutral':neutral,
                  'content':view['content'],'trigger':view['trigger'],'criteria':used})
            for key,c in credits.items():evidence_event(key,sid,'downstream_credit',c.positive_credit,c.negative_credit,
                 record['initial_rubric'],record['initial_judgment'],key,sample_index)
        else:
            require(not record['downstream_credits'],'unknown Judge must not credit')
            for key,view in selected.items():exposures.append({'sample_index':sample_index,'sample_id':sid,'experience_id':key,
                 'root_id':root_id(key,final),'task':original['task'],'selected':True,'cited':None,
                 'numerical_evidence':False,'neutral':False,'unknown_judge':True})
        for app in record['application_attempts']:
            if not app['judgment']:continue
            r=Rubric.from_dict(app['rubric']);j=RubricJudgment.from_dict(app['judgment'])
            require(calculate_rubric_judgment(r,j.criterion_judgments)==j,'application judgment recomputation')
            c=calculate_experience_credits(r,j).get(app['candidate_id'],Credit())
            require(c.to_dict()==app['credit'],'application credit recomputation')
            require(app['eligible']==(app['candidate_id'] in positive_experience_sources(r,j)),'candidate qualification')
        key=record['admitted_experience_id']
        if key:
            app=next(a for a in record['application_attempts'] if a['candidate_id']==record['admitted_candidate_id'])
            require(app['eligible'],'admission not eligible')
            candidate=next(p['candidate'] for p in record['proposal_attempts'] if p['candidate'] and p['candidate']['candidate_id']==app['candidate_id'])
            origin=next(c for c in final[key]['cases'] if c['sample_id']==sid)
            e={'experience_id':key,'content':candidate['content'],'trigger':candidate['trigger'],
              'alpha':1.,'beta':1.,'cases':[deepcopy(origin)],'text_version':1,'update_history':[],
              'parent_id':None,'child_ids':[],'inherited_credit':None}
            register(e,sample_index)
            evidence_event(key,sid,'application_credit',app['credit']['positive_credit'],app['credit']['negative_credit'],
                           app['rubric'],app['judgment'],app['candidate_id'],sample_index,app['round_number'])
            stats[key]['birth_confidence']=confidence(state[key]);stats[key]['birth_alpha']=state[key]['alpha'];stats[key]['birth_beta']=state[key]['beta']
            stats[key]['versions'][0]['confidence']=confidence(state[key])
        for key,event in record['experience_updates'].items():
            e=state[key];require(not e['child_ids'],'update retired')
            require((event['previous_content'],event['previous_trigger'],event['previous_version'])==
                    (e['content'],e['trigger'],e['text_version']),'update previous version mismatch')
            require(key in record['downstream_credits'],'update must follow current R0 credit')
            c=record['downstream_credits'][key];require(c['positive_credit']>c['negative_credit'],'update requires netpositive')
            case=next(c for c in final[key]['cases'] if c['sample_id']==sid)
            require(case['sample_id'] not in {c['sample_id'] for c in e['cases']},'duplicate collected case')
            require((case['content'],case['trigger'],case['text_version'])==(e['content'],e['trigger'],e['text_version']),'case applied version')
            e['cases'].append(deepcopy(case));require(event['case_ids']==[c['sample_id'] for c in e['cases']],'case order')
            before=confidence(e)
            event_info={'sample_index':sample_index,'experience_id':key,'root_id':root_id(key,final),
                'confidence_before':before,'case_count':len(e['cases']),**deepcopy(event),'children':[]}
            e['update_history'].append(deepcopy(event))
            if event['child_ids']:
                require(len(event['child_ids'])==2,'split2')
                e['child_ids']=event['child_ids'];stats[key]['retired_sample']=sample_index
                original_case_ids={c['sample_id'] for c in e['cases']};assigned=[]
                inherited_p=inherited_n=0.
                for child_key in event['child_ids']:
                    stored=final[child_key];ic=stored['inherited_credit']
                    initial_cases=[c for c in stored['cases'] if c['sample_id'] in original_case_ids]
                    assigned += [c['sample_id'] for c in initial_cases]
                    fraction=len(initial_cases)/len(e['cases'])
                    require(close(ic['case_positive_credit'],fsum(c['credit']['positive_credit'] for c in initial_cases)),'child caseP')
                    remaining=e['alpha']-1-fsum(c['credit']['positive_credit'] for c in e['cases'])
                    require(close(ic['remaining_positive_credit'],max(remaining,0)*fraction),'remainingP allocation')
                    require(close(ic['negative_credit'],(e['beta']-1)*fraction),'N allocation')
                    child_p=ic['case_positive_credit']+ic['remaining_positive_credit'];child_n=ic['negative_credit']
                    inherited_p+=child_p;inherited_n+=child_n
                    text=stored['update_history'][0] if stored['update_history'] else None
                    child={'experience_id':child_key,'content':text['previous_content'] if text else stored['content'],
                      'trigger':text['previous_trigger'] if text else stored['trigger'],'alpha':1+child_p,'beta':1+child_n,
                      'cases':deepcopy(initial_cases),'text_version':1,'update_history':[],'parent_id':key,'child_ids':[],
                      'inherited_credit':deepcopy(ic)}
                    register(child,sample_index)
                    event_info['children'].append({'experience_id':child_key,'content':child['content'],'trigger':child['trigger'],
                      'sample_ids':[c['sample_id'] for c in initial_cases],'confidence':confidence(child),
                      'confidence_delta_vs_parent':confidence(child)-before,'positive_inherited':child_p,'negative_inherited':child_n})
                require(set(assigned)==original_case_ids and len(assigned)==len(set(assigned)),'partition complete')
                require(close(inherited_p,e['alpha']-1) and close(inherited_n,e['beta']-1),'split credit conserved')
            else:
                e.update(content=event['content'],trigger=event['trigger'],text_version=event['text_version'])
                require(confidence(e)==before,'text change alone must not change confidence')
                stats[key]['versions'].append({'sample_index':sample_index,'text_version':e['text_version'],
                  'content':e['content'],'trigger':e['trigger'],'confidence':before,'cause':'text_update'})
            updates.append(event_info)
        active=[e for e in state.values() if not e['child_ids']]
        snapshots.append({'sample_index':sample_index,'active_count':len(active),
          'positive_evidence_total':fsum(e['alpha']-1 for e in active),'negative_evidence_total':fsum(e['beta']-1 for e in active),
          'active':[{ 'experience_id':e['experience_id'],'text_version':e['text_version'],'confidence':confidence(e)} for e in active]})
    require(set(state)==set(final),'all IDs reconstructed')
    for key,e in state.items():
        f=final[key]
        require(close(e['alpha'],f['alpha']) and close(e['beta'],f['beta']),'final Beta mismatch')
        for field in ('content','trigger','text_version','parent_id','child_ids','cases','update_history'):
            require(e[field]==f[field],'final mismatch:'+field)
        actual=[(x['sample_id'],x['evidence_type'],x['positive_credit'],x['negative_credit'],x['text_version']) for x in f['evidence_history']]
        require(actual==expected_evidence[key],'evidence history mismatch')
        s=stats[key];s.update(final_confidence=confidence(e),final_alpha=e['alpha'],final_beta=e['beta'],
          content=e['content'],trigger=e['trigger'],active=not e['child_ids'],child_ids=e['child_ids'],
          case_count=len(e['cases']),text_version=e['text_version'],remaining_sample_opportunities=100-s['birth_sample'],
          confidence_delta_since_birth=confidence(e)-s['birth_confidence'],inherited_credit=e['inherited_credit'])
    positive=[e for e in events if e['evidence_type']=='downstream_credit' and e['positive_credit']>e['negative_credit']]
    # Balanced timeline plus all represented roots, then evenly fill to20. No new outputs used.
    chosen=[];seen=set()
    for root in sorted({e['root_id'] for e in positive}):
        option=next(e for e in positive if e['root_id']==root)
        chosen.append(option);seen.add(option['event_key'])
    for i in range(20):
        option=positive[round(i*(len(positive)-1)/19)]
        if option['event_key'] not in seen:chosen.append(option);seen.add(option['event_key'])
        if len(chosen)==20:break
    for option in positive:
        if len(chosen)>=20:break
        if option['event_key'] not in seen:chosen.append(option);seen.add(option['event_key'])
    require(len(chosen)==20,'20 positive audit events')
    selected_keys={e['event_key'] for e in chosen}
    cards=[]
    for e in events:
        if e['evidence_type']=='application_credit' or e['negative_credit']>e['positive_credit'] or e['event_key'] in selected_keys:
            key=e['experience_id'];e=deepcopy(e)
            s=stats[key];e['selection_reason']='all_admissions' if e['evidence_type']=='application_credit' else 'all_netnegative' if e['negative_credit']>e['positive_credit'] else '20_spread_netpositive'
            e['later_numeric_events']=[x for x in events if x['experience_id']==key and x['sample_index']>e['sample_index']]
            e['following_update']=next((u for u in updates if u['experience_id']==key and u['sample_index']==e['sample_index']),None)
            e['experience_final']=s;cards.append(e)
    active=[s for s in stats.values() if s['active']]
    families=[]
    for root in sorted(k for k,e in final.items() if e['parent_id'] is None):
        group=[s for s in stats.values() if s['root_id']==root]
        downstream=[e for e in events if e['root_id']==root and e['evidence_type']=='downstream_credit']
        family={'root_id':root,'origin_sample':stats[root]['birth_sample'],'origin_content':birth_text[root][0],
          'members':[s['experience_id'] for s in group],'active_descendants':[s['experience_id'] for s in group if s['active']],
          'distinct_downstream_tasks':len({e['sample_id'] for e in downstream}),
          'downstream_numeric_events':len(downstream),'positive_credit':fsum(e['positive_credit'] for e in downstream),
          'negative_credit':fsum(e['negative_credit'] for e in downstream),
          'selected_uses':sum(s['selected'] for s in group),'cited_uses':sum(s['cited'] for s in group),
          'neutral_uses':sum(s['neutral_uses'] for s in group)}
        families.append(family)
    negative=[e for e in events if e['evidence_type']=='downstream_credit' and e['negative_credit']>e['positive_credit']]
    summary={'run':str(RUN),'samples':100,'experiences_total':len(state),'root_candidates_admitted':len(families),
      'active_experiences':len(active),'retired_experiences':len(state)-len(active),'updates':len(updates),
      'splits':sum(bool(u['children']) for u in updates),'same_ID_text_updates':sum(not u['children'] for u in updates),
      'admission_events':sum(e['evidence_type']=='application_credit' for e in events),
      'downstream_numeric_events':sum(e['evidence_type']=='downstream_credit' for e in events),
      'netpositive_downstream':len(positive),'netnegative_downstream':len(negative),
      'negative_events_confidence_down':sum(e['confidence_delta']<0 for e in negative),
      'positive_events_confidence_down':sum(e['confidence_delta']<0 for e in positive),
      'active_without_own_numeric_downstream':sum(s['positive_events']+s['negative_events']==0 for s in active),
      'active_own_downstream_count_distribution':dict(Counter(s['positive_events']+s['negative_events'] for s in active)),
      'neutral_cited_uses':sum(s['neutral_uses'] for s in stats.values()),
      'selected_uses':sum(s['selected'] for s in stats.values()),'cited_uses':sum(s['cited'] for s in stats.values()),
      'final_active_confidence_range':[min(s['final_confidence'] for s in active),max(s['final_confidence'] for s in active)],
      'active_below_half':sum(s['final_confidence']<.5 for s in active),
      'active_numeric_evidence_totals':{'P':fsum(s['final_alpha']-1 for s in active),'N':fsum(s['final_beta']-1 for s in active)},
      'review_cards':len(cards),'selected_positive_events':20,
      'limitations':'100 sequential tasks; source/admission and inherited credit are not fresh validation of each new text; no known quality gold or convergence proof'}
    require(close(summary['active_numeric_evidence_totals']['P'],fsum(e['positive_credit'] for e in events)),'all activeP conserved')
    require(close(summary['active_numeric_evidence_totals']['N'],fsum(e['negative_credit'] for e in events)),'all activeN conserved')
    OUTPUT.mkdir(parents=True,exist_ok=False);OUTPUT.chmod(0o700)
    for name,value in [('numeric_events.json',events),('exposures.json',exposures),('text_evolution.json',updates),
       ('experience_stats.json',list(stats.values())),('families.json',families),('snapshots.json',snapshots),
       ('review_cards.json',cards),('summary.json',summary)]:save(OUTPUT/name,value)
    require(all(digest(Path(p))==h for p,h in hashes.items()),'source changed')
    save(OUTPUT/'audit_manifest.json',{'source_sha256':hashes,'all_checks_passed':True,
      'scope':'read-only historical100 run; no LLM/pipeline/production changes','run_prompt':manifest['prompt_version'] if 'prompt_version' in manifest else manifest['execution'].get('prompt_version'),
      'integrity_checks':'judgment/credit/eligibility, selected text/confidence, ordered original samples, all Beta/versions/evidence/cases/updates, parent child allocation and active total conservation'})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
