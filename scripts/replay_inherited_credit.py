"""Plan68: isolated accounting replay on real feedback; no LLM or PipelineRunner.

The old pool rewrote text in place. Its same-ID five-event windows test accounting
only, not the current Updater's immutable-text input contract. Keep/replace/split
are fixed control branches, not invented historical model decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from math import fsum, isclose
from pathlib import Path
import shutil

from verge.calculator import allocate_evolution_credit, calculate_experience_credits
from verge.models import EvidenceRecord, Experience, ExperienceCase, Rubric, RubricJudgment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    path.chmod(0o600)


def replay(source: Path, destination: Path):
    if destination.exists():
        raise ValueError('destination already exists')
    root = source/'artifacts'
    manifest_path = source/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['format_version'] != 8:
        raise ValueError('this explicit historical projection requires protocol8')
    train_path = Path(manifest['config']['dataset']['config']['paths']['train'])
    protected = [manifest_path, train_path, *root.glob('*.json*')]
    hashes = {str(p): sha(p) for p in protected}
    memory = json.loads((root/'experience_memory.json').read_text())
    old = {e['experience_id']: e for e in memory['experiences']}
    records = [json.loads(line) for line in (root/'trajectories.jsonl').read_text().splitlines()]
    tasks = {s['sample_id']: s for s in map(json.loads, train_path.read_text().splitlines())}
    ledger = defaultdict(list)
    windows = defaultdict(list)
    inputs = []
    used_tasks = set()
    for r in records:
        for eid, c in r['downstream_credits'].items():
            if c['positive_credit']+c['negative_credit'] == 0:
                continue
            view = next(v for v in r['selected_experiences'] if v['experience_id'] == eid)
            credit = calculate_experience_credits(Rubric.from_dict(r['initial_rubric']),
                                                 RubricJudgment.from_dict(r['initial_judgment']))[eid]
            if credit.to_dict() != c:
                raise ValueError('historical feedback does not match actual rubric/judgment')
            if not ledger[eid]:
                origins = [v for v in old[eid]['evidence_history'] if v['evidence_type'] == 'application_credit']
                ledger[eid].extend(EvidenceRecord.from_dict(v) for v in origins)
            ledger[eid].append(EvidenceRecord(r['sample_id'], 'downstream_credit',
                                             credit.positive_credit, credit.negative_credit))
            task = tasks[r['sample_id']]
            case = ExperienceCase(r['sample_id'], task['task'], task['chosen'], task['rejected'],
                                 eid, 1, view['content'], view['trigger'], Rubric.from_dict(r['initial_rubric']),
                                 RubricJudgment.from_dict(r['initial_judgment']), credit)
            windows[eid].append(case)
            if len(windows[eid]) < 5 or task['task'] in used_tasks:
                continue
            used_tasks.add(task['task'])
            history = deepcopy(ledger[eid])
            experience = Experience(eid, view['content'], view['trigger'],
                1+fsum(v.positive_credit for v in history), 1+fsum(v.negative_credit for v in history),
                [], history, deepcopy(windows[eid][-5:]))
            inputs.append({'sample_id': r['sample_id'], 'experience': experience.to_dict(),
                           'same_text_window': len({(c.content,c.trigger) for c in experience.cases}) == 1})
    if len(inputs) < 20:
        raise ValueError(f'only {len(inputs)} task-deduplicated real feedback inputs')
    write(destination/'inputs.json', inputs)
    code = [*Path('verge').glob('*.py'), Path(__file__).resolve().relative_to(Path.cwd().resolve())]
    for p in code:
        target=destination/'code_snapshot'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    write(destination/'manifest.json', {
        'source':str(source),'source_sha256':hashes,'code_sha256':{str(p):sha(p) for p in code},
        'input_sha256':sha(destination/'inputs.json'),'llm_calls':0,'pipeline_run':False,
        'scope':'historical accounting projection, fixed keep/replace/split control outputs',
        'conversion':'Protocol8 raw JSON to current typed cases, local text_version=1. Historical text is retained per case; no production compatibility.',
        'limitations':['Old same-ID windows may span text changes; not a new immutable-text Updater replay.',
                       'No child method quality, allocation attribution or downstream benefit claim.'],
    })
    outputs=[]
    for i, item in enumerate(inputs):
        e=Experience.from_dict(item['experience'])
        snapshot=e.to_json()
        branches=[]
        for k in (1,2,3):
            amount, penalty, allocations=allocate_evolution_credit(e,f'replay_event_{i}_{k}',k)
            expected_window=fsum(c.credit.amount for c in e.cases)
            assert isclose(amount,expected_window,abs_tol=1e-9)
            assert isclose(penalty,.2*expected_window,abs_tol=1e-9)
            assert isclose(fsum(a.positive_credit for a in allocations),e.alpha-1,abs_tol=1e-9)
            assert isclose(fsum(a.negative_credit for a in allocations),e.beta-1,abs_tol=1e-9)
            assert isclose(fsum(a.evolution_penalty for a in allocations),penalty,abs_tol=1e-9)
            children=[]
            for a in allocations:
                alpha=1+a.positive_credit
                beta=1+a.negative_credit+a.inherited_penalty+a.evolution_penalty
                children.append({'allocation':a.to_dict(),'alpha':alpha,'beta':beta,
                                 'confidence':alpha/(alpha+beta)})
            branches.append({'result':'replace' if k==1 else 'split','child_count':k,
                             'window_credit_amount':amount,'total_penalty':penalty,'children':children})
        assert e.to_json()==snapshot
        outputs.append({'sample_id':item['sample_id'],'experience_id':e.experience_id,
                        'before':{'alpha':e.alpha,'beta':e.beta,'confidence':e.confidence},
                        'keep':{'alpha':e.alpha,'beta':e.beta,'penalty':0},'branches':branches})
    if hashes != {p:sha(Path(p)) for p in hashes}:
        raise ValueError('source artifacts changed')
    penalties=[r['branches'][0]['total_penalty'] for r in outputs]
    summary={'inputs':len(inputs),'distinct_tasks':len(used_tasks),'fixed_control_branches':4*len(inputs),
             'same_text_windows':sum(x['same_text_window'] for x in inputs),
             'actual_feedback_directions':dict(Counter('mixed' if c.credit.positive_credit and c.credit.negative_credit
                    else 'positive' if c.credit.positive_credit else 'negative'
                    for x in inputs for c in Experience.from_dict(x['experience']).cases)),
             'penalty_range':[min(penalties),max(penalties)],'checks_passed':True,'source_unchanged':True,
             'llm_calls':0,'pipeline_run':False}
    write(destination/'results.json',outputs)
    write(destination/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('runs/helpsteer3-general-atomic-experience-qwen-flash-200'))
    parser.add_argument('--destination',type=Path,default=Path('runs/inherited-credit-replay-68'))
    args=parser.parse_args()
    print(json.dumps(replay(args.source,args.destination),ensure_ascii=False,indent=2))
