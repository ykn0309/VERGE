"""Read-only descriptive audit of this fixed run; never calls a model."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json

from verge.feedback import CreditFeedback

ROOT = Path('runs/history-learning-general500-subset120')
OUT = Path('artifacts/general500-subset120')
state = json.loads((ROOT / 'checkpoint.json').read_text())


def success(path):
    if not path or path.get('error') or not path.get('judgment'):
        return None
    return path['judgment']['success']


def pairing(actual, baseline):
    if actual is None or baseline is None:
        return 'unknown'
    if actual and not baseline:
        return 'improved'
    if baseline and not actual:
        return 'regressed'
    return 'both_correct' if actual else 'both_wrong'


by_selection = defaultdict(Counter)
historical = Counter()
admission_reasons = Counter()
rounds_by_task = Counter()
joint = Counter()
errors = defaultdict(Counter)
scored_selected_examples = []
source_repairs = Counter()
for row in state['trajectories']:
    actual = row['actual'] or row['baseline'] or {}
    ids = actual.get('provided_experience_ids', [])
    a, b = success(actual), success(row['baseline'])
    by_selection[len(ids)][pairing(a, b)] += 1
    retrieval = row.get('retrieval') or {}
    scored = {x['experience_id']: x for x in retrieval.get('historical', {}).get('scored', [])}
    offered = {x['experience_id'] for x in retrieval.get('candidates', []) if x['source'] == 'historical'}
    if retrieval.get('task_groups'):
        historical['group_hit_tasks'] += 1
    if offered:
        historical['historical_offered_tasks'] += 1
        historical['offered_' + pairing(a, b)] += 1
    if offered.intersection(ids):
        historical['selected_historical_tasks'] += 1
        historical['selected_' + pairing(a, b)] += 1
        scored_selected_examples.append({
            'sample_id': row['sample']['sample_id'],
            'pair': pairing(a, b),
            'selected': [{'id': x, 'score': scored[x]['score']} for x in ids if x in scored],
        })
    if len(ids) == 2:
        outcomes = [success(path) for path in row['subset_paths']]
        if None not in outcomes:
            r0, r1, r2, r12 = map(int, outcomes)
            joint['complete'] += 1
            joint['only_combination_correct'] += (r0, r1, r2, r12) == (0, 0, 0, 1)
            joint['combination_worse_than_best_singleton'] += r12 < max(r1, r2)
            joint['both_singletons_correct_combination_wrong'] += r1 == r2 == 1 and r12 == 0
            joint['interaction_' + str(r12-r1-r2+r0)] += 1
        else:
            joint['incomplete'] += 1
    rounds = row['critic_rounds']
    rounds_by_task[len(rounds)] += 1
    if rounds:
        repaired = any(at['admitted'] for rnd in rounds for at in rnd['attempts'])
        source_repairs['repaired' if repaired else 'not_repaired'] += 1
        if repaired:
            first = next(rnd['round_number'] for rnd in rounds if any(at['admitted'] for at in rnd['attempts']))
            source_repairs['first_success_round_' + str(first)] += 1
    if row.get('error'):
        errors['preselection'][row['error']] += 1
    for path in row['subset_paths']:
        if path.get('error'):
            errors['initial_path'][path['error']] += 1
    for rnd in rounds:
        if rnd.get('error'):
            errors['critic'][rnd['error']] += 1
        for path in rnd['subset_paths']:
            if path.get('provided_experience_ids') and path.get('error'):
                errors['candidate_path'][path['error']] += 1
        for attempt in rnd['attempts']:
            admission_reasons[attempt['reason']] += 1

credits = defaultdict(list)
for row in state['feedback']:
    parsed = CreditFeedback(**row)
    for eid, value in (parsed.credits or {}).items():
        credits[eid].append(value)
credit_totals = Counter()
for eid in [x['experience_id'] for x in state['experiences']]:
    values = credits[eid]
    credit_totals['no_valid_credit' if not values else 'observed_experiences'] += 1
    if values:
        credit_totals['positive_mean' if sum(values)>0 else 'negative_mean' if sum(values)<0 else 'zero_mean'] += 1

request_status = defaultdict(Counter)
request_retries = Counter()
for line in (ROOT / 'llm_requests.jsonl').open():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    request_status[row.get('role')][str(row.get('status'))] += 1
    if row.get('attempt', 1) > 1:
        request_retries[row['role']] += 1

validation = defaultdict(Counter)
invalid_reasons = defaultdict(Counter)
intervals = []
for line in (ROOT / 'role_io.jsonl').open():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get('event_type') == 'contract_validation':
        validation[row['role']][row['validation_status']] += 1
        if row['validation_status'] == 'invalid':
            invalid_reasons[row['role']][row['error']] += 1
    if row.get('event_type') == 'request_timing':
        end = datetime.fromisoformat(row['finished_at']).timestamp()
        intervals.append((end - row['request_seconds'], end))

# Session spans separate long pauses; network union is separately exact from logs.
segments = []
occupied = []
for start, end in sorted(intervals):
    if not segments or start - segments[-1][1] > 600:
        segments.append([start, end])
    else:
        segments[-1][1] = max(end, segments[-1][1])
    if not occupied or start > occupied[-1][1]:
        occupied.append([start, end])
    else:
        occupied[-1][1] = max(end, occupied[-1][1])

summary = {
    'completed': len(state['trajectories']),
    'paired_by_selected_count': {str(k):dict(v) for k,v in by_selection.items()},
    'historical': dict(historical),
    'historical_selection_examples': scored_selected_examples,
    'two_experience_paths': dict(joint),
    'critic_rounds_per_task': dict(rounds_by_task),
    'source_repairs': dict(source_repairs),
    'candidate_attempt_reasons': dict(admission_reasons),
    'credit_coverage': dict(credit_totals),
    'final_error_messages': {k:dict(v) for k,v in errors.items()},
    'http_request_status': {k:dict(v) for k,v in request_status.items()},
    'http_extra_attempts': dict(request_retries),
    'contract_validation': {k:dict(v) for k,v in validation.items()},
    'invalid_contract_reasons': {k:dict(v) for k,v in invalid_reasons.items()},
    'timing': {
        'request_session_spans_10min_gap_rule': segments,
        'session_span_minutes': sum(b-a for a,b in segments)/60,
        'any_http_in_flight_minutes': sum(b-a for a,b in occupied)/60,
        'long_gaps_minutes': [(segments[i][0]-segments[i-1][1])/60 for i in range(1,len(segments))],
        'scope': 'HTTP timing logs; session spans include short review and computation gaps; excludes gaps over10min',
    },
}
(OUT / 'details.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('historical_selection_examples','final_error_messages','invalid_contract_reasons')},ensure_ascii=False,indent=2))
