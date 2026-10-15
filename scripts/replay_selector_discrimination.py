"""Paired Selector-only replay with strictly prior R0 discrimination statistics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

from scripts.replay_experience_splits import digest, save
from scripts.replay_role_few_shot import validate
from scripts.run_module_model_replay import RecordedClient
from verge.llm import LLMFormatError, LLMSampleError, OpenAICompatibleBackend


TARGET = 'exp_094583091100442786f2d699273c5ac0'
# Frozen before model calls. These task-only assessments are not preference labels.
GROUPS = {
    'context_or_clarification': [73, 84, 89, 105, 123, 130, 132, 165],
    'creative_or_supplied_material': [72, 102, 136, 149, 154, 161, 186, 198],
    'boundary': [60, 62, 63, 93, 94, 148, 160, 185],
}
GUIDANCE = '''附加历史统计（仅包含当前任务开始前的训练R0反馈）：
区分率 = 引用该经验且两回答Judge结果不同的标准数 / 引用该经验的全部标准数。
正确区分和反向区分均计入分子，因此区分率不代表方向正确率，也不代表经验本身的质量或因果效果。暂无历史时标为“未知”。
这些统计只反映历史应用；不能据此推断当前任务的两个回答是否有差异。仍先判断当前task中的适用性和具体标准设计帮助，再把历史区分率作为辅助信息；不设硬门槛，不因区分率低就排除明确适用的经验。
'''


def verify(output: Path):
    manifest = json.loads((output / 'replay_manifest.json').read_text())
    if digest(output / 'prepared_inputs.json') != manifest['prepared_sha256']:
        raise ValueError('frozen replay inputs changed')
    if any(digest(Path(p)) != value for p, value in manifest['protected_sha256'].items()):
        raise ValueError('protected source or implementation changed')
    return manifest


def prepare(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError('output must be independent of source')
    mf = source / 'run_manifest.json'
    original = json.loads(mf.read_text())
    artifacts = source / original['config']['store']['config']['root_dir']
    trajectories = [json.loads(s) for s in (artifacts / 'trajectories.jsonl').read_text().splitlines()]
    events = {}
    for number, line in enumerate((artifacts / 'role_io.jsonl').read_text().splitlines(), 1):
        event = json.loads(line)
        if event['role'] == 'selector' and event.get('output') is not None and not event.get('error'):
            events.setdefault(event['sample_id'], (number, event))
    groups = {i: group for group, indices in GROUPS.items() for i in indices}
    if len(groups) != 24 or len({trajectories[i-1]['sample_id'] for i in groups}) != 24:
        raise ValueError('expected 24 different historical tasks')
    stats, prepared = defaultdict(Counter), []
    for index, trajectory in enumerate(trajectories, 1):
        if index in groups:
            line, event = events[trajectory['sample_id']]
            payload = deepcopy(event['input'])
            real_ids = trajectory['retrieved_experience_ids']
            candidates = payload['candidates']
            if len(candidates) != len(real_ids) or TARGET not in real_ids:
                raise ValueError('target must have been actually recalled')
            mapping = {str(i): eid for i, eid in enumerate(real_ids, 1)}
            if [c['experience_id'] for c in candidates] != list(mapping):
                raise ValueError('historical local-ID projection is not positional')
            selected = {e['experience_id']: e for e in trajectory['selected_experiences']}
            if {mapping[key] for key in event['output']['selected_experience_ids']} != set(selected):
                raise ValueError('historical selection mapping differs from trajectory')
            for candidate in candidates:
                previous = selected.get(mapping[candidate['experience_id']])
                if previous and any(candidate[key] != previous[key] for key in ('content', 'trigger', 'confidence', 'evidence_amount')):
                    raise ValueError('historical selected snapshot mismatch')
            evidence = []
            for candidate in candidates:
                counts = stats[mapping[candidate['experience_id']]]
                total = sum(counts.values())
                different = counts['positive'] + counts['negative']
                evidence.append({'experience_id': candidate['experience_id'],
                    'discriminating_criteria': different, 'cited_criteria': total,
                    'discrimination_rate': different / total if total else None})
            additions = ['\n\n' + GUIDANCE]
            for item in evidence:
                rate = '未知' if item['discrimination_rate'] is None else f"{item['discrimination_rate']:.4%}"
                additions.append(f"经验{item['experience_id']}：历史引用标准 {item['cited_criteria']} 条，其中区分 {item['discriminating_criteria']} 条，区分率 {rate}。")
            base = deepcopy(event['messages'])
            added = deepcopy(base)
            added[-1]['content'] += '\n'.join(additions)
            if any(sum(len(m['content'].encode()) for m in messages) > 100_000 for messages in (base, added)):
                raise ValueError('input exceeds budget; no truncation allowed')
            target_local = next(k for k, eid in mapping.items() if eid == TARGET)
            prepared.append({'case_key': f'case_{index:03}', 'sample_index': index,
                'sample_id': trajectory['sample_id'], 'trace_line': line,
                'group': groups[index], 'input': payload, 'id_mapping': mapping,
                'target_local_id': target_local,
                'historical_output': event['output'], 'prior_statistics': evidence,
                'messages': {'baseline': base, 'with_rate': added}})
        # Update only AFTER freezing this sample's inputs. No R1/R2 or origin credit.
        rubric, judgment = trajectory.get('initial_rubric'), trajectory.get('initial_judgment')
        if rubric and judgment:
            verdicts = {v['criterion_id']: v for v in judgment['criterion_judgments']}
            for criterion in rubric['criteria']:
                v = verdicts[criterion['criterion_id']]
                direction = 'positive' if v['chosen'] > v['rejected'] else 'negative' if v['chosen'] < v['rejected'] else 'neutral'
                for eid in criterion['source_experiences']:
                    stats[eid][direction] += 1
    if len(prepared) != 24:
        raise ValueError('incomplete frozen selection')
    spec = original['config']['models']['default']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('fixed Qwen3.5-Flash non-thinking required')
    protected = [mf, *(artifacts / n for n in ('role_io.jsonl', 'trajectories.jsonl', 'experience_memory.json', 'llm_requests.jsonl')),
                 Path(__file__).resolve(), *Path('verge').glob('*.py'), Path('AGENTS.md')]
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    save(output / 'prepared_inputs.json', prepared)
    save(output / 'replay_manifest.json', {'target_experience_id': TARGET, 'model_config': spec,
        'variants': ['baseline', 'with_rate'], 'unique_tasks': 24, 'planned_outputs': 48,
        'max_http_attempts': 48 * (spec['max_retries'] + 1),
        'rate_definition': 'unweighted per cited R0 criterion; (positive+reverse)/(positive+reverse+neutral); no origin/application credit',
        'history_boundary': 'strictly preceding dataset tasks; values pooled across historical text versions, as existing confidence is',
        'change': 'baseline is exact historical messages; treatment appends metric definition and prior counts for all recalled experiences to user message; all existing fields/order unchanged',
        'groups': GROUPS, 'prepared_sha256': digest(output / 'prepared_inputs.json'),
        'protected_sha256': {str(p.resolve()): digest(p) for p in protected},
        'target_final_prior_counts': dict(stats[TARGET]),
        'scope': 'Selector only; no Generator/Judge/Critic/Updater calls, no memory writes; one paired draw per task; task groups are audit strata, not gold applicability labels'})
    print(f'prepared 24 tasks / 48 outputs at {output}', flush=True)


def run_variant(output: Path, variant: str):
    manifest = verify(output)
    spec = manifest['model_config']
    cases = json.loads((output / 'prepared_inputs.json').read_text())
    directory = output / variant
    directory.mkdir(exist_ok=True)
    directory.chmod(0o700)
    context = {'variant': variant}

    def append(name, event):
        path = directory / name
        with path.open('a', encoding='utf-8') as stream:
            path.chmod(0o600)
            stream.write(json.dumps({**context, **event}, ensure_ascii=False, allow_nan=False) + '\n')

    client = RecordedClient(spec, append)
    backend = OpenAICompatibleBackend(spec['model'],
        **{k: spec[k] for k in ('base_url', 'api_key_env', 'timeout_seconds', 'max_retries', 'enable_thinking')},
        client=client, trace_hook=lambda e: append('role_io.jsonl', e),
        metadata_hook=lambda e: append('llm_requests.jsonl', e))
    results = []
    try:
        for case in cases:
            context['case_key'] = case['case_key']
            path = directory / (case['case_key'] + '.json')
            if path.exists():
                raise ValueError('existing results; no silent resume or duplicate calls')
            result, error = None, None
            started = time.monotonic()
            try:
                with patch('verge.llm.llm_messages', lambda *args, **kwargs: deepcopy(case['messages'][variant])):
                    result = backend.generate('selector', case['input'])
                validate('selector', case['input'], result)
            except (LLMFormatError, LLMSampleError, ValueError) as exc:
                result, error = None, type(exc).__name__ + ': ' + str(exc)[:500]
            row = {'case_key': case['case_key'], 'sample_index': case['sample_index'],
                'sample_id': case['sample_id'], 'variant': variant, 'output': result,
                'error': error, 'elapsed_seconds': round(time.monotonic() - started, 3),
                'target_selected': case['target_local_id'] in result['selected_experience_ids'] if result else None}
            save(path, row)
            results.append(row)
            save(directory / 'progress.json', {'completed': len(results), 'valid': sum(r['error'] is None for r in results)})
            print(f'{variant}: {len(results)}/24', flush=True)
    finally:
        backend.close()
        client.close()
        verify(output)
    return {'variant': variant, 'completed': len(results), 'valid': sum(r['error'] is None for r in results)}


def summarize(output: Path):
    verify(output)
    cases = json.loads((output / 'prepared_inputs.json').read_text())
    pairs = []
    for case in cases:
        rows = {v: json.loads((output / v / (case['case_key'] + '.json')).read_text()) for v in ('baseline', 'with_rate')}
        target_stats = next(s for s in case['prior_statistics'] if s['experience_id'] == case['target_local_id'])
        pairs.append({**{k: case[k] for k in ('sample_index', 'sample_id', 'group', 'target_local_id')},
            'task': case['input']['task'], 'target_content': next(c['content'] for c in case['input']['candidates'] if c['experience_id'] == case['target_local_id']),
            'prior_statistics': target_stats,
            'historical_selected': case['target_local_id'] in case['historical_output']['selected_experience_ids'],
            'baseline_selected': rows['baseline']['target_selected'], 'with_rate_selected': rows['with_rate']['target_selected'],
            'baseline_output': rows['baseline']['output'], 'with_rate_output': rows['with_rate']['output'],
            'errors': {v: r['error'] for v, r in rows.items() if r['error']}})
    summary = {'paired_tasks': len(pairs), 'valid_pairs': sum(not p['errors'] for p in pairs),
        'baseline_target_selected': sum(p['baseline_selected'] is True for p in pairs),
        'with_rate_target_selected': sum(p['with_rate_selected'] is True for p in pairs),
        'dropped': [p['sample_index'] for p in pairs if p['baseline_selected'] is True and p['with_rate_selected'] is False],
        'added': [p['sample_index'] for p in pairs if p['baseline_selected'] is False and p['with_rate_selected'] is True],
        'all_selection_sets_changed': sum(set(p['baseline_output']['selected_experience_ids']) != set(p['with_rate_output']['selected_experience_ids']) for p in pairs if not p['errors']),
        'groups': {g: {v: sum(p[v] is True for p in pairs if p['group'] == g) for v in ('historical_selected', 'baseline_selected', 'with_rate_selected')} for g in GROUPS},
        'protected_sources_unchanged': True}
    save(output / 'paired_results.json', pairs)
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.source_run is None:
            parser.error('--source-run required')
        prepare(args.source_run, args.output_dir)
    elif args.mode == 'run':
        verify(args.output_dir)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_variant, args.output_dir, v) for v in ('baseline', 'with_rate')]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        summarize(args.output_dir)
    else:
        summarize(args.output_dir)


if __name__ == '__main__':
    main()
