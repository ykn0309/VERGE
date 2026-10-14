"""Bounded Generator examples and Critic presentation ablation, outside the pipeline."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.replay_experience_splits import digest, fingerprint, save
from scripts.replay_role_few_shot import example_suffix, validate
from verge.llm import LLMFormatError, LLMRefusalError, OpenAICompatibleBackend
from verge import prompts


LABELS = {'chosen': '回答A', 'rejected': '回答B',
          'chosen_reason': '回答A判断依据', 'rejected_reason': '回答B判断依据',
          'chosen_score': '回答A得分', 'rejected_score': '回答B得分'}
GROUNDING = ('证据核对提醒：偏好身份不代表逐项判定。声称Judge误判之前，先核对该标准的实际判定与理由；'
             '引用回答内容之前，核对它属于哪份原回答。Judge理由和自己上一轮分析都可能出错，'
             '不能替代task或回答原文。仅在现有analysis中简述必要核对，不增加字段或逐项长分析。')
VARIANTS = ('baseline', 'layout_only', 'labels_only', 'grounding_only', 'combined')


def grouped_evidence(payload):
    """Only reorganize each criterion's two Boolean judgments and verbatim reasons."""
    value = prompts._inline_critic_evidence(payload)

    def visit(item):
        if isinstance(item, dict):
            result = {key: visit(child) for key, child in item.items()}
            if 'criterion_id' in result and 'question' in result:
                for side in ('chosen', 'rejected'):
                    if isinstance(result.get(side), bool):
                        judgment = result[side]
                        reason = result.pop(side + '_reason')
                        result[side] = {'判定': '满足' if judgment else '未满足', '依据': reason}
            return result
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value)


def critic_messages(event, first_event, variant):
    if variant not in VARIANTS:
        raise ValueError('unknown presentation variant')
    layout, labels = variant in ('layout_only', 'combined'), variant in ('labels_only', 'combined')
    current = grouped_evidence(event['input']) if layout else deepcopy(event['input'])
    first = grouped_evidence(first_event['input']) if layout else deepcopy(first_event['input'])
    with patch.dict(prompts._INPUT_TITLES, LABELS if labels else {}):
        messages = prompts.llm_messages('critic', first)
        if len(event['messages']) == 4:
            # Actual first assistant is retained even if it contains factual mistakes.
            messages.append(deepcopy(event['messages'][2]))
            messages = prompts.llm_messages('critic', current, history=messages)
        elif len(event['messages']) != 2:
            raise ValueError('only first/second semantic turns are selected')
    if labels:
        messages[0]['content'] += ('\n\n本次证据中回答A对应chosen（偏好较高），回答B对应rejected（偏好较低）；'
                                   '历史消息中的优选回答=A，非优选回答=B。逐项是否满足仍以各项判定为准。')
    if variant in ('grounding_only', 'combined'):
        messages[0]['content'] += '\n\n' + GROUNDING
    return messages


def main():
    source = Path('runs/helpsteer3-general-split-experience-qwen-flash-100-resume3').resolve()
    output = Path('runs/role-prompt-ablation').resolve()
    generator_dir = Path('runs/module-simplified-input-replay/generator').resolve()
    manifest_path = source / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    root = source / manifest['config']['store']['config']['root_dir']
    events = [json.loads(x) for x in (root / 'role_io.jsonl').read_text().splitlines()]
    spec = manifest['config']['models']['default']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('requires fixed Flash non-thinking')
    protected = [manifest_path, root / 'role_io.jsonl', root / 'experience_memory.json',
                 root / 'trajectories.jsonl', generator_dir / 'few_shot-proposal.json',
                 generator_dir / 'results/replay_manifest.json']
    prepared = []
    suffix = example_suffix('initial_generator', json.loads(protected[4].read_text()))
    for key in ('invented-era', 'source-confusion', 'normal-feelings'):
        path = generator_dir / 'results' / (key + '.json')
        protected.append(path)
        row = json.loads(path.read_text())
        payload = row['new_input']
        for variant in ('baseline', 'few_shot'):
            messages = prompts.llm_messages('initial_generator', payload)
            if variant == 'few_shot':
                messages[0]['content'] += '\n\n' + suffix
            prepared.append({'role': 'initial_generator', 'case_key': key, 'variant': variant,
                             'input': payload, 'messages': messages, 'trace_line': row['selection']['trace_line'],
                             'historical_output': row['baseline_output'], 'previous_output': row['new_output']})
    critic_selection = [(163, 'first-turn-content-confusion'), (167, 'second-turn-history'),
                        (724, 'video-read-hear'), (4, 'feelings-fixed-list'), (390, 'story-normal-control')]
    for line, key in critic_selection:
        event = events[line - 1]
        if event['role'] != 'critic' or event['output'] is None or event['error'] is not None:
            raise ValueError('source must be successful Critic event')
        first = event
        if len(event['messages']) == 4:
            matches = [r for r in events[:line - 1] if r['role'] == 'critic'
                       and r['sample_id'] == event['sample_id'] and r['messages'] == event['messages'][:2]]
            if len(matches) != 1:
                raise ValueError('first-turn exact history must have one source')
            first = matches[0]
        baseline = critic_messages(event, first, 'baseline')
        # No unrecorded regeneration of historical user evidence.
        if baseline[1:] != event['messages'][1:]:
            raise ValueError('baseline user/history must match actual source byte-for-byte')
        for variant in VARIANTS:
            prepared.append({'role': 'critic', 'case_key': key, 'variant': variant,
                             'input': event['input'], 'messages': critic_messages(event, first, variant),
                             'trace_line': line, 'sample_id': event['sample_id'],
                             'historical_output': event['output']})
    if len(prepared) != 31 or any(sum(len(m['content'].encode()) for m in r['messages']) > 100_000 for r in prepared):
        raise ValueError('bounded selection/context budget violated; never truncate')
    hashes = {str(p): digest(p) for p in protected}
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    save(output / 'replay_manifest.json', {'model_config': spec, 'prompt_version': prompts.PROMPT_VERSION,
        'script_sha256': digest(Path(__file__)), 'production_prompts_sha256': digest(Path(prompts.__file__)),
        'source_sha256': hashes, 'planned_cases': len(prepared), 'max_calls_per_case': 9,
        'scope': '31 outputs; backend up to 3 transport/format attempts per call; up to 3 structure attempts; no pipeline',
        'critic_task_count': len({r['sample_id'] for r in prepared if r['role'] == 'critic'}),
        'variants': list(VARIANTS), 'grounding': GROUNDING, 'labels': LABELS,
        'generator_examples': json.loads(protected[4].read_text()),
        'case_message_hashes': [fingerprint(r) for r in prepared]})
    context = {}

    def append(name, event):
        path = output / name
        with path.open('a', encoding='utf-8') as stream:
            path.chmod(0o600)
            stream.write(json.dumps({**context, **event}, ensure_ascii=False, allow_nan=False) + '\n')

    backend = OpenAICompatibleBackend(spec['model'], **{k: spec[k] for k in
        ('base_url', 'api_key_env', 'timeout_seconds', 'max_retries', 'enable_thinking') if k in spec},
        trace_hook=lambda e: append('role_io.jsonl', e), metadata_hook=lambda e: append('llm_requests.jsonl', e))
    results = []
    try:
        for case in prepared:
            context.update(role=case['role'], case_key=case['case_key'], variant=case['variant'])
            messages = deepcopy(case['messages'])
            started = time.monotonic()
            result, error = None, None
            for retry in range(3):
                try:
                    with patch('verge.llm.llm_messages', lambda *a, **kw: deepcopy(messages)):
                        raw = backend.generate(case['role'], case['input'])
                    validate(case['role'], case['input'], raw)
                    result = raw
                    break
                except (LLMFormatError, LLMRefusalError) as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                    break
                except ValueError as exc:
                    error = str(exc)
                    if retry == 2:
                        break
                    messages.extend([{'role': 'assistant', 'content': json.dumps(raw, ensure_ascii=False)},
                                     {'role': 'user', 'content': '输出校验失败：' + error[:160] + '。请返回完整合法结果。'}])
            row = {**case, 'new_output': result, 'error': None if result is not None else error,
                   'structure_retries': retry, 'elapsed_seconds': round(time.monotonic() - started, 2)}
            save(output / f"{case['role']}_{case['case_key']}_{case['variant']}.json", row)
            results.append(row)
            print(f"{case['role']} {case['case_key']} {case['variant']}: {'valid' if result is not None else 'invalid'}", flush=True)
    finally:
        backend.close()
        unchanged = all(digest(Path(p)) == h for p, h in hashes.items())
        save(output / 'progress.json', {'completed': len(results), 'source_unchanged': unchanged})
        if not unchanged:
            raise ValueError('source changed')
    save(output / 'summary.json', {'completed': len(results), 'valid': sum(r['new_output'] is not None for r in results),
         'valid_by_role_variant': dict(Counter(r['role'] + '/' + r['variant'] for r in results if r['new_output'] is not None)),
         'structure_retries': sum(r['structure_retries'] for r in results), 'source_unchanged': True,
         'limitations': 'One output per fixed input/variant; manual evidence review needed; no overall accuracy or stability claim'})


if __name__ == '__main__':
    main()
