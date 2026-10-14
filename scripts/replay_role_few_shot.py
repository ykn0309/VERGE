"""Bounded fixed-message role replay; never runs a pipeline or writes source artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.replay_experience_splits import digest, fingerprint, save
from verge.llm import LLMFormatError, LLMRefusalError, OpenAICompatibleBackend, _valid_shape
from verge.models import Rubric
from verge.prompts import PROMPT_VERSION, NumberedIDCodec, llm_messages


def validate(role, payload, output):
    """Check existing role contracts; no semantic gate or length rejection."""
    if not _valid_shape(role, output):
        raise ValueError('invalid role output shape')
    if role == 'initial_generator':
        rubric = Rubric.from_dict({'rubric_id': 'replay', 'criteria': [
            {**item, 'criterion_id': str(i)} for i, item in enumerate(output['criteria'], 1)]})
        allowed = {x['experience_id'] for x in payload['selected_experiences']}
        if 'candidate_experience' in payload:
            allowed.add(payload['candidate_experience']['experience_id'])
        if any(set(c.source_experiences) - allowed for c in rubric.criteria):
            raise ValueError('unknown source experience')
    elif role == 'judge':
        actual = [x['criterion_id'] for x in output['criterion_judgments']]
        expected = [x['criterion_id'] for x in payload['criteria']]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError('each input criterion must appear exactly once')
    elif role == 'selector':
        selected = output['selected_experience_ids']
        known = {x['experience_id'] for x in payload['candidates']}
        if len(selected) > payload['max_selected'] or len(selected) != len(set(selected)) or set(selected) - known:
            raise ValueError('selection must use distinct input IDs within max_selected')
    elif role == 'experience_updater':
        from verge.models import ExperienceUpdate
        ExperienceUpdate.from_dict(output)


def example_suffix(role, examples):
    parts = ['以下为独立构造的教学示例，不是当前任务或证据；编号仅在各示例内有效。只学习设计方法，不复制示例要求到当前任务。']
    for i, example in enumerate(examples, 1):
        validate(role, example['input'], example['output'])
        parts.extend([f'示例{i}输入：', llm_messages(role, example['input'])[1]['content'],
                      f'示例{i}输出：', json.dumps(example['output'], ensure_ascii=False)])
    return '\n\n'.join(parts)


def replay_messages(role, event, suffix=''):
    payload = deepcopy(event['input'])
    messages = deepcopy(event['messages'])
    if not messages or messages[0]['role'] != 'system':
        raise ValueError('source request must have its actual system message')
    # Older Updater logs precede short-case IDs: explicitly record this conversion.
    if role == 'experience_updater':
        payload = NumberedIDCodec(role, payload).project(payload)
        messages = llm_messages(role, payload)
    messages[0] = llm_messages(role, payload)[0]
    if suffix:
        messages[0]['content'] += '\n\n' + suffix
    if sum(len(m['content'].encode()) for m in messages) > 100_000:
        raise ValueError('actual replay messages exceed context budget; no truncation')
    return payload, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error('output must be separate from source')
    selection = json.loads(args.selection.read_text())
    role, cases = selection['role'], selection['cases']
    if not 1 <= len(cases) <= 4 or len({x['case_key'] for x in cases}) != len(cases):
        raise ValueError('select 1 to 4 distinct cases per role')
    if any('/' in x['case_key'] or '\\' in x['case_key'] or x['case_key'] in ('.', '..') for x in cases):
        raise ValueError('case_key must be a filename component')
    examples = json.loads(args.examples.read_text())
    if not 1 <= len(examples) <= 3:
        raise ValueError('provide 1 to 3 independent examples')
    suffix = example_suffix(role, examples)
    manifest_file = source / 'run_manifest.json'
    manifest = json.loads(manifest_file.read_text())
    root = source / manifest['config']['store']['config']['root_dir']
    trace_file = root / 'role_io.jsonl'
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    spec = manifest['config']['models']['default']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('requires fixed Qwen3.5-Flash without thinking')
    protected = [manifest_file, trace_file, root / 'experience_memory.json', root / 'trajectories.jsonl',
                 args.selection.resolve(), args.examples.resolve()]
    hashes = {str(p): digest(p) for p in protected}
    prepared = []
    for case in cases:
        event = events[case['trace_line'] - 1]
        if event['role'] != role or event['output'] is None or event['error'] is not None:
            raise ValueError('selection must identify a successful actual request of this role')
        payload, baseline = replay_messages(role, event)
        _, few_shot = replay_messages(role, event, suffix)
        prepared.append((case, event, payload, baseline, few_shot))
    replay_manifest = {'source_run': str(source), 'source_sha256': hashes, 'role': role,
        'prompt_version': PROMPT_VERSION, 'model_config': spec, 'selection': selection,
        'examples': examples, 'input_message_hashes': [fingerprint(x[3:]) for x in prepared],
        'conversion': 'Updater case IDs projected to local digits; other historical non-system messages byte-preserved',
        'scope': 'independent fixed inputs; baseline vs few-shot; no pipeline, pool writes or causal estimate',
        'script_sha256': digest(Path(__file__))}
    if args.resume:
        if json.loads((output / 'replay_manifest.json').read_text()) != replay_manifest:
            raise ValueError('resume source, inputs, prompt, script or model differs')
    else:
        output.mkdir(parents=True, exist_ok=False)
        output.chmod(0o700)
        save(output / 'replay_manifest.json', replay_manifest)
    context = {}

    def append(name, event):
        path = output / name
        with path.open('a', encoding='utf-8') as stream:
            path.chmod(0o600)
            stream.write(json.dumps({**context, **event}, ensure_ascii=False, allow_nan=False) + '\n')

    backend = OpenAICompatibleBackend(spec['model'], **{k: spec[k] for k in (
        'base_url', 'api_key_env', 'timeout_seconds', 'max_retries', 'enable_thinking') if k in spec},
        metadata_hook=lambda e: append('llm_requests.jsonl', e),
        trace_hook=lambda e: append('role_io.jsonl', e))
    results = []
    try:
        for case, original, payload, baseline, few_shot in prepared:
            for variant, initial_messages in [('baseline', baseline), ('few_shot', few_shot)]:
                context.update(case_key=case['case_key'], variant=variant)
                path = output / f"{case['case_key']}_{variant}.json"
                if args.resume and path.exists():
                    results.append(json.loads(path.read_text()))
                    continue
                messages = deepcopy(initial_messages)
                started = time.monotonic()
                result, error = None, None
                for retry in range(3):
                    try:
                        # Script-local adapter keeps actual historical Critic history intact.
                        with patch('verge.llm.llm_messages', lambda *a, **kw: deepcopy(messages)):
                            raw = backend.generate(role, payload)
                        validate(role, payload, raw)
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
                            {'role': 'user', 'content': '上次输出校验失败：' + error[:160] + '。请返回完整合法结果。'}])
                row = {**context, 'source_trace_line': case['trace_line'], 'selection': case,
                    'original_input': original['input'], 'replay_input': payload,
                    'messages': initial_messages, 'historical_output': original['output'],
                    'new_output': result, 'error': None if result is not None else error,
                    'structure_retries': retry, 'elapsed_seconds': round(time.monotonic()-started, 2)}
                save(path, row)
                results.append(row)
                print(f"{role} {case['case_key']} {variant}: {'valid' if result is not None else 'invalid'}", flush=True)
    finally:
        backend.close()
        unchanged = all(digest(Path(p)) == expected for p, expected in hashes.items())
        save(output / 'progress.json', {'completed': len(results), 'source_unchanged': unchanged})
        if not unchanged:
            raise ValueError('source artifacts changed during replay')
    save(output / 'summary.json', {'role': role, 'completed': len(results), 'source_unchanged': True,
        'valid_by_variant': dict(Counter(r['variant'] for r in results if r['new_output'] is not None)),
        'structure_retries': sum(r['structure_retries'] for r in results),
        'limitations': 'One replay per variant; semantic review required; no full-chain or generalization claim'})


if __name__ == '__main__':
    main()
