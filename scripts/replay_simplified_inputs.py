"""Replay only simplified Updater/Generator inputs against saved plan52 baselines."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from scripts.replay_experience_splits import digest, save
from scripts.replay_role_few_shot import validate
from verge.llm import LLMFormatError, LLMRefusalError, OpenAICompatibleBackend
from verge.prompts import PROMPT_VERSION, llm_messages


def simplified_payload(role, original):
    """Explicit historical input conversion; no changes to task/answers/credit storage."""
    payload = deepcopy(original)
    if role == 'experience_updater':
        for case in payload['cases']:
            case.pop('credit')
            for criterion in case['criteria']:
                criterion.pop('weight')
                criterion.pop('source_count')
    elif role == 'initial_generator':
        for experience in payload['selected_experiences']:
            experience.pop('confidence')
            experience.pop('evidence_amount')
    else:
        raise ValueError('only Updater/Generator inputs change')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.baseline_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents:
        parser.error('output must be separate from baseline directory')
    previous = json.loads((source / 'replay_manifest.json').read_text())
    role, spec = previous['role'], previous['model_config']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('requires Qwen3.5-Flash without thinking')
    rows = []
    protected = {**previous['source_sha256'], str(source / 'replay_manifest.json'): digest(source / 'replay_manifest.json')}
    for case in previous['selection']['cases']:
        path = source / (case['case_key'] + '_baseline.json')
        protected[str(path)] = digest(path)
        row = json.loads(path.read_text())
        if row['new_output'] is None:
            raise ValueError('baseline must be valid')
        payload = simplified_payload(role, row['replay_input'])
        messages = llm_messages(role, payload)
        if sum(len(m['content'].encode()) for m in messages) > 100_000:
            raise ValueError('context budget exceeded; no truncation')
        rows.append((row, payload, messages))
    if not 1 <= len(rows) <= 4:
        raise ValueError('bounded to 1–4 fixed cases per role')
    if any(digest(Path(p)) != expected for p, expected in protected.items()):
        raise ValueError('source changed since original replay')
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    save(output / 'replay_manifest.json', {'baseline_dir': str(source), 'role': role,
        'source_sha256': protected, 'model_config': spec, 'prompt_version': PROMPT_VERSION,
        'script_sha256': digest(Path(__file__)),
        'scope': 'only simplified input replay; saved original baseline reused; no few-shot or pipeline',
        'conversion': 'Updater removes credit/weight/source_count; Generator removes confidence/evidence_amount'})
    context = {}

    def append(name, event):
        path = output / name
        with path.open('a', encoding='utf-8') as stream:
            path.chmod(0o600)
            stream.write(json.dumps({**context, **event}, ensure_ascii=False, allow_nan=False)+'\n')

    backend = OpenAICompatibleBackend(spec['model'], **{key: spec[key] for key in
        ('base_url', 'api_key_env', 'timeout_seconds', 'max_retries', 'enable_thinking') if key in spec},
        trace_hook=lambda e: append('role_io.jsonl', e),
        metadata_hook=lambda e: append('llm_requests.jsonl', e))
    completed = []
    try:
        for baseline, payload, messages in rows:
            context.update(case_key=baseline['case_key'])
            started = time.monotonic()
            result, error = None, None
            for retry in range(3):
                try:
                    raw = backend.generate(role, payload)
                    validate(role, payload, raw)
                    result = raw
                    break
                except (LLMFormatError, LLMRefusalError) as exc:
                    error = type(exc).__name__+': '+str(exc)
                    break
                except ValueError as exc:
                    error = str(exc)
                    if retry == 2:
                        break
                    payload = {**payload, 'validation_feedback': error[:160]}
            row = {**context, 'selection': baseline['selection'],
                'baseline_input': baseline['replay_input'], 'new_input': payload,
                'baseline_messages': baseline['messages'], 'new_messages': messages,
                'baseline_output': baseline['new_output'], 'new_output': result,
                'structure_retries': retry, 'error': None if result is not None else error,
                'elapsed_seconds': round(time.monotonic()-started, 2)}
            save(output / (baseline['case_key']+'.json'), row)
            completed.append(row)
            print(f"{role} {baseline['case_key']}: {'valid' if result is not None else 'invalid'}", flush=True)
    finally:
        backend.close()
        unchanged = all(digest(Path(p)) == expected for p, expected in protected.items())
        save(output / 'progress.json', {'completed': len(completed), 'source_unchanged': unchanged})
        if not unchanged:
            raise ValueError('source artifacts changed')
    save(output / 'summary.json', {'completed': len(completed), 'valid': sum(r['new_output'] is not None for r in completed),
        'source_unchanged': True, 'structure_retries': sum(r['structure_retries'] for r in completed),
        'limitations': 'Saved baseline reused, not contemporaneous; local semantic review required, no causal claim'})


if __name__ == '__main__':
    main()
