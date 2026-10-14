"""Replay the 27 historical split inputs without writing their Memory or trajectories.

python -m scripts.replay_experience_splits --source-run <run> --output-dir <new-dir>
Use --resume only to finish this same bounded replay after an infrastructure error.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from verge.llm import LLMFormatError, OpenAICompatibleBackend
from verge.models import Experience, ExperienceUpdate
from verge.prompts import PROMPT_VERSION, NumberedIDCodec, llm_messages


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path: Path, value) -> None:
    with path.open('w', encoding='utf-8') as stream:
        path.chmod(0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error('output must be separate from the source run')
    review = source / 'semantic_review'
    files = [review / name for name in ('split_evidence_01_14.json', 'split_evidence_15_27.json')]
    items = sorted(sum([json.loads(p.read_text()) for p in files], []), key=lambda x: x['split_index'])
    if [i['split_index'] for i in items] != list(range(1, 28)):
        raise ValueError('expected exactly the 27 original splits')
    manifest_file = source / 'run_manifest.json'
    source_manifest = json.loads(manifest_file.read_text())
    root = source / source_manifest['config']['store']['config']['root_dir']
    memory_file, trace_file = root / 'experience_memory.json', root / 'role_io.jsonl'
    parents = {e['experience_id']: Experience.from_dict(e)
               for e in json.loads(memory_file.read_text())['experiences']}
    # Verify that each replay card is an actual recorded request, not a rebuilt approximation.
    recorded = set()
    with trace_file.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['role'] == 'experience_updater':
                recorded.add(fingerprint(event['input']))
    for item in items:
        payload = item['actual_input']
        if fingerprint(payload) not in recorded:
            raise ValueError(f"split {item['split_index']}: input not found in raw trace")
        cases = parents[item['parent_id']].cases
        if [c.sample_id for c in cases] != [c['sample_id'] for c in payload['cases']]:
            raise ValueError('source case order does not match')
        ExperienceUpdate.from_dict(item['actual_output']).validate_cases(cases)
        projected = NumberedIDCodec('experience_updater', payload).project(payload)
        if not 2 <= len(cases) <= 10 or sum(len(m['content'].encode()) for m in
                llm_messages('experience_updater', projected)) > 100_000:
            raise ValueError('replay exceeds production case or context budget')
    judgments_file = review / 'review_summary.json'
    judgments = {r['split_index']: r['verdict'] for r in
                 json.loads(judgments_file.read_text())['split_decisions']}
    spec = source_manifest['config']['models']['default']
    if spec['model'] != 'qwen3.5-flash' or spec['enable_thinking'] is not False:
        raise ValueError('replay requires Qwen3.5-Flash without thinking')
    protected = [*files, judgments_file, manifest_file, memory_file, trace_file, root / 'trajectories.jsonl']
    hashes = {str(p): digest(p) for p in protected}
    replay_manifest = {
        'source_run': str(source), 'source_sha256': hashes, 'prompt_version': PROMPT_VERSION,
        'system_sha256': fingerprint(llm_messages('experience_updater', items[0]['actual_input'])[0]),
        'model_config': spec, 'split_indices': list(range(1, 28)),
        'scope': 'independent fixed-input replay; no pool writes, training, held-out or causal estimate',
    }
    if args.resume:
        if json.loads((output / 'replay_manifest.json').read_text()) != replay_manifest:
            raise ValueError('resume source, prompt or model differs')
    else:
        output.mkdir(parents=True, exist_ok=False)
        output.chmod(0o700)
        save(output / 'replay_manifest.json', replay_manifest)
    current_index = None

    def append(name: str, event) -> None:
        path = output / name
        with path.open('a', encoding='utf-8') as stream:
            path.chmod(0o600)
            stream.write(json.dumps({'split_index': current_index, **event},
                                   ensure_ascii=False, allow_nan=False) + '\n')

    backend = OpenAICompatibleBackend(
        spec['model'], **{key: spec[key] for key in ('base_url', 'api_key_env',
            'timeout_seconds', 'max_retries', 'enable_thinking') if key in spec},
        metadata_hook=lambda event: append('llm_requests.jsonl', event),
        trace_hook=lambda event: append('role_io.jsonl', event),
    )
    results = []
    try:
        for item in items:
            current_index = item['split_index']
            path = output / f'split_{current_index:02d}.json'
            if args.resume and path.exists():
                row = json.loads(path.read_text())
                if row['new_output'] is not None:
                    ExperienceUpdate.from_dict(row['new_output']).validate_cases(parents[item['parent_id']].cases)
                elif row['decision'] != 'invalid':
                    raise ValueError('invalid saved replay result')
                results.append(row)
                continue
            payload = item['actual_input']
            codec = NumberedIDCodec('experience_updater', payload)
            payload = codec.project(payload)
            started = time.monotonic()
            update, error = None, None
            for retry in range(3):
                try:
                    candidate = ExperienceUpdate.from_dict(codec.restore(backend.generate('experience_updater', payload)))
                    candidate.validate_cases(parents[item['parent_id']].cases)
                    update = candidate
                    break
                except LLMFormatError as exc:
                    error = str(exc)
                    break
                except ValueError as exc:
                    if retry == 2:
                        error = str(exc)
                        break
                    payload = {**payload, 'validation_feedback': '上次文本校验失败：' + str(exc)[:160]}
            row = {'split_index': current_index, 'parent_id': item['parent_id'],
                   'sample_id': item['sample_id'], 'old_review': judgments[current_index],
                   'old_output': item['actual_output'], 'new_output': update.to_dict() if update else None,
                   'decision': ('split' if len(update.experiences) == 2 else 'one') if update else 'invalid',
                   'error': error,
                   'structure_retries': retry, 'elapsed_seconds': round(time.monotonic() - started, 2)}
            save(path, row)
            results.append(row)
            counts = Counter(r['decision'] for r in results)
            print(f"{current_index}/27: {row['decision']}; cumulative {dict(counts)}", flush=True)
    finally:
        backend.close()
        unchanged = {str(p): digest(p) for p in protected} == hashes
        save(output / 'progress.json', {'completed': len(results), 'source_unchanged': unchanged})
        if not unchanged:
            raise ValueError('source artifacts changed during replay')
    save(output / 'summary.json', {
        'completed': len(results), 'source_unchanged': True,
        'decisions': dict(Counter(r['decision'] for r in results)),
        'by_old_review': {label: dict(Counter(r['decision'] for r in results if r['old_review'] == label))
                          for label in ('unnecessary', 'reasonable', 'uncertain')},
        'structure_retries': sum(r['structure_retries'] for r in results),
        'limitations': 'Old qualitative reviews are references, not gold labels; new text needs semantic review.',
    })


if __name__ == '__main__':
    main()
