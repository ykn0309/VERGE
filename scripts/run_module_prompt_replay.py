"""Bounded process-isolated Qwen prompt comparisons, with separate confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

from scripts.replay_experience_splits import digest, save
import scripts.run_module_model_replay as worker

ROOT=Path('runs/module-prompt-replay-20')

def verify(root):
    manifest=json.loads((root/'replay_manifest.json').read_text())
    if digest(root/'prepared_inputs.json')!=manifest['input_sha256']: raise ValueError('frozen inputs changed')
    if any(digest(Path(p))!=h for p,h in manifest['source_sha256'].items()): raise ValueError('protected file changed')
    return manifest

def run(root,job):
    worker.ROOT=root
    return worker.run_group(job)

def main(root):
    manifest=verify(root)
    # Existing worker expects its model map, all roles and actual-message frozen rows.
    jobs=[(role,'qwen',variant) for role in manifest['roles'] for variant in manifest['variants'][role]]
    completed=[];futures={}
    pool=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))
    try:
        futures={pool.submit(run,root,job):job for job in jobs}
        for future in as_completed(futures):
            completed.append(future.result())
            save(root/'progress.json',{'completed_groups':len(completed),'planned_groups':len(jobs),'groups':completed})
    except BaseException:
        for future in futures: future.cancel()
        for process in list(pool._processes.values()): process.terminate()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True);verify(root)
    save(root/'summary.json',{'groups':completed,'planned_outputs':manifest['planned_outputs'],
        'completed_outputs':sum(r['completed'] for r in completed),'valid_outputs':sum(r['valid'] for r in completed),
        'protected_sources_unchanged':True})
    print('all planned prompt outputs completed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    main(args.root)
