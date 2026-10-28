"""Wait for a successful training manifest, then run a frozen paired test."""
import argparse, json, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('config','test','train-dir','output'):p.add_argument('--'+name,required=True)
p.add_argument('--workers',type=int,default=8)
a=p.parse_args();train=Path(a.train_dir)
while True:
    manifest=train/'run_manifest.json'
    if manifest.exists():
        state=json.loads(manifest.read_text()).get('state')
        if state=='training_completed':break
        if state in ('failed','interrupted'):raise RuntimeError(f'training did not complete: {state}')
    time.sleep(30)
subprocess.run(['/opt/anaconda3/bin/python','-u','scripts/midterm_paired_evaluation.py','--config',a.config,'--test',a.test,'--train-dir',str(train),'--output',a.output,'--workers',str(a.workers)],check=True)
