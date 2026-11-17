"""Read-only protocol gate; no metric threshold for semantic correctness."""
from pathlib import Path
import json,hashlib
import numpy as np
from verge.memory import ExperienceMemory

def check(root):
 root=Path(root);s=ExperienceMemory.load(root/'checkpoint.json');c=s.clusters
 for kind in ['task','experience']:
  ids,matrix=c._matrix(kind)
  for i,left in enumerate(ids):
   for j in range(i):assert abs(matrix[i,j]-c.similarity(kind,left,ids[j]))<1e-12
 for eid in s.experiences:
  if not c.references('experience',eid):
   gid=c.assignments('experience')[eid];assert c.groups('experience')[gid]==(eid,)
 for r in s.trajectories.values():
  assert len(r.paths)<=4 and all(len(p.provided_experience_ids)<=1 for p in r.paths)
  if r.critic_rounds:assert r.paths and all(p.correct is False for p in r.paths)
 tests=[json.loads(x) for x in (root/'test.jsonl').read_text().splitlines()]
 assert all(x['mode']=='inference' and len(x['paths'])<=1 and not x['critic_rounds'] for x in tests)
 result=json.loads((root/'test_controls/manifest.json').read_text());assert result['status']=='completed'
 assert hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest()==result['frozen_sha256']
 return {'flow_passed':True,'train':len(s.trajectories),'test':len(tests),'pool':len(s.experiences),'task_groups':len(c.groups('task')),'experience_groups':len(c.groups('experience')),'result':result}
if __name__=='__main__':print(json.dumps(check('runs/reference127'),ensure_ascii=False))
