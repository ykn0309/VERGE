from pathlib import Path
from collections import Counter
import json,numpy as np
from verge.memory import ExperienceMemory
s=ExperienceMemory.load('runs/v31-general500-125/checkpoint.json');c=s.clusters
out={};lines=[]
for kind in ['task','experience']:
 ids,a=c._matrix(kind);_,t=c._matrix(kind,text_only=True);ix={oid:i for i,oid in enumerate(ids)};groups=sorted(c.groups(kind).items(),key=lambda g:(-len(g[1]),g[0]));result=[]
 for num,(gid,members) in enumerate(groups,1):
  pos=[ix[x] for x in members];mask=np.triu_indices(len(pos),1);sm=a[np.ix_(pos,pos)];tm=t[np.ix_(pos,pos)]
  row={'number':num,'group':gid,'size':len(pos),'members':list(members)}
  if len(pos)>1:
   row.update(mixed_mean=float(sm[mask].mean()),text_mean=float(tm[mask].mean()),mixed_min=float(sm[mask].min()),text_min=float(tm[mask].min()),minimum_member_average=float((sm.sum(1)/(len(pos)-1)).min()),low_text_pair_fraction=float((tm[mask]<.2).mean()))
   pairs=sorted((float(tm[i,j]),members[i],members[j],float(sm[i,j])) for i in range(len(pos)) for j in range(i))
   row['lowest_text_pairs']=pairs[:4]
  result.append(row)
  lines.append(f'\n{kind} GROUP {num} n={len(members)} '+json.dumps({k:v for k,v in row.items() if k not in ['members','lowest_text_pairs']}))
  for oid in members:
   if kind=='task':
    r=s.trajectories[oid];text=r.sample.task;label=r.sample.sample_id
   else:
    e=s.experiences[oid];text=json.dumps(e.method.to_dict(),ensure_ascii=False);label=oid
   lines.append(label+' '+text.replace('\n',' ')[:650])
 out[kind]=result
Path('artifacts/cluster-audit126/statistics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
Path('artifacts/cluster-audit126/members.txt').write_text('\n'.join(lines))
for kind in out:
 print(kind)
 for x in out[kind][:8]:print({k:v for k,v in x.items() if k not in ['members','lowest_text_pairs']})
