from pathlib import Path
import json,sys
root=Path(__file__).resolve().parent;cases=json.loads((root/'cases.json').read_text());start=int(sys.argv[1]);stop=int(sys.argv[2]);priority={'added_obligation':0,'lost_requirement':1,'useful_check':2,'judge_reading_issue':3,'judge_rule_drift':4,'scope_issue':5,'requires_external_fact':6,'uncertain':7}
for c in cases[start:stop]:
 name=f"{c['input_index']:03d}-{c['sample_id'].split(':')[-1][:12]}";f=root/'v2'/'reviews'/(name+'.json');p=root/'v2'/'packets'/(name+'.json');task=' '.join(c['task'].split());print('\n'+str(c['input_index'])+(' + ' if c['outcome']=='improved' else ' - ')+task[:75]+(' … '+task[-60:] if len(task)>300 else ''))
 print('m',c['baseline']['judgment']['margin'],'>',c['actual']['judgment']['margin'])
 if not f.exists():print('PENDING');continue
 r=json.loads(f.read_text());
 if r['status']!='valid':print('FAILED');continue
 v=r['review'];packet=json.loads(p.read_text());refs=packet['refs'];print('sel',','.join(x['fit'] for x in v['selection']))
 obs=sorted(v['observations'],key=lambda o:priority[o['kind']])[:2]
 for o in obs:
  qs=[]
  for ref in o['refs']:
   key=ref.split('.')[0]
   if key in refs and key[:1] in ('N','W') and key[1:].isdigit() and key not in qs:qs.append(key)
  print(o['kind'],o['note'][:70], 'Q', ' / '.join(k+':'+refs[k][:70] for k in qs[:2]))
 if not any(o['kind']=='requires_external_fact' for o in obs) and any(o['kind']=='requires_external_fact' for o in v['observations']):print('external_fact_unverified')
