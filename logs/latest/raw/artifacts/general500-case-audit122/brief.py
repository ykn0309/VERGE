from pathlib import Path
import json,sys
root=Path(__file__).resolve().parent
cases=json.loads((root/'cases.json').read_text());start=int(sys.argv[1]) if len(sys.argv)>1 else 0;stop=int(sys.argv[2]) if len(sys.argv)>2 else len(cases)
for c in cases[start:stop]:
 key=f"{c['input_index']:03d}-{c['sample_id'].split(':')[-1][:12]}";f=root/'v2'/'reviews'/(key+'.json');packet=root/'v2'/'packets'/(key+'.json')
 print('\n',c['input_index'],c['outcome'],c['task'].replace('\n',' ')[:115], (' ... '+c['task'].replace('\n',' ')[-110:] if len(c['task'])>350 else ''))
 print(' E', ' / '.join(e['method']['trigger'][:70] for e in c['selected_experiences']))
 b=c['baseline']['judgment'];a=c['actual']['judgment'];print(' Δ',b['margin'],'->',a['margin'])
 if not f.exists():print(' PENDING');continue
 rev=json.loads(f.read_text())
 if rev['status']!='valid':print(' FAILED',rev.get('errors'));continue
 refs=json.loads(packet.read_text())['refs'];v=rev['review']
 print(' SEL',','.join(e['id']+':'+e['fit'] for e in v['selection']))
 for obs in v['observations']:
  if obs['kind'] in ('requires_external_fact','uncertain'):continue
  keyrefs=[r for r in obs['refs'] if (r.startswith('W') or r.startswith('N')) and '.reason' not in r and '.analysis' not in r]
  print(' ?',obs['kind'],obs['note'][:100], ' | ', ' / '.join(refs[r][:90] for r in keyrefs[:2]))
