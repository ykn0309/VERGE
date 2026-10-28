"""Task-only TF-IDF retrieval for a fixed, deliberately similar held-out set."""
from collections import Counter, defaultdict
from math import log, sqrt
import hashlib, json, random, re
from pathlib import Path
from verge.dataset import JsonlDataset
from scripts.prepare_general_collection import stratified_general

def tokens(text):
    words=re.findall(r'[a-z0-9]+',text.lower())
    stop=set('the a an of to and in is are for with on it this that be as i you your please'.split())
    words=[w for w in words if w not in stop]
    return words+[' '.join(words[i:i+2]) for i in range(len(words)-1)]

def prepare(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=Path('runs/helpsteer3-general-consensus-data/train.jsonl')
    population=list(JsonlDataset({'train':source}).load('train'))
    excluded={s.task_id for s in JsonlDataset({'train':'runs/helpsteer3-general-v2-collection-30-data/train.jsonl'}).load('train')}
    population=[s for s in population if s.task_id not in excluded]
    train,sampling=stratified_general(population,300,42)
    ids={s.task_id for s in train}
    candidates=[s for s in population if s.task_id not in ids]
    # Vocabulary and document frequency fitted on training TASKS ONLY.
    counts=[Counter(tokens(s.task)) for s in train]
    df=Counter(w for c in counts for w in c)
    idf={w:log((1+len(train))/(1+n))+1 for w,n in df.items()}
    def vector(task):
        values={w:(1+log(n))*idf[w] for w,n in Counter(tokens(task)).items() if w in idf}
        norm=sqrt(sum(v*v for v in values.values())) or 1
        return {w:v/norm for w,v in values.items()}
    inverted=defaultdict(list)
    for i,s in enumerate(train):
        for w,v in vector(s.task).items():inverted[w].append((i,v))
    ranked=[]
    for s in candidates:
        scores=defaultdict(float)
        for w,v in vector(s.task).items():
            for i,tv in inverted[w]:scores[i]+=v*tv
        nearest,similarity=max(scores.items(),key=lambda x:(x[1],-x[0])) if scores else (0,0)
        ranked.append((similarity,s.sample_id,nearest,s))
    ranked.sort(key=lambda r:(-r[0],r[1]))
    selected=[];per_train=Counter()
    for similarity,_,nearest,s in ranked:
        if similarity >= .90 or per_train[nearest]>=3:continue
        selected.append((s,nearest,similarity));per_train[nearest]+=1
        if len(selected)==200:break
    assert len(selected)==200
    random.Random(43).shuffle(selected)
    for name,rows in [('train',train),('test',[r[0] for r in selected])]:
        (out/f'{name}.jsonl').write_text(''.join(s.to_json()+'\n' for s in rows))
    audit={'scope':'deliberately similar held-out General subset of original train; not official test or representative overall distribution','selection':'training-task-fitted word+bigram TF-IDF cosine nearest neighbor; exclude cosine>=0.90 near duplicates; top200 with maximum3 per nearest training task; seed43 order','training_sampling':sampling,'excluded_pilot_tasks':len(excluded),'labels_or_answers_used_for_test_ranking':False,'matches':[{'sample_id':s.sample_id,'nearest_train_id':train[i].sample_id,'similarity':v} for s,i,v in selected],'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}}
    (out/'sampling.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('Prepared train300/test200; task-only similarity range',min(r[2] for r in selected),max(r[2] for r in selected))
if __name__=='__main__':prepare('runs/midterm-general-300-similar200-data')
