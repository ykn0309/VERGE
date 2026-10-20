"""Frozen retrieval comparison: vector-only vs hybrid on the SAME document index."""
import json
from pathlib import Path
from collections import Counter
from verge.modules import HybridRetriever
from verge.models import Experience,ExperienceCategory
from verge.llm import OpenAICompatibleEmbedding
from scripts.replay_versioned_evolution import SOURCE,DEST,write,sha

def main():
    root=DEST/'retrieval-comparison'
    if root.exists():raise ValueError('retrieval comparison exists')
    root.mkdir()
    source=json.loads((DEST/'manifest.json').read_text())
    for p,h in source['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('source changed')
    rows=json.loads((DEST/'inputs.json').read_text());events=[];cache={}
    def metadata(x):events.append(x)
    backend=OpenAICompatibleEmbedding('text-embedding-v4',base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',api_key_env='DASHSCOPE_API_KEY_2',dimensions=1024,timeout_seconds=120,max_retries=2,metadata_hook=metadata)
    class Cached:
        def embed(self,text):
            if text not in cache:cache[text]=backend.embed(text)
            return cache[text]
    retriever=HybridRetriever(Cached());results=[]
    try:
        for row in rows:
            if row['role'] not in ('selector','category_organizer'):continue
            p=row['input'];ex=[];cats=[]
            if row['role']=='selector':
                for u in p['candidates']:
                    grouped='members' in u
                    if grouped:cats.append(ExperienceCategory(u['category_id'],u['name'],u['description']))
                    for e in u.get('members',[u]):ex.append(Experience(e['experience_id'],e['content'],e['trigger'],1,1,[],[],category_id=u['category_id'] if grouped else None))
                query=p['task'];kw={};kind='task'
            else:
                ex=[Experience.from_dict(x) for x in p['pool']];cats=[ExperienceCategory.from_dict(x) for x in p['categories']]
                e=p['target'];query=e['trigger']+'\n'+e['content'];kw={'category_only':True,'exclude':e['experience_id']};kind='classification'
            hybrid,_=retriever.rank(query,ex,cats,**kw)
            vector={}
            for d in retriever.last_trace:vector[d['unit_id']]=max(vector.get(d['unit_id'],-2),d['vector_score'])
            dense=sorted(vector,key=lambda k:(-vector[k],k))
            results.append({'key':row['key'],'kind':kind,'sample_id':row['sample_id'],'query':query,
                'unit_count':len(hybrid),'dense':dense,'hybrid':hybrid,'trace':retriever.last_trace,
                'top5_changed':set(dense[:5])!=set(hybrid[:5]),'lexical_hits':sum(d['bm25_rank'] is not None for d in retriever.last_trace)})
            print(row['key'],kind,'units',len(hybrid),'top5_changed',results[-1]['top5_changed'],flush=True)
    finally:backend.close()
    for p,h in source['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('source changed')
    write(root/'results.json',results);write(root/'embedding_metadata.json',events)
    write(root/'summary.json',{'queries':len(results),'kinds':dict(Counter(r['kind'] for r in results)),
        'top5_candidate_set_changes':sum(r['top5_changed'] for r in results),'zero_lexical_query_count':sum(r['lexical_hits']==0 for r in results),
        'embedding_requests':len(events),'distinct_embedding_texts':len(cache),'source_unchanged':True,
        'limitation':'No human relevance gold set. Candidate/order differences do NOT establish precision or recall improvement. Small pools make Top20 full recall.'})

if __name__=='__main__':main()
