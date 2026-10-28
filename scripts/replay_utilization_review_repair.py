"""Generator-only review/one-repair experiment; frozen draft, no Selector/Judge."""
import json,sys,hashlib,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import replay_experience_utilization as common
import verge.llm as llm
from verge.cli import backend
from verge.config import load_config
ROOT=common.ROOT
ORIGINAL=llm.llm_messages
REVIEW='''你是独立的评分标准任务依据核对员。只看task、候选经验和草稿标准，不看回答或偏好。检查最终标准是否忠实核验用户要回答完成的任务，而非把历史经验变成额外义务。重点检查：1.是否凭陌生名称断言实体存疑、虚构或不存在；2.是否强制任务没有要求的声明、标签、禁止建议或复述；3.是否把输入中已发生的背景改成输出必须再完成的事；4.是否遗漏明确的数量、对象、文体或用户保留的操作权。通用事实准确性、合理连贯性不必在task逐字出现；保持原文风格不等于必须每种标点不变。不要编造外部事实核验或新增要求。approved为true仅当未发现实质问题，否则false；reason简短指出草稿中的标准序号、对应task证据和需要去掉/恢复什么。仅返回JSON：{"approved":true,"reason":"简短可核对结论"}。判断可能不确定时保守，不因个人风格偏好拒绝。'''
def messages(role,payload,**kwargs):
    if role=='experience_checker':
        return [{'role':'system','content':REVIEW},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
    return ORIGINAL(role,payload,**kwargs)
llm.llm_messages=messages

def run():
    out=ROOT/'round7';out.mkdir(exist_ok=False)
    cases=[c for c in json.loads((ROOT/'fixed_inputs.json').read_text()) if c['split']=='development' and c['role']=='initial_generator']
    drafts={r['sample_id']:r for r in map(json.loads,(ROOT/'round5/results.jsonl').read_text().splitlines()) if r['role']=='initial_generator'}
    spec=load_config('configs/midterm-general-v2-assumed-train-300-batch3.yaml')['models']['default']
    lock=threading.Lock()
    def append(name,r):
        with lock:
            with (out/name).open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    def one(c):
        b=backend(spec);sid=c['sample_id'];ctx={'sample_id':sid,'variant':'review_repair','split':'development'}
        b.trace_hook=lambda e:append('role_io.jsonl',{**ctx,**e})
        b.metadata_hook=lambda e:append('llm_requests.jsonl',{**ctx,**e})
        draft=drafts[sid]['output']; row={**ctx,'role':'initial_generator','group':c['group'],'draft':draft}
        token=common.VARIANT.set('debiased')
        try:
            review=b.generate('experience_checker',{'task':c['input']['task'],'experiences':c['input']['experiences'],'draft':draft,'_message_budget':100000})
            assert type(review['approved']) is bool and isinstance(review['reason'],str)
            row['review']=review
            if review['approved']:result=draft
            else:
                payload={**c['input'],'draft_to_repair':draft,'independent_review':review,'repair_instruction':'核对报告也可能有误。逐条与task比对，只修正可确认的问题；去掉经验造成的额外义务，恢复task明确要求，保留草稿已有正确标准。不要因报告声称就编造事实。完整返回analysis和criteria。'}
                result=b.generate('initial_generator',payload)
            common.valid('initial_generator',result,c['input'])
            row.update(valid=True,output=result)
        except Exception as exc:row.update(valid=False,error={'type':type(exc).__name__,'message':str(exc)})
        finally:common.VARIANT.reset(token);b.close()
        append('results.jsonl',row);return row
    (out/'protocol.json').write_text(json.dumps({'model':spec,'review_prompt':REVIEW,'cases':30,'workflow':'reuse fixed round5 actual draft -> one independent review -> at most one repair; no other roles; no preference/answers','draft_sha256':hashlib.sha256((ROOT/'round5/results.jsonl').read_bytes()).hexdigest(),'added_calls':'one review per case; rejected draft adds one repair; draft generation cost excluded from replay, must add in deployment'},ensure_ascii=False,indent=2))
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(one,c) for c in cases]
        for i,f in enumerate(as_completed(fs),1):f.result();print(f'{i}/30',flush=True)
if __name__=='__main__':run()
