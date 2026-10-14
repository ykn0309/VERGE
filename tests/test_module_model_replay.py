from copy import deepcopy
import json
import httpx
from scripts.prepare_module_model_replay import project,SELECTION
from scripts.run_module_model_replay import RecordedClient


def test_provider_adapter_records_actual_non_thinking_request_without_credentials(monkeypatch):
    recorded=[];sent=[]
    def post(client,url,*,headers,json):
        sent.append(deepcopy(json))
        return httpx.Response(200,json={'choices':[{'message':{'content':'{}'}}]})
    monkeypatch.setattr(httpx.Client,'post',post)
    client=RecordedClient({'base_url':'https://api.deepseek.com','timeout_seconds':30,
                          'request_extras':{'thinking':{'type':'disabled'},'max_tokens':8192}},
                         lambda name,event:recorded.append((name,event)))
    original={'model':'deepseek-flash','messages':[{'role':'user','content':'原文'}],
              'response_format':{'type':'json_object'}}
    previous=deepcopy(original)
    try:
        client.post('chat/completions',headers={'Authorization':'Bearer private-test-secret'},json=original)
    finally:client.close()
    assert original==previous
    assert sent[0]==recorded[0][1]['request_body']
    assert sent[0]['thinking']=={'type':'disabled'} and 'enable_thinking' not in sent[0]
    assert 'private-test-secret' not in json.dumps(recorded)
    assert sent[0]['messages']==original['messages']
    assert recorded[1][1]['status']==200


def test_current_input_projection_preserves_raw_text_and_actual_method():
    original={'current_experience':{'content':'方法','trigger':'条件'},'cases':[
      {'sample_id':'helpsteer3:'+'x'*64,'task':'完整\n任务','chosen':'甲完整文本','rejected':'乙完整文本',
       'applied_experience':{'content':'实际旧方法','trigger':'旧条件'},'credit':{'positive_credit':.5},
       'criteria':[{'question':'问题？','weight':.5,'source_count':2,'chosen':True,'rejected':False,
                    'chosen_reason':'完整甲理由','rejected_reason':'完整乙理由'}]}]}
    previous=deepcopy(original);value=project('experience_updater',original)
    assert original==previous
    case=value['cases'][0]
    assert case['sample_id']=='1' and 'credit' not in case
    for key in ('task','chosen','rejected','applied_experience'):assert case[key]==original['cases'][0][key]
    assert 'weight' not in case['criteria'][0] and 'source_count' not in case['criteria'][0]
    assert case['criteria'][0]['chosen_reason']=='完整甲理由'


def test_selection_covers_twenty_cases_in_each_real_module_and_current_candidate_inputs():
    # Budget and coverage are part of the new experiment contract, not production model semantics.
    assert set(SELECTION)=={'selector','initial_generator','judge','critic','experience_updater'}
    for cases in SELECTION.values():
        assert len(cases)==20 and len({line for line,_,_ in cases})==20
        assert set(stratum for _,stratum,_ in cases)=={'problem','control','boundary'}
    assert sum('当前候选' in note for _,_,note in SELECTION['initial_generator'])==5
