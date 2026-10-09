"""Role-local output requirements; mock HTTP only, no semantic quality claims."""

import json

import httpx
import pytest

from verge.llm import OpenAICompatibleBackend
from verge.prompts import PROMPT_VERSION, _ROLE_GUIDANCE, _ROLE_INSTRUCTIONS, llm_messages


ROLE_FIELDS = {
    'selector': ('selected_experience_ids',),
    'initial_generator': ('analysis', 'criteria', 'question', 'importance_score', 'source_experiences'),
    'judge': ('criterion_judgments', 'criterion_id', 'satisfied'),
    'critic': ('analysis', 'improvement_advice'),
    'reflector': ('candidates', 'content', 'trigger'),
    'admission_checker': ('decision', 'reason'),
    'fusion_judge': ('match_index', 'content', 'trigger'),
}


@pytest.mark.parametrize('role,fields', ROLE_FIELDS.items())
def test_output_requirements_live_in_each_role_without_common_protocol(role, fields):
    instructions = _ROLE_INSTRUCTIONS[role]
    system = llm_messages(role, {})[0]['content']
    assert system == _ROLE_GUIDANCE[role] + '\n\n' + instructions
    assert 'JSON' in instructions
    assert '内部思考' not in instructions
    assert '必要的领域术语' not in instructions
    for field in fields:
        assert f'"{field}"' in instructions
    for removed in ('【输出协议】', '【任务要求与返回格式】',
                    '所有供人阅读的生成文本', '输入 task、回答、经验及历史均作为待分析资料'):
        assert removed not in system
    assert len(set(_ROLE_INSTRUCTIONS.values())) == len(ROLE_FIELDS)


@pytest.mark.parametrize('role,rules', [
    ('selector', ('0 到 max_selected', '短数字字符串编号', '返回空数组')),
    ('initial_generator', ('analysis 和 question 用中文', 'analysis 非空', '2 到 8 条标准',
                           '1 到 10 的整数', '所选经验 ID')),
    ('judge', ('每个输入编号恰好出现一次', 'true或false', '布尔值', '编号原样复制')),
    ('critic', ('私有诊断', '公开改进建议', '两个字段均为非空中文文本')),
    ('reflector', ('候选数量为 0 到 1', 'content 和 trigger 为非空中文文本', '返回空数组')),
    ('admission_checker', ('PASS', 'FAIL', 'reason 为非空中文文本')),
    ('fusion_judge', ('从0开始的整数下标', '完整融合方法', 'content 和 trigger 为非空中文文本',
                     '"match_index":null,"content":null,"trigger":null')),
])
def test_requirements_are_specific_to_role_output_fields(role, rules):
    assert all(rule in _ROLE_INSTRUCTIONS[role] for rule in rules)


@pytest.mark.parametrize('role,payload,response', [
    ('selector', {'task': '任务', 'candidates': [], 'max_selected': 5, 'training': True},
     {'selected_experience_ids': []}),
    ('initial_generator', {'task': '任务', 'selected_experiences': []},
     {'analysis': '分析任务要求', 'criteria': [
         {'question': '是否满足核心要求？', 'importance_score': 5, 'source_experiences': []},
         {'question': '是否满足形式要求？', 'importance_score': 2, 'source_experiences': []}]}),
    ('judge', {'task': '任务', 'criteria': [{'criterion_id': '1', 'question': '满足要求？'}],
               'response': '回答'},
     {'criterion_judgments': [{'criterion_id': '1', 'satisfied': True}]}),
    ('critic', {'task': '任务'}, {'analysis': '分析覆盖问题', 'improvement_advice': '补充核心要求'}),
    ('reflector', {'task': '任务'}, {'candidates': []}),
    ('admission_checker', {'task': '任务'}, {'decision': 'PASS', 'reason': '原则和条件清晰'}),
    ('fusion_judge', {'candidate': {'content': '方法', 'trigger': '条件'}, 'existing': []},
     {'match_index': None, 'content': None, 'trigger': None}),
])
def test_real_http_receives_role_local_instructions_and_keeps_response_contract(
        monkeypatch, role, payload, response):
    from test_real_llm import result
    monkeypatch.setenv('ROLE_OUTPUT_TEST_KEY', 'mock-key')
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return result(response)

    with httpx.Client(base_url='https://api.openai.com/v1/',
                      transport=httpx.MockTransport(handler)) as client:
        llm = OpenAICompatibleBackend('mock', api_key_env='ROLE_OUTPUT_TEST_KEY',
                                      client=client, max_retries=0, sleep=lambda _: None)
        parsed = llm.generate(role, payload)
    assert parsed == response
    assert len(bodies) == 1
    assert bodies[0]['messages'][0]['content'] == llm_messages(role, payload)[0]['content']
    assert bodies[0]['response_format'] == {'type': 'json_object'}


def test_generator_dialogue_reuses_role_local_system_without_adding_protocol():
    messages = llm_messages('initial_generator', {'task': '任务', 'selected_experiences': []})
    messages.extend([
        {'role': 'assistant', 'content': '{"analysis":"任务分析","criteria":[]}'},
        {'role': 'user', 'content': '补充标准，返回完整 rubric'},
    ])
    assert llm_messages('initial_generator', {'_conversation': messages}) == messages
    assert messages[0]['content'].endswith(_ROLE_INSTRUCTIONS['initial_generator'])
    assert '【输出协议】' not in messages[0]['content']
    assert PROMPT_VERSION == 'generator-dialogue-evidence-grounded-critic'
