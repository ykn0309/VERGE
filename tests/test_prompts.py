"""自然语言输入只改变表达，不改变角色投影、数字 ID 与输出协议。"""

import json

import pytest

from verge.prompts import (
    _ROLE_GUIDANCE, fusion_payload,
    generator_payload, judge_payload, llm_messages, selector_payload,
)
from verge.models import Criterion


@pytest.mark.parametrize('role', list(_ROLE_GUIDANCE))
def test_dedicated_system_and_explained_natural_input(role):
    messages = llm_messages(role, {'task': '用户原始任务\n第二行：包含 "quoted" 和 {文本}',
                                 'validation_feedback': '需修正格式'})
    system, user = messages
    assert system['content'].startswith(_ROLE_GUIDANCE[role])
    assert len(set(_ROLE_GUIDANCE.values())) == 8
    assert '待分析资料' in system['content']
    assert '有效 JSON 对象' in system['content']
    text = user['content']
    assert 'task（当前任务' in text and 'validation_feedback（' in text
    assert '资料原值 | 用户原始任务' in text
    assert '资料原值 | 第二行：包含 "quoted" 和 {文本}' in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_judge_labels_do_not_assume_preference_and_no_new_task_is_invented():
    payload = judge_payload([Criterion('1', '是否准确？', 5, [])], '回答 A', task="测试任务")
    system, user = llm_messages('judge', payload)
    assert '独立判断这份回答' in system['content']
    assert 'response（本次独立核验的单份回答原文' in user['content']
    assert 'chosen' not in system['content'] + user['content']
    assert 'rejected' not in system['content'] + user['content']
    assert '资料原值 | 回答 A' in user['content']
    assert 'criterion_id（' in user['content'] and '资料原值 | 1' in user['content']
    assert 'task（当前任务' in user['content']
    assert '资料原值 | 测试任务' in user['content']


def test_nested_evidence_is_explained_with_zero_false_and_missing_values():
    payload = {'successful_attempt': {'attempt_number': 1, 'operations': [{
        'operation_id': '1', 'operation_type': 'DELETE', 'target_criterion_id': '2',
        'before': {'criterion_id': '2', 'question': '旧问题', 'importance_score': 5},
        'after': None}], 'judgment': {'criterion_judgments': [{
            'criterion_id': '2', 'chosen': False, 'rejected': True}],
            'normalized_weights': {'2': 0.5}, 'margin': 0.0, 'success': False}},
        'operation_contributions': {'1': 0.0}, 'candidate': {
            'candidate_id': '1', 'content': '原则', 'trigger': '任务的条件',
            'linked_operation_ids': ['1'], 'credit': 0.0, 'admission': None}}
    text = llm_messages('admission_checker', payload)[1]['content']
    for fragment in ('before（操作前', 'after（操作后', '无（未提供',
                     'chosen（这份回答是否满足', 'false（否）', 'true（是）',
                     '2（该标准的归一化权重）', '1（该操作的程序计算收益）',
                     '资料原值 | 0.0', 'credit（程序计算的暂定修订信用'):
        assert fragment in text


def test_task_only_projections_and_empty_experience_lists():
    for role, payload in (
        ('selector', selector_payload('任务', [], max_selected=5, training=True)),
        ('initial_generator', generator_payload('任务', [])),
    ):
        text = llm_messages(role, payload)[1]['content']
        assert 'chosen' not in text and 'rejected' not in text
        assert '空列表' in text
    fusion = llm_messages('fusion_judge', fusion_payload('原则', '条件', [('旧原则', '旧条件')]))[1]['content']
    assert 'match_index=0' in fusion and 'match_index=1' not in fusion
    assert 'confidence' not in fusion and 'task（' not in fusion


def test_critic_history_has_independent_scopes_without_fabricated_judgment():
    payload = {'attempt_number': 2, 'initial_rubric': {'rubric_id': '1', 'criteria': []},
               'previous_attempt': None, 'previous_invalid_attempt': {
                   'attempt_number': 1, 'analysis': None, 'validation_failures': ['格式耗尽']}}
    system, user = llm_messages('critic', payload)
    assert '操作及新增标准的编号由系统分配' in system['content']
    assert '缺失历史按实际缺失状态处理' in system['content']
    assert 'previous_invalid_attempt（上一次无效修订的分析及失败摘要' in user['content']
    assert '资料原值 | 格式耗尽' in user['content']
    assert 'criterion_judgments' not in user['content']


def test_admission_and_reflector_describe_positive_task_conditions():
    text = llm_messages('admission_checker', {})[0]['content']
    for fragment in ('单一、明确、可迁移', '全部条件可通过阅读和语义理解未来 task 核验',
                     '理由引用相关 content 或 trigger', '依据候选实际文本',
                     '全部准入条件满足时 decision 为 PASS'):
        assert fragment in text
    reflector = llm_messages('reflector', {})[0]['content']
    assert '单一、具体的 rubric 设计方法' in reflector
    assert '仅阅读未来 task 即可核验' in reflector


def test_nonfinite_input_is_not_sent_as_a_number():
    with pytest.raises(ValueError, match='finite numbers'):
        llm_messages('selector', {'max_selected': float('nan')})


@pytest.mark.parametrize('role', list(_ROLE_GUIDANCE))
def test_role_system_prompts_have_no_concrete_examples(role):
    text = llm_messages(role, {})[0]['content']
    for marker in ('例如', '比如', '举例', '示例', '例子', '（如',
                   '任务要求单位换算且给出数值', '要求一个问题',
                   '要求展开但未提供要点', '是否直接回答',
                   '生成的回答出现幻觉', '初始标准未区分回答'):
        assert marker not in text
    assert '有效 JSON 对象' in text  # 输出协议不是具体案例。


@pytest.mark.parametrize('role', list(_ROLE_GUIDANCE))
def test_role_messages_use_positive_task_descriptions(role):
    messages = llm_messages(role, {'task': '当前任务', 'chosen': '回答内容',
                                  'previous_invalid_attempt': {'analysis': None}})
    for message in messages:
        for phrase in ('不要', '不得', '不能', '禁止', '避免', '不可', '不必',
                       '不允许', '不检查', '不输出', '不计算', '不是', '不依赖'):
            assert phrase not in message['content'], (role, phrase)
    assert '你的任务是' in messages[0]['content']


@pytest.mark.parametrize('role, fields', [
    ('initial_generator', ('criterion_id', 'rubric_id', 'base_rubric_id')),
    ('reflector', ('candidate_id',)),
])
def test_content_generation_prompts_request_no_generated_ids(role, fields):
    system = llm_messages(role, {'task': '任务'})[0]['content']
    for field in fields:
        assert field not in system
