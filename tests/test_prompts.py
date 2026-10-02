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
    assert len(set(_ROLE_GUIDANCE.values())) == 7
    assert '待分析资料' in system['content']
    assert '有效 JSON 对象' in system['content']
    text = user['content']
    assert '任务：' in text and '本次需修正的问题' in text
    assert '用户原始任务\n第二行：包含' in text
    assert '第二行：包含 "quoted" 和 {文本}' in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_judge_labels_do_not_assume_preference_and_no_new_task_is_invented():
    payload = judge_payload([Criterion('1', '是否准确？', 5, [])], '回答 A', task="测试任务")
    system, user = llm_messages('judge', payload)
    assert '独立判断这份回答' in system['content']
    assert '待评价回答：' in user['content']
    assert 'chosen' not in system['content'] + user['content']
    assert 'rejected' not in system['content'] + user['content']
    assert '回答 A' in user['content']
    assert '1：' in user['content']
    assert '任务：' in user['content']
    assert '测试任务' in user['content']


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
    for fragment in ('修改前', '修改后', '未提供', '优选回答：未满足', '非优选回答：满足',
                     '2：0.5', '1：0.0', '修订信用：0.0'):
        assert fragment in text


def test_task_only_projections_and_empty_experience_lists():
    for role, payload in (
        ('selector', selector_payload('任务', [], max_selected=5, training=True)),
        ('initial_generator', generator_payload('任务', [])),
    ):
        text = llm_messages(role, payload)[1]['content']
        assert 'chosen' not in text and 'rejected' not in text
        assert '暂无' in text
    fusion = llm_messages('fusion_judge', fusion_payload('原则', '条件', [('旧原则', '旧条件')]))[1]['content']
    assert '- 0：' in fusion and '- 1：' not in fusion
    assert 'confidence' not in fusion and 'task（' not in fusion


def test_critic_history_has_independent_scopes_without_fabricated_judgment():
    payload = {'attempt_number': 2, 'initial_rubric': {'rubric_id': '1', 'criteria': []},
               'previous_attempt': None, 'previous_invalid_attempt': {
                   'attempt_number': 1, 'analysis': None, 'validation_failures': ['格式耗尽']}}
    system, user = llm_messages('critic', payload)
    assert '操作及新增标准的编号由系统分配' in system['content']
    assert '缺失历史按实际缺失状态处理' in system['content']
    assert '上次无效修订' in user['content']
    assert '格式耗尽' in user['content']
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


# Critic 的候选隔离规则按用户要求使用显式禁止措辞。
@pytest.mark.parametrize('role', [role for role in _ROLE_GUIDANCE if role != 'critic'])
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


@pytest.mark.parametrize('role', list(_ROLE_GUIDANCE))
def test_natural_inputs_have_no_repeated_role_or_field_explanations(role):
    system, user = llm_messages(role, {'task': '原始任务\n保留第二行',
                                     'chosen': '回答原文', 'criteria': [
        {'criterion_id': '1', 'question': '核验问题', 'source_experiences': ['2']}]})
    for marker in ('角色标识', '当前角色：', '资料原值', '字段说明', '（当前任务',
                   '这份回答是否满足该标准的已记录判断'):
        assert marker not in system['content'] + user['content']
    assert '原始任务\n保留第二行' in user['content']
    assert '1：' in user['content'] and '来源经验：2' in user['content']


def test_technical_explanations_appear_once_for_multiple_experiences():
    experience = {'experience_id': '1', 'content': '原则', 'trigger': '适用条件',
                  'confidence': 0.5, 'evidence_amount': 0}
    text = llm_messages('selector', {'task': '任务', 'candidates': [experience, experience]})[1]['content']
    assert text.count('历史信用的 Beta 证据分数') == 1
    assert text.count('置信度：0.5') == 2


@pytest.mark.parametrize('role', list(_ROLE_GUIDANCE))
def test_input_objects_use_compact_lists_without_generated_markdown_headings(role):
    payload = {'task': '原始任务', 'criteria': [
        {'criterion_id': '1', 'question': '第一项要求？', 'importance_score': 5,
         'source_experiences': ['2']},
        {'criterion_id': '2', 'question': '第二项要求？', 'importance_score': 0,
         'source_experiences': []}],
        'selected_experiences': [{'experience_id': '2', 'content': '经验原则',
                                  'trigger': '明确条件', 'confidence': 0.0, 'evidence_amount': 0}],
        'previous_attempt': {'analysis': '私有分析', 'operations': [
            {'operation_id': '1', 'operation_type': 'MODIFY', 'target_criterion_id': '2',
             'before': {'criterion_id': '2', 'question': '旧问题', 'importance_score': 3},
             'after': {'criterion_id': '2', 'question': '新问题', 'importance_score': 5}}],
            'judgment': {'criterion_judgments': [
                {'criterion_id': '2', 'chosen': False, 'rejected': True}], 'margin': 0.0}},
        'previous_invalid_attempt': None}
    text = llm_messages(role, payload)[1]['content']
    assert all(not line.lstrip().startswith('#') for line in text.splitlines())
    assert '- 1：第一项要求？' in text
    assert '- 2：第二项要求？' in text
    assert '重要性：5；来源经验：2' in text
    assert '重要性：0；来源经验：暂无' in text
    assert '- 2：经验原则' in text
    assert '置信度：0.0；证据量：0' in text
    assert '- 1（MODIFY），目标标准：2' in text
    assert '修改前：' in text and '2：旧问题' in text
    assert '修改后：' in text and '2：新问题' in text
    assert '优选回答：未满足；非优选回答：满足' in text
    assert '上次无效修订：未提供' in text


def test_raw_markdown_and_multiline_text_remain_unmodified():
    raw = '# 原始资料标题\n\n- 用户提供的列表\n  原始缩进\n```python\nx = 1\n```'
    text = llm_messages('judge', {'task': raw, 'response': raw, 'criteria': []})[1]['content']
    assert text.count(raw) == 2
    assert text.startswith('任务：\n' + raw)


def test_fusion_list_keeps_zero_based_positions_without_item_headings():
    text = llm_messages('fusion_judge', fusion_payload(
        '新原则', '新条件', [('第一原则', '第一条件'), ('第二原则', '第二条件')]))[1]['content']
    assert '- 0：' in text
    assert '- 1：' in text
    assert text.index('第一原则') < text.index('第二原则')
    assert '第一条件' in text and '第二条件' in text
    assert all(not line.lstrip().startswith('#') for line in text.splitlines())


def test_critic_targets_are_explicitly_scoped_to_r0_not_history():
    payload = {"initial_rubric": {"criteria": [
        {"criterion_id": "1", "question": "当前问题", "importance_score": 5}]},
        "previous_attempt": {"rubric": {"criteria": [
            {"criterion_id": "2", "question": "历史新增标准", "importance_score": 5}]}}}
    text = llm_messages("critic", payload)[1]["content"]
    assert text.startswith("本轮从初始标准 R0 重新修订，目标编号取自 R0：1。")
    assert "- 2：历史新增标准" in text


def test_judge_criterion_references_have_numeric_only_labels():
    text = llm_messages("judge", judge_payload([
        Criterion("1", "核验要求甲？", 5, []),
        Criterion("2", "核验要求乙？", 5, [])], "回答", task="任务"))[1]["content"]
    assert "- 1：核验要求甲？" in text and "- 2：核验要求乙？" in text
    assert "标准 1" not in text and "标准 2" not in text


@pytest.mark.parametrize('role,history_key', [('critic', 'previous_attempt'),
                                           ('reflector', 'successful_attempt')])
def test_revision_evidence_is_joined_by_id_and_displayed_once(role, history_key):
    from copy import deepcopy
    rubric = {'rubric_id': 'private-r', 'base_rubric_id': 'private-base', 'criteria': [
        {'criterion_id': '1', 'question': '要求甲？', 'importance_score': 8},
        {'criterion_id': '2', 'question': '要求乙？', 'importance_score': 3}]}
    judgment = {'rubric_id': 'private-r', 'normalized_weights': {'1': 0.72, '2': 0.28},
                'criterion_judgments': [
                    {'criterion_id': '2', 'chosen': False, 'rejected': True},
                    {'criterion_id': '1', 'chosen': True, 'rejected': False}],
                'chosen_score': 0.72, 'rejected_score': 0.28, 'margin': 0.44, 'success': True}
    payload = {'task': '原始任务', 'initial_rubric': rubric, 'initial_judgment': judgment,
               history_key: {'rubric': rubric, 'judgment': judgment, 'operations': []}}
    original = deepcopy(payload)
    text = llm_messages(role, payload)[1]['content']
    assert text.count('- 1：要求甲？\n      重要性：8；优选回答：满足；非优选回答：未满足') == 1
    # The history rubric has deeper indentation but the same correct ID-based association.
    assert text.count('重要性：8；优选回答：满足；非优选回答：未满足') == 2
    assert text.count('重要性：3；优选回答：未满足；非优选回答：满足') == 2
    for redundant in ('标准集编号', '起点标准集', 'private-r', 'private-base', '标准权重',
                      'normalized_weights', '逐项判断', '补充说明'):
        assert redundant not in text
    assert text.count('得分差：0.44') == 2
    assert payload == original


@pytest.mark.parametrize('role', ['critic', 'reflector'])
def test_missing_judgment_remains_missing_not_fabricated(role):
    text = llm_messages(role, {'initial_rubric': {'criteria': [
        {'criterion_id': '1', 'question': '要求？', 'importance_score': 5}]},
        'initial_judgment': None})[1]['content']
    assert '- 1：要求？' in text
    assert '初始评价结果：未提供' in text
    assert '优选回答：满足' not in text and '非优选回答：未满足' not in text


@pytest.mark.parametrize('role', ['critic', 'reflector'])
def test_scores_are_rounded_only_for_prompt_display(role):
    payload = {'chosen_score': 0.7142857142857143,
               'rejected_score': 0.3333333333333333,
               'margin': -0.00000001}
    original = dict(payload)
    text = llm_messages(role, payload)[1]['content']
    assert '0.7143' in text and '0.3333' in text
    assert '得分差：0' in text
    assert '714285714' not in text and '-0' not in text
    assert payload == original


def test_reflection_merges_changes_and_strict_op_references():
    from copy import deepcopy
    from verge.prompts import NumberedIDCodec
    old = {'criterion_id': 'a', 'question': '原要求', 'importance_score': 5}
    new = {'criterion_id': 'a', 'question': '新要求', 'importance_score': 6}
    payload = {'initial_rubric': {'criteria': [old]},
               'initial_judgment': {'criterion_judgments': [
                   {'criterion_id': 'a', 'chosen': False, 'rejected': True}]},
               'successful_attempt': {'rubric': {'criteria': [new]},
                   'judgment': {'criterion_judgments': [
                       {'criterion_id': 'a', 'chosen': True, 'rejected': False}]},
                   'operations': [{'operation_id': 'private-op', 'operation_type': 'MODIFY',
                                   'target_criterion_id': 'a', 'before': old, 'after': new}]}}
    original = deepcopy(payload)
    codec = NumberedIDCodec('reflector', payload)
    text = llm_messages('reflector', codec.project(payload))[1]['content']
    assert text.count('原要求') == text.count('新要求') == 1
    assert '- op1（MODIFY）' in text
    assert '优选回答：满足' in text and '优选回答：未满足' in text
    assert payload == original
    candidate = {'content': '方法', 'trigger': '条件', 'linked_operation_ids': ['op1']}
    assert codec.restore({'candidates': [candidate]})['candidates'][0]['linked_operation_ids'] == ['private-op']
    for invalid in ('1', 'op2', 1):
        with pytest.raises(ValueError):
            codec.restore({'candidates': [{**candidate, 'linked_operation_ids': [invalid]}]})


def test_critic_explicitly_separates_candidate_analysis_from_operations():
    messages = llm_messages('critic', {'task': '任务'})
    text = messages[0]['content']
    for fragment in ('严禁泄露候选回答内容', '两份回答的比较评价', '候选专有事实',
                     'task 本身或独立于候选回答的领域知识', '只保留在 analysis'):
        assert fragment in text


def test_reflector_requires_mechanism_level_transfer_not_sample_specific_rules():
    text = llm_messages('reflector', {'task': '任务'})[0]['content']
    for phrase in ('评价设计变化', '用语义类别替代', '迁移性指同类任务间复用',
                   '操作体现的关键机制', '不同对象后', '不写入候选'):
        assert phrase in text
