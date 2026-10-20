from copy import deepcopy

import pytest

from scripts.replay_role_few_shot import example_suffix, replay_messages, validate
from scripts.replay_simplified_inputs import simplified_payload
from scripts.replay_role_prompt_ablation import critic_messages, grouped_evidence, GROUNDING
from scripts.replay_critic_evidence_ablation import preference_messages, without_reasons
from verge.prompts import llm_messages


def test_replay_preserves_actual_critic_history_and_isolates_variants():
    event = {'input': {'task': '原任务', 'round_number': 2}, 'messages': [
        {'role': 'system', 'content': '旧提示词'},
        {'role': 'user', 'content': '首轮原始输入'},
        {'role': 'assistant', 'content': '首轮真实回复'},
        {'role': 'user', 'content': '第二轮真实反馈'}]}
    original = deepcopy(event)
    _, baseline = replay_messages('critic', event)
    _, variant = replay_messages('critic', event, '独立示例')
    assert baseline[1:] == variant[1:] == event['messages'][1:]
    assert variant[0]['content'] == baseline[0]['content'] + '\n\n独立示例'
    variant[1]['content'] = 'changed'
    assert event == original
    assert baseline[1]['content'] == '首轮原始输入'


def test_updater_converts_only_case_ids_and_keeps_original_text():
    task = '含原始task标识 xyz 的原文'
    payload = {'current_experience': {'action':'keep','draft':None},
               'cases': [{'sample_id': 'helpsteer3:' + 'a' * 64, 'task': task}]}
    event = {'input': payload, 'messages': [{'role': 'system', 'content': 'old'}]}
    projected, messages = replay_messages('experience_updater', event)
    assert projected['cases'][0]['sample_id'] == '1'
    assert projected['cases'][0]['task'] == task
    assert task in messages[1]['content']
    assert 'a' * 64 not in messages[1]['content']
    assert payload['cases'][0]['sample_id'].startswith('helpsteer3:')


def test_role_contracts_reject_unknown_duplicate_and_missing_ids():
    with pytest.raises(ValueError):
        validate('selector', {'candidates': [{'experience_id': '1'}], 'max_selected': 1},
                 {'selected_experience_ids': ['2']})
    with pytest.raises(ValueError):
        validate('judge', {'criteria': [{'criterion_id': '1'}]}, {'criterion_judgments': []})
    with pytest.raises(ValueError):
        validate('experience_updater', {'cases': [{'sample_id': '1'}, {'sample_id': '2'}]},
                 {'experiences': [{'content': '方法', 'trigger': '条件', 'sample_ids': ['1', '1']}]})
    # Existing prompt-only length constraints remain prompt-only.
    validate('experience_updater', {'cases': [{'sample_id': '1'}]},
             {'action':'revise','draft':{'content': '长' * 200, 'trigger': '条件'}})


def test_examples_have_real_role_contracts_and_no_new_schema():
    assert '示例1输出' in example_suffix('selector', [
        {'input': {'task': '测试', 'candidates': [], 'max_selected': 1},
         'output': {'selected_experience_ids': []}}])
    with pytest.raises(ValueError):
        example_suffix('critic', [{'input': {}, 'output': {'candidates': []}}])


def test_simplified_replay_preserves_case_evidence_and_original_audit_values():
    original = {'current_experience': {'action':'keep','draft':None}, 'cases': [
        {'sample_id': '1', 'task': '完整任务', 'chosen': '完整甲', 'rejected': '完整乙',
         'applied_experience': {'content': '实际旧方法', 'trigger': '实际旧条件'},
         'credit': {'positive_credit': .7, 'negative_credit': .1}, 'criteria': [
             {'question': '核验动作？', 'weight': .8, 'source_count': 2,
              'chosen': True, 'rejected': False, 'chosen_reason': '甲依据', 'rejected_reason': '乙依据'}]}]}
    previous = deepcopy(original)
    simplified = simplified_payload('experience_updater', original)
    expected = deepcopy(previous)
    expected['cases'][0].pop('credit')
    expected['cases'][0]['criteria'][0].pop('weight')
    expected['cases'][0]['criteria'][0].pop('source_count')
    assert simplified == expected and original == previous


def test_simplified_generator_keeps_current_candidate_and_task():
    original = {'task': '完整比较任务', 'selected_experiences': [
        {'experience_id': '1', 'content': '已有方法', 'trigger': '条件', 'confidence': .8, 'evidence_amount': 5}],
        'candidate_experiences': {'experience_id': '2', 'content': '当前方法', 'trigger': '当前条件',
                                 'status': '尚未验证'}}
    previous = deepcopy(original)
    simple = simplified_payload('initial_generator', original)
    assert simple['task'] == original['task']
    assert simple['candidate_experiences'] == original['candidate_experiences']
    assert set(simple['selected_experiences'][0]) == {'experience_id', 'content', 'trigger'}
    assert original == previous


def critic_event():
    payload = {'task': '含回答A和优选回答字样的原任务', 'chosen': '原文甲：未满足',
               'rejected': '原文乙：满足', 'round_number': 1,
               'current_rubric': {'criteria': [{'criterion_id': '1', 'question': '动作是否准确？',
                                              'weight': 1, 'source_experiences': ['1']}]},
               'current_judgment': {'criterion_judgments': [{'criterion_id': '1', 'chosen': False,
                   'rejected': True, 'chosen_reason': '依据甲；保留原文', 'rejected_reason': '依据乙'}]}}
    return {'input': payload, 'messages': llm_messages('critic', payload)}


def test_critic_layout_groups_actual_status_with_reason_without_changing_source():
    event = critic_event()
    original = deepcopy(event)
    grouped = grouped_evidence(event['input'])['current_rubric']['criteria'][0]
    assert grouped['chosen'] == {'判定': '未满足', '依据': '依据甲；保留原文'}
    assert grouped['rejected'] == {'判定': '满足', '依据': '依据乙'}
    messages = critic_messages(event, event, 'layout_only')
    assert messages[0] == event['messages'][0]
    for key in ('task', 'chosen', 'rejected'):
        assert event['input'][key] in messages[1]['content']
    assert event == original


def test_critic_label_variant_preserves_raw_text_and_does_not_leak_titles():
    event = critic_event()
    labels = critic_messages(event, event, 'labels_only')
    for key in ('task', 'chosen', 'rejected'):
        assert event['input'][key] in labels[1]['content']
    assert '回答A：\n        Judge判定：未满足' in labels[1]['content']
    assert '回答B：\n        Judge判定：满足' in labels[1]['content']
    assert '回答A对应chosen' in labels[0]['content']
    assert critic_messages(event, event, 'baseline') == event['messages']


def test_critic_grounding_changes_only_system_and_second_turn_keeps_real_assistant():
    first = critic_event()
    second = deepcopy(first)
    second['input']['round_number'] = 2
    history = first['messages'] + [{'role': 'assistant', 'content': '真实旧分析（可能有错误）'}]
    second['messages'] = llm_messages('critic', second['input'], history=history)
    for variant in ('baseline', 'layout_only', 'labels_only', 'grounding_only', 'combined'):
        new = critic_messages(second, first, variant)
        assert new[2] == second['messages'][2]
        assert len(new) == 4
    grounded = critic_messages(second, first, 'grounding_only')
    assert grounded[1:] == second['messages'][1:]
    assert grounded[0]['content'] == first['messages'][0]['content'] + '\n\n' + GROUNDING


def test_critic_reason_ablation_removes_only_model_explanations():
    event = critic_event()
    original = deepcopy(event['input'])
    changed = without_reasons(original)
    expected = deepcopy(original)
    judgment = expected['current_judgment']['criterion_judgments'][0]
    judgment.pop('chosen_reason')
    judgment.pop('rejected_reason')
    assert changed == expected
    assert event['input'] == original


def test_critic_preference_pair_has_identical_evidence_and_only_mapping_differs():
    event = critic_event()
    event['input']['current_judgment'].update(chosen_score=0, rejected_score=1, margin=-1, success=False)
    original = deepcopy(event)
    visible_input, visible = preference_messages(event, True)
    hidden_input, hidden = preference_messages(event, False)
    assert visible_input == hidden_input
    assert visible[1:] == hidden[1:]
    assert visible[0]['content'] == hidden[0]['content'] + '\n\n回答A是偏好较高的chosen，回答B是偏好较低的rejected。'
    assert not set(('chosen_score', 'rejected_score', 'margin', 'success')) & set(hidden_input['current_judgment'])
    for key in ('task', 'chosen', 'rejected'):
        assert original['input'][key] in hidden[1]['content']
    assert event == original


def test_readable_critic_keeps_full_raw_evidence_and_side_associations():
    from scripts.replay_critic_readable_inputs import messages_for, VARIANTS
    event = critic_event()
    judgment = event['input']['current_judgment']['criterion_judgments'][0]
    judgment['chosen_reason'] = '甲理由第一行\n甲理由第二行'
    original = deepcopy(event)
    for variant in VARIANTS:
        messages = messages_for(event, event, variant)
        text = messages[1]['content']
        for key in ('task', 'chosen', 'rejected'):
            assert event['input'][key] in text
        assert '动作是否准确？' in text
        assert '来源经验：1' in text
        if variant == 'words_absent':
            assert judgment['chosen_reason'] not in text and judgment['rejected_reason'] not in text
        else:
            assert judgment['chosen_reason'] in text and judgment['rejected_reason'] in text
        first_label = '回答A' if variant.startswith('ab_') else '优选回答'
        second_label = '回答B' if variant.startswith('ab_') else '非优选回答'
        evidence = text.split('动作是否准确？', 1)[1]
        a, b = evidence.split(second_label + '：', 1)
        assert first_label + '：' in a
        assert 'Judge判定：' + ('false' if 'boolean' in variant else '未满足') in a
        assert 'Judge判定：' + ('true' if 'boolean' in variant else '满足') in b
    assert event == original


def test_readable_replay_retains_actual_assistant_and_controls_legend():
    from scripts.replay_critic_readable_inputs import messages_for, VARIANTS
    first = critic_event()
    second = deepcopy(first)
    second['input']['round_number'] = 2
    second['messages'] = first['messages'] + [
        {'role': 'assistant', 'content': '真实旧回复，含优选回答和false'},
        {'role': 'user', 'content': '实际反馈'}]
    systems = []
    for variant in VARIANTS:
        messages = messages_for(second, first, variant)
        assert len(messages) == 4
        assert messages[2] == second['messages'][2]
        assert '原文甲：未满足' in messages[1]['content']
        assert '原文甲：未满足' not in messages[3]['content']
        systems.append(messages[0])
    assert all(s == systems[0] for s in systems)


def test_grounded_few_shot_pair_changes_only_system_and_keeps_actual_second_turn():
    from scripts.replay_critic_grounded_few_shot import paired_messages
    first = critic_event()
    second = deepcopy(first)
    second['input']['round_number'] = 2
    second['messages'] = first['messages'] + [
        {'role': 'assistant', 'content': '真实历史，可能把B的事实当成A的事实'},
        {'role': 'user', 'content': '旧反馈展示'}]
    original = deepcopy(second)
    baseline, variant = paired_messages(second, first, '独立教学示例')
    assert baseline == llm_messages('critic', second['input'], history=[
        *llm_messages('critic', first['input']), second['messages'][2]])
    assert baseline[1:] == variant[1:]
    assert variant[0]['content'] == baseline[0]['content'] + '\n\n独立教学示例'
    assert variant[2] == second['messages'][2]
    assert second == original
    variant[1]['content'] = '修改副本'
    assert baseline[1]['content'] != '修改副本'


def test_grounded_few_shot_baseline_uses_current_production_without_diagnostic_legend():
    from scripts.replay_critic_grounded_few_shot import paired_messages
    event = critic_event()
    baseline, variant = paired_messages(event, event, '两个原创示例')
    assert baseline == llm_messages('critic', event['input'])
    assert '编号说明：回答A=' not in baseline[0]['content']
    assert variant[1:] == event['messages'][1:]
