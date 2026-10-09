"""Critic证据去重、真实HTTP消息及角色契约；不验证LLM语义准确率。"""

from copy import deepcopy
from dataclasses import replace
import json

import httpx
import pytest

from verge.calculator import calculate_rubric_judgment
from verge.llm import OpenAICompatibleBackend
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import Criterion, CriterionJudgment, InvalidRevisionAttempt, PreferenceSample, RevisionAttempt, Rubric
from verge.modules import LLMCritic, make_fake_modules
from verge.pipeline import PipelineRunner
from verge.prompts import NumberedIDCodec, critic_payload, llm_messages, reflection_payload


@pytest.fixture
def evidence():
    r0 = Rubric('private-r0', None, [
        Criterion('old-a', '原有数量要求？', 10, []),
        Criterion('old-b', '原有表达要求？', 3, [])])
    j0 = calculate_rubric_judgment(r0, [
        CriterionJudgment('old-b', True, True),
        CriterionJudgment('old-a', False, True)])
    r1 = Rubric('private-r1', r0.rubric_id, [
        Criterion('new-a', '原有数量要求？', 10, []),
        Criterion('new-b', '调整后的表达要求？', 5, [])])
    j1 = calculate_rubric_judgment(r1, [
        CriterionJudgment('new-b', True, True),
        CriterionJudgment('new-a', False, True)])
    attempt = RevisionAttempt(1, '旧公开建议哨兵', [], r1, j1,
                              validation_failures=['已恢复校验错误哨兵'],
                              analysis='旧私有诊断哨兵', previous_rubric=r0, previous_judgment=j0)
    return r0, j0, attempt


def projected_messages(payload):
    codec = NumberedIDCodec('critic', payload)
    return codec, llm_messages('critic', codec.project(payload))


def test_first_round_contains_only_current_r0(evidence):
    r0, j0, _ = evidence
    payload = critic_payload('任务原文', '私有优选原文', '私有非优选原文', r0, j0, 1)
    _, messages = projected_messages(payload)
    text = messages[1]['content']
    assert payload['current_baseline'] == 'R0'
    assert 'previous_attempt' not in payload and 'previous_invalid_attempt' not in payload
    assert '即将进行的修订轮次：1；当前有效基线：R0' in text
    assert text.count('原有数量要求？') == text.count('原有表达要求？') == 1
    assert '上一轮修改记录' not in text and '（修改后）' not in text


def test_second_round_shows_each_version_once_and_only_public_history(evidence):
    r0, j0, attempt = evidence
    snapshot = deepcopy(attempt.to_dict())
    payload = critic_payload('任务\n原文', '私有优选原文', '私有非优选原文',
                             attempt.rubric, attempt.judgment, 2, previous_attempt=attempt)
    before_render = deepcopy(payload)
    _, messages = projected_messages(payload)
    text = messages[1]['content']
    history = payload['previous_attempt']
    assert set(history) == {'attempt_number', 'previous_rubric', 'previous_judgment', 'improvement_advice'}
    assert history['previous_rubric']['rubric_id'] == r0.rubric_id
    assert history['previous_judgment'] == j0.to_dict()
    assert payload['current_baseline'] == 'R1'
    assert text.count('原有表达要求？') == text.count('调整后的表达要求？') == 1
    assert text.count('原有数量要求？') == 2  # unchanged question, one per version
    assert text.count('旧公开建议哨兵') == 1
    assert text.count('得分差：') == 2
    assert text.index('修改前标准：') < text.index('当前有效标准（修改后）：')
    assert '即将进行的修订轮次：2；当前有效基线：R1' in text
    for forbidden in ('旧私有诊断哨兵', '已恢复校验错误哨兵', 'private-r0', 'private-r1',
                      '逐项判断', '标准权重'):
        assert forbidden not in text
    assert payload == before_render and attempt.to_dict() == snapshot
    assert '任务\n原文' in text and '私有优选原文' in text and '私有非优选原文' in text


def test_numbering_joins_judgments_by_id_not_criterion_position(evidence):
    _, _, attempt = evidence
    payload = critic_payload('任务', '好', '坏', attempt.rubric, attempt.judgment, 2,
                             previous_attempt=attempt)
    codec, messages = projected_messages(payload)
    text = messages[1]['content']
    for criterion, weight, chosen, rejected in [
        ('old-a', 10, '未满足', '满足'), ('old-b', 3, '满足', '满足'),
        ('new-a', 10, '未满足', '满足'), ('new-b', 5, '满足', '满足')]:
        rubric = attempt.previous_rubric if criterion.startswith('old') else attempt.rubric
        question = next(c.question for c in rubric.criteria if c.criterion_id == criterion)
        assert f'- {codec.forward["criterion"][criterion]}：{question}' in text
        assert f'重要性：{weight}；优选回答：{chosen}；非优选回答：{rejected}' in text
    assert len(codec.forward['criterion']) == 4
    assert codec.forward['criterion']['old-a'] != codec.forward['criterion']['new-a']


@pytest.mark.parametrize('has_private_diagnosis', [False, True])
def test_invalid_first_round_is_summary_only_and_current_baseline_stays_r0(evidence, has_private_diagnosis):
    r0, j0, _ = evidence
    invalid = InvalidRevisionAttempt(1, '无效公开建议哨兵' if has_private_diagnosis else None,
                                     ['结构校验耗尽'], '无效私有诊断哨兵' if has_private_diagnosis else None)
    snapshot = deepcopy(invalid.to_dict())
    payload = critic_payload('任务', '好', '坏', r0, j0, 2, previous_invalid_attempt=invalid)
    _, messages = projected_messages(payload)
    summary = payload['previous_invalid_attempt']
    assert set(summary) == {'attempt_number', 'validation_failures', 'judgment_status'}
    assert summary['validation_failures'] == invalid.validation_failures
    assert payload['current_baseline'] == 'R0' and payload['previous_attempt'] is None
    text = messages[1]['content']
    assert '未形成有效修订，无Judge结果' in text and '结构校验耗尽' in text
    assert text.count('原有数量要求？') == 1 and text.count('得分差：') == 1
    assert '无效公开建议哨兵' not in text and '无效私有诊断哨兵' not in text
    assert invalid.to_dict() == snapshot
    summary['validation_failures'].append('只修改副本')
    assert invalid.to_dict() == snapshot


def test_missing_before_evidence_is_not_reconstructed(evidence):
    _, _, attempt = evidence
    legacy = replace(attempt, previous_rubric=None, previous_judgment=None)
    payload = critic_payload('任务', '好', '坏', legacy.rubric, legacy.judgment, 2, previous_attempt=legacy)
    _, messages = projected_messages(payload)
    assert '修改前标准：未提供' in messages[1]['content']
    assert '修改前得分：未提供' in messages[1]['content']
    assert messages[1]['content'].count('得分差：') == 1


def test_actual_http_input_keeps_contract_without_old_analysis(monkeypatch, evidence):
    _, _, attempt = evidence
    monkeypatch.setenv('CRITIC_EVIDENCE_TEST_KEY', 'mock')
    requests, trace = [], []
    reply = {'analysis': '本轮私有诊断', 'improvement_advice': '保留合理数量要求，调整表达维度。'}

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(reply, ensure_ascii=False)},
                                                     'finish_reason': 'stop'}]})

    with httpx.Client(base_url='https://api.openai.com/v1/', transport=httpx.MockTransport(handler)) as client:
        backend = OpenAICompatibleBackend('mock', api_key_env='CRITIC_EVIDENCE_TEST_KEY', client=client,
                                          max_retries=0, trace_hook=trace.append)
        feedback = LLMCritic(backend).critique('任务', '好', '坏', attempt.rubric, attempt.judgment, 2,
                                             previous_attempt=attempt)
    assert feedback.to_dict() == reply
    assert len(requests) == 1 and trace[0]['output'] == reply
    assert json.loads(trace[0]['raw_output']) == reply
    assert requests[0]['messages'] == trace[0]['messages']
    text = requests[0]['messages'][1]['content']
    assert text.count('调整后的表达要求？') == 1 and '旧私有诊断哨兵' not in text
    assert requests[0]['response_format'] == {'type': 'json_object'}


@pytest.mark.parametrize('scenario', ['attempt_2_success', 'final_failure'])
def test_pipeline_keeps_full_audit_but_not_old_analysis_in_next_critic(tmp_path, scenario):
    modules = make_fake_modules(scenario)
    memory, log = ExperienceMemory(tmp_path), TrajectoryLog(tmp_path)
    record = PipelineRunner(modules, memory, log, experience_enabled=False).run_sample(
        PreferenceSample('one', '任务原文', '优选哨兵', '非优选哨兵', {}))
    first, second = record.revision_attempts
    critic_calls = [p for role, p in modules.backend.requests if role == 'critic']
    assert first.analysis and first.feedback
    assert 'analysis' not in critic_calls[1]['previous_attempt']
    assert first.analysis not in json.dumps(critic_calls[1], ensure_ascii=False)
    assert log.get('one').to_dict() == record.to_dict()
    assert second.previous_rubric == first.rubric and critic_calls[1]['current_baseline'] == 'R1'
    generator_calls = [p for role, p in modules.backend.requests if role == 'initial_generator']
    for p in generator_calls:
        content = json.dumps(p, ensure_ascii=False)
        assert '优选哨兵' not in content and '非优选哨兵' not in content and first.analysis not in content
    assert generator_calls[2]['_conversation'][0] == generator_calls[1]['_conversation'][0]
    reflection = reflection_payload('任务', record.initial_rubric, record.initial_judgment,
                                    first, chosen='好', rejected='坏')
    assert reflection['successful_attempt']['analysis'] == first.analysis
    assert reflection['successful_attempt']['improvement_advice'] == first.feedback


def test_preservation_advice_does_not_add_a_skip_revision_branch(tmp_path):
    modules = make_fake_modules('attempt_1_success')
    original = modules.backend.generate

    def generate(role, payload):
        if role == 'critic':
            modules.backend.requests.append((role, deepcopy(payload)))
            return {'analysis': '证据不足以认定标准错误', 'improvement_advice': '保留合理标准，不新增无依据要求。'}
        return original(role, payload)

    modules.backend.generate = generate
    record = PipelineRunner(modules, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path),
                            experience_enabled=False).run_sample(PreferenceSample('one', '任务', '好', '坏', {}))
    assert record.state == 'revision_success_1'
    assert len(record.revision_attempts) == 1
    assert sum(role == 'initial_generator' for role, _ in modules.backend.requests) == 2


def test_critic_role_is_evidence_grounded_and_public_boundary_is_explicit():
    system = llm_messages('critic', {})[0]['content']
    for phrase in ('判别失败本身不证明标准有错', '对照实际修改前后标准与判断',
                   '只是待核验的修改意图', '重要性归一化加权的是/否判断',
                   '不建议部分得分、一票否决或不适用项自动排除', '不要求每条标准都区分两侧',
                   '任务、上下文或合理领域推理依据', '如何调整及任务依据',
                   '允许 task 本身的信息和独立领域知识', '候选特有内容、偏好身份、评分或比较评价',
                   '无可靠修改依据时建议保留合理标准'):
        assert phrase in system
    assert '说明当前标准存在的问题' not in system


def test_generator_preserves_task_priority_in_same_role_system():
    system = llm_messages('initial_generator', {})[0]['content']
    for phrase in ('任务目标与明确约束优先于经验和修订建议', '结合 task 判断建议是否适用',
                   '保留合理标准，仅调整有依据的问题', 'analysis 中简述关键调整依据',
                   '不为提高判别分差降低明确约束的重要性'):
        assert phrase in system
    assert '程序统一置空' not in system and '一票否决' not in system
