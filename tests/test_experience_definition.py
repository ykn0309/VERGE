"""Prompt-boundary regression checks, not measured semantic accuracy."""
import json

import pytest

from verge.models import CandidateExperience
from verge.modules import LLMAdmissionChecker
from verge.prompts import (
    PROMPT_VERSION, admission_payload, fusion_payload,
    generator_payload, llm_messages, selector_payload,
)


@pytest.mark.parametrize('role', ['reflector', 'admission_checker', 'fusion_judge'])
def test_experience_roles_use_prospective_guidance_without_shared_definition(role):
    import verge.prompts as prompts
    assert not hasattr(prompts, '_EXPERIENCE_DEFINITION')
    system = llm_messages(role, {})[0]['content']
    assert '首次生成标准' in system
    assert '已有 rubric' in system and 'Judge 结果' in system
    assert '已观察到的候选特征' in system or '已经观察到这些性质' in system
    assert '单一设计机制' not in system
    assert '仅凭未来 task 核验' not in system
    assert '协同动作' in system


def test_admission_allows_reasoning_and_complexity_but_rejects_concrete_defects():
    system = llm_messages('admission_checker', {})[0]['content']
    for rule in ('表达是否可用，而非提前证明实际有效',
                 '具体、可复用且指导连贯', '允许结合任务语义',
                 '实际输入、上下文和领域知识', '进行合理推理',
                 '多个协同动作本身不足以拒绝',
                 'content 可以指导核验未来回答的性质',
                 'trigger 不应以已经观察到这些性质为前提',
                 '允许设计评价维度、安排重要性及排除不当要求',
                 '尊重任务目标与明确约束', '只因明确缺陷返回 FAIL',
                 '事后触发条件', '样本答案细节或偏好身份',
                 '指导与适用条件不匹配', '拼接互不相关的指导',
                 '程序不支持的评分机制', '理由引用实际 content 或 trigger',
                 '不能仅因复杂、具体或属于评分标准设计方法而拒绝',
                 'PASS 保持候选文本，不等于实际有效'):
        assert rule in system
    assert '固定数值阈值' not in system
    assert '无需二次推断' not in system
    assert '"decision":"PASS"或"FAIL"' in system


def test_reflection_prioritizes_observed_design_changes_over_critic_story():
    system = llm_messages('reflector', {})[0]['content']
    for rule in ('对照修订前后标准与两侧 Judge 判断', '实际产生区分的评价维度',
                 'Critic 分析只作参考', '不把全部修订动作都解释为有效方法',
                 '不把判别成功解释为经验的因果贡献',
                 'content 说明首次生成时如何设计评价标准',
                 'trigger 说明生成标准之前可识别的适用场景',
                 '将修订诊断转化为事前指导', '不要求只有一个操作',
                 '不保留样本答案细节或偏好身份'):
        assert rule in system
    assert '0 到 1' in system and 'linked_operation_ids' not in system


def test_fusion_defaults_to_add_and_deduplicates_instead_of_concatenating():
    system, user = llm_messages('fusion_judge', fusion_payload('方法', '条件', [('旧方法', '旧条件')]))
    for rule in ('识别候选与已有经验是否重复', '默认独立新增候选',
                 '同一评价设计问题', '指导等价或属于同一方法的必要细化',
                 '适用条件兼容', '共享质量目标、主题或关键词不足以合并',
                 '不确定是否重复时返回空匹配', '去重提炼为统一指导',
                 '不逐条追加无关原则', '扩展无关场景或修复事后触发条件',
                 '历史证据不认证新增内容'):
        assert rule in system['content']
    assert set(fusion_payload('方法', '条件', [])) == {'candidate', 'existing'}
    assert '任务：' not in user['content']
    assert '"match_index":null,"content":null,"trigger":null' in system['content']


@pytest.mark.parametrize('role', ['selector', 'initial_generator'])
def test_task_analysis_and_explicit_constraints_remain_preference_blind(role):
    payload = selector_payload('总结所附材料', [], max_selected=5, training=True) if role == 'selector' else generator_payload('总结所附材料', [])
    system, user = llm_messages(role, payload)
    assert '总结所附材料' in user['content']
    assert 'chosen' not in user['content'] and 'rejected' not in user['content']
    if role == 'selector':
        assert '允许合理推理' in system['content']
        assert '不要求条件由 task 逐字声明' in system['content']
        assert '高置信度不能代替适用性' in system['content']
    else:
        assert '实际输入与完整性' in system['content']
        assert '当前条件下合理的回答行为' in system['content']
        assert '任务目标与明确约束优先于经验指导' in system['content']


@pytest.mark.parametrize('decision,content,trigger,reason', [
    ('PASS', '核心要求与形式要求按任务主次调整相对权重', '任务明确区分核心与形式要求', '权重设计是具体可复用机制，条件支持主次调整'),
    ('PASS', '核验不同地域约定对应的单位标准', '任务涉及不同地域约定下的单位换算', '限定地域差异并不影响同类任务迁移'),
    ('FAIL', '立即执行的操作步骤必须最高权重', '任务寻求帮助', '“寻求帮助”条件不足以支持最高权重的强制指导'),
    ('FAIL', '强化科学事实核验', '候选回答包含伪科学', '“候选回答包含伪科学”依赖未来回答'),
])
def test_real_admission_http_uses_aligned_system_and_unchanged_projection(
        monkeypatch, decision, content, trigger, reason):
    from test_modules import initial_case, revise
    from test_real_llm import backend, result
    modules, initial, judgment = initial_case()
    attempt = revise(modules, initial, judgment, 1)
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return result({'decision': decision, 'reason': reason})
    candidate = CandidateExperience('internal-candidate-id', content, trigger)
    checker = LLMAdmissionChecker(backend(monkeypatch, handler, max_retries=0))
    reviewed = checker.check('当前任务', candidate, attempt)
    assert (reviewed.admission, reviewed.admission_reason) == (decision, reason)
    assert reviewed.content == content and reviewed.trigger == trigger
    assert candidate.admission is None and reviewed.credit is None
    assert reviewed.fusion_record is None
    assert len(bodies) == 1
    system, user = bodies[0]['messages']
    assert '允许结合任务语义' in system['content']
    assert '生成标准之前可识别' in system['content']
    assert content in user['content'] and trigger in user['content']
    for hidden in ('internal-candidate-id', 'chosen', 'rejected', 'improvement_advice',
                   'fusion_credit', 'evidence_weight', 'successful_attempt'):
        assert hidden not in user['content']
    assert set(admission_payload('当前任务', candidate)) == {'task', 'candidate'}
    assert PROMPT_VERSION == 'generator-dialogue-program-owned-revision-sources'


@pytest.mark.parametrize('version', ['generator-dialogue-critic-feedback',
                                     'generator-dialogue-fusion-revision-evidence',
                                     'generator-dialogue-task-grounded-experience',
                                     'generator-dialogue-task-grounded-experience-compact',
                                     'generator-dialogue-initial-design-experience',
                                     'generator-dialogue-initial-design-experience-v2',
                                     'generator-dialogue-inline-experience-definition',
                                     'generator-dialogue-role-specific-output',
                                     'generator-dialogue-role-specific-output-compact',
                                     'generator-dialogue-concise-revision-feedback',
                                     'generator-dialogue-prospective-experience-conservative-fusion'])
def test_read_only_web_accepts_known_previous_versions_without_rewriting(tmp_path, version):
    from test_web import _run
    from verge.web import load_run
    directory, _ = _run(tmp_path)
    manifest_path = directory / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['prompt_versions'] = dict.fromkeys(manifest['prompt_versions'], version)
    manifest_path.write_text(json.dumps(manifest))
    files = list((directory / 'artifacts').iterdir()) + [manifest_path]
    original = {path: path.read_bytes() for path in files}
    assert len(load_run(directory)['samples']) == 3
    assert all(path.read_bytes() == value for path, value in original.items())
