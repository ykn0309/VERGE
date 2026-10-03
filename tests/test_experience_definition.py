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
def test_experience_definition_is_inline_in_each_role_system_message(role):
    import verge.prompts as prompts
    assert not hasattr(prompts, '_EXPERIENCE_DEFINITION')
    system = llm_messages(role, {})[0]['content']
    definition = 'content 说明如何设计评价标准；trigger 说明该经验适用于哪些任务或输入条件。'
    assert system.count(definition) == 1
    assert '首次生成标准' in system
    assert '具体、可复用设计方法' in system
    assert '如何设计或调整标准' not in system
    assert 'trigger 仅凭未来 task 判断，并支撑指导的范围与强度' not in system


def test_admission_reviews_expression_and_explicit_defects_not_universal_effectiveness():
    system = llm_messages('admission_checker', {})[0]['content']
    for rule in ('审核供 rubric generator 首次生成标准使用的具体、可复用设计方法',
                 '实际有效性由后续下游任务验证',
                 '允许设计评价维度、安排标准重要性、排除不当要求',
                 '任务类别有限、需要语义理解或机制具体，本身均不足以拒绝',
                 '完整 rubric 和回答生成指南属于其他产物',
                 '理由引用相关 content 或 trigger',
                 '条件不足以支撑强制指导', '改变任务目标',
                 '依赖未来回答或既有标准/Judge', '保留样本答案或专有细节',
                 '可独立使用的多个原则', '程序不支持的评分机制',
                 '“太具体”', '“非通用原则”', '均不足以构成拒绝理由',
                 '固定数值阈值仅在机制确有需要时核对',
                 'PASS 表示经验表达合格，不等于实际有效'):
        assert rule in system
    assert '适用于未来任务和回答' not in system
    assert '保持候选文本' in system
    assert '"decision":"PASS"或"FAIL"' in system


def test_reflection_keeps_mechanism_and_scope_support():
    system = llm_messages('reflector', {})[0]['content']
    assert '用于未来任务首次生成标准' in system
    assert '修订轨迹是提炼依据' in system
    assert '如何构造或调整判断标准' not in system
    assert '保留有效具体机制与协同动作' in system
    assert 'trigger 如何支撑 content 的范围及指导强度' in system
    assert '0 到 1' in system
    assert 'linked_operation_ids' not in system


def test_fusion_keeps_conditional_branches_and_does_not_certify_effectiveness():
    system, user = llm_messages('fusion_judge', fusion_payload('方法', '条件', [('旧方法', '旧条件')]))
    assert '条件分支应分别支撑对应指导' in system['content']
    assert '条件性规则仍在其条件内适用' in system['content']
    assert '而非宣称扩展已被历史信用验证' in system['content']
    assert set(fusion_payload('方法', '条件', [])) == {'candidate', 'existing'}
    assert '任务：' not in user['content']
    assert '"match_index":null,"content":null,"trigger":null' in system['content']


@pytest.mark.parametrize('role', ['selector', 'initial_generator'])
def test_task_operation_and_material_topic_are_distinct_in_system(role):
    payload = selector_payload('总结所附材料', [], max_selected=5, training=True) if role == 'selector' else generator_payload('总结所附材料', [])
    system, user = llm_messages(role, payload)
    assert '指令要求执行的任务操作与输入材料讲述的主题' in system['content']
    assert '总结所附材料' in user['content']
    assert 'chosen' not in user['content'] and 'rejected' not in user['content']
    if role == 'selector':
        assert '材料主题相似本身不足以支持选中' in system['content']
        assert '指导强度与当前条件匹配' in system['content']
    else:
        assert '任务目标与明确约束优先于经验指导' in system['content']
        assert '材料中描述的行为与回答应执行的行为分别处理' in system['content']
        assert '与任务冲突的指导不采用' in system['content']
        assert '维度与权重的强度由当前 task 支撑' in system['content']


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
    assert 'content 说明如何设计评价标准；trigger 说明该经验适用于哪些任务或输入条件。' in system['content']
    assert content in user['content'] and trigger in user['content']
    for hidden in ('internal-candidate-id', 'chosen', 'rejected', 'improvement_advice',
                   'fusion_credit', 'evidence_weight', 'successful_attempt'):
        assert hidden not in user['content']
    assert set(admission_payload('当前任务', candidate)) == {'task', 'candidate'}
    assert PROMPT_VERSION == 'generator-dialogue-inline-experience-definition'


@pytest.mark.parametrize('version', ['generator-dialogue-critic-feedback',
                                     'generator-dialogue-fusion-revision-evidence',
                                     'generator-dialogue-task-grounded-experience',
                                     'generator-dialogue-task-grounded-experience-compact',
                                     'generator-dialogue-initial-design-experience',
                                     'generator-dialogue-initial-design-experience-v2'])
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
