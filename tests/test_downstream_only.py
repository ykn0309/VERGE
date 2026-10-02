"""新方法：单候选、中性先验、融合不增加信用。"""
from dataclasses import replace
import pytest
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, PreferenceSample
from verge.modules import make_fake_modules
from verge.pipeline import PipelineRunner
from verge.prompts import NumberedIDCodec, llm_messages


def sample(key):
    return PreferenceSample(key, '满足关键要求', '较好回答', '较差回答', {})


def test_add_and_merge_preserve_downstream_evidence(tmp_path):
    memory = ExperienceMemory(tmp_path)
    candidate = CandidateExperience('c', '设计方法', '适用条件', admission='PASS')
    added = memory.fuse(candidate, 'origin')
    e = memory.get(added.experience_id)
    assert (e.alpha, e.beta, e.confidence, e.evidence_history) == (1, 1, .5, [])
    memory.apply_downstream_credits('used', {e.experience_id: -.4})
    before = memory.get(e.experience_id)
    memory.fuse(replace(candidate, candidate_id='c2'), 'other', e.experience_id)
    after = memory.get(e.experience_id)
    assert (after.alpha, after.beta, after.evidence_history) == (before.alpha, before.beta, before.evidence_history)
    assert after.source_trajectories == ['origin', 'other']


@pytest.mark.parametrize('count', [0, 1, 2])
def test_reflection_count_boundary_and_continuation(tmp_path, count):
    modules = make_fake_modules('attempt_1_success', reflect_count=count)
    memory = ExperienceMemory(tmp_path)
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))
    rows = runner.run([sample('first'), sample('second')])
    assert len(rows) == 2
    if count > 1:
        assert all(row.sample_error['stage'] == 'reflection' for row in rows)
        assert memory.all() == []
    else:
        assert not any(row.sample_error for row in rows)
        assert len(rows[0].reflected_candidates) == count
        assert rows[0].operation_contributions == {}
        assert not hasattr(modules, 'revision_credit_assigner')


def test_content_only_protocol_rejects_old_binding_fields():
    codec = NumberedIDCodec('reflector', {})
    assert codec.restore({'candidates': [{'content': '方法', 'trigger': '条件'}]})['candidates'][0]['candidate_id'] == 'candidate_1'
    for field, value in [('linked_operation_ids', ['op1']), ('credit', .2)]:
        with pytest.raises(ValueError):
            codec.restore({'candidates': [{'content': '方法', 'trigger': '条件', field: value}]})
    system = llm_messages('reflector', {})[0]['content']
    assert '0 到 1' in system and 'linked_operation_ids' not in system


def test_real_codec_rejects_multiple_candidates():
    codec = NumberedIDCodec('reflector', {})
    with pytest.raises(ValueError, match='0..1'):
        codec.restore({'candidates': [{'content': '方法', 'trigger': '条件'}] * 2})


def test_dashboard_no_longer_displays_binding_or_provisional_credit():
    from pathlib import Path
    text = Path('verge/web_assets/app.js').read_text()
    for obsolete in ('绑定操作', '暂定信用', '信用未绑定', 'CREDIT_FAIL', 'REFLECTION → CREDIT'):
        assert obsolete not in text
