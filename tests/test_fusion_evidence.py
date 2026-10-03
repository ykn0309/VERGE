"""Expandable text fusion and discounted revision evidence; no real requests."""
from dataclasses import replace
import json

import pytest

from verge.calculator import calculate_fusion_credit
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, Experience, FusionResult, PreferenceSample
from verge.modules import LLMFusionJudge, make_fake_modules
from verge.pipeline import PipelineRunner
from verge.prompts import fusion_payload
from verge.web import _evolution


def candidate(key='c'):
    return CandidateExperience(key, '单独核验核心要求', '任务有核心要求', admission='PASS')


def test_add_merge_update_text_beta_and_preserve_history(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = memory.fuse(candidate(), 'a', fusion_credit=.1)
    eid = added.experience_id
    memory.apply_downstream_credits('b', {eid: -.4})
    before = memory.get(eid)
    merged = memory.fuse(candidate('c2'), 'c', eid, merged_content='核心要求与兼容的新维度分别核验',
                         merged_trigger='任务包含核心要求或新增维度', fusion_credit=.2)
    after = ExperienceMemory(tmp_path).get(eid)
    assert (after.content, after.trigger) == (merged.fusion_record.content, merged.fusion_record.trigger)
    assert (after.alpha, after.beta, after.confidence) == pytest.approx((1.3, 1.4, 1.3/2.7))
    assert after.evidence_history[:2] == before.evidence_history
    assert [e.evidence_type for e in after.evidence_history] == ['fusion_credit', 'downstream_credit', 'fusion_credit']
    assert after.evidence_history[-1].candidate_id == 'c2'
    assert after.source_trajectories == ['a', 'c']
    assert len(after.fusion_history) == 2
    assert after.fusion_history[-1].previous_content == before.content
    assert after.fusion_history[-1].previous_trigger == before.trigger
    assert before.content == candidate().content
    after.fusion_history[-1].content = 'mutated snapshot'
    assert memory.get(eid).fusion_history[-1].content != 'mutated snapshot'


@pytest.mark.parametrize('credit', [-.1, .26, float('nan'), float('inf'), True])
def test_invalid_credit_cannot_partially_modify_memory(tmp_path, credit):
    memory = ExperienceMemory(tmp_path)
    eid = memory.fuse(candidate(), 'a', fusion_credit=.1).experience_id
    original = memory.path.read_bytes()
    with pytest.raises(ValueError):
        memory.fuse(candidate('c2'), 'b', eid, merged_content='更新文本', merged_trigger='条件', fusion_credit=credit)
    assert memory.path.read_bytes() == original
    assert memory.get(eid).content == candidate().content


@pytest.mark.parametrize('scenario', ['attempt_1_success', 'attempt_2_success'])
def test_credit_uses_successful_round_valid_base_and_test_is_frozen(tmp_path, scenario):
    modules = make_fake_modules(scenario)
    memory = ExperienceMemory(tmp_path)
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))
    sample = PreferenceSample('a', '核心要求', '好', '坏', {})
    record = runner.run_sample(sample)
    attempt = record.revision_attempts[-1]
    expected = .25 * min(1, attempt.judgment.margin - attempt.previous_judgment.margin)
    assert calculate_fusion_credit(attempt.previous_rubric, attempt.previous_judgment,
                                   attempt.rubric, attempt.judgment) == pytest.approx(expected)
    assert record.reflected_candidates[0].fusion_record.credit == pytest.approx(expected)
    assert memory.all()[0].alpha == pytest.approx(1 + expected)
    original = memory.path.read_bytes()
    frozen = PipelineRunner(make_fake_modules(scenario), ExperienceMemory(tmp_path, read_only=True),
                            TrajectoryLog(tmp_path), training=False)
    frozen.run_sample(replace(sample, sample_id='test'))
    assert memory.path.read_bytes() == original


@pytest.mark.parametrize('output', [
    {'match_index': True, 'content':'c', 'trigger':'t'},
    {'match_index': 2, 'content':'c', 'trigger':'t'},
    {'match_index': 0, 'content':'', 'trigger':'t'},
    {'match_index': None, 'content':'c', 'trigger':'t'},
    {'match_index': 0},
    {'match_index': 0, 'content':'c', 'trigger':'t', 'credit':1},
])
def test_fusion_rejects_invalid_result(output):
    class Backend:
        def generate(self, role, payload):
            return output
    with pytest.raises(ValueError):
        LLMFusionJudge(Backend()).find_match('c', 't', [('old', 'condition')])


def test_fusion_adapter_returns_complete_text_and_only_text_projection():
    calls = []
    class Backend:
        def generate(self, role, payload):
            calls.append(payload)
            return {'match_index': 0, 'content':'兼容扩展', 'trigger':'事前条件'}
    module = LLMFusionJudge(Backend())
    assert module.find_match('新指导', '条件', [('旧指导', '旧条件')]) == FusionResult(0, '兼容扩展', '事前条件')
    assert calls == [fusion_payload('新指导', '条件', [('旧指导', '旧条件')])]
    assert module.find_match('c', 't', []) == FusionResult(None, None, None)
    assert len(calls) == 1


def test_old_memory_decodes_without_fabricating_new_evidence():
    old = {'experience_id':'e', 'content':'旧方法', 'trigger':'旧条件', 'alpha':1, 'beta':1,
           'source_trajectories':['a'], 'evidence_history':[], 'confidence':.5}
    experience = Experience.from_dict(old)
    assert experience.fusion_history == [] and experience.evidence_history == []


def test_web_fusion_evidence_links_text_history_even_without_complete_trajectory(tmp_path):
    memory = ExperienceMemory(tmp_path)
    eid = memory.fuse(candidate(), 'a', fusion_credit=.1).experience_id
    memory.fuse(candidate('c2'), 'b', eid, merged_content='扩展方法', merged_trigger='扩展条件', fusion_credit=.2)
    warnings = []
    events = _evolution(memory.get(eid).to_dict(), [], {'a':0}, warnings)
    assert not warnings
    assert [ev['kind'] for ev in events] == ['prior', 'fusion_credit', 'fusion_credit']
    assert events[-1]['fusion'] == 'MERGE'
    assert events[-1]['fusion_record']['content'] == '扩展方法'
    assert not events[-1]['has_trajectory']
    assert events[-1]['alpha'] == pytest.approx(1.3)


def test_invalid_fusion_keeps_admission_but_never_writes_fusion_credit(tmp_path):
    memory = ExperienceMemory(tmp_path)
    memory.fuse(candidate(), 'origin')
    modules = make_fake_modules('attempt_1_success')
    modules.fusion_judge.find_match = lambda *args: FusionResult(20, '扩展方法', '条件')
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))
    record = runner.run_sample(PreferenceSample('a', '核心要求', '好', '坏', {}))
    assert record.sample_error['stage'] == 'fusion_judge'
    assert record.reflected_candidates[0].admission == 'PASS'
    assert record.reflected_candidates[0].fusion is None
    assert not any(ev.evidence_type == 'fusion_credit' for e in memory.all() for ev in e.evidence_history)


def test_real_http_fusion_retries_old_shape_and_preserves_full_text(monkeypatch):
    from test_real_llm import backend, result
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return result({'match_index':0} if len(calls) == 1 else
                      {'match_index':0, 'content':'融合方法', 'trigger':'融合条件'})
    module = LLMFusionJudge(backend(monkeypatch, handler, max_retries=1))
    assert module.find_match('新指导', '条件', [('旧指导', '旧条件')]) == FusionResult(0, '融合方法', '融合条件')
    assert len(calls) == 2
    assert '新指导' in calls[0]['messages'][1]['content']
    assert '旧指导' in calls[0]['messages'][1]['content']
    assert 'fusion_credit' not in calls[0]['messages'][1]['content']


def test_pipeline_applies_expanded_fusion_and_next_sample_reads_latest_text(tmp_path):
    memory = ExperienceMemory(tmp_path)
    eid = memory.fuse(candidate(), 'seed').experience_id
    modules = make_fake_modules('attempt_1_success')
    modules.selector.select = lambda task, views, **kwargs: list(views[:1])
    modules.fusion_judge.find_match = lambda *args: FusionResult(0, '兼容扩展后的标准设计', '任务包含核心要求或兼容分支')
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))
    rows = runner.run([PreferenceSample('a', '核心要求', '好', '坏', {}),
                       PreferenceSample('b', '核心要求', '好', '坏', {})])
    assert all(row.reflected_candidates[0].fusion == 'MERGE' for row in rows)
    assert memory.get(eid).content == '兼容扩展后的标准设计'
    generators = [payload for role, payload in modules.backend.requests
                  if role == 'initial_generator' and not payload.get('revision_number')]
    assert generators[1]['selected_experiences'][0]['content'] == '兼容扩展后的标准设计'
    from verge.calculator import calculate_existing_experience_credit
    credits = [calculate_existing_experience_credit(r.initial_rubric, r.initial_judgment).get(eid, 0) for r in rows]
    assert memory.get(eid).alpha == pytest.approx(1 + sum(max(0, c) for c in credits) + 2 * .25 * 2/3)
    assert memory.get(eid).beta == pytest.approx(1 + sum(max(0, -c) for c in credits))


def test_invalid_first_round_uses_r0_for_second_round_credit(tmp_path):
    from verge.llm import LLMFormatError
    modules = make_fake_modules('attempt_2_success')
    revise = modules.initial_generator.revise
    def fail_first(previous, advice, number, **kwargs):
        if number == 1:
            raise LLMFormatError('no valid first revision')
        return revise(previous, advice, number, **kwargs)
    modules.initial_generator.revise = fail_first
    memory = ExperienceMemory(tmp_path)
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))
    record = runner.run_sample(PreferenceSample('a', '核心要求', '好', '坏', {}))
    attempt = record.revision_attempts[-1]
    assert attempt.previous_rubric == record.initial_rubric
    expected = .25 * min(1, attempt.judgment.margin - record.initial_judgment.margin)
    assert memory.all()[0].evidence_history[0].credit == pytest.approx(expected)


def test_previous_dialogue_run_is_read_only_compatible(tmp_path):
    from test_web import _run
    from verge.web import load_run
    directory, _ = _run(tmp_path)
    path = directory / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['prompt_versions'] = dict.fromkeys(manifest['prompt_versions'], 'generator-dialogue-critic-feedback')
    path.write_text(json.dumps(manifest))
    assert len(load_run(directory)['samples']) == 3
