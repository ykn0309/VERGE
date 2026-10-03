"""Per-sample contract failures continue without invented results or duplicate memory updates."""
from dataclasses import replace

import pytest

from verge.llm import LLMFormatError, LLMRefusalError, LLMRequestError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import PreferenceSample, TrajectoryRecord
from verge.modules import LLMInitialRubricGenerator, make_fake_modules
from verge.pipeline import PipelineRunner


def sample(key):
    return PreferenceSample(key, '任务要求', '好', '坏', {})


def runner_at(path, scenario='attempt_1_success'):
    modules = make_fake_modules(scenario)
    memory = ExperienceMemory(path)
    log = TrajectoryLog(path)
    return PipelineRunner(modules, memory, log), modules, memory, log


@pytest.mark.parametrize('role,method,stage', [
    ('initial_generator', 'generate', 'initial_generator'),
    ('reflector', 'reflect', 'reflection'),
    ('admission_checker', 'check', 'admission'),
    ('fusion_judge', 'find_match', 'fusion_judge'),
])
def test_role_contract_failure_continues_and_preserves_completed_stages(tmp_path, role, method, stage):
    runner, modules, memory, log = runner_at(tmp_path)
    module = getattr(modules, role)
    original = getattr(module, method)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('bad structured role output')
        return original(*args, **kwargs)

    setattr(module, method, fail_once)
    first, second = runner.run([sample('a'), sample('b')])
    assert first.state == 'sample_error' and first.sample_error['stage'] == stage
    assert second.state == 'revision_success_1'
    assert list(log.all()) == [first, second]
    assert TrajectoryRecord.from_dict(first.to_dict()) == first
    if role == 'initial_generator':
        assert first.initial_rubric is None and first.initial_judgment is None
        assert first.revision_attempts == []
    else:
        assert first.initial_judgment is not None
        assert first.revision_attempts[0].judgment.success
    if role in ('admission_checker', 'fusion_judge'):
        assert first.reflected_candidates
    if role == 'fusion_judge':
        assert first.reflected_candidates[0].admission == 'PASS'
        assert first.reflected_candidates[0].fusion is None
    assert len(memory.all()) == 1


@pytest.mark.parametrize('exception', [LLMFormatError('invalid JSON after 3 attempts'),
                                       LLMRefusalError('model refused the request')])
def test_format_exhaustion_and_refusal_are_sample_errors(tmp_path, exception):
    runner, modules, _, _ = runner_at(tmp_path)
    modules.initial_generator.generate = lambda *_: (_ for _ in ()).throw(exception)
    records = runner.run([sample('a'), sample('b')])
    assert len(records) == 2
    assert all(r.state == 'sample_error' and r.initial_judgment is None for r in records)


def test_generator_default_semantic_retries_are_bounded(tmp_path):
    runner, modules, _, _ = runner_at(tmp_path, 'initial_success')
    backend = modules.backend
    original = backend.generate
    count = 0

    def invalid(role, payload):
        nonlocal count
        value = original(role, payload)
        if role == 'initial_generator':
            count += 1
            value['rubric']['criteria'] = value['rubric']['criteria'][:1]
        return value

    backend.generate = invalid
    record = runner.run_sample(sample('a'))
    assert count == 3
    assert record.state == 'sample_error' and record.initial_rubric is None


def test_format_retries_not_multiplied_by_generator_semantic_retries():
    class Backend:
        count = 0
        def generate(self, *args):
            self.count += 1
            raise LLMFormatError('backend format retries exhausted')
    backend = Backend()
    with pytest.raises(LLMFormatError):
        LLMInitialRubricGenerator(backend).generate('任务', [])
    assert backend.count == 1


def test_partial_fusion_is_recorded_and_not_rolled_back(tmp_path):
    runner, modules, memory, log = runner_at(tmp_path)
    reflect = modules.reflector.reflect
    modules.reflector.reflect = lambda *args, **kwargs: [*reflect(*args, **kwargs),
        replace(reflect(*args, **kwargs)[0], candidate_id='second', content='另一原则')]
    check = modules.admission_checker.check
    calls = 0
    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError('second candidate invalid')
        return check(*args)
    modules.admission_checker.check = fail_second
    record = runner.run_sample(sample('a'))
    assert record.state == 'sample_error' and record.sample_error['stage'] == 'admission'
    assert record.reflected_candidates[0].fusion == 'ADD'
    assert record.reflected_candidates[1].admission is None
    assert len(memory.all()) == 1 and memory.all()[0].source_trajectories == ['a']
    assert len(list(log.all())) == 1


@pytest.mark.parametrize('exc', [LLMRequestError('HTTP 401'), OSError('disk full'),
                                 RuntimeError('implementation bug'), TypeError('bad code')])
def test_global_service_io_and_programming_errors_still_abort(tmp_path, exc):
    runner, modules, _, log = runner_at(tmp_path)
    modules.initial_generator.generate = lambda *_: (_ for _ in ()).throw(exc)
    with pytest.raises(type(exc)):
        runner.run_sample(sample('a'))
    assert list(log.all()) == []


def test_storage_value_error_is_not_misclassified_as_role_error(tmp_path):
    runner, _, memory, log = runner_at(tmp_path)
    memory.fuse = lambda *_, **kwargs: (_ for _ in ()).throw(ValueError('invalid memory'))
    with pytest.raises(ValueError, match='invalid memory'):
        runner.run_sample(sample('a'))
    assert list(log.all()) == []


def test_downstream_updates_survive_later_error_exactly_once(tmp_path):
    runner, modules, memory, log = runner_at(tmp_path)
    runner.run_sample(sample('a'))
    original = modules.initial_generator.generate
    def with_source(*args):
        rubric = original(*args)
        for criterion in rubric.criteria:
            criterion.source_experiences = [memory.all()[0].experience_id]
        return rubric
    modules.initial_generator.generate = with_source
    modules.reflector.reflect = lambda *_, **kwargs: (_ for _ in ()).throw(ValueError('reflection invalid'))
    record = runner.run_sample(sample('b'))
    assert record.state == 'sample_error' and record.initial_judgment is not None
    events = [ev for exp in memory.all() for ev in exp.evidence_history
              if ev.sample_id == 'b' and ev.evidence_type == 'downstream_credit']
    assert len(events) == 1
    assert log.get('b') == record


def test_reflection_binding_receive_original_pair_without_leaking_to_early_roles(tmp_path):
    runner, modules, memory, log = runner_at(tmp_path)
    original = sample('context')
    record = runner.run_sample(original)
    assert record.state == 'revision_success_1'
    seen = set()
    for role, payload in modules.backend.requests:
        if role == 'reflector':
            seen.add(role)
            assert payload['task'] == original.task
            assert payload['chosen'] == original.chosen
            assert payload['rejected'] == original.rejected
            assert 'operation_contributions' not in payload and 'budget' not in payload
        if role in ('selector', 'initial_generator', 'admission_checker', 'fusion_judge'):
            assert 'chosen' not in payload and 'rejected' not in payload
    assert seen == {'reflector'}
    assert all(role != 'revision_credit_assigner' for role, _ in modules.backend.requests)
