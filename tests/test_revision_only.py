"""Independent experience switch; temporary Fake runs only, no real experiment."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from verge.cli import main
from verge.config import load_config
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience
from verge.modules import make_fake_modules
from verge.pipeline import PipelineRunner


def forbidden(*args, **kwargs):
    raise AssertionError('experience stage must not execute')


def guard_experience_calls(monkeypatch, modules, memory):
    for component, method in [('retriever', 'retrieve'), ('selector', 'select'),
                              ('reflector', 'reflect'), ('admission_checker', 'check'),
                              ('fusion_judge', 'find_match')]:
        monkeypatch.setattr(getattr(modules, component), method, forbidden)
    for method in ['all', 'apply_downstream_credits', 'fuse']:
        monkeypatch.setattr(memory, method, forbidden)
    monkeypatch.setattr('verge.pipeline.calculate_existing_experience_credit', forbidden)
    monkeypatch.setattr('verge.pipeline.calculate_fusion_credit', forbidden)


@pytest.mark.parametrize('scenario,state,attempts', [
    ('initial_success', 'initial_success', 0),
    ('attempt_1_success', 'revision_success_1', 1),
    ('attempt_2_success', 'revision_success_2', 2),
    ('final_failure', 'final_failure', 2),
])
def test_disabled_experience_keeps_revision_and_ignores_seeded_read_only_pool(
        tmp_path, monkeypatch, scenario, state, attempts):
    from test_pipeline import sample
    memory = ExperienceMemory(tmp_path)
    memory.fuse(CandidateExperience('seed', '已有设计经验', '已有适用条件', admission='PASS'), 'origin')
    before = memory.path.read_bytes()
    memory = ExperienceMemory(tmp_path, read_only=True)
    modules = make_fake_modules(scenario)
    guard_experience_calls(monkeypatch, modules, memory)
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, memory, log, experience_enabled=False)
    record = runner.run_sample(sample())
    assert record.state == state and len(record.revision_attempts) == attempts
    assert record.retrieved_experience_ids == record.selected_experience_ids == []
    assert record.reflected_candidates == [] and record.operation_contributions == {}
    assert all(c.source_experiences == [] for c in record.initial_rubric.criteria)
    assert memory.path.read_bytes() == before
    assert log.get(record.sample_id) == record
    assert {role for role, _ in modules.backend.requests} <= {'initial_generator', 'judge', 'critic'}
    calls = [p for role, p in modules.backend.requests if role == 'initial_generator']
    assert calls[0]['selected_experiences'] == [] and len(calls) == attempts + 1
    if attempts == 2:
        first, second = record.revision_attempts
        assert second.previous_rubric == first.rubric
        assert second.rubric.base_rubric_id == first.rubric.rubric_id


def test_disabled_mode_preserves_invalid_round_and_latest_valid_baseline(tmp_path, monkeypatch):
    from test_pipeline import sample
    modules = make_fake_modules('attempt_2_success')
    memory = ExperienceMemory(tmp_path)
    guard_experience_calls(monkeypatch, modules, memory)
    original = modules.backend.generate
    def invalid_first(role, payload):
        output = original(role, payload)
        if role == 'initial_generator' and payload.get('revision_number') == 1:
            output['rubric']['criteria'] = output['rubric']['criteria'][:1]
        return output
    modules.backend.generate = invalid_first
    record = PipelineRunner(modules, memory, TrajectoryLog(tmp_path),
                            experience_enabled=False).run_sample(sample())
    assert record.state == 'revision_success_2'
    assert len(record.invalid_revision_attempts) == 1
    assert len(record.invalid_revision_attempts[0].validation_failures) == 3
    assert record.revision_attempts[0].rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert record.reflected_candidates == []


def test_disabled_mode_does_not_accept_custom_initial_provenance(tmp_path):
    from test_pipeline import sample
    modules = make_fake_modules('initial_success')
    original = modules.initial_generator.generate
    def stale(task, selected):
        rubric = original(task, selected)
        return replace(rubric, criteria=[replace(c, source_experiences=['stale']) for c in rubric.criteria])
    modules.initial_generator.generate = stale
    record = PipelineRunner(modules, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path),
                            experience_enabled=False).run_sample(sample())
    assert record.state == 'sample_error' and record.sample_error['stage'] == 'initial_generator'
    assert 'empty sources' in record.sample_error['message']


@pytest.mark.parametrize('name,bad', [('experience_enabled', 0), ('experience_enabled', 'false'),
                                    ('experience_enabled', None), ('experience_enabled', []),
                                    ('stop_after_train', 0), ('stop_after_train', -1),
                                    ('stop_after_train', True), ('stop_after_train', None),
                                    ('stop_after_train', '100')])
def test_mode_and_prefix_config_types_are_strict(tmp_path, name, bad):
    from test_cli import experiment
    path, config = experiment(tmp_path, with_test=False)
    config['run'][name] = bad
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='run.' + name):
        load_config(path)
    assert not Path(config['run']['output_dir']).exists()


@pytest.mark.parametrize('bad', [None, 0, 1, 'false'])
def test_runner_requires_boolean_switch(tmp_path, bad):
    with pytest.raises(ValueError, match='experience_enabled must be Boolean'):
        PipelineRunner(make_fake_modules(), ExperienceMemory(tmp_path), TrajectoryLog(tmp_path),
                       experience_enabled=bad)


def test_cli_fake_100_prefix_runs_only_required_roles_and_saves_mode(tmp_path, capsys):
    from test_cli import experiment
    path, config = experiment(tmp_path, with_test=False, scenario='attempt_2_success')
    config['run'].update(experience_enabled=False, stop_after_train=100)
    train = Path(config['dataset']['config']['paths']['train'])
    train.write_text(''.join(json.dumps({'sample_id':f'item-{i}', 'task':f'合成任务{i}',
                                        'chosen':'好', 'rejected':'差'}, ensure_ascii=False)+'\n'
                             for i in range(101)))
    path.write_text(yaml.safe_dump(config))
    assert main(['run', '--config', str(path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['train_samples'] == 100 and summary['test_samples'] == 0
    assert summary['status'] == 'paused_after_train_prefix'
    root = Path(summary['run_dir'])
    manifest = json.loads((root/'run_manifest.json').read_text())
    assert manifest['execution']['experience_enabled'] is False
    assert manifest['execution']['stop_after_train'] == 100
    records = list(TrajectoryLog(root/'artifacts').all())
    assert len(records) == 100 and all(r.state == 'revision_success_2' for r in records)
    assert all(not r.selected_experience_ids and not r.reflected_candidates for r in records)
    assert ExperienceMemory(root/'artifacts', read_only=True).all() == []
    events = [json.loads(l) for l in (root/'artifacts/role_io.jsonl').read_text().splitlines()]
    assert {e['role'] for e in events} == {'initial_generator', 'judge', 'critic'}
    assert {e['sample_id'] for e in events} == {r.sample_id for r in records}


@pytest.mark.parametrize('config_limit,cli_limit,count', [(1, 2, 2), (2, 1, 1)])
def test_cli_prefix_overrides_config_limit(tmp_path, capsys, config_limit, cli_limit, count):
    from test_cli import experiment
    path, config = experiment(tmp_path, with_test=False)
    config['run'].update(experience_enabled=False, stop_after_train=config_limit)
    path.write_text(yaml.safe_dump(config))
    assert main(['run', '--config', str(path), '--stop-after-train', str(cli_limit)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['train_samples'] == count
    manifest = json.loads((Path(result['run_dir'])/'run_manifest.json').read_text())
    assert manifest['execution']['stop_after_train'] == cli_limit


@pytest.mark.parametrize('workers', [1, 2])
def test_disabled_mode_propagates_to_sequential_and_parallel_test(tmp_path, monkeypatch, capsys, workers):
    from test_cli import experiment, _test_rows
    from verge.modules import FakeRetriever, LLMSelector, LLMReflector, LLMAdmissionChecker, LLMFusionJudge
    path, config = experiment(tmp_path)
    config['run'].update(experience_enabled=False, test_workers=workers)
    _test_rows(tmp_path, config, count=3)
    path.write_text(yaml.safe_dump(config))
    for cls, method in [(FakeRetriever, 'retrieve'), (LLMSelector, 'select'), (LLMReflector, 'reflect'),
                        (LLMAdmissionChecker, 'check'), (LLMFusionJudge, 'find_match')]:
        monkeypatch.setattr(cls, method, forbidden)
    assert main(['run', '--config', str(path)]) == 0
    capsys.readouterr()
    root = Path(config['run']['output_dir'])/'artifacts'
    rows = list(TrajectoryLog(root).all())
    assert [r.state for r in rows] == ['revision_success_1']*2 + ['initial_failure']*3
    assert all(not r.retrieved_experience_ids and not r.selected_experience_ids and
               not r.reflected_candidates for r in rows)
    assert ExperienceMemory(root, read_only=True).all() == []


def test_disabled_vector_mode_never_constructs_embedding_client(tmp_path, monkeypatch):
    from verge.cli import _components
    from verge.modules import FakeRetriever
    config = load_config('configs/rewardbench-chat-hard-qwen-vector-program-owned-revision-sources.yaml')
    config['run'].update(experience_enabled=False, output_dir=str(tmp_path/'fresh'))
    monkeypatch.setenv('DASHSCOPE_API_KEY_2', 'fake-no-network')
    monkeypatch.setattr('verge.cli.OpenAICompatibleEmbedding', forbidden)
    _, modules, _, _, _ = _components(config)
    try:
        assert isinstance(modules.retriever, FakeRetriever)
        assert not Path(config['run']['output_dir']).exists()
    finally:
        modules.backend.close()


def test_shipped_100_config_is_disabled_and_uses_existing_data_without_test():
    config = load_config('configs/rewardbench-chat-hard-qwen-revision-only-100.yaml')
    assert config['run']['experience_enabled'] is False and config['run']['stop_after_train'] == 100
    assert config['modules']['revision']['config']['max_attempts'] == 2
    assert config['models']['default']['model'] == 'qwen3.5-flash'
    assert set(config['dataset']['config']['paths']) == {'train'}
    # The authorized run now exists; the contract is a distinct output path, not
    # the absence of local artifacts. Tests must never depend on/delete real runs.
    assert config['run']['output_dir'] == 'runs/rewardbench-chat-hard-qwen-revision-only-100'
    for other in ('task-grounded-experience', 'prospective-experience'):
        previous = load_config(f'configs/rewardbench-chat-hard-qwen-vector-{other}.yaml')
        assert config['run']['output_dir'] != previous['run']['output_dir']


def test_resume_preserves_disabled_mode_and_rejects_switching_it(tmp_path, capsys):
    import hashlib
    from test_cli import experiment
    path, config = experiment(tmp_path, with_test=False)
    config['run'].update(experience_enabled=False, stop_after_train=1)
    path.write_text(yaml.safe_dump(config))
    assert main(['run', '--config', str(path)]) == 0
    capsys.readouterr()
    parent = Path(config['run']['output_dir'])
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
    config['run'].update(output_dir=str(tmp_path/'child'), experience_enabled=True)
    path.write_text(yaml.safe_dump(config))
    assert main(['run', '--config', str(path), '--resume-from', str(parent)]) == 2
    assert 'config or dataset changed' in capsys.readouterr().err
    assert not Path(config['run']['output_dir']).exists()
    config['run']['experience_enabled'] = False
    path.write_text(yaml.safe_dump(config))
    assert main(['run', '--config', str(path), '--resume-from', str(parent),
                 '--stop-after-train', '2']) == 0
    assert json.loads(capsys.readouterr().out)['train_samples'] == 2
    child = Path(config['run']['output_dir'])
    rows = list(TrajectoryLog(child/'artifacts').all())
    assert len(rows) == 2 and all(r.state == 'revision_success_1' for r in rows)
    assert ExperienceMemory(child/'artifacts', read_only=True).all() == []
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
