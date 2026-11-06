import importlib.util
from pathlib import Path
import pytest


def test_slot_contract_rejects_unknown_requirements_and_retains_method(monkeypatch):
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    payload = {'task': '解释术语', 'experiences': [{'experience_id': '1'}], 'requirements': [{'id': '1'}]}
    result = {'analysis': '', 'criteria': [{'question': '定义准确吗', 'weight': .5, 'source_experiences': ['1'], 'requirement_ids': ['1']}, {'question': '回应任务吗', 'weight': .5, 'source_experiences': [], 'requirement_ids': ['1']}]}
    m.valid(result, payload, 'slots')
    result['criteria'][0]['requirement_ids'] = ['2']
    with pytest.raises(AssertionError):
        m.valid(result, payload, 'slots')
    m.valid({'requirements': [{'id': '1', 'requirement': '先质疑名称', 'basis': 'explicit', 'task_quote': '名称错误'}]}, {'task': '解释术语'}, 'contract')
    m.valid({'requirements': [], 'uncertainties': ['目标不明确']}, {'task': '一句背景'}, 'contract')


@pytest.mark.parametrize("arm,thinking", [("plus_fixed", False), ("plus_reasoned_fixed", True)])
def test_plus_arm_uses_pinned_backend_and_fixed_draft(monkeypatch, tmp_path, arm, thinking):
    import json
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95_dispatch', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT = tmp_path
    m.FIXTURES = tmp_path / 'inputs.json'
    m.FIXTURES.write_text(json.dumps([{'sample_id': 't', 'split': 'development', 'role': 'initial_generator', 'group': 'normal', 'input': {'task': '解释术语', 'experiences': []}}]))
    (tmp_path / 'round6').mkdir()
    draft = {'criteria': [{'question': '定义准确吗', 'weight': .5, 'source_experiences': []}, {'question': '回应主题吗', 'weight': .5, 'source_experiences': []}]}
    (tmp_path / 'round6/intermediates.jsonl').write_text(json.dumps({'sample_id': 't', 'repetition': 0, 'output': draft}) + '\n')
    monkeypatch.setattr(m, 'load_config', lambda _: {'models': {'default': {'model': 'qwen3.5-flash'}}})
    seen = []
    class Fake:
        def __init__(self, config):
            seen.append(config)
        def generate(self, role, payload):
            assert 'experiences' not in payload
            assert payload['draft_criteria'] == draft['criteria']
            return {'analysis': '说明', **draft}
        def close(self):
            pass
    monkeypatch.setattr(m, 'backend', Fake)
    m.run('test', [arm])
    assert len(seen) == 1
    assert seen[0]['model'] == 'qwen3.5-plus-2026-04-20'
    assert seen[0]['enable_thinking'] is thinking


def test_final_repair_allows_one_meaningful_check_but_rejects_schema_text(monkeypatch):
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95_checks', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    payload = {'task': '现在是哪年', 'draft_criteria': []}
    result = {'analysis': '核验实际年份', 'criteria': [{'question': '回答的年份是否正确？', 'weight': 1, 'source_experiences': []}]}
    m.valid(result, payload, 'repair_plus')
    result['criteria'][0]['question'] = 'true表示满足的标准'
    with pytest.raises(ValueError, match='placeholder'):
        m.valid(result, payload, 'repair_plus')
    with pytest.raises(ValueError, match='invalid experiment'):
        m.run('must_not_run', ['misspelled_arm'])


def test_numeric_task_constraints_survive_counterexample_pruning(monkeypatch):
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95_guard', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    original = {'analysis': '', 'criteria': [{'question': '是否写明1080x1080和3种变体？', 'weight': .5, 'source_experiences': ['1']}, {'question': '是否必须先声明名称错误？', 'weight': .5, 'source_experiences': []}]}
    audit = {'checks': [{'index': 0, 'decision': 'drop', 'question': ''}, {'index': 1, 'decision': 'weaken', 'question': '是否合理回应含义问题？'}]}
    out, protected = m.apply_counterexamples(original, audit, '作品1080x1080，有3种变体', protect_numbers=True)
    assert out['criteria'][0] == original['criteria'][0]
    assert out['criteria'][1]['question'] == '是否合理回应含义问题？'
    assert len(protected) == 1

@pytest.mark.parametrize('arm', ['guarded_full', 'direct_plus', 'taskonly_plus'])
def test_fresh_arms_do_not_read_fixed_development_drafts(monkeypatch, tmp_path, arm):
    import json
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95_fresh', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.chdir(tmp_path)
    m.ROOT = tmp_path / 'output'
    m.FIXTURES = tmp_path / 'inputs.json'
    m.FIXTURES.write_text(json.dumps([{'sample_id': 'new', 'split': 'development', 'role': 'initial_generator', 'group': 'normal', 'input': {'task': '解释术语', 'experiences': [{'experience_id': '1', 'trigger': '原始适用条件', 'principle': '原始方法'}]}, 'id_mapping': {'reverse': {'experience': {'1': 'old'}}}}]))
    n = tmp_path / 'artifacts/experience-utilization-audit/normalized_methods.json'
    n.parent.mkdir(parents=True)
    n.write_text(json.dumps([{'experience_id': 'old', 'method': {'trigger': '新适用条件', 'principle': '新方法'}}]))
    monkeypatch.setattr(m, 'load_config', lambda _: {'models': {'default': {'model': 'qwen3.5-flash'}}})
    seen = []
    criteria = [{'question': '定义是否实质准确？', 'weight': .5, 'source_experiences': ['1']}, {'question': '说明是否回应问题？', 'weight': .5, 'source_experiences': []}]
    class Fake:
        def __init__(self, config):
            self.config = config
        def generate(self, role, payload):
            stage = m.STAGE.get()
            seen.append(stage)
            if stage == 'contract':
                return {'requirements': [{'id': '1', 'requirement': '解释术语', 'basis': 'explicit', 'task_quote': '解释术语'}]}
            if stage == 'counterexample':
                return {'checks': [{'index': i, 'decision': 'keep', 'question': c['question']} for i, c in enumerate(criteria)]}
            if stage == 'slots':
                assert payload['experiences'][0]['principle'] == '新方法'
                return {'analysis': '说明', 'criteria': [{**c, 'requirement_ids': ['1']} for c in criteria]}
            if stage == 'repair_plus_reasoned':
                assert 'experiences' not in payload
            if stage == 'direct_plus':
                assert self.config['enable_thinking'] is True
                assert self.config['model'] == 'qwen3.5-plus-2026-04-20'
                assert len(payload['experiences']) == (0 if arm == 'taskonly_plus' else 1)
            out = [{**c, 'source_experiences': []} for c in criteria] if arm == 'taskonly_plus' else criteria
            return {'analysis': '说明', 'criteria': out}
        def close(self):
            pass
    monkeypatch.setattr(m, 'backend', Fake)
    m.run('test', [arm])
    rows = [json.loads(l) for l in (m.ROOT / 'test/results.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['valid'], rows
    assert seen == (['contract', 'slots', 'repair_plus_reasoned', 'counterexample'] if arm == 'guarded_full' else ['direct_plus'])


def test_rebuild_is_task_blind_and_generator_keeps_selected_ids(monkeypatch, tmp_path):
    import json
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95_rebuild', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT = tmp_path / 'output'
    m.FIXTURES = tmp_path / 'inputs.json'
    m.FIXTURES.write_text(json.dumps([{'sample_id': 't', 'split': 'development', 'role': 'initial_generator', 'group': 'normal', 'input': {'task': '不准给编辑器看到的题目', 'experiences': [{'experience_id': '7', 'trigger': '旧条件', 'principle': '旧方法'}]}}]))
    monkeypatch.setattr(m, 'load_config', lambda _: {'models': {'default': {'model': 'qwen3.5-flash', 'enable_thinking': False}}})
    seen = []
    class Fake:
        def __init__(self, config):
            self.config = config
        def generate(self, role, payload):
            stage = m.STAGE.get()
            seen.append(stage)
            if stage == 'rebuild':
                assert set(payload) == {'trigger', 'principle'}
                assert self.config['model'] == 'qwen3.5-plus-2026-04-20'
                assert self.config['enable_thinking'] is True
                return {'trigger': '新条件', 'principle': '新方法'}
            assert self.config['model'] == 'qwen3.5-flash'
            assert payload['experiences'] == [{'experience_id': '7', 'trigger': '新条件', 'principle': '新方法'}]
            return {'analysis': '', 'criteria': [{'question': '事实是否准确？', 'weight': .5, 'source_experiences': ['7']}, {'question': '回答是否切题？', 'weight': .5, 'source_experiences': []}]}
        def close(self):
            pass
    monkeypatch.setattr(m, 'backend', Fake)
    m.run('test', ['rebuilt'], repeats=2)
    assert seen.count('rebuild') == 1
    assert seen.count('baseline') == 2


@pytest.mark.parametrize('arm', ['gated', 'necessity_gated'])
def test_gate_only_filters_given_methods_before_original_generator(monkeypatch, tmp_path, arm):
    import json
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    spec = importlib.util.spec_from_file_location('research95_gate', Path(__file__).parents[1] / 'scripts/research_premise95.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT = tmp_path / 'output'
    m.FIXTURES = tmp_path / 'inputs.json'
    original = [{'experience_id': '1', 'trigger': 'a', 'principle': '不成立前提'}, {'experience_id': '2', 'trigger': 'b', 'principle': '有用核验'}]
    m.FIXTURES.write_text(json.dumps([{'sample_id': 't', 'split': 'development', 'role': 'initial_generator', 'group': 'normal', 'input': {'task': 'task', 'experiences': original}}]))
    monkeypatch.setattr(m, 'load_config', lambda _: {'models': {'default': {'model': 'qwen3.5-flash', 'enable_thinking': False}}})
    calls = []
    class Fake:
        def __init__(self, config):
            self.config = config
        def generate(self, role, payload):
            calls.append(m.STAGE.get())
            if m.STAGE.get() in ('gate', 'necessity_gate'):
                assert payload['experiences'] == original
                assert self.config['model'] == 'qwen3.5-plus-2026-04-20'
                assert self.config['enable_thinking'] is (arm == 'necessity_gated')
                if arm == 'necessity_gated':
                    return {'decisions': [{'experience_id': '1', 'premises_supported': True, 'counterexample_found': True, 'counterexample': '合理但违反原限制的回答', 'reason': '不必要'}, {'experience_id': '2', 'premises_supported': True, 'counterexample_found': False, 'counterexample': '无法构造', 'reason': '明确要求'}]}
                return {'decisions': [{'experience_id': '1', 'use': False, 'reason': '前提未知'}, {'experience_id': '2', 'use': True, 'reason': '明确要求'}]}
            assert role == 'initial_generator'
            assert self.config['model'] == 'qwen3.5-flash'
            assert payload['experiences'] == [original[1]]
            return {'analysis': '', 'criteria': [{'question': '内容是否正确？', 'weight': .5, 'source_experiences': ['2']}, {'question': '回答是否切题？', 'weight': .5, 'source_experiences': []}]}
        def close(self):
            pass
    monkeypatch.setattr(m, 'backend', Fake)
    m.run('test', [arm])
    assert calls == ['necessity_gate' if arm == 'necessity_gated' else 'gate', 'baseline']
    row = json.loads((m.ROOT / 'test/results.jsonl').read_text())
    assert row['valid']
    bad = {'decisions': [{'experience_id': '1', 'use': 'false', 'reason': 'bad'}, {'experience_id': '2', 'use': True, 'reason': 'good'}]}
    with pytest.raises(AssertionError):
        m.valid(bad, {'experiences': original}, 'gate')
