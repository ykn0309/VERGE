from copy import deepcopy
import importlib.util
from pathlib import Path


def load_replay():
    spec = importlib.util.spec_from_file_location('utilization_replay', Path(__file__).parents[1] / 'scripts/replay_experience_utilization.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry_preserves_intervention_and_does_not_mutate_input(monkeypatch):
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    m = load_replay()
    payload = {'task': '任务', 'experiences': [{'experience_id': '1', 'trigger': '规范化条件', 'principle': '规范化方法'}]}
    original = deepcopy(payload)
    class Fake:
        calls = []
        def generate(self, role, data):
            self.calls.append(deepcopy(data))
            data['experiences'][0]['principle'] = '后端修改'
            if len(self.calls) == 1:
                return {'analysis': '', 'criteria': []}
            return {'analysis': '说明', 'criteria': [{'question': '检查一', 'weight': .5, 'source_experiences': ['1']}, {'question': '检查二', 'weight': .5, 'source_experiences': []}]}
    f = Fake()
    m.generate_validated(f, 'initial_generator', payload)
    assert len(f.calls) == 2
    assert f.calls[1]['experiences'] == original['experiences']
    assert 'validation_feedback' in f.calls[1]
    assert payload == original


def test_debiased_retry_feedback_reaches_messages(monkeypatch):
    import verge.llm as llm
    monkeypatch.setattr(llm, 'llm_messages', llm.llm_messages)
    m = load_replay()
    token = m.VARIANT.set('normalized_debiased')
    try:
        messages = m.messages('selector', {'task': '任务', 'max_selected': 1, 'candidates': [], 'validation_feedback': '请纠正编号'})
        assert '请纠正编号' in messages[-1]['content']
    finally:
        m.VARIANT.reset(token)
