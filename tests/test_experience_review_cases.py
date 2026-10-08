"""Human-reviewed development cases; mocks test transport, NOT semantic accuracy."""

import json
from pathlib import Path

import pytest

from verge.models import CandidateExperience
from verge.modules import LLMAdmissionChecker, LLMFusionJudge
from verge.prompts import PROMPT_VERSION, llm_messages


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/prospective_experience_cases.json').read_text())


def test_curated_cases_are_explicitly_development_only_and_cover_both_decisions():
    assert FIXTURE['not_held_out'] is True
    assert '不证明真实模型' in FIXTURE['purpose']
    for key in ('admission_cases', 'fusion_cases'):
        cases = FIXTURE[key]
        assert len({c['id'] for c in cases}) == len(cases)
        assert all(c['review_note'] for c in cases)
    assert {c['expected_decision'] for c in FIXTURE['admission_cases']} == {'PASS', 'FAIL'}
    assert {c['expected_match_index'] for c in FIXTURE['fusion_cases']} == {None, 0}
    observed = [c for c in FIXTURE['admission_cases'] if c['source_sample_id']]
    assert len(observed) == 8
    assert next(c for c in observed if c['id'] == 'observed-2857')['expected_decision'] == 'PASS'
    assert next(c for c in observed if c['id'] == 'observed-2501')['expected_decision'] == 'FAIL'


@pytest.mark.parametrize('case', FIXTURE['admission_cases'], ids=lambda c: c['id'])
def test_admission_case_projection_and_text_preservation_with_preset_decision(monkeypatch, case):
    from test_modules import initial_case, revise
    from test_real_llm import backend, result
    modules, initial, judgment = initial_case()
    attempt = revise(modules, initial, judgment, 1)
    requests = []
    # Gold labels are supplied by the mock. Passing does not evaluate model judgment.
    def handler(request):
        requests.append(json.loads(request.content))
        return result({'decision': case['expected_decision'], 'reason': case['review_note']})

    candidate = CandidateExperience('internal-candidate', case['content'], case['trigger'])
    checker = LLMAdmissionChecker(backend(monkeypatch, handler, max_retries=0))
    reviewed = checker.check(case['task'], candidate, attempt)
    assert reviewed.admission == case['expected_decision']
    assert reviewed.admission_reason == case['review_note']
    assert (reviewed.content, reviewed.trigger) == (candidate.content, candidate.trigger)
    assert candidate.admission is None and reviewed.credit is None and reviewed.fusion_record is None
    assert len(requests) == 1
    system, user = requests[0]['messages']
    assert system == llm_messages('admission_checker', {})[0]
    assert case['task'] in user['content']
    assert case['content'] in user['content'] and case['trigger'] in user['content']
    assert 'internal-candidate' not in user['content']
    for hidden in ('successful_attempt', 'fusion_credit', 'evidence_weight', 'chosen', 'rejected'):
        assert hidden not in user['content']


@pytest.mark.parametrize('case', FIXTURE['fusion_cases'], ids=lambda c: c['id'])
def test_fusion_case_contract_with_preset_add_or_merge(monkeypatch, case):
    from test_real_llm import backend, result
    requests = []
    index = case['expected_match_index']
    expected = ({'match_index': None, 'content': None, 'trigger': None} if index is None else
                {'match_index': index, **case['existing'][index]})

    def handler(request):
        requests.append(json.loads(request.content))
        return result(expected)

    judge = LLMFusionJudge(backend(monkeypatch, handler, max_retries=0))
    candidate = case['candidate']
    actual = judge.find_match(candidate['content'], candidate['trigger'],
                             [(e['content'], e['trigger']) for e in case['existing']])
    assert actual.to_dict() == expected
    assert len(requests) == 1
    system, user = requests[0]['messages']
    assert system == llm_messages('fusion_judge', {})[0]
    assert candidate['content'] in user['content'] and candidate['trigger'] in user['content']
    assert all(e['content'] in user['content'] and e['trigger'] in user['content'] for e in case['existing'])
    for hidden in ('confidence', 'credit', 'source_trajectories', 'review_note', 'expected_match_index'):
        assert hidden not in user['content']
    assert PROMPT_VERSION == 'generator-dialogue-program-owned-revision-sources'
