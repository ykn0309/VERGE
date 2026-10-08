"""Revision provenance is program-owned; HTTP mocks and fixed-output replay only."""

from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from verge.llm import LLMFormatError, OpenAICompatibleBackend
from verge.modules import LLMInitialRubricGenerator, make_fake_modules
from verge.prompts import ExperienceView


VIEW = ExperienceView('real-exp', '标准设计指导', '事前适用条件', 0.5, 0.0)
REPLAY = json.loads((Path(__file__).parent / 'fixtures/revision_source_replies.json').read_text())


def reply(sources):
    return {'analysis': '依据任务设计标准', 'criteria': [
        {'question': '是否满足核心要求？', 'importance_score': 8, 'source_experiences': sources},
        {'question': '表达是否清晰？', 'importance_score': 3, 'source_experiences': []},
    ]}


def http_generator(monkeypatch, replies, bodies, traces):
    monkeypatch.setenv('REVISION_SOURCE_KEY', 'mock-key')
    def handler(request):
        bodies.append(json.loads(request.content))
        raw = replies[len(bodies) - 1]
        content = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        return httpx.Response(200, json={'choices': [
            {'message': {'content': content}, 'finish_reason': 'stop'}]})
    client = httpx.Client(base_url='https://api.openai.com/v1/', transport=httpx.MockTransport(handler))
    backend = OpenAICompatibleBackend('mock', api_key_env='REVISION_SOURCE_KEY', client=client,
                                      max_retries=0, trace_hook=traces.append, sleep=lambda _: None)
    return LLMInitialRubricGenerator(backend, invalid_output_retries=0), client


@pytest.mark.parametrize('sources', [[1], ['1'], ['999'], ['1', '999', '1'],
                                    ['not-a-number'], []])
def test_http_revision_clears_sources_before_decode_without_mutating_trace(monkeypatch, sources):
    raw0 = json.dumps(reply(['1']), ensure_ascii=False)
    raw1 = json.dumps(reply(sources), ensure_ascii=False)
    bodies, traces = [], []
    generator, client = http_generator(monkeypatch, [raw0, raw1, raw1], bodies, traces)
    with client:
        r0 = generator.generate('任务', [VIEW])
        r1 = generator.revise(r0, '改善核心标准', 1)
        r2 = generator.revise(r1, '完善表达标准', 2)
    assert len(bodies) == 3  # no source repair/retry calls
    assert r0.criteria[0].source_experiences == ['real-exp']
    for revised in [r1, r2]:
        assert all(c.source_experiences == [] for c in revised.criteria)
        assert [c.question for c in revised.criteria] == [c['question'] for c in reply(sources)['criteria']]
        assert [c.importance_score for c in revised.criteria] == [8, 3]
    assert r1.base_rubric_id == r0.rubric_id and r2.base_rubric_id == r1.rubric_id
    assert traces[1]['raw_output'] == traces[2]['raw_output'] == raw1
    normalized = [str(x) if type(x) is int else x for x in sources]
    assert traces[1]['output']['criteria'][0]['source_experiences'] == normalized
    assert traces[2]['output']['criteria'][0]['source_experiences'] == normalized
    assert bodies[1]['messages'][2]['content'] == raw0
    assert bodies[2]['messages'][-2]['content'] == raw1
    assert generator._conversation.get()[-1]['content'] == raw1


@pytest.mark.parametrize('sources, error', [(['999'], 'unknown numbered experience'),
                                           (['1', '1'], 'distinct')])
def test_r0_reference_validation_remains_strict(monkeypatch, sources, error):
    bodies, traces = [], []
    generator, client = http_generator(monkeypatch, [reply(sources)], bodies, traces)
    with client, pytest.raises(ValueError, match=error):
        generator.generate('任务', [VIEW])
    assert len(bodies) == 1
    assert generator._conversation.get() is None
    assert traces[0]['output']['criteria'][0]['source_experiences'] == sources


@pytest.mark.parametrize('kind', ['missing_source', 'null_source', 'string_source', 'bad_source_item',
                                  'extra_field', 'bad_score', 'bad_question', 'empty_analysis',
                                  'blank_question', 'too_few'])
def test_revision_still_rejects_other_output_errors(monkeypatch, kind):
    bad = reply(['1'])
    item = bad['criteria'][0]
    if kind == 'missing_source':
        item.pop('source_experiences')
    elif kind == 'null_source':
        item['source_experiences'] = None
    elif kind == 'string_source':
        item['source_experiences'] = '1'
    elif kind == 'bad_source_item':
        item['source_experiences'] = [False]
    elif kind == 'extra_field':
        item['extra'] = True
    elif kind == 'bad_score':
        item['importance_score'] = 11
    elif kind == 'bad_question':
        item['question'] = None
    elif kind == 'empty_analysis':
        bad['analysis'] = ''
    elif kind == 'blank_question':
        item['question'] = ' '
    else:
        bad['criteria'] = bad['criteria'][:1]
    bodies, traces = [], []
    generator, client = http_generator(monkeypatch, [reply([]), bad], bodies, traces)
    with client:
        r0 = generator.generate('任务', [])
        with pytest.raises(ValueError) as failure:
            generator.revise(r0, '修订标准', 1)
    assert 'unknown numbered experience' not in str(failure.value)
    assert len(bodies) == 2
    if kind not in {'blank_question', 'too_few'}:
        assert isinstance(failure.value, LLMFormatError)


def test_fake_pipeline_clears_sources_and_keeps_backend_and_conversation_originals(tmp_path):
    from test_generator_dialogue import run
    modules = make_fake_modules('attempt_2_success')
    original = modules.backend.generate
    outputs = []
    def with_sources(role, payload):
        response = original(role, payload)
        if role == 'initial_generator' and payload.get('revision_number'):
            response['rubric']['criteria'][0]['source_experiences'] = ['old-exp', 'unknown-exp']
            outputs.append((response, deepcopy(response)))
        return response
    modules.backend.generate = with_sources
    record, _, _ = run(tmp_path, modules=modules)
    assert record.state == 'revision_success_2'
    assert record.invalid_revision_attempts == []
    first, second = record.revision_attempts
    assert second.rubric.base_rubric_id == first.rubric.rubric_id
    assert all(not a.validation_failures for a in record.revision_attempts)
    assert all(c.source_experiences == [] for a in record.revision_attempts for c in a.rubric.criteria)
    assert all(output == snapshot for output, snapshot in outputs)
    calls = [p for role, p in modules.backend.requests if role == 'initial_generator']
    assert len(calls) == 3
    raw_first = json.loads(calls[2]['_conversation'][-2]['content'])
    assert raw_first == outputs[0][1]
    assert raw_first['rubric']['criteria'][0]['source_experiences'] == ['old-exp', 'unknown-exp']


@pytest.mark.parametrize('sources', [None, 'old-exp', [None], [True], {'id': 'old-exp'}])
def test_fake_revision_does_not_hide_bad_source_types(sources):
    modules = make_fake_modules('attempt_1_success')
    original = modules.backend.generate
    def malformed(role, payload):
        response = original(role, payload)
        if role == 'initial_generator' and payload.get('revision_number'):
            response['rubric']['criteria'][0]['source_experiences'] = sources
        return response
    modules.backend.generate = malformed
    r0 = modules.initial_generator.generate('任务', [])
    with pytest.raises(ValueError, match='list of strings'):
        modules.initial_generator.revise(r0, '修订', 1)


@pytest.mark.parametrize('case', REPLAY['cases'], ids=lambda c: c['sample_id'])
def test_recorded_problem_replies_decode_without_retry_in_offline_replay(monkeypatch, case):
    bodies, traces = [], []
    generator, client = http_generator(monkeypatch, case['raw_replies'], bodies, traces)
    with client:
        r0 = generator.generate(case['task'], [VIEW])
        r1 = generator.revise(r0, '调整评价维度', 1)
        r2 = generator.revise(r1, '完善评价维度', 2)
    assert len(bodies) == 3
    assert all(c.source_experiences == [] for r in [r1, r2] for c in r.criteria)
    assert r2.base_rubric_id == r1.rubric_id
    for trace, raw in zip(traces, case['raw_replies']):
        assert trace['raw_output'] == raw
    for revised, raw in zip([r1, r2], case['raw_replies'][1:]):
        original = json.loads(raw)['criteria']
        assert any(c['source_experiences'] for c in original)
        assert [(c.question, c.importance_score) for c in revised.criteria] == [
            (c['question'], c['importance_score']) for c in original]
