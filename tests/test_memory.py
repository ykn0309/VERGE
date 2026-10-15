"""Persistent bounded cases, separate credit, frozen text and new-only protocol."""
import json
import pytest
from experience_helpers import seed, case_for
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import CandidateExperience, Credit, TrajectoryRecord


def candidate(sid='candidate_1'):
    return CandidateExperience(sid, '独立关键要求标准', '*')


def test_text_updates_preserve_id_credit_and_versions(tmp_path):
    memory = ExperienceMemory(tmp_path)
    added = seed(memory, candidate(), 's1', p=.3, n=.2)
    eid = added.experience_id
    memory.apply_downstream_credits('s2', {eid: Credit(.3, .2)})
    e = memory.get(eid)
    case = case_for(eid, 's2', e.content, e.trigger, e.text_version, .3, .2)
    assert memory.collect_case(eid, case)
    for sid in ('s3', 's4'):
        memory.apply_downstream_credits(sid, {eid: Credit(.3, .2)})
        memory.collect_case(eid, case_for(eid, sid, e.content, e.trigger, 1, .3, .2))
    event = memory.update_text(eid, '跨案例共同方法', '必要条件')
    e = ExperienceMemory(tmp_path).get(eid)
    assert e.experience_id == eid and (e.alpha, e.beta) == pytest.approx((2.2, 1.8))
    assert e.text_version == 2 and e.update_history == [event]
    assert len(e.cases) == 4 and [ev.text_version for ev in e.evidence_history] == [1, 1, 1, 1]
    assert json.loads(memory.path.read_text())['schema_version'] == 6
    assert not hasattr(memory, 'fuse')


def test_ten_cases_freeze_text_and_dedup_unchanged_updates(tmp_path):
    m = ExperienceMemory(tmp_path)
    eid = seed(m, candidate(), 's0').experience_id
    for i in range(1, 10):
        e = m.get(eid)
        case = case_for(eid, f's{i}', e.content, e.trigger, e.text_version)
        assert m.collect_case(eid, case)
        assert not m.collect_case(eid, case)
        if m.get(eid).update_due:
            m.update_text(eid, e.content, e.trigger)  # unchanged still consumes a case
    e = m.get(eid)
    assert len(e.cases) == 10 and len(e.update_history) == 3
    assert e.text_frozen and e.text_version == 1
    assert not m.collect_case(eid, case_for(eid, 's10', e.content, e.trigger))
    with pytest.raises(ValueError, match='pending'):
        m.update_text(eid, '绕过冻结', '*')
    m.apply_downstream_credits('s10', {eid: Credit(.2, .1)})
    assert m.get(eid).alpha == pytest.approx(1.4)
    assert m.get(eid).content == e.content


def test_readonly_rejects_mutations(tmp_path):
    m = ExperienceMemory(tmp_path)
    c = candidate()
    eid = seed(m, c).experience_id
    before = m.path.read_bytes()
    frozen = ExperienceMemory(tmp_path, read_only=True)
    case = case_for(eid, 'next', c.content, c.trigger)
    for action in [lambda: frozen.apply_downstream_credits('s', {eid: Credit(.2, 0)}),
                   lambda: frozen.collect_case(eid, case),
                   lambda: seed(frozen, candidate('c2'), 's'),
                   lambda: frozen.update_text(eid, '方法', '条件')]:
        with pytest.raises(ValueError):
            action()
    assert m.path.read_bytes() == before


@pytest.mark.parametrize('version', [None, 1, 2, 3, 4])
def test_old_memory_rejected_and_trajectory_roundtrip(tmp_path, version):
    data = {'experiences': []}
    if version is not None:
        data['schema_version'] = version
    (tmp_path/'experience_memory.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    record = TrajectoryRecord('s', 'initial_failure')
    log.append(record)
    assert log.get('s') == record


def test_origin_case_and_duplicate_admission_are_validated_before_write(tmp_path):
    m = ExperienceMemory(tmp_path)
    added = seed(m, candidate(), 's1')
    before = m.path.read_bytes()
    with pytest.raises(ValueError, match='already'):
        m.add(added, m.get(added.experience_id).cases[0])
    with pytest.raises(ValueError, match='one candidate'):
        seed(m, candidate('c2'), 's1')
    assert m.path.read_bytes() == before
