from threading import Barrier
from copy import deepcopy
from verge.pipeline import run_training_batch
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import make_fake_modules
from verge.models import ExperienceReview
from test_v2 import sample, experience, params


def test_batch_admissions_invisible_until_next_batch_and_ordered(tmp_path):
    memory = ExperienceMemory(tmp_path, parameters=params())
    log = TrajectoryLog(tmp_path)
    barrier = Barrier(3)
    seen = {}
    def factory(s):
        m = make_fake_modules()
        original = m.retriever.retrieve
        def retrieve(task, pool):
            seen[task] = [e.experience_id for e in pool]
            barrier.wait(timeout=10)
            return original(task, pool)
        m.retriever.retrieve = retrieve
        return m, lambda: None
    records = run_training_batch([sample(i) for i in range(3)], factory, memory, log,
                                runner_options={'revision_enabled': False})
    assert all(not ids for ids in seen.values())
    assert len(memory.all()) == 3
    assert [r.sample_id for r in log.all()] == ['s0','s1','s2']
    assert all(len(r.admitted_experience_ids) == 1 for r in records)
    assert all(e.q_posterior is not None for e in memory.all())
    # A subsequent batch now sees all previous admissions.
    observed = []
    def second(s):
        m = make_fake_modules('initial_success')
        original = m.retriever.retrieve
        def retrieve(task,pool):
            observed.append(len(pool));return original(task,pool)
        m.retriever.retrieve = retrieve
        return m, lambda: None
    run_training_batch([sample(4)],second,memory,log,runner_options={'revision_enabled':False})
    assert observed == [3]


def test_shared_experience_feedback_not_overwritten_and_q_recomputed(tmp_path):
    p=params(); memory=ExperienceMemory(tmp_path,parameters=p); log=TrajectoryLog(tmp_path)
    e=experience(); source=sample(99); e.source_sample_id=source.sample_id
    memory.add(e,source,ExperienceReview(True,'合理'))
    def factory(s):return make_fake_modules('initial_success'),lambda:None
    records=run_training_batch([sample(i) for i in range(3)],factory,memory,log,
                              runner_options={'revision_enabled':False})
    current=memory.get(e.experience_id)
    assert len(current.applications)==3
    expected=deepcopy(current)
    from verge.calculator import rebuild_quality
    rebuild_quality(expected,p)
    assert abs(current.q_posterior-expected.q_posterior)<1e-12
    assert [a.sample.sample_id for a in current.applications]==['s0','s1','s2']
    assert records[1].applications[0].q_before==records[0].applications[0].q_after


def test_cli_batch_partial_tail_and_config_validation(tmp_path):
    import json, yaml, pytest
    from pathlib import Path
    from verge.cli import run
    from verge.config import load_config
    data=tmp_path/'train.jsonl'
    data.write_text(''.join(sample(i).to_json()+'\n' for i in range(5)))
    c=yaml.safe_load(Path('configs/default.yaml').read_text())
    c['run'].update(output_dir=str(tmp_path/'run'),train_workers=3)
    c['dataset']['config']['paths']={'train':str(data)}
    c['modules']['application']['config']['revision_enabled']=False
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(c))
    result=run(p)
    assert result['train']==5 and result['active_experiences']==5
    log=TrajectoryLog(tmp_path/'run'/'artifacts')
    rows=list(log.all())
    assert [r.sample_id for r in rows]==['s0','s1','s2','s3','s4']
    assert all(not r.retrieved_experience_ids for r in rows[:3])
    assert all(len(r.retrieved_experience_ids)==3 for r in rows[3:])
    for value in (0,-1,True,1.5):
        c['run']['train_workers']=value;p.write_text(yaml.safe_dump(c))
        with pytest.raises(ValueError,match='train_workers'):load_config(p)
