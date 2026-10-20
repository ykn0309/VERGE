"""Exercise independent ledger/window reconstruction on deterministic pipeline fixtures."""
import json
from pathlib import Path
import pytest
from test_cli import experiment, rewrite
from verge.cli import run
from scripts.audit_versioned_run import audit


def fixture_run(tmp_path):
    p,c=experiment(tmp_path,with_test=False)
    source=Path(c['dataset']['config']['paths']['train'])
    source.write_text(''.join(json.dumps({'sample_id':f's{i}','task':f'任务包含明确关键要求。案例{i}',
                                        'chosen':'好','rejected':'差'},ensure_ascii=False)+'\n' for i in range(12)))
    rewrite(p,c);run(p)
    directory=Path(c['run']['output_dir'])
    # Only this isolated deterministic fixture uses a model label to exercise the audit's model guard.
    role_file=directory/'artifacts/role_io.jsonl'
    rows=[json.loads(x) for x in role_file.read_text().splitlines()]
    for row in rows:row['model']='qwen3.5-flash'
    role_file.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    return directory


def test_delayed_audit_reconstructs_credit_windows_versions_and_does_not_write(tmp_path):
    directory=fixture_run(tmp_path)
    result=audit(directory,12)
    assert result['samples']==12 and result['experiences']>0
    assert result['updates']>0 and result['active_experiences']>0
    assert all(n==5 for n in result['window_sizes'])
    assert result['final_successes']==12
    with pytest.raises(ValueError,match='sample count'):audit(directory,200)


def test_delayed_audit_rejects_corrupted_final_beta(tmp_path):
    directory=fixture_run(tmp_path)
    path=directory/'artifacts/experience_memory.json';raw=json.loads(path.read_text())
    raw['experiences'][0]['alpha']+=.1
    e=raw['experiences'][0];e['confidence']=e['alpha']/(e['alpha']+e['beta'])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Beta'):audit(directory,12)


def test_audit_reconstructs_multiple_generations_of_split_credit(tmp_path,monkeypatch):
    from verge.modules import LLMAtomizer,LLMExperienceUpdater,LLMSelector
    from verge.models import AtomicResult,AtomicMethod,ExperienceUpdate
    original=LLMAtomizer.atomize
    def atomize(self,draft,*,previous=None):
        if previous is None:return original(self,draft)
        return AtomicResult([AtomicMethod(draft.content+'甲',draft.trigger),
                             AtomicMethod(draft.content+'乙',draft.trigger)])
    monkeypatch.setattr(LLMAtomizer,'atomize',atomize)
    monkeypatch.setattr(LLMExperienceUpdater,'update',lambda self,e:ExperienceUpdate('revise',AtomicMethod(e.content+'修订',e.trigger)))
    select=LLMSelector.select
    # This deterministic fixture must exercise descendants, rather than repeatedly
    # assigning Fake's single cited criterion to the oldest unrelated candidate.
    monkeypatch.setattr(LLMSelector,'select',lambda self,task,candidates,**kw:
        select(self,task,sorted(candidates,key=lambda v:-v.content.count('修订')),**kw))
    directory=fixture_run(tmp_path)
    result=audit(directory,12)
    assert result['retired_experiences']>=2
    path=directory/'artifacts/experience_memory.json'
    raw=json.loads(path.read_text())
    descendants=[e for e in raw['experiences'] if e['inherited_credit']]
    assert len(descendants)>=4 and any(e['inherited_credit']['inherited_penalty']>0 for e in descendants)
    descendants[-1]['inherited_credit']['negative_credit']+=.01
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='birth credit allocation'):audit(directory,12)
