"""Exercise independent ledger/window reconstruction on deterministic pipeline fixtures."""
import json
from pathlib import Path
import pytest
from test_cli import experiment, rewrite
from verge.cli import run
from scripts.audit_delayed_run import audit


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
    assert result['updates']>0 and result['frozen_experiences']>0
    assert all(n>=3 for n in result['window_sizes'])
    assert result['final_successes']==12
    with pytest.raises(ValueError,match='sample count'):audit(directory,200)


def test_delayed_audit_rejects_corrupted_final_beta(tmp_path):
    directory=fixture_run(tmp_path)
    path=directory/'artifacts/experience_memory.json';raw=json.loads(path.read_text())
    raw['experiences'][0]['alpha']+=.1
    e=raw['experiences'][0];e['confidence']=e['alpha']/(e['alpha']+e['beta'])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Beta'):audit(directory,12)
