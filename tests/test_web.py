"""Read-only run dashboard: data lineage, replay, and HTTP boundaries."""

import json
import shutil
from hashlib import sha256
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import yaml

from verge.cli import run
from verge.web import DashboardHandler, _evolution, load_run


PROJECT = Path(__file__).resolve().parents[1]


def _run(tmp_path):
    paths = {}
    for split, ids in (("train", ["a", "b"]), ("test", ["c"])):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"sample_id": key, "task": "<script>not code</script>",
                                    "chosen": "好", "rejected": "差"}, ensure_ascii=False) + "\n"
                                for key in ids), encoding="utf-8")
        paths[split] = str(path)
    config = yaml.safe_load((PROJECT / "configs/default.yaml").read_text(encoding="utf-8"))
    config["run"]["output_dir"] = str(tmp_path / "run")
    config["dataset"]["config"]["paths"] = paths
    config["models"]["default"] = {"backend": "fake", "model": "attempt_1_success"}
    config["modules"]["retrieval"]["implementation"] = "fake"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    run(path)
    return tmp_path / "run", paths


def test_load_run_sample_trace_and_experience_evolution(tmp_path):
    directory, _ = _run(tmp_path)
    payload = load_run(directory)
    assert len(payload["samples"]) == 3 and len(payload["experiences"]) == 1
    assert payload["warnings"] == []
    a, b, c = payload["samples"]
    assert a["split"] == "train" and c["split"] == "test"
    assert a["task"] == "<script>not code</script>"
    assert a["revision_attempts"] and a["reflected_candidates"][0]["fusion"] == "ADD"
    assert a["reflected_candidates"][0]["admission_reason"] == "候选满足准入条件。"
    assert b["reflected_candidates"][0]["fusion"] == "MERGE"
    assert b["reflected_candidates"][0]["admission_reason"] == "候选满足准入条件。"
    assert c["state"] == "initial_failure" and c["revision_attempts"] == []
    exp = payload["experiences"][0]
    assert [item["fusion"] for item in exp["evolution"][1:]] == ["ADD", "MERGE"]
    assert exp["evolution"][-1]["alpha"] == exp["alpha"]
    assert exp["evolution"][-1]["beta"] == exp["beta"]
    assert exp["evolution"][-1]["confidence"] == exp["confidence"]
    js = (PROJECT / "verge/web_assets/app.js").read_text(encoding="utf-8")
    assert '准入理由：' in js and '旧轨迹未记录判断理由' not in js
    assert 'h(c.admission_reason' in js  # HTML-escape model text before rendering.


def test_missing_or_changed_dataset_never_displays_unverified_text(tmp_path):
    directory, paths = _run(tmp_path)
    train = Path(paths["train"])
    train.write_text('"changed"\n', encoding="utf-8")
    result = load_run(directory)
    assert result["samples"][0].get("task") is None
    assert result["samples"][0].get("split") is None
    assert result["samples"][-1]["split"] == "test"
    assert any("哈希不匹配" in message for message in result["warnings"])


def test_invalid_and_duplicate_dataset_rows_are_ignored(tmp_path):
    directory, paths = _run(tmp_path)
    train = Path(paths["train"])
    with train.open("a", encoding="utf-8") as stream:
        stream.write('{bad json\n')
        stream.write(json.dumps({"sample_id": "a", "task": "wrong", "chosen": "x", "rejected": "y"}) + "\n")
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"]["train"] = sha256(train.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = load_run(directory)
    assert result["samples"][0].get("task") is None
    assert result["samples"][1]["task"] == "<script>not code</script>"
    assert any("第 3 行无效" in warning for warning in result["warnings"])
    assert any("重复样本 ID" in warning for warning in result["warnings"])


def test_truncated_logs_and_unfinished_request_are_visible(tmp_path):
    directory, _ = _run(tmp_path)
    root = directory / "artifacts"
    with (root / "trajectories.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"partial":\n')
    with (root / "llm_requests.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"sample_id": "uncompleted", "role": "judge", "status": 429,
                                 "prompt": "SECRET", "api_key": "SECRET"}) + "\n")
    result = load_run(directory)
    assert len(result["samples"]) == 4
    assert result["samples"][-1]["state"] == "interrupted"
    assert "SECRET" not in json.dumps(result)
    assert any("trajectories.jsonl" in item for item in result["warnings"])


def test_zero_credit_merge_keeps_origin_without_creating_evidence():
    exp = {"experience_id": "e", "alpha": 1.25, "beta": 1.,
           "evidence_history": [{"sample_id": "a", "evidence_type": "revision_credit",
                                 "credit": .25, "evidence_weight": 1.}]}
    records = [{"sample_id": "a", "reflected_candidates": [{"experience_id": "e", "fusion": "ADD",
                "candidate_id": "c1", "credit": .25, "content": "first", "trigger": "when"}]},
               {"sample_id": "b", "reflected_candidates": [{"experience_id": "e", "fusion": "MERGE",
                "candidate_id": "c2", "credit": 0., "content": "later", "trigger": "when"}]}]
    warnings = []
    events = _evolution(exp, records, {"a": 0, "b": 1}, warnings)
    assert [ev["kind"] for ev in events] == ["prior", "revision_credit", "fusion"]
    assert events[0]["sample_id"] == "a" and events[0]["origin_kind"] == "add"
    assert events[0]["confidence"] == .5 and events[0]["has_trajectory"]
    assert events[-1]["fusion"] == "MERGE" and events[-1]["confidence"] == events[-2]["confidence"]
    assert not warnings


def test_prior_uses_first_source_if_add_trajectory_is_unavailable():
    exp = {"experience_id": "e", "alpha": 1., "beta": 1.,
           "source_trajectories": ["original"], "evidence_history": []}
    events = _evolution(exp, [], {}, [])
    assert events == [{"kind": "prior", "alpha": 1., "beta": 1., "confidence": .5,
                       "sample_id": "original", "origin_kind": "source", "has_trajectory": False}]


def test_equal_credit_candidates_are_not_misattributed():
    exp = {"experience_id": "e", "alpha": 1.4, "beta": 1.,
           "evidence_history": [{"sample_id": "a", "evidence_type": "revision_credit",
                                 "credit": .2, "evidence_weight": 1.} for _ in range(2)]}
    record = {"sample_id": "a", "reflected_candidates": [
        {"experience_id": "e", "fusion": kind, "candidate_id": candidate,
         "credit": .2, "content": candidate, "trigger": "when"}
        for kind, candidate in (("ADD", "c1"), ("MERGE", "c2"))]}
    warnings = []
    events = _evolution(exp, [record], {"a": 0}, warnings)
    assert [item["kind"] for item in events[1:]] == ["revision_credit", "revision_credit", "fusion", "fusion"]
    assert events[-1]["confidence"] == exp["alpha"] / (exp["alpha"] + exp["beta"])
    assert all("fusion" not in item for item in events[1:3])
    assert any("无法唯一绑定" in warning for warning in warnings)


def test_http_can_switch_runs_without_path_traversal(tmp_path):
    parent = tmp_path / "runs"
    for name in ("run-a", "run-b"):
        source = tmp_path / name
        source.mkdir()
        directory, _ = _run(source)
        parent.mkdir(exist_ok=True)
        shutil.move(str(directory), parent / name)
    (parent / "linked").symlink_to(parent / "run-a", target_is_directory=True)
    handler = type("SwitcherHandler", (DashboardHandler,),
                   {"run_dir": parent / "run-a", "runs_root": parent})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port)
            client.request("GET", "/api/runs")
            response = client.getresponse()
            assert json.loads(response.read()) == {"runs": ["run-a", "run-b"], "default": "run-a"}
            client.request("GET", "/api/run?name=run-b")
            response = client.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["run_name"] == "run-b"
            for name in ("..%2Frun-a", "linked", "missing"):
                client.request("GET", f"/api/run?name={name}")
                response = client.getresponse()
                assert response.status == 404
                response.read()
            client.close()
        finally:
            server.shutdown()
            thread.join()


def test_http_only_exposes_fixed_routes_and_rejects_foreign_host(tmp_path):
    directory, _ = _run(tmp_path)
    handler = type("TestHandler", (DashboardHandler,), {"run_dir": directory})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port)
            client.request("GET", "/api/run")
            response = client.getresponse()
            assert response.status == 200
            assert response.getheader("Cache-Control") == "no-store"
            assert len(json.loads(response.read())["samples"]) == 3
            client.request("GET", "/../run_manifest.json")
            response = client.getresponse()
            assert response.status == 404
            response.read()
            client.request("GET", "/api/run", headers={"Host": "untrusted.example"})
            response = client.getresponse()
            assert response.status == 403
            response.read()
            client.close()
        finally:
            server.shutdown()
            thread.join()


def test_role_io_records_real_fake_role_inputs_outputs_without_credentials(tmp_path):
    from verge.web import load_role_io

    directory, _ = _run(tmp_path)
    root = directory / "artifacts"
    assert (root / "role_io.jsonl").stat().st_mode & 0o077 == 0
    a = load_role_io(directory, "a")
    assert a["available"] and a["entries"]
    assert [row["role"] for row in a["entries"]][:4] == ["initial_generator", "judge", "judge", "critic"]
    initial = a["entries"][0]
    assert initial["input"]["task"] == "<script>not code</script>"
    assert "chosen" not in initial["input"] and "rejected" not in initial["input"]
    assert initial["output"]["rubric"]["rubric_id"] == "r0"
    assert initial["messages"] is None and initial["backend"] == "fake"
    assert [row["role"] for row in load_role_io(directory, "c")["entries"]] == [
        "selector", "initial_generator", "judge"] or [
        "initial_generator", "judge"]


def test_role_io_not_yet_written_and_truncated_lines_are_distinguished(tmp_path):
    from verge.web import load_role_io

    directory, _ = _run(tmp_path)
    sidecar = directory / "artifacts" / "role_io.jsonl"
    sidecar.unlink()
    assert load_role_io(directory, "a") == {"available": False, "entries": [], "warnings": []}
    sidecar.write_text(json.dumps({"sample_id": "a", "role": "critic", "input": {"x": "y"},
                                   "output": {"analysis": "诊断", "operations": []}, "Authorization": "Bearer secret"}) +
                       "\n{" + "\n", encoding="utf-8")
    result = load_role_io(directory, "a")
    assert result["available"] and len(result["entries"]) == 1
    assert "Authorization" not in result["entries"][0]
    assert result["warnings"]


def test_role_io_bounds_large_logs_and_truncates_sample_calls(tmp_path, monkeypatch):
    from verge.web import load_role_io

    directory, _ = _run(tmp_path)
    monkeypatch.setattr("verge.web._MAX_ROLE_ENTRIES", 1)
    truncated = load_role_io(directory, "a")
    assert len(truncated["entries"]) == 1
    assert any("仅展示前 1 次" in warning for warning in truncated["warnings"])
    monkeypatch.setattr("verge.web._MAX_ROLE_IO_BYTES", 1)
    oversized = load_role_io(directory, "a")
    assert oversized["error"] and not oversized["entries"]


def test_http_role_io_only_allows_selected_run_and_sample(tmp_path):
    directory, _ = _run(tmp_path)
    handler = type("RoleHandler", (DashboardHandler,), {"run_dir": directory})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port)
            client.request("GET", f"/api/role-io?name={directory.name}&sample_id=a")
            response = client.getresponse()
            assert response.status == 200 and json.loads(response.read())["entries"]
            client.request("GET", f"/api/role-io?name={directory.name}&sample_id=missing")
            response = client.getresponse()
            assert response.status == 404
            response.read()
            client.request("GET", "/api/role-io?name=..%2F..&sample_id=a")
            response = client.getresponse()
            assert response.status == 404
            response.read()
            client.request("GET", f"/api/role-io?name={directory.name}&sample_id=a&sample_id=b")
            response = client.getresponse()
            assert response.status == 400
            response.read()
            client.close()
        finally:
            server.shutdown()
            thread.join()


def test_overview_replays_pool_without_new_log_and_uses_pre_task_confidence(tmp_path):
    directory, paths = _run(tmp_path)
    trajectory = directory / 'artifacts/trajectories.jsonl'
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    eid = records[0]['reflected_candidates'][0]['experience_id']
    records[1]['selected_experience_ids'] = [eid]  # 未引用仍是零信用的使用记录。
    trajectory.write_text(''.join(json.dumps(r) + '\n' for r in records))
    payload = load_run(directory)
    o = payload['overview']
    assert not (directory / 'artifacts/evolution.jsonl').exists()
    assert o['train_completed'] == o['train_total'] == 2
    assert o['initial']['valid'] == 2
    assert o['adds'] == 1 and o['merges'] == 1 and o['merge_rate'] == .5
    assert [p['pool'] for p in o['curve']] == [1, 1]
    assert o['replay_available']
    assert sum(b['uses'] for b in o['confidence_bins']) == 1
    # task b 使用的是 a 写入后的状态，不能包含 b 的合并信用。
    exp = payload['experiences'][0]
    credit = exp['evidence_history'][0]['credit']
    pre = (1 + max(credit, 0)) / (2 + abs(credit))
    assert o['confidence_bins'][min(9, int(pre * 10))]['uses'] == 1
    assert o['scale_bins'][0]['tasks'] == 1
    assert o['scale_bins'][1]['tasks'] == 1
    assert o['segments'][0]['total'] == 2
    assert o['segments'][0]['errors'] == 0
    # 数据丢失时不猜测训练身份或使用前状态。
    Path(paths['train']).unlink()
    missing = load_run(directory)['overview']
    assert missing['train_completed'] == 0
    assert missing['confidence_bins'] is None


def test_overview_missing_evidence_prefix_suppresses_confidence(tmp_path):
    directory, _ = _run(tmp_path)
    path = directory / 'artifacts/experience_memory.json'
    memory = json.loads(path.read_text())
    memory['experiences'][0]['evidence_history'][0]['sample_id'] = 'unfinished'
    path.write_text(json.dumps(memory))
    result = load_run(directory)
    assert result['overview']['confidence_bins'] is None
    assert result['overview']['scale_bins'] is None
    assert any('无法与完整训练轨迹对齐' in w for w in result['warnings'])


def test_overview_judge_errors_and_zero_denominators():
    from verge.web import _overview_metrics
    sample = {'sample_id': 'a', 'split': 'train', 'state': 'judge_error_initial',
              'initial_judgment': None, 'initial_rubric': {'criteria': []}}
    manifest = {'config': {'dataset': {'config': {'paths': {'train': 'unused'}}}}}
    o = _overview_metrics([sample], [], {'a': {'split': 'train'}}, manifest, [])
    assert o['initial'] == {'total': 1, 'valid': 0, 'correct': 0, 'errors': 1, 'accuracy': None}
    assert o['curve'][0]['rolling']['accuracy'] is None
    assert o['repair_rate'] is None and o['merge_rate'] is None
    assert o['mode'] == 'train_only'
    assert o['scale_bins'][0]['selection_rate'] is None


def test_train_only_empty_completed_and_resume(tmp_path):
    from verge.cli import run
    config = yaml.safe_load((PROJECT / 'configs/default.yaml').read_text())
    dataset = tmp_path / 'train.jsonl'
    dataset.write_text(''.join(json.dumps({'sample_id': key, 'task': 'test task',
                                          'chosen': 'good', 'rejected': 'bad'}) + '\n'
                               for key in ('a', 'b')))
    config['dataset']['config']['paths'] = {'train': str(dataset)}
    config['run']['output_dir'] = str(tmp_path / 'only')
    config['models']['default'] = {'backend': 'fake', 'model': 'initial_success'}
    config['modules']['retrieval']['implementation'] = 'fake'
    path = tmp_path / 'only.yaml'
    path.write_text(yaml.safe_dump(config))
    run(path)
    o = load_run(tmp_path / 'only')['overview']
    assert o['mode'] == 'train_only' and o['train_completed'] == o['train_total'] == 2
    # 构造首任务完整提交的前缀，无经验池写入；已完成运行本身不得续跑。
    trajectory = tmp_path / 'only/artifacts/trajectories.jsonl'
    trajectory.write_text(trajectory.read_text().splitlines()[0] + '\n')
    role_io = tmp_path / 'only/artifacts/role_io.jsonl'
    role_io.write_text(''.join(line + '\n' for line in role_io.read_text().splitlines()
                               if json.loads(line).get('sample_id') == 'a'))
    config['run']['output_dir'] = str(tmp_path / 'resume')
    path.write_text(yaml.safe_dump(config))
    run(path, resume_from=tmp_path / 'only')
    assert load_run(tmp_path / 'resume')['overview']['curve'] == o['curve']
    empty = tmp_path / 'empty'
    shutil.copytree(tmp_path / 'only', empty)
    (empty / 'artifacts/trajectories.jsonl').write_text('')
    (empty / 'artifacts/experience_memory.json').write_text('{"experiences": []}')
    result = load_run(empty)['overview']
    assert result['train_completed'] == 0 and result['train_total'] == 2
    assert result['initial']['accuracy'] is None and result['curve'] == []


def test_generator_analysis_and_experience_status_from_existing_logs(tmp_path):
    directory, _ = _run(tmp_path)
    sidecar = directory / 'artifacts/role_io.jsonl'
    entries = [json.loads(line) for line in sidecar.read_text().splitlines()]
    for entry in entries:
        if entry.get('sample_id') == 'a' and entry.get('role') == 'initial_generator':
            entry['output']['analysis'] = '<script>分析原文</script>'
    sidecar.write_text(''.join(json.dumps(e) + '\n' for e in entries))
    a, b, c = load_run(directory)['samples']
    assert a['generator_analysis'] == '<script>分析原文</script>'
    assert a['experience_status']['generated_candidates'] == 1
    assert a['experience_status']['added'] == 1
    assert b['experience_status']['merged'] == 1
    assert c['generator_analysis'] is None
    # Empty candidate output differs from missing older role logs.
    sidecar.unlink()
    a = load_run(directory)['samples'][0]
    assert a['generator_analysis'] is None
    assert a['experience_status']['generated_candidates'] is None


def test_experience_status_distinguishes_selected_referenced_and_admitted():
    from verge.web import _sample_experience_status
    sample = {'state': 'revision_success_1', 'selected_experience_ids': ['e1', 'e2'],
              'initial_rubric': {'criteria': [{'source_experiences': ['e1', 'e1']}]},
              'reflected_candidates': [{'fusion': 'FAIL'}]}
    status = _sample_experience_status(sample, {'generated_candidates': 3})
    assert status == {'generated_candidates': 3,
                      'added': 0, 'merged': 0, 'selected': 2, 'referenced': 1}
    assert _sample_experience_status({'state': 'interrupted'}, {})['referenced'] is None


def test_generation_summary_latest_retry_empty_and_corrupt_records(tmp_path, monkeypatch):
    from verge.web import _generation_summaries
    sidecar = tmp_path / 'role_io.jsonl'
    entries = [
        {'sample_id': 'a', 'role': 'initial_generator', 'output': {'analysis': '第一次'}},
        {'sample_id': 'a', 'role': 'initial_generator', 'error': 'format', 'output': {'analysis': '错误'}},
        {'sample_id': 'a', 'role': 'initial_generator', 'output': {'analysis': '最终分析'}},
        {'sample_id': 'a', 'role': 'reflector', 'output': {'candidates': []}},
        {'sample_id': 'b', 'role': 'initial_generator', 'output': {'analysis': 123}},
    ]
    sidecar.write_text(''.join(json.dumps(e) + '\n' for e in entries) + '{truncated\n')
    warnings = []
    summaries = _generation_summaries(tmp_path, warnings)
    assert summaries['a'] == {'generator_analysis': '最终分析', 'generated_candidates': 0}
    assert summaries['b']['generator_analysis'] is None
    assert warnings
    monkeypatch.setattr('verge.web._MAX_ROLE_IO_BYTES', 1)
    assert _generation_summaries(tmp_path, []) == {}


def test_web_rejects_previous_prompt_version(tmp_path):
    import pytest
    directory, _ = _run(tmp_path)
    path = directory / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['prompt_versions'] = {role: 'v16' for role in manifest['prompt_versions']}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='current prompt version'):
        load_run(directory)
    js = (PROJECT / 'verge/web_assets/app.js').read_text()
    assert 'revision_generator' not in js
    assert 'bad.feedback' not in js and 'attempt.feedback' not in js


def test_empty_runs_http_still_serves_dashboard(tmp_path):
    handler = type('EmptyDashboard', (DashboardHandler,),
                   {'run_dir': tmp_path / '__pending__', 'runs_root': tmp_path})
    with ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection('127.0.0.1', server.server_port)
            connection.request('GET', '/api/runs')
            result = connection.getresponse()
            assert result.status == 200 and json.loads(result.read())['runs'] == []
            connection.request('GET', '/')
            result = connection.getresponse()
            assert result.status == 200 and b'app.js' in result.read()
            connection.close()
        finally:
            server.shutdown()
            thread.join()


def test_sample_error_before_r0_is_visible_and_excluded_from_initial_accuracy(tmp_path):
    directory, _ = _run(tmp_path)
    path = directory / 'artifacts/trajectories.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1].update(state='sample_error', initial_rubric=None, initial_judgment=None,
                   revision_attempts=[], reflected_candidates=[], operation_contributions={},
                   sample_error={'stage':'initial_generator', 'type':'ValueError',
                                 'message':'criteria count must be in [2, 8]'})
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = load_run(directory)
    assert result['overview']['sample_errors'] == 1
    assert result['overview']['initial']['valid'] == 1
    assert result['overview']['initial']['errors'] == 1
    assert result['samples'][1]['sample_error']['stage'] == 'initial_generator'
    assert result['samples'][1]['experience_status']['referenced'] is None
