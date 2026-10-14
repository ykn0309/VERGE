"""Initial-only baseline: no source writes, hidden model calls, or false pairing."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import verge.baseline as baseline
from verge.llm import FakeLLMBackend
from verge.models import TrajectoryRecord
from verge.modules import JudgeOutputError
from verge.prompts import PROMPT_VERSION
from verge.web import load_run


def source_run(tmp_path, count=3):
    source = tmp_path / "main"
    source.mkdir()
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"s{i}",
                    "task": f"task {i}",
                    "chosen": "good",
                    "rejected": "bad",
                    "metadata": {"subset": "llmbar-natural"},
                }
            )
            + "\n"
            for i in range(count)
        )
    )
    config = {
        "run": {"output_dir": str(source), "seed": 42},
        "dataset": {
            "implementation": "jsonl",
            "config": {"paths": {"train": str(data)}},
        },
        "models": {
            "default": {"backend": "fake", "model": "initial_success"},
            "roles": {},
        },
        "modules": {},
        "store": {"config": {"root_dir": "artifacts"}},
    }
    manifest = {
        "format_version": 7,
        "config": config,
        "dataset_sha256": {"train": baseline._sha(data)},
        "prompt_versions": {r: PROMPT_VERSION for r in baseline.ROLES},
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    root = source / "artifacts"
    root.mkdir()
    (root / "experience_memory.json").write_text(
        '{"schema_version": 5, "experiences": []}\n'
    )
    return source, tmp_path / "baseline", data


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_baseline_only_two_roles_and_no_source_writes(tmp_path, monkeypatch):
    source, output, _ = source_run(tmp_path)
    before = snapshot(source)
    backends = []

    def backend(spec, path):
        result = FakeLLMBackend(spec["model"])
        backends.append(result)
        return result

    monkeypatch.setattr(baseline, "_backend", backend)
    result = baseline.run_baseline(source, output)
    assert result["correct"] == result["valid"] == result["samples"] == 3
    assert result["accuracy"] == 1
    assert snapshot(source) == before
    assert len(backends) == 2
    assert [r for r, _ in backends[0].requests] == ["initial_generator"] * 3
    assert [r for r, _ in backends[1].requests] == ["judge"] * 6
    for _, payload in backends[0].requests:
        assert payload["selected_experiences"] == []
        assert "chosen" not in payload and "rejected" not in payload
    records = list(baseline.TrajectoryLog(output / "artifacts").all())
    assert all(
        not r.application_attempts
        and not r.proposal_attempts
        and not r.selected_experiences
        and not r.downstream_credits
        for r in records
    )
    assert (
        output / "artifacts/experience_memory.json"
    ).read_text() == '{"schema_version": 5, "experiences": []}\n'
    assert (output / "artifacts/role_io.jsonl").stat().st_mode & 0o777 == 0o600
    assert load_run(output)["experiences"] == []


@pytest.mark.parametrize("problem", ["existing", "nested", "changed_data", "prompt"])
def test_preflight_rejects_without_backend_or_request(tmp_path, monkeypatch, problem):
    source, output, data = source_run(tmp_path)
    if problem == "existing":
        output.mkdir()
    elif problem == "nested":
        output = source / "baseline"
    elif problem == "changed_data":
        data.write_text("changed")
    else:
        path = source / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["prompt_versions"]["judge"] = "old"
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(baseline, "_backend", lambda *args: pytest.fail("unexpected backend"))
    with pytest.raises(ValueError):
        baseline.run_baseline(source, output)


def test_role_override_and_initial_failure_do_not_trigger_revision(tmp_path):
    source, output, _ = source_run(tmp_path)
    path = source / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["models"]["roles"]["judge"] = {"backend": "fake", "model": "final_failure"}
    path.write_text(json.dumps(manifest))
    result = baseline.run_baseline(source, output)
    assert result["accuracy"] == 0 and result["valid"] == 3
    events = [json.loads(line) for line in (output / "artifacts/role_io.jsonl").read_text().splitlines()]
    assert {e["role"] for e in events} == set(baseline.ROLES)
    assert all(r.state == "initial_failure" for r in baseline.TrajectoryLog(output / "artifacts").all())


def test_judge_error_is_unknown_and_continues(tmp_path, monkeypatch):
    source, output, _ = source_run(tmp_path)
    original = baseline.LLMJudge.judge
    calls = 0

    def judge(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise JudgeOutputError("invalid IDs")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(baseline.LLMJudge, "judge", judge)
    result = baseline.run_baseline(source, output)
    assert result["judge_errors"] == 1 and result["valid"] == 2
    assert result["accuracy"] is None and result["valid_accuracy"] == 1
    first = next(baseline.TrajectoryLog(output / "artifacts").all())
    assert first.state == "judge_error_initial" and first.initial_judgment is None


def test_backend_fault_stops_without_fabricated_trajectory(tmp_path, monkeypatch):
    source, output, _ = source_run(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(baseline.LLMJudge, "judge", broken)
    with pytest.raises(RuntimeError, match="network unavailable"):
        baseline.run_baseline(source, output)
    assert list(baseline.TrajectoryLog(output / "artifacts").all()) == []
    assert not (output / "baseline_summary.json").exists()


def test_pairing_uses_initial_judgments_and_valid_pairs_only(tmp_path, monkeypatch):
    source, output, _ = source_run(tmp_path, 4)
    baseline.run_baseline(source, output)
    path = output / "artifacts/trajectories.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]

    def failed(row):
        row["initial_judgment"].update(
            chosen_score=0, rejected_score=1, margin=-1, success=False
        )

    main = deepcopy(records)
    failed(main[2])
    failed(main[3])
    failed(records[1])
    failed(records[3])
    # A repair success must not replace a failed initial judgment.
    main[2]["state"] = "success_without_candidate"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    source_path = source / "artifacts/trajectories.jsonl"
    source_path.write_text("".join(json.dumps(r) + "\n" for r in main))
    monkeypatch.setattr(
        baseline, "_backend", lambda *args: pytest.fail("compare makes requests")
    )
    result = baseline.compare_runs(source, output)
    assert [
        result[k]
        for k in (
            "both_correct",
            "experience_only_correct",
            "baseline_only_correct",
            "both_wrong",
        )
    ] == [1, 1, 1, 1]
    assert result["net_accuracy_gain"] == 0 and not result["provisional"]
    source_path.write_text(json.dumps(main[0]) + "\n" + '{"partial":')
    result = baseline.compare_runs(source, output)
    assert result["paired_valid"] == 1 and result["excluded_or_pending"] == 3
    assert result["provisional"]
    main[0]["initial_judgment"] = None
    main[0]["state"] = "judge_error_initial"
    main[0]["judge_error"] = "IDs"
    source_path.write_text(json.dumps(main[0]) + "\n")
    result = baseline.compare_runs(source, output)
    assert result["paired_valid"] == 0 and result["net_accuracy_gain"] is None
    assert result["source_initial_errors"] == 1


def test_pairing_rejects_wrong_order_or_source(tmp_path):
    source, output, _ = source_run(tmp_path)
    baseline.run_baseline(source, output)
    path = output / "artifacts/trajectories.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n")
    with pytest.raises(ValueError, match="ordered"):
        baseline.compare_runs(source, output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["baseline_source"]["manifest_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="match"):
        baseline.compare_runs(source, output)


def test_mock_http_logs_numeric_ids_and_no_secrets(tmp_path, monkeypatch, wire_payload):
    import httpx
    from verge.llm import OpenAICompatibleBackend

    source, output, _ = source_run(tmp_path, 1)
    monkeypatch.setenv('BASELINE_TEST_KEY', 'secret-not-in-artifacts')
    calls = []
    clients = []

    def handler(request):
        body = json.loads(request.content)
        payload = wire_payload(body)
        calls.append((body['model'], payload))
        if body['model'] == 'generator-model':
            assert payload['selected_experiences'] == []
            response = {'analysis': '按任务要求组织评价标准。', 'criteria': [
                {'question': f'是否满足要求{i}？',
                 'weight': 5, 'source_experiences': []} for i in (1, 2)]}
        else:
            assert [c['criterion_id'] for c in payload['criteria']] == ['1', '2']
            response = {'criterion_judgments': [
                {"reason": "fixture evidence", 'criterion_id': c['criterion_id'], 'satisfied': payload['response'] == 'good'}
                for c in payload['criteria']]}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(response)}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}})

    def backend(spec, path):
        client = httpx.Client(base_url='https://api.openai.com/v1/',
                              transport=httpx.MockTransport(handler))
        clients.append(client)
        result = OpenAICompatibleBackend('generator-model' if 'initial_generator' in path else 'judge-model',
                                         client=client, api_key_env='BASELINE_TEST_KEY')
        result._owns_client = True  # 测试工厂把 mock client 的所有权交给 backend。
        return result

    monkeypatch.setattr(baseline, '_backend', backend)
    assert baseline.run_baseline(source, output)['accuracy'] == 1
    assert [model for model, _ in calls] == ['generator-model', 'judge-model', 'judge-model']
    root = output / 'artifacts'
    requests = [json.loads(line) for line in (root / 'llm_requests.jsonl').read_text().splitlines()]
    assert {r['role'] for r in requests} == set(baseline.ROLES)
    assert all(r['sample_id'] == 's0' and r['usage']['total_tokens'] == 30 for r in requests)
    assert all(c.is_closed for c in clients)
    assert all(b'secret-not-in-artifacts' not in content for content in snapshot(output).values())


def test_generator_exhaustion_records_unknown_and_continues(tmp_path, monkeypatch):
    source, output, _ = source_run(tmp_path)
    original = baseline.LLMInitialRubricGenerator.generate
    calls = 0
    def invalid_first(self, *args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('criteria count must be in [2, 8]')
        return original(self, *args)
    monkeypatch.setattr(baseline.LLMInitialRubricGenerator, 'generate', invalid_first)
    result = baseline.run_baseline(source, output)
    from verge.memory import TrajectoryLog
    records = list(TrajectoryLog(output / 'artifacts').all())
    assert result['samples'] == 3 and result['sample_errors'] == 1 and result['valid'] == 2
    assert result['accuracy'] is None
    assert records[0].state == 'sample_error' and records[0].initial_rubric is None
    assert records[0].sample_error['stage'] == 'initial_generator'
