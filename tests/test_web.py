"""The new checkpoint viewer: read-only data, calls, path safety and translation."""

import json, threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from verge.cli import run
from verge.web import DashboardHandler, discover_runs, load_run, load_role_io
from verge.web_observation import load_observation
from test_learning_workflow import config_file


@pytest.fixture
def web(tmp_path):
    config = config_file(tmp_path, tasks=2)
    run(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    server.runs_root = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path
    server.shutdown()
    server.server_close()
    thread.join()


def test_new_run_projects_actual_pair_and_group_evidence(tmp_path):
    run(config_file(tmp_path, tasks=2))
    data = load_run(tmp_path / "run")
    assert len(data["samples"]) == 2 and len(data["experiences"]) == 1
    assert (
        len(data["feedback"]) == 1 and data["group_statistics"][0]["mean_reward"] == 1
    )
    assert not any("q_posterior" in e for e in data["experiences"])
    assert discover_runs(tmp_path)[0]["name"] == "run"


def test_role_io_includes_prompt_output_and_failure_evidence(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    entries = [
        {
            "sample_id": "s",
            "call_id": "c",
            "role": "generator",
            "messages": [{"role": "user", "content": "task\nsecond line"}],
            "raw_output": "{ broken JSON\n",
            "output": {"broken": True},
        },
        {
            "sample_id": "s",
            "call_id": "c",
            "event_type": "contract_validation",
            "validation_status": "invalid",
            "error": "wrong fields",
        },
        {"sample_id": "other", "role": "judge"},
    ]
    (root / "role_io.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n{bad"
    )
    value = load_role_io(root, "s")
    assert len(value["entries"]) == 2 and value["warnings"]
    assert value["entries"][0] == entries[0]
    assert (
        value["entries"][0]["messages"]
        and value["entries"][1]["validation_status"] == "invalid"
    )


def test_http_read_only_routes_and_traversal_protection(web):
    base, root = web
    with urlopen(base + "/api/runs") as response:
        assert json.load(response)["runs"][0]["name"] == "run"
    with urlopen(base + "/api/run?run=run") as response:
        assert len(json.load(response)["samples"]) == 2
    for route in ["/api/run?run=../", "/../checkpoint.json", "/config.json"]:
        with pytest.raises(HTTPError):
            urlopen(base + route)
    assert not (root / "checkpoint.json").exists()


def test_translation_cors_validation_and_local_service_response(web, monkeypatch):
    base, _ = web
    monkeypatch.setattr("verge.web._translate_apple", lambda text: "本地译文")
    request = Request(
        base + "/api/translate",
        data=b'{"text":"hello"}',
        headers={"Content-Type": "application/json", "Origin": base},
    )
    with urlopen(request) as response:
        assert json.load(response)["translation"] == "本地译文"
    request = Request(
        base + "/api/translate",
        data=b'{"text":"hello"}',
        headers={"Origin": "https://example.com"},
    )
    with pytest.raises(HTTPError) as exc:
        urlopen(request)
    assert exc.value.code == 403


def test_cluster_membership_references_and_sources_are_traceable(tmp_path):
    run(config_file(tmp_path, tasks=2))
    data = load_run(tmp_path / "run")
    source = data["experiences"][0]["source_task_id"]
    assert source in data["task_index"]
    assert data["task_index"][source] in {
        r["sample"]["sample_id"] for r in data["samples"]
    }
    assert set(data["references"]) == set(data["task_groups"])
    assert all(
        set(es) <= set(data["experience_groups"]) for es in data["references"].values()
    )


def test_observation_counts_reported_requests_without_counting_a_retry_as_a_sample(
    tmp_path,
):
    run(config_file(tmp_path, tasks=2))
    root = tmp_path / "run"
    (root / "llm_requests.jsonl").write_text(
        json.dumps(
            {"sample_id": "s-pending", "role": "generator", "attempt": 1, "status": 500}
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "s-pending",
                "role": "generator",
                "attempt": 2,
                "status": 200,
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )
        + "\n"
    )
    (root / "role_io.jsonl").write_text(
        json.dumps(
            {"event_type": "request_timing", "call_id": "c", "request_seconds": 2.5}
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "s-pending",
                "call_id": "c",
                "role": "generator",
                "attempt": 1,
                "messages": [{"role": "user", "content": "task"}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "event_type": "contract_validation",
                "call_id": "c",
                "validation_status": "invalid",
                "error": "invalid ID",
            }
        )
        + "\n{unfinished"
    )
    before = (root / "checkpoint.json").read_bytes()
    data = load_observation(root)
    assert data["target_tasks"] == 2
    assert data["requests"] == {
        "total": 2,
        "transport_retries": 1,
        "invalid_contracts": 1,
        "input_tokens": 12,
        "output_tokens": 4,
        "by_role": {"generator": 2},
    }
    assert data["recent_calls"][0]["seconds"] == 2.5
    assert data["recent_calls"][0]["error"] == "invalid ID"
    assert data["uncommitted"][0]["sample_id"] == "s-pending"
    assert (root / "checkpoint.json").read_bytes() == before
    with (root / "llm_requests.jsonl").open("a") as stream:
        stream.write(json.dumps({"role": "judge", "attempt": 1}) + "\n")
    assert load_observation(root)["requests"]["total"] == 3


def test_observation_endpoint_and_static_modules_preserve_path_boundaries(web):
    base, root = web
    before = (root / "run/checkpoint.json").read_bytes()
    with urlopen(base + "/api/observation?run=run") as response:
        data = json.load(response)
        assert data["target_tasks"] == 2
        assert "api_key_env" not in json.dumps(data["settings"])
    for module in ["metrics.mjs", "calls.mjs", "trace.mjs", "views.mjs"]:
        with urlopen(base + "/" + module) as response:
            assert response.status == 200
            assert "javascript" in response.headers["Content-Type"]
            assert response.headers["Cache-Control"] == "no-store"
    with pytest.raises(HTTPError):
        urlopen(base + "/api/observation?run=../")
    assert (root / "run/checkpoint.json").read_bytes() == before


def test_legacy_projection_is_read_only_and_never_a_training_checkpoint(tmp_path):
    from verge.memory import ExperienceMemory

    legacy = {
        "schema": "verge-history-learning-v1",
        "protocol_id": "archival",
        "clustering": {
            "assignments": {"task": {"t": "g"}, "experience": {"e": "h"}},
            "citations": [["t", "e"]],
        },
        "trajectories": [],
        "experiences": [],
        "feedback": [
            {
                "phase": "downstream",
                "record_id": "s",
                "task_id": "t",
                "experience_ids": ["e"],
                "baseline_correct": False,
                "actual_correct": True,
                "error": None,
            }
        ],
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(legacy))
    (tmp_path / "manifest.json").write_text('{"status":"completed"}')
    before = (tmp_path / "checkpoint.json").read_bytes()
    value = load_run(tmp_path)
    assert value["schema"] == "verge-history-learning-v1"
    assert value["references"] == {"t": ["e"]}
    assert value["group_statistics"][0]["mean_reward"] == 1
    assert "credits" not in value["feedback"][0]
    assert (tmp_path / "checkpoint.json").read_bytes() == before
    with pytest.raises(ValueError, match="unsupported"):
        ExperienceMemory.load(tmp_path / "checkpoint.json")
