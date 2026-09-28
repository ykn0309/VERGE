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
    assert b["reflected_candidates"][0]["fusion"] == "MERGE"
    assert c["state"] == "initial_failure" and c["revision_attempts"] == []
    exp = payload["experiences"][0]
    assert [item["fusion"] for item in exp["evolution"][1:]] == ["ADD", "MERGE"]
    assert exp["evolution"][-1]["alpha"] == exp["alpha"]
    assert exp["evolution"][-1]["beta"] == exp["beta"]
    assert exp["evolution"][-1]["confidence"] == exp["confidence"]


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
    assert events[-1]["fusion"] == "MERGE" and events[-1]["confidence"] == events[-2]["confidence"]
    assert not warnings


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
