"""New-only read-only dashboard, dual credit replay and role IO boundary."""

import hashlib, json
from pathlib import Path
import pytest, yaml
from verge.cli import run
from verge.web import load_run, load_role_io, discover_runs


def _run(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sid,
                    "task": "<script>not code</script>",
                    "chosen": "好",
                    "rejected": "差",
                }
            )
            + "\n"
            for sid in ("a", "b")
        )
    )
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["run"]["output_dir"] = str(tmp_path / "run")
    config["dataset"]["config"]["paths"] = {"train": str(data)}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(config))
    run(p)
    return tmp_path / "run", {"train": data}


def test_dashboard_candidates_adoption_dual_credit_and_no_writes(tmp_path):
    directory, _ = _run(tmp_path)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result = load_run(directory)
    assert result["warnings"] == [] and result["counts"] == {"application_success": 2}
    assert result["overview"]["adopted"] == 2 and result["overview"]["candidates"] == 2
    assert result["overview"]["adds"] == 1 and result["overview"]["merges"] == 1
    e = result["experiences"][0]
    assert e["evolution"][-1]["alpha"] == pytest.approx(e["alpha"])
    assert e["evolution"][-1]["beta"] == pytest.approx(e["beta"])
    assert all("credit" not in event for event in e["evolution"])
    assert all("reflected_candidates" not in s for s in result["samples"])
    io = load_role_io(directory, "a")
    assert io["available"] and any(e["role"] == "critic" for e in io["entries"])
    assert {p: p.read_bytes() for p in before} == before


def test_old_manifest_hidden_without_deleting_directory(tmp_path):
    directory, _ = _run(tmp_path)
    old = tmp_path / "old"
    old.mkdir()
    (old / "run_manifest.json").write_text('{"format_version":1}')
    assert discover_runs(tmp_path) == ["run"] and old.exists()
    with pytest.raises(ValueError):
        load_run(old)
    m = json.loads((directory / "run_manifest.json").read_text())
    m["prompt_versions"]["critic"] = "old"
    (directory / "run_manifest.json").write_text(json.dumps(m))
    assert discover_runs(tmp_path) == []


def test_dataset_hash_mismatch_hides_unverified_context(tmp_path):
    directory, paths = _run(tmp_path)
    paths["train"].write_text("{}\n")
    result = load_run(directory)
    assert result["warnings"] and all("task" not in s for s in result["samples"])


def test_role_io_only_selected_sample_and_no_headers(tmp_path):
    directory, _ = _run(tmp_path)
    file = directory / "artifacts/role_io.jsonl"
    with file.open("a") as f:
        f.write(
            json.dumps(
                {
                    "sample_id": "a",
                    "role": "critic",
                    "headers": {"secret": "never-show"},
                    "input": {},
                    "output": {},
                }
            )
            + "\n"
        )
    result = load_role_io(directory, "a")
    assert "never-show" not in str(result) and all(
        "sample_id" not in e for e in result["entries"]
    )


def test_truncated_jsonl_warning_no_fake_trajectory(tmp_path):
    directory, _ = _run(tmp_path)
    file = directory / "artifacts/trajectories.jsonl"
    with file.open("a") as f:
        f.write('{"sample_id":')
    result = load_run(directory)
    assert len(result["samples"]) == 2 and result["warnings"]
