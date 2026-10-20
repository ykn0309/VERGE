"""CLI v4, ordered prefixes, bounded workers, new-only checkpoint/config contract."""

import json
from pathlib import Path
from threading import Barrier, Lock
import pytest
import yaml
from verge.cli import main
from verge.llm import FakeLLMBackend, LLMSampleError
from verge.prompts import PROMPT_VERSION

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT = PROJECT / "configs/default.yaml"


def experiment(tmp_path, *, with_test=True, scenario="attempt_1_success"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sid,
                    "task": "任务包含明确关键要求。",
                    "chosen": "好",
                    "rejected": "差",
                }
            )
            + "\n"
            for sid in ("train-a", "train-b")
        )
    )
    paths = {"train": str(train)}
    if with_test:
        test.write_text(
            json.dumps(
                {
                    "sample_id": "test-a",
                    "task": "任务包含明确关键要求。",
                    "chosen": "好",
                    "rejected": "差",
                }
            )
            + "\n"
        )
        paths["test"] = str(test)
    config = yaml.safe_load(DEFAULT.read_text())
    config["run"]["output_dir"] = str(tmp_path / "result")
    config["dataset"] = {"implementation": "jsonl", "config": {"paths": paths}}
    config["models"]["default"] = {"backend": "fake", "model": scenario}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True))
    return path, config


def rewrite(path, config):
    path.write_text(yaml.safe_dump(config, allow_unicode=True))


def records(directory):
    return [
        json.loads(line)
        for line in (directory / "artifacts/trajectories.jsonl")
        .read_text()
        .splitlines()
    ]


def test_cli_train_frozen_test_inspect_and_role_manifest(tmp_path, capsys):
    path, config = experiment(tmp_path)
    assert main(["run", "--config", str(path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    directory = Path(summary["run_dir"])
    assert summary["train_samples"] == 2 and summary["test_samples"] == 1
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert (
        manifest["format_version"] == 10
        and manifest["method"] == "immutable_atomic_evolution"
    )
    assert set(manifest["prompt_versions"].values()) == {PROMPT_VERSION}
    assert set(manifest["prompt_versions"]) == {
        "selector",
        "initial_generator",
        "judge",
        "critic",
        "experience_updater",
        "category_organizer",
        "atomizer",
        "category_describer",
    }
    assert manifest["execution"]["max_application_rounds"] == 2
    assert manifest["execution"]["critic_proposals_per_round"] == 1
    assert "admission_regenerations_per_round" not in manifest["execution"]
    assert "experience_enabled" not in manifest["execution"]
    rs = records(directory)
    assert [r["state"] for r in rs] == [
        "application_success",
        "application_success",
        "initial_failure",
    ]
    assert all(rs[i]["admitted_experience_ids"] for i in (0, 1))
    assert rs[0]["admitted_experience_ids"] != rs[1]["admitted_experience_ids"]
    assert not rs[2]["application_attempts"] and not rs[2]["downstream_credits"]
    pool = json.loads((directory / "artifacts/experience_memory.json").read_text())[
        "experiences"
    ]
    assert len(pool) == 2 and all(
        ev["sample_id"] != "test-a" for ev in pool[0]["evidence_history"]
    )
    assert (
        main(["inspect-sample", "--run-dir", str(directory), "--sample-id", "train-a"])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == rs[0]
    assert main(["run", "--config", str(path)]) == 2
    assert "already exists" in capsys.readouterr().err
    io = directory / "artifacts/role_io.jsonl"
    assert io.stat().st_mode & 0o777 == 0o600
    assert all(
        json.loads(line)["role"] != "reflector" for line in io.read_text().splitlines()
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["run"].update(experience_enabled=False),
        lambda c: c["modules"].update(revision=c["modules"].pop("application")),
        lambda c: c["modules"]["application"].update(
            implementation="critic_generator_dialogue"
        ),
        lambda c: c["models"]["roles"].update(
            reflector={"backend": "fake", "model": "initial_success"}
        ),
        lambda c: c["modules"]["application"]["config"].update(max_rounds=3),
        lambda c: c["modules"]["application"]["config"].update(
            invalid_output_retries=True
        ),
    ],
)
def test_rejects_old_or_invalid_configuration_before_output(tmp_path, mutate, capsys):
    p, c = experiment(tmp_path)
    mutate(c)
    rewrite(p, c)
    assert main(["run", "--config", str(p)]) == 2
    assert not (tmp_path / "result").exists()
    assert capsys.readouterr().err


def test_pause_then_resume_exact_prefix_new_directory(tmp_path, capsys):
    p, c = experiment(tmp_path)
    assert main(["run", "--config", str(p), "--stop-after-train", "1"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (
        summary["status"] == "paused_after_train_prefix"
        and summary["test_samples"] == 0
    )
    source = tmp_path / "result"
    before = {
        str(f.relative_to(source)): f.read_bytes()
        for f in source.rglob("*")
        if f.is_file()
    }
    c["run"]["output_dir"] = str(tmp_path / "resumed")
    rewrite(p, c)
    assert (
        main(["run", "--config", str(p), "--resume-from", str(source)]) == 0
    ), capsys.readouterr().err
    capsys.readouterr()
    resumed = tmp_path / "resumed"
    assert len(records(resumed)) == 3
    assert (
        (resumed / "artifacts/trajectories.jsonl")
        .read_bytes()
        .startswith(before["artifacts/trajectories.jsonl"])
    )
    assert {
        str(f.relative_to(source)): f.read_bytes()
        for f in source.rglob("*")
        if f.is_file()
    } == before
    manifest = json.loads((resumed / "run_manifest.json").read_text())
    assert manifest["resumed_from"]["completed_train"] == 1


@pytest.mark.parametrize(
    "problem",
    ["old_schema", "previous_schema", "prompt", "code", "execution", "memory_pending", "dataset", "order"],
)
def test_resume_rejects_uncertified_parent_without_touching_it(
    tmp_path, problem, capsys
):
    p, c = experiment(tmp_path)
    assert main(["run", "--config", str(p), "--stop-after-train", "1"]) == 0
    capsys.readouterr()
    source = tmp_path / "result"
    mf = source / "run_manifest.json"
    m = json.loads(mf.read_text())
    if problem in {"old_schema", "previous_schema"}:
        m["format_version"] = 1 if problem == "old_schema" else 5
        mf.write_text(json.dumps(m))
    elif problem == "prompt":
        m["prompt_versions"]["critic"] = "old"
        mf.write_text(json.dumps(m))
    elif problem == "code":
        m["code_version"] = "other-code"
        mf.write_text(json.dumps(m))
    elif problem == "execution":
        m["execution"]["initial_output_retries"] = 0
        mf.write_text(json.dumps(m))
    elif problem == "memory_pending":
        f = source / "artifacts/experience_memory.json"
        v = json.loads(f.read_text())
        v["experiences"][0]["source_trajectories"].append("train-b")
        f.write_text(json.dumps(v))
    elif problem == "dataset":
        Path(c["dataset"]["config"]["paths"]["train"]).write_text("{}\n")
    else:
        f = source / "artifacts/trajectories.jsonl"
        v = json.loads(f.read_text())
        v["sample_id"] = "train-b"
        f.write_text(json.dumps(v) + "\n")
    before = {
        str(f.relative_to(source)): f.read_bytes()
        for f in source.rglob("*")
        if f.is_file()
    }
    c["run"]["output_dir"] = str(tmp_path / "resumed")
    rewrite(p, c)
    assert main(["run", "--config", str(p), "--resume-from", str(source)]) == 2
    assert not (tmp_path / "resumed").exists()
    assert {
        str(f.relative_to(source)): f.read_bytes()
        for f in source.rglob("*")
        if f.is_file()
    } == before


def test_errors_count_toward_processed_prefix_and_pause_skips_test(
    tmp_path, monkeypatch, capsys
):
    p, c = experiment(tmp_path)
    original = FakeLLMBackend.generate

    def fail(self, role, payload):
        if role == "initial_generator":
            raise ValueError("invalid external output")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail)
    assert main(["run", "--config", str(p), "--stop-after-train", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["train_samples"] == 1 and result["test_samples"] == 0
    rs = records(tmp_path / "result")
    assert len(rs) == 1 and rs[0]["state"] == "sample_error"


@pytest.mark.parametrize("role", ["initial_generator", "judge"])
def test_request_exhaustion_does_not_abort_cli_run(tmp_path, monkeypatch, capsys, role):
    path, _ = experiment(tmp_path)
    original = FakeLLMBackend.generate
    failed = False
    lock = Lock()

    def fail_once(self, requested_role, payload, **kwargs):
        nonlocal failed
        with lock:
            if requested_role == role and not failed:
                failed = True
                raise LLMSampleError("request failed after 3 attempts (ReadTimeout)")
        return original(self, requested_role, payload, **kwargs)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_once)
    assert main(["run", "--config", str(path)]) == 0
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert summary["train_samples"] == 2 and summary["test_samples"] == 1
    saved = records(tmp_path / "result")
    assert [r["sample_id"] for r in saved] == ["train-a", "train-b", "test-a"]
    assert saved[0]["state"] == "sample_error" and saved[0]["sample_error"]["stage"] == ("initial_judge" if role == "judge" else role)
    assert saved[1]["state"] == "application_success"
    assert saved[2]["initial_judgment"] is not None
    assert "recorded, continuing" in output.err


def test_parallel_evaluation_keeps_order_and_no_candidate_calls(
    tmp_path, monkeypatch, capsys
):
    p, c = experiment(tmp_path, scenario="initial_success")
    test = Path(c["dataset"]["config"]["paths"]["test"])
    test.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"test-{i}",
                    "task": "task",
                    "chosen": "好",
                    "rejected": "差",
                }
            )
            + "\n"
            for i in range(4)
        )
    )
    c["run"]["test_workers"] = 2
    rewrite(p, c)
    original = FakeLLMBackend.generate
    barrier = Barrier(2)
    calls = []
    lock = Lock()

    def generate(self, role, payload):
        if role == "initial_generator" and payload["task"] == "task":
            with lock:
                calls.append(role)
            barrier.wait(timeout=3)
        if role == "critic":
            pytest.fail("initial successes and frozen evaluation must not propose")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", generate)
    assert main(["run", "--config", str(p)]) == 0, capsys.readouterr().err
    assert [r["sample_id"] for r in records(tmp_path / "result")][-4:] == [
        f"test-{i}" for i in range(4)
    ]
    assert len(calls) == 4
