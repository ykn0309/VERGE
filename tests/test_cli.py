"""CLI 离线运行、按样本审计和错误退出码。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from verge.cli import main


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT = PROJECT / "configs" / "default.yaml"


def experiment(tmp_path, *, with_test: bool = True, scenario: str = "attempt_1_success"):
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps({
        "sample_id": key, "task": "任务包含明确的关键要求。", "chosen": "好", "rejected": "差",
    }, ensure_ascii=False) for key in ("train-a", "train-b")) + "\n", encoding="utf-8")
    paths = {"train": str(train)}
    if with_test:
        test = tmp_path / "test.jsonl"
        test.write_text(json.dumps({
            "sample_id": "test-a", "task": "任务包含明确的关键要求。",
            "chosen": "好", "rejected": "差",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        paths["test"] = str(test)
    config = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    config["run"]["output_dir"] = str(tmp_path / "result")
    config["dataset"] = {"implementation": "jsonl", "config": {"paths": paths}}
    config["models"]["default"] = {"backend": "fake", "model": scenario}
    config["modules"]["retrieval"]["implementation"] = "fake"
    config["modules"]["selection"]["config"]["exploration_slots"] = 0
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return file, config


def command(*args: str):
    return subprocess.run([sys.executable, "-m", "verge.cli", *args],
                          cwd=PROJECT, capture_output=True, text=True, check=False)


def test_run_trains_then_evaluates_frozen_memory_and_inspects_full_stages(tmp_path):
    config, _ = experiment(tmp_path)
    result = command("run", "--config", str(config))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary == {"run_dir": str(tmp_path / "result"), "train_samples": 2,
                       "test_samples": 1}
    run_dir = tmp_path / "result"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["run"]["seed"] == 42
    assert "+sha256:" in manifest["code_version"]
    assert len(manifest["code_version"].split("+sha256:", 1)[1]) == 64
    stored = json.loads((run_dir / "artifacts" / "experience_memory.json").read_text(encoding="utf-8"))
    assert len(stored["experiences"]) == 1
    # 两次训练 ADD+MERGE，评估样本不会写入 evidence。
    assert {evidence["sample_id"] for evidence in stored["experiences"][0]["evidence_history"]} == {
        "train-a", "train-b"}

    inspected = command("inspect-sample", "--run-dir", str(run_dir), "--sample-id", "train-a")
    assert inspected.returncode == 0, inspected.stderr
    record = json.loads(inspected.stdout)
    assert record["state"] == "revision_success_1"
    assert record["initial_rubric"]["criteria"]
    assert record["initial_judgment"]["criterion_judgments"]
    assert record["revision_attempts"][0]["operations"][0]["before_criterion"] is not None
    assert record["operation_contributions"]
    assert record["reflected_candidates"][0]["credit"] > 0
    assert record["reflected_candidates"][0]["admission"] == "PASS"
    assert record["reflected_candidates"][0]["fusion"] == "ADD"
    next_record = json.loads(command("inspect-sample", "--run-dir", str(run_dir),
                                     "--sample-id", "train-b").stdout)
    assert next_record["reflected_candidates"][0]["fusion"] == "MERGE"
    eval_record = json.loads(command("inspect-sample", "--run-dir", str(run_dir),
                                     "--sample-id", "test-a").stdout)
    assert eval_record["state"] == "initial_failure" and eval_record["revision_attempts"] == []


def test_config_error_exits_2_without_creating_run_directory(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    data["models"]["default"]["backend"] = "not-implemented"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "models.default" in result.stderr
    assert not (tmp_path / "result").exists()


def test_unrecognized_secret_config_is_rejected_before_manifest_write(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    data["models"]["default"]["api_key"] = "should-never-be-persisted"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "models.default" in result.stderr
    assert not (tmp_path / "result").exists()


def test_shipped_default_config_runs_with_a_fresh_output_directory(tmp_path):
    data = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    data["run"]["output_dir"] = str(tmp_path / "default-run")
    config = tmp_path / "shipped-default.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["train_samples"] == 1
    assert json.loads(result.stdout)["test_samples"] == 1


def test_unsupported_dataset_is_not_silently_runnable(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    data["dataset"]["implementation"] = "placeholder"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "dataset.implementation" in result.stderr
    assert not (tmp_path / "result").exists()


def test_role_specific_backend_overrides_default_scenario(tmp_path):
    config, data = experiment(tmp_path, with_test=False, scenario="final_failure")
    data["models"]["roles"]["revision_generator"] = {
        "backend": "fake", "model": "attempt_1_success",
    }
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 0, result.stderr
    record = json.loads(command("inspect-sample", "--run-dir", str(tmp_path / "result"),
                                "--sample-id", "train-a").stdout)
    assert record["state"] == "revision_success_1"


def test_missing_sample_and_usage_exit_codes(tmp_path):
    config, _ = experiment(tmp_path, with_test=False, scenario="initial_success")
    assert command("run", "--config", str(config)).returncode == 0
    missing = command("inspect-sample", "--run-dir", str(tmp_path / "result"),
                      "--sample-id", "does-not-exist")
    assert missing.returncode == 1 and "does-not-exist" in missing.stderr
    assert command("inspect-sample", "--run-dir", str(tmp_path / "result")).returncode == 2
    duplicate_run = command("run", "--config", str(config))
    assert duplicate_run.returncode == 2 and "already exists" in duplicate_run.stderr


def test_failing_sample_reports_id_and_exact_stage_and_keeps_previous_write(tmp_path, capsys,
                                                                                monkeypatch):
    config, _ = experiment(tmp_path, with_test=False)
    from verge.llm import FakeLLMBackend

    original = FakeLLMBackend.generate

    def broken_critic(self, role, payload):
        if role == "critic":
            raise ValueError("deliberate critic failure")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", broken_critic)
    assert main(["run", "--config", str(config)]) == 1
    error = capsys.readouterr().err
    assert "sample 'train-a'" in error and "stage revision_1.critic" in error
    assert "deliberate critic failure" in error
    assert (tmp_path / "result" / "artifacts" / "experience_memory.json").exists()
    # 非事务性写盘；失败样本的 trajectory 不会假装已完成。
    assert not (tmp_path / "result" / "artifacts" / "trajectories.jsonl").exists()


def test_overlapping_train_test_ids_are_rejected_before_ambiguous_inspection(tmp_path):
    config, data = experiment(tmp_path)
    Path(data["dataset"]["config"]["paths"]["test"]).write_text(json.dumps({
        "sample_id": "train-a", "task": "任务包含明确的关键要求。",
        "chosen": "好", "rejected": "差",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 1
    assert "sample 'train-a', stage dataset: duplicate ID across splits" in result.stderr


def test_invalid_dataset_row_reports_file_stage_and_id_when_known(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    path = Path(data["dataset"]["config"]["paths"]["train"])
    path.write_text('{"sample_id":"bad-row","task":"hi"}\n', encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 1 and "stage dataset" in result.stderr
    assert "train.jsonl:1" in result.stderr and "sample_id='bad-row'" in result.stderr
    path.write_text("{broken}\n", encoding="utf-8")
    # 非法 JSON 无法得知 ID；仍报告数据位置和阶段。
    data["run"]["output_dir"] = str(tmp_path / "bad-json")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 1 and "stage dataset" in result.stderr
    assert "train.jsonl:1" in result.stderr
