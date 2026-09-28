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


def test_invalid_revision_retry_config_rejected_before_creating_run_directory(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    data["modules"]["revision"]["config"]["invalid_output_retries"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "invalid_output_retries" in result.stderr
    assert not (tmp_path / "result").exists()


def test_vector_retrieval_requires_real_backend_and_reuses_qwen_key(tmp_path, monkeypatch):
    config, data = experiment(tmp_path, with_test=False)
    data["modules"]["retrieval"] = {"implementation": "vector", "config": {
        "top_l": 20, "embedding_model": "text-embedding-v4", "dimensions": 1024,
    }}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "vector requires" in result.stderr
    assert not (tmp_path / "result").exists()
    monkeypatch.setenv("DASHSCOPE_API_KEY_2", "private-test-key")
    data["models"]["default"] = {
        "backend": "openai_compatible", "model": "qwen3.5-flash",
        "base_url": "https://api.openai.com/v1", "api_key_env": "DASHSCOPE_API_KEY_2",
    }
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "requires a DashScope" in result.stderr
    assert not (tmp_path / "result").exists()
    data["models"]["default"]["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    from verge.cli import _components
    from verge.modules import VectorRetriever
    _, modules, _, _, _ = _components(data)
    try:
        assert isinstance(modules.retriever, VectorRetriever)
        assert modules.retriever.backend._key == modules.backend._key == "private-test-key"
        assert modules.retriever.retrieve("task", []) == []  # 无经验时不产生网络请求
    finally:
        modules.retriever.backend.close()
        modules.backend.close()
    data["modules"]["retrieval"]["config"]["dimensions"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "dimensions" in result.stderr
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


def test_resume_copies_verified_prefix_without_repeating_train_or_mutating_parent(tmp_path,
                                                                                     monkeypatch, capsys):
    config, data = experiment(tmp_path)
    from verge.llm import FakeLLMBackend
    from verge.memory import TrajectoryLog

    original = FakeLLMBackend.generate
    calls = 0

    def fail_second_initial(self, role, payload):
        nonlocal calls
        if role == "initial_generator":
            calls += 1
            if calls == 2:
                raise ValueError("invalid initial rubric")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_second_initial)
    assert main(["run", "--config", str(config)]) == 1
    assert "sample 'train-b', stage initial_generator" in capsys.readouterr().err
    parent = tmp_path / "result"
    old_trajectory = (parent / "artifacts" / "trajectories.jsonl").read_bytes()
    old_memory = (parent / "artifacts" / "experience_memory.json").read_bytes()
    monkeypatch.setattr(FakeLLMBackend, "generate", original)
    data["run"]["output_dir"] = str(tmp_path / "continued")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["train_samples"] == 2 and summary["test_samples"] == 1
    child = tmp_path / "continued"
    records = list(TrajectoryLog(child / "artifacts").all())
    assert [record.sample_id for record in records] == ["train-a", "train-b", "test-a"]
    assert (parent / "artifacts" / "trajectories.jsonl").read_bytes() == old_trajectory
    assert (parent / "artifacts" / "experience_memory.json").read_bytes() == old_memory
    assert (child / "artifacts" / "trajectories.jsonl").read_bytes().startswith(old_trajectory)
    lineage = json.loads((child / "run_manifest.json").read_text())["resumed_from"]
    assert lineage["completed_train"] == 1 and lineage["completed_test"] == 0
    assert lineage["initial_output_retries"] == 2
    assert lineage["checkpoint_sha256"]["experience_memory.json"]
    data["run"]["seed"] = 43
    data["run"]["output_dir"] = str(tmp_path / "bad-config")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 2
    assert "config or dataset changed" in capsys.readouterr().err
    assert not (tmp_path / "bad-config").exists()


def test_resume_after_all_train_skips_training_and_keeps_frozen_memory(tmp_path,
                                                                            monkeypatch, capsys):
    config, data = experiment(tmp_path)
    from verge.llm import FakeLLMBackend
    from verge.memory import TrajectoryLog
    original = FakeLLMBackend.generate
    calls = 0

    def fail_test_initial(self, role, payload):
        nonlocal calls
        if role == "initial_generator":
            calls += 1
            if calls == 3:
                raise ValueError("bad test initial")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_test_initial)
    assert main(["run", "--config", str(config)]) == 1
    capsys.readouterr()
    parent = tmp_path / "result"
    memory = (parent / "artifacts" / "experience_memory.json").read_bytes()
    monkeypatch.setattr(FakeLLMBackend, "generate", original)
    data["run"]["output_dir"] = str(tmp_path / "continued")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 0
    assert json.loads(capsys.readouterr().out)["test_samples"] == 1
    assert [item.sample_id for item in TrajectoryLog(tmp_path / "continued" / "artifacts").all()] == [
        "train-a", "train-b", "test-a",
    ]
    assert (tmp_path / "continued" / "artifacts" / "experience_memory.json").read_bytes() == memory


def test_resume_rejects_memory_write_after_last_complete_trajectory(tmp_path,
                                                                       monkeypatch, capsys):
    config, data = experiment(tmp_path)
    from verge.llm import FakeLLMBackend
    from verge.memory import ExperienceMemory
    original = FakeLLMBackend.generate
    calls = 0

    def fail_second_initial(self, role, payload):
        nonlocal calls
        if role == "initial_generator":
            calls += 1
            if calls == 2:
                raise ValueError("invalid initial rubric")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_second_initial)
    assert main(["run", "--config", str(config)]) == 1
    capsys.readouterr()
    parent = tmp_path / "result"
    memory = ExperienceMemory(parent / "artifacts")
    experience_id = memory.all()[0].experience_id
    memory.apply_downstream_credits("train-b", {experience_id: 0.5})
    data["run"]["output_dir"] = str(tmp_path / "continued")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 2
    assert "Memory contains evidence beyond completed train prefix" in capsys.readouterr().err
    assert not (tmp_path / "continued").exists()


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
