"""CLI 离线运行、按样本审计和错误退出码。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from threading import Event

import yaml
import pytest

from verge.cli import main
from verge.llm import FakeLLMBackend, OpenAICompatibleBackend


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


def _test_rows(tmp_path, config, count=3):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps({
        "sample_id": f"test-{index}", "task": f"测试任务 {index}",
        "chosen": f"chosen-{index}", "rejected": f"rejected-{index}",
    }, ensure_ascii=False) + "\n" for index in range(count)), encoding="utf-8")
    config["dataset"]["config"]["paths"]["test"] = str(path)


def test_test_workers_run_concurrently_commit_order_and_show_progress(tmp_path, monkeypatch, capsys):
    path, config = experiment(tmp_path)
    _test_rows(tmp_path, config)
    config["run"]["test_workers"] = 2
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    original = FakeLLMBackend.generate
    second_started = Event()

    def overlapping(self, role, payload):
        if role == "initial_generator":
            if payload["task"] == "测试任务 0":
                assert second_started.wait(5), "test samples did not overlap"
            elif payload["task"] == "测试任务 1":
                second_started.set()
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", overlapping)
    assert main(["run", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["test_samples"] == 3
    assert "train: [" in captured.err and "2/2" in captured.err
    assert "test: [" in captured.err and "3/3" in captured.err
    root = tmp_path / "result" / "artifacts"
    records = [json.loads(line) for line in (root / "trajectories.jsonl").read_text().splitlines()]
    assert [item["sample_id"] for item in records] == [
        "train-a", "train-b", "test-0", "test-1", "test-2",
    ]
    role_events = [json.loads(line) for line in (root / "role_io.jsonl").read_text().splitlines()]
    test_events = [event for event in role_events if event["split"] == "test"]
    assert test_events
    for event in test_events:
        suffix = event["sample_id"].removeprefix("test-")
        if "task" in event["input"]:
            assert event["input"]["task"] == "测试任务 " + suffix
        if "chosen" in event["input"]:
            assert event["input"]["chosen"] == "chosen-" + suffix
    memory = json.loads((root / "experience_memory.json").read_text())
    assert all(evidence["sample_id"].startswith("train-") for item in memory["experiences"]
               for evidence in item["evidence_history"])


def test_parallel_real_backend_metadata_uses_worker_sample_context(tmp_path, monkeypatch, capsys):
    path, config = experiment(tmp_path)
    _test_rows(tmp_path, config, count=2)
    config["run"]["test_workers"] = 2
    monkeypatch.setenv("TEST_BACKEND_API_KEY", "secret-for-test")
    config["models"]["default"] = {"backend": "openai_compatible", "model": "offline-stub",
                                   "base_url": "http://localhost:12345/v1",
                                   "api_key_env": "TEST_BACKEND_API_KEY"}
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    other_started = Event()

    def offline_generate(self, role, payload):
        from verge.llm import JUDGE_RESPONSE_SIDE
        if role == "initial_generator":
            if payload["task"] == "测试任务 0":
                assert other_started.wait(5), "real-backend test requests did not overlap"
            elif payload["task"] == "测试任务 1":
                other_started.set()
            result = {"analysis": "根据任务分析评价要求。", "criteria": [{"question": f"问题{i}？",
                                               "importance_score": 5, "source_experiences": []}
                                              for i in range(1, 5)]}
        elif role == "judge":
            result = {"criterion_judgments": [
                {"criterion_id": item["criterion_id"], "satisfied": JUDGE_RESPONSE_SIDE.get() == "chosen"}
                for item in payload["criteria"]]}
        else:
            raise AssertionError(f"unexpected role: {role}")
        self._trace(role, payload, [], 1, status=200, output=result)
        self._metadata(role, 1, status=200)
        return result

    monkeypatch.setattr(OpenAICompatibleBackend, "generate", offline_generate)
    assert main(["run", "--config", str(path)]) == 0
    capsys.readouterr()
    root = tmp_path / "result" / "artifacts"
    events = [json.loads(line) for line in (root / "llm_requests.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in (root / "role_io.jsonl").read_text().splitlines()]
    for sample_id in ("test-0", "test-1"):
        assert [item["role"] for item in events if item["sample_id"] == sample_id] == [
            "initial_generator", "judge", "judge"]
        assert [item["role"] for item in traces if item["sample_id"] == sample_id] == [
            "initial_generator", "judge", "judge"]
        assert next(item for item in traces if item["sample_id"] == sample_id
                    and item["role"] == "initial_generator")["input"]["task"] == (
                        "测试任务 " + sample_id[-1])
    assert [json.loads(line)["sample_id"] for line in
            (root / "trajectories.jsonl").read_text().splitlines()] == [
                "train-a", "train-b", "test-0", "test-1"]


def test_sequential_and_parallel_test_trajectories_match(tmp_path, capsys):
    path, config = experiment(tmp_path, scenario="initial_success")
    _test_rows(tmp_path, config)
    config["run"]["test_workers"] = 1
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(path)]) == 0
    capsys.readouterr()
    old = (tmp_path / "result" / "artifacts" / "trajectories.jsonl").read_bytes()
    config["run"].update(test_workers=4, output_dir=str(tmp_path / "parallel"))
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(path)]) == 0
    assert old == (tmp_path / "parallel" / "artifacts" / "trajectories.jsonl").read_bytes()


@pytest.mark.parametrize("failed_index", [0, 1])
def test_parallel_test_failure_keeps_prefix_and_can_resume_with_later_requests(
        tmp_path, monkeypatch, capsys, failed_index):
    path, config = experiment(tmp_path)
    _test_rows(tmp_path, config, count=failed_index + 2)
    config["run"]["test_workers"] = 2
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    original = FakeLLMBackend.generate
    second_finished = Event()

    def fail_mid_test(self, role, payload):
        if role == "initial_generator" and payload["task"] == f"测试任务 {failed_index}":
            assert second_finished.wait(5), "later test did not finish first"
            raise RuntimeError("simulated test failure")
        result = original(self, role, payload)
        if role == "judge" and payload["response"] == f"chosen-{failed_index + 1}":
            second_finished.set()
        return result

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_mid_test)
    assert main(["run", "--config", str(path)]) == 1
    assert f"sample 'test-{failed_index}', stage initial_generator" in capsys.readouterr().err
    root = tmp_path / "result" / "artifacts"
    prefix = ["train-a", "train-b"] + [f"test-{i}" for i in range(failed_index)]
    assert [json.loads(line)["sample_id"] for line in
            (root / "trajectories.jsonl").read_text().splitlines()] == prefix
    assert any(json.loads(line)["sample_id"] == f"test-{failed_index + 1}" for line in
               (root / "role_io.jsonl").read_text().splitlines())
    monkeypatch.setattr(FakeLLMBackend, "generate", original)
    config["run"]["output_dir"] = str(tmp_path / "continued")
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(path), "--resume-from", str(tmp_path / "result")]) == 0
    assert [json.loads(line)["sample_id"] for line in
            (tmp_path / "continued" / "artifacts" / "trajectories.jsonl").read_text().splitlines()
            ] == prefix + [f"test-{i}" for i in range(failed_index, failed_index + 2)]


def test_test_workers_config_validation(tmp_path):
    path, config = experiment(tmp_path)
    for value in (True, 0, 17, 1.5):
        config["run"]["test_workers"] = value
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = command("run", "--config", str(path))
        assert result.returncode == 2 and "run.test_workers" in result.stderr
        assert not (tmp_path / "result").exists()


def test_config_error_exits_2_without_creating_run_directory(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    data["models"]["default"]["backend"] = "not-implemented"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "models.default" in result.stderr
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("name", [
    "default.yaml", "openai.example.yaml", "qwen3.5-flash.yaml",
    "rewardbench-chat-hard-qwen.yaml", "rewardbench-chat-hard-qwen-vector.yaml",
])
def test_active_configs_enable_revision_structure_retries(name):
    config = yaml.safe_load((PROJECT / "configs" / name).read_text(encoding="utf-8"))
    assert config["modules"]["revision"]["config"]["invalid_output_retries"] == 2


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
    data["models"]["roles"]["critic"] = {
        "backend": "fake", "model": "attempt_1_success",
    }
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 0, result.stderr
    record = json.loads(command("inspect-sample", "--run-dir", str(tmp_path / "result"),
                                "--sample-id", "train-a").stdout)
    assert record["state"] == "revision_success_1"


def test_removed_generator_role_config_is_rejected(tmp_path):
    config, data = experiment(tmp_path, with_test=False)
    data["models"]["roles"]["revision_generator"] = {
        "backend": "fake", "model": "attempt_1_success",
    }
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    result = command("run", "--config", str(config))
    assert result.returncode == 2 and "models.roles" in result.stderr
    assert not (tmp_path / "result").exists()


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
    assert main(["run", "--config", str(config)]) == 0
    records = [json.loads(line) for line in (tmp_path / "result/artifacts/trajectories.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in records] == ["train-a", "train-b"]
    assert all(r["state"] == "sample_error" for r in records)
    assert records[0]["sample_error"]["stage"] == "revision_1.critic"
    assert "deliberate critic failure" in records[0]["sample_error"]["message"]
    assert (tmp_path / "result/artifacts/experience_memory.json").exists()


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
                raise RuntimeError("invalid initial rubric")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_second_initial)
    assert main(["run", "--config", str(config)]) == 1
    assert "sample 'train-b', stage initial_generator" in capsys.readouterr().err
    parent = tmp_path / "result"
    old_trajectory = (parent / "artifacts" / "trajectories.jsonl").read_bytes()
    old_memory = (parent / "artifacts" / "experience_memory.json").read_bytes()
    old_role_io = (parent / "artifacts" / "role_io.jsonl").read_bytes()
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
    assert (parent / "artifacts" / "role_io.jsonl").read_bytes() == old_role_io
    assert (child / "artifacts" / "trajectories.jsonl").read_bytes().startswith(old_trajectory)
    assert (child / "artifacts" / "role_io.jsonl").read_bytes().startswith(old_role_io)
    assert (child / "artifacts" / "role_io.jsonl").stat().st_mode & 0o077 == 0
    lineage = json.loads((child / "run_manifest.json").read_text())["resumed_from"]
    assert lineage["completed_train"] == 1 and lineage["completed_test"] == 0
    assert lineage["initial_output_retries"] == 2
    assert lineage["checkpoint_sha256"]["experience_memory.json"]
    assert lineage["checkpoint_sha256"]["role_io.jsonl"]
    data["run"]["seed"] = 43
    data["run"]["output_dir"] = str(tmp_path / "bad-config")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 2
    assert "config or dataset changed" in capsys.readouterr().err
    assert not (tmp_path / "bad-config").exists()


def test_resume_rejects_prior_prompt_protocol_before_any_model_call(tmp_path, capsys):
    config, data = experiment(tmp_path)
    assert main(["run", "--config", str(config)]) == 0, capsys.readouterr().err
    parent = tmp_path / "result"
    manifest_file = parent / "run_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["prompt_versions"]["critic"] = "v9-numbered-ids"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    data["run"]["output_dir"] = str(tmp_path / "new-run")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 2
    assert "prompt protocol changed" in capsys.readouterr().err
    assert not (tmp_path / "new-run").exists()


def test_resume_after_failed_revision_without_memory_write(tmp_path, monkeypatch, capsys):
    config, data = experiment(tmp_path)
    from verge.llm import FakeLLMBackend
    from verge.memory import TrajectoryLog

    original = FakeLLMBackend.generate
    selections = 0
    revisions = 0

    def fail_second_revision(self, role, payload):
        nonlocal selections, revisions
        if role == "selector":
            selections += 1
            if selections == 1:  # 首条样本的池为空，首次 Selector 调用发生在 train-b。
                return {"selected_experience_ids": []}
        if role == "critic":
            revisions += 1
            if revisions >= 2:
                raise RuntimeError("simulated revision timeout")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_second_revision)
    assert main(["run", "--config", str(config)]) == 1
    assert "sample 'train-b', stage revision_1.critic" in capsys.readouterr().err
    parent = tmp_path / "result"
    root = parent / "artifacts"
    assert [record.sample_id for record in TrajectoryLog(root).all()] == ["train-a"]
    # 模拟真实 LLM 在该修订角色超时后仍留下 request / role IO 诊断。
    for filename in ("llm_requests.jsonl", "role_io.jsonl"):
        with (root / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"sample_id": "train-b", "role": "critic"}) + "\n")
    old_memory = (root / "experience_memory.json").read_bytes()
    assert all(ev["sample_id"] != "train-b" for exp in json.loads(old_memory)["experiences"]
               for ev in exp["evidence_history"])
    monkeypatch.setattr(FakeLLMBackend, "generate", original)
    data["run"]["output_dir"] = str(tmp_path / "continued")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 0
    assert json.loads(capsys.readouterr().out)["train_samples"] == 2
    assert [record.sample_id for record in TrajectoryLog(tmp_path / "continued" / "artifacts").all()] == [
        "train-a", "train-b", "test-a",
    ]
    assert (root / "experience_memory.json").read_bytes() == old_memory


@pytest.mark.parametrize("log_file,error", [
    ("llm_requests.jsonl", "request event beyond next sample"),
    ("role_io.jsonl", "role IO event beyond next sample"),
])
def test_resume_rejects_request_or_role_io_beyond_next_sample(tmp_path, monkeypatch, capsys,
                                                              log_file, error):
    config, data = experiment(tmp_path)
    from verge.llm import FakeLLMBackend
    original = FakeLLMBackend.generate
    calls = 0

    def fail_second_initial(self, role, payload):
        nonlocal calls
        if role == "initial_generator":
            calls += 1
            if calls == 2:
                raise RuntimeError("stop before second sample")
        return original(self, role, payload)

    monkeypatch.setattr(FakeLLMBackend, "generate", fail_second_initial)
    assert main(["run", "--config", str(config)]) == 1
    capsys.readouterr()
    parent = tmp_path / "result"
    with (parent / "artifacts" / log_file).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"sample_id": "test-a", "role": "judge"}) + "\n")
    monkeypatch.setattr(FakeLLMBackend, "generate", original)
    data["run"]["output_dir"] = str(tmp_path / "continued")
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(config), "--resume-from", str(parent)]) == 2
    assert error in capsys.readouterr().err
    assert not (tmp_path / "continued").exists()


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
                raise RuntimeError("bad test initial")
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
                raise RuntimeError("invalid initial rubric")
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


def test_stop_at_committed_train_prefix_and_resume_with_full_dataset(tmp_path, capsys):
    from verge.cli import run
    file, config = experiment(tmp_path)
    original_data = Path(config['dataset']['config']['paths']['train']).read_bytes()
    result = run(file, stop_after_train=1)
    assert result['status'] == 'paused_after_train_prefix'
    assert result['train_samples'] == 1 and result['train_total'] == 2
    assert result['test_samples'] == 0
    parent = Path(result['run_dir'])
    trajectory = parent / 'artifacts/trajectories.jsonl'
    original_checkpoint = trajectory.read_bytes()
    assert [json.loads(line)['sample_id'] for line in original_checkpoint.splitlines()] == ['train-a']
    entries = [json.loads(line) for line in (parent / 'artifacts/role_io.jsonl').read_text().splitlines()]
    assert {e['sample_id'] for e in entries} == {'train-a'}
    manifest = json.loads((parent / 'run_manifest.json').read_text())
    assert manifest['execution']['stop_after_train'] == 1
    assert set(manifest['config']['dataset']['config']['paths']) == {'train', 'test'}
    config['run']['output_dir'] = str(tmp_path / 'continued')
    file.write_text(yaml.safe_dump(config, allow_unicode=True))
    # Absolute prefix cap counts the copied task, not only new tasks.
    continued = run(file, resume_from=parent, stop_after_train=2)
    assert continued['train_samples'] == 2 and continued['test_samples'] == 1
    assert 'status' not in continued
    records = [json.loads(line) for line in
               (Path(continued['run_dir']) / 'artifacts/trajectories.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in records] == ['train-a', 'train-b', 'test-a']
    assert trajectory.read_bytes() == original_checkpoint
    assert Path(config['dataset']['config']['paths']['train']).read_bytes() == original_data


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_stop_after_train_rejected_before_creating_run(tmp_path, limit):
    from verge.cli import ConfigError, run
    file, config = experiment(tmp_path)
    with pytest.raises(ConfigError, match='positive integer'):
        run(file, stop_after_train=limit)
    assert not Path(config['run']['output_dir']).exists()


def test_stop_after_train_cli_and_train_only(tmp_path, capsys):
    file, config = experiment(tmp_path, with_test=False)
    assert main(['run', '--config', str(file), '--stop-after-train', '1']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['status'] == 'paused_after_train_prefix'
    assert output['train_samples'] == 1 and output['test_samples'] == 0


def test_parallel_test_sample_contract_error_is_committed_and_other_tests_continue(tmp_path, monkeypatch, capsys):
    path, config = experiment(tmp_path, scenario='initial_success')
    _test_rows(tmp_path, config, count=3)
    config['run']['test_workers'] = 2
    path.write_text(yaml.safe_dump(config, allow_unicode=True))
    original = FakeLLMBackend.generate
    def invalid(self, role, payload):
        if role == 'initial_generator' and payload['task'] == '测试任务 0':
            raise ValueError('invalid response')
        return original(self, role, payload)
    monkeypatch.setattr(FakeLLMBackend, 'generate', invalid)
    assert main(['run', '--config', str(path)]) == 0
    assert 'recorded, continuing' in capsys.readouterr().err
    from verge.memory import TrajectoryLog
    records = list(TrajectoryLog(tmp_path/'result/artifacts').all())
    assert [r.sample_id for r in records] == ['train-a','train-b','test-0','test-1','test-2']
    assert records[2].state == 'sample_error' and records[2].initial_judgment is None
    assert all(r.initial_judgment.success for r in records[3:])


def test_error_record_is_completed_prefix_and_not_retried_on_resume(tmp_path, monkeypatch, capsys):
    path, config = experiment(tmp_path, with_test=False, scenario='initial_success')
    original = FakeLLMBackend.generate
    def invalid(self, role, payload):
        if role == 'initial_generator':
            raise ValueError('bad R0')
        return original(self, role, payload)
    monkeypatch.setattr(FakeLLMBackend, 'generate', invalid)
    assert main(['run','--config',str(path),'--stop-after-train','1']) == 0
    assert json.loads(capsys.readouterr().out)['train_samples'] == 1
    monkeypatch.setattr(FakeLLMBackend, 'generate', original)
    config['run']['output_dir'] = str(tmp_path/'resumed')
    path.write_text(yaml.safe_dump(config,allow_unicode=True))
    assert main(['run','--config',str(path),'--resume-from',str(tmp_path/'result')]) == 0
    from verge.memory import TrajectoryLog
    records = list(TrajectoryLog(tmp_path/'resumed/artifacts').all())
    assert records[0].state == 'sample_error' and records[1].state == 'initial_success'
    assert [r.sample_id for r in records] == ['train-a','train-b']
