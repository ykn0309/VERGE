"""真实 backend 用 MockTransport 测试；不调用外网、不依赖真实凭据。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from verge.cli import main
from verge.llm import FakeLLMBackend, LLMFormatError, LLMRequestError, OpenAICompatibleBackend
from verge.prompts import PROMPT_VERSION, llm_messages


def result(value, *, status=200, reason="stop", headers=None):
    return httpx.Response(status, headers=headers, json={
        "choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)},
                     "finish_reason": reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    })


def backend(monkeypatch, handler, **options):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-log")
    client = httpx.Client(base_url="https://api.openai.com/v1/",
                          transport=httpx.MockTransport(handler))
    return OpenAICompatibleBackend("mock-model", client=client,
                                   sleep=options.pop("sleep", lambda _: None), **options)


def test_structured_success_metadata_prompt_version_and_privacy(monkeypatch):
    events = []
    observed = []

    def handler(request):
        observed.append(request)
        return result({"feedback": "具体修订建议"}, headers={"x-request-id": "req-123"})

    llm = backend(monkeypatch, handler, metadata_hook=events.append)
    assert llm.generate("critic", {"task": "private task"}) == {"feedback": "具体修订建议"}
    assert observed[0].url.path == "/v1/chat/completions"
    assert observed[0].headers["authorization"] == "Bearer test-secret-never-log"
    sent = json.loads(observed[0].content)
    assert sent["model"] == "mock-model"
    assert sent["response_format"] == {"type": "json_object"}
    assert len(sent["messages"]) == 2 and sent["messages"][1]["content"] == '{"task": "private task"}'
    assert events == [{"role": "critic", "model": "mock-model", "prompt_version": PROMPT_VERSION,
                       "attempt": 1, "status": 200, "request_id": "req-123",
                       "finish_reason": "stop", "usage": {
                           "prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}}]
    assert "test-secret-never-log" not in json.dumps(events)

    # 即使不可信服务把凭据反射进头部/元数据，也不能写入本地 sidecar。
    reflected = []
    echo = backend(monkeypatch, lambda _: result({"feedback": "ok"},
        reason="test-secret-never-log", headers={"x-request-id": "test-secret-never-log"}),
        metadata_hook=reflected.append)
    assert echo.generate("critic", {"task": "x"}) == {"feedback": "ok"}
    assert "test-secret-never-log" not in json.dumps(reflected)


def test_format_retry_on_missing_nested_field_and_limit(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        value = {"criterion_judgments": [{"criterion_id": "c1", "chosen": True}]} if len(calls) == 1 else {
            "criterion_judgments": [{"criterion_id": "c1", "chosen": True, "rejected": False}]}
        return result(value)

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate("judge", {"criteria": [{"criterion_id": "c1"}]})["criterion_judgments"][0]["rejected"] is False
    assert len(calls) == 2 and "上次输出格式无效" in calls[1]["messages"][-1]["content"]

    bad = backend(monkeypatch, lambda _: result({"wrong": 1}), max_retries=1)
    with pytest.raises(LLMFormatError, match="after 2 attempts"):
        bad.generate("critic", {"task": "x"})

    # 多余的模型字段也是格式错误（领域 Model.from_dict 会严格拒绝）。
    extra = backend(monkeypatch, lambda _: result({"candidates": [{
        "candidate_id": "candidate_1", "content": "建议", "trigger": "条件", "credit": 0.9,
    }]}), max_retries=1)
    with pytest.raises(LLMFormatError, match="after 2 attempts"):
        extra.generate("reflector", {"task": "x"})


def test_invalid_json_retries_then_succeeds(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"},
                                                        "finish_reason": "stop"}]})
        return result({"feedback": "修正后的建议"})

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate("critic", {"task": "x"}) == {"feedback": "修正后的建议"}
    assert len(calls) == 2


def test_rate_limit_timeout_and_terminal_errors(monkeypatch):
    waits = []
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="private error body")
        if len(attempts) == 2:
            raise httpx.ReadTimeout("private network details")
        return result({"decision": "FAIL"})

    llm = backend(monkeypatch, handler, max_retries=2, sleep=waits.append)
    assert llm.generate("admission_checker", {"candidate": {}}) == {"decision": "FAIL"}
    assert len(attempts) == 3 and waits == [0.0, 2]
    assert len(attempts) == 3  # FAIL 是语义决策，不重试。

    unauthorized = backend(monkeypatch, lambda _: httpx.Response(401, text="test-secret-never-log"))
    with pytest.raises(LLMRequestError, match="HTTP 401") as exc:
        unauthorized.generate("critic", {"task": "x"})
    assert "test-secret-never-log" not in str(exc.value)

    exhausted = []

    def unavailable(request):
        exhausted.append(request)
        return httpx.Response(503, text="private service body")

    flaky = backend(monkeypatch, unavailable, max_retries=1)
    with pytest.raises(LLMRequestError, match="HTTP 503 after 2 attempts"):
        flaky.generate("critic", {"task": "x"})
    assert len(exhausted) == 2


def test_empty_binding_and_admission_fail_are_not_format_failures(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return result({"bindings": {"candidate_0": []}})

    llm = backend(monkeypatch, handler)
    assert llm.generate("revision_credit_assigner", {"candidates": []}) == {
        "bindings": {"candidate_0": []}}
    assert len(calls) == 1


def test_missing_key_and_unsafe_url_fail_before_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing"):
        OpenAICompatibleBackend("mock-model")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with pytest.raises(ValueError, match="base_url requires HTTPS"):
        OpenAICompatibleBackend("mock-model", base_url="http://remote.example/v1")
    with pytest.raises(ValueError, match="base_url requires HTTPS"):
        OpenAICompatibleBackend("mock-model", base_url="https://user:secret@api.example/v1")
    # 测试注入的 client 也不能绕过目标地址检查或自动转发凭据。
    with httpx.Client(base_url="http://remote.example/v1") as unsafe:
        with pytest.raises(ValueError, match="client must use"):
            OpenAICompatibleBackend("mock-model", client=unsafe)
    with httpx.Client(base_url="https://api.openai.com/v1", follow_redirects=True) as redirecting:
        with pytest.raises(ValueError, match="client must use"):
            OpenAICompatibleBackend("mock-model", client=redirecting)


def test_role_prompts_are_isolated_and_versioned():
    for role in ("selector", "initial_generator"):
        messages = llm_messages(role, {"task": "hello"})
        assert len(messages) == 2
        assert "chosen" not in json.dumps(messages) and "rejected" not in json.dumps(messages)
        assert role in messages[0]["content"] and "JSON" in messages[0]["content"]
    with pytest.raises(ValueError, match="unknown LLM role"):
        llm_messages("not_a_role", {})


def test_cli_role_specific_backend_and_missing_key(tmp_path, monkeypatch, capsys):
    default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(default.read_text(encoding="utf-8"))
    config["run"]["output_dir"] = str(tmp_path / "missing-key")
    config["models"]["roles"]["judge"] = {"backend": "openai_compatible", "model": "judge-model"}
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["run", "--config", str(file)]) == 2
    assert not (tmp_path / "missing-key").exists()
    assert "models.roles.judge" in capsys.readouterr().err
    config["models"]["roles"]["judge"]["api_key"] = "should-never-be-written"
    file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 2
    assert "should-never-be-written" not in capsys.readouterr().err
    assert not (tmp_path / "missing-key").exists()
    del config["models"]["roles"]["judge"]["api_key"]

    fake = FakeLLMBackend("initial_success")

    def handler(request):
        data = json.loads(request.content)
        assert data["model"] == "judge-model"
        return result(fake.generate("judge", json.loads(data["messages"][1]["content"])))

    mock_client = httpx.Client(base_url="https://api.openai.com/v1/",
                               transport=httpx.MockTransport(handler))
    monkeypatch.setattr("verge.llm.httpx.Client", lambda **kwargs: mock_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-log")
    config["run"]["output_dir"] = str(tmp_path / "role-run")
    config["dataset"]["config"]["paths"] = {
        "train": str(Path(__file__).resolve().parents[1] / "configs/data/demo_train.jsonl")}
    file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 0, capsys.readouterr().err
    trajectory = json.loads((tmp_path / "role-run/artifacts/trajectories.jsonl").read_text(encoding="utf-8"))
    assert trajectory["state"] == "initial_success"
    events = (tmp_path / "role-run/artifacts/llm_requests.jsonl").read_text(encoding="utf-8")
    assert {json.loads(line)["role"] for line in events.splitlines()} == {"judge"}


def test_mocked_real_backend_completes_cli_trajectory_without_pipeline_changes(tmp_path, monkeypatch,
                                                                                     capsys):
    # HTTP transport 模拟真实 Chat Completions JSON 接口，语义结果借用固定场景。
    fake = FakeLLMBackend("attempt_1_success")
    sent = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-secret-never-log"
        body = json.loads(request.content)
        role = body["messages"][0]["content"].split("当前角色：", 1)[1].split("。", 1)[0]
        payload = json.loads(body["messages"][1]["content"])
        sent.append((role, payload))
        return result(fake.generate(role, payload), headers={"x-request-id": f"req-{len(sent)}"})

    mock_client = httpx.Client(base_url="https://api.openai.com/v1/",
                               transport=httpx.MockTransport(handler))
    monkeypatch.setattr("verge.llm.httpx.Client", lambda **kwargs: mock_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-log")
    demo = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(demo.read_text(encoding="utf-8"))
    config["run"]["output_dir"] = str(tmp_path / "run")
    config["dataset"]["config"]["paths"] = {
        "train": str(Path(__file__).resolve().parents[1] / "configs/data/demo_train.jsonl")}
    config["models"]["default"] = {"backend": "openai_compatible", "model": "mock-model"}
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 0, capsys.readouterr().err
    record_file = tmp_path / "run" / "artifacts" / "trajectories.jsonl"
    trajectory = json.loads(record_file.read_text(encoding="utf-8").strip())
    assert trajectory["state"] == "revision_success_1"
    assert trajectory["reflected_candidates"][0]["admission"] == "PASS"
    assert trajectory["reflected_candidates"][0]["fusion"] == "ADD"
    metadata = (tmp_path / "run" / "artifacts" / "llm_requests.jsonl").read_text(encoding="utf-8")
    assert "test-secret-never-log" not in metadata
    requests = [json.loads(line) for line in metadata.splitlines()]
    assert requests and all(row["sample_id"] == "demo-train-1" for row in requests)
    assert {row["role"] for row in requests} >= {"initial_generator", "judge", "critic",
                                                  "revision_generator", "reflector"}
    assert "test-secret-never-log" not in record_file.read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt_versions"]["judge"] == PROMPT_VERSION
    assert "test-secret-never-log" not in json.dumps(manifest)
