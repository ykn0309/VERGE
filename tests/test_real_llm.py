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


def numbered_fake_response(fake, role, payload):
    """MockTransport 的模型桩遵循 v9 数字协议；Fake 本体仍用于内部 ID 测试。"""
    if role == "judge":
        internal = {**payload, "criteria": [
            {**item, "criterion_id": ("c2" if item["question"] == "是否逐项满足关键要求？"
                                      else "c" + item["criterion_id"])}
            for item in payload["criteria"]]}
        result_value = fake.generate(role, internal)
        for answer, requested in zip(result_value["criterion_judgments"], payload["criteria"]):
            answer["criterion_id"] = requested["criterion_id"]
        return result_value
    result_value = fake.generate(role, payload)
    if role == "initial_generator":
        result_value["rubric"]["rubric_id"] = "1"
        for index, item in enumerate(result_value["rubric"]["criteria"], 1):
            item["criterion_id"] = str(index)
    elif role == "critic":
        for index, item in enumerate(result_value["operations"], 1):
            item["operation_id"] = str(index)
            item["target_criterion_id"] = "2"
            item["new_or_updated_criterion"]["criterion_id"] = "2"
    elif role == "reflector":
        for index, item in enumerate(result_value["candidates"], 1):
            item["candidate_id"] = str(index)
    return result_value


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
        return result({"analysis": "具体修订建议", "operations": []}, headers={"x-request-id": "req-123"})

    llm = backend(monkeypatch, handler, metadata_hook=events.append)
    assert llm.generate("critic", {"task": "private task"}) == {"analysis": "具体修订建议", "operations": []}
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
    echo = backend(monkeypatch, lambda _: result({"analysis": "好的", "operations": []},
        reason="test-secret-never-log", headers={"x-request-id": "test-secret-never-log"}),
        metadata_hook=reflected.append)
    assert echo.generate("critic", {"task": "x"}) == {"analysis": "好的", "operations": []}
    assert "test-secret-never-log" not in json.dumps(reflected)


@pytest.mark.parametrize("role, output, expected", [
    ("selector", {"selected_experience_ids": [1, 2]},
     {"selected_experience_ids": ["1", "2"]}),
    ("initial_generator", {"rubric": {"rubric_id": 1, "base_rubric_id": None,
         "criteria": [{"criterion_id": 2, "question": "准则？", "importance_score": 8,
                       "source_experiences": [3]}]}},
     {"rubric": {"rubric_id": "1", "base_rubric_id": None,
         "criteria": [{"criterion_id": "2", "question": "准则？", "importance_score": 8,
                       "source_experiences": ["3"]}]}}),
    ("judge", {"criterion_judgments": [{"criterion_id": 1, "chosen": True, "rejected": False}]},
     {"criterion_judgments": [{"criterion_id": "1", "chosen": True, "rejected": False}]}),
    ("reflector", {"candidates": [{"candidate_id": 1, "content": "经验", "trigger": "条件"}]},
     {"candidates": [{"candidate_id": "1", "content": "经验", "trigger": "条件"}]}),
    ("revision_credit_assigner", {"bindings": {"1": [2]}},
     {"bindings": {"1": ["2"]}}),
])
def test_model_can_return_positive_integer_ids_for_any_role(monkeypatch, role, output, expected):
    llm = backend(monkeypatch, lambda _: result(output), max_retries=0)
    assert llm.generate(role, {"task": "x"}) == expected


def test_critic_accepts_integer_operation_id_but_not_boolean_or_invalid_experience(monkeypatch):
    operation = {
        "operation_id": 1, "operation_type": "ADD", "target_criterion_id": None,
        "new_or_updated_criterion": {
            "criterion_id": "5", "question": "明确说明推导过程？",
            "importance_score": 8, "source_experiences": [],
        },
    }
    traces = []
    llm = backend(monkeypatch, lambda _: result({"analysis": "加强推导", "operations": [operation]}),
                  max_retries=0, trace_hook=traces.append)
    parsed = llm.generate("critic", {"task": "证明"})
    assert parsed["operations"][0]["operation_id"] == "1"
    assert json.loads(traces[0]["raw_output"])["operations"][0]["operation_id"] == 1

    for broken in ({**operation, "operation_id": True},
                   {**operation, "new_or_updated_criterion": {
                       **operation["new_or_updated_criterion"], "source_experiences": [False]}}):
        invalid = backend(monkeypatch, lambda _: result({"analysis": "加强推导", "operations": [broken]}),
                          max_retries=0)
        with pytest.raises(LLMFormatError):
            invalid.generate("critic", {"task": "证明"})


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
        return result({"analysis": "修正后的建议", "operations": []})

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate("critic", {"task": "x"}) == {"analysis": "修正后的建议", "operations": []}
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
        return result({"decision": "FAIL", "reason": "触发条件依赖回答内容。"})

    llm = backend(monkeypatch, handler, max_retries=2, sleep=waits.append)
    assert llm.generate("admission_checker", {"candidate": {}}) == {
        "decision": "FAIL", "reason": "触发条件依赖回答内容。"}
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


def test_admission_requires_nonempty_reason_and_retries_format(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return result({"decision": "PASS"})
        if len(calls) == 2:
            return result({"decision": "PASS", "reason": "  "})
        return result({"decision": "PASS", "reason": "绑定操作支撑且触发条件可核验。"})

    llm = backend(monkeypatch, handler, max_retries=2)
    assert llm.generate("admission_checker", {"candidate": {}}) == {
        "decision": "PASS", "reason": "绑定操作支撑且触发条件可核验。"}
    assert len(calls) == 3


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
    with pytest.raises(ValueError, match="enable_thinking must be Boolean"):
        backend(monkeypatch, lambda _: result({"analysis": "ok", "operations": []}), enable_thinking="false")


def test_role_prompts_are_isolated_and_versioned():
    for role in ("selector", "initial_generator", "judge", "critic",
                 "reflector", "revision_credit_assigner", "admission_checker", "fusion_judge"):
        messages = llm_messages(role, {"task": "hello"})
        assert len(messages) == 2
        if role in ("selector", "initial_generator"):
            assert "chosen" not in json.dumps(messages) and "rejected" not in json.dumps(messages)
        assert role in messages[0]["content"] and "JSON" in messages[0]["content"]
        assert "必须以中文表达" in messages[0]["content"]
        assert "机器协议值保持规定形式" in messages[0]["content"]
    with pytest.raises(ValueError, match="unknown LLM role"):
        llm_messages("not_a_role", {})


def test_admission_selection_and_provenance_prompt_requirements():
    assert PROMPT_VERSION == "v10-critic-atomic"
    selector = llm_messages("selector", {"task": "task"})[0]["content"]
    assert all(text in selector for text in ("task 本身核验", "高信度不能覆盖不匹配", "可选零条"))
    reflector = llm_messages("reflector", {"task": "task"})[0]["content"]
    admission = llm_messages("admission_checker", {"task": "task"})[0]["content"]
    assert "无法写出明确的事前适用条件时不输出" in reflector
    assert "即使混有 task 条件也 FAIL" in admission
    assert "不改写 trigger" in admission
    assert '"reason"' in admission
    critic = llm_messages("critic", {"task": "task"})[0]["content"]
    assert "第一次操作不会自动继承" in critic
    assert "针对 R0 重新提出可独立执行的完整操作集" in critic
    assert "analysis" in critic and "operations" in critic
    generator = llm_messages("initial_generator", {"task": "task"})[0]["content"]
    assert "source_experiences 仍为 []" in generator
    assert "未使用的经验不得挂名" in generator


@pytest.mark.parametrize("role,output", [
    ("critic", {"analysis": "Identify missing criteria.", "operations": []}),
    ("initial_generator", {"rubric": {"rubric_id": "r0", "base_rubric_id": None,
        "criteria": [{"criterion_id": "c1", "question": "Be clear?", "importance_score": 4,
                      "source_experiences": []}]}}),

    ("reflector", {"candidates": [{"candidate_id": "e1", "content": "Be clear",
                                       "trigger": "Explanatory task"}]}),
])
def test_backend_does_not_gate_text_by_language(monkeypatch, role, output):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return result(output)

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate(role, {"task": "private task"}) == output
    assert len(requests) == 1  # 提示词要求中文，但合法结构不会因文本语言触发重试。
    assert "必须以中文表达" in requests[0]["messages"][0]["content"]


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
        return result(numbered_fake_response(fake, "judge", json.loads(data["messages"][1]["content"])))

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


def test_qwen_config_sends_non_thinking_dashscope_request(tmp_path, monkeypatch, capsys):
    config_file = Path(__file__).resolve().parents[1] / "configs/qwen3.5-flash.yaml"
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    spec = config["models"]["default"]
    assert spec["model"] == "qwen3.5-flash"
    assert spec["api_key_env"] == "DASHSCOPE_API_KEY_2"
    assert spec["enable_thinking"] is False
    config["run"]["output_dir"] = str(tmp_path / "qwen-run")
    config["dataset"]["config"]["paths"].pop("test")
    file = tmp_path / "qwen.yaml"
    file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    fake = FakeLLMBackend("initial_success")
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append((request.url, body))
        role = body["messages"][0]["content"].split("当前角色：", 1)[1].split("。", 1)[0]
        return result(numbered_fake_response(fake, role, json.loads(body["messages"][1]["content"])))

    mock_client = httpx.Client(base_url=spec["base_url"] + "/",
                               transport=httpx.MockTransport(handler))
    monkeypatch.setattr("verge.llm.httpx.Client", lambda **kwargs: mock_client)
    monkeypatch.setenv("DASHSCOPE_API_KEY_2", "test-secret-never-log")
    assert main(["run", "--config", str(file)]) == 0, capsys.readouterr().err
    assert sent and all(url.path == "/compatible-mode/v1/chat/completions" for url, _ in sent)
    assert all(body["model"] == "qwen3.5-flash" and body["enable_thinking"] is False
               and body["response_format"] == {"type": "json_object"} for _, body in sent)
    assert "test-secret-never-log" not in (tmp_path / "qwen-run/run_manifest.json").read_text()


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
        return result(numbered_fake_response(fake, role, payload), headers={"x-request-id": f"req-{len(sent)}"})

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
    assert trajectory["reflected_candidates"][0]["admission_reason"]
    assert trajectory["reflected_candidates"][0]["fusion"] == "ADD"
    metadata = (tmp_path / "run" / "artifacts" / "llm_requests.jsonl").read_text(encoding="utf-8")
    assert "test-secret-never-log" not in metadata
    requests = [json.loads(line) for line in metadata.splitlines()]
    assert requests and all(row["sample_id"] == "demo-train-1" for row in requests)
    assert {row["role"] for row in requests} >= {"initial_generator", "judge", "critic", "reflector"}
    assert "revision_generator" not in {row["role"] for row in requests}
    assert "test-secret-never-log" not in record_file.read_text(encoding="utf-8")
    role_path = tmp_path / "run" / "artifacts" / "role_io.jsonl"
    role_events = [json.loads(line) for line in role_path.read_text(encoding="utf-8").splitlines()]
    assert role_events and all(event["sample_id"] == "demo-train-1" for event in role_events)
    assert "test-secret-never-log" not in role_path.read_text(encoding="utf-8")
    assert role_path.stat().st_mode & 0o077 == 0
    assert role_events[0]["messages"][0]["role"] == "system"
    assert role_events[0]["raw_output"] and role_events[0]["output"]
    assert "chosen" not in role_events[0]["input"] and "rejected" not in role_events[0]["input"]
    assert any("chosen" in event["input"] for event in role_events if event["role"] == "judge")
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt_versions"]["judge"] == PROMPT_VERSION
    assert "test-secret-never-log" not in json.dumps(manifest)


def test_role_trace_preserves_actual_prompt_response_and_format_retry(monkeypatch):
    trace = []
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"},
                                                          "finish_reason": "stop"}]})
        return result({"analysis": "请调整判定标准", "operations": []})

    llm = backend(monkeypatch, handler, max_retries=1, trace_hook=trace.append)
    assert llm.generate("critic", {"task": "private task"}) == {"analysis": "请调整判定标准", "operations": []}
    assert len(trace) == 2
    assert trace[0]["input"] == {"task": "private task"}
    assert trace[0]["messages"] == calls[0]["messages"]
    assert trace[0]["raw_output"] == "not JSON" and trace[0]["error"] == "invalid structured output"
    assert trace[1]["messages"] == calls[1]["messages"]
    assert "上次输出格式无效" in trace[1]["messages"][-1]["content"]
    assert trace[1]["raw_output"] == '{"analysis": "请调整判定标准", "operations": []}'
    assert trace[1]["output"] == {"analysis": "请调整判定标准", "operations": []}
    assert "test-secret-never-log" not in json.dumps(trace, ensure_ascii=False)


def test_role_trace_does_not_save_http_error_bodies_or_headers(monkeypatch):
    trace = []
    llm = backend(monkeypatch, lambda _: httpx.Response(401, text="secret error body",
                   headers={"x-request-id": "private-id"}), trace_hook=trace.append)
    with pytest.raises(LLMRequestError, match="HTTP 401"):
        llm.generate("critic", {"task": "x"})
    assert trace[0]["raw_output"] is None and trace[0]["error"] == "HTTP 401"
    assert "secret error body" not in json.dumps(trace) and "private-id" not in json.dumps(trace)
