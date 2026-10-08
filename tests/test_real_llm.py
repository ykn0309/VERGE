"""真实 backend 用 MockTransport 测试；不调用外网、不依赖真实凭据。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from verge.cli import main
from verge.llm import JUDGE_RESPONSE_SIDE, FakeLLMBackend, LLMFormatError, LLMRequestError, OpenAICompatibleBackend
from verge.prompts import PROMPT_VERSION, _ROLE_GUIDANCE, _ROLE_INSTRUCTIONS, llm_messages


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
        result_value = {"analysis": "根据任务拆解核心要求。", "criteria": [{k: v for k, v in item.items() if k != "criterion_id"}
                                     for item in result_value["rubric"]["criteria"]]}
    elif role == "reflector":
        for item in result_value["candidates"]:
            item.pop("candidate_id")
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
        return result({"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}, headers={"x-request-id": "req-123"})

    llm = backend(monkeypatch, handler, metadata_hook=events.append)
    assert llm.generate("critic", {"task": "private task"}) == {"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}
    assert observed[0].url.path == "/v1/chat/completions"
    assert observed[0].headers["authorization"] == "Bearer test-secret-never-log"
    sent = json.loads(observed[0].content)
    assert sent["model"] == "mock-model"
    assert sent["response_format"] == {"type": "json_object"}
    assert len(sent["messages"]) == 2
    assert "任务：" in sent["messages"][1]["content"]
    assert "private task" in sent["messages"][1]["content"]
    assert events == [{"role": "critic", "model": "mock-model", "prompt_version": PROMPT_VERSION,
                       "attempt": 1, "status": 200, "request_id": "req-123",
                       "finish_reason": "stop", "usage": {
                           "prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}}]
    assert "test-secret-never-log" not in json.dumps(events)

    # 即使不可信服务把凭据反射进头部/元数据，也不能写入本地 sidecar。
    reflected = []
    echo = backend(monkeypatch, lambda _: result({"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"},
        reason="test-secret-never-log", headers={"x-request-id": "test-secret-never-log"}),
        metadata_hook=reflected.append)
    assert echo.generate("critic", {"task": "x"}) == {"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}
    assert "test-secret-never-log" not in json.dumps(reflected)


@pytest.mark.parametrize("role, output, expected", [
    ("selector", {"selected_experience_ids": [1, 2]},
     {"selected_experience_ids": ["1", "2"]}),
    ("initial_generator", {"analysis": "任务分析", "criteria": [{"question": "准则？", "importance_score": 8,
                                        "source_experiences": [3]}]},
     {"analysis": "任务分析", "criteria": [{"question": "准则？", "importance_score": 8,
                    "source_experiences": ["3"]}]}),
    ("judge", {"criterion_judgments": [{"criterion_id": 1, "satisfied": True}]},
     {"criterion_judgments": [{"criterion_id": "1", "satisfied": True}]}),
    ("reflector", {"candidates": [{"content": "经验", "trigger": "条件"}]},
     {"candidates": [{"content": "经验", "trigger": "条件"}]}),
])
def test_model_can_return_positive_integer_ids_for_any_role(monkeypatch, role, output, expected):
    llm = backend(monkeypatch, lambda _: result(output), max_retries=0)
    assert llm.generate(role, {"task": "x"}) == expected




def test_format_retry_on_missing_nested_field_and_limit(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        value = {"criterion_judgments": [{"criterion_id": "c1"}]} if len(calls) == 1 else {
            "criterion_judgments": [{"criterion_id": "c1", "satisfied": True}]}
        return result(value)

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate("judge", {"criteria": [{"criterion_id": "c1"}]})["criterion_judgments"][0]["satisfied"] is True
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
        return result({"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"})

    llm = backend(monkeypatch, handler, max_retries=1)
    assert llm.generate("critic", {"task": "x"}) == {"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}
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
        return result({"candidates": [{"content": "方法", "trigger": "条件"}]})

    llm = backend(monkeypatch, handler)
    assert llm.generate("reflector", {"task": "任务"}) == {
        "candidates": [{"content": "方法", "trigger": "条件"}]}
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
        backend(monkeypatch, lambda _: result({"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}), enable_thinking="false")


def test_role_prompts_are_isolated_and_versioned():
    for role in ("selector", "initial_generator", "judge", "critic",
                 "reflector", "admission_checker", "fusion_judge"):
        messages = llm_messages(role, {"task": "hello"})
        assert len(messages) == 2
        if role in ("selector", "initial_generator"):
            assert "chosen" not in json.dumps(messages) and "rejected" not in json.dumps(messages)
        assert messages[0]["content"].startswith(_ROLE_GUIDANCE[role]) and "JSON" in messages[0]["content"]
        assert messages[0]["content"] == _ROLE_GUIDANCE[role] + "\n\n" + _ROLE_INSTRUCTIONS[role]
        if role not in ("selector", "judge"):
            assert "中文" in _ROLE_INSTRUCTIONS[role]
        assert "【输出协议】" not in messages[0]["content"]
    with pytest.raises(ValueError, match="unknown LLM role"):
        llm_messages("not_a_role", {})


def test_admission_selection_and_provenance_prompt_requirements():
    assert PROMPT_VERSION == "generator-dialogue-program-owned-revision-sources"
    selector = llm_messages("selector", {"task": "task"})[0]["content"]
    assert all(text in selector for text in ('允许合理推理', '选择数量为 0 到 max_selected',
                                               '高置信度不能代替适用性'))
    reflector = llm_messages("reflector", {"task": "task"})[0]["content"]
    admission = llm_messages("admission_checker", {"task": "task"})[0]["content"]
    assert "生成标准之前可识别的适用场景" in reflector
    assert "trigger 不应以已经观察到这些性质为前提" in admission
    assert '"reason"' in admission
    critic = llm_messages("critic", {"task": "task"})[0]["content"]
    assert "improvement_advice" in critic
    assert "如果修订轮次为 2" in critic
    assert "analysis" in critic and "严禁引用" in critic
    generator = llm_messages("initial_generator", {"task": "task"})[0]["content"]
    assert "仅由 task 推导的标准使用空数组" in generator
    assert "实质新增或改变该标准的所选经验 ID" in generator


@pytest.mark.parametrize("role,output", [
    ("critic", {"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}),
    ("initial_generator", {"analysis": "任务分析", "criteria": [{"question": "Be clear?", "importance_score": 4,
                                         "source_experiences": []}]}),

    ("reflector", {"candidates": [{"content": "Be clear",
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
    language_rule = {
        "critic": "两个字段均为非空中文文本",
        "initial_generator": "analysis 和 question 用中文",
        "reflector": "content 和 trigger 为非空中文文本",
    }[role]
    assert language_rule in requests[0]["messages"][0]["content"]


def test_cli_role_specific_backend_and_missing_key(tmp_path, monkeypatch, capsys, wire_payload):
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
        return result(numbered_fake_response(fake, "judge", wire_payload(data)))

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


def test_qwen_config_sends_non_thinking_dashscope_request(tmp_path, monkeypatch, capsys, wire_payload):
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
        role = next(role for role, guide in _ROLE_GUIDANCE.items()
                    if body["messages"][0]["content"].startswith(guide))
        return result(numbered_fake_response(fake, role, wire_payload(body)))

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
                                                                                     capsys, wire_payload):
    # HTTP transport 模拟真实 Chat Completions JSON 接口，语义结果借用固定场景。
    fake = FakeLLMBackend("attempt_1_success")
    sent = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-secret-never-log"
        body = json.loads(request.content)
        role = next(role for role, guide in _ROLE_GUIDANCE.items()
                    if body["messages"][0]["content"].startswith(guide))
        payload = wire_payload(body)
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
    assert all(set(event["input"]) == {"task", "criteria", "response"}
               for event in role_events if event["role"] == "judge")
    assert {event["response_side"] for event in role_events if event["role"] == "judge"} == {"chosen", "rejected"}
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
        return result({"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"})

    llm = backend(monkeypatch, handler, max_retries=1, trace_hook=trace.append)
    assert llm.generate("critic", {"task": "private task"}) == {"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}
    assert len(trace) == 2
    assert trace[0]["input"] == {"task": "private task"}
    assert trace[0]["messages"] == calls[0]["messages"]
    assert trace[0]["raw_output"] == "not JSON" and trace[0]["error"] == "invalid structured output"
    assert trace[1]["messages"] == calls[1]["messages"]
    assert "上次输出格式无效" in trace[1]["messages"][-1]["content"]
    assert json.loads(trace[1]['raw_output']) == trace[1]['output']
    assert trace[1]["output"] == {"analysis": "诊断覆盖不足", "improvement_advice": "按任务要求完善标准"}
    assert "test-secret-never-log" not in json.dumps(trace, ensure_ascii=False)


def test_role_trace_does_not_save_http_error_bodies_or_headers(monkeypatch):
    trace = []
    llm = backend(monkeypatch, lambda _: httpx.Response(401, text="secret error body",
                   headers={"x-request-id": "private-id"}), trace_hook=trace.append)
    with pytest.raises(LLMRequestError, match="HTTP 401"):
        llm.generate("critic", {"task": "x"})
    assert trace[0]["raw_output"] is None and trace[0]["error"] == "HTTP 401"
    assert "secret error body" not in json.dumps(trace) and "private-id" not in json.dumps(trace)


def test_critic_before_after_schema_failure_is_logged_and_not_fatal(tmp_path, monkeypatch, wire_payload):
    from verge.memory import ExperienceMemory, TrajectoryLog
    from verge.models import PreferenceSample
    from verge.modules import make_fake_modules
    from verge.pipeline import PipelineRunner

    calls = []

    def handler(request):
        payload = wire_payload(json.loads(request.content))
        calls.append(payload)
        # Reproduce the real failure: MODIFY using before/after instead of required schema.
        return result({"analysis": "诊断", "operations": []})

    critic_backend = backend(monkeypatch, handler)
    modules = make_fake_modules('attempt_1_success')
    modules.critic.backend = critic_backend
    memory, log = ExperienceMemory(tmp_path), TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, memory, log)
    record = runner.run_sample(PreferenceSample('schema', '任务', 'chosen', 'rejected', {}))
    assert record.state == 'final_failure' and len(record.invalid_revision_attempts) == 2
    assert len(calls) == 6  # 每轮仅 backend 的首次 + 2 次重试。
    assert calls[3]['previous_invalid_attempt']['analysis'] is None
    assert not record.revision_attempts and not record.reflected_candidates
    assert memory.all() == [] and log.get('schema') == record
    critic_backend._client.close()


@pytest.mark.parametrize('role, invalid, valid', [
    ('initial_generator', {"analysis": "任务分析", 'criteria': [{'criterion_id': '1', 'question': '问题',
                                         'importance_score': 5, 'source_experiences': []}]},
     {"analysis": "任务分析", 'criteria': [{'question': '问题', 'importance_score': 5, 'source_experiences': []}]}),
    ('reflector', {'candidates': [{'candidate_id': '1', 'content': '原则', 'trigger': '条件'}]},
     {'candidates': [{'content': '原则', 'trigger': '条件'}]}),
])
def test_backend_retries_generated_id_fields_and_returns_content_only(monkeypatch, role, invalid, valid):
    responses = iter([invalid, valid])
    llm = backend(monkeypatch, lambda _: result(next(responses)), max_retries=1)
    assert llm.generate(role, {'task': '任务'}) == valid
