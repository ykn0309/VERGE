import json
from pathlib import Path
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from copy import deepcopy
import httpx, pytest, yaml
from verge.cli import run, components, ConfigError
from verge.llm import OpenAICompatibleBackend, LLMRequestError, LLMSampleError
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import LLMJudge, LLMSelector, LLMCheckers, LLMInitialRubricGenerator
from verge.models import *
from verge.calculator import ProbabilityModel, update_quality, rebuild_quality
from verge.web import load_run, DashboardHandler, discover_runs
from test_v2 import params, experience, app, reports, rubric, sample


def config(tmp_path):
    data = yaml.safe_load(Path("configs/default.yaml").read_text())
    data["run"]["output_dir"] = str(tmp_path / "run")
    return data


def write_config(tmp_path, data):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_cli_fake_train_frozen_and_resume(tmp_path):
    c = config(tmp_path)
    path = write_config(tmp_path, c)
    summary = run(path)
    assert summary["train"] == 1 and summary["test"] == 1
    d = load_run(tmp_path / "run")
    assert d["metrics"]["tasks"] == 2
    assert all(e["q_posterior"] is None for e in d["experiences"])
    assert discover_runs(tmp_path) == ["run"]
    with pytest.raises(ConfigError):
        run(path)


def test_resume_requires_matching_code_and_certified_memory(tmp_path):
    c = config(tmp_path)
    c["dataset"]["config"]["paths"].pop("test")
    path = write_config(tmp_path, c)
    run(path, stop_after_train=1)
    c["run"]["output_dir"] = str(tmp_path / "resumed")
    path = write_config(tmp_path, c)
    assert run(path, resume_from=tmp_path / "run", stop_after_train=1)["train"] == 1
    m = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    m["state"] = "running"
    (tmp_path / "run" / "run_manifest.json").write_text(json.dumps(m))
    c["run"]["output_dir"] = str(tmp_path / "invalid")
    path = write_config(tmp_path, c)
    with pytest.raises(ConfigError, match="uncertified"):
        run(path, resume_from=tmp_path / "run")


def test_old_memory_trajectory_and_config_rejected(tmp_path):
    (tmp_path / "experience_memory.json").write_text('{"schema_version":8}')
    with pytest.raises(ValueError):
        ExperienceMemory(tmp_path)
    with pytest.raises(ValueError):
        TrajectoryRecord.from_dict({"schema_version": 10, "sample_id": "x"})
    c = config(tmp_path)
    c["modules"]["application"]["implementation"] = "immutable_atomic_evolution"
    from verge.config import load_config

    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, c))


def test_http_judge_visibility_and_anonymous_independent_calls(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "not-a-real-key")
    seen = []

    def handler(req):
        b = json.loads(req.content)
        seen.append(b)
        assert "chosen" not in json.dumps(
            b["messages"]
        ) and "rejected" not in json.dumps(b["messages"])
        # The input projection has stable short IDs, and never includes another answer.
        current = b["messages"][-1]["content"]
        assert ("秘密A" in current) ^ ("秘密B" in current)
        result = {
            "criterion_judgments": [
                {"criterion_id": str(i), "satisfied": True, "reason": "相关依据"}
                for i in (1, 2)
            ]
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps(result)},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    client = httpx.Client(
        base_url="https://example.test/v1/", transport=httpx.MockTransport(handler)
    )
    b = OpenAICompatibleBackend(
        "test",
        base_url="https://example.test/v1",
        api_key_env="TEST_API_KEY",
        client=client,
        max_retries=0,
    )
    result = LLMJudge(b, retries=0).judge(rubric(), "秘密A", "秘密B", task="核验内容")
    assert len(seen) == 2 and not result.success
    client.close()


@pytest.mark.parametrize(
    "code,error",
    [
        (401, LLMRequestError),
        (429, LLMSampleError),
        (500, LLMSampleError),
        (400, LLMSampleError),
    ],
)
def test_transport_error_classification_no_key_in_trace(monkeypatch, code, error):
    monkeypatch.setenv("TEST_API_KEY", "secret-key-value")
    events = []
    client = httpx.Client(
        base_url="https://example.test/v1/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(code, json={"error": "safe"})
        ),
    )
    b = OpenAICompatibleBackend(
        "test",
        base_url="https://example.test/v1",
        api_key_env="TEST_API_KEY",
        client=client,
        max_retries=0,
        trace_hook=events.append,
    )
    with pytest.raises(error):
        b.generate("selector", {"task": "x", "candidates": []})
    assert "secret-key-value" not in json.dumps(events)
    client.close()


def test_unknown_checker_is_valid_not_call_failure():
    class B:
        def generate(self, *a, **kw):
            return {"status": "unknown", "evidence": "资料不足", "details": []}

    result = LLMCheckers({s: B() for s in ("selector", "generator", "judge")}).check(
        app()
    )
    assert all(r.status == "unknown" for r in result)


def test_generator_output_cannot_choose_its_own_criterion_ids():
    class B:
        def generate(self, *a, **kw):
            return {
                "analysis": "a",
                "criteria": [
                    {
                        "criterion_id": "1",
                        "question": "q",
                        "weight": 1,
                        "source_experiences": [],
                        "bindings": {},
                        "binding_evidence": {},
                    }
                ]
                * 2,
            }

    from verge.llm import LLMFormatError

    with pytest.raises(LLMFormatError):
        LLMInitialRubricGenerator(B(), retries=0).generate("任务", [])


def test_log_odds_recovers_after_display_probability_rounds_to_one():
    e = experience()
    p = params()
    for n in range(150):
        a = app(e, n)
        a.reports = reports()
        update_quality(e, a, p)
        e.applications.append(a)
    assert e.q_posterior == 1 and e.q_log_odds > 0
    for n in range(150, 700):
        a = app(e, n, states=((False, True), (False, True)))
        a.reports = reports()
        update_quality(e, a, p)
    assert e.q_posterior < 0.01 and e.q_log_odds < 0


def test_sampler_source_ancestry_and_boundary_fallback():
    from verge.pipeline import select_revision_cases

    parent = experience()
    source = app(parent, 0, phase="admission")
    source.sample = PreferenceSample("source", "原任务", "a", "b", {})
    parent.applications = [source]
    child = Experience(
        "child", Method("t", "p"), "source", parent_id=parent.experience_id
    )
    for n in range(7):
        child.applications.append(app(child, n))
    cases, kinds = select_revision_cases(
        child, {parent.experience_id: parent, "child": child}
    )
    assert len(cases) == 6 and kinds["source"] == "source"
    assert len({a.sample.sample_id for a in cases}) == 6


def test_web_translate_and_host_restrictions(tmp_path, monkeypatch):
    c = config(tmp_path)
    run(write_config(tmp_path, c))
    directory = tmp_path / "run"
    monkeypatch.setattr("verge.web._translate_apple", lambda t: "译文:" + t)
    handler = type(
        "TestHandler",
        (DashboardHandler,),
        {"run_dir": directory, "runs_root": tmp_path},
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = HTTPConnection("127.0.0.1", server.server_port)
        try:
            client.request("GET", "/api/run?name=run")
            resp = client.getresponse()
            assert resp.status == 200
            resp.read()
            payload = json.dumps(
                {"run": "run", "sample_id": "demo-train-1", "field": "task"}
            )
            client.request(
                "POST", "/api/translate", payload, {"Content-Type": "application/json"}
            )
            resp = client.getresponse()
            assert resp.status == 200
            assert "译文" in resp.read().decode()
            client.request("GET", "/api/run?name=run", headers={"Host": "evil.test"})
            resp = client.getresponse()
            assert resp.status == 403
            resp.read()
        finally:
            client.close()
            server.shutdown()
            worker.join()


def test_frontend_dom_contract():
    import subprocess

    result = subprocess.run(
        ["node", "tests/test_v2_frontend.cjs"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
