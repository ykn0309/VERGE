"""Current schema drops redundant fields; call validation remains auditable."""

import json
from pathlib import Path
import pytest
from verge.models import Method, Criterion
from verge.modules import LLMInitialRubricGenerator
from verge.llm import FakeLLMBackend
from verge.web import load_role_io, load_run
from test_v2_io import config, write_config
from verge.cli import run


def test_removed_fields_are_rejected_and_two_field_method_remains():
    assert Method("任务", "核验动作").to_dict() == {
        "trigger": "任务",
        "principle": "核验动作",
    }
    with pytest.raises(ValueError, match="unknown"):
        Method.from_dict({"trigger": "任务", "principle": "方法", "parameters": {}})
    with pytest.raises(ValueError, match="unknown"):
        Criterion.from_dict(
            {
                "criterion_id": "1",
                "question": "符合？",
                "weight": 1,
                "binding_evidence": {},
            }
        )


def test_contract_validation_log_is_linked_and_field_error_is_specific(tmp_path):
    class Bad(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            result["criteria"][0]["weight"] = "bad"
            return result

    b = Bad()
    events = []
    b.trace_hook = events.append
    from verge.llm import LLMFormatError

    with pytest.raises(LLMFormatError, match="weight: expected"):
        LLMInitialRubricGenerator(b, retries=0).generate("task", [])
    validation = events[-1]
    assert validation["event_type"] == "contract_validation"
    assert validation["validation_status"] == "invalid"
    assert validation["call_id"] == events[0]["call_id"]
    c = config(tmp_path)
    run(write_config(tmp_path, c))
    directory = tmp_path / "run"
    data = load_run(directory)
    sid = data["samples"][0]["sample_id"]
    path = directory / "artifacts/role_io.jsonl"
    path.write_text("".join(json.dumps({**e, "sample_id": sid}) + "\n" for e in events))
    rows = load_role_io(directory, sid)["entries"]
    assert len(rows) == 1 and rows[0]["validation_status"] == "invalid"
    assert "weight" in rows[0]["contract_error"]


def test_historical_template_run_display_without_current_decoder(tmp_path):
    directory = tmp_path / "run"
    run(write_config(tmp_path, config(tmp_path)))
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 15
    manifest_path.write_text(json.dumps(manifest))
    memory_path = directory / "artifacts/experience_memory.json"
    memory = json.loads(memory_path.read_text())
    memory["schema_version"] = 13
    for e in memory["experiences"]:
        e["method"].update(template=None, parameters={})
    memory_path.write_text(json.dumps(memory))
    path = directory / "artifacts/trajectories.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for r in records:
        r["schema_version"] = 15
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    data = load_run(directory)
    assert len(data["samples"]) == 2
    assert "template" in data["experiences"][0]["method"]
    from verge.memory import ExperienceMemory

    with pytest.raises(ValueError, match="schema"):
        ExperienceMemory(directory / "artifacts")
