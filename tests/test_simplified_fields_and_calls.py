"""Current schema drops redundant fields; call validation remains auditable."""

import json
from pathlib import Path
import pytest
from verge.models import Method, Parameter, Criterion
from verge.modules import LLMInitialRubricGenerator
from verge.llm import FakeLLMBackend
from verge.web import load_role_io, load_run
from test_v2_io import config, write_config
from verge.cli import run


def test_removed_fields_are_rejected_and_description_remains():
    assert Parameter("数量").to_dict() == {"description": "数量"}
    with pytest.raises(ValueError, match="unknown"):
        Parameter.from_dict({"description": "数量", "source": "task"})
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
            result["criteria"][0]["bindings"] = "bad"
            return result

    b = Bad()
    events = []
    b.trace_hook = events.append
    from verge.llm import LLMFormatError

    with pytest.raises(LLMFormatError, match="bindings: expected a dictionary"):
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
    assert "bindings" in rows[0]["contract_error"]


def test_historical_v10_display_preserves_old_data_without_current_decoder():
    directory = Path("runs/helpsteer3-general-v2-prompt-v10-qwen-flash-30")
    if not directory.exists():
        pytest.skip("local historical experiment absent")
    data = load_run(directory)
    rows = load_role_io(directory, data["samples"][0]["sample_id"])["entries"]
    failures = [r for r in rows if r.get("validation_status") == "invalid"]
    assert len(data["samples"]) == 30
    assert failures and any("dictionary" in r["contract_error"] for r in failures)
