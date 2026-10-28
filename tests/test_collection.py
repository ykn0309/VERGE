from copy import deepcopy
from dataclasses import replace
import json
import pytest
from test_v2 import experience, sample
from test_v2_io import config, write_config
from verge.cli import run
from verge.config import load_config
from verge.llm import FakeLLMBackend
from verge.modules import LLMSelector
from verge.memory import ExperienceMemory, TrajectoryLog
from scripts.prepare_general_collection import stratified_general


def test_selector_collection_never_exposes_quality_or_usage():
    b = FakeLLMBackend()
    traces = []
    b.trace_hook = traces.append
    e = experience()
    e.q_prior = 0.6
    e.q_posterior = 0.7
    e.parameter_version = "private"
    e.parameter_digest = "private-hash"
    LLMSelector(b, include_quality=False).select("任务", [e])
    candidate = [t for t in traces if "input" in t][-1]["input"]["candidates"][0]
    assert set(candidate) == {
        "experience_id",
        "trigger",
        "principle",
    }
    LLMSelector(b).select("任务", [e])
    assert (
        "q_posterior"
        in [t for t in traces if "input" in t][-1]["input"]["candidates"][0]
    )


def collection_config(tmp_path):
    c = config(tmp_path)
    c["run"]["mode"] = "collection"
    c["dataset"]["config"]["paths"].pop("test")
    c["modules"]["application"]["config"]["revision_enabled"] = False
    return c


@pytest.mark.parametrize("invalid", ["parameters", "revision", "held_out"])
def test_collection_incompatible_options_rejected(tmp_path, invalid):
    c = collection_config(tmp_path)
    if invalid == "parameters":
        c["modules"]["reliability"]["config"]["parameter_file"] = "unused.json"
    elif invalid == "revision":
        c["modules"]["application"]["config"]["revision_enabled"] = True
    else:
        c["dataset"]["config"]["paths"]["test"] = "never-open.jsonl"
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, c))


def test_collection_fake_records_without_q_or_revisions(tmp_path):
    c = collection_config(tmp_path)
    result = run(write_config(tmp_path, c))
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert (
        manifest["run_mode"] == "collection"
        and manifest["state"] == "training_completed"
    )
    m = ExperienceMemory(tmp_path / "run/artifacts", read_only=True)
    assert not m.revisions and all(e.q_posterior is None for e in m.all())
    assert result["test"] == 0


def test_general_stratification_deterministic_proportional_and_unique():
    population = []
    for n in range(60):
        s = sample(n)
        population.append(
            replace(
                s,
                task="x" * (n + 1),
                metadata={
                    "domain": "general",
                    "overall_preference": 2 if n % 3 else -3,
                },
            )
        )
    first, summary = stratified_general(population, 30, 42)
    second, again = stratified_general(list(reversed(population)), 30, 42)
    assert summary == again and first == second
    assert len({s.task_id for s in first}) == 30
    assert sum(summary["sample_counts"].values()) == 30
    for key, n in summary["population_counts"].items():
        assert abs(summary["sample_counts"][key] - 30 * n / 60) < 1
    with pytest.raises(ValueError):
        stratified_general(
            [
                replace(
                    population[0], metadata={"domain": "code", "overall_preference": 2}
                )
            ],
            1,
        )
