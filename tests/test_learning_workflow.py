"""v3.1 control flow, information boundaries, persistence, and frozen inference."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
import json
import pytest
import yaml

from verge.calculator import calculate_rubric_judgment
from verge.cli import Components, run, evaluate, execution_protocol
from verge.clustering import OnlineClustering
from verge.config import load_config
from verge.llm import FakeEmbedding, FakeLLMBackend, LLMSampleError, LLMFormatError
from verge.memory import ExperienceMemory, downstream_feedback
from verge.models import PreferenceSample, Rubric, Criterion, CriterionJudgment
from verge.modules import LLMSelector, LLMGenerator, LLMJudge, LLMCritic, RoleModules
from verge.pipeline import PipelineRunner, preference_evidence
from verge.scheduling import run_training_stream


def sample(i=0, metadata=None):
    return PreferenceSample(
        f"s{i}", f"task {i}", "preferred answer", "other answer", metadata or {}
    )


def runner(backend=None, selected=3):
    b = backend or FakeLLMBackend()
    return PipelineRunner(
        RoleModules(
            LLMSelector(b, selected), LLMGenerator(b), LLMJudge(b), LLMCritic(b)
        ),
        FakeEmbedding(),
    )


def memory(tmp_path=None):
    return ExperienceMemory(
        "p", OnlineClustering(), path=tmp_path / "checkpoint.json" if tmp_path else None
    )


def seeded(count=3):
    m = memory()
    for i in range(count):
        r = runner(selected=0).run_sample(sample(i), m.snapshot())
        m.commit(r)
    assert len(m.experiences) == count
    return m


class TrialGenerator:
    def __init__(self):
        self.drafts = 0
        self.inputs = []

    def draft(self, task):
        self.drafts += 1
        return Rubric("draft", [Criterion("draft-c", "task-only", 1)], "draft analysis")

    def revise(self, task, draft, experiences):
        assert len(experiences) <= 1
        self.inputs.append((draft.to_dict(), [e.experience_id for e in experiences]))
        key = experiences[0].experience_id if experiences else "baseline"
        return Rubric(key, [Criterion(key + "-c", key, 1)], "revision analysis")


class TrialJudge:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def judge(self, task, rubric, chosen, rejected):
        value = self.outcomes.get(rubric.rubric_id, True)
        if value is None:
            raise LLMSampleError("offline timeout")
        return calculate_rubric_judgment(
            rubric,
            [
                CriterionJudgment(c.criterion_id, value, False, "fixture", "fixture")
                for c in rubric.criteria
            ],
        )


@pytest.mark.parametrize("values", list(product([False, True, None], repeat=4)))
def test_three_equal_trials_independent_credit_and_all_failed_gate(values):
    m = seeded()
    ids = list(m.experiences)
    b = FakeLLMBackend()
    r = runner(b)
    r.modules.generator = TrialGenerator()
    r.modules.judge = TrialJudge(dict(zip(["baseline", *ids], values)))
    result = r.run_sample(sample(10), m.snapshot())
    paths = result.record.paths
    assert len(paths) == 4 and all(len(p.provided_experience_ids) <= 1 for p in paths)
    assert result.record.prediction_path is None
    assert r.modules.generator.drafts == 1
    assert all(
        draft == result.record.draft.to_dict()
        for draft, _ in r.modules.generator.inputs
    )
    assert bool(result.record.critic_rounds) == all(v is False for v in values)
    feedback = downstream_feedback(result.record, "p")
    for row in feedback:
        index = ids.index(row.experience_id) + 1
        expected = (
            None
            if values[0] is None or values[index] is None
            else {row.experience_id: int(values[index]) - int(values[0])}
        )
        assert row.credits == expected
    m.commit(result)
    assert all(
        m.clusters.references("experience", eid) == {sample(10).task_id} for eid in ids
    )


def test_source_reuses_draft_no_credit_or_association_then_downstream(tmp_path):
    m = memory(tmp_path)
    b = FakeLLMBackend()
    r = runner(b)
    first = r.run_sample(sample(), m.snapshot())
    m.commit(first)
    assert len(first.record.paths) == 1 and len(first.record.critic_rounds) == 1
    eid = next(iter(m.experiences))
    assert m.clusters.references("experience", eid) == set()
    assert not m.ledger.records
    generated = [payload for role, payload in b.calls if role == "generator"]
    assert [p["stage"] for p in generated] == ["draft", "revise", "revise"]
    assert generated[1]["draft"] == generated[2]["draft"]
    second = r.run_sample(sample(1), m.snapshot())
    m.commit(second)
    assert len(m.ledger.records) == 1 and m.ledger.records[0].credits == {eid: 1}
    assert ExperienceMemory.load(m.path).to_dict() == m.to_dict()


def test_two_round_failure_reuses_draft_and_baseline():
    m = memory()
    b = FakeLLMBackend("unrepairable")
    result = runner(b).run_sample(sample(), m.snapshot())
    assert len(result.record.critic_rounds) == 2
    calls = [p for role, p in b.calls if role == "generator"]
    assert [p["stage"] for p in calls] == ["draft", "revise", "revise", "revise"]
    assert len({json.dumps(p["draft"], sort_keys=True) for p in calls[1:]}) == 1
    m.commit(result)
    assert not m.experiences


@pytest.mark.parametrize("stage", ["draft", "candidate"])
def test_technical_failure_never_becomes_critic_content_evidence(stage):
    class Fail(FakeLLMBackend):
        def _generate(self, role, payload, **kw):
            if role == "generator" and (
                (stage == "draft" and payload["stage"] == "draft")
                or (stage == "candidate" and payload.get("experiences"))
            ):
                raise LLMSampleError("fixture transport failure")
            return super()._generate(role, payload, **kw)

    m = memory()
    r = runner(Fail()).run_sample(sample(), m.snapshot())
    m.commit(r)
    if stage == "draft":
        assert r.record.error and not r.record.paths and not r.record.critic_rounds
    else:
        assert (
            len(r.record.critic_rounds) == 1
            and r.record.critic_rounds[0].attempts[0].path.correct is None
        )
    assert not m.ledger.records


def test_inference_selects_one_only_no_baseline_or_mutation():
    m = seeded()
    before = m.to_dict()
    b = FakeLLMBackend()
    result = runner(b).run_sample(sample(20), m.snapshot(), training=False)
    assert result.record.mode == "inference" and len(result.record.paths) == 1
    assert len(result.record.prediction_path.provided_experience_ids) == 1
    assert result.record.baseline is None and not result.record.critic_rounds
    assert m.to_dict() == before
    assert next(p for role, p in b.calls if role == "selector")["max_selected"] == 1
    with pytest.raises(ValueError, match="inference"):
        m.commit(result)


def test_human_reasons_only_reach_critic_with_response_mapping():
    metadata = {
        "chosen_response": 2,
        "individual_preference": [
            {"reasoning": "unique annotation", "score": 2},
            {"reasoning": "disagreement", "score": -1},
        ],
    }
    s = sample(metadata=metadata)
    b = FakeLLMBackend()
    runner(b).run_sample(s, memory().snapshot())
    critic = next(p for role, p in b.calls if role == "critic")
    assert (
        critic["preference_evidence"]["individual_preference"]
        == metadata["individual_preference"]
    )
    assert critic["preference_evidence"]["original_response_mapping"] == {
        "response_1": "rejected",
        "response_2": "chosen",
    }
    for role, p in b.calls:
        if role != "critic":
            assert "unique annotation" not in json.dumps(p)
        if role in ("generator", "selector"):
            assert s.chosen not in json.dumps(p) and s.rejected not in json.dumps(p)
        if role == "judge":
            assert set(p) == {"task", "criteria", "response"}
    assert (
        preference_evidence(sample())["original_response_mapping"]["response_1"]
        == "unknown"
    )


def test_critic_can_abstain_on_judge_only_failure():
    class Abstain(FakeLLMBackend):
        def _generate(self, role, payload, **kw):
            if role == "critic":
                return {
                    "analysis": "Only Judge execution is unsupported; no rubric defect established.",
                    "candidates": [],
                }
            return super()._generate(role, payload, **kw)

    result = runner(Abstain()).run_sample(sample(), memory())
    assert len(result.record.critic_rounds) == 1 and not result.admitted


def test_checkpoint_rejects_source_edge_and_tampered_feedback(tmp_path):
    m = seeded(1)
    m.path = tmp_path / "checkpoint.json"
    m.save()
    raw = m.to_dict()
    eid = next(iter(m.experiences))
    raw["clustering"]["citations"] = [[m.experiences[eid].source_task_id, eid]]
    m.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="associations"):
        ExperienceMemory.load(m.path)
    m.commit(runner().run_sample(sample(30), m.snapshot()))
    raw = m.to_dict()
    raw["feedback"][0]["experience_correct"] = False
    m.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="feedback"):
        ExperienceMemory.load(m.path)


def test_atomic_publish_rollback(tmp_path, monkeypatch):
    m = memory(tmp_path)
    before = m.to_dict()
    result = runner().run_sample(sample(), m.snapshot())
    monkeypatch.setattr(
        "verge.memory.atomic_json",
        lambda *_: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        m.commit(result)
    assert m.to_dict() == before


def test_unknown_id_retry_and_weight_overflow_normalization():
    class Hallucinate(FakeLLMBackend):
        def _generate(self, role, payload, **kw):
            result = super()._generate(role, payload, **kw)
            if role == "selector" and "validation_feedback" not in payload:
                result["selected_experience_ids"] = ["nonexistent"]
            if role == "generator":
                result["criteria"] = [
                    {"question": "one", "weight": 1e308},
                    {"question": "two", "weight": 1e308},
                ]
            return result

    b = Hallucinate()
    m = seeded()
    result = runner(b).run_sample(sample(40), m.snapshot())
    assert len([1 for role, _ in b.calls if role == "selector"]) == 2
    assert [c.weight for c in result.record.draft.criteria] == [0.5, 0.5]


def test_cli_pause_resume_and_frozen_evaluation(tmp_path):
    config = load_config("configs/default.yaml")
    config["run"]["output_dir"] = str(tmp_path / "run")
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(sample(i).to_json() for i in range(3)) + "\n")
    config["dataset"]["paths"]["train"] = str(train)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    first = run(path, stop_after_train=1)
    assert first["status"] == "paused"
    final = run(path, resume=True)
    assert final["status"] == "completed" and final["test"]["tasks"] > 0
    cp = tmp_path / "run" / "checkpoint.json"
    before = cp.read_bytes()
    report = evaluate(path, cp, "configs/data/demo_test.jsonl", tmp_path / "evaluation")
    assert report["status"] == "completed" and cp.read_bytes() == before
    assert ExperienceMemory.load(cp).protocol_id == execution_protocol(config)


def config_file(tmp_path, tasks=2):
    config = load_config("configs/default.yaml")
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(sample(i).to_json() for i in range(tasks)) + "\n")
    config["dataset"]["paths"] = {"train": str(data)}
    config["run"].update(output_dir=str(tmp_path / "run"), train_workers=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path
