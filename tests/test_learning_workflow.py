"""The new complete protocol, using deterministic backend fixtures only."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import json
import pytest
import yaml

from verge.calculator import calculate_rubric_judgment
from verge.cli import Components, run
from verge.clustering import ClusterItem, OnlineClustering
from verge.config import load_config
from verge.llm import FakeEmbedding, FakeLLMBackend, LLMFormatError, LLMSampleError
from verge.memory import ExperienceMemory
from verge.models import (
    Method,
    Experience,
    PreferenceSample,
    ExecutionPath,
    Criterion,
    Rubric,
    CriterionJudgment,
    TrajectoryRecord,
)
from verge.modules import (
    LLMSelector,
    LLMGenerator,
    LLMJudge,
    LLMCritic,
    RoleModules,
    CriticSession,
)
from verge.pipeline import PipelineRunner, SampleResult
from verge.retrieval import retrieve_candidates
from verge.scheduling import run_training_stream


def sample(i=0):
    return PreferenceSample(f"s{i}", f"task {i}", "chosen text", "rejected text", {})


def state(tmp_path=None):
    return ExperienceMemory(
        "p",
        OnlineClustering(
            reference_weight=0.3, task_threshold=0.75, experience_threshold=0.75
        ),
        path=tmp_path / "checkpoint.json" if tmp_path else None,
    )


def runner(backend=None):
    b = backend or FakeLLMBackend()
    return PipelineRunner(
        RoleModules(LLMSelector(b), LLMGenerator(b), LLMJudge(b), LLMCritic(b)),
        FakeEmbedding(),
    )


def test_source_candidate_cited_and_correct_is_admitted_without_reapplication(tmp_path):
    b = FakeLLMBackend()
    r = runner(b)
    s = state(tmp_path)
    result = r.run_sample(sample(), s.snapshot())
    assert result.record.baseline.correct is False and len(result.admitted) == 1
    assert len(result.record.critic_rounds) == 1
    assert result.record.critic_rounds[0].attempts[0].admitted
    assert [role for role, _ in b.calls].count("generator") == 2
    assert [role for role, _ in b.calls].count("judge") == 4
    assert not any("checker" in role or "updater" in role for role, _ in b.calls)
    s.commit(result)
    assert len(s.experiences) == 1 and s.ledger.records == ()
    assert len(s.clusters.references("task", sample().task_id)) == 0
    loaded = ExperienceMemory.load(s.path)
    assert loaded.to_dict() == s.to_dict()
    with pytest.raises(ValueError, match="read-only"):
        loaded.commit(result)


def test_two_round_budget_stops_failed_candidates_and_passes_failure_trace():
    b = FakeLLMBackend("unrepairable")
    s = state()
    result = runner(b).run_sample(sample(), s)
    assert not result.admitted and len(result.record.critic_rounds) == 2
    critic_calls = [p for role, p in b.calls if role == "critic"]
    assert "chosen" in critic_calls[0] and "previous_attempts" in critic_calls[1]
    assert "task" not in critic_calls[1] and "chosen" not in critic_calls[1]
    assert all(not a.admitted for r in result.record.critic_rounds for a in r.attempts)



def test_normal_downstream_pairs_are_accounted_once_and_restart_is_exact(tmp_path):
    s = state(tmp_path)
    r = runner()
    s.commit(r.run_sample(sample(0), s.snapshot()))
    second = r.run_sample(sample(1), s.snapshot())
    s.commit(second)
    assert second.record.actual is not None and second.record.actual.correct is True
    assert len(s.ledger.records) == 1
    eid = next(iter(s.experiences))
    assert s.ledger.for_experience(eid).tasks == 1
    assert s.ledger.for_experience(eid).mean_reward == 1
    with pytest.raises(ValueError, match="already committed"):
        s.commit(second)
    loaded = ExperienceMemory.load(s.path, read_only=False)
    assert loaded.to_dict() == s.to_dict()
    third = r.run_sample(sample(2), loaded.snapshot())
    loaded.commit(third)
    assert loaded.ledger.for_experience(eid).tasks == 2


def test_inference_runs_only_the_used_path_and_never_mutates_memory():
    s = state()
    b = FakeLLMBackend()
    r = runner(b)
    s.commit(r.run_sample(sample(), s.snapshot()))
    before = deepcopy(s.to_dict())
    b.calls = []
    record = r.run_sample(sample(1), s.snapshot(), training=False).record
    assert record.baseline is None and record.actual.correct
    assert [role for role, _ in b.calls].count("generator") == 1
    assert not record.critic_rounds and s.to_dict() == before


def test_true_initial_success_does_not_call_critic():
    b = FakeLLMBackend("correct")
    result = runner(b).run_sample(sample(), state())
    assert (
        result.record.baseline.correct
        and not result.record.critic_rounds
        and not result.admitted
    )


def test_two_candidates_are_individually_validated_and_both_can_enter():
    class Two(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            result = super().generate(role, payload, **kw)
            if role == "critic":
                result["candidates"].append(
                    {
                        "method": {
                            "trigger": "另一种适用条件",
                            "principle": "另一条核心方法",
                            "application_guide": "另一条使用指南",
                        },
                        "parent_ids": [],
                    }
                )
            return result

    b = Two()
    result = runner(b).run_sample(sample(), state())
    assert len(result.admitted) == 2 and len(result.record.critic_rounds) == 1
    candidate_generations = [
        p for role, p in b.calls if role == "generator" and p["experiences"]
    ]
    assert sorted(len(p["experiences"]) for p in candidate_generations) == [1, 1, 2]


def test_technical_failures_do_not_become_zero_reward_or_content_revision():
    class Timeout(FakeLLMBackend):
        def generate(self, role, payload, **kw):
            raise LLMSampleError("timeout")

    b = Timeout()
    result = runner(b).run_sample(sample(), state())
    assert result.record.baseline.error and not result.record.critic_rounds


def test_checkpoint_publication_failure_leaves_live_state_unmodified(
    tmp_path, monkeypatch
):
    s = state(tmp_path)
    result = runner().run_sample(sample(), s.snapshot())
    before = s.to_dict()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("verge.memory.atomic_json", fail)
    with pytest.raises(OSError):
        s.commit(result)
    assert s.to_dict() == before


def test_cluster_snapshot_does_not_recluster_or_renormalize_saved_bits():
    s = state()
    s.clusters.update(
        [
            ClusterItem("t", "task", "a", (0.123, 0.456, 0.789)),
            ClusterItem("e", "experience", "b", (0.2, 0.3, 0.4)),
        ],
        [("t", "e")],
    )
    restored = OnlineClustering.from_dict(s.clusters.to_dict())
    assert restored.to_dict() == s.clusters.to_dict()
    assert (
        restored.update(
            [ClusterItem("t", "task", "a", (0.123, 0.456, 0.789))]
        ).new_items
        == 0
    )


def test_rolling_workers_refill_before_slow_peer_finishes():
    slow = Event()
    third = Event()
    s = state()
    starts = []

    class Controlled:
        def run_sample(self, x, snapshot, **kw):
            starts.append(x.sample_id)
            if x.sample_id == "s0":
                assert third.wait(3)
                slow.set()
            if x.sample_id == "s2":
                third.set()
            return SampleResult(
                TrajectoryRecord(x, {}, {}, None, None, error="fixture failure"), None, []
            )

    run_training_stream(
        [sample(i) for i in range(3)], s, lambda _: Controlled(), workers=2
    )
    assert slow.is_set() and len(s.trajectories) == 3


def test_new_cases_see_committed_experience_and_not_unfinished_peer():
    s = state()
    counts = []

    class Observe:
        def __init__(self, x):
            self.x = x

        def run_sample(self, x, snapshot, **kw):
            counts.append(len(snapshot.experiences))
            return runner().run_sample(x, snapshot)

    run_training_stream([sample(i) for i in range(3)], s, Observe, workers=1)
    assert counts == [0, 1, 1]


def test_selector_generator_judge_visibility_and_weights():
    b = FakeLLMBackend()
    r = runner(b)
    s = state()
    s.commit(r.run_sample(sample(), s.snapshot()))
    b.calls = []
    r.run_sample(sample(1), s.snapshot(), training=False)
    for role, payload in b.calls:
        if role in ("selector", "generator"):
            assert (
                "chosen" not in payload
                and "rejected" not in payload
                and "metadata" not in payload
            )
        if role == "selector":
            assert all("q_posterior" not in e for e in payload["candidates"])
        if role == "judge":
            assert set(payload) == {"task", "criteria", "response"} and all(
                set(c) == {"criterion_id", "question"} for c in payload["criteria"]
            )


def test_scoring_normalizes_exact_decimal_weights_and_tie_is_not_success():
    rubric = Rubric("r", [Criterion("a", "a", 2), Criterion("b", "b", 1)])
    result = calculate_rubric_judgment(
        rubric,
        [
            CriterionJudgment("a", True, False, "yes", "no"),
            CriterionJudgment("b", False, True, "no", "yes"),
        ],
    )
    assert result.chosen_score == pytest.approx(
        2 / 3
    ) and result.margin == pytest.approx(1 / 3)
    tie = calculate_rubric_judgment(
        rubric,
        [
            CriterionJudgment("a", True, True, "yes", "yes"),
            CriterionJudgment("b", False, False, "no", "no"),
        ],
    )
    assert tie.margin == 0 and not tie.success


def config_file(tmp_path, tasks=3):
    train = tmp_path / "train.jsonl"
    train.write_text(
        "".join(json.dumps(sample(i).to_dict()) + "\n" for i in range(tasks))
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_dir": str(tmp_path / "run"), "train_workers": 1},
                "dataset": {"paths": {"train": str(train)}},
            }
        )
    )
    return path


def test_cli_pause_resume_and_protocol_hash_guard(tmp_path):
    path = config_file(tmp_path)
    manifest = run(path, stop_after_train=1)
    assert manifest["status"] == "paused" and manifest["train_completed"] == 1
    manifest = run(path, resume=True)
    assert (
        manifest["status"] == "training_completed" and manifest["train_completed"] == 3
    )
    state_data = ExperienceMemory.load(tmp_path / "run/checkpoint.json")
    assert len(state_data.ledger.records) == 2
    with pytest.raises(ValueError, match="completed"):
        run(path, resume=True)


def test_cli_limit_must_be_positive(tmp_path):
    path = config_file(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        run(path, stop_after_train=0)
    assert not (tmp_path / "run").exists()


def test_old_memory_and_configuration_are_rejected(tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"schema_version":14}')
    with pytest.raises(ValueError, match="unsupported"):
        ExperienceMemory.load(old)
    config = tmp_path / "old.yaml"
    config.write_text("modules: {reliability: {implementation: bayesian}}")
    with pytest.raises(ValueError, match="unsupported"):
        load_config(config)


def test_corrupt_checkpoint_does_not_change_feedback_or_embedded_method_silently(
    tmp_path,
):
    s = state(tmp_path)
    r = runner()
    s.commit(r.run_sample(sample(0), s.snapshot()))
    s.commit(r.run_sample(sample(1), s.snapshot()))
    original = json.loads(s.path.read_text())
    broken = deepcopy(original)
    broken["experiences"][0]["method"]["principle"] = "已被改写的正文"
    s.path.write_text(json.dumps(broken, ensure_ascii=False))
    with pytest.raises(ValueError, match="embedded text|admitted candidates"):
        ExperienceMemory.load(s.path)
    broken = deepcopy(original)
    broken["feedback"][0]["actual_correct"] = False
    s.path.write_text(json.dumps(broken, ensure_ascii=False))
    with pytest.raises(ValueError, match="saved subset|single experience"):
        ExperienceMemory.load(s.path)
    broken = deepcopy(original)
    broken["feedback"].append(broken["feedback"][0])
    s.path.write_text(json.dumps(broken, ensure_ascii=False))
    with pytest.raises(ValueError, match="duplicate feedback"):
        ExperienceMemory.load(s.path)


def test_large_finite_generator_weights_cannot_overflow_normalization():
    class Large(FakeLLMBackend):
        def generate(self, role, payload, **kwargs):
            result = super().generate(role, payload, **kwargs)
            if role == "generator":
                row = result["criteria"][0]
                row["weight"] = 1e308
                result["criteria"].append(dict(row))
            return result

    rubric = LLMGenerator(Large()).generate("task", [])
    assert [c.weight for c in rubric.criteria] == [0.5, 0.5]


def test_cli_rejects_duplicate_task_before_creating_a_run(tmp_path):
    path = config_file(tmp_path, tasks=3)
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(sample(0).to_dict())
        + "\n"
        + json.dumps(sample(1).to_dict())
        + "\n"
        + json.dumps({**sample(0).to_dict(), "sample_id": "duplicate"})
        + "\n"
    )
    with pytest.raises(ValueError, match="duplicate normalized task"):
        run(path)
    assert not (tmp_path / "run").exists()


def test_separate_frozen_evaluation_never_retrains_or_changes_pool(tmp_path):
    from verge.cli import evaluate, digest

    config = config_file(tmp_path)
    run(config)
    checkpoint = tmp_path / "run/checkpoint.json"
    before = digest(checkpoint)
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps(sample(20).to_dict()) + "\n")
    manifest = evaluate(config, checkpoint, test, tmp_path / "evaluation")
    assert manifest["status"] == "completed" and manifest["test"]["tasks"] == 1
    assert manifest["training_overlap"] == 0 and digest(checkpoint) == before
    assert manifest["test"]["errors"] == 0


def test_evaluation_rejects_changed_execution_protocol(tmp_path):
    from verge.cli import evaluate

    path = config_file(tmp_path)
    run(path)
    config = yaml.safe_load(path.read_text())
    config["selection"] = {"max_selected": 1}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="protocol changed"):
        evaluate(
            path,
            tmp_path / "run/checkpoint.json",
            tmp_path / "train.jsonl",
            tmp_path / "evaluation",
        )
    assert not (tmp_path / "evaluation").exists()



def test_critic_receives_real_selection_explanation_and_short_provided_ids():
    s = state()
    r = runner()
    s.commit(r.run_sample(sample(), s.snapshot()))
    b = FakeLLMBackend("unrepairable")
    runner(b).run_sample(sample(1), s.snapshot())
    payload = next(p for role, p in b.calls if role == "critic")
    assert payload["selection_reasons"] and payload["retrieval_candidates"]
    assert all(
        not value.startswith("exp_")
        for value in payload["actual_experience_ids"]
    )


def test_candidate_transport_failure_does_not_trigger_content_rewrite():
    class FailingValidation(FakeLLMBackend):
        def generate(self, role, payload, **kwargs):
            if role == "generator" and payload["experiences"]:
                raise LLMSampleError("temporary transport failure")
            return super().generate(role, payload, **kwargs)

    result = runner(FailingValidation()).run_sample(sample(), state())
    assert len(result.record.critic_rounds) == 1
    assert result.record.critic_rounds[0].attempts[0].reason == "technical_failure"
    assert not result.admitted


@pytest.mark.parametrize("exhaust", [False, True])
def test_unknown_experience_retries_and_does_not_stop_following_samples(exhaust):
    class HallucinatedUsage(FakeLLMBackend):
        def __init__(self):
            super().__init__("correct")
            self.first_calls = 0

        def _generate(self, role, payload, **kwargs):
            output = super()._generate(role, payload, **kwargs)
            if role == "generator" and payload["task"] == "task 0":
                self.first_calls += 1
                if exhaust or self.first_calls == 1:
                    output["experience_usage"] = [
                        {"experience_id": "999", "used": False, "reason": "invalid ID"}
                    ]
            return output

    b = HallucinatedUsage()
    memory = state()
    run_training_stream([sample(0), sample(1)], memory, lambda _: runner(b), workers=1)
    assert len(memory.trajectories) == 2
    assert b.first_calls == (3 if exhaust else 2)
    first = memory.trajectories[sample(0).task_id]
    second = memory.trajectories[sample(1).task_id]
    assert bool(first.baseline.error) is exhaust
    assert second.baseline.correct is True
