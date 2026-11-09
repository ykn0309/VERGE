"""Four-path execution, joint admission and source/downstream separation."""

from dataclasses import replace
from itertools import product
import json
import pytest

from test_learning_workflow import runner, sample, state
from verge.calculator import calculate_rubric_judgment
from verge.llm import FakeLLMBackend, LLMSampleError
from verge.models import Criterion, CriterionJudgment, ExecutionPath, Rubric
from verge.memory import ExperienceMemory
from verge.pipeline import PipelineRunner


class TwoCandidates(FakeLLMBackend):
    def _generate(self, role, payload, **kwargs):
        value = super()._generate(role, payload, **kwargs)
        if role == "critic":
            item = value["candidates"][0]
            left = {**item, "method": {**item["method"], "principle": "left"}}
            right = {**item, "method": {**item["method"], "principle": "right"}}
            value["candidates"] = [left, right]
        return value


class ControlledRunner(PipelineRunner):
    def __init__(self, outcomes, backend=None):
        base = runner(backend or TwoCandidates())
        super().__init__(base.modules, base.embedder)
        self.outcomes = outcomes
        self.executions = []

    def execute(self, sample_, experiences):
        key = tuple(sorted(e.method.principle for e in experiences))
        self.executions.append((sample_.task_id, key))
        result = self.outcomes[key]
        path = ExecutionPath([e.experience_id for e in experiences])
        if result is None:
            path.error = "fixture transport failure"
            return path
        path.rubric = Rubric(
            "r" + str(len(self.executions)), [Criterion("c", "condition?", 1)]
        )
        path.judgment = calculate_rubric_judgment(
            path.rubric, [CriterionJudgment("c", result, False, "fixture", "fixture")]
        )
        return path


def seed_pair(memory):
    r = ControlledRunner(
        {(): False, ("left",): False, ("right",): False, ("left", "right"): True}
    )
    result = r.run_sample(sample(0), memory.snapshot())
    memory.commit(result)
    return r, result


def test_joint_only_admission_retains_all_subsets_but_no_source_credit_or_edges(
    tmp_path,
):
    memory = state(tmp_path)
    r, result = seed_pair(memory)
    assert len(result.admitted) == 2 and len(r.executions) == 4
    assert len(result.record.critic_rounds[0].subset_paths) == 4
    assert all(
        a.reason == "joint_correct" for a in result.record.critic_rounds[0].attempts
    )
    assert memory.ledger.records == ()
    assert memory.clusters.references("task", sample(0).task_id) == frozenset()
    for eid in memory.experiences:
        assert memory.clusters.references("experience", eid) == frozenset()
        assert memory.ledger.for_experience(eid).mean_reward is None
    assert ExperienceMemory.load(memory.path).to_dict() == memory.to_dict()


@pytest.mark.parametrize("values", list(product([False, True], repeat=4)))
def test_initial_four_paths_credit_and_predetermined_actual_for_all_outcomes(
    tmp_path, values
):
    memory = state(tmp_path)
    seed_pair(memory)
    keys = [(), ("left",), ("right",), ("left", "right")]
    r = ControlledRunner(dict(zip(keys, values)))
    r.critic_rounds = 0
    result = r.run_sample(sample(1), memory.snapshot())
    assert len(r.executions) == 4
    assert result.record.actual.correct is values[3]
    memory.commit(result)
    feedback = memory.ledger.records[0]
    assert feedback.actual_correct is values[3]
    r0, r1, r2, r12 = map(int, values)
    by_principle = {
        e.method.principle: e.experience_id for e in memory.experiences.values()
    }
    assert feedback.credits[by_principle["left"]] == ((r1 - r0) + (r12 - r2)) / 2
    assert feedback.credits[by_principle["right"]] == ((r2 - r0) + (r12 - r1)) / 2
    assert memory.clusters.references("task", sample(1).task_id) == frozenset(
        by_principle.values()
    )
    assert ExperienceMemory.load(memory.path).to_dict() == memory.to_dict()


def test_incomplete_downstream_has_edges_but_no_valid_credit():
    memory = state()
    seed_pair(memory)
    r = ControlledRunner(
        {(): False, ("left",): None, ("right",): False, ("left", "right"): True}
    )
    result = r.run_sample(sample(1), memory.snapshot())
    memory.commit(result)
    assert memory.ledger.records[0].credits is None
    assert len(memory.clusters.references("task", sample(1).task_id)) == 2
    assert all(
        memory.ledger.for_experience(eid).tasks == 0 for eid in memory.experiences
    )


def test_singleton_success_does_not_admit_unhelpful_companion():
    memory = state()
    r = ControlledRunner(
        {(): False, ("left",): True, ("right",): False, ("left", "right"): True}
    )
    result = r.run_sample(sample(), memory.snapshot())
    assert [e.method.principle for e, _ in result.admitted] == ["left"]
    memory.commit(result)
    assert memory.ledger.records == ()


def test_successful_pair_preserves_unknown_singleton_without_filling_zero():
    memory = state()
    r = ControlledRunner(
        {(): False, ("left",): None, ("right",): False, ("left", "right"): True}
    )
    result = r.run_sample(sample(), memory.snapshot())
    assert len(result.admitted) == 2
    assert result.record.critic_rounds[0].attempts[0].path.correct is None
    memory.commit(result)
    assert memory.ledger.records == ()


def test_pair_embedding_failure_cannot_publish_half_pair():
    memory = state()
    r = ControlledRunner(
        {(): False, ("left",): False, ("right",): False, ("left", "right"): True}
    )
    original = r.embedder.embed

    def embed(text):
        if '"principle": "right"' in text:
            raise LLMSampleError("fixture embedding failure")
        return original(text)

    r.embedder.embed = embed
    result = r.run_sample(sample(), memory.snapshot())
    assert result.admitted == []
    assert all(not a.admitted for a in result.record.critic_rounds[0].attempts)
    memory.commit(result)
    assert not memory.experiences


def test_baseline_is_reused_across_two_critic_rounds():
    memory = state()
    r = ControlledRunner(
        {(): False, ("left",): False, ("right",): False, ("left", "right"): False}
    )
    result = r.run_sample(sample(), memory.snapshot())
    assert len(result.record.critic_rounds) == 2
    assert len(r.executions) == 7
    assert sum(key == () for _, key in r.executions) == 1


def test_frozen_inference_has_only_full_selection_and_no_history_updates():
    memory = state()
    seed_pair(memory)
    r = ControlledRunner({("left", "right"): True})
    before = memory.to_dict()
    result = r.run_sample(sample(1), memory.snapshot(), training=False)
    assert len(r.executions) == 1 and result.record.baseline is None
    assert not result.record.critic_rounds and memory.to_dict() == before


def test_checkpoint_rejects_source_edge_injection_and_false_subset(tmp_path):
    memory = state(tmp_path)
    seed_pair(memory)
    original = json.loads(memory.path.read_text())
    eid = next(iter(memory.experiences))
    raw = json.loads(json.dumps(original))
    raw["clustering"]["citations"].append([sample(0).task_id, eid])
    memory.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="associations"):
        ExperienceMemory.load(memory.path)


def test_joint_admission_rejects_half_published_checkpoint(tmp_path):
    memory = state(tmp_path)
    seed_pair(memory)
    data = memory.to_dict()
    partner = data["trajectories"][0]["critic_rounds"][0]["attempts"][1]
    partner["admitted"] = False
    partner["admission_subset"] = []
    memory.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="joint admission"):
        ExperienceMemory.load(memory.path)
