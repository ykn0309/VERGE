"""用 Fake Modules 覆盖 Runner 的训练分支、隔离边界与写盘时机。"""

from dataclasses import replace

import pytest

from verge.models import CandidateExperience
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import Criterion, PreferenceSample, RevisionOperation, Rubric
from verge.modules import make_fake_modules
from verge.pipeline import PipelineRunner, RevisionExecutor


def sample(name: str = "sample") -> PreferenceSample:
    return PreferenceSample(name, "任务包含明确的关键要求。", "优质回答", "较差回答", {})


def make_runner(tmp_path, scenario="attempt_1_success", **options):
    modules = make_fake_modules(scenario, **options)
    memory = ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    return PipelineRunner(modules, memory, log), modules, memory, log


def test_initial_success_stops_before_revision_and_writes_no_experience(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, "initial_success")
    record = runner.run_sample(sample())
    assert record.state == "initial_success"
    assert record.initial_judgment.success
    assert (record.revision_attempts, record.reflected_candidates, record.operation_contributions) == ([], [], {})
    assert memory.all() == []
    assert [role for role, _ in modules.backend.requests] == ["initial_generator", "judge"]
    assert log.get("sample") == record


def test_attempt_one_success_adds_experience_and_full_trajectory(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_1"
    assert not record.initial_judgment.success
    assert len(record.revision_attempts) == 1
    attempt = record.revision_attempts[0]
    assert (attempt.attempt_number, attempt.rubric.base_rubric_id) == (1, record.initial_rubric.rubric_id)
    assert attempt.judgment.success
    assert len(attempt.operations) == 1
    assert attempt.operations[0].before_criterion == record.initial_rubric.criteria[1]
    assert attempt.operations[0].operation_id in record.operation_contributions
    result = record.reflected_candidates[0]
    assert result.linked_operation_ids == ["op1"]
    assert result.credit is not None and result.credit > 0
    assert (result.admission, result.fusion) == ("PASS", "ADD")
    stored = memory.get(result.experience_id)
    assert stored.alpha == pytest.approx(1 + result.credit)
    assert stored.source_trajectories == ["sample"]
    assert log.get("sample") == record
    # pool 为空，无需调用 fusion_judge 后端。
    assert [role for role, _ in modules.backend.requests][-2:] == [
        "revision_credit_assigner", "admission_checker",
    ]


def test_attempt_two_uses_r0_not_failed_r1_and_only_successful_operations(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, "attempt_2_success")
    record = runner.run_sample(sample())
    assert record.state == "revision_success_2"
    assert len(record.revision_attempts) == 2
    failed, successful = record.revision_attempts
    assert not failed.judgment.success and successful.judgment.success
    assert failed.rubric.rubric_id != successful.rubric.rubric_id
    assert failed.rubric.base_rubric_id == successful.rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert {op.operation_id for op in successful.operations} == set(record.operation_contributions) == {"op2"}
    assert record.reflected_candidates[0].linked_operation_ids == ["op2"]
    assert [item.experience_id for item in memory.all()] == [record.reflected_candidates[0].experience_id]
    assert log.get("sample") == record
    assert [payload["initial_rubric"]["rubric_id"] for role, payload in
            modules.backend.requests if role == "revision_generator"] == ["r0", "r0"]


def test_empty_first_revision_is_a_failed_attempt_not_an_abort(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, "attempt_2_success")
    generate = modules.backend.generate

    def empty_first_revision(role, payload):
        if role == "revision_generator" and payload["attempt_number"] == 1:
            return {"operations": []}
        return generate(role, payload)

    modules.backend.generate = empty_first_revision
    record = runner.run_sample(sample())
    assert record.state == "revision_success_2"
    assert len(record.revision_attempts) == 2
    first, second = record.revision_attempts
    assert first.operations == [] and not first.judgment.success
    assert first.rubric.base_rubric_id == second.rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert second.operations[0].operation_id in record.operation_contributions
    assert log.get("sample") == record


def test_invalid_revision_output_retries_without_consuming_an_attempt(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    original = modules.backend.generate
    calls = []

    def invalid_once(role, payload):
        if role == "revision_generator":
            calls.append(payload)
            if len(calls) == 1:
                return {"operations": [{"operation_id": "bad", "operation_type": "DELETE",
                                        "target_criterion_id": "c1",
                                        "new_or_updated_criterion": None}]}
        return original(role, payload)

    modules.backend.generate = invalid_once
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log,
                            max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_1"
    assert len(record.revision_attempts) == 1
    assert [operation.operation_id for operation in record.revision_attempts[0].operations] == ["op1"]
    assert len(calls) == 2
    assert "revised rubric must contain 4..8 criteria" in calls[1]["feedback"]
    assert record.revision_attempts[0].validation_failures == [
        "revised rubric must contain 4..8 criteria"]
    assert log.get("sample").revision_attempts[0].validation_failures == [
        "revised rubric must contain 4..8 criteria"]
    assert "优质回答" not in str(calls[1]) and "较差回答" not in str(calls[1])
    assert log.get("sample") == record


def test_invalid_revision_output_exhausts_retries_without_trajectory(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    calls = []

    def always_invalid(role, payload):
        if role == "revision_generator":
            calls.append(payload)
            return {"operations": [{"operation_id": "bad", "operation_type": "DELETE",
                                    "target_criterion_id": "c1",
                                    "new_or_updated_criterion": None}]}
        return original(role, payload)

    original = modules.backend.generate
    modules.backend.generate = always_invalid
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log,
                            max_revision_output_retries=2)
    with pytest.raises(ValueError, match="revised rubric must contain 4..8 criteria"):
        runner.run_sample(sample())
    assert len(calls) == 3
    assert log.get("sample") is None


def test_final_failure_writes_trajectory_without_new_experience(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, "final_failure")
    record = runner.run_sample(sample())
    assert record.state == "final_failure"
    assert len(record.revision_attempts) == 2
    assert all(not attempt.judgment.success for attempt in record.revision_attempts)
    assert record.reflected_candidates == [] and record.operation_contributions == {}
    assert not any(role in ("reflector", "fusion_judge") for role, _ in modules.backend.requests)
    assert memory.all() == [] and log.get("sample") == record


@pytest.mark.parametrize("options,expected_roles", [
    ({"reflect_count": 0}, {"reflector"}),
    ({"unbound": True}, {"reflector", "revision_credit_assigner"}),
])
def test_no_candidates_or_unbound_skips_admission_and_fusion(tmp_path, options, expected_roles):
    runner, modules, memory, log = make_runner(tmp_path, **options)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_1"
    roles = {role for role, _ in modules.backend.requests}
    assert expected_roles <= roles and "admission_checker" not in roles and "fusion_judge" not in roles
    assert not memory.all()
    if options.get("unbound"):
        assert len(record.reflected_candidates) == 1
        assert record.reflected_candidates[0].linked_operation_ids == []
        assert record.reflected_candidates[0].credit is None
    else:
        assert record.reflected_candidates == []
    assert log.get("sample") == record


def test_admission_fail_keeps_provisional_credit_in_log_without_memory(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, admission_pass=False)
    record = runner.run_sample(sample())
    rejected = record.reflected_candidates[0]
    assert rejected.credit > 0 and rejected.admission == "FAIL"
    assert rejected.fusion is None and rejected.experience_id is None
    assert memory.all() == [] and log.get("sample") == record
    assert not any(role == "fusion_judge" for role, _ in modules.backend.requests)


def test_multiple_candidates_share_operation_budget_and_fuse_add_then_merge(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, reflect_count=2)
    record = runner.run_sample(sample())
    first, second = record.reflected_candidates
    assert (first.fusion, second.fusion) == ("ADD", "MERGE")
    assert first.experience_id == second.experience_id
    assert first.credit == pytest.approx(second.credit)
    assert first.credit + second.credit <= 1
    stored = memory.get(first.experience_id)
    assert stored.alpha == pytest.approx(1 + first.credit + second.credit)
    assert stored.source_trajectories == ["sample", "sample"]
    assert len(stored.evidence_history) == 2
    assert log.get("sample") == record
    fusion_requests = [payload for role, payload in modules.backend.requests if role == "fusion_judge"]
    assert len(fusion_requests) == 1  # 第一条候选时经验池为空。
    assert set(fusion_requests[0]) == {"candidate", "existing"}
    assert set(fusion_requests[0]["existing"][0]) == {"content", "trigger"}


def test_downstream_persisted_before_later_revision_failure(tmp_path):
    memory = ExperienceMemory(tmp_path)
    created = memory.fuse(make_candidate(), "origin")
    before = memory.get(created.experience_id).beta
    modules = make_fake_modules("attempt_1_success")
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))

    class FailedCritic:
        def critique(self, *args, **kwargs):
            raise RuntimeError("failure after downstream write")

    modules.critic = FailedCritic()
    with pytest.raises(RuntimeError, match="failure after downstream write"):
        runner.run_sample(sample())
    assert ExperienceMemory(tmp_path).get(created.experience_id).beta > before
    assert TrajectoryLog(tmp_path).get("sample") is None
    assert modules.backend.requests[0][0] == "selector"


def make_candidate() -> CandidateExperience:
    return CandidateExperience("seed", "参考经验", "任务包含明确的关键要求。", ["seed-op"],
                               0.2, "PASS")


def test_downstream_credit_on_initial_success_and_sample_order(tmp_path):
    memory = ExperienceMemory(tmp_path)
    seed = memory.fuse(make_candidate(), "origin")
    runner = PipelineRunner(make_fake_modules("initial_success"), memory, TrajectoryLog(tmp_path),
                            evidence_weight=0.5)
    first, second = runner.run([sample("one"), sample("two")])
    assert first.selected_experience_ids == second.selected_experience_ids == [seed.experience_id]
    stored = ExperienceMemory(tmp_path).get(seed.experience_id)
    # c1 的权重为 2/12；每个样本证据权重为 0.5。
    assert stored.alpha == pytest.approx(1.2 + 2 * (2 / 12) * 0.5)
    assert [item.sample_id for item in stored.evidence_history] == ["origin", "one", "two"]
    assert [item.evidence_type for item in stored.evidence_history] == [
        "revision_credit", "downstream_credit", "downstream_credit"]


@pytest.mark.parametrize("scenario", ["initial_success", "final_failure"])
def test_eval_only_initial_judgment_never_writes_memory(tmp_path, scenario):
    seeded = ExperienceMemory(tmp_path).fuse(make_candidate(), "origin")
    before = (tmp_path / "experience_memory.json").read_bytes()
    memory = ExperienceMemory(tmp_path, read_only=True)
    modules = make_fake_modules(scenario)
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, memory, log, training=False)
    record = runner.run_sample(sample())
    assert record.selected_experience_ids == [seeded.experience_id]
    assert record.state == ("initial_success" if scenario == "initial_success" else "initial_failure")
    assert record.revision_attempts == [] and record.reflected_candidates == []
    assert (tmp_path / "experience_memory.json").read_bytes() == before
    assert not any(role in ("critic", "revision_generator", "reflector")
                   for role, _ in modules.backend.requests)
    selector_payload = next(payload for role, payload in modules.backend.requests if role == "selector")
    assert selector_payload["training"] is False
    assert log.get("sample") == record


def test_admission_fail_does_not_redistribute_other_candidates_credit(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, reflect_count=2)

    class MixedAdmission:
        def check(self, task, candidate, successful_attempt, operation_contributions):
            return replace(candidate, admission="FAIL" if candidate.candidate_id == "candidate_0"
                           else "PASS")

    modules.admission_checker = MixedAdmission()
    record = runner.run_sample(sample())
    failed, passed = record.reflected_candidates
    assert failed.admission == "FAIL" and failed.fusion is None
    assert passed.admission == "PASS" and passed.fusion == "ADD"
    assert failed.credit == pytest.approx(passed.credit)
    assert memory.get(passed.experience_id).alpha == pytest.approx(1 + passed.credit)
    assert log.get("sample") == record


def test_preference_blind_roles_receive_only_task_and_public_experiences(tmp_path):
    memory = ExperienceMemory(tmp_path)
    memory.fuse(make_candidate(), "origin")
    modules = make_fake_modules("initial_success")
    runner = PipelineRunner(modules, memory, TrajectoryLog(tmp_path))
    runner.run_sample(sample())
    selector = next(payload for role, payload in modules.backend.requests if role == "selector")
    generator = next(payload for role, payload in modules.backend.requests
                     if role == "initial_generator")
    assert set(selector) == {"task", "candidates", "max_selected", "training"}
    assert set(generator) == {"task", "selected_experiences"}
    assert set(generator["selected_experiences"][0]) == {
        "experience_id", "content", "trigger", "confidence", "evidence_amount"}
    assert "优质回答" not in str(selector) + str(generator)
    assert "较差回答" not in str(selector) + str(generator)


def test_cross_id_identical_question_cannot_gain_credit_by_judge_drift(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    original_generate = modules.backend.generate

    def drifting_judge(role, payload):
        if role == "revision_generator":
            return {"operations": [
                {"operation_id": "del", "operation_type": "DELETE", "target_criterion_id": "c2",
                 "new_or_updated_criterion": None},
                {"operation_id": "add", "operation_type": "ADD", "target_criterion_id": None,
                 "new_or_updated_criterion": {
                     "criterion_id": "c5", "question": "回答是否符合要求？",
                     "importance_score": 8, "source_experiences": []}},
            ]}
        result = original_generate(role, payload)
        if role == "judge":
            for item in result["criterion_judgments"]:
                if item["criterion_id"] == "c5":
                    item["chosen"] = True
        return result

    modules.backend.generate = drifting_judge
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), TrajectoryLog(tmp_path))
    with pytest.raises(ValueError, match="unchanged criterion question"):
        runner.run_sample(sample())
    assert ExperienceMemory(tmp_path).all() == []


def test_executor_validates_conflicts_and_populates_only_one_operation_list():
    initial = Rubric("r0", None, [Criterion(f"c{i}", f"Q{i}", 2, ["exp"])
                                   for i in range(4)])
    executor = RevisionExecutor()
    modified = RevisionOperation("m", "MODIFY", "c1", Criterion("c1", "new", 2, ["external"]))
    revised, executed = executor.execute(initial, [modified], 1)
    assert revised.base_rubric_id == "r0"
    assert revised.criteria[1].source_experiences == []
    assert executed[0].before_criterion == initial.criteria[1]
    assert modified.before_criterion is None  # 未改变 Generator 输出。
    with pytest.raises(ValueError, match="no-op MODIFY"):
        executor.execute(initial, [RevisionOperation("noop", "MODIFY", "c1",
                                                     Criterion("c1", "Q1", 2, []))], 1)
    with pytest.raises(ValueError, match="conflicting"):
        executor.execute(initial, [modified, modified], 1)
    with pytest.raises(ValueError, match="4..8"):
        executor.execute(initial, [RevisionOperation("drop", "DELETE", "c1", None)], 1)
    with pytest.raises(ValueError, match="new criterion ID"):
        executor.execute(initial, [RevisionOperation("add", "ADD", None,
                                                     Criterion("c1", "new", 3, []))], 1)
