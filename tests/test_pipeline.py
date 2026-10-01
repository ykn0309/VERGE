"""用 Fake Modules 覆盖 Runner 的训练分支、隔离边界与写盘时机。"""

from dataclasses import replace

import pytest

from verge.models import CandidateExperience
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import Criterion, PreferenceSample, RevisionOperation, Rubric
from verge.modules import JudgeOutputError, make_fake_modules
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
    assert [role for role, _ in modules.backend.requests] == ["initial_generator", "judge", "judge"]
    assert log.get("sample") == record


def test_initial_judge_error_records_failure_and_continues_without_credit(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, "initial_success")
    original = modules.judge.judge
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise JudgeOutputError("invalid numbered criterion")
        return original(*args, **kwargs)

    modules.judge.judge = fail_once
    first, second = runner.run([sample("first"), sample("second")])
    assert first.state == "judge_error_initial" and first.initial_judgment is None
    assert first.judge_error == "invalid numbered criterion" and log.get("first") == first
    assert second.state == "initial_success" and log.get("second") == second
    assert memory.all() == []


def test_revision_judge_error_preserves_initial_credit_and_skips_reflection(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path)
    original = modules.judge.judge
    calls = 0

    def fail_revision(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise JudgeOutputError("invalid numbered criterion")
        return original(*args, **kwargs)

    modules.judge.judge = fail_revision
    record = runner.run_sample(sample())
    assert record.state == "judge_error_revision" and record.initial_judgment is not None
    assert record.judge_error and not record.revision_attempts
    assert not record.reflected_candidates and not memory.all()
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
    assert attempt.analysis and attempt.feedback is None
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
        "reflector", "admission_checker",
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
            modules.backend.requests if role == "critic"] == ["r0", "r0"]


def test_empty_first_revision_is_a_failed_attempt_not_an_abort(tmp_path):
    runner, modules, memory, log = make_runner(tmp_path, "attempt_2_success")
    generate = modules.backend.generate

    def empty_first_revision(role, payload):
        if role == "critic" and payload["attempt_number"] == 1:
            return {"analysis": "需要重新设计关键标准。", "operations": []}
        return generate(role, payload)

    modules.backend.generate = empty_first_revision
    record = runner.run_sample(sample())
    assert record.state == "revision_success_2"
    assert [attempt.attempt_number for attempt in record.revision_attempts] == [2]
    assert [attempt.attempt_number for attempt in record.invalid_revision_attempts] == [1]
    assert record.invalid_revision_attempts[0].analysis
    assert record.invalid_revision_attempts[0].validation_failures == [
        "revision operations must not be empty"] * 3
    second = record.revision_attempts[0]
    assert second.rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert second.operations[0].operation_id in record.operation_contributions
    assert log.get("sample") == record


def delete_to_one():
    return {"analysis": "部分标准未能区分候选回答。", "operations": [{"operation_id": f"drop{i}", "operation_type": "DELETE",
                            "target_criterion_id": f"c{i}", "new_or_updated_criterion": None}
                           for i in (1, 2, 3)]}


def test_invalid_revision_output_retries_without_consuming_an_attempt(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    original = modules.backend.generate
    calls = []

    def invalid_once(role, payload):
        if role == "critic":
            calls.append(payload)
            if len(calls) == 1:
                return delete_to_one()
        return original(role, payload)

    modules.backend.generate = invalid_once
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log,
                            max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_1"
    assert len(record.revision_attempts) == 1 and record.invalid_revision_attempts == []
    assert [operation.operation_id for operation in record.revision_attempts[0].operations] == ["op1"]
    assert len(calls) == 2
    assert "revised rubric must contain 2..8 criteria" in calls[1]["validation_feedback"]
    assert record.revision_attempts[0].validation_failures == [
        "revised rubric must contain 2..8 criteria"]
    assert log.get("sample").revision_attempts[0].validation_failures == [
        "revised rubric must contain 2..8 criteria"]
    assert calls[1]["chosen"] == sample().chosen and calls[1]["rejected"] == sample().rejected
    assert "优质回答" not in calls[1]["validation_feedback"]
    assert "较差回答" not in calls[1]["validation_feedback"]
    assert log.get("sample") == record


def test_critic_operation_model_error_retries_without_aborting(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    original = modules.backend.generate
    requests = []

    def wrong_modified_id(role, payload):
        if role == "critic":
            requests.append(payload)
            if len(requests) == 1:
                return {"analysis": "应重新检查标准。", "operations": [{
                    "operation_id": "op1", "operation_type": "MODIFY", "target_criterion_id": "c2",
                    "new_or_updated_criterion": {"criterion_id": "c3", "question": "新问题",
                                                 "importance_score": 7, "source_experiences": []},
                }]}
        return original(role, payload)

    modules.backend.generate = wrong_modified_id
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_1"
    assert len(requests) == 2
    assert "MODIFY must preserve target_criterion_id" in requests[1]["validation_feedback"]
    assert record.revision_attempts[0].validation_failures == [
        "RevisionOperation: MODIFY must preserve target_criterion_id"]
    assert log.get("sample") == record


def test_invalid_size_first_attempt_can_succeed_on_second_from_r0(tmp_path):
    modules = make_fake_modules("attempt_2_success")
    original = modules.backend.generate
    calls = []

    def invalid_first(role, payload):
        if role == "critic":
            calls.append(payload)
            if payload["attempt_number"] == 1:
                return delete_to_one()
        return original(role, payload)

    modules.backend.generate = invalid_first
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log,
                            max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_2"
    assert [attempt.attempt_number for attempt in record.revision_attempts] == [2]
    assert record.revision_attempts[0].rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert [attempt.attempt_number for attempt in record.invalid_revision_attempts] == [1]
    assert record.invalid_revision_attempts[0].analysis
    assert record.invalid_revision_attempts[0].feedback is None
    assert len(record.invalid_revision_attempts[0].validation_failures) == 3
    assert [payload["attempt_number"] for payload in calls] == [1, 1, 1, 2]
    assert calls[0]["chosen"] == sample().chosen and "previous_attempt" not in calls[0]
    assert calls[-1]["chosen"] == sample().chosen
    assert calls[-1]["previous_attempt"] is None
    assert calls[-1]["previous_invalid_attempt"]["attempt_number"] == 1
    assert len(calls[-1]["previous_invalid_attempt"]["validation_failures"]) == 3
    assert [role for role, _ in modules.backend.requests].count("judge") == 4
    assert log.get("sample") == record


def test_both_size_invalid_attempts_are_final_failure_without_judging(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    calls = []

    def always_invalid(role, payload):
        if role == "critic":
            calls.append(payload)
            return delete_to_one()
        return original(role, payload)

    original = modules.backend.generate
    modules.backend.generate = always_invalid
    log = TrajectoryLog(tmp_path)
    memory = ExperienceMemory(tmp_path)
    runner = PipelineRunner(modules, memory, log, max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "final_failure"
    assert record.revision_attempts == [] and record.reflected_candidates == []
    assert [item.attempt_number for item in record.invalid_revision_attempts] == [1, 2]
    assert [len(item.validation_failures) for item in record.invalid_revision_attempts] == [3, 3]
    assert [payload["attempt_number"] for payload in calls] == [1, 1, 1, 2, 2, 2]
    assert [role for role, _ in modules.backend.requests].count("judge") == 2
    assert memory.all() == [] and log.get("sample") == record


@pytest.mark.parametrize("operations,error", [
    ({"operations": [{"operation_id": "bad", "operation_type": "DELETE",
                     "target_criterion_id": "missing", "new_or_updated_criterion": None}]},
     "DELETE target must be in R0"),
    ({"operations": [{"operation_id": "dup", "operation_type": "DELETE",
                     "target_criterion_id": "c1", "new_or_updated_criterion": None},
                    {"operation_id": "dup", "operation_type": "DELETE",
                     "target_criterion_id": "c2", "new_or_updated_criterion": None}]},
     "conflicting or already executed revision operations"),
])
def test_invalid_executor_operations_exhaust_retries_without_aborting_run(tmp_path, operations, error):
    modules = make_fake_modules("final_failure")
    original = modules.backend.generate
    calls = []

    def invalid_second(role, payload):
        if role == "critic":
            calls.append(payload)
            if payload["attempt_number"] == 2:
                return {"analysis": "上次修订没有改善结果。", **operations}
        return original(role, payload)

    modules.backend.generate = invalid_second
    memory = ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, memory, log, max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "final_failure"
    assert [item.attempt_number for item in record.revision_attempts] == [1]
    assert not record.revision_attempts[0].judgment.success
    assert [item.attempt_number for item in record.invalid_revision_attempts] == [2]
    assert record.invalid_revision_attempts[0].validation_failures == [error] * 3
    assert [payload["attempt_number"] for payload in calls] == [1, 2, 2, 2]
    assert error in calls[-1]["validation_feedback"]
    assert len([role for role, _ in modules.backend.requests if role == "judge"]) == 4
    assert memory.all() == [] and log.get("sample") == record


def test_conflicting_first_revision_can_recover_on_second_from_r0(tmp_path):
    modules = make_fake_modules("attempt_2_success")
    original = modules.backend.generate
    calls = []

    def conflicting_first(role, payload):
        if role == "critic":
            calls.append(payload)
            if payload["attempt_number"] == 1:
                return {"analysis": "需要校正操作目标。", "operations": [{"operation_id": "dup", "operation_type": "DELETE",
                                        "target_criterion_id": "c1", "new_or_updated_criterion": None},
                                       {"operation_id": "dup", "operation_type": "DELETE",
                                        "target_criterion_id": "c2", "new_or_updated_criterion": None}]}
        return original(role, payload)

    modules.backend.generate = conflicting_first
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log,
                            max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "revision_success_2"
    assert [item.attempt_number for item in record.invalid_revision_attempts] == [1]
    assert [item.attempt_number for item in record.revision_attempts] == [2]
    assert [payload["attempt_number"] for payload in calls] == [1, 1, 1, 2]
    assert calls[-1]["previous_attempt"] is None
    assert len(calls[-1]["previous_invalid_attempt"]["validation_failures"]) == 3
    assert record.revision_attempts[0].rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert log.get("sample") == record


def test_critic_contract_error_is_recorded_without_executor_retries(tmp_path):
    modules = make_fake_modules("attempt_1_success")
    original = modules.backend.generate
    calls = []

    def bad_generator(role, payload):
        if role == "critic":
            calls.append(payload)
            return {"analysis": "诊断", "operations": "not a list"}
        return original(role, payload)

    modules.backend.generate = bad_generator
    log = TrajectoryLog(tmp_path)
    runner = PipelineRunner(modules, ExperienceMemory(tmp_path), log,
                            max_revision_output_retries=2)
    record = runner.run_sample(sample())
    assert record.state == "sample_error"
    assert record.sample_error["stage"] == "revision_1.critic"
    assert "operations must be a list" in record.sample_error["message"]
    assert len(calls) == 1 and log.get("sample") == record  # 不叠加 Executor 重试。


def test_second_revision_receives_full_failed_revision_history_but_starts_from_r0(tmp_path):
    runner, modules, _, _ = make_runner(tmp_path, "final_failure")
    record = runner.run_sample(sample())
    requests = [payload for role, payload in modules.backend.requests
                if role == "critic"]
    first, second = requests
    assert set(first) == {"task", "chosen", "rejected", "initial_rubric", "initial_judgment", "attempt_number"}
    assert second["chosen"] == sample().chosen and second["rejected"] == sample().rejected
    assert first["initial_judgment"] == record.initial_judgment.to_dict()
    assert second["initial_rubric"]["rubric_id"] == record.initial_rubric.rubric_id
    assert second["initial_judgment"] == record.initial_judgment.to_dict()
    prior = second["previous_attempt"]
    assert prior["analysis"] == record.revision_attempts[0].analysis
    assert prior["operations"][0]["operation_id"] == record.revision_attempts[0].operations[0].operation_id
    assert prior["rubric"]["rubric_id"] == record.revision_attempts[0].rubric.rubric_id
    assert prior["judgment"] == record.revision_attempts[0].judgment.to_dict()
    assert prior["validation_failures"] == record.revision_attempts[0].validation_failures
    assert second["previous_invalid_attempt"] is None
    assert "validation_feedback" not in second and record.revision_attempts[1].analysis
    assert record.revision_attempts[1].rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert "retrieved_experience_ids" not in second and "operation_contributions" not in second


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
    ({"unbound": True}, {"reflector"}),
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


@pytest.mark.parametrize("trigger", [
    "任务要求解释，而且初始标准未能区分回答",
    "初始标准未能区分回答",
])
def test_admission_fail_path_for_scripted_posthoc_trigger(tmp_path, trigger):
    runner, modules, memory, log = make_runner(tmp_path)
    original = modules.backend.generate

    def reject_mixed_trigger(role, payload):
        response = original(role, payload)
        if role == "reflector":
            response["candidates"][0]["trigger"] = trigger
        if role == "admission_checker":
            assert payload["candidate"]["trigger"] == trigger
            # 只测试模型返回 FAIL 后的写盘路径；不把 mock 当作语义判别器。
            return {"decision": "FAIL", "reason": "条件依赖修订后才能观察的信息。"}
        return response

    modules.backend.generate = reject_mixed_trigger
    record = runner.run_sample(sample())
    assert record.reflected_candidates[0].admission == "FAIL"
    assert record.reflected_candidates[0].admission_reason == "条件依赖修订后才能观察的信息。"
    assert memory.all() == [] and log.get("sample") == record
    assert "fusion_judge" not in [role for role, _ in modules.backend.requests]


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


def test_unmatched_high_confidence_experience_can_be_left_unselected(tmp_path):
    memory = ExperienceMemory(tmp_path)
    unrelated = memory.fuse(CandidateExperience(
        "seed", "辨别法律条文引用是否准确", "任务明确要求引用法律条文", ["seed-op"],
        1.0, "PASS"), "origin")
    runner = PipelineRunner(make_fake_modules("initial_success"), memory, TrajectoryLog(tmp_path))
    record = runner.run_sample(sample())
    assert memory.get(unrelated.experience_id).confidence > 0.5
    assert record.retrieved_experience_ids == [unrelated.experience_id]
    assert record.selected_experience_ids == []
    assert all(not criterion.source_experiences for criterion in record.initial_rubric.criteria)
    assert len(memory.get(unrelated.experience_id).evidence_history) == 1


def test_selected_but_unused_experience_does_not_receive_downstream_credit(tmp_path):
    memory = ExperienceMemory(tmp_path)
    seeded = memory.fuse(make_candidate(), "origin")
    modules = make_fake_modules("initial_success")
    original = modules.backend.generate

    def leave_selected_experience_unused(role, payload):
        answer = original(role, payload)
        if role == "initial_generator":
            # 模拟 Generator 判断标准仅靠 task 就足够，无需虚构来源。
            for criterion in answer["rubric"]["criteria"]:
                criterion["source_experiences"] = []
        return answer

    modules.backend.generate = leave_selected_experience_unused
    record = PipelineRunner(modules, memory, TrajectoryLog(tmp_path)).run_sample(sample())
    assert record.selected_experience_ids == [seeded.experience_id]
    assert all(not item.source_experiences for item in record.initial_rubric.criteria)
    assert len(memory.get(seeded.experience_id).evidence_history) == 1
    assert [item for role, payload in modules.backend.requests if role == "initial_generator"
            for item in payload["selected_experiences"]][0]["experience_id"] == seeded.experience_id


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
    assert not any(role in ("critic", "reflector")
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
        if role == "critic":
            return {"analysis": "必须直接检验关键要求。", "operations": [
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
    record = runner.run_sample(sample())
    previous = next(item for item in record.initial_rubric.criteria
                    if item.question == "回答是否符合要求？")
    base_answer = next(item for item in record.initial_judgment.criterion_judgments
                       if item.criterion_id == previous.criterion_id)
    revised_answer = next(item for item in record.revision_attempts[0].judgment.criterion_judgments
                          if item.criterion_id == "c5")
    assert (revised_answer.chosen, revised_answer.rejected) == (
        base_answer.chosen, base_answer.rejected)
    assert all("c5" not in [item["criterion_id"] for item in payload["criteria"]]
               for role, payload in modules.backend.requests if role == "judge")


def test_executor_validates_conflicts_and_populates_only_one_operation_list():
    initial = Rubric("r0", None, [Criterion(f"c{i}", f"Q{i}", 2, ["exp"])
                                   for i in range(4)])
    executor = RevisionExecutor()
    modified = RevisionOperation("m", "MODIFY", "c1", Criterion("c1", "new", 2, ["external"]))
    revised, executed = executor.execute(initial, [modified], 1)
    assert revised.base_rubric_id == "r0"
    assert revised.criteria[1].source_experiences == []
    assert executed[0].before_criterion == initial.criteria[1]
    assert modified.before_criterion is None  # 未改变 Critic 输出。
    with pytest.raises(ValueError, match="no-op MODIFY"):
        executor.execute(initial, [RevisionOperation("noop", "MODIFY", "c1",
                                                     Criterion("c1", "Q1", 2, []))], 1)
    with pytest.raises(ValueError, match="conflicting"):
        executor.execute(initial, [modified, modified], 1)
    two, _ = executor.execute(initial, [RevisionOperation(f"drop{i}", "DELETE", f"c{i}", None)
                                        for i in (1, 2)], 1)
    assert len(two.criteria) == 2
    with pytest.raises(ValueError, match="2..8"):
        executor.execute(initial, [RevisionOperation(f"drop{i}", "DELETE", f"c{i}", None)
                                   for i in (1, 2, 3)], 1)
    with pytest.raises(ValueError, match="2..8"):
        executor.execute(initial, [RevisionOperation(f"add{i}", "ADD", None,
                                                     Criterion(f"new{i}", "new", 3, []))
                                   for i in range(5)], 1)
    with pytest.raises(ValueError, match="new criterion ID"):
        executor.execute(initial, [RevisionOperation("add", "ADD", None,
                                                     Criterion("c1", "new", 3, []))], 1)


def test_critic_format_exhaustion_records_unknown_analysis_then_second_succeeds(tmp_path):
    from verge.llm import LLMFormatError
    runner, modules, memory, log = make_runner(tmp_path, 'attempt_2_success')
    original = modules.backend.generate
    calls = []

    def generate(role, payload):
        if role == 'critic':
            calls.append(payload)
            if payload['attempt_number'] == 1:
                raise LLMFormatError('critic: invalid structured output after 3 attempts')
        return original(role, payload)

    modules.backend.generate = generate
    record = runner.run_sample(sample())
    assert record.state == 'revision_success_2'
    assert len(calls) == 2  # 格式重试已经耗尽，不能再叠加 2 次操作重试。
    failed = record.invalid_revision_attempts[0]
    assert failed.analysis is None and failed.feedback is None
    assert failed.validation_failures == ['critic: invalid structured output after 3 attempts']
    assert calls[1]['previous_invalid_attempt']['analysis'] is None
    assert calls[1]['previous_attempt'] is None
    assert len(record.revision_attempts) == 1 and record.revision_attempts[0].attempt_number == 2
    assert memory.all() and log.get('sample') == record


def test_both_critic_format_failures_do_not_abort_next_sample(tmp_path):
    from verge.llm import LLMFormatError
    runner, modules, memory, log = make_runner(tmp_path)
    original = modules.backend.generate
    calls = []

    def generate(role, payload):
        if role == 'critic':
            calls.append(payload['attempt_number'])
            raise LLMFormatError('critic: invalid structured output after 3 attempts')
        return original(role, payload)

    modules.backend.generate = generate
    records = runner.run([sample('a'), sample('b')])
    assert calls == [1, 2, 1, 2]
    assert all(r.state == 'final_failure' and len(r.invalid_revision_attempts) == 2 for r in records)
    assert all(not r.revision_attempts and not r.reflected_candidates for r in records)
    assert memory.all() == [] and len(list(log.all())) == 2
    assert [r for r, _ in modules.backend.requests] == ['initial_generator', 'judge', 'judge'] * 2


def test_critic_network_error_still_aborts(tmp_path):
    from verge.llm import LLMRequestError
    runner, modules, memory, log = make_runner(tmp_path)
    original = modules.backend.generate

    def generate(role, payload):
        if role == 'critic':
            raise LLMRequestError('critic: HTTP 401')
        return original(role, payload)

    modules.backend.generate = generate
    with pytest.raises(LLMRequestError, match='HTTP 401'):
        runner.run_sample(sample())
    assert list(log.all()) == [] and memory.all() == []
