"""用 Fake Modules 覆盖 Runner 的训练分支、隔离边界与写盘时机。"""

from dataclasses import replace

import pytest

from verge.models import CandidateExperience
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import Criterion, PreferenceSample, RevisionOperation, Rubric
from verge.modules import JudgeOutputError, make_fake_modules
from verge.pipeline import PipelineRunner


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
    assert attempt.operations == []
    assert attempt.analysis and attempt.feedback
    assert record.operation_contributions == {}
    result = record.reflected_candidates[0]
    assert result.linked_operation_ids == []
    assert result.credit is None
    assert (result.admission, result.fusion) == ("PASS", "ADD")
    stored = memory.get(result.experience_id)
    assert stored.alpha == pytest.approx(1 + .25 * (attempt.judgment.margin - record.initial_judgment.margin))
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
    assert failed.rubric.base_rubric_id == record.initial_rubric.rubric_id
    assert successful.rubric.base_rubric_id == failed.rubric.rubric_id
    assert successful.operations == []
    assert record.operation_contributions == {}
    assert record.reflected_candidates[0].linked_operation_ids == []
    assert [item.experience_id for item in memory.all()] == [record.reflected_candidates[0].experience_id]
    assert log.get("sample") == record
    assert [payload["current_rubric"]["rubric_id"] for role, payload in
            modules.backend.requests if role == "critic"] == ["r0", failed.rubric.rubric_id]




def delete_to_one():
    return {"analysis": "部分标准未能区分候选回答。", "operations": [{"operation_id": f"drop{i}", "operation_type": "DELETE",
                            "target_criterion_id": f"c{i}", "new_or_updated_criterion": None}
                           for i in (1, 2, 3)]}














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
    assert "structured fields" in record.sample_error["message"]
    assert len(calls) == 1 and log.get("sample") == record  # 不叠加 Executor 重试。




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
    assert rejected.credit is None and rejected.admission == "FAIL"
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
    return CandidateExperience("seed", "参考经验", "任务包含明确的关键要求。", [],
                               None, "PASS")


def test_unmatched_high_confidence_experience_can_be_left_unselected(tmp_path):
    memory = ExperienceMemory(tmp_path)
    unrelated = memory.fuse(CandidateExperience(
        "seed", "辨别法律条文引用是否准确", "任务明确要求引用法律条文", [],
        None, "PASS"), "origin")
    memory.apply_downstream_credits("verified", {unrelated.experience_id: 1.0})
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
    assert len(memory.get(seeded.experience_id).evidence_history) == 0
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
    assert stored.alpha == pytest.approx(1 + 2 * (2 / 12) * 0.5)
    assert [item.sample_id for item in stored.evidence_history] == ["one", "two"]
    assert [item.evidence_type for item in stored.evidence_history] == [
        "downstream_credit", "downstream_credit"]


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
