from copy import deepcopy
from dataclasses import replace
from threading import Event
import json, time
import pytest
from verge.memory import ExperienceMemory, TrajectoryLog, atomic_json
from verge.modules import make_fake_modules, HybridRetriever
from verge.pipeline import PipelineRunner
from verge.scheduling import run_training_stream, commit_sample, recover_sample_commit
from verge.models import (
    ExperienceReview,
    CriticProposal,
    Experience,
    Method,
    ExperienceUpdate,
)
from verge.calculator import rebuild_quality
from test_v2 import sample, experience, params, app, reports


def seed(path):
    m = ExperienceMemory(path, parameters=params())
    e = experience()
    s = sample(99)
    e.source_sample_id = s.sample_id
    m.add(e, s, ExperienceReview(True, "测试"))
    return m, e


def test_rolling_slot_refilled_before_slow_sample_finishes(tmp_path):
    started_next = Event()
    seen = {}
    m = ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)

    def factory(s):
        mods = make_fake_modules()
        original = mods.initial_generator.generate

        def generate(task, selected, *, round_number=0):
            if not round_number:
                seen[s.sample_id] = [e.experience_id for e in selected]
                if s.sample_id == "s1":
                    assert started_next.wait(10)
                if s.sample_id == "s2":
                    started_next.set()
            return original(task, selected, round_number=round_number)

        mods.initial_generator.generate = generate
        return mods, lambda: None

    rows = run_training_stream(
        [sample(i) for i in range(3)],
        factory,
        m,
        log,
        workers=2,
        runner_options={"revision_enabled": False},
    )
    assert started_next.is_set() and seen["s2"] and not seen["s0"]
    assert rows[0].sample_id == "s0" and len(m.all()) == 3
    assert [r.completion_sequence for r in log.all()] == [1, 2, 3]
    assert all(r.snapshot_digest for r in rows)


def test_merge_does_not_double_count_worker_log_odds(tmp_path):
    p = params()
    m = ExperienceMemory(tmp_path, parameters=p)
    log = TrajectoryLog(tmp_path)
    run_training_stream(
        [sample()],
        lambda s: (make_fake_modules(), lambda: None),
        m,
        log,
        workers=1,
        runner_options={"revision_enabled": False},
    )
    e = m.all()[0]
    expected = deepcopy(e)
    rebuild_quality(expected, p)
    assert e.q_posterior == pytest.approx(expected.q_posterior)
    assert e.q_log_odds == pytest.approx(expected.q_log_odds)
    assert e.evidence_counts["source_validation"] == 0
    assert not e.applications and e.q_posterior == p.q_prior


def test_multisample_evidence_accumulates_at_current_q(tmp_path):
    m, e = seed(tmp_path)
    run_training_stream(
        [sample(i) for i in range(5)],
        lambda s: (make_fake_modules("initial_success"), lambda: None),
        m,
        TrajectoryLog(tmp_path),
        workers=5,
        runner_options={"revision_enabled": False},
    )
    e = m.get(e.experience_id)
    expected = deepcopy(e)
    rebuild_quality(expected, params())
    assert len(e.applications) == 5 and e.q_posterior == pytest.approx(
        expected.q_posterior
    )
    assert len({r.completion_sequence for r in TrajectoryLog(tmp_path).all()}) == 5


def test_pending_sample_commit_recovers_without_rerunning_calls(tmp_path, monkeypatch):
    m = ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    import os

    replace_file = os.replace

    def fail_log(src, dst):
        if str(dst) == str(log.path):
            data = __import__("pathlib").Path(src).read_bytes()
            log.path.write_bytes(data[: len(data) // 2])
            raise OSError("partial write")
        return replace_file(src, dst)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", fail_log)
        with pytest.raises(OSError):
            run_training_stream(
                [sample()],
                lambda s: (make_fake_modules(), lambda: None),
                m,
                log,
                runner_options={"revision_enabled": False},
            )
    assert (tmp_path / "pending_sample_commit.json").exists() and len(m.all()) == 1
    recover_sample_commit(m, log)
    recover_sample_commit(m, log)
    assert (
        len(list(log.all())) == 1
        and not (tmp_path / "pending_sample_commit.json").exists()
    )
    assert len(m.all()[0].applications) == 0


def test_retired_version_accepts_inflight_evidence_without_transferring(tmp_path):
    from verge.models import RevisionRecord

    m, e = seed(tmp_path)
    a = app(e, 0)
    a.reports = reports()
    child = Experience(
        "child",
        Method("新适用", "新方法"),
        e.source_sample_id,
        parent_id=e.experience_id,
        status="candidate",
    )
    event = RevisionRecord(
        "event", e.experience_id, "s1", [], [], {}, candidate=child, accepted=True
    )
    m.finish_revision(event)
    m.record(a)
    assert (
        len(m.get(e.experience_id).applications) == 1
        and m.get("child").applications == []
    )
    assert m.get("child").source_sample is not None


def test_multi_experience_partial_review_and_one_rewrite(tmp_path):
    m = ExperienceMemory(tmp_path)
    mods = make_fake_modules()
    original = mods.critic.extract

    def extract(*a, **kw):
        result = original(*a, **kw)
        e, refs, why = deepcopy(result[0])
        e.experience_id = "bad"
        e.method.principle = "多个方法"
        return [*result, (e, refs, why)]

    mods.critic.extract = extract
    original_review = mods.experience_checker.check_many

    def review(sample,evidence,es):
        result=original_review(sample,evidence,es)
        for e in es:
            if e.experience_id=="bad":result[e.experience_id]=ExperienceReview(False,"多个方法")
        return result
    mods.experience_checker.check_many=review
    rec = PipelineRunner(
        mods, m, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    entries = rec.proposal_attempts[0].extractions
    assert len(entries) == 2 and [x["admitted"] for x in entries] == [True, False]
    assert len(entries[1]["reviews"]) == 2 and len(m.all()) == 1


def test_critic_final_booleans_determine_margin_not_raw_judge(tmp_path):
    mods = make_fake_modules()
    judge = mods.judge.judge

    def wrong(r, *a, **kw):
        result = judge(r, *a, **kw)
        if r.rubric_id.startswith("repair_"):
            for j in result.criterion_judgments:
                j.chosen = False
                j.rejected = False
            from verge.calculator import calculate_rubric_judgment

            return calculate_rubric_judgment(r, result.criterion_judgments)
        return result

    mods.judge.judge = wrong
    assess = mods.critic.assess

    def correct(s, r, j, c=None):
        if r.rubric_id.startswith("repair_"):
            from verge.calculator import calculate_rubric_judgment
            from verge.models import CriterionJudgment

            return (
                calculate_rubric_judgment(
                    r,
                    [
                        CriterionJudgment(
                            x.criterion_id, True, False, "有依据", "不满足"
                        )
                        for x in r.criteria
                    ],
                ),
                "更正Judge",
            )
        return assess(s, r, j, c)

    mods.critic.assess = correct
    m = ExperienceMemory(tmp_path)
    rec = PipelineRunner(
        mods, m, TrajectoryLog(tmp_path), revision_enabled=False
    ).run_sample(sample())
    a = rec.proposal_attempts[0]
    assert (
        a.repair_judgment.margin == 0
        and a.final_judgment.margin == 1
        and a.margin_gain == 1
        and len(m.all()) == 1
    )


def test_embedding_single_flight_across_workers():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    class Emb:
        calls = 0

        def embed(self, t):
            self.calls += 1
            time.sleep(0.02)
            return (1.0, 0.0)

    b = Emb()
    r = HybridRetriever(b)
    with ThreadPoolExecutor(max_workers=5) as pool:
        vs = list(pool.map(lambda _: r.fork()._vector("same"), range(5)))
    assert b.calls == 1 and len(vs) == 5


def test_revision_runs_without_blocking_new_samples(tmp_path):
    m, e = seed(tmp_path)
    next_started = Event()
    revision_started = Event()

    def factory(s):
        mods = make_fake_modules("initial_success")
        gen = mods.initial_generator.generate

        def generate(task, selected, *, round_number=0):
            if task == sample(4).task:
                next_started.set()
            return gen(task, selected, round_number=round_number)

        mods.initial_generator.generate = generate
        mods.checkers.check = lambda a: reports(("correct", "error", "correct"))

        def update(*a, **kw):
            revision_started.set()
            assert next_started.wait(10)
            return ExperienceUpdate("keep", None, [], "保留")

        mods.experience_updater.update = update
        return mods, lambda: None

    run_training_stream(
        [sample(i) for i in range(6)],
        factory,
        m,
        TrajectoryLog(tmp_path),
        workers=2,
        runner_options={"error_threshold": 3},
    )
    assert (
        revision_started.is_set()
        and next_started.is_set()
        and m.get(e.experience_id).status == "active"
    )
    assert m.revisions


def test_weight_only_changes_reuse_judgments_but_recompute_scores():
    from verge.models import Rubric, Criterion
    from verge.modules import make_fake_modules

    mods = make_fake_modules("initial_success")
    first = Rubric(
        "a", [Criterion("a1", "验证通过", 0.5), Criterion("a2", "基础核验", 0.5)]
    )
    second = Rubric(
        "b", [Criterion("b1", "验证通过", 0.8), Criterion("b2", "基础核验", 0.2)]
    )
    j1 = mods.judge.judge(first, "甲", "乙", task="任务")
    calls = len(mods.backend.calls)
    j2 = mods.judge.judge(second, "甲", "乙", task="任务")
    assert len(mods.backend.calls) == calls and j1.margin == 0.5 and j2.margin == 0.8
    assert [x.criterion_id for x in j2.criterion_judgments] == ["b1", "b2"]


def test_real_transport_accepts_repair_history(monkeypatch):
    import httpx, json
    from verge.llm import OpenAICompatibleBackend

    monkeypatch.setenv("VERGE_TEST_KEY", "test")
    client = httpx.Client(
        base_url="https://example.test/v1/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"analysis": "a", "criteria": [], "changes": []}
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        ),
    )
    b = OpenAICompatibleBackend(
        "test",
        base_url="https://example.test/v1",
        api_key_env="VERGE_TEST_KEY",
        client=client,
        max_retries=0,
    )
    history = []
    b.generate("critic_repair", {"task": "任务"}, conversation=history)
    assert history[-1]["role"] == "assistant"
    client.close()


def test_updater_missing_guides_or_undeclared_changes_rejected():
    from verge.modules import LLMExperienceUpdater
    from verge.llm import LLMFormatError

    class B:
        def generate(self, *a, **kw):
            return {
                "action": "revise",
                "method": {"trigger": "different", "principle": "核对"},
                "target_fields": ["trigger"],
                "reason": "调整",
            }

    with pytest.raises(LLMFormatError, match="five-field"):
        LLMExperienceUpdater(B(), retries=0).update(experience(), [], ["错误"])


def test_recovered_sample_not_executed_again(tmp_path, monkeypatch):
    import os

    m = ExperienceMemory(tmp_path)
    log = TrajectoryLog(tmp_path)
    real = os.replace

    def fail(src, dst):
        if str(dst) == str(log.path):
            raise OSError("before log")
        return real(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail)
        with pytest.raises(OSError):
            run_training_stream(
                [sample()],
                lambda s: (make_fake_modules(), lambda: None),
                m,
                log,
                runner_options={"revision_enabled": False},
            )

    def forbidden(s):
        raise AssertionError("must not repeat completed sample")

    run_training_stream(
        [sample()], forbidden, m, log, runner_options={"revision_enabled": False}
    )
    assert len(list(log.all())) == 1 and len(m.all()) == 1


def test_explicit_recovery_certifies_copy_without_llm_calls(tmp_path):
    import yaml
    from pathlib import Path
    from verge.cli import run, recover_run

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    cfg["dataset"]["config"]["paths"] = {"train": "configs/data/demo_train.jsonl"}
    cfg["run"]["output_dir"] = str(tmp_path / "run")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run(p)
    manifest = tmp_path / "run/run_manifest.json"
    r = json.loads(manifest.read_text())
    r["state"] = "running"
    manifest.write_text(json.dumps(r))
    result = recover_run(tmp_path / "run", tmp_path / "recovered")
    assert (
        result["model_calls"] == 0
        and result["state"] == "paused"
        and result["train"] == 1
    )
    assert json.loads(manifest.read_text())["state"] == "running"


def test_incomplete_recovery_refuses_to_repeat_model_calls(tmp_path):
    import yaml
    from pathlib import Path
    from verge.cli import run, recover_run, ConfigError

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    cfg["dataset"]["config"]["paths"] = {"train": "configs/data/demo_train.jsonl"}
    cfg["run"]["output_dir"] = str(tmp_path / "run")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run(p)
    atomic_json(
        tmp_path / "run/artifacts/sample_jobs.json",
        [{"sample_id": "unfinished", "state": "running"}],
    )
    with pytest.raises(ConfigError, match="refusing repeat calls"):
        recover_run(tmp_path / "run", tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()
