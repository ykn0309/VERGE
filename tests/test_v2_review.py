from verge.models import ExperienceReview

"""Review regressions: independent task identity, ledger rebuilding and revision audit."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pytest
import yaml
from test_v2 import sample, experience, app, params, reports, rubric, judged
from test_v2_io import config, write_config
from verge.calculator import rebuild_quality, update_quality, application
from verge.config import load_config
from verge.cli import run, components, ConfigError, reestimate_run
from verge.models import (
    CheckerReport,
    CheckerDetail,
    ExperienceUpdate,
    Method,
    PreferenceSample,
)
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.pipeline import revision_reasons, select_revision_cases, PipelineRunner
from verge.modules import make_fake_modules, LLMCheckers
from verge.prompts import experience_payload, evidence_payload


def admitted(tmp_path, p=None):
    m = ExperienceMemory(tmp_path, parameters=p)
    e = experience()
    a = app(e, 0, phase="source_validation")
    a.sample = replace(a.sample, sample_id="source")
    m.add(e, a.sample, ExperienceReview(True, "测试核对通过"))
    return m, m.get(e.experience_id)


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.yaml")))
def test_every_shipped_yaml_loads_and_assembles(path, monkeypatch):
    import verge.cli as cli
    from verge.llm import FakeLLMBackend

    monkeypatch.setattr(cli, "backend", lambda spec: FakeLLMBackend())

    class Embedding:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(cli, "OpenAICompatibleEmbedding", Embedding)
    modules = components(load_config(path))
    assert modules.initial_generator


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limit_before_creating_output(tmp_path, limit):
    path = write_config(tmp_path, config(tmp_path))
    with pytest.raises(ConfigError, match="positive"):
        run(path, stop_after_train=limit)
    assert not (tmp_path / "run").exists()


def test_full_limit_runs_frozen_test_and_train_only_is_explicit(tmp_path):
    c = config(tmp_path)
    p = write_config(tmp_path, c)
    n = sum(
        bool(line.strip())
        for line in Path(c["dataset"]["config"]["paths"]["train"])
        .read_text()
        .splitlines()
    )
    result = run(p, stop_after_train=n)
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert result["test"] > 0 and manifest["state"] == "completed"
    c["run"]["output_dir"] = str(tmp_path / "only")
    del c["dataset"]["config"]["paths"]["test"]
    run(write_config(tmp_path, c), stop_after_train=n)
    manifest = json.loads((tmp_path / "only/run_manifest.json").read_text())
    assert (
        manifest["state"] == "training_completed"
        and manifest["evaluation_mode"] == "training_only"
    )


def test_normalized_task_overlap_rejected(tmp_path):
    c = config(tmp_path)
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train.write_text(sample(0).to_json() + "\n")
    test.write_text(
        replace(
            sample(0), sample_id="alias", task=" \n" + sample(0).task + "  "
        ).to_json()
        + "\n"
    )
    c["dataset"]["config"]["paths"] = {"train": str(train), "test": str(test)}
    with pytest.raises(ConfigError, match="overlap"):
        run(write_config(tmp_path, c))


def test_duplicate_task_and_source_survive_reestimate(tmp_path):
    p = params()
    m, e = admitted(tmp_path, p)
    source_alias = app(e, 0)
    source_alias.sample = replace(
        source_alias.sample, sample_id="source-alias", task=" 任务0\n"
    )
    source_alias.reports = reports()
    m.record(source_alias)
    first = app(e, 1)
    first.reports = reports()
    m.record(first)
    alias = app(e, 1)
    alias.sample = replace(alias.sample, sample_id="alias", task=" 任务1\n")
    alias.reports = reports()
    m.record(alias)
    before = m.get(e.experience_id)
    m.reestimate(p)
    after = m.get(e.experience_id)
    assert after.q_posterior == before.q_posterior
    assert (
        after.evidence_counts
        == before.evidence_counts
        == {"source_validation": 0, "downstream": 1, "replay": 0}
    )
    assert len(after.downstream) == 1
    assert [a.update_reason for a in after.applications] == [
        "duplicate_task",
        "updated",
        "duplicate_task",
    ]
    assert (
        not after.applications[-1].likelihoods
        and after.applications[-1].parameter_version is None
    )
    m.reestimate(p)
    assert m.get(e.experience_id).to_dict() == after.to_dict()


def test_old_errors_consumed_and_new_errors_accumulate(tmp_path):
    m, e = admitted(tmp_path)
    for n in range(1, 4):
        a = app(e, n)
        a.reports = reports(("correct", "error", "correct"))
        m.record(a)
    assert revision_reasons(m.get(e.experience_id)) == ["generator_errors"]
    m.mark_checked(e.experience_id, ["generator_errors"])
    for n in range(4, 7):
        a = app(e, n)
        a.reports = reports()
        m.record(a)
    assert revision_reasons(m.get(e.experience_id)) == []
    m.mark_checked(e.experience_id)
    for n in range(7, 10):
        a = app(e, n)
        a.reports = reports(("correct", "error", "correct")) if n < 9 else reports()
        m.record(a)
    assert revision_reasons(m.get(e.experience_id)) == []
    m.mark_checked(e.experience_id)
    for n in range(10, 13):
        a = app(e, n)
        a.reports = reports(("correct", "error", "correct")) if n == 10 else reports()
        m.record(a)
    current = m.get(e.experience_id)
    assert revision_reasons(current) == ["generator_errors"]
    cases, kinds = select_revision_cases(
        current, {current.experience_id: current}, reasons=["generator_errors"]
    )
    problems = {
        a.sample.sample_id for a in cases if kinds[a.sample.sample_id] == "problem"
    }
    assert problems == {"s7", "s8", "s10"}


def test_prior_only_never_quality_risk(tmp_path):
    p = replace(params(), q_prior=0.4)
    m, e = admitted(tmp_path, p)
    for n in range(1, 4):
        a = app(e, n)
        a.reports = reports(("unknown", "correct", "correct"))
        m.record(a)
    assert revision_reasons(m.get(e.experience_id)) == []
    a = app(e, 4)
    a.reports = reports()
    m.record(a)
    current = m.get(e.experience_id)
    current.q_posterior = 0.3
    assert "quality_risk" in revision_reasons(current)


def test_parameter_contents_and_status_change_require_explicit_rebuild(tmp_path):
    p = params()
    m, e = admitted(tmp_path, p)
    altered = replace(p, q_prior=0.4)
    assert p.fingerprint != altered.fingerprint
    for incoming in (altered, replace(p, status="experimental"), None):
        with pytest.raises(ValueError, match="parameter contents"):
            ExperienceMemory(tmp_path, parameters=incoming)
    m.reestimate(altered)
    assert len(m.parameter_events) == 1
    m.reestimate(altered)
    assert len(m.parameter_events) == 1
    assert (
        ExperienceMemory(tmp_path, parameters=altered).get(e.experience_id).q_posterior
        == 0.4
    )
    m.reestimate(replace(p, status="experimental"))
    assert m.get(e.experience_id).q_posterior is None


def test_reestimate_cli_keeps_source_immutable_and_certifies_copy(tmp_path):
    c = config(tmp_path)
    run(write_config(tmp_path, c))
    source = tmp_path / "run"
    raw = (source / "artifacts/experience_memory.json").read_bytes()
    p = tmp_path / "params.json"
    p.write_text(params().to_json())
    target = tmp_path / "reestimated"
    result = reestimate_run(source, p, target)
    assert result["events"] == 1
    assert (source / "artifacts/experience_memory.json").read_bytes() == raw
    manifest = json.loads((target / "run_manifest.json").read_text())
    assert manifest["reestimated_from"]["run"] == str(source)
    memory = ExperienceMemory(target / "artifacts", parameters=params())
    for e in memory.all():
        assert e.parameter_digest == params().fingerprint


def test_checker_details_cannot_hide_errors():
    detail = CheckerDetail("c1", "error", "错误依据")
    with pytest.raises(ValueError, match="top-level error"):
        CheckerReport("generator", "correct", "汇总", [detail])
    report = CheckerReport("generator", "error", "汇总", [detail])
    assert CheckerReport.from_dict(report.to_dict()) == report


def test_zero_weight_citation_gets_checks_and_unused_reason():
    e = experience()
    r = rubric(e.experience_id, weights=(0, 1))
    r.criteria[1].source_experiences = []
    a = application(sample(), e, r, judged(r))
    assert (
        a.cited
        and a.feedback == "unused"
        and a.no_evidence_reason == "zero_weight_only"
    )
    checked = make_fake_modules().checkers.check(a)
    assert len(checked) == 3
    a.reports = checked
    update_quality(e, a, params())
    assert a.update_reason == "unused" and e.q_posterior is None


def test_real_replay_preserves_recall_selection_and_isolates_index(tmp_path):
    m, e = admitted(tmp_path)
    modules = make_fake_modules()
    modules.retriever.last_trace = [{"sentinel": True}]
    runner = PipelineRunner(modules, m, TrajectoryLog(tmp_path))
    original = m.get(e.experience_id).to_dict()
    record = runner._replay(sample(5), e, m.all(active_only=True))
    assert record.recalled and record.selected
    assert record.retrieval_rank == 1
    assert modules.retriever.last_trace == [{"sentinel": True}]
    assert m.get(e.experience_id).to_dict() == original
    # A successful generation which declines the experience is valid replay with explicit state.
    modules.selector.select = lambda *args, **kwargs: []
    record = runner._replay(sample(5), e, m.all(active_only=True))
    assert record.recalled and not record.selected and not record.cited
    assert record.no_evidence_reason == "not_selected"


def test_projection_exposes_program_evidence_and_source_counts():
    e = experience()
    a = app(e, 1)
    a.reports = reports()
    update_quality(e, a, params())
    e.applications.append(a)
    replay = app(e, 2, phase="replay")
    replay.reports = reports()
    update_quality(e, replay, params())
    e.applications.append(replay)
    payload = experience_payload(e, quality=True)
    assert payload["effective_tasks"] == 1 and payload["replay_tasks"] == 1
    projected = evidence_payload(a)
    assert (
        projected["observation"] == "correct"
        and projected["counted_as_quality_evidence"]
    )
    assert (
        projected["outcomes"] == a.outcomes
        and projected["positive_weight"] == a.positive_weight
    )


def test_revise_requires_target_fields():
    with pytest.raises(ValueError, match="target_fields"):
        ExperienceUpdate("revise", Method("t", "p"), [], "修订")


def test_dataset_rejects_same_task_with_different_ids(tmp_path):
    from verge.dataset import JsonlDataset

    path = tmp_path / "rows.jsonl"
    a = sample(0)
    alias = replace(a, sample_id="different", task="\n  " + a.task + "\t")
    path.write_text(a.to_json() + "\n" + alias.to_json() + "\n")
    with pytest.raises(ValueError, match="duplicate normalized task"):
        list(JsonlDataset({"train": path}).load("train"))


def test_accepted_revision_snapshot_matches_birth_state_and_lineage(tmp_path):
    from verge.models import RevisionRecord, Experience

    memory, parent = admitted(tmp_path)
    child = Experience(
        "child",
        Method("条件", "改进方法"),
        "source",
        parent_id=parent.experience_id,
        status="candidate",
    )
    event = RevisionRecord(
        "event",
        parent.experience_id,
        "trigger",
        [],
        [],
        {},
        candidate=child,
        accepted=True,
    )
    saved = memory.finish_revision(event)
    assert (
        saved.candidate.status == "active"
        and saved.candidate.created_by_event_id == "event"
    )
    active = memory.get("child")
    assert active.to_dict() == saved.candidate.to_dict()
    data = json.loads(memory.path.read_text())
    data["experiences"][0]["retired_by_event_id"] = "missing"
    memory.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="retirement"):
        ExperienceMemory(tmp_path)


def test_updater_receives_actual_case_kind_and_program_aggregates():
    from verge.modules import LLMExperienceUpdater
    from verge.llm import FakeLLMBackend

    b = FakeLLMBackend()
    seen = []
    b.trace_hook = seen.append
    e = experience()
    a = app(e, 1)
    a.reports = reports()
    update_quality(e, a, params())
    LLMExperienceUpdater(b).update(
        e, [a], ["quality_risk"], case_kinds={a.sample.sample_id: "problem"}
    )
    payload = seen[0]["input"]["cases"][0]
    assert payload["case_kind"] == "problem" and payload["observation"] == "correct"
    assert payload["counted_as_quality_evidence"] and payload["round_number"] == 0


def test_zero_weight_checker_errors_do_not_hide_valid_other_error():
    from verge.calculator import aggregate_reports

    r = CheckerReport(
        "generator", "error", "细项有错", [CheckerDetail("c1", "error", "原文依据")]
    )
    assert (
        aggregate_reports([r, CheckerReport("judge", "failed", "timeout")]) == "error"
    )


def test_embedding_trace_preserves_success_and_failure(monkeypatch):
    import httpx
    from verge.llm import OpenAICompatibleEmbedding, LLMFormatError

    monkeypatch.setenv("TEST_EMBED_KEY", "not-a-secret")
    traces = []
    metadata = []

    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [1.0, 2.0] + [0.0] * 62}],
                "usage": {"total_tokens": 2},
            },
        )

    client = httpx.Client(
        base_url="https://example.com/v1/",
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    emb = OpenAICompatibleEmbedding(
        "model",
        base_url="https://example.com/v1",
        api_key_env="TEST_EMBED_KEY",
        dimensions=64,
        client=client,
    )
    emb.trace_hook = traces.append
    emb.metadata_hook = metadata.append
    assert emb.embed("原始文本") == (1.0, 2.0) + (0.0,) * 62
    assert traces[0]["input"]["text"] == "原始文本"
    assert traces[0]["output"] == [1.0, 2.0] + [0.0] * 62
    client.close()
    client = httpx.Client(
        base_url="https://example.com/v1/",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": []})
        ),
        follow_redirects=False,
    )
    emb = OpenAICompatibleEmbedding(
        "model",
        base_url="https://example.com/v1",
        api_key_env="TEST_EMBED_KEY",
        dimensions=64,
        client=client,
    )
    emb.trace_hook = traces.append
    emb.metadata_hook = metadata.append
    with pytest.raises(LLMFormatError):
        emb.embed("失败文本")
    assert traces[-1]["input"]["text"] == "失败文本" and "error" in traces[-1]
    assert "not-a-secret" not in json.dumps(traces)
    client.close()


@pytest.mark.parametrize(
    "states", [((True, False), (False, True)), ((True, True), (False, False))]
)
def test_both_net_ties_update_q_and_count_as_downstream(tmp_path, states):
    memory, e = admitted(tmp_path, params())
    for n in range(1, 4):
        a = app(e, n, states=states)
        a.reports = reports()
        memory.record(a)
        assert a.feedback == "tie" and a.update_reason == "updated"
        assert a.no_evidence_reason is None
    current = memory.get(e.experience_id)
    assert len(current.downstream) == 3
    assert current.evidence_counts["downstream"] == 3
    assert revision_reasons(current) is not None
    from verge.calculator import replay_score

    assert replay_score(current.downstream[-1]) == 0
    q = current.q_posterior
    memory.reestimate(params())
    assert memory.get(e.experience_id).q_posterior == pytest.approx(q)


def test_zero_weight_all_cited_still_unused_while_positive_weight_tie_is_evidence():
    e = experience()
    r = rubric(e.experience_id, weights=(0, 1))
    r.criteria[1].source_experiences = []
    unused = application(sample(), e, r, judged(r))
    assert unused.cited and unused.feedback == "unused"
    assert unused.no_evidence_reason == "zero_weight_only"
    r.criteria[0].weight = 1
    tied = application(sample(), e, r, judged(r, ((True, True), (True, False))))
    assert tied.feedback == "tie" and tied.no_evidence_reason is None


def test_old_probability_table_definition_is_rejected():
    from verge.calculator import ProbabilityModel

    data = params().to_dict()
    del data["metadata"]["feedback_protocol"]
    with pytest.raises(ValueError, match="feedback_protocol"):
        ProbabilityModel.from_dict(data)
