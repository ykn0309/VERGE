"""Atomic ownership, category partition/retrieval and neutral feedback boundaries."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from experience_helpers import seed, case_for
from test_bounded_experience import existing_runner
from test_delayed_experience import directional_rubric
from test_pipeline import sample
from test_real_llm import backend, result
from verge.calculator import calculate_experience_credits
from verge.llm import LLMSampleError, LLMRequestError, FakeLLMBackend
from verge.memory import ExperienceMemory
from verge.models import (CandidateExperience, CategoryDraft, CategoryOrganization,
                          ExperienceCategory, Experience, ExperienceUpdate)
from verge.modules import LLMCategoryOrganizer, LLMSelector, VectorRetriever, retrieval_units
from verge.prompts import (ExperienceView, NumberedIDCodec, category_organization_payload,
                           experience_update_payload, generator_payload, selector_payload)


def pool(tmp_path, count=5):
    memory = ExperienceMemory(tmp_path)
    for i in range(count):
        seed(memory, CandidateExperience(f"candidate-long-{i}", f"方法{i}", "*"), f"origin{i}")
    return memory


def one_group(memory, *, category_id=None):
    return CategoryOrganization([CategoryDraft(category_id, "依据核验", "核验材料和事实的依据边界",
        [e.experience_id for e in memory.all()])], [])


def test_neutral_r0_keeps_beta_capacity_and_window_but_records_both_kinds(tmp_path):
    runner, eid = existing_runner(tmp_path)
    initial = runner.memory.get(eid)
    active = {"direction": "both_true"}
    runner.modules.initial_generator.generate = lambda *a: directional_rubric(eid, active["direction"])[0]
    runner.modules.judge.judge = lambda *a, **kw: directional_rubric(eid, active["direction"])[1]
    for i in range(15):
        active["direction"] = "both_true" if i % 2 else "both_false"
        record = runner.run_sample(sample(str(i)))
        assert not record.experience_updates
        assert record.downstream_credits[eid].amount == 0
    e = ExperienceMemory(tmp_path).get(eid)
    assert (e.alpha, e.beta, e.cases, e.pending_feedback_count) == (initial.alpha, initial.beta, initial.cases, 0)
    assert len(e.evidence_history) == 1 and len(e.usage_history) == 16
    assert sum(u.both_true for u in e.usage_history) == 7
    assert sum(u.both_false for u in e.usage_history) == 8


def test_mixed_case_only_exposes_discriminating_criteria_to_updater(tmp_path):
    from verge.models import Criterion, CriterionJudgment, Rubric, ExperienceCase
    from verge.calculator import calculate_rubric_judgment
    m = pool(tmp_path, 1)
    e = m.all()[0]
    r = Rubric("r", [Criterion("p", "正确区分", .2, [e.experience_id]),
        Criterion("tt", "双方满足", .4, [e.experience_id]),
        Criterion("ff", "双方未满足", .4, [e.experience_id])])
    j = calculate_rubric_judgment(r, [CriterionJudgment("p", True, False, "甲", "乙"),
        CriterionJudgment("tt", True, True, "甲", "乙"), CriterionJudgment("ff", False, False, "甲", "乙")])
    case = ExperienceCase("next", "原任务", "完整甲", "完整乙", e.experience_id, 1, e.content, e.trigger,
                          r, j, calculate_experience_credits(r, j)[e.experience_id])
    assert m.collect_case(e.experience_id, case)
    card = experience_update_payload(m.get(e.experience_id))["cases"][-1]
    assert [c["question"] for c in card["criteria"]] == ["正确区分"]
    assert len(m.get(e.experience_id).cases[-1].rubric.criteria) == 3


def test_partition_is_persisted_and_changes_no_leaf_credit_text_or_cases(tmp_path):
    m = pool(tmp_path)
    original = m.all()
    assert m.organization_due(5) and not m.organization_due(6)
    event = m.organize("origin4", one_group(m))
    key = event.categories[0].category_id
    restored = ExperienceMemory(tmp_path)
    assert len(restored.categories()) == 1 and not restored.pending_organization
    assert [replace(e, category_id=None) for e in restored.all()] == original
    assert all(e.category_id == key for e in restored.all())
    assert not hasattr(restored.categories()[0], "confidence")
    # Explicit reorganization can rename an existing category without changing its stable ID.
    restored.pending_organization[original[0].experience_id] = 1
    plan = one_group(restored, category_id=key)
    plan.categories[0].name = "信息依据"
    restored.organize("origin4", plan)
    assert restored.categories()[0].category_id == key and restored.categories()[0].name == "信息依据"


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unknown_category", "empty"])
def test_invalid_partition_never_partially_writes(tmp_path, problem):
    m = pool(tmp_path)
    plan = one_group(m)
    before = m.path.read_bytes()
    if problem == "duplicate":
        plan.ungrouped_experience_ids = [m.all()[0].experience_id]
    elif problem == "missing":
        plan.categories[0].experience_ids.pop()
    elif problem == "unknown_category":
        plan.categories[0].category_id = "unknown"
    else:
        plan = CategoryOrganization([], [])
    with pytest.raises(ValueError):
        m.organize("origin4", plan)
    assert m.path.read_bytes() == before and not m.categories()


def test_pending_queue_is_distinct_versions_only_and_credit_does_not_requeue(tmp_path):
    m = pool(tmp_path, 1)
    e = m.all()[0]
    m.organize("origin0", CategoryOrganization([], [e.experience_id]))
    for i in range(3):
        case = case_for(e.experience_id, str(i), e.content, e.trigger)
        m.apply_downstream_credits(str(i), {e.experience_id: case.credit})
        m.collect_case(e.experience_id, case)
        assert not m.pending_organization
    m.apply_update(e.experience_id, ExperienceUpdate("修正方法", "*"))
    assert m.pending_organization == {e.experience_id: 2}
    assert not m.organization_due(5) and m.organization_due(5, flush=True)
    m.organize("2", None, error="timeout")
    assert not m.organization_due(1, flush=True)
    assert ExperienceMemory(tmp_path).pending_organization == {e.experience_id: 2}
    seed(m, CandidateExperience("fresh", "新方法", "*"), "new-origin")
    assert m.organization_due(1)


def test_category_and_ungrouped_units_expand_fully_and_embed_no_ids_or_credit():
    class Embed:
        def __init__(self): self.calls = []
        def embed(self, text): self.calls.append(text); return (1, 0)
    cat = ExperienceCategory("cat-long", "依据", "依据核验")
    leaves = [Experience(f"exp-long-{i}", f"规则{i}", "适用", 2, 1, [], [], category_id=cat.category_id)
              for i in range(21)]
    lone = Experience("exp-lone", "单独方法", "适用", 1, 1, [], [])
    embed = Embed()
    retriever = VectorRetriever(embed)
    assert len(retrieval_units([*leaves, lone], [cat])) == 2
    result = retriever.retrieve("任务", [*leaves, lone], categories=[cat])
    assert len(result) == 22  # Top20 counts retrieval units, never cuts a category's leaves.
    assert len(embed.calls) == 3
    assert "规则20" in embed.calls[1] and "依据核验" in embed.calls[1]
    assert "exp-long" not in str(embed.calls) and "cat-long" not in str(embed.calls)
    retriever.retrieve("任务", [*leaves, lone], categories=[cat])
    assert len(embed.calls) == 4
    leaves[0] = replace(leaves[0], content="修正后的原子规则")
    retriever.retrieve("任务", [*leaves, lone], categories=[cat])
    assert len(embed.calls) == 6 and "修正后的原子规则" in embed.calls[-1]
    retriever.retrieve("任务", [*leaves, lone], categories=[replace(cat, description="新的范围")])
    assert len(embed.calls) == 8


def test_selector_grouping_short_ids_only_and_generator_receives_leaves(tmp_path):
    m = pool(tmp_path, 3)
    m.organize("origin2", one_group(m))
    cat = m.categories()[0]
    views = [ExperienceView.from_experience(e, cat) for e in m.all()]
    payload = selector_payload("任务", views, max_selected=5, training=True)
    assert len(payload["candidates"]) == 1 and len(payload["candidates"][0]["members"]) == 3
    codec = NumberedIDCodec("selector", payload)
    projected = codec.project(payload)
    assert "exp_" not in json.dumps(projected) and "cat_" not in json.dumps(projected)
    assert codec.restore({"selected_experience_ids": ["2"]})["selected_experience_ids"] == [views[1].experience_id]
    generated = generator_payload("任务", [views[1]])
    assert set(generated["selected_experiences"][0]) == {"experience_id", "content", "trigger"}
    assert "category" not in json.dumps(generated)


def test_selector_budget_keeps_full_category_and_does_not_call_model(tmp_path):
    m = pool(tmp_path, 2)
    m.organize("origin1", one_group(m))
    b = FakeLLMBackend()
    selector = LLMSelector(b, input_bytes=10)
    with pytest.raises(ValueError, match="context budget"):
        selector.select("任务", [ExperienceView.from_experience(e, m.categories()[0]) for e in m.all()])
    assert not b.requests


def test_http_organizer_uses_short_ids_and_retries_invalid_partition(monkeypatch, tmp_path, wire_payload):
    m = pool(tmp_path, 3)
    calls = []
    def respond(request):
        body = json.loads(request.content)
        payload = wire_payload(body)
        assert all(e["experience_id"] == str(i) for i, e in enumerate(payload["experiences"], 1))
        assert "exp_" not in str(body) and "origin" not in str(body)
        assert "confidence" not in str(body) and "chosen" not in str(body)
        calls.append(body)
        ids = ["1", "2"] if len(calls) == 1 else ["1", "2", "3"]
        return result({"categories": [{"category_id": None, "name": "事实依据", "description": "核验事实来源", "experience_ids": ids}],
                       "ungrouped_experience_ids": []})
    organizer = LLMCategoryOrganizer(backend(monkeypatch, respond, max_retries=0))
    plan = organizer.organize(m.all(), [], m.pending_organization)
    assert len(calls) == 2 and set(plan.categories[0].experience_ids) == {e.experience_id for e in m.all()}
    assert not m.categories()  # Adapter never writes the pool.


@pytest.mark.parametrize("error", [ValueError, LLMSampleError])
def test_organizer_failure_retains_queue_and_does_not_repeat_each_sample(tmp_path, error):
    r, _ = existing_runner(tmp_path)
    r.organization_batch_size = 1
    calls = []
    def fail(*args): calls.append(args); raise error("bounded failure")
    r.modules.category_organizer.organize = fail
    first = r.run_sample(sample("one"))
    assert first.state == "initial_success" and first.category_organization.error == "bounded failure"
    second = r.run_sample(sample("two"))
    assert second.category_organization is None and len(calls) == 1
    assert r.memory.pending_organization and not r.memory.categories()
    assert r.flush_organization("two") is None


def test_flush_organizes_below_batch_and_evaluation_is_read_only(tmp_path):
    r, _ = existing_runner(tmp_path)
    r.modules.category_organizer.organize = lambda *args: one_group(r.memory)
    record = r.run_sample(sample("one"))
    assert record.category_organization is None
    assert r.flush_organization("one") is not None
    original = r.memory.path.read_bytes()
    r.training = False
    assert r.flush_organization("one") is None and r.memory.path.read_bytes() == original


def test_fatal_organizer_auth_error_is_not_hidden(tmp_path):
    r, _ = existing_runner(tmp_path)
    r.modules.category_organizer.organize = lambda *a: (_ for _ in ()).throw(LLMRequestError("HTTP 401"))
    with pytest.raises(LLMRequestError):
        r.flush_organization("seed")
