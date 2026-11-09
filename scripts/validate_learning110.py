"""Replay 20 fixed historical pairs through the new pure learning components.

Uses saved embeddings and judgments only. No API requests or pipeline runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verge.clustering import ClusterItem, OnlineClustering
from verge.feedback import FeedbackLedger, PairedFeedback


def load_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_id(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source, args.output
    if output.exists():
        raise ValueError("use a new output directory")
    protected = [
        source / name
        for name in ("results.jsonl", "role_io.jsonl", "frozen_experience_memory.json")
    ]
    before = {str(path): digest(path) for path in protected}
    rows = load_lines(protected[0])
    pairs = {}
    for row in rows:
        key = row["task_id"], row["arm"]
        if key in pairs:
            raise ValueError("duplicate source task/arm")
        pairs[key] = row
    cases = []
    quota = Counter()
    for baseline in rows:
        if baseline["arm"] != "baseline":
            continue
        actual = pairs[(baseline["task_id"], "experience")]
        if not baseline["valid"] or not actual["valid"]:
            continue
        outcome = (baseline["correct"], actual["correct"])
        if quota[outcome] == 5:
            continue
        quota[outcome] += 1
        cases.append((baseline, actual))
    if len(cases) != 20 or len(quota) != 4:
        raise ValueError("need five distinct tasks for each paired outcome")

    vectors = {}
    for call in load_lines(protected[1]):
        if call.get("role") == "embedding":
            vectors[call["input"]["text"]] = tuple(call["output"])
    task_texts = {task_id(text): text for text in vectors}
    experiences = json.loads(protected[2].read_text())["experiences"]
    experience_texts = {
        e["experience_id"]: json.dumps(e["method"], ensure_ascii=False, sort_keys=True)
        for e in experiences
    }
    fixed = []
    for baseline, actual in cases:
        for arm in (baseline, actual):
            judgment = arm["judgment"]
            assert arm["correct"] == (
                judgment["chosen_score"] > judgment["rejected_score"]
            )
        cited = sorted(
            {
                eid
                for criterion in actual["rubric"]["criteria"]
                for eid in criterion["source_experiences"]
            }
        )
        text = task_texts[actual["task_id"]]
        for eid in cited:
            assert (
                experience_texts[eid] in vectors
            ), "missing historical experience embedding"
        fixed.append(
            {
                "sample_id": actual["sample_id"],
                "task_id": actual["task_id"],
                "task": text,
                "experience_ids": cited,
                "baseline_correct": baseline["correct"],
                "actual_correct": actual["correct"],
            }
        )
    output.mkdir(parents=True)
    (output / "fixed_inputs.json").write_text(
        json.dumps(fixed, ensure_ascii=False, indent=2)
    )
    protocol = {
        "source_sha256": before,
        "selection": "first five valid tasks per paired outcome",
        "tasks": 20,
        "model_calls": 0,
        "limitation": "saved embeddings represent historical two-field methods, not the new prompt or method format",
        "parameters": {
            "reference_weight": 0.3,
            "task_threshold": 0.75,
            "experience_threshold": 0.75,
            "max_rounds": 5,
        },
        "code_sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("verge/feedback.py"),
                Path("verge/clustering.py"),
            )
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    ledger = FeedbackLedger("historical-paired200-learning110")
    clusters = OnlineClustering(**protocol["parameters"])
    known_experiences = set()
    updates = []
    expected = {}
    for case in fixed:
        items = [
            ClusterItem(case["task_id"], "task", case["task"], vectors[case["task"]])
        ]
        for eid in case["experience_ids"]:
            if eid not in known_experiences:
                text = experience_texts[eid]
                items.append(ClusterItem(eid, "experience", text, vectors[text]))
                known_experiences.add(eid)
        update = clusters.update(
            items, [(case["task_id"], eid) for eid in case["experience_ids"]]
        )
        updates.append({"task_id": case["task_id"], **asdict(update)})
        feedback = PairedFeedback(
            case["sample_id"],
            case["task_id"],
            ledger.protocol_id,
            tuple(case["experience_ids"]),
            case["baseline_correct"],
            case["actual_correct"],
        )
        ledger.append(feedback)
        reward = int(case["actual_correct"]) - int(case["baseline_correct"])
        for eid in case["experience_ids"]:
            expected.setdefault(eid, []).append(reward)
    for eid, rewards in expected.items():
        stats = ledger.for_experience(eid)
        assert stats.tasks == len(rewards)
        assert stats.reward_sum == sum(rewards)
        assert abs(stats.mean_reward - sum(rewards) / len(rewards)) < 1e-12
    task_groups = clusters.assignments("task")
    exp_groups = clusters.assignments("experience")
    group_stats = ledger.by_group_pair(task_groups, exp_groups)
    for (tg, eg), stats in group_stats.items():
        expected_tasks = {
            case["task_id"]
            for case in fixed
            if task_groups[case["task_id"]] == tg
            and any(exp_groups[eid] == eg for eid in case["experience_ids"])
        }
        assert stats.tasks == len(expected_tasks)
    assert before == {str(path): digest(path) for path in protected}
    summary = {
        "tasks": len(fixed),
        "cited_experiences": len(expected),
        "participation_records": sum(len(value) for value in expected.values()),
        "task_groups": len(clusters.groups("task")),
        "experience_groups": len(clusters.groups("experience")),
        "unstable_updates": sum(not update["stable"] for update in updates),
        "source_unchanged": True,
        "checks_passed": True,
        "note": "tests bookkeeping and grouping contracts, not semantic quality or prediction improvement",
    }
    (output / "updates.json").write_text(json.dumps(updates, indent=2))
    (output / "experience_statistics.json").write_text(
        json.dumps(
            {eid: asdict(ledger.for_experience(eid)) for eid in expected}, indent=2
        )
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
