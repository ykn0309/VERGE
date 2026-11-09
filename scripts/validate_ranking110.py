"""Check group-first ranking on 20 saved real queries, without model calls.

Query tasks are excluded from the entire historical state. This is a diagnostic
calculation replay, not a test of end-to-end accuracy or semantic applicability.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from math import fsum, hypot, isclose
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_learning110 import digest, load_lines, task_id
from verge.clustering import ClusterItem, OnlineClustering
from verge.feedback import FeedbackLedger, PairedFeedback
from verge.ranking import rank_experiences
from verge.retrieval import CandidateQuota, retrieve_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new output directory")
    protected = [args.queries] + [
        args.source / name
        for name in ("results.jsonl", "role_io.jsonl", "frozen_experience_memory.json")
    ]
    before = {str(path): digest(path) for path in protected}
    queries = json.loads(args.queries.read_text())
    excluded = {case["task_id"] for case in queries}
    if len(queries) != 20 or len(excluded) != 20:
        raise ValueError("need 20 task-deduplicated real queries")
    pairs = {}
    for row in load_lines(args.source / "results.jsonl"):
        key = row["task_id"], row["arm"]
        if key in pairs:
            raise ValueError("duplicate task/arm in source")
        pairs[key] = row
    vectors = {
        call["input"]["text"]: tuple(call["output"])
        for call in load_lines(args.source / "role_io.jsonl")
        if call.get("role") == "embedding"
    }
    task_texts = {task_id(text): text for text in vectors}
    experiences = json.loads(
        (args.source / "frozen_experience_memory.json").read_text()
    )["experiences"]
    experience_texts = {
        exp["experience_id"]: json.dumps(
            exp["method"], ensure_ascii=False, sort_keys=True
        )
        for exp in experiences
        if exp["status"] == "active"
    }
    history = []
    for (tid, arm), actual in pairs.items():
        if arm != "experience" or tid in excluded:
            continue
        baseline = pairs[(tid, "baseline")]
        if not actual["valid"] or not baseline["valid"]:
            continue
        cited = sorted(
            {
                eid
                for criterion in actual["rubric"]["criteria"]
                for eid in criterion["source_experiences"]
            }
        )
        history.append(
            {
                "task_id": tid,
                "experience_ids": cited,
                "baseline_correct": baseline["correct"],
                "actual_correct": actual["correct"],
            }
        )
    assert not excluded.intersection(case["task_id"] for case in history)
    quota = CandidateQuota()
    parameters = dict(
        reference_weight=0.3,
        task_threshold=0.75,
        experience_threshold=0.75,
        max_rounds=5,
    )
    code = [
        Path(__file__),
        Path("verge/ranking.py"),
        Path("verge/feedback.py"),
        Path("verge/clustering.py"),
        Path("verge/retrieval.py"),
    ]
    protocol = {
        "source_sha256": before,
        "code_sha256": {str(p): digest(p) for p in code},
        "model_calls": 0,
        "queries": 20,
        "history_tasks": len(history),
        "parameters": parameters,
        "query_group_limit": 3,
        "semantic_limit": quota.total,
        "candidate_quota": asdict(quota),
        "retrieval_budget_note": "user-confirmed candidate quota: six historical, four semantic; shortages filled semantically",
        "ranking": "similarity-weighted task-group means, not raw-record weighting",
        "history_initialization": "one batch of all valid non-query tasks and citations",
        "limitations": [
            "queries selected as five per historical paired outcome, not representative",
            "saved vectors use historical two-field experiences",
            "all query task feedback/citations excluded before clustering and ranking",
            "no prediction rerun, applicability assessment or effectiveness measurement",
            "uncited experiences without cached embeddings stay rank candidates but are not clustered",
            "candidate list uses only the 86 active entries of the historical frozen run",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "code").mkdir()
    for path in code:
        (args.output / "code" / path.name).write_bytes(path.read_bytes())
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2)
    )
    (args.output / "fixed_inputs.json").write_text(
        json.dumps(
            {
                "query_tasks": [
                    {"task_id": q["task_id"], "task": q["task"]} for q in queries
                ],
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    clusters = OnlineClustering(**parameters)
    items = [
        ClusterItem(eid, "experience", text, vectors[text])
        for eid, text in experience_texts.items()
        if text in vectors
    ]
    items.extend(
        ClusterItem(
            case["task_id"],
            "task",
            task_texts[case["task_id"]],
            vectors[task_texts[case["task_id"]]],
        )
        for case in history
    )
    citations = [
        (case["task_id"], eid) for case in history for eid in case["experience_ids"]
    ]
    known_experiences = {item.object_id for item in items if item.kind == "experience"}
    assert all(
        eid in known_experiences for _, eid in citations
    ), "missing embedding for an actually cited experience"
    update = clusters.update(items, citations)
    task_groups = clusters.assignments("task")
    ledger = FeedbackLedger(
        "ranking110-offline",
        [
            PairedFeedback(
                case["task_id"],
                case["task_id"],
                "ranking110-offline",
                tuple(case["experience_ids"]),
                case["baseline_correct"],
                case["actual_correct"],
            )
            for case in history
        ],
    )
    statistics = ledger.by_task_group(task_groups)
    output = []
    for query in queries:
        groups = dict(clusters.query_task_groups(vectors[query["task"]], limit=3))
        ranking = rank_experiences(experience_texts, groups, statistics)
        routes = retrieve_candidates(
            vectors[query["task"]],
            clusters,
            ledger,
            task_group_limit=3,
            quota=quota,
        )
        assert routes.historical == ranking
        assert routes.semantic_only == (not groups)
        raw_query = vectors[query["task"]]
        query_norm = hypot(*raw_query)
        semantic_expected = []
        for eid, text in experience_texts.items():
            raw_experience = vectors[text]
            norm = hypot(*raw_experience)
            cosine = fsum(
                (a / query_norm) * (b / norm) for a, b in zip(raw_query, raw_experience)
            )
            semantic_expected.append((eid, max(0.0, min(1.0, cosine))))
        semantic_expected.sort(key=lambda item: (-item[1], item[0]))
        assert len(routes.semantic) == min(quota.total, len(experience_texts))
        for actual, expected_semantic in zip(
            routes.semantic, semantic_expected[: quota.total]
        ):
            assert actual[0] == expected_semantic[0]
            assert isclose(actual[1], expected_semantic[1], abs_tol=1e-12)

        expected = {}
        for eid in experience_texts:
            numerator = denominator = 0.0
            for group_id, similarity in groups.items():
                rewards = [
                    int(case["actual_correct"]) - int(case["baseline_correct"])
                    for case in history
                    if task_groups[case["task_id"]] == group_id
                    and eid in case["experience_ids"]
                ]
                if not rewards:
                    continue
                numerator += similarity * sum(rewards) / len(rewards)
                denominator += similarity
            expected[eid] = numerator / denominator if denominator else None
        assert set(ranking.unknown_ids) == {
            eid for eid, value in expected.items() if value is None
        }
        for result in ranking.scored:
            assert isclose(result.score, expected[result.experience_id], abs_tol=1e-12)
            assert -1 - 1e-12 <= result.score <= 1 + 1e-12
            assert not excluded.intersection(
                tid for e in result.evidence for tid in e.statistics.task_ids
            )
        expected_ids = [
            eid
            for eid in sorted(
                expected,
                key=lambda eid: (
                    -(expected[eid] if expected[eid] is not None else -2),
                    eid,
                ),
            )
            if expected[eid] is not None
        ][: quota.historical]
        for eid, _ in semantic_expected:
            if len(expected_ids) == min(quota.total, len(experience_texts)):
                break
            if eid not in expected_ids:
                expected_ids.append(eid)
        assert [
            candidate.experience_id for candidate in routes.candidates
        ] == expected_ids
        assert len(set(expected_ids)) == len(expected_ids)
        if routes.semantic_only:
            assert all(
                candidate.source == "semantic" and candidate.historical_score is None
                for candidate in routes.candidates
            )
        output.append(
            {
                "task_id": query["task_id"],
                "task": query["task"],
                "task_groups": groups,
                "semantic_only": routes.semantic_only,
                "semantic": routes.semantic,
                "candidates": [asdict(candidate) for candidate in routes.candidates],
                **asdict(ranking),
                "scored": [
                    {**asdict(item), "tasks": item.tasks} for item in ranking.scored
                ],
            }
        )
    assert before == {str(path): digest(path) for path in protected}
    summary = {
        "queries": len(output),
        "history_tasks": len(history),
        "experiences": len(experience_texts),
        "queries_with_matching_groups": sum(bool(row["task_groups"]) for row in output),
        "queries_with_scored_experiences": sum(bool(row["scored"]) for row in output),
        "scored_entries": sum(len(row["scored"]) for row in output),
        "queries_with_semantic_candidates": sum(
            bool(row["semantic"]) for row in output
        ),
        "semantic_only_queries": sum(row["semantic_only"] for row in output),
        "semantic_entries": sum(len(row["semantic"]) for row in output),
        "candidate_entries": sum(len(row["candidates"]) for row in output),
        "historical_candidate_entries": sum(
            c["source"] == "historical" for row in output for c in row["candidates"]
        ),
        "semantic_candidate_entries": sum(
            c["source"] == "semantic" for row in output for c in row["candidates"]
        ),
        "task_groups": len(clusters.groups("task")),
        "clustering_update": asdict(update),
        "query_feedback_excluded": True,
        "source_unchanged": True,
        "reference_calculation_passed": True,
        "model_calls": 0,
    }
    (args.output / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
