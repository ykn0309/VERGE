"""Read-only collection run audit; aggregate model reports without treating them as truth."""

from __future__ import annotations
import argparse, json, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.web import load_role_io


def audit(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    root = run_dir / manifest["config"]["store"]["config"].get("root_dir", "artifacts")
    memory = ExperienceMemory(root, read_only=True)
    records = list(TrajectoryLog(root).all())
    applications = [a for r in records for a in r.applications]
    source_applications = [
        a
        for r in records
        for p in r.proposal_attempts
        for a in p.applications
        if a.phase == "source_validation"
    ]
    cited = [a for a in applications if a.cited]
    traces = [json.loads(l) for l in (root / "role_io.jsonl").read_text().splitlines()]
    traces = [r for r in traces if r.get("event_type") != "contract_validation"]
    selector_traces = [r for r in traces if r["role"] == "selector"]
    selector_keys = {"experience_id", "trigger", "principle", "template", "parameters"}
    hidden = all(
        set(e) == selector_keys
        for r in selector_traces
        for e in r["input"]["candidates"]
    )
    q_disabled = all(
        e.q_posterior is None
        and not e.evidence_counts["downstream"]
        and not e.evidence_counts["replay"]
        for e in memory.all()
    )
    no_revisions = not memory.revisions and not any(
        r.experience_updates for r in records
    )
    sample_path = Path(manifest["config"]["dataset"]["config"]["paths"]["train"])
    sampling = json.loads((sample_path.parent / "sampling.json").read_text())
    source = Path(sampling["source"])
    unchanged = (
        hashlib.sha256(source.read_bytes()).hexdigest() == sampling["source_sha256"]
        and hashlib.sha256(sample_path.read_bytes()).hexdigest()
        == sampling["sample_sha256"]
    )
    assert hidden and q_disabled and no_revisions and unchanged
    statuses = {
        stage: dict(
            Counter(r.status for a in cited for r in a.reports if r.stage == stage)
        )
        for stage in ("selector", "generator", "judge")
    }
    missing = [
        {"sample_id": a.sample.sample_id, "version_id": a.version_id}
        for a in cited
        if {r.stage for r in a.reports} != {"selector", "generator", "judge"}
    ]
    reasons = Counter(a.update_reason for a in applications)
    summary = {
        "run": str(run_dir),
        "state": manifest["state"],
        "processed": len(records),
        "sample_errors": [
            {"sample_id": r.sample_id, **r.sample_error}
            for r in records
            if r.sample_error
        ],
        "states": dict(Counter(r.state for r in records)),
        "initial_correct": sum(
            bool(r.initial_judgment and r.initial_judgment.success) for r in records
        ),
        "final_correct": sum(
            bool(
                next(
                    (p.judgment for p in reversed(r.proposal_attempts) if p.judgment),
                    r.initial_judgment,
                )
                and next(
                    (p.judgment for p in reversed(r.proposal_attempts) if p.judgment),
                    r.initial_judgment,
                ).success
            )
            for r in records
        ),
        "source_validation_applications": len(source_applications),
        "source_validation_feedback": dict(
            Counter(a.feedback for a in source_applications)
        ),
        "source_checker_reports": sum(len(a.reports) for a in source_applications),
        "experience_review_decisions": dict(
            Counter(
                str(p.review.approved)
                for r in records
                for p in r.proposal_attempts
                if p.review
            )
        ),
        "versions": len(memory.all()),
        "active_versions": len(memory.all(active_only=True)),
        "admitted": sum(len(r.admitted_experience_ids) for r in records),
        "candidate_rounds": sum(len(r.proposal_attempts) for r in records),
        "tasks_with_selected": sum(bool(r.selected_experiences) for r in records),
        "tasks_with_citation": len({a.sample.task_id for a in cited}),
        "selected_applications": len(applications),
        "cited_applications": len(cited),
        "feedback": dict(Counter(a.feedback for a in cited)),
        "observations": dict(Counter(a.observation for a in cited)),
        "checker_statuses": statuses,
        "missing_checker_triplets": missing,
        "update_reasons": dict(reasons),
        "quality_disabled": q_disabled,
        "no_revisions": no_revisions,
        "selector_private_statistics_absent": hidden,
        "selector_requests": len(selector_traces),
        "source_hashes_unchanged": unchanged,
        "role_request_counts": dict(Counter(r["role"] for r in traces)),
    }
    samples = []
    for i, r in enumerate(records, 1):
        apps = [a for a in r.applications if a.cited]
        samples.append(
            {
                "number": i,
                "sample_id": r.sample_id,
                "task": r.sample.task,
                "state": r.state,
                "initial_correct": bool(
                    r.initial_judgment and r.initial_judgment.success
                ),
                "candidate_rounds": len(r.proposal_attempts),
                "selected": len(r.selected_experiences),
                "cited": len(apps),
                "checkers": [
                    {
                        "version_id": a.version_id,
                        "method": a.method.to_dict(),
                        "feedback": a.feedback,
                        "reports": [p.to_dict() for p in a.reports],
                    }
                    for a in apps
                ],
            }
        )
    return summary, samples


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    summary, samples = audit(a.run_dir)
    out = Path(a.output)
    out.mkdir(exist_ok=True, parents=True)
    (out / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "sample_review.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
