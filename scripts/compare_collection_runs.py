"""Read-only paired description of two collection runs; not a causal evaluation."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_collection import audit
from verge.memory import ExperienceMemory


def extras(run):
    run = Path(run)
    m = json.loads((run / "run_manifest.json").read_text())
    root = run / m["config"]["store"]["config"].get("root_dir", "artifacts")
    memory = ExperienceMemory(root, read_only=True)
    traces = [json.loads(l) for l in (root / "role_io.jsonl").read_text().splitlines()]
    traces = [r for r in traces if r.get("event_type") != "contract_validation"]
    weights = []
    invalid_weights = []
    for r in traces:
        if r["role"] != "initial_generator" or not isinstance(r.get("output"), dict):
            continue
        criteria = r["output"].get("criteria", [])
        if criteria:
            if not isinstance(criteria, list) or any(
                not isinstance(c, dict) or type(c.get("weight")) not in (int, float)
                for c in criteria
            ):
                invalid_weights.append(
                    {"sample_id": r["sample_id"], "criteria": criteria}
                )
                continue
            values = [c["weight"] for c in criteria]
            weights.append(
                {
                    "sample_id": r["sample_id"],
                    "weights": values,
                    "sum_one": abs(sum(values) - 1) < 1e-9,
                    "all_one": all(w == 1 for w in values),
                }
            )
    return {
        "methods": [
            {"id": e.experience_id, **e.method.to_dict()} for e in memory.all()
        ],
        "generator_outputs": len(weights),
        "generator_nonnumeric_weight_outputs": invalid_weights,
        "generator_sum_one": sum(w["sum_one"] for w in weights),
        "generator_all_one": sum(w["all_one"] for w in weights),
        "weights": weights,
        "role_errors": [
            {"role": r["role"], "sample_id": r["sample_id"], "error": r["error"]}
            for r in traces
            if r.get("error")
        ],
        "role_requests": dict(Counter(r["role"] for r in traces)),
        "prompt_versions": dict(Counter(r.get("prompt_version") for r in traces)),
    }


def compare(old, new, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    audits = {}
    cases = {}
    for label, path in [("old", old), ("new", new)]:
        summary, samples = audit(path)
        audits[label] = {"summary": summary, "extras": extras(path)}
        cases[label] = samples
    assert [r["sample_id"] for r in cases["old"]] == [
        r["sample_id"] for r in cases["new"]
    ]
    pairs = []
    for a, b in zip(cases["old"], cases["new"]):
        pairs.append(
            {
                "number": a["number"],
                "sample_id": a["sample_id"],
                "old_initial": a["initial_correct"],
                "new_initial": b["initial_correct"],
                "old_state": a["state"],
                "new_state": b["state"],
                "old_selected": a["selected"],
                "new_selected": b["selected"],
                "old_cited": a["cited"],
                "new_cited": b["cited"],
            }
        )
    hashes = json.loads((out / "source_hashes.json").read_text())
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()
    )
    result = {
        "runs": audits,
        "paired": pairs,
        "source_hashes_unchanged": True,
        "initial_transitions": dict(
            Counter(f"{a['old_initial']}→{a['new_initial']}" for a in pairs)
        ),
    }
    (out / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    for label, rows in cases.items():
        (out / f"{label}_sample_review.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        )
    print(
        json.dumps(
            {
                k: {
                    "summary": v["summary"],
                    "generator_sum_one": v["extras"]["generator_sum_one"],
                    "generator_outputs": v["extras"]["generator_outputs"],
                    "generator_all_one": v["extras"]["generator_all_one"],
                }
                for k, v in audits.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(result["initial_transitions"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    compare(a.old, a.new, a.output)
