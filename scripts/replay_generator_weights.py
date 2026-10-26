"""Plan76: fixed 20-task Generator-only replay; reuse production adapter and audit runner."""

from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import replay_v2_modules as runner
from verge.models import task_identity
from verge.prompts import PROMPT_VERSION


def main():
    source = Path("artifacts/v2-module-replay/inputs.json")
    out = Path("artifacts/v2-generator-weights-76")
    if out.exists():
        raise ValueError("new output required")
    rows = [
        r for r in json.loads(source.read_text()) if r["role"] == "initial_generator"
    ]
    assert (
        len(rows) == 20 and len({task_identity(r["input"]["task"]) for r in rows}) == 20
    )
    out.mkdir(parents=True)
    (out / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((out / "inputs.json").read_bytes()).hexdigest(),
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "role": "initial_generator",
        "tasks": 20,
        "prompt_version": PROMPT_VERSION,
        "independent": True,
        "source_conversion": "Frozen plan70 actual historical task and supplied experience input reused unchanged; no new output feeds another role or case.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    runner.OUT = out
    runner.execute(workers=4)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    results = [
        json.loads(p.read_text()) for p in sorted((out / "results").glob("*.json"))
    ]
    audit = []
    for r in results:
        if not r["valid"]:
            audit.append({"key": r["key"], "valid": False})
            continue
        criteria = r["output"]["criteria"]
        weights = [c["weight"] for c in criteria]
        audit.append(
            {
                "key": r["key"],
                "valid": True,
                "count": len(weights),
                "weights": weights,
                "sum": sum(weights),
                "sum_one": abs(sum(weights) - 1) < 1e-9,
                "all_one": all(w == 1 for w in weights),
                "equal_weights": len(set(weights)) == 1,
                "matches_example": weights == [0.6, 0.25, 0.15],
            }
        )
    (out / "weight_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "valid": sum(r["valid"] for r in audit),
                "sum_one": sum(r.get("sum_one", False) for r in audit),
                "all_one": sum(r.get("all_one", False) for r in audit),
                "equal_weights": sum(r.get("equal_weights", False) for r in audit),
                "matches_example": sum(r.get("matches_example", False) for r in audit),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
