"""Plan81: isolated 20-task role replays with explicit removed-field conversion."""

import json, sys, hashlib, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import replay_v2_modules as core, replay_v2_review as review
from verge.prompts import PROMPT_VERSION
from verge.models import task_identity


def convert(x):
    if isinstance(x, list):
        return [convert(v) for v in x]
    if not isinstance(x, dict):
        return x
    out = {k: convert(v) for k, v in x.items() if k != "binding_evidence"}
    if isinstance(out.get("parameters"), dict):
        out["parameters"] = {
            k: {"description": v["description"]} for k, v in out["parameters"].items()
        }
    if out.get("schema_version") == 13:
        out["schema_version"] = 14
    return out


def main():
    root = Path("artifacts/v2-simplified-fields-81")
    if root.exists():
        raise ValueError("new directory required")
    sources = [
        Path("artifacts/v2-module-replay/inputs.json"),
        Path("artifacts/v2-review-71/inputs.json"),
        Path("artifacts/v2-semantic-templates-79/inputs.json"),
    ]
    rows0 = json.loads(sources[0].read_text())
    rows1 = json.loads(sources[1].read_text())
    rows2 = json.loads(sources[2].read_text())
    groups = {
        "core": [r for r in rows0 if r["role"] in ("judge", "critic")]
        + [r for r in rows2 if r["role"] == "initial_generator"],
        "review": rows1,
    }
    root.mkdir(parents=True)
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "independent": True,
        "tasks_per_role": 20,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "conversion": "Offline copies remove criterion binding_evidence and method parameter source; embedded trajectory schema13→14 only. Task/answers, other fields and input grouping unchanged. No output feeds another role/case. Generator uses actual run78 requests, other roles fixed plan70/71 historical inputs and documented conversions.",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    for group, rows in groups.items():
        rows = convert(rows)
        for role in {r["role"] for r in rows}:
            rs = [r for r in rows if r["role"] == role]

            def task(r):
                p = r["input"]
                if "task" in p:
                    return p["task"]
                if "sample" in p:
                    return p["sample"]["task"]
                if "application" in p:
                    return p["application"]["sample"]["task"]
                return p["cases"][0]["sample"]["task"]

            assert len(rs) == 20 and len({task_identity(task(r)) for r in rs}) == 20
        out = root / group
        out.mkdir()
        (out / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "input_sha256": hashlib.sha256(
                        (out / "inputs.json").read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
        )
        if group == "core":
            core.OUT = out
            core.execute(workers=4)
        else:
            shutil.copytree(
                "verge",
                out / "code_snapshot/verge",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            review.OUT = out
            review.execute(rows)
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        for p, h in manifest["source_sha256"].items()
    )
    summary = {}
    for group in groups:
        summary.update(json.loads((root / group / "summary.json").read_text()))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
