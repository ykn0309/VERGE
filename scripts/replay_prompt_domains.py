"""Plan77: eight independent role replays, 20 frozen historical tasks per role."""

from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import replay_v2_modules as core
from scripts import replay_v2_review as review
from verge.models import task_identity
from verge.prompts import PROMPT_VERSION

ROOT = Path("artifacts/v2-prompt-domains-77")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if ROOT.exists():
        raise ValueError("new output directory required")
    sources = [
        Path("artifacts/v2-module-replay/inputs.json"),
        Path("artifacts/v2-review-71/inputs.json"),
    ]
    groups = {
        "core": [
            r
            for r in json.loads(sources[0].read_text())
            if r["role"] in ("initial_generator", "judge", "critic")
        ],
        "review": json.loads(sources[1].read_text()),
    }
    for rows in groups.values():
        for role in {r["role"] for r in rows}:
            group = [r for r in rows if r["role"] == role]

            def task(r):
                p = r["input"]
                if "task" in p:
                    return p["task"]
                if "sample" in p:
                    return p["sample"]["task"]
                if "application" in p:
                    return p["application"]["sample"]["task"]
                return p["cases"][0]["sample"]["task"]

            assert (
                len(group) == 20 and len({task_identity(task(r)) for r in group}) == 20
            )
    ROOT.mkdir(parents=True)
    hashes = {str(p): sha(p) for p in sources}
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "tasks_per_role": 20,
        "roles": 8,
        "independent": True,
        "source_sha256": hashes,
        "scope": "Frozen actual historical inputs from plans70/71; their documented V2 conversions retained. No generated output feeds another role or case. Checkers use no-call stubs in other slots. Judge tests both original answers independently.",
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    for name, rows in groups.items():
        out = ROOT / name
        out.mkdir()
        (out / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        (out / "manifest.json").write_text(
            json.dumps({**manifest, "input_sha256": sha(out / "inputs.json")}, indent=2)
        )
        if name == "core":
            core.OUT = out
            core.execute(workers=4)
        else:
            # Full production Updater projection rather than direct raw requests.
            import shutil

            shutil.copytree(
                "verge",
                out / "code_snapshot/verge",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            review.OUT = out
            review.execute(rows)
        assert (
            sha(out / "inputs.json")
            == json.loads((out / "manifest.json").read_text())["input_sha256"]
        )
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    summary = {}
    for name in groups:
        summary.update(json.loads((ROOT / name / "summary.json").read_text()))
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
