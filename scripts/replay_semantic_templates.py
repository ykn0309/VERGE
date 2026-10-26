"""Plan79 isolated Generator/Generator Checker replay, no pipeline writes."""

import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import replay_v2_modules as runner
from verge.models import Experience, Method, task_identity
from verge.prompts import PROMPT_VERSION


def main():
    out = Path("artifacts/v2-semantic-templates-79")
    if out.exists():
        raise ValueError("new directory required")
    source = Path(
        "runs/helpsteer3-general-v2-prompt-v9-qwen-flash-30/artifacts/role_io.jsonl"
    )
    checker = Path("artifacts/v2-review-71/inputs.json")
    traces = [json.loads(l) for l in source.read_text().splitlines()]
    groups = {}
    for t in traces:
        if t["role"] == "initial_generator":
            groups.setdefault(t["sample_id"], []).append(t)
    chosen = list(groups)[:19] + [list(groups)[25]]
    rows = []
    for i, sid in enumerate(chosen, 1):
        options = groups[sid]
        t = next(
            (
                t
                for t in options
                if any(e["template"] for e in t["input"]["experiences"])
            ),
            options[0],
        )
        p = t["input"]
        es = [
            Experience(
                e["experience_id"],
                Method.from_dict(
                    {
                        k: e[k]
                        for k in ("trigger", "principle", "template", "parameters")
                    }
                ),
                sid,
            ).to_dict()
            for e in p["experiences"]
        ]
        rows.append(
            {
                "key": f"initial_generator-{i:02}",
                "role": "initial_generator",
                "sample_id": sid,
                "input": {"task": p["task"], "experiences": es},
                "conversion": "Actual run78 task and supplied methods; local IDs retained; add unavailable private quality defaults only. Prefer first templated candidate request if present. No new generation feeds Checker.",
            }
        )
    rows += [
        r for r in json.loads(checker.read_text()) if r["role"] == "generator_checker"
    ]
    assert len(rows) == 40
    for role in ("initial_generator", "generator_checker"):
        rs = [r for r in rows if r["role"] == role]
        tasks = [
            (
                r["input"]["task"]
                if role == "initial_generator"
                else r["input"]["application"]["sample"]["task"]
            )
            for r in rs
        ]
        assert len({task_identity(t) for t in tasks}) == 20
    out.mkdir(parents=True)
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, checker)
    }
    (out / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "source_sha256": hashes,
                "model": "qwen3.5-flash",
                "enable_thinking": False,
                "independent": True,
                "tasks_per_role": 20,
                "generator_sampling": "run78 tasks1-19 plus26, first templated candidate request preferred; covers failed template uses4/6/10/26 and ordinary initial requests",
                "checker_sampling": "plan71 fixed20 actual historical applications; old optional templates absent, cannot fully validate newly relaxed template semantics",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    runner.OUT = out
    runner.execute(workers=4)
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()
    )


if __name__ == "__main__":
    main()
