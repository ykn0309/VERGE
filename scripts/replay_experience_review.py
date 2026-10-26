"""Plan82 fixed historical Critic and Experience Checker replays; no admission writes."""

from __future__ import annotations
import json, hashlib, shutil, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import (
    PreferenceSample,
    Rubric,
    RubricJudgment,
    Method,
    Experience,
    Model,
    task_identity,
)
from verge.modules import LLMCritic, LLMExperienceChecker
from verge.prompts import PROMPT_VERSION, NumberedIDCodec
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMSampleError

ROOT = Path("artifacts/v2-experience-review-82")


def main(only_role=None):
    if ROOT.exists():
        raise ValueError("new output required")
    critics = Path("artifacts/v2-simplified-fields-81/core/inputs.json")
    legacy = Path("artifacts/v2-module-replay/source/role_io.jsonl")
    manifestpath = Path("artifacts/v2-module-replay/source/run_manifest.json")
    oldmanifest = json.loads(manifestpath.read_text())
    dataset = Path(oldmanifest["config"]["dataset"]["config"]["paths"]["train"])
    samples = {
        s["sample_id"]: s for s in map(json.loads, dataset.read_text().splitlines())
    }
    rows = [r for r in json.loads(critics.read_text()) if r["role"] == "critic"]
    logs = [json.loads(l) for l in legacy.read_text().splitlines()]
    seen = set()
    checks = []
    for r in logs:
        if r["role"] != "critic" or not r.get("output", {}).get("candidates"):
            continue
        p = r["input"]
        tid = task_identity(p["task"])
        if tid in seen or any(
            len(c["source_experiences"]) > 1 for c in p["current_rubric"]["criteria"]
        ):
            continue
        seen.add(tid)
        c = r["output"]["candidates"][0]
        method = Method(c["trigger"], c["content"])
        rubric = {
            **p["current_rubric"],
            "criteria": [
                {**c, "bindings": {}} for c in p["current_rubric"]["criteria"]
            ],
        }
        checks.append(
            {
                "key": f"experience_checker-{len(checks)+1:02}",
                "role": "experience_checker",
                "sample_id": r["sample_id"],
                "input": {
                    "sample": samples[r["sample_id"]],
                    "rubric": rubric,
                    "judgment": p["current_judgment"],
                    "candidate": Experience(
                        "historical-candidate", method, r["sample_id"]
                    ).to_dict(),
                },
                "conversion": "Actual historical Critic candidate+initial failure trajectory, first candidate only, content→principle; optional template null/parameters{} because absent, bindings{} absent in old criteria. No new result used as labels. Fixed 20 distinct tasks, no hand truth labels.",
            }
        )
        if len(checks) == 20:
            break
    rows += checks
    for role in ("critic", "experience_checker"):
        group = [r for r in rows if r["role"] == role]
        assert len(group) == 20
        assert len({task_identity(r["input"]["sample"]["task"]) for r in group}) == 20
    if only_role:
        rows = [r for r in rows if r["role"] == only_role]
    ROOT.mkdir(parents=True)
    (ROOT / "results").mkdir()
    (ROOT / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    sources = [critics, legacy, manifestpath, dataset]
    manifest = {
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "prompt_version": PROMPT_VERSION,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "independent": True,
        "tasks_per_role": 20,
        "scope": "Critic first-turn fixed inputs plus20 actual old candidates. Independent outputs, no feedback loop/pipeline, no manual truth labels or new quality claims. Checker includes task-specific triggers/multi-rule methods and narrower valid scopes from historical outputs; semantic audit remains necessary.",
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    shutil.copytree(
        "verge",
        ROOT / "code_snapshot/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    lock = threading.Lock()

    def one(row):
        b = OpenAICompatibleBackend(
            "qwen3.5-flash",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY_2",
            enable_thinking=False,
            timeout_seconds=120,
            max_retries=2,
        )

        def log(name, event):
            with lock:
                with (ROOT / name).open("a") as f:
                    f.write(
                        json.dumps(
                            {"key": row["key"], "sample_id": row["sample_id"], **event},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        b.trace_hook = lambda e: log("role_io.jsonl", e)
        b.metadata_hook = lambda e: log("llm_requests.jsonl", e)
        p = row["input"]
        result = {k: row[k] for k in ("key", "role", "sample_id")}
        try:
            s = PreferenceSample.from_dict(p["sample"])
            r = Rubric.from_dict(p["rubric"])
            j = RubricJudgment.from_dict(p["judgment"])
            if row["role"] == "critic":
                value = LLMCritic(b).propose(
                    s, r, j, 1, conversation=[], codec=NumberedIDCodec()
                )
            else:
                value = LLMExperienceChecker(b).check(
                    s, r, j, Experience.from_dict(p["candidate"])
                )
            result.update(valid=True, output=value.to_dict())
        except (LLMFormatError, LLMSampleError) as e:
            result.update(valid=False, error=str(e))
        finally:
            b.close()
        (ROOT / "results" / f"{row['key']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, rows))
    summary = {
        role: {
            "tasks": sum(r["role"] == role for r in results),
            "valid": sum(r["valid"] for r in results if r["role"] == role),
        }
        for role in sorted({r["role"] for r in results})
    }
    summary["review_decisions"] = {
        str(v): sum(
            r["valid"] and r["output"]["approved"] == v
            for r in results
            if r["role"] == "experience_checker"
        )
        for v in (True, False)
    }
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        for p, h in manifest["source_sha256"].items()
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("critic", "experience_checker"))
    p.add_argument("--output", default=str(ROOT))
    a = p.parse_args()
    ROOT = Path(a.output)
    main(a.role)
