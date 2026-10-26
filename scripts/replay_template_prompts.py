"""Plan75 isolated Critic/Updater contract replays, never a pipeline experiment."""

from __future__ import annotations
import json, hashlib, shutil, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import *
from verge.modules import LLMCritic, LLMExperienceUpdater
from verge.prompts import NumberedIDCodec, PROMPT_VERSION
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMSampleError

ROOT = Path("artifacts/v2-template-prompts-75")
CURRENT = ContextVar("case", default={})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(only_role=None):
    if ROOT.exists():
        raise ValueError("output must be new")
    sources = [
        Path("artifacts/v2-module-replay/inputs.json"),
        Path("artifacts/v2-review-71/inputs.json"),
    ]
    rows = [r for r in json.loads(sources[0].read_text()) if r["role"] == "critic"]
    rows += [
        r
        for r in json.loads(sources[1].read_text())
        if r["role"] == "experience_updater"
    ]
    # Include the observed nonempty-but-invalid parameter case; retain its actual first-round request.
    run = Path(
        "runs/helpsteer3-general-v2-collection-qwen-flash-30/artifacts/trajectories.jsonl"
    )
    records = [json.loads(l) for l in run.open()]
    problem = records[28]
    rows = [
        r
        for r in rows
        if r["role"] != "critic" or r != [x for x in rows if x["role"] == "critic"][-1]
    ]
    rows.append(
        {
            "key": "critic-20",
            "role": "critic",
            "sample_id": problem["sample_id"],
            "input": {
                "sample": problem["sample"],
                "rubric": problem["initial_rubric"],
                "judgment": problem["initial_judgment"],
            },
            "conversion": "Plan73 sample29 first Critic request; task/answers/rubric/judgment unchanged, no previous rounds reconstructed.",
        }
    )
    sources.append(run)
    for role in ("critic", "experience_updater"):
        group = [r for r in rows if r["role"] == role]
        assert len(group) == 20
        assert (
            len(
                {
                    task_identity(
                        r["input"]["sample"]["task"]
                        if role == "critic"
                        else r["input"]["cases"][0]["sample"]["task"]
                    )
                    for r in group
                }
            )
            == 20
        )
    if only_role:
        rows = [r for r in rows if r["role"] == only_role]
    ROOT.mkdir(parents=True)
    (ROOT / "results").mkdir()
    (ROOT / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    manifest = {
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "roles": 2,
        "tasks_per_role": 20,
        "independent": True,
        "prompt_version": PROMPT_VERSION,
        "source_hashes": {str(p): sha(p) for p in sources},
        "input_sha256": sha(ROOT / "inputs.json"),
        "critic_history": "single-turn genuine input; no generated outputs feed another case",
        "updater_conversion": "Historical balanced-direction feedback mapped to tie in replay copies only; Q remains original unavailable data.",
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    shutil.copytree(
        "verge",
        ROOT / "code_snapshot/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, ROOT / "code_snapshot/replay_template_prompts.py")
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
        result = {k: row[k] for k in ("key", "role", "sample_id")}
        p = row["input"]
        try:
            if row["role"] == "critic":
                value = LLMCritic(b).propose(
                    PreferenceSample.from_dict(p["sample"]),
                    Rubric.from_dict(p["rubric"]),
                    RubricJudgment.from_dict(p["judgment"]),
                    1,
                    conversation=[],
                    codec=NumberedIDCodec(),
                )
            else:
                cases = []
                for raw in p["cases"]:
                    raw = json.loads(json.dumps(raw))
                    if raw["feedback"] == "cancellation":
                        raw["feedback"] = "tie"
                        raw["no_evidence_reason"] = None
                    cases.append(ApplicationRecord.from_dict(raw))
                value = LLMExperienceUpdater(b).update(
                    Experience.from_dict(p["experience"]),
                    cases,
                    p["reasons"],
                    case_kinds=p.get("case_kinds"),
                )
            result.update(valid=True, output=value.to_dict())
        except (LLMFormatError, LLMSampleError) as e:
            result.update(valid=False, error=str(e))
        finally:
            b.close()
        (ROOT / "results" / f'{row["key"]}.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        print(row["key"], result["valid"], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, rows))
    summary = {}
    for role in ("critic", "experience_updater"):
        group = [r for r in results if r["role"] == role]
        methods = (
            [
                c["method"]
                for r in group
                if r["valid"]
                for c in r["output"]["candidates"]
            ]
            if role == "critic"
            else [
                r["output"]["method"]
                for r in group
                if r["valid"] and r["output"]["method"]
            ]
        )
        summary[role] = {
            "tasks": len(group),
            "valid": sum(r["valid"] for r in group),
            "methods": len(methods),
            "templates": sum(bool(m["template"]) for m in methods),
        }
    assert all(sha(Path(p)) == h for p, h in manifest["source_hashes"].items())
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT))
    parser.add_argument("--role", choices=("critic", "experience_updater"))
    args = parser.parse_args()
    ROOT = Path(args.output)
    main(args.role)
