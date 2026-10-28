"""Paired, isolated Critic replay on the twenty audited historical inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge import llm, prompts
from verge.llm import LLMFormatError, LLMSampleError, OpenAICompatibleBackend
from verge.models import PreferenceSample, Rubric, RubricJudgment, task_identity
from verge.modules import LLMCritic
from verge.prompts import NumberedIDCodec


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="artifacts/v2-critic-evidence-86")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["baseline", "candidate"],
        default=["baseline", "candidate"],
    )
    parser.add_argument("--candidate-guidance", type=Path, required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--candidate-schema", type=Path)
    parser.add_argument("--output-reminder", type=Path)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    root = Path(args.output)
    source = Path("artifacts/v2-trigger-principle-84")
    input_path = source / "inputs.json"
    baseline_path = source / "code_snapshot/verge/prompts.py"
    baseline = runpy.run_path(str(baseline_path))
    rows = [r for r in json.loads(input_path.read_text()) if r["role"] == "critic"]
    assert len(rows) == 20
    assert len({task_identity(r["input"]["sample"]["task"]) for r in rows}) == 20
    protected = [
        input_path,
        baseline_path,
        source / "semantic-audit/critic_quality.json",
    ]
    protected += [source / "results" / f"{r['key']}.json" for r in rows]
    run = Path("runs/helpsteer3-general-v2-experience-review-qwen-flash-30")
    protected += [
        run / "artifacts" / name
        for name in ("experience_memory.json", "trajectories.jsonl", "role_io.jsonl")
    ]
    hashes = {str(p): digest(p) for p in protected}
    root.mkdir(exist_ok=False)
    (root / "results").mkdir()
    (root / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    shutil.copytree(
        "verge",
        root / "code_snapshot/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, root / "code_snapshot/replay_critic_evidence.py")
    original_guidance = prompts._GUIDANCE["critic"]
    original_version = prompts.PROMPT_VERSION
    original_schema = prompts._SCHEMA["critic"]
    original_reminder = prompts.CRITIC_OUTPUT_REMINDER
    candidate_guidance = (
        args.candidate_guidance.read_text()
        if args.candidate_guidance
        else original_guidance
    )
    candidate_version = args.candidate_version or original_version
    (root / "candidate_guidance.txt").write_text(candidate_guidance)
    arms = [
        ("baseline", baseline["_GUIDANCE"]["critic"], baseline["PROMPT_VERSION"]),
        ("candidate", candidate_guidance, candidate_version),
    ]
    arms = [arm for arm in arms if arm[0] in args.arms]
    manifest = {
        "model": "qwen3.5-flash",
        "enable_thinking": args.thinking,
        "timeout_seconds": args.timeout,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_network_retries": 2,
        "max_contract_retries": 2,
        "workers": 4,
        "tasks": 20,
        "arms": [a[0] for a in arms],
        "scope": "Critic only; same frozen inputs; no downstream calls or pool writes",
        "source_sha256": hashes,
        "input_sha256": digest(root / "inputs.json"),
        "historical_baseline": str(source / "results"),
        "selection": "15 initially failed tasks plus5 successful boundary tasks; not random prevalence sample",
        "candidate_first": args.candidate_first,
        "candidate_schema": (
            json.loads(args.candidate_schema.read_text())
            if args.candidate_schema
            else None
        ),
        "output_reminder": (
            args.output_reminder.read_text() if args.output_reminder else ""
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    lock = threading.Lock()
    all_results = []
    try:
        # Whole arms are sequential: guidance never changes while workers are active.
        for arm, guidance, version in arms:
            prompts._GUIDANCE["critic"] = guidance
            prompts.PROMPT_VERSION = version
            llm.PROMPT_VERSION = version
            prompts.CRITIC_OUTPUT_REMINDER = (
                manifest["output_reminder"]
                if arm == "candidate"
                else baseline.get("CRITIC_OUTPUT_REMINDER", "")
            )
            (root / f"{arm}_output_reminder.txt").write_text(
                prompts.CRITIC_OUTPUT_REMINDER
            )
            if arm == "candidate" and args.candidate_first:
                prompts._SCHEMA["critic"] = {
                    "candidates": original_schema["candidates"],
                    "analysis": original_schema["analysis"],
                }
            if arm == "candidate" and manifest["candidate_schema"] is not None:
                prompts._SCHEMA["critic"] = manifest["candidate_schema"]
            (root / f"{arm}_system.txt").write_text(
                prompts.llm_messages("critic", {})[0]["content"]
            )

            def one(row):
                def log(name, event):
                    with lock:
                        with (root / name).open("a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "arm": arm,
                                        "key": row["key"],
                                        "sample_id": row["sample_id"],
                                        **event,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                backend = OpenAICompatibleBackend(
                    "qwen3.5-flash",
                    base_url=manifest["base_url"],
                    api_key_env="DASHSCOPE_API_KEY_2",
                    enable_thinking=args.thinking,
                    timeout_seconds=args.timeout,
                    max_retries=2,
                    trace_hook=lambda e: log("role_io.jsonl", e),
                    metadata_hook=lambda e: log("llm_requests.jsonl", e),
                )
                p = row["input"]
                result = {"arm": arm, "key": row["key"], "sample_id": row["sample_id"]}
                started = time.perf_counter()
                try:
                    proposal = LLMCritic(backend).propose(
                        PreferenceSample.from_dict(p["sample"]),
                        Rubric.from_dict(p["rubric"]),
                        RubricJudgment.from_dict(p["judgment"]),
                        1,
                        conversation=[],
                        codec=NumberedIDCodec(),
                    )
                    result.update(valid=True, output=proposal.to_dict())
                except (LLMFormatError, LLMSampleError) as exc:
                    result.update(valid=False, error=str(exc))
                finally:
                    backend.close()
                    result["elapsed_seconds"] = time.perf_counter() - started
                (root / "results" / f"{arm}-{row['key']}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2)
                )
                return result

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(one, row) for row in rows]
                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)
                    print(
                        arm,
                        result["key"],
                        "valid" if result["valid"] else "invalid",
                        flush=True,
                    )
    finally:
        prompts._GUIDANCE["critic"] = original_guidance
        prompts.PROMPT_VERSION = original_version
        llm.PROMPT_VERSION = original_version
        prompts._SCHEMA["critic"] = original_schema
        prompts.CRITIC_OUTPUT_REMINDER = original_reminder
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    summary = {}
    for arm, _, version in arms:
        results = [r for r in all_results if r["arm"] == arm]
        summary[arm] = {
            "prompt_version": version,
            "tasks": len(results),
            "structurally_valid": sum(r["valid"] for r in results),
            "candidates": sum(
                len(r.get("output", {}).get("candidates", [])) for r in results
            ),
        }
    summary["source_hashes_unchanged"] = True
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
