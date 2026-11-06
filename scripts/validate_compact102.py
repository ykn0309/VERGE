"""Fixed historical role replays for102; no training or live-memory writes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, hashlib, shutil, threading, time, argparse
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from verge.cli import components
from verge.models import (
    PreferenceSample,
    Rubric,
    RubricJudgment,
    Experience,
    ExperienceReview,
)
from verge.prompts import NumberedIDCodec, PROMPT_VERSION
from verge.llm import configure_request_limit

ROOT = Path(__file__).resolve().parents[1]
JOB = ContextVar("compact102_job", default={})


def prepare(out):
    old = ROOT / "artifacts/protocol100-module-replay-v1"
    original = json.loads((old / "fixtures.json").read_text())
    extractions = json.loads((old / "extraction_fixtures.json").read_text())
    outputs = [json.loads(l) for l in (old / "results.jsonl").open()]
    jobs = []
    for role in ("critic_repair", "critic_assess", "critic_rewrite"):
        rows = [j for j in original if j["role"] == role]
        assert len(rows) == 20 and len({j["sample"]["task"] for j in rows}) == 20
        for j in rows:
            j = json.loads(json.dumps(j))
            if role == "critic_rewrite":
                j["role"] = "critic_rewrite_batch"
            jobs.append(j)
    jobs += extractions
    byid = {
        r["sample_id"]: r
        for r in outputs
        if r["role"] == "critic_extract" and not r["error"]
    }
    for j in extractions:
        es = byid[j["task_id"]]["output"][:2]
        assert es
        evidence = {
            **j["repair_evidence"],
            "rubric": j["rubric"],
            "final_judgment": j["judgment"],
            "changes": j["changes"],
            "candidate_support": [
                {
                    "experience_id": x["candidate"]["experience_id"],
                    "change_indices": x["change_indices"],
                    "reason": x["reason"],
                }
                for x in es
            ],
        }
        jobs.append(
            {
                "role": "experience_checker_batch",
                "task_id": j["task_id"],
                "sample": j["sample"],
                "repair_evidence": evidence,
                "candidates": [x["candidate"] for x in es],
            }
        )
    assert len(jobs) == 100
    out.mkdir(exist_ok=False, parents=True)
    (out / "fixtures.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "model": "qwen3.5-flash",
                "thinking": False,
                "workers": 8,
                "limit": 12,
                "prompt": PROMPT_VERSION,
                "scope": "5 changed roles x20 task-deduplicated actual historic inputs; isolated role replays, not source repair success or held-out evaluation",
                "batch_inputs": "20 same-task batches of up to2 original candidates, not paired across unrelated tasks; rewrite uses20 historical rejected candidates",
                "protected": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        old / "fixtures.json",
                        old / "extraction_fixtures.json",
                        old / "results.jsonl",
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run(out):
    c = yaml.safe_load(
        (ROOT / "configs/helpsteer3-general-role-guides-qwen-flash-20.yaml").read_text()
    )
    c["modules"]["retrieval"]["implementation"] = "fake"
    configure_request_limit(12)
    lock = threading.Lock()
    if (out / "results.jsonl").exists():
        raise ValueError("new output required")
    shutil.copytree(
        ROOT / "verge",
        out / ".source/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    def save(name, x):
        with lock:
            with (out / name).open("a") as f:
                f.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n")

    def work(j):
        JOB.set({"role_under_test": j["role"], "sample_id": j["task_id"]})
        m = components(c)
        m.backend.trace_hook = lambda e: save("role_io.jsonl", {**JOB.get(), **e})
        m.backend.metadata_hook = lambda e: save("requests.jsonl", {**JOB.get(), **e})
        begin = time.monotonic()
        row = {"role": j["role"], "sample_id": j["task_id"], "input": j, "error": None}
        try:
            sample = PreferenceSample.from_dict(j["sample"])
            role = j["role"]
            if role == "critic_repair":
                base, reason, r, changes = m.critic.start(
                    sample,
                    Rubric.from_dict(j["rubric"]),
                    RubricJudgment.from_dict(j["judgment"]),
                    conversation=[],
                    codec=NumberedIDCodec(),
                )
                value = {
                    "baseline_judgment": base.to_dict(),
                    "baseline_assessment": reason,
                    "diagnosis": m.critic.last_diagnosis,
                    "rubric": r.to_dict(),
                    "changes": changes,
                }
            elif role == "critic_assess":
                result, reason = m.critic.assess(
                    sample,
                    Rubric.from_dict(j["rubric"]),
                    RubricJudgment.from_dict(j["judgment"]),
                )
                value = {
                    "judgment": result.to_dict(),
                    "reason": reason,
                    "diagnosis": m.critic.last_diagnosis,
                }
            elif role == "critic_extract":
                xs = m.critic.extract(
                    sample,
                    Rubric.from_dict(j["rubric"]),
                    RubricJudgment.from_dict(j["judgment"]),
                    j["changes"],
                    j["repair_evidence"],
                )
                value = [
                    {"candidate": e.to_dict(), "change_indices": refs, "reason": why}
                    for e, refs, why in xs
                ]
            elif role == "experience_checker_batch":
                result = m.experience_checker.check_many(
                    sample,
                    j["repair_evidence"],
                    [Experience.from_dict(x) for x in j["candidates"]],
                )
                value = {
                    k: (
                        v.to_dict()
                        if v is not None
                        else {"technical_error": m.experience_checker.last_errors[k]}
                    )
                    for k, v in result.items()
                }
            else:
                e = Experience.from_dict(j["candidate"])
                reviews = {e.experience_id: ExperienceReview.from_dict(j["review"])}
                context = {
                    "task": sample.task,
                    "chosen": sample.chosen,
                    "rejected": sample.rejected,
                    "rubric": j["rubric"],
                    "judgment": j["judgment"],
                }
                result = m.critic.rewrite_many([e], reviews, context)
                value = {k: v.to_dict() for k, v in result.items()}
            row["output"] = value
        except Exception as e:
            row["error"] = {"type": type(e).__name__, "message": str(e)[:500]}
        finally:
            m.backend.close()
            row["seconds"] = time.monotonic() - begin
        save("results.jsonl", row)
        return row

    jobs = json.loads((out / "fixtures.json").read_text())
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs = [pool.submit(work, j) for j in jobs]
        for i, f in enumerate(as_completed(fs), 1):
            f.result()
            print(i, len(jobs), flush=True)
    save("timing.jsonl", {"seconds": time.monotonic() - start})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "run"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    prepare(out) if a.stage == "prepare" else run(out)
