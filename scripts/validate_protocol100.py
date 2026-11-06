"""Isolated role replays for protocol100; never writes historical runs or live pools."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, hashlib, time, threading, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from copy import deepcopy
from verge.cli import components
from verge.models import (
    Experience,
    Method,
    PreferenceSample,
    Rubric,
    RubricJudgment,
    ApplicationRecord,
    CheckerReport,
    ExperienceReview,
)
from verge.prompts import NumberedIDCodec
from verge.llm import configure_request_limit

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "runs/midterm-general-v2-assumed-train-300-batch3/artifacts"
FIXED = ROOT / "artifacts/experience-utilization-audit/fixed_inputs.json"
JOB = ContextVar("module100_job", default={})


def unique(rows, key, limit=20):
    seen = set()
    out = []
    for x in rows:
        k = key(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) == limit:
            break
    if len(out) < limit:
        raise ValueError("not enough distinct historical tasks")
    return out


def prepare(out):
    rows = [json.loads(l) for l in (HIST / "trajectories.jsonl").open()]
    frozen = json.loads(
        (
            ROOT
            / "runs/midterm-general-v2-frozen-early-stop/artifacts/experience_memory.json"
        ).read_text()
    )
    versions = {
        e["experience_id"]: Experience.from_dict(e) for e in frozen["experiences"]
    }
    audit = json.loads(FIXED.read_text())
    jobs = []
    # Fixed 12 regression + 4 improvement + 4 normal per selector/generator.
    for role in ("selector", "initial_generator"):
        subset = [r for r in audit if r["role"] == role and r["split"] == "development"]
        selected = sum(
            (
                [r for r in subset if r["group"] == g][:n]
                for g, n in [("regression", 12), ("improvement", 4), ("normal", 4)]
            ),
            [],
        )
        assert len(selected) == 20
        for r in selected:
            inp = deepcopy(r["input"])
            key = "candidates" if role == "selector" else "experiences"
            reverse = r["id_mapping"]["reverse"].get("experience", {})
            es = []
            for item in inp[key]:
                eid = reverse.get(
                    str(item["experience_id"]), str(item["experience_id"])
                )
                e = deepcopy(versions[eid])
                es.append(e.to_dict())
            jobs.append(
                {
                    "role": role,
                    "task_id": r["sample_id"],
                    "group": r["group"],
                    "task": inp["task"],
                    "experiences": es,
                    "original_output": r["original_output"],
                }
            )
    failed = unique(
        [
            r
            for r in rows
            if r.get("initial_judgment") and not r["initial_judgment"]["success"]
        ],
        lambda r: r["sample"]["task"],
    )
    for r in failed:
        jobs.append(
            {
                "role": "critic_repair",
                "task_id": r["sample_id"],
                "sample": r["sample"],
                "rubric": r["initial_rubric"],
                "judgment": r["initial_judgment"],
            }
        )
        jobs.append(
            {
                "role": "critic_assess",
                "task_id": r["sample_id"],
                "sample": r["sample"],
                "rubric": r["initial_rubric"],
                "judgment": r["initial_judgment"],
            }
        )
    apps = unique(
        [a for r in rows for a in r.get("applications", []) if a.get("cited")],
        lambda a: a["sample"]["task"],
    )
    for a in apps:
        for role in ("execution_checkers", "experience_updater"):
            jobs.append(
                {
                    "role": role,
                    "task_id": a["sample"]["sample_id"],
                    "application": a,
                    "experience": versions[a["version_id"]].to_dict(),
                }
            )
    proposals = unique(
        [
            {
                "sample": r["sample"],
                "rubric": r["initial_rubric"],
                "judgment": r["initial_judgment"],
                "candidate": a["proposal"]["candidates"][0],
                "review": a.get("review"),
            }
            for r in rows
            for a in r.get("proposal_attempts", [])
            if a.get("proposal")
            and a["proposal"].get("candidates")
            and a.get("review")
            and not a["review"]["approved"]
        ],
        lambda x: x["sample"]["task"],
    )
    for x in proposals:
        jobs.append(
            {"role": "critic_rewrite", "task_id": x["sample"]["sample_id"], **x}
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "fixtures.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            FIXED,
            HIST / "trajectories.jsonl",
            ROOT
            / "runs/midterm-general-v2-frozen-early-stop/artifacts/experience_memory.json",
        ]
    }
    (out / "protected_hashes.json").write_text(json.dumps(hashes, indent=2))
    print("prepared", len(jobs), flush=True)


def run(out):
    config = __import__("yaml").safe_load(
        (
            ROOT / "configs/helpsteer3-general-v2-critic-zh-short-qwen-flash-30.yaml"
        ).read_text()
    )
    config["modules"]["retrieval"]["implementation"] = "fake"
    # Test call roles individually; no pipeline, no training/evolution.
    configure_request_limit(12)
    lock = threading.Lock()

    def append(name, obj):
        with lock:
            with (out / name).open("a") as f:
                f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")

    def worker(job):
        JOB.set({"role_under_test": job["role"], "sample_id": job["task_id"]})
        m = components(config)
        bs = {id(m.backend): m.backend}
        m.backend.trace_hook = lambda e: append("role_io.jsonl", {**JOB.get(), **e})
        m.backend.metadata_hook = lambda e: append("requests.jsonl", {**JOB.get(), **e})
        begin = time.monotonic()
        result = {
            "role": job["role"],
            "sample_id": job["task_id"],
            "input": job,
            "error": None,
        }
        try:
            role = job["role"]
            if role in ("selector", "initial_generator"):
                es = [Experience.from_dict(x) for x in job["experiences"]]
                if role == "selector":
                    selected = m.selector.select(job["task"], es)
                    value = {
                        "selected_experience_ids": [e.experience_id for e in selected],
                        "reasons": m.selector.last_reasons,
                    }
                else:
                    value = m.initial_generator.generate(job["task"], es).to_dict()
            elif role in (
                "critic_repair",
                "critic_assess",
                "critic_extract",
                "experience_checker",
                "critic_rewrite",
            ):
                sample = PreferenceSample.from_dict(job["sample"])
                rubric = Rubric.from_dict(job["rubric"])
                judgment = RubricJudgment.from_dict(job["judgment"])
                if role == "critic_repair":
                    r, changes = m.critic.repair(
                        sample,
                        rubric,
                        judgment,
                        1,
                        conversation=[],
                        codec=NumberedIDCodec(),
                    )
                    value = {"rubric": r.to_dict(), "changes": changes}
                elif role == "critic_assess":
                    checker = CheckerReport(
                        "judge",
                        "unknown",
                        "历史样本未记录整套rubric核对，独立复测不伪造核对结论",
                    )
                    j, reason = m.critic.assess(sample, rubric, judgment, checker)
                    value = {"judgment": j.to_dict(), "reason": reason}
                elif role == "critic_extract":
                    items = m.critic.extract(
                        sample, rubric, judgment, job["changes"], job["repair_evidence"]
                    )
                    value = [
                        {
                            "candidate": e.to_dict(),
                            "change_indices": refs,
                            "reason": reason,
                        }
                        for e, refs, reason in items
                    ]
                elif role == "experience_checker":
                    candidate = Experience.from_dict(job["candidate"])
                    value = m.experience_checker.check(
                        sample,
                        rubric,
                        judgment,
                        candidate,
                        repair_evidence=job.get("repair_evidence"),
                    ).to_dict()
                else:
                    candidate = Experience.from_dict(job["candidate"])
                    value = m.critic.rewrite(
                        candidate,
                        ExperienceReview.from_dict(job["review"]),
                        {
                            "task": sample.task,
                            "rubric": rubric.to_dict(),
                            "judgment": judgment.to_dict(),
                        },
                    ).to_dict()
            else:
                a = ApplicationRecord.from_dict(job["application"])
                if role == "execution_checkers":
                    value = [r.to_dict() for r in m.checkers.check(a)]
                else:
                    value = m.experience_updater.update(
                        Experience.from_dict(job["experience"]),
                        [a],
                        ["历史异常独立复测"],
                        case_kinds={a.sample.sample_id: "problem"},
                    ).to_dict()
            result["output"] = value
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        finally:
            result["seconds"] = time.monotonic() - begin
            m.backend.close()
        append("results.jsonl", result)
        return result

    fixtures = json.loads((out / "fixtures.json").read_text())
    if (out / "results.jsonl").exists():
        raise ValueError("new output required; no blind resume")
    shutil.copytree(
        ROOT / "verge",
        out / ".source/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "model": "qwen3.5-flash",
                "thinking": False,
                "workers": 8,
                "request_limit": 12,
                "roles": "fixed independent historical inputs; derived extraction inputs frozen after repair outputs",
                "guide_migration": "preserve trigger/principle, new application guide defaults; optional negatives unknown; not learned reorganization",
                "source_hashes": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / "verge").glob("*.py")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, j) for j in fixtures]
        results = []
        for f in as_completed(futures):
            results.append(f.result())
            print(len(results), len(fixtures), flush=True)
    # New extract/checker roles have no old equivalent inputs; freeze task-level repair outputs first.
    derived = []
    for r in results:
        if r["role"] != "critic_repair" or r["error"]:
            continue
        j = r["input"]
        base = {
            "sample": j["sample"],
            "rubric": r["output"]["rubric"],
            "judgment": j["judgment"],
            "changes": r["output"]["changes"],
            "repair_evidence": {
                "initial_rubric": j["rubric"],
                "initial_judgment": j["judgment"],
                "note": "独立提炼输入，不冒称修复已验证",
            },
        }
        # Match revised IDs with neutral unknown original judgments? Actual Judge is required; run independently and save it.
        # Extraction below uses historical initial criterion judgments only as before evidence; revised judgment generated by separate Judge.
        derived.append({"role": "critic_extract", "task_id": r["sample_id"], **base})
    # Explicitly run actual Judge on fixed repaired rubrics; no selection/training occurs.
    judged = []

    def judge_input(j):
        m = components(config)
        m.backend.trace_hook = lambda e: append(
            "role_io.jsonl",
            {"role_under_test": "judge_fixture", "sample_id": j["task_id"], **e},
        )
        m.backend.metadata_hook = lambda e: append(
            "requests.jsonl",
            {"role_under_test": "judge_fixture", "sample_id": j["task_id"], **e},
        )
        try:
            s = PreferenceSample.from_dict(j["sample"])
            r = Rubric.from_dict(j["rubric"])
            jj = m.judge.judge(r, s.chosen, s.rejected, task=s.task)
            j["judgment"] = jj.to_dict()
            return j
        except Exception as exc:
            append(
                "fixture_errors.jsonl", {"sample_id": j["task_id"], "error": str(exc)}
            )
            return None
        finally:
            m.backend.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        judged = [x for x in pool.map(judge_input, derived) if x]
    (out / "extraction_fixtures.json").write_text(
        json.dumps(judged, ensure_ascii=False, indent=2)
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        extracts = list(pool.map(worker, judged))
    checks = []
    for r in extracts:
        if r["error"] or not r.get("output"):
            continue
        j = r["input"]
        c = r["output"][0]
        checks.append(
            {
                "role": "experience_checker",
                "task_id": r["sample_id"],
                "sample": j["sample"],
                "rubric": j["rubric"],
                "judgment": j["judgment"],
                "candidate": c["candidate"],
                "repair_evidence": {
                    **j["repair_evidence"],
                    "changes": j["changes"],
                    "change_indices": c["change_indices"],
                    "extraction_reason": c["reason"],
                },
            }
        )
    (out / "review_fixtures.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2)
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, checks))
    print("finished derived", len(judged), len(checks), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "run"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    prepare(out) if a.stage == "prepare" else run(out)
