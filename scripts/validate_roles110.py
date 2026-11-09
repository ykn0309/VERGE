"""Fixed independent historical replays of the four new LLM role contracts."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from threading import Lock
import argparse, hashlib, json, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.cli import build_backend
from verge.config import load_config
from verge.models import Method, Experience, Rubric
from verge.modules import (
    LLMSelector,
    LLMGenerator,
    LLMJudge,
    LLMCritic,
    CriticSession,
    experience_payload,
)
from verge.prompts import PROMPT_VERSION
from verge.ranking import ExperienceRanking, ExperienceScore, GroupEvidence
from verge.feedback import ParticipationStats
from verge.retrieval import CandidateRoutes, CandidateQuota, ExperienceCandidate
from verge.llm import configure_request_limit

JOB = ContextVar("role_job", default={})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, queries, output):
    if output.exists():
        raise ValueError("new output required")
    selected = json.loads(queries.read_text())
    rows = [json.loads(line) for line in (source / "results.jsonl").open()]
    paired = {(r["task_id"], r["arm"]): r for r in rows}
    memory = json.loads((source / "frozen_experience_memory.json").read_text())
    methods = {e["experience_id"]: e["method"] for e in memory["experiences"]}
    previous = json.loads(
        Path("artifacts/candidates110-historical20-v1/results.json").read_text()
    )
    candidates = {r["task_id"]: r["candidates"] for r in previous}
    saved_routes = {r["task_id"]: r for r in previous}

    def experience(eid):
        raw = methods[eid]
        # Fixture-only projection: the old frozen pool had no third field.
        return Experience(
            eid,
            Method(
                raw["trigger"],
                raw["principle"],
                "结合当前任务落实核验方法，不把适用条件当作事实，不新增任务要求。",
            ),
            "historical-source",
        ).to_dict()

    jobs = []
    for q in selected:
        tid = q["task_id"]
        base = paired[tid, "baseline"]
        actual = paired[tid, "experience"]
        provided = [e["experience_id"] for e in candidates[tid]]
        cited = sorted(
            {
                eid
                for c in actual["rubric"]["criteria"]
                for eid in c["source_experiences"]
            }
        )
        common = {"task_id": tid, "task": q["task"]}
        jobs.append(
            {
                **common,
                "role": "selector",
                "experiences": [experience(eid) for eid in provided],
                "route": saved_routes[tid],
            }
        )
        jobs.append(
            {
                **common,
                "role": "generator",
                "experiences": [experience(eid) for eid in actual["selected"]],
            }
        )
        jobs.append(
            {
                **common,
                "role": "judge",
                "rubric": actual["rubric"],
                "chosen": q.get("chosen"),
                "sample_id": q["sample_id"],
            }
        )
    # Resolve raw answers from the source's existing dataset, never from labels.
    dataset = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in Path("runs/helpsteer3-general-consensus-data/train.jsonl").open()
    }
    for job in jobs:
        match = next(q for q in selected if q["task_id"] == job["task_id"])
        sample = dataset[match["sample_id"]]
        if job["role"] == "judge":
            job["chosen"] = sample["chosen"]
            job["rejected"] = sample["rejected"]
    failures = [
        r for r in rows if r["arm"] == "experience" and r["valid"] and not r["correct"]
    ][:20]
    for actual in failures:
        sample = dataset[actual["sample_id"]]
        base = paired[actual["task_id"], "baseline"]
        cited = sorted(
            {
                eid
                for c in actual["rubric"]["criteria"]
                for eid in c["source_experiences"]
            }
        )
        jobs.append(
            {
                "role": "critic",
                "task_id": actual["task_id"],
                "task": sample["task"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "baseline": {"rubric": base["rubric"], "judgment": base["judgment"]},
                "actual": {"rubric": actual["rubric"], "judgment": actual["judgment"]},
                "experiences": [experience(eid) for eid in cited],
            }
        )
    assert len(jobs) == 80 and all(
        len({j["task_id"] for j in jobs if j["role"] == r}) == 20
        for r in ("selector", "generator", "judge", "critic")
    )
    output.mkdir(parents=True)
    (output / "fixtures.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2)
    )
    (output / "code").mkdir()
    paths = list(Path("verge").glob("*.py")) + [Path(__file__)]
    for path in paths:
        (output / "code" / path.name).write_bytes(path.read_bytes())
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "model": "qwen3.5-flash",
                "thinking": False,
                "prompt": PROMPT_VERSION,
                "workers": 8,
                "tasks_per_role": 20,
                "independent_roles": True,
                "no_training": True,
                "fixture_limit": "historic two-field methods receive the same generic application guide for this input adaptation only; this does not test newly learned guides",
                "source_sha256": {
                    str(p): digest(p)
                    for p in [
                        source / "results.jsonl",
                        source / "frozen_experience_memory.json",
                        queries,
                    ]
                },
                "code_sha256": {str(p): digest(p) for p in paths},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def replay(output, config, roles=None):
    if (output / "results.jsonl").exists():
        raise ValueError("results already exist")
    cfg = load_config(config)
    backend = build_backend(cfg["model"])
    lock = Lock()

    def append(name, event):
        with lock:
            with (output / name).open("a") as stream:
                stream.write(
                    json.dumps(
                        {**JOB.get(), **event}, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                )

    backend.trace_hook = lambda e: append("role_io.jsonl", e)
    backend.metadata_hook = lambda e: append("llm_requests.jsonl", e)
    configure_request_limit(12)
    jobs = json.loads((output / "fixtures.json").read_text())
    if roles:
        jobs = [job for job in jobs if job["role"] in roles]

    def work(job):
        JOB.set({"task_id": job["task_id"], "role": job["role"]})
        start = time.monotonic()
        result = {
            "task_id": job["task_id"],
            "role": job["role"],
            "output": None,
            "error": None,
        }
        try:
            es = [Experience.from_dict(e) for e in job.get("experiences", [])]
            if job["role"] == "selector":
                route = job["route"]
                evidence = []
                for item in route["scored"]:
                    groups = []
                    for group in item["evidence"]:
                        statistics = dict(group["statistics"])
                        for key in ("record_ids", "task_ids"):
                            statistics[key] = tuple(statistics[key])
                        groups.append(
                            GroupEvidence(
                                group["group_id"],
                                group["similarity"],
                                ParticipationStats(**statistics),
                            )
                        )
                    evidence.append(
                        ExperienceScore(
                            item["experience_id"], item["score"], tuple(groups)
                        )
                    )
                routes = CandidateRoutes(
                    tuple(route["task_groups"].items()),
                    ExperienceRanking(tuple(evidence), tuple(route["unknown_ids"])),
                    tuple(tuple(x) for x in route["semantic"]),
                    tuple(ExperienceCandidate(**c) for c in route["candidates"]),
                    CandidateQuota(),
                )
                selected, reasons = LLMSelector(backend).select(
                    job["task"], routes, {e.experience_id: e for e in es}
                )
                value = {
                    "selected_experience_ids": [e.experience_id for e in selected],
                    "reasons": reasons,
                }
            elif job["role"] == "generator":
                value = LLMGenerator(backend).generate(job["task"], es).to_dict()
            elif job["role"] == "judge":
                value = (
                    LLMJudge(backend)
                    .judge(
                        job["task"],
                        Rubric.from_dict(job["rubric"]),
                        job["chosen"],
                        job["rejected"],
                    )
                    .to_dict()
                )
            else:
                payload = {
                    k: job[k]
                    for k in ("task", "chosen", "rejected", "baseline", "actual")
                }
                payload["experiences"] = [experience_payload(e) for e in es]
                analysis, proposals = LLMCritic(backend).propose(
                    payload,
                    source_task_id=job["task_id"],
                    allowed_parent_ids=[e.experience_id for e in es],
                    session=CriticSession(),
                )
                value = {
                    "analysis": analysis,
                    "candidates": [e.to_dict() for e in proposals],
                }
            result["output"] = value
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.monotonic() - start
        append("results.jsonl", result)
        return result

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(work, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 10 == 0:
                    print(f"{index}/{len(jobs)}", flush=True)
    finally:
        backend.close()
    results = [json.loads(line) for line in (output / "results.jsonl").open()]
    summary = {
        role: {
            "tasks": sum(r["role"] == role for r in results),
            "errors": sum(
                r["role"] == role and r["error"] is not None for r in results
            ),
        }
        for role in ("selector", "generator", "judge", "critic")
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/qwen-flash.yaml")
    parser.add_argument(
        "--roles", nargs="+", choices=["selector", "generator", "judge", "critic"]
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            Path("runs/midterm-general-v2-assumed-early-stop-paired200"),
            Path("artifacts/learning110-historical20-v1/fixed_inputs.json"),
            args.output,
        )
    else:
        replay(args.output, args.config, args.roles)
