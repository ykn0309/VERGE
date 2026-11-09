"""Independent real-input module replay; never runs or mutates training state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import hashlib
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.cli import build_backend
from verge.config import load_config
from verge.models import Experience, Method
from verge.modules import LLMSelector, LLMGenerator, LLMCritic, CriticSession
from verge.prompts import PROMPT_VERSION

JOB = ContextVar("replay119_job", default={})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source):
    by_role = {role: {} for role in ("selector", "generator", "critic")}
    for line in source.open():
        row = json.loads(line)
        role, payload = row.get("role"), row.get("input")
        if (
            role not in by_role
            or not payload
            or "task" not in payload
            or row.get("attempt") != 1
        ):
            continue
        task = payload["task"]
        old = by_role[role].get(task)
        # Prefer an actual experience application, bounded to this protocol.
        if old is None or role == "generator" and 0 < len(payload["experiences"]) <= 2:
            by_role[role][task] = row
    jobs = []
    for role, tasks in by_role.items():
        rows = list(tasks.values())
        if role == "generator":
            with_e = [r for r in rows if r["input"]["experiences"]]
            without = [r for r in rows if not r["input"]["experiences"]]
            rows = with_e[:15] + without[:5]
            if len(rows) < 20:
                rows += [r for r in tasks.values() if r not in rows][: 20 - len(rows)]
        for index, row in enumerate(rows[:20], 1):
            p = json.loads(json.dumps(row["input"]))
            adaptations = []
            if role == "selector":
                p["max_selected"] = 2
                adaptations.append("max_selected=2；保留历史候选与收益信息")
            elif role == "generator" and len(p["experiences"]) > 2:
                p["experiences"] = p["experiences"][:2]
                adaptations.append("超过两条时仅保留历史输入前两条")
            elif role == "critic":
                p["initial_paths"] = [p.pop("baseline")]
                actual = p.pop("actual", None)
                if actual:
                    p["initial_paths"].append(actual)
                p["actual_experience_ids"] = p.pop("selected_experience_ids", [])
                p["historical_trace_note"] = (
                    "旧实验的真实失败轨迹；只提供已实际运行的路径，缺少的单条/组合对照不可推断。"
                )
                for path in p["initial_paths"]:
                    rubric = path.get("rubric")
                    if rubric:
                        rubric.pop("experience_usage", None)
                        for criterion in rubric["criteria"]:
                            criterion.pop("source_experiences", None)
                adaptations.append(
                    "旧baseline/actual投影为initial_paths；去除旧自报引用，未补造缺失子集结果"
                )
            jobs.append(
                {
                    "role": role,
                    "index": index,
                    "sample_id": row["sample_id"],
                    "payload": p,
                    "adaptations": adaptations,
                }
            )
    return jobs


def replay(source, output):
    output.mkdir(parents=True, exist_ok=False)
    config = load_config("configs/qwen-flash.yaml")
    jobs = prepare(source)
    (output / "inputs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    assert all(
        len({j["payload"]["task"] for j in jobs if j["role"] == role}) == 20
        for role in ("selector", "generator", "critic")
    )
    metadata = {
        "prompt_version": PROMPT_VERSION,
        "model": config["model"]["model"],
        "enable_thinking": config["model"].get("enable_thinking"),
        "source": str(source),
        "source_sha256": sha(source),
        "input_sha256": sha(output / "inputs.json"),
        "code_sha256": {
            str(p): sha(p) for p in [Path("verge/prompts.py"), Path("verge/modules.py")]
        },
        "network_retries": config["model"]["max_retries"],
        "contract_retries": config["requests"]["retries"],
        "scope": "60 independent module jobs, no Generator/Judge pipeline or training updates",
    }
    (output / "protocol.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    lock = Lock()
    backend = build_backend(config["model"])

    def trace(row):
        with lock, (output / "role_io.jsonl").open("a") as f:
            f.write(json.dumps({**JOB.get(), **row}, ensure_ascii=False) + "\n")

    backend.trace_hook = trace

    def run(job):
        token = JOB.set(
            {
                "sample_id": job["sample_id"],
                "replay_role": job["role"],
                "replay_index": job["index"],
            }
        )
        start = time.monotonic()
        p, role = job["payload"], job["role"]
        try:
            if role == "generator":
                experiences = [
                    Experience(
                        e["experience_id"],
                        Method.from_dict(
                            {
                                k: e[k]
                                for k in ("trigger", "principle", "application_guide")
                            }
                        ),
                        "historical",
                    )
                    for e in p["experiences"]
                ]
                value = (
                    LLMGenerator(backend, **config["requests"])
                    .generate(p["task"], experiences)
                    .to_dict()
                )
            elif role == "selector":
                pool = {
                    e["experience_id"]: Experience(
                        e["experience_id"],
                        Method.from_dict(
                            {
                                k: e[k]
                                for k in ("trigger", "principle", "application_guide")
                            }
                        ),
                        "historical",
                    )
                    for e in p["candidates"]
                }
                candidates = [
                    SimpleNamespace(
                        experience_id=e["experience_id"],
                        source=e["retrieval_source"],
                        historical_score=e["historical_reward"],
                    )
                    for e in p["candidates"]
                ]
                scored = [
                    SimpleNamespace(
                        experience_id=e["experience_id"], tasks=e["evidence_tasks"]
                    )
                    for e in p["candidates"]
                ]
                routes = SimpleNamespace(
                    candidates=candidates, historical=SimpleNamespace(scored=scored)
                )
                selected, reasons = LLMSelector(
                    backend, 2, **config["requests"]
                ).select(p["task"], routes, pool)
                value = {
                    "selected_experience_ids": [e.experience_id for e in selected],
                    "selection_reasons": reasons,
                }
            else:
                analysis, candidates = LLMCritic(
                    backend, 2, **config["requests"]
                ).propose(
                    p,
                    source_task_id=job["sample_id"],
                    allowed_parent_ids=p.get("allowed_parent_ids", []),
                    session=CriticSession(),
                )
                value = {
                    "analysis": analysis,
                    "candidates": [e.to_dict() for e in candidates],
                }
            result = {**job, "output": value, "error": None}
        except Exception as exc:
            result = {**job, "output": None, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            JOB.reset(token)
        result["seconds"] = time.monotonic() - start
        with lock, (output / "results.jsonl").open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for future in as_completed([executor.submit(run, j) for j in jobs]):
            result = future.result()
            results.append(result)
            print(
                f'{len(results)}/60 {result["role"]} {result["index"]} {"ERROR" if result["error"] else "ok"}',
                flush=True,
            )
    (output / "summary.json").write_text(
        json.dumps(
            {
                role: {
                    "total": 20,
                    "valid": sum(
                        r["error"] is None for r in results if r["role"] == role
                    ),
                }
                for role in ("selector", "generator", "critic")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    replay(
        Path("runs/history-learning-general100-qwen-116/role_io.jsonl"),
        Path(sys.argv[1]),
    )
