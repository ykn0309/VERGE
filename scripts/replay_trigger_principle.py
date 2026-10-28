"""Independent method-only role replays from the retained real 30-task run."""

from __future__ import annotations
import hashlib, json, sys, shutil, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import (
    PreferenceSample,
    Rubric,
    RubricJudgment,
    Experience,
    Method,
    ApplicationRecord,
    Model,
    task_identity,
)
from verge.modules import (
    LLMInitialRubricGenerator,
    LLMCritic,
    LLMSelector,
    LLMExperienceChecker,
    LLMExperienceUpdater,
    LLMCheckers,
)
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMSampleError
from verge.prompts import NumberedIDCodec, PROMPT_VERSION

ROOT = Path("artifacts/v2-trigger-principle-84")
SOURCE = Path("runs/helpsteer3-general-v2-experience-review-qwen-flash-30/artifacts")


def project(x):
    if isinstance(x, list):
        return [project(v) for v in x]
    if not isinstance(x, dict):
        return x
    out = {k: project(v) for k, v in x.items()}
    if "trigger" in out and "principle" in out:
        out.pop("template", None)
        out.pop("parameters", None)
    if "question" in out and "source_experiences" in out:
        out.pop("bindings", None)
        out.pop("binding_evidence", None)
    return out


def main():
    if ROOT.exists():
        raise ValueError("new output directory required")
    sources = [
        SOURCE / "trajectories.jsonl",
        SOURCE / "role_io.jsonl",
        SOURCE / "experience_memory.json",
    ]
    records = [json.loads(l) for l in sources[0].read_text().splitlines()]
    traces = [
        json.loads(l)
        for l in sources[1].read_text().splitlines()
        if json.loads(l).get("event_type") != "contract_validation"
    ]
    memory = json.loads(sources[2].read_text())
    pool = {e["experience_id"]: e for e in memory["experiences"]}
    rows = []

    def add(role, sample_id, data, note):
        n = 1 + sum(r["role"] == role for r in rows)
        rows.append(
            {
                "key": f"{role}-{n:02}",
                "role": role,
                "sample_id": sample_id,
                "input": project(data),
                "conversion": note,
            }
        )

    chosen = [r for r in records if not r["initial_judgment"]["success"]] + [
        r for r in records if r["initial_judgment"]["success"]
    ][:5]
    assert len(chosen) == 20
    for r in chosen:
        options = [
            t
            for t in traces
            if t.get("sample_id") == r["sample_id"] and t["role"] == "initial_generator"
        ]
        t = next((t for t in options if t["input"].get("experiences")), options[0])
        experiences = [
            {
                "experience_id": e["experience_id"],
                "method": {k: e[k] for k in ("trigger", "principle")},
                "source_sample_id": "unavailable-history-metadata",
            }
            for e in t["input"]["experiences"]
        ]
        add(
            "initial_generator",
            r["sample_id"],
            {"task": t["input"]["task"], "experiences": experiences},
            "Actual stored Generator request, first with supplied experience preferred. Delete template/parameters/bindings, no textual reconstruction; source ID placeholder is private adapter metadata never sent to model.",
        )
        add(
            "critic",
            r["sample_id"],
            {
                "sample": r["sample"],
                "rubric": r["initial_rubric"],
                "judgment": r["initial_judgment"],
            },
            "Exact real initial task/answers/rubric/Judge snapshot. 15 failed and5 successful tasks. Successful cases are boundary replay inputs, not claims Critic was historically called on success. No failure manufactured.",
        )
    seen = set()
    for t in traces:
        if t["role"] != "selector" or t["sample_id"] in seen:
            continue
        seen.add(t["sample_id"])
        p = t["input"]
        add(
            "selector",
            t["sample_id"],
            {
                "task": p["task"],
                "experiences": [
                    {
                        "experience_id": e["experience_id"],
                        "method": {k: e[k] for k in ("trigger", "principle")},
                        "source_sample_id": "unavailable-history-metadata",
                    }
                    for e in p["candidates"]
                ],
            },
            "Actual collection Selector request, first per task; supplied local IDs retained, optional fields removed; no Q/counts added to visible payload.",
        )
        if len(seen) == 20:
            break
    for r in records:
        attempt = next(
            (
                p
                for p in r["proposal_attempts"]
                if p.get("review") and p.get("proposal")
            ),
            None,
        )
        if attempt:
            add(
                "experience_checker",
                r["sample_id"],
                {
                    "sample": r["sample"],
                    "rubric": r["initial_rubric"],
                    "judgment": r["initial_judgment"],
                    "candidate": attempt["proposal"]["candidates"][0],
                },
                "Actual first candidate reviewed on this task; remove template/parameters only, no generated proposal supplied.",
            )
    apps = [a for e in memory["experiences"] for a in e["applications"] if a["cited"]]
    unique = {
        (
            a["sample"]["task_id"]
            if "task_id" in a["sample"]
            else task_identity(a["sample"]["task"])
        ): a
        for a in apps
    }
    for a in unique.values():
        for role in ("selector_checker", "generator_checker", "judge_checker"):
            add(
                role,
                a["sample"]["sample_id"],
                {"application": a},
                "Actual cited immutable-version application; source/downstream evidence preserved; delete optional method/criterion fields only. Other checker slots use no-call stubs.",
            )
        related = [a] + [
            b
            for b in apps
            if b["version_id"] == a["version_id"]
            and b["sample"]["sample_id"] != a["sample"]["sample_id"]
        ]
        related = related[:6]
        add(
            "experience_updater",
            a["sample"]["sample_id"],
            {
                "experience": pool[a["version_id"]],
                "cases": related,
                "reasons": ["historical_review"],
                "case_kinds": {
                    b["sample"]["sample_id"]: (
                        "source"
                        if b["phase"] == "source_validation"
                        else "positive" if b["feedback"] == "positive" else "problem"
                    )
                    for b in related
                },
            },
            "Updater input projected from actual experience and its real applications; collection did not invoke Updater. No online trigger or Q inferred. Case purpose derived from stored phase/feedback.",
        )
    for role in {r["role"] for r in rows}:
        tasks = []
        for r in (r for r in rows if r["role"] == role):
            p = r["input"]
            tasks.append(
                p["task"]
                if "task" in p
                else (
                    p["sample"]["task"]
                    if "sample" in p
                    else (
                        p["application"]["sample"]["task"]
                        if "application" in p
                        else p["cases"][0]["sample"]["task"]
                    )
                )
            )
        assert len(tasks) == len({task_identity(t) for t in tasks})
    ROOT.mkdir()
    (ROOT / "results").mkdir()
    (ROOT / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    manifest = {
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "prompt_version": PROMPT_VERSION,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "input_sha256": hashlib.sha256((ROOT / "inputs.json").read_bytes()).hexdigest(),
        "independent": True,
        "coverage_shortfall": "Old logs explicitly deleted. Only retained run83 available:13 cited applications across12 distinct tasks and11 distinct reviewed-candidate tasks. Those roles tested exhaustively without repeating tasks or inventing historical calls; below AGENTS20 requirement. Updater projection not historical actual invocation.",
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    shutil.copytree(
        "verge",
        ROOT / "code_snapshot/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, ROOT / "code_snapshot/replay_trigger_principle.py")
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
        role = row["role"]
        result = {k: row[k] for k in ("key", "role", "sample_id")}
        try:
            if role == "initial_generator":
                v = LLMInitialRubricGenerator(b).generate(
                    p["task"], [Experience.from_dict(e) for e in p["experiences"]]
                )
            elif role == "selector":
                v = [
                    e.experience_id
                    for e in LLMSelector(b, include_quality=False).select(
                        p["task"], [Experience.from_dict(e) for e in p["experiences"]]
                    )
                ]
            elif role in ("critic", "experience_checker"):
                s = PreferenceSample.from_dict(p["sample"])
                r = Rubric.from_dict(p["rubric"])
                j = RubricJudgment.from_dict(p["judgment"])
                v = (
                    LLMCritic(b).propose(
                        s, r, j, 1, conversation=[], codec=NumberedIDCodec()
                    )
                    if role == "critic"
                    else LLMExperienceChecker(b).check(
                        s, r, j, Experience.from_dict(p["candidate"])
                    )
                )
            elif role == "experience_updater":
                v = LLMExperienceUpdater(b).update(
                    Experience.from_dict(p["experience"]),
                    [ApplicationRecord.from_dict(a) for a in p["cases"]],
                    p["reasons"],
                    case_kinds=p["case_kinds"],
                )
            else:

                class NoCall:
                    def generate(self, *args, **kwargs):
                        return {
                            "status": "unknown",
                            "evidence": "not under test",
                            "details": [],
                        }

                stage = role.removesuffix("_checker")
                bs = {s: NoCall() for s in ("selector", "generator", "judge")}
                bs[stage] = b
                v = next(
                    r
                    for r in LLMCheckers(bs).check(
                        ApplicationRecord.from_dict(p["application"])
                    )
                    if r.stage == stage
                )
            result.update(
                valid=getattr(v, "status", None) != "failed",
                output=v.to_dict() if isinstance(v, Model) else v,
            )
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
            "valid": sum(r["role"] == role and r["valid"] for r in results),
        }
        for role in sorted({r["role"] for r in results})
    }
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        for p, h in manifest["source_sha256"].items()
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT))
    args = parser.parse_args()
    ROOT = Path(args.output)
    main()
