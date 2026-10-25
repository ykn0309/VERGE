"""Independent V2 replays. Inputs are frozen before calls; outputs never feed another role."""

from __future__ import annotations
import argparse, json, hashlib, os, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import *
from verge.modules import *
from verge.calculator import application, calculate_rubric_judgment
from verge.prompts import experience_payload, NumberedIDCodec
from verge.llm import OpenAICompatibleBackend, JUDGE_RESPONSE_SIDE

SOURCE = Path("artifacts/v2-module-replay/source")
OUT = Path("artifacts/v2-module-replay")
CURRENT = ContextVar("replay_case", default={})


def application_record(raw):
    # Offline-only conversion of frozen plan70 inputs; production has no legacy migration.
    raw = json.loads(json.dumps(raw))
    cited = any(
        raw["version_id"] in c["source_experiences"] for c in raw["rubric"]["criteria"]
    )
    raw.update(recalled=cited, selected=cited, cited=cited)
    # Offline conversion: old positive/reverse balance becomes the new tie.
    if raw["feedback"] == "cancellation":
        raw["feedback"] = "tie"
    return ApplicationRecord.from_dict(raw)


def method(old):
    return Method(old["trigger"], old.get("principle", old.get("content", "")))


def exp(old, source="historical"):
    return Experience(old.get("experience_id", "exp_historical"), method(old), source)


def rconvert(old):
    return Rubric(
        old["rubric_id"],
        [
            Criterion(**{**c, "bindings": {}, "binding_evidence": {}})
            for c in old["criteria"]
        ],
    )


def evenly(rows, n=20):
    # Preserve task deduplication. Fixed evenly spaced positions cover early/late pool states.
    seen = set()
    unique = []
    for x in rows:
        key = x["sample_id"]
        if key not in seen:
            seen.add(key)
            unique.append(x)
    if len(unique) < n:
        raise ValueError(f"only {len(unique)} historical tasks")
    return [unique[round(i * (len(unique) - 1) / (n - 1))] for i in range(n)]


def prepare():
    logs = [json.loads(l) for l in (SOURCE / "role_io.jsonl").open()]
    dataset = json.loads((SOURCE / "run_manifest.json").read_text())["config"][
        "dataset"
    ]["config"]["paths"]["train"]
    samples = {
        x["sample_id"]: x
        for x in map(json.loads, Path(dataset).read_text().splitlines())
    }
    inputs = []
    for role in ("selector", "initial_generator", "judge", "critic"):
        rows = [x for x in logs if x["role"] == role and x.get("error") is None]
        if role == "critic":
            rows = [
                x
                for x in rows
                if all(
                    len(c["source_experiences"]) <= 1
                    for c in x["input"]["current_rubric"]["criteria"]
                )
            ]
        for i, row in enumerate(evenly(rows), 1):
            p = row["input"]
            data = {"task": p["task"]}
            if role == "selector":
                members = []
                for item in p["candidates"]:
                    members.extend(item.get("members", [item]))
                data["experiences"] = [exp(e).to_dict() for e in members]
            elif role == "initial_generator":
                data["experiences"] = [
                    exp(e).to_dict() for e in p["selected_experiences"]
                ]
            elif role == "judge":
                data = {
                    "sample": samples[row["sample_id"]],
                    "rubric": Rubric(
                        "historical_rubric",
                        [
                            Criterion(c["criterion_id"], c["question"], 1, [])
                            for c in p["criteria"]
                        ],
                    ).to_dict(),
                }
            elif role == "critic":
                data = {
                    "sample": samples[row["sample_id"]],
                    "rubric": rconvert(p["current_rubric"]).to_dict(),
                    "judgment": p["current_judgment"],
                }
            inputs.append(
                {
                    "key": f"{role}-{i:02}",
                    "role": role,
                    "sample_id": row["sample_id"],
                    "input": data,
                    "source": "plan69 role_io.jsonl",
                    "conversion": "content→principle; template=null/parameters={}; old P/N/categories removed; old quality not transferred. Judge uses original pair, criteria weights unavailable in Judge input are set to 1 only for its irrelevant score container. Critic is single-turn replay of actual input; history not reconstructed.",
                }
            )
    older = json.loads((SOURCE / "module67_inputs.json").read_text())
    for i, row in enumerate(
        evenly([x for x in older if x["role"] == "experience_updater"]), 1
    ):
        p = row["input"]
        data = {
            "experience": experience_payload(
                exp(p["current_experience"]), quality=True
            ),
            "cases": p["cases"],
            "reasons": ["historical_review"],
        }
        inputs.append(
            {
                "key": f"experience_updater-{i:02}",
                "role": "experience_updater",
                "sample_id": row["sample_id"],
                "input": data,
                "source": "plan67 independent replay inputs.json",
                "conversion": "content→principle, no optional template; retain all actual case texts and criterion judgments. Old input omitted weights and Checker reports; neither fabricated nor inferred. Q unestimated. Original window preserved rather than fabricating six cases.",
            }
        )
    trajectories = [json.loads(l) for l in (SOURCE / "trajectories.jsonl").open()]
    usable = []
    for t in trajectories:
        if not t["initial_judgment"] or not t["selected_experiences"]:
            continue
        if any(
            len(c["source_experiences"]) > 1 for c in t["initial_rubric"]["criteria"]
        ):
            continue
        for old in t["selected_experiences"]:
            eid = old["experience_id"]
            if any(
                eid in c["source_experiences"] for c in t["initial_rubric"]["criteria"]
            ):
                usable.append(
                    {"sample_id": t["sample_id"], "trajectory": t, "experience": old}
                )
                break
    # Stratify actual cited records by direction, then evenly sample each stratum.
    apps = []
    for row in usable:
        t = row["trajectory"]
        e = exp(row["experience"])
        s = PreferenceSample.from_dict(samples[row["sample_id"]])
        r = rconvert(t["initial_rubric"])
        j = RubricJudgment.from_dict(t["initial_judgment"])
        a = application(s, e, r, j)
        apps.append(
            {"sample_id": s.sample_id, "app": a.to_dict(), "direction": a.feedback}
        )
    selected = []
    for label in ("positive", "reverse", "tie"):
        group = [a for a in apps if a["direction"] == label]
        if group:
            selected.extend(
                evenly(group, min(5, len(group))) if len(group) > 1 else group
            )
    selected_ids = {a["sample_id"] for a in selected}
    for a in evenly(apps):
        if len(selected) >= 20:
            break
        if a["sample_id"] not in selected_ids:
            selected.append(a)
            selected_ids.add(a["sample_id"])
    if len(selected) != 20:
        raise ValueError("need 20 distinct cited tasks")
    for stage in ("selector", "generator", "judge"):
        role = stage + "_checker"
        for i, row in enumerate(selected, 1):
            inputs.append(
                {
                    "key": f"{role}-{i:02}",
                    "role": role,
                    "sample_id": row["sample_id"],
                    "input": {"application": row["app"]},
                    "source": "plan69 actual R0 trajectory and selected method snapshot",
                    "conversion": "Historical method renamed to Principle, no new model outputs used; current Checker projects only its allowed evidence. Four raw outcomes and original criterion weights preserved.",
                }
            )
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "model": "qwen3.5-flash",
                "enable_thinking": False,
                "per_role": 20,
                "roles": 8,
                "retries": 2,
                "independent": True,
                "baseline_protocol": "plan68",
                "conversion": "explicit per input; old cancellation maps to net-direction tie during isolated conversion only",
                "output_feedback": False,
                "input_sha256": hashlib.sha256(
                    (OUT / "inputs.json").read_bytes()
                ).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Frozen inputs:", len(inputs), flush=True)


def execute(workers=4):
    import shutil

    snapshot = OUT / "code_snapshot"
    if not snapshot.exists():
        shutil.copytree(
            "verge", snapshot / "verge", ignore=shutil.ignore_patterns("__pycache__")
        )
        shutil.copy2(__file__, snapshot / "replay_v2_modules.py")
    hashes = {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*.py")
    }
    (OUT / "execution_code.json").write_text(json.dumps(hashes, indent=2))
    rows = json.loads((OUT / "inputs.json").read_text())
    lock = threading.Lock()
    (OUT / "results").mkdir(exist_ok=True)

    def run(row):
        target = OUT / "results" / (row["key"] + ".json")
        if target.exists():
            return json.loads(target.read_text())
        token = CURRENT.set({"key": row["key"], "sample_id": row["sample_id"]})
        b = OpenAICompatibleBackend(
            "qwen3.5-flash",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY_2",
            enable_thinking=False,
            timeout_seconds=120,
            max_retries=2,
        )

        def trace(event):
            with lock:
                with (OUT / "role_io.jsonl").open("a") as f:
                    f.write(
                        json.dumps({**CURRENT.get(), **event}, ensure_ascii=False)
                        + "\n"
                    )

        b.trace_hook = trace

        def metadata(event):
            with lock:
                with (OUT / "llm_requests.jsonl").open("a") as f:
                    f.write(
                        json.dumps({**CURRENT.get(), **event}, ensure_ascii=False)
                        + "\n"
                    )

        b.metadata_hook = metadata
        p = row["input"]
        role = row["role"]
        result = {"key": row["key"], "role": role, "sample_id": row["sample_id"]}
        try:
            if role == "selector":
                value = LLMSelector(b).select(
                    p["task"], [Experience.from_dict(e) for e in p["experiences"]]
                )
                value = [e.experience_id for e in value]
            elif role == "initial_generator":
                value = LLMInitialRubricGenerator(b).generate(
                    p["task"], [Experience.from_dict(e) for e in p["experiences"]]
                )
            elif role == "judge":
                s = PreferenceSample.from_dict(p["sample"])
                value = LLMJudge(b).judge(
                    Rubric.from_dict(p["rubric"]), s.chosen, s.rejected, task=s.task
                )
            elif role == "critic":
                value = LLMCritic(b).propose(
                    PreferenceSample.from_dict(p["sample"]),
                    Rubric.from_dict(p["rubric"]),
                    RubricJudgment.from_dict(p["judgment"]),
                    1,
                    conversation=[],
                    codec=NumberedIDCodec(),
                )
            elif role == "experience_updater":
                value = request(b, role, p, ExperienceUpdate.from_dict)
            else:
                # Run just this Checker; other two slots are deterministic no-call fixtures.
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
                value = next(
                    r
                    for r in LLMCheckers(bs).check(application_record(p["application"]))
                    if r.stage == stage
                )
            result["output"] = value.to_dict() if isinstance(value, Model) else value
            result["valid"] = not (
                isinstance(value, CheckerReport) and value.status == "failed"
            )
        except (LLMFormatError, LLMSampleError) as exc:
            result.update(valid=False, error=str(exc))
        finally:
            b.close()
            CURRENT.reset(token)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(run, r) for r in rows]), 1):
            result = f.result()
            if i % 20 == 0:
                print(
                    f'{i}/{len(rows)}; latest={result["key"]},valid={result["valid"]}',
                    flush=True,
                )
    results = [
        json.loads((OUT / "results" / (row["key"] + ".json")).read_text())
        for row in rows
    ]
    summary = {
        r: {
            "tasks": sum(x["role"] == r for x in results),
            "valid": sum(x["role"] == r and x["valid"] for x in results),
        }
        for r in sorted({r["role"] for r in rows})
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output)
    if args.prepare:
        prepare()
    else:
        execute(args.workers)
