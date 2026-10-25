"""Plan71: isolated role contract replays; never modifies the frozen source corpus."""

from __future__ import annotations
import json, hashlib, sys, threading, shutil
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import ApplicationRecord, Experience, Model
from verge.modules import LLMCheckers, LLMExperienceUpdater, LLMSelector
from verge.prompts import PROMPT_VERSION
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMSampleError
from verge.calculator import application

SOURCE = Path("artifacts/v2-module-replay/inputs.json")
OUT = Path("artifacts/v2-review-71")
CURRENT = ContextVar("case", default={})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise ValueError("use a new output directory")
    old = json.loads(SOURCE.read_text())
    rows = []
    apps = []
    for row in old:
        if row["role"] == "generator_checker":
            raw = deepcopy(row["input"]["application"])
            cited = any(
                raw["version_id"] in c["source_experiences"]
                for c in raw["rubric"]["criteria"]
            )
            raw.update(recalled=cited, selected=cited, cited=cited)
            # Offline conversion: old positive/reverse balance becomes the new tie.
            if raw["feedback"] == "cancellation":
                raw["feedback"] = "tie"
            original = ApplicationRecord.from_dict(raw)
            e = Experience(
                original.version_id, original.method, "historical-source-unavailable"
            )
            a = application(
                original.sample, e, original.rubric, original.judgment, original.phase
            )
            # Retain raw actual reports and round if present; no new Checker result is invented.
            a.reports = original.reports
            a.round_number = original.round_number
            apps.append(a)
    for role in ("selector", "selector_checker", "generator_checker", "judge_checker"):
        for row in [r for r in old if r["role"] == role]:
            data = row["input"]
            if role.endswith("_checker"):
                a = next(a for a in apps if a.sample.sample_id == row["sample_id"])
                data = {"application": a.to_dict()}
            rows.append(
                {
                    **row,
                    "input": data,
                    "conversion": row["conversion"]
                    + " Plan72: old cancellation converted to net-direction tie. Plan71: populate actual citation state from preserved rubric; new audit fields default to unavailable rather than inventing historical evidence.",
                }
            )
    for i, a in enumerate(apps, 1):
        related = [a] + [
            b
            for b in apps
            if b.version_id == a.version_id and b.sample.task_id != a.sample.task_id
        ]
        related = related[:6]
        e = Experience(a.version_id, a.method, "historical-source-unavailable")
        kinds = {
            b.sample.sample_id: (
                "problem"
                if b.feedback == "reverse"
                else "positive" if b.feedback == "positive" else "recent"
            )
            for b in related
        }
        rows.append(
            {
                "key": f"experience_updater-{i:02}",
                "role": "experience_updater",
                "sample_id": a.sample.sample_id,
                "input": {
                    "experience": e.to_dict(),
                    "cases": [b.to_dict() for b in related],
                    "reasons": ["historical_review"],
                    "case_kinds": kinds,
                },
                "conversion": "Plan69 actual applications converted to V2 semistructured methods as in plan70; full raw task/answers/weights/Boolean judgments preserved. Group at most6 same-version cases. Case kinds derived from actual feedback; no historical Checker/Q/trigger inferred. This is contract replay, not a reproduction of an online revision trigger.",
            }
        )
    for role in {r["role"] for r in rows}:
        group = [r for r in rows if r["role"] == role]
        assert len(group) == 20
        tasks = [
            (
                r["input"]["task"]
                if role == "selector"
                else (
                    r["input"]["application"]["sample"]["task"]
                    if role.endswith("_checker")
                    else r["input"]["cases"][0]["sample"]["task"]
                )
            )
            for r in group
        ]
        assert len({" ".join(t.split()) for t in tasks}) == 20
    OUT.mkdir(parents=True)
    (OUT / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    sources = {str(SOURCE): sha(SOURCE)}
    for p in Path("artifacts/v2-module-replay/source").glob("*"):
        if p.is_file():
            sources[str(p)] = sha(p)
    manifest = {
        "source_sha256": sources,
        "input_sha256": sha(OUT / "inputs.json"),
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "prompt_version": PROMPT_VERSION,
        "roles": 5,
        "tasks_per_role": 20,
        "independent": True,
        "source_conversion": "Explicit per row. Updater uses complete task-deduplicated plan69 application inputs, not the older plan67 inputs missing program evidence.",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    shutil.copytree(
        "verge",
        OUT / "code_snapshot/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, OUT / "code_snapshot/replay_v2_review.py")
    return rows


def execute(rows):
    lock = threading.Lock()
    (OUT / "results").mkdir(exist_ok=True)

    def append(name, event):
        with lock:
            with (OUT / name).open("a") as f:
                f.write(
                    json.dumps({**CURRENT.get(), **event}, ensure_ascii=False) + "\n"
                )

    def one(row):
        token = CURRENT.set({"key": row["key"], "sample_id": row["sample_id"]})
        b = OpenAICompatibleBackend(
            "qwen3.5-flash",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY_2",
            enable_thinking=False,
            timeout_seconds=120,
            max_retries=2,
        )
        b.trace_hook = lambda e: append("role_io.jsonl", e)
        b.metadata_hook = lambda e: append("llm_requests.jsonl", e)
        role = row["role"]
        p = row["input"]
        result = {k: row[k] for k in ("key", "role", "sample_id")}
        try:
            if role == "selector":
                v = LLMSelector(b).select(
                    p["task"], [Experience.from_dict(x) for x in p["experiences"]]
                )
                output = [e.experience_id for e in v]
            elif role == "experience_updater":
                output = LLMExperienceUpdater(b).update(
                    Experience.from_dict(p["experience"]),
                    [ApplicationRecord.from_dict(x) for x in p["cases"]],
                    p["reasons"],
                    case_kinds=p["case_kinds"],
                )
            else:

                class NoCall:
                    def generate(self, *args, **kw):
                        return {
                            "status": "unknown",
                            "evidence": "not under test",
                            "details": [],
                        }

                stage = role.removesuffix("_checker")
                bs = {s: NoCall() for s in ("selector", "generator", "judge")}
                bs[stage] = b
                output = next(
                    r
                    for r in LLMCheckers(bs).check(
                        ApplicationRecord.from_dict(p["application"])
                    )
                    if r.stage == stage
                )
            result.update(
                valid=getattr(output, "status", None) != "failed",
                output=output.to_dict() if isinstance(output, Model) else output,
            )
        except (LLMFormatError, LLMSampleError) as e:
            result.update(valid=False, error=str(e))
        finally:
            b.close()
            CURRENT.reset(token)
        (OUT / "results" / f'{row["key"]}.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one, r) for r in rows]):
            result = f.result()
            results.append(result)
            if len(results) % 20 == 0:
                print(f"{len(results)}/{len(rows)}", flush=True)
    summary = {
        role: {
            "tasks": sum(r["role"] == role for r in results),
            "valid": sum(r["role"] == role and r["valid"] for r in results),
        }
        for role in sorted({r["role"] for r in rows})
    }
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert all(sha(Path(p)) == h for p, h in manifest["source_sha256"].items())
    assert sha(OUT / "inputs.json") == manifest["input_sha256"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    execute(prepare())
