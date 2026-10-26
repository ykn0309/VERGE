"""Independent 20-task validation of the applicability-only Selector projection."""

from __future__ import annotations
import hashlib, json, shutil, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import Experience, task_identity
from verge.modules import LLMSelector
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMSampleError
from verge.prompts import PROMPT_VERSION


def execute():
    source = Path("artifacts/v2-review-71/inputs.json")
    out = Path("artifacts/v2-collection-73/selector-replay")
    if out.exists():
        raise ValueError("new output required")
    raw = source.read_bytes()
    rows = [r for r in json.loads(raw) if r["role"] == "selector"]
    assert (
        len(rows) == 20 and len({task_identity(r["input"]["task"]) for r in rows}) == 20
    )
    out.mkdir(parents=True)
    (out / "results").mkdir()
    (out / "inputs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model": "qwen3.5-flash",
        "enable_thinking": False,
        "include_quality": False,
        "prompt_version": PROMPT_VERSION,
        "tasks": 20,
        "independent": True,
        "outputs_feed_online": False,
        "conversion": "Production Selector strips all Q, counts and quality states; actual task and experience texts unchanged.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    shutil.copytree(
        "verge",
        out / "code_snapshot/verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, out / "code_snapshot/replay_collection_selector.py")
    lock = threading.Lock()

    def one(row):
        p = row["input"]
        b = OpenAICompatibleBackend(
            "qwen3.5-flash",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY_2",
            enable_thinking=False,
            timeout_seconds=120,
            max_retries=2,
        )

        def append(name, event):
            with lock:
                with (out / name).open("a") as f:
                    f.write(
                        json.dumps(
                            {"key": row["key"], "sample_id": row["sample_id"], **event},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        b.trace_hook = lambda e: append("role_io.jsonl", e)
        b.metadata_hook = lambda e: append("llm_requests.jsonl", e)
        result = {"key": row["key"], "sample_id": row["sample_id"]}
        try:
            chosen = LLMSelector(b, include_quality=False).select(
                p["task"], [Experience.from_dict(x) for x in p["experiences"]]
            )
            result.update(valid=True, selected=[x.experience_id for x in chosen])
        except (LLMFormatError, LLMSampleError) as e:
            result.update(valid=False, error=str(e))
        finally:
            b.close()
        (out / "results" / f'{row["key"]}.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, rows))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    for line in (out / "role_io.jsonl").read_text().splitlines():
        trace = json.loads(line)
        for e in trace["input"]["candidates"]:
            assert set(e) == {
                "experience_id",
                "trigger",
                "principle",
                "template",
                "parameters",
            }
    summary = {
        "tasks": 20,
        "valid": sum(r["valid"] for r in results),
        "private_statistics_absent": True,
        "source_unchanged": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    execute()
