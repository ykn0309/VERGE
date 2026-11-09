"""Replay the frozen Generator inputs against one isolated code snapshot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import sys
from threading import Lock
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("arm", choices=("baseline", "candidate"))
    args = parser.parse_args()
    root = args.root.resolve()
    code = root / f"{args.arm}-code"
    sys.path.insert(0, str(code))
    from verge.cli import build_backend
    from verge.config import load_config
    from verge.llm import configure_request_limit
    from verge.models import Experience
    from verge.modules import LLMGenerator
    from verge.prompts import PROMPT_VERSION

    out = root / args.arm
    out.mkdir(exist_ok=False)
    job_context = ContextVar("generator_job", default={})
    lock = Lock()

    def append(name, event):
        with lock:
            with (out / name).open("a") as stream:
                stream.write(
                    json.dumps(
                        {**job_context.get(), **event},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

    cfg = load_config(root / "config.yaml")
    configure_request_limit(8)
    b = build_backend(cfg["model"])
    b.trace_hook = lambda event: append("role_io.jsonl", event)
    b.metadata_hook = lambda event: append("llm_requests.jsonl", event)
    fixtures = json.loads((root / "fixtures.json").read_text())
    manifest = {
        "arm": args.arm,
        "prompt_version": PROMPT_VERSION,
        "model": b.model,
        "enable_thinking": b.enable_thinking,
        "input_sha256": hashlib.sha256(
            (root / "fixtures.json").read_bytes()
        ).hexdigest(),
        "code_sha256": {
            str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (code / "verge").glob("*.py")
        },
        "started": time.time(),
        "status": "running",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    def work(job):
        meta = {k: job[k] for k in ("sample_id", "task_id", "case", "path")}
        job_context.set(meta)
        start = time.monotonic()
        result = {**meta, "output": None, "error": None}
        try:
            es = [Experience.from_dict(e) for e in job["experiences"]]
            result["output"] = (
                LLMGenerator(b, **cfg["requests"]).generate(job["task"], es).to_dict()
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = time.monotonic() - start
        append("results.jsonl", result)
        return result

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(work, job) for job in fixtures]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(
                    f'{i}/{len(futures)} {result["case"]} error={result["error"]}',
                    flush=True,
                )
        manifest.update(status="completed", finished=time.time())
    finally:
        b.close()
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
