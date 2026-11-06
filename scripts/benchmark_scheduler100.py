"""Controlled scheduling benchmark with identical fake work, never a model-effect claim."""

import sys, time, json, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tempfile import TemporaryDirectory
from verge.pipeline import run_training_batch
from verge.scheduling import run_training_stream
from verge.models import PreferenceSample
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import make_fake_modules

if __name__ == "__main__":
    data = Path("artifacts/protocol100-flow-data/train20.jsonl")
    samples = [PreferenceSample.from_dict(json.loads(l)) for l in data.open()]
    rows = []
    for repeat in range(3):
        for mode, workers in [("batch", 3), ("stream", 3), ("stream", 5)]:
            indices = {s.sample_id: i for i, s in enumerate(samples)}
            state = {"active": 0, "peak": 0}
            lock = threading.Lock()

            def factory(s):
                m = make_fake_modules("initial_success")
                original = m.initial_generator.generate

                def gen(*args, **kwargs):
                    with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    try:
                        time.sleep(0.24 if indices[s.sample_id] % 3 == 0 else 0.04)
                        return original(*args, **kwargs)
                    finally:
                        with lock:
                            state["active"] -= 1

                m.initial_generator.generate = gen
                return m, lambda: None

            with TemporaryDirectory() as root:
                memory = ExperienceMemory(root)
                log = TrajectoryLog(root)
                start = time.monotonic()
                if mode == "batch":
                    for offset in range(0, len(samples), workers):
                        run_training_batch(
                            samples[offset : offset + workers],
                            factory,
                            memory,
                            log,
                            runner_options={"revision_enabled": False},
                        )
                else:
                    run_training_stream(
                        samples,
                        factory,
                        memory,
                        log,
                        workers=workers,
                        runner_options={"revision_enabled": False},
                    )
                elapsed = time.monotonic() - start
                assert len(list(log.all())) == len(samples)
                rows.append(
                    {
                        "mode": mode,
                        "workers": workers,
                        "repeat": repeat,
                        "tasks": len(samples),
                        "seconds": elapsed,
                        "peak_active": state["peak"],
                    }
                )
    out = Path("artifacts/protocol100-scheduling-benchmark")
    out.mkdir(exist_ok=False)
    (out / "results.json").write_text(
        json.dumps(
            {
                "scope": "Controlled fixed fake delay, same real20 tasks, no API/network cost; NOT observed cloud speedup",
                "results": rows,
            },
            indent=2,
        )
    )
    print(rows)
