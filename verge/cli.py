"""Run and inspect the history-based learning protocol."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from pathlib import Path
from threading import Lock
import argparse, hashlib, json, time
from verge.clustering import OnlineClustering
from verge.config import load_config
from verge.dataset import JsonlDataset
from verge.llm import (
    FakeLLMBackend,
    FakeEmbedding,
    OpenAICompatibleBackend,
    OpenAICompatibleEmbedding,
    configure_request_limit,
)
from verge.memory import ExperienceMemory, atomic_json
from verge.modules import LLMSelector, LLMGenerator, LLMJudge, LLMCritic, RoleModules
from verge.pipeline import PipelineRunner
from verge.prompts import PROMPT_VERSION
from verge.retrieval import CandidateQuota
from verge.scheduling import run_training_stream

CURRENT_SAMPLE = ContextVar("current_sample", default=None)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class JSONLogs:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = Lock()

    def append(self, name, event):
        with self.lock:
            with (self.root / name).open("a") as stream:
                stream.write(
                    json.dumps(
                        {"sample_id": CURRENT_SAMPLE.get(), **event},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )


def build_backend(spec, *, embedding=False):
    spec = dict(spec)
    kind = spec.pop("backend", "openai_compatible")
    if kind == "fake":
        return (
            FakeEmbedding()
            if embedding
            else FakeLLMBackend(spec.pop("model", "repairable"))
        )
    if kind != "openai_compatible":
        raise ValueError("unsupported backend")
    return (OpenAICompatibleEmbedding if embedding else OpenAICompatibleBackend)(**spec)


class Components:
    def __init__(self, config, logs=None):
        self.config = config
        self.backends = {}
        shared = build_backend(config["model"])
        for role in ("selector", "generator", "judge", "critic"):
            self.backends[role] = (
                build_backend({**config["model"], **config["roles"][role]})
                if role in config["roles"]
                else shared
            )
        self.embedding = build_backend(config["embedding"], embedding=True)
        for backend in {*self.backends.values(), self.embedding}:
            if logs:
                backend.trace_hook = lambda event: logs.append("role_io.jsonl", event)
                backend.metadata_hook = lambda event: logs.append(
                    "llm_requests.jsonl", event
                )

    def modules(self):
        options = self.config["requests"]
        return RoleModules(
            LLMSelector(
                self.backends["selector"], **self.config["selection"], **options
            ),
            LLMGenerator(self.backends["generator"], **options),
            LLMJudge(self.backends["judge"], **options),
            LLMCritic(
                self.backends["critic"],
                max_candidates=self.config["critic"]["max_candidates"],
                **options,
            ),
        )

    def runner(self, sample):
        CURRENT_SAMPLE.set(sample.sample_id)
        retrieval = self.config["retrieval"]
        return PipelineRunner(
            self.modules(),
            self.embedding,
            task_group_limit=retrieval["task_group_limit"],
            quota=CandidateQuota(retrieval["historical"], retrieval["semantic"]),
            critic_rounds=self.config["critic"]["max_rounds"],
        )

    def close(self):
        for backend in {*self.backends.values(), self.embedding}:
            backend.close()


def execution_protocol(config):
    return fingerprint(
        {key: value for key, value in config.items() if key not in ("run", "dataset")}
        | {"prompt": PROMPT_VERSION}
    )


def evaluate_records(components, samples, frozen, workers):
    def evaluate_one(sample):
        return (
            components.runner(sample).run_sample(sample, frozen, training=False).record
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(copy_context().run, evaluate_one, sample)
            for sample in samples
        ]
        return [future.result() for future in futures]


def evaluation_summary(records, checkpoint_hash):
    return {
        "tasks": len(records),
        "correct": sum(
            r.prediction_path is not None and r.prediction_path.correct is True
            for r in records
        ),
        "errors": sum(
            r.error is not None
            or r.prediction_path is None
            or r.prediction_path.error is not None
            for r in records
        ),
        "frozen_sha256": checkpoint_hash,
    }


def run(config_path, *, resume=False, stop_after_train=None):
    config = load_config(config_path)
    if stop_after_train is not None and (
        type(stop_after_train) is not int or stop_after_train < 1
    ):
        raise ValueError("stop_after_train must be positive")

    dataset = JsonlDataset(config["dataset"]["paths"])
    train = list(dataset.load("train"))
    tests = list(dataset.load("test")) if "test" in config["dataset"]["paths"] else []
    if {s.task_id for s in train} & {s.task_id for s in tests}:
        raise ValueError("train/test task overlap")
    protocol = execution_protocol(config)
    data_hashes = {
        name: digest(path) for name, path in config["dataset"]["paths"].items()
    }
    code_hash = fingerprint(
        {str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )
    root = Path(config["run"]["output_dir"])
    checkpoint = root / "checkpoint.json"
    manifest_path = root / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["code_hash"] != code_hash
            or manifest["data_hashes"] != data_hashes
            or manifest["protocol_id"] != protocol
            or manifest["checkpoint_sha256"] != digest(checkpoint)
        ):
            raise ValueError(
                "checkpoint/config/code digest mismatch; no automatic recovery"
            )
        if manifest["status"] not in ("paused", "failed", "running"):
            raise ValueError("run already completed")
        state = ExperienceMemory.load(checkpoint, protocol_id=protocol, read_only=False)
    else:
        if root.exists():
            raise ValueError("output directory already exists")
        root.mkdir(parents=True)
        state = ExperienceMemory(
            protocol, OnlineClustering(**config["clustering"]), path=checkpoint
        )
        state.save()
        atomic_json(root / "config.json", config)
        manifest = {
            "protocol_id": protocol,
            "prompt_version": PROMPT_VERSION,
            "data_hashes": data_hashes,
            "code_hash": code_hash,
            "status": "running",
            "started": time.time(),
        }
    logs = JSONLogs(root)
    configure_request_limit(config["run"]["max_inflight_requests"])

    def progress(record=None):
        manifest.update(
            train_completed=len(state.trajectories),
            experiences=len(state.experiences),
            checkpoint_sha256=digest(checkpoint),
        )
        atomic_json(manifest_path, manifest)

    progress()
    components = None
    try:
        components = Components(config, logs)
        target = train[:stop_after_train] if stop_after_train else train
        manifest["status"] = "running"
        progress()
        run_training_stream(
            target,
            state,
            components.runner,
            workers=config["run"]["train_workers"],
            on_commit=progress,
        )
        if len(state.trajectories) < len(train):
            manifest["status"] = "paused"
            progress()
            return manifest
        # The stream drains all source validation before this frozen checkpoint.
        manifest["status"] = "training_completed"
        progress()
        if tests:
            frozen = state.snapshot()
            before = digest(checkpoint)

            records = evaluate_records(
                components, tests, frozen, config["run"]["test_workers"]
            )
            for record in records:
                logs.append(
                    "test.jsonl",
                    {"sample_id": record.sample.sample_id, **record.to_dict()},
                )
            if digest(checkpoint) != before:
                raise ValueError("frozen pool changed during evaluation")
            manifest["test"] = evaluation_summary(records, before)
            manifest["status"] = "completed"
        manifest["finished"] = time.time()
        progress()
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        progress()
        raise
    finally:
        if components is not None:
            components.close()


def evaluate(config_path, checkpoint, input_path, output_dir):
    config = load_config(config_path)
    state = ExperienceMemory.load(
        checkpoint, protocol_id=execution_protocol(config), read_only=True
    )
    samples = list(JsonlDataset({"test": input_path}).load("test"))
    output = Path(output_dir)
    if output.exists():
        raise ValueError("evaluation output already exists")
    output.mkdir(parents=True)
    logs = JSONLogs(output)
    before = digest(checkpoint)
    configure_request_limit(config["run"]["max_inflight_requests"])
    components = Components(config, logs)
    manifest = {
        "status": "running",
        "mode": "frozen_evaluation",
        "protocol_id": state.protocol_id,
        "data_sha256": digest(input_path),
        "training_overlap": sum(s.task_id in state.trajectories for s in samples),
    }
    atomic_json(output / "manifest.json", manifest)
    try:
        records = evaluate_records(
            components, samples, state, config["run"]["test_workers"]
        )
        for record in records:
            logs.append(
                "test.jsonl", {"sample_id": record.sample.sample_id, **record.to_dict()}
            )
        if digest(checkpoint) != before:
            raise ValueError("frozen pool changed during evaluation")
        manifest.update(status="completed", test=evaluation_summary(records, before))
        atomic_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        components.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    command.add_argument("config")
    command.add_argument("--resume", action="store_true")
    command.add_argument("--stop-after-train", type=int)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("checkpoint")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("config")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument("input")
    evaluation.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "run":
        print(
            json.dumps(
                run(
                    args.config,
                    resume=args.resume,
                    stop_after_train=args.stop_after_train,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate(args.config, args.checkpoint, args.input, args.output),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        state = ExperienceMemory.load(args.checkpoint)
        print(
            json.dumps(
                {
                    "tasks": len(state.trajectories),
                    "experiences": len(state.experiences),
                    "feedback": len(state.ledger.records),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
