"""V2 CLI: immutable runs, explicit protocol fingerprints and certified-prefix resume."""

from __future__ import annotations
import argparse, hashlib, json, subprocess, shutil, sys, threading
from contextvars import ContextVar
from pathlib import Path
from verge.config import load_config
from verge.dataset import JsonlDataset
from verge.llm import (
    FakeLLMBackend,
    OpenAICompatibleBackend,
    OpenAICompatibleEmbedding,
    APPLICATION_CONTEXT,
)
from verge.modules import make_llm_modules, HybridRetriever
from verge.calculator import ProbabilityModel
from verge.memory import ExperienceMemory, TrajectoryLog, atomic_json
from verge.pipeline import PipelineRunner, run_training_batch
from verge.scheduling import run_training_stream
from verge.prompts import PROMPT_VERSION

METHOD = "semistructured_bayesian_v2"
FORMAT = 17
CURRENT_SAMPLE = ContextVar("v2_sample", default={})
ROLES = (
    "selector",
    "initial_generator",
    "judge",
    "critic",
    "critic_repair",
    "critic_assess",
    "critic_extract",
    "critic_rewrite",
    "experience_checker_batch",
    "critic_rewrite_batch",
    "experience_checker",
    "experience_updater",
    "selector_checker",
    "generator_checker",
    "judge_checker",
)


class ConfigError(ValueError):
    pass


def _completed_judgment(record):
    """Latest completed judgment, including Critic reviews; never infer from admission."""
    for attempt in reversed(record.proposal_attempts):
        judgment = attempt.final_judgment or attempt.judgment
        if judgment is not None:
            return judgment
    return record.baseline_judgment or record.initial_judgment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_version():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def backend(spec):
    if set(spec) - {
        "backend",
        "model",
        "base_url",
        "api_key_env",
        "timeout_seconds",
        "max_retries",
        "enable_thinking",
    }:
        raise ConfigError("unknown backend option")
    if spec["backend"] == "fake":
        return FakeLLMBackend(spec["model"])
    if spec["backend"] != "openai_compatible":
        raise ConfigError("unknown backend")
    return OpenAICompatibleBackend(**{k: v for k, v in spec.items() if k != "backend"})


def components(config):
    specs = config["models"]
    roles = specs.get("roles", {})
    if set(roles) - set(ROLES):
        raise ConfigError("unknown model role")
    b = backend(specs["default"])
    bs = {k: backend(v) for k, v in roles.items()}
    app = config["modules"]["application"]["config"]
    retrieval = config["modules"]["retrieval"]
    r = None
    if retrieval["implementation"] == "hybrid":
        spec = specs["default"]
        opts = retrieval["config"]
        emb = OpenAICompatibleEmbedding(
            opts.get("embedding_model", "text-embedding-v4"),
            base_url=spec["base_url"],
            api_key_env=spec["api_key_env"],
            dimensions=opts.get("dimensions", 1024),
            timeout_seconds=spec.get("timeout_seconds", 60),
        )
        r = HybridRetriever(emb, rrf_k=opts.get("rrf_k", 60))
    modules = make_llm_modules(
        b,
        role_backends=bs,
        retriever=r,
        max_selected=config["modules"]["selection"]["config"].get("max_selected", 5),
        max_candidates=app.get("max_candidates", 1),
        retries=app.get("invalid_output_retries", 2),
        input_bytes=app.get("input_bytes", 100000),
        selector_include_quality=config["run"].get("mode") != "collection",
    )
    return modules


def run(config_path, *, resume_from=None, stop_after_train=None):
    if stop_after_train is not None and (
        type(stop_after_train) is not int or stop_after_train < 1
    ):
        raise ConfigError("stop_after_train must be a positive integer")
    config = load_config(config_path)
    from verge.llm import configure_request_limit

    configure_request_limit(config["run"].get("max_inflight_requests", 16))
    out = Path(config["run"]["output_dir"]).resolve()
    if out.exists():
        raise ConfigError("output directory already exists; use a new directory")
    data = JsonlDataset(**config["dataset"]["config"])
    train = list(data.load("train"))
    test = list(data.load("test")) if "test" in data.paths else []
    if {s.sample_id for s in train} & {s.sample_id for s in test} or {
        s.task_id for s in train
    } & {s.task_id for s in test}:
        raise ConfigError("train/test overlap")
    if not train:
        raise ConfigError("training split must not be empty")
    if "test" in data.paths and not test:
        raise ConfigError("configured test split must not be empty")
    parameters_path = config["modules"]["reliability"]["config"].get("parameter_file")
    parameters = ProbabilityModel.load(parameters_path) if parameters_path else None
    fingerprint = {**config}
    fingerprint = json.loads(json.dumps(fingerprint))
    fingerprint["run"].pop("output_dir")
    fingerprint["run"].pop("stop_after_train", None)
    param_digest = digest(parameters_path) if parameters_path else None
    protocol = {
        "method": METHOD,
        "format_version": FORMAT,
        "code_version": code_version(),
        "prompt_version": PROMPT_VERSION,
        "dataset_sha256": {k: digest(v) for k, v in data.paths.items()},
        "parameters_sha256": param_digest,
        "config": fingerprint,
    }
    store_rel = Path(config["store"]["config"].get("root_dir", "artifacts"))
    if store_rel.is_absolute() or ".." in store_rel.parts:
        raise ConfigError("store root must be inside run directory")
    source = None
    old_records = []
    if resume_from:
        source = Path(resume_from).resolve()
        manifest = json.loads((source / "run_manifest.json").read_text())
        if (
            manifest.get("protocol") != protocol
            or manifest.get("format_version") != FORMAT
        ):
            raise ConfigError(
                "resume requires identical V2 code/prompt/data/config/parameters"
            )
        if manifest.get("state") not in ("completed", "training_completed", "paused"):
            raise ConfigError(
                "uncertified checkpoint; cross-file recovery is not automatic"
            )
        old_records = list(TrajectoryLog(source / store_rel).all())
        training_records = [r for r in old_records if r.split == "train"]
        ids = [r.sample_id for r in training_records]
        if (
            len(set(ids)) != len(ids)
            or set(ids) - {s.sample_id for s in train}
            or any(r.split != "train" for r in old_records)
        ):
            raise ConfigError(
                "resume requires unique train-only records from the same dataset"
            )
        if manifest.get("memory_sha256") != digest(
            source / store_rel / "experience_memory.json"
        ) or manifest.get("trajectory_sha256") != digest(
            source / store_rel / "trajectories.jsonl"
        ):
            raise ConfigError("checkpoint memory/trajectory fingerprint mismatch")
    modules = components(config)
    out.mkdir(parents=True)
    store = out / store_rel
    store.mkdir()
    if source:
        for name in (
            "experience_memory.json",
            "trajectories.jsonl",
            "role_io.jsonl",
            "llm_requests.jsonl",
            "revision_jobs.json",
            "pending_sample_commit.json",
        ):
            p = source / store_rel / name
            if p.exists():
                shutil.copy2(p, store / name)
    if source and (source / store_rel / "revision_results").exists():
        shutil.copytree(
            source / store_rel / "revision_results", store / "revision_results"
        )
    # Snapshot executed code/config; never credentials or environment values.
    shutil.copytree(
        Path(__file__).parent,
        out / ".source" / "verge",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    manifest = {
        "format_version": FORMAT,
        "method": METHOD,
        "protocol": protocol,
        "config": config,
        "state": "running",
        "run_mode": config["run"].get("mode", "learning"),
        "evaluation_mode": "train_test" if test else "training_only",
    }
    if parameters:
        atomic_json(out / "probability_model.json", parameters.to_dict())
    atomic_json(out / "run_manifest.json", manifest)
    lock = threading.Lock()

    def append(name, event):
        event = {
            **CURRENT_SAMPLE.get(),
            **APPLICATION_CONTEXT.get(),
            "cost": None,
            "cost_status": "pricing_not_configured",
            **event,
        }
        with lock:
            with (store / name).open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    backends = {
        id(x.backend): x.backend
        for x in (
            modules.selector,
            modules.initial_generator,
            modules.judge,
            modules.critic,
            modules.experience_updater,
            modules.experience_checker,
        )
    }
    backends.update({id(b): b for b in modules.checkers.backends.values()})
    for b in backends.values():
        b.trace_hook = lambda e: append("role_io.jsonl", e)
        b.metadata_hook = lambda e: append("llm_requests.jsonl", e)
    if isinstance(modules.retriever, HybridRetriever):
        modules.retriever.backend.metadata_hook = lambda e: append(
            "llm_requests.jsonl", e
        )
        modules.retriever.backend.trace_hook = lambda e: append("role_io.jsonl", e)
    atomic_json(out / "run.lock", {"pid": __import__("os").getpid()})
    memory = ExperienceMemory(store, parameters=parameters)
    log = TrajectoryLog(store)
    opts = config["modules"]["application"]["config"]
    kwargs = {
        k: opts[name]
        for k, name in [
            ("max_rounds", "max_rounds"),
            ("interval", "check_interval"),
            ("window", "error_window"),
            ("error_threshold", "error_threshold"),
            ("q_threshold", "q_threshold"),
            ("min_replay", "min_replay"),
            ("revision_enabled", "revision_enabled"),
            ("margin_threshold", "margin_threshold"),
        ]
        if name in opts
    }
    limit = (
        stop_after_train
        if stop_after_train is not None
        else config["run"].get("stop_after_train")
    )
    n = len(old_records)
    records = list(old_records)
    try:
        runner = PipelineRunner(modules, memory, log, **kwargs)
        workers = config["run"].get("train_workers", 1)

        def batch_factory(sample):
            token = CURRENT_SAMPLE.set(
                {"sample_id": sample.sample_id, "split": "train"}
            )
            m = components(config)
            m.judge.share_cache(modules.judge)
            bs = {
                id(x.backend): x.backend
                for x in (
                    m.selector,
                    m.initial_generator,
                    m.judge,
                    m.critic,
                    m.experience_updater,
                    m.experience_checker,
                )
            }
            bs.update({id(b): b for b in m.checkers.backends.values()})
            if isinstance(m.retriever, HybridRetriever):
                bs[id(m.retriever.backend)] = m.retriever.backend
                with lock:
                    m.retriever.share_cache(modules.retriever)
            for b in bs.values():
                b.trace_hook = lambda e: append("role_io.jsonl", e)
                b.metadata_hook = lambda e: append("llm_requests.jsonl", e)

            def cleanup():
                try:
                    if isinstance(m.retriever, HybridRetriever):
                        with lock:
                            modules.retriever.cache.update(m.retriever.cache)
                    for b in bs.values():
                        if hasattr(b, "close"):
                            b.close()
                finally:
                    CURRENT_SAMPLE.reset(token)

            return m, cleanup

        completed_ids = {r.sample_id for r in records}
        remaining = [
            s
            for s in (train[:limit] if limit is not None else train)
            if s.sample_id not in completed_ids
        ]
        import signal

        stopping = threading.Event()
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.signal(signal.SIGINT, lambda *_: stopping.set())
        try:
            new_records = run_training_stream(
                remaining,
                batch_factory,
                memory,
                log,
                workers=workers,
                revision_workers=config["run"].get("revision_workers", 1),
                runner_options=kwargs,
                stop_event=stopping,
                on_complete=lambda rec, i: print(
                    f"train {i}/{len(train)} {rec.state}", flush=True
                ),
            )
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
        records.extend(new_records)
        paused = len(records) < len(train)
        if not paused and test:
            frozen = ExperienceMemory(store, read_only=True, parameters=parameters)
            runner = PipelineRunner(modules, frozen, log, training=False)
            for sample in test:
                token = CURRENT_SAMPLE.set(
                    {"sample_id": sample.sample_id, "split": "test"}
                )
                try:
                    records.append(runner.run_sample(sample))
                finally:
                    CURRENT_SAMPLE.reset(token)
        manifest["state"] = (
            "paused" if paused else "completed" if test else "training_completed"
        )
        manifest["evaluation_mode"] = "train_test" if test else "training_only"
        manifest["memory_sha256"] = digest(memory.path)
        manifest["trajectory_sha256"] = digest(log.path) if log.path.exists() else None
        atomic_json(out / "run_manifest.json", manifest)
        summary = {
            "train": sum(r.split == "train" for r in records),
            "test": sum(r.split == "test" for r in records),
            "initial_correct": sum(
                bool(r.initial_judgment and r.initial_judgment.success) for r in records
            ),
            "final_correct": sum(
                bool(_completed_judgment(r) and _completed_judgment(r).success)
                for r in records
            ),
            "sample_errors": sum(r.sample_error is not None for r in records),
            "active_experiences": len(memory.all(active_only=True)),
        }
        atomic_json(out / "summary.json", summary)
        return summary
    finally:
        (out / "run.lock").unlink(missing_ok=True)
        for b in backends.values():
            if hasattr(b, "close"):
                b.close()
        if isinstance(modules.retriever, HybridRetriever):
            modules.retriever.backend.close()


def recover_run(run_dir, output_dir):
    """Certify a new copy from durable completed work only. Never repeats model calls."""
    from verge.scheduling import recover_sample_commit, commit_sample
    from verge.models import Experience, RevisionRecord, TrajectoryRecord
    import os

    source, target = Path(run_dir).resolve(), Path(output_dir).resolve()
    if target.exists() or source == target or source in target.parents:
        raise ConfigError("recovery needs a new output outside source")
    manifest = json.loads((source / "run_manifest.json").read_text())
    protocol = manifest.get("protocol", {})
    if (
        manifest.get("format_version") != FORMAT
        or protocol.get("code_version") != code_version()
        or protocol.get("prompt_version") != PROMPT_VERSION
    ):
        raise ConfigError("recovery requires identical current code and prompt")
    lock = source / "run.lock"
    if lock.exists():
        pid = json.loads(lock.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ConfigError("source process is alive; recovery refused")
    relative = Path(manifest["config"]["store"]["config"].get("root_dir", "artifacts"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError("invalid recovery store")
    store = source / relative
    # Check before copying. Unfinished calls cannot be safely reconstructed or replayed.
    for kind in ("sample", "revision"):
        jobs_path = store / (kind + "_jobs.json")
        jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
        for job in jobs:
            if job["state"] != "running":
                continue
            name = (
                __import__("hashlib").sha256(job["sample_id"].encode()).hexdigest()
                if kind == "sample"
                else job["job_id"]
            )
            if not (store / (kind + "_results") / (name + ".json")).exists():
                raise ConfigError(
                    kind
                    + " has unfinished calls without a durable result; refusing repeat calls"
                )
    parameter_path = source / "probability_model.json"
    parameters = (
        ProbabilityModel.load(parameter_path) if parameter_path.exists() else None
    )
    if parameters and digest(parameter_path) != protocol.get("parameters_sha256"):
        raise ConfigError("parameter snapshot differs")
    shutil.copytree(source, target)
    (target / "run.lock").unlink(missing_ok=True)
    root = target / relative
    memory = ExperienceMemory(root, parameters=parameters)
    log = TrajectoryLog(root)
    recover_sample_commit(memory, log)
    sample_jobs = (
        json.loads((root / "sample_jobs.json").read_text())
        if (root / "sample_jobs.json").exists()
        else []
    )
    for job in sample_jobs:
        if job["state"] != "running":
            continue
        result = json.loads(
            (
                root
                / "sample_results"
                / (
                    __import__("hashlib").sha256(job["sample_id"].encode()).hexdigest()
                    + ".json"
                )
            ).read_text()
        )
        rec = TrajectoryRecord.from_dict(result["record"])
        if log.get(rec.sample_id) is None:
            rec.completion_sequence = sum(1 for _ in log.all()) + 1
            commit_sample(
                memory, log, rec, [Experience.from_dict(e) for e in result["added"]]
            )
        job["state"] = "completed"
    atomic_json(root / "sample_jobs.json", sample_jobs)
    jobs = (
        json.loads((root / "revision_jobs.json").read_text())
        if (root / "revision_jobs.json").exists()
        else []
    )
    for job in jobs:
        if job["state"] != "running":
            continue
        event = RevisionRecord.from_dict(
            json.loads(
                (root / "revision_results" / (job["job_id"] + ".json")).read_text()
            )
        )
        if event.event_id not in {r.event_id for r in memory.revisions}:
            if memory.get(event.parent_id).status == "active":
                memory.finish_revision(event)
            else:
                job["state"] = "stale_parent"
        if job["state"] == "running":
            job["state"] = "completed"
            job["event_id"] = event.event_id
    atomic_json(root / "revision_jobs.json", jobs)
    records = list(log.all())
    if any(r.split != "train" for r in records) or len(
        {r.sample_id for r in records}
    ) != len(records):
        raise ConfigError("recovery needs unique train-only completed records")
    manifest.update(
        state="paused",
        memory_sha256=digest(memory.path),
        trajectory_sha256=digest(log.path) if log.path.exists() else None,
        recovered_from=str(source),
    )
    atomic_json(target / "run_manifest.json", manifest)
    return {
        "state": "paused",
        "train": len(records),
        "output_dir": str(target),
        "model_calls": 0,
    }


def reestimate_run(run_dir, parameter_file, output_dir):
    """Rebuild only quality state in a new certified run copy, without model calls."""
    source, target = Path(run_dir).resolve(), Path(output_dir).resolve()
    if target.exists() or source == target or source in target.parents:
        raise ConfigError("reestimate requires a new output outside the source run")
    manifest = json.loads((source / "run_manifest.json").read_text())
    if manifest.get("format_version") != FORMAT or manifest.get("state") not in (
        "paused",
        "completed",
        "training_completed",
    ):
        raise ConfigError("reestimate requires a certified current-protocol run")
    store = Path(manifest["config"]["store"]["config"].get("root_dir", "artifacts"))
    if store.is_absolute() or ".." in store.parts:
        raise ConfigError("invalid store path")
    if manifest.get("memory_sha256") != digest(
        source / store / "experience_memory.json"
    ) or manifest.get("trajectory_sha256") != digest(
        source / store / "trajectories.jsonl"
    ):
        raise ConfigError("checkpoint fingerprint mismatch")
    parameters = ProbabilityModel.load(parameter_file)
    # Parse and validate before creating the output; source remains read-only.
    memory = ExperienceMemory(source / store, read_only=True)
    shutil.copytree(source, target)
    memory = ExperienceMemory(
        target / store, parameters=parameters, allow_reestimate=True
    )
    memory.reestimate(parameters)
    atomic_json(target / "probability_model.json", parameters.to_dict())
    manifest["config"]["run"]["output_dir"] = str(target)
    manifest["config"]["modules"]["reliability"]["config"]["parameter_file"] = str(
        target / "probability_model.json"
    )
    manifest["protocol"]["config"]["modules"]["reliability"]["config"][
        "parameter_file"
    ] = str(target / "probability_model.json")
    manifest["protocol"]["parameters_sha256"] = digest(
        target / "probability_model.json"
    )
    manifest["protocol"]["code_version"] = code_version()
    manifest["protocol"]["prompt_version"] = PROMPT_VERSION
    manifest["memory_sha256"] = digest(memory.path)
    manifest["reestimated_from"] = {
        "run": str(source),
        "memory_sha256": digest(source / store / "experience_memory.json"),
    }
    # Evaluation scores were produced with the old table; retain their original provenance.
    manifest["evaluation_parameters_sha256"] = manifest.get(
        "evaluation_parameters_sha256",
        json.loads((source / "run_manifest.json").read_text())["protocol"].get(
            "parameters_sha256"
        ),
    )
    snapshot = target / ".source" / "verge"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(
        Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns("__pycache__")
    )
    atomic_json(target / "run_manifest.json", manifest)
    return {
        "output_dir": str(target),
        "parameters": parameters.fingerprint,
        "events": len(memory.parameter_events),
    }


def inspect_sample(run_dir, sample_id):
    root = Path(run_dir)
    m = json.loads((root / "run_manifest.json").read_text())
    return (
        TrajectoryLog(
            root / m["config"]["store"]["config"].get("root_dir", "artifacts")
        )
        .get(sample_id)
        .to_dict()
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--resume-from")
    r.add_argument("--stop-after-train", type=int)
    i = sub.add_parser("inspect")
    i.add_argument("--run-dir", required=True)
    i.add_argument("--sample-id", required=True)
    e = sub.add_parser("reestimate")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--parameter-file", required=True)
    e.add_argument("--output-dir", required=True)
    recover = sub.add_parser("recover")
    recover.add_argument("--run-dir", required=True)
    recover.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "recover":
            print(
                json.dumps(
                    recover_run(args.run_dir, args.output_dir),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "reestimate":
            print(
                json.dumps(
                    reestimate_run(args.run_dir, args.parameter_file, args.output_dir),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        result = (
            run(
                args.config,
                resume_from=args.resume_from,
                stop_after_train=args.stop_after_train,
            )
            if args.command == "run"
            else inspect_sample(args.run_dir, args.sample_id)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
