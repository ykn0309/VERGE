"""命令行入口：按配置运行 Fake/真实 LLM 实验，并按样本检查轨迹。"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Lock, local
from time import monotonic
from typing import Any, Sequence
from urllib.parse import urlsplit

from verge.config import load_config
from verge.dataset import CHAT_HARD_SUBSETS, JsonlDataset, prepare_rewardbench_chat_hard
from verge.llm import FakeLLMBackend, LLMBackend, OpenAICompatibleBackend, OpenAICompatibleEmbedding
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import VectorRetriever, make_llm_modules
from verge.pipeline import PipelineRunner
from verge.prompts import PROMPT_VERSION


_ROLES = (
    "selector", "initial_generator", "judge", "critic",
    "reflector", "admission_checker", "fusion_judge",
)
_FAKE_SCENARIOS = {"initial_success", "attempt_1_success", "attempt_2_success", "final_failure"}
_IMPLEMENTATIONS = {
    "retrieval": "fake", "selection": "llm", "revision": "critic_atomic_same_base",
    "revision_credit": "operation_bound", "reliability": "beta",
}


class ConfigError(ValueError):
    """配置或组件组合无效；CLI 使用退出码 2。"""


def _only_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path}: unsupported fields: {', '.join(sorted(unknown))}")


def _backend(spec: dict[str, Any], path: str) -> LLMBackend:
    if spec["backend"] == "fake":
        _only_keys(spec, {"backend", "model"}, path)
        if spec["model"] not in _FAKE_SCENARIOS:
            raise ValueError(f"{path}.model: unsupported fake scenario")
        return FakeLLMBackend(spec["model"])
    if spec["backend"] == "openai_compatible":
        _only_keys(spec, {"backend", "model", "base_url", "api_key_env",
                          "timeout_seconds", "max_retries", "enable_thinking"}, path)
        options = {key: spec[key] for key in ("base_url", "api_key_env", "timeout_seconds",
                                             "max_retries", "enable_thinking")
                   if key in spec}
        try:
            return OpenAICompatibleBackend(spec["model"], **options)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    raise ValueError(f"{path}.backend: supported backends are fake and openai_compatible")


def _components(config: dict[str, Any]) -> tuple[JsonlDataset, Any, Path, Path, int]:
    """运行前校验所有当前可装配的组件，不把未实现的配置静默降级。"""
    _only_keys(config, {"run", "dataset", "models", "modules", "store"}, "config")
    _only_keys(config["run"], {"seed", "output_dir", "test_workers"}, "run")
    workers = config["run"].get("test_workers", 4)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("run.test_workers: expected an integer in [1, 16]")
    dataset_config = config["dataset"]
    _only_keys(dataset_config, {"implementation", "config"}, "dataset")
    if dataset_config["implementation"] != "jsonl":
        raise ValueError("dataset.implementation: V1 supports only jsonl")
    options = dataset_config["config"]
    if set(options) - {"paths", "field_map", "prepare"} or "paths" not in options:
        raise ValueError("dataset.config: expected paths, optional field_map and prepare")
    if "prepare" in options:
        _check_preparation_paths(options)
    dataset = JsonlDataset(options["paths"], options.get("field_map"))
    if "train" not in dataset.paths:
        raise ValueError("dataset.config.paths.train: required for run")
    if set(dataset.paths) - {"train", "test"}:
        raise ValueError("dataset.config.paths: V1 supports only train and optional test")
    for split, path in dataset.paths.items():
        if not path.is_file():
            raise ValueError(f"dataset.config.paths.{split}: file not found: {path}")

    models = config["models"]
    _only_keys(models, {"default", "roles"}, "models")
    if set(models.get("roles", {})) - set(_ROLES):
        raise ValueError("models.roles: unknown LLM role")

    configured = config["modules"]
    if set(configured) != set(_IMPLEMENTATIONS):
        raise ValueError(f"modules: expected exactly {', '.join(_IMPLEMENTATIONS)}")
    for name, implementation in _IMPLEMENTATIONS.items():
        _only_keys(configured[name], {"implementation", "config"}, f"modules.{name}")
        if (configured[name]["implementation"] != implementation
                and not (name == "retrieval" and configured[name]["implementation"] == "vector")):
            raise ValueError(f"modules.{name}.implementation: V1 supports only {implementation}"
                             + (" or vector" if name == "retrieval" else ""))
    retrieval = configured["retrieval"]["config"]
    vector_retrieval = configured["retrieval"]["implementation"] == "vector"
    retrieval_keys = {"top_l", "embedding_model", "dimensions"} if vector_retrieval else {"top_l"}
    if vector_retrieval:
        if models["default"]["backend"] != "openai_compatible":
            raise ValueError("modules.retrieval: vector requires an openai_compatible default model")
        embedding_url = models["default"].get("base_url")
        parsed = urlsplit(embedding_url) if isinstance(embedding_url, str) else None
        if (parsed is None or parsed.hostname not in {
                "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
                "dashscope-us.aliyuncs.com",
            } or parsed.path.rstrip("/") != "/compatible-mode/v1"):
            raise ValueError("modules.retrieval: vector requires a DashScope compatible-mode/v1 base_url")
        if retrieval.get("embedding_model") != "text-embedding-v4":
            raise ValueError("modules.retrieval.config.embedding_model: expected text-embedding-v4")
        if (type(retrieval.get("dimensions", 1024)) is not int
                or retrieval.get("dimensions", 1024) not in (64, 128, 256, 512, 768, 1024, 1536, 2048)):
            raise ValueError("modules.retrieval.config.dimensions: unsupported text-embedding-v4 dimension")
    selection = configured["selection"]["config"]
    revision = configured["revision"]["config"]
    credit = configured["revision_credit"]["config"]
    if (set(retrieval) - retrieval_keys or set(selection) - {"max_selected", "exploration_slots"}
            or set(revision) - {"max_attempts", "invalid_output_retries"} or set(credit) - {"max_budget"}
            or configured["reliability"]["config"]):
        raise ValueError("modules: unsupported V1 component options")
    if selection.get("exploration_slots", 0) != 0:
        raise ValueError("modules.selection.config.exploration_slots: V1 selector does not explore")
    retries = revision.get("invalid_output_retries", 2)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError("modules.revision.config.invalid_output_retries: expected integer in [0, 3]")
    if credit.get("max_budget", 1.0) != 1.0:
        raise ValueError("modules.revision_credit.config.max_budget: V1 requires 1.0")

    store = config["store"]
    _only_keys(store, {"implementation", "config"}, "store")
    if store["implementation"] != "local" or set(store["config"]) - {"root_dir"}:
        raise ValueError("store: V1 supports only local with root_dir")
    relative_store = Path(store["config"].get("root_dir", "artifacts"))
    if relative_store.is_absolute() or ".." in relative_store.parts or relative_store == Path("."):
        raise ValueError("store.config.root_dir: must be a relative subdirectory of run.output_dir")
    run_dir = Path(config["run"]["output_dir"])
    if run_dir.exists():
        raise ValueError(f"run.output_dir: already exists: {run_dir}; use a fresh directory")

    created: list[OpenAICompatibleBackend | OpenAICompatibleEmbedding] = []
    try:
        backend = _backend(models["default"], "models.default")
        if isinstance(backend, OpenAICompatibleBackend):
            created.append(backend)
        retriever = None
        if vector_retrieval:
            default = models["default"]
            embedder = OpenAICompatibleEmbedding(
                retrieval["embedding_model"],
                base_url=default.get("base_url", "https://api.openai.com/v1"),
                api_key_env=default.get("api_key_env", "OPENAI_API_KEY"),
                dimensions=retrieval.get("dimensions", 1024),
                timeout_seconds=default.get("timeout_seconds", 30),
                max_retries=default.get("max_retries", 2),
            )
            created.append(embedder)
            retriever = VectorRetriever(embedder, retrieval.get("top_l", 20))
        modules = make_llm_modules(backend, top_l=retrieval.get("top_l", 20),
                                   max_selected=selection.get("max_selected", 5),
                                   retriever=retriever)
        for role, spec in models.get("roles", {}).items():
            role_backend = _backend(spec, f"models.roles.{role}")
            if isinstance(role_backend, OpenAICompatibleBackend):
                created.append(role_backend)
            getattr(modules, role).backend = role_backend
    except Exception:
        for item in created:
            item.close()
        raise
    return dataset, modules, run_dir, relative_store, revision.get("max_attempts", 2)


def _code_version() -> str:
    # Git HEAD 无法区别两个不同的未提交修改；对实际运行的 Python 源码取指纹。
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        commit = "unknown"
    return f"{commit}+sha256:{digest.hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_preparation_paths(options: dict[str, Any]) -> None:
    """The prepared dataset must be precisely the files named by the experiment."""
    prepare = options["prepare"]
    directory = Path(prepare["output_dir"])
    paths = options["paths"]
    splits = ("train",) if prepare.get("mode") == "full_train" else ("train", "test")
    if (options.get("field_map") is not None or not isinstance(paths, dict)
            or set(paths) != set(splits) or any(
                not isinstance(paths[split], str) or Path(paths[split]).resolve() !=
                (directory / f"{split}.jsonl").resolve() for split in splits)):
        raise ValueError("dataset.config.prepare: paths must point to output_dir/train.jsonl "
                         "and, in split mode, output_dir/test.jsonl with no field_map")


def _verify_prepared_dataset(config: dict[str, Any], dataset: JsonlDataset) -> dict[str, Any] | None:
    """Check provenance and exact bytes before any run directory or paid request is created."""
    options = config["dataset"]["config"]
    if "prepare" not in options:
        return None
    prepare = options["prepare"]
    provenance_file = Path(prepare["output_dir"]) / "provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("prepared dataset provenance must be an object")
    expected = {"dataset": "allenai/reward-bench", "source_split": "filtered",
                "category": "Chat Hard", "revision": prepare["revision"],
                "seed": prepare["seed"], "per_subset": prepare.get("per_subset"),
                "train_per_subset": prepare.get("train_per_subset")}
    full = prepare.get("mode") == "full_train"
    if full:
        expected.pop("per_subset")
        expected.pop("train_per_subset")
        expected["mode"] = "full_train"
    if any(type(provenance.get(key)) is not type(value) or provenance[key] != value
           for key, value in expected.items()):
        raise ValueError("prepared dataset provenance differs from dataset.config.prepare")
    counts = (provenance.get("samples", {}) if full else
              {"train": 6 * prepare["train_per_subset"],
               "test": 6 * (prepare["per_subset"] - prepare["train_per_subset"])})
    if full and (not isinstance(counts, dict) or set(counts) != {"train"} or type(counts["train"]) is not int or counts["train"] <= 0):
        raise ValueError("invalid full_train sample count")
    if provenance.get("samples") != counts:
        raise ValueError("prepared dataset provenance has inconsistent sample counts")
    hashes = {split: _sha256_file(path) for split, path in dataset.paths.items()}
    if provenance.get("sha256") != hashes:
        raise ValueError("prepared dataset files differ from provenance SHA-256")
    sample_ids: set[str] = set()
    for split in dataset.paths:
        samples = list(dataset.load(split))
        if len(samples) != counts[split]:
            raise ValueError(f"prepared dataset {split} has inconsistent sample count")
        subset_counts = {subset: 0 for subset in CHAT_HARD_SUBSETS}
        for sample in samples:
            subset = sample.metadata.get("subset")
            source_id = sample.metadata.get("source_id")
            if (subset not in subset_counts or type(source_id) is not int or
                    sample.sample_id != f"rewardbench:{subset}:{source_id}" or
                    sample.metadata.get("revision") != prepare["revision"]):
                raise ValueError(f"prepared dataset {split} has invalid RewardBench metadata")
            if sample.sample_id in sample_ids:
                raise ValueError("prepared dataset has duplicate sample IDs across splits")
            sample_ids.add(sample.sample_id)
            subset_counts[subset] += 1
        if full and (any(n <= 0 for n in subset_counts.values()) or
                     provenance.get("subset_counts") != subset_counts):
            raise ValueError("full_train subset counts differ from provenance")
        if not full and any(n != counts[split] // 6 for n in subset_counts.values()):
            raise ValueError(f"prepared dataset {split} is not balanced across Chat Hard subsets")
    return provenance


def _ensure_prepared_dataset(config: dict[str, Any], *, resuming: bool) -> None:
    """Only a wholly absent prepared dataset may be created; never repair partial inputs."""
    options = config["dataset"]["config"]
    prepare = options.get("prepare")
    if prepare is None:
        return
    if config["dataset"]["implementation"] != "jsonl":
        raise ValueError("dataset.config.prepare requires a jsonl dataset")
    _check_preparation_paths(options)
    directory = Path(prepare["output_dir"])
    if directory.exists():
        if not directory.is_dir() or not all((directory / name).is_file() for name in
                                              ([f"{split}.jsonl" for split in options["paths"]] + ["provenance.json"])):
            raise ValueError("prepared dataset directory is incomplete; refusing to overwrite or repair")
        return  # _verify_prepared_dataset checks provenance and bytes before any model request.
    if resuming:
        raise ValueError("resume requires the original prepared dataset; refusing to recreate it")
    prepare_rewardbench_chat_hard(
        directory, seed=prepare["seed"], per_subset=prepare.get("per_subset", 2),
        train_per_subset=prepare.get("train_per_subset", 1), revision=prepare["revision"],
        mode=prepare.get("mode", "split"),
        preserve_splits_from=prepare.get("preserve_splits_from"),
    )


def _resume_checkpoint(source: Path, config: dict[str, Any], dataset: JsonlDataset,
                       manifest: dict[str, Any]) -> tuple[Path, set[str], dict[str, Any]]:
    """只接受有序完整轨迹前缀、且下一样本未改写 Memory 的旧运行。"""
    old = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if (old.get("format_version") != 1 or old.get("dataset_version") != manifest["dataset_version"]
            or not isinstance(old.get("code_version"), str)):
        raise ValueError("resume: unsupported parent manifest")
    if old.get("prompt_versions") != manifest["prompt_versions"]:
        raise ValueError("resume: prompt protocol changed; start a fresh run")
    old_config = old.get("config")
    if not isinstance(old_config, dict) or not isinstance(old_config.get("run"), dict):
        raise ValueError("resume: invalid parent config")
    expected = {**old_config, "run": {**old_config["run"], "output_dir": config["run"]["output_dir"]}}
    # Older runs predate the preparation declaration; the same bytes and verified
    # provenance permit adding metadata without altering any executed parameter.
    if ("prepare" in config["dataset"]["config"] and
            "prepare" not in old_config["dataset"]["config"] and
            "dataset_provenance" in manifest):
        expected = {**expected, "dataset": {**old_config["dataset"], "config": {
            **old_config["dataset"]["config"], "prepare": config["dataset"]["config"]["prepare"],
        }}}
    if expected != config or old.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("resume: config or dataset changed (only run.output_dir may differ)")
    if source.resolve() != Path(old_config["run"]["output_dir"]).resolve():
        raise ValueError("resume: source does not match parent manifest output_dir")
    root = source / config["store"]["config"].get("root_dir", "artifacts")
    memory_file = root / "experience_memory.json"
    trajectory_file = root / "trajectories.jsonl"
    request_file = root / "llm_requests.jsonl"
    role_io_file = root / "role_io.jsonl"
    if not memory_file.is_file() or not trajectory_file.is_file():
        raise ValueError("resume: missing checkpoint memory or trajectories")
    samples = [*dataset.load("train")]
    train_ids = [sample.sample_id for sample in samples]
    test_ids = [sample.sample_id for sample in dataset.load("test")] if "test" in dataset.paths else []
    ordered = train_ids + test_ids
    if len(ordered) != len(set(ordered)):
        raise ValueError("resume: duplicate dataset sample IDs")
    completed = [record.sample_id for record in TrajectoryLog(root).all()]
    if not completed or len(completed) >= len(ordered) or completed != ordered[:len(completed)]:
        raise ValueError("resume: trajectories must be a nonempty strict dataset prefix")
    # 每条 train 完整轨迹先于下一条样本；测试不得写入 Beta evidence。
    completed_train = set(train_ids[:min(len(completed), len(train_ids))])
    memory = ExperienceMemory(root, read_only=True)
    for item in memory.all():
        if (any(key not in completed_train for key in item.source_trajectories)
                or any(ev.sample_id not in completed_train for ev in item.evidence_history)):
            raise ValueError("resume: Memory contains evidence beyond completed train prefix")
    next_id = ordered[len(completed)]
    parent_models = old_config["models"]
    if (any(spec["backend"] == "openai_compatible" for spec in
            [parent_models["default"], *parent_models.get("roles", {}).values()])
            and not request_file.is_file()):
        raise ValueError("resume: real backend requires request metadata to verify failed stage")
    # 前面的 Memory 检查已排除未完成样本的 credit / fusion 写盘；
    # 即使它已调用 Judge、Critic 或修订角色，也可从头重试这一整条样本。
    # 训练不得出现未来请求；只有 manifest 记载了并行测试时，才允许
    # 轨迹前缀之外的测试请求（可能已发出但尚未按序提交）。
    if request_file.exists():
        with request_file.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("sample_id"), str):
                    raise ValueError(f"resume: invalid request event at line {number}")
                if event["sample_id"] not in completed and event["sample_id"] != next_id:
                    # Only runs declaring parallel test execution may log beyond the prefix.
                    if not (len(completed) >= len(train_ids) and
                            old.get("execution", {}).get("test_workers", 1) > 1 and
                            event["sample_id"] in test_ids):
                        raise ValueError("resume: request event beyond next sample")
    if role_io_file.exists():
        with role_io_file.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("sample_id"), str):
                    raise ValueError(f"resume: invalid role IO event at line {number}")
                if event["sample_id"] not in completed and event["sample_id"] != next_id:
                    if not (len(completed) >= len(train_ids) and
                            old.get("execution", {}).get("test_workers", 1) > 1 and
                            event["sample_id"] in test_ids):
                        raise ValueError("resume: role IO event beyond next sample")
    copied = {name: _sha256_file(path) for name, path in (
        ("experience_memory.json", memory_file), ("trajectories.jsonl", trajectory_file),
        ("llm_requests.jsonl", request_file), ("role_io.jsonl", role_io_file),
    ) if path.exists()}
    lineage = {"source_run": str(source.resolve()), "source_code_version": old.get("code_version"),
               "completed_train": min(len(completed), len(train_ids)),
               "completed_test": max(0, len(completed) - len(train_ids)),
               "checkpoint_sha256": copied}
    return root, set(completed), lineage


class _Progress:
    """终端原地刷新；重定向时最多输出约 20 行，不污染 stdout JSON。"""

    def __init__(self, split: str, total: int) -> None:
        self.split, self.total, self.done = split, total, 0
        self.terminal = sys.stderr.isatty()
        self.started = monotonic()
        self._show()

    def _show(self) -> None:
        width = 20
        filled = round(width * self.done / self.total) if self.total else width
        elapsed = int(monotonic() - self.started)
        line = (f"{self.split}: [{'#' * filled}{'-' * (width - filled)}] "
                f"{self.done}/{self.total} ({elapsed}s)")
        if self.terminal:
            print(f"\r{line}", file=sys.stderr, end="\n" if self.done == self.total else "", flush=True)
        elif self.done == 0 or self.done == self.total or self.done % max(1, self.total // 20) == 0:
            print(line, file=sys.stderr, flush=True)

    def advance(self) -> None:
        self.done += 1
        self._show()

    def close(self) -> None:
        if self.terminal and self.done != self.total:
            print(file=sys.stderr, flush=True)


def _split_total(dataset: JsonlDataset, split: str) -> int:
    with dataset.paths[split].open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _report_sample_error(record: Any) -> None:
    if record.sample_error:
        error = record.sample_error
        print(f"warning: sample {record.sample_id!r}, stage {error['stage']}: "
              f"{error['type']}: {error['message']}; recorded, continuing", file=sys.stderr)


def _process_split(runner: PipelineRunner, dataset: JsonlDataset, split: str,
                   seen_ids: set[str], request_context: ContextVar[dict[str, str]],
                   progress: _Progress, skip_ids: set[str] | None = None,
                   stop_after: int | None = None) -> int:
    count = 0
    iterator = iter(dataset.load(split))
    while stop_after is None or count < stop_after:
        try:
            sample = next(iterator)
        except StopIteration:
            break
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"split {split!r}, stage dataset: {exc}") from exc
        if sample.sample_id in seen_ids:
            raise RuntimeError(f"sample {sample.sample_id!r}, stage dataset: duplicate ID across splits")
        seen_ids.add(sample.sample_id)
        if skip_ids is not None and sample.sample_id in skip_ids:
            count += 1
            progress.advance()
            continue
        token = request_context.set({"sample_id": sample.sample_id, "split": split})
        try:
            record = runner.run_sample(sample)
            _report_sample_error(record)
        except Exception as exc:
            raise RuntimeError(
                f"sample {sample.sample_id!r}, stage {runner.current_stage}: {exc}"
            ) from exc
        finally:
            request_context.reset(token)
        count += 1
        progress.advance()
    return count


class _DeferredTrajectoryLog:
    """Worker 不写共享轨迹；只由主线程按数据集顺序落盘。"""

    def append(self, record: Any) -> None:
        pass


def _process_test_parallel(runner: PipelineRunner, dataset: JsonlDataset,
                           seen_ids: set[str], request_context: ContextVar[dict[str, str]],
                           progress: _Progress, skip_ids: set[str], workers: int) -> int:
    if runner.training or not runner.memory.read_only:
        raise ValueError("parallel evaluation requires frozen read-only memory")
    thread_state = local()
    pending: deque[tuple[Any, Future[Any] | None]] = deque()
    count = 0

    def evaluate(sample: Any) -> Any:
        worker = getattr(thread_state, "runner", None)
        if worker is None:
            retriever = runner.modules.retriever
            if isinstance(retriever, VectorRetriever):
                # Copy the frozen training cache once per worker; no shared mutation or
                # re-embedding of already cached experiences.
                retriever = retriever.fork()
            modules = replace(runner.modules, retriever=retriever)
            worker = PipelineRunner(modules, runner.memory, _DeferredTrajectoryLog(),
                                    training=False, max_attempts=runner.max_attempts)
            thread_state.runner = worker
        token = request_context.set({"sample_id": sample.sample_id, "split": "test"})
        try:
            return worker.run_sample(sample)
        except Exception as exc:
            raise RuntimeError(f"sample {sample.sample_id!r}, stage {worker.current_stage}: {exc}") from exc
        finally:
            request_context.reset(token)

    def commit_next() -> None:
        nonlocal count
        _, future = pending.popleft()
        if future is not None:
            record = future.result()
            runner.trajectories.append(record)
            _report_sample_error(record)
        count += 1
        progress.advance()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            iterator = iter(dataset.load("test"))
            while True:
                try:
                    sample = next(iterator)
                except StopIteration:
                    break
                except (ValueError, OSError) as exc:
                    raise RuntimeError(f"split 'test', stage dataset: {exc}") from exc
                if sample.sample_id in seen_ids:
                    raise RuntimeError(f"sample {sample.sample_id!r}, stage dataset: duplicate ID across splits")
                seen_ids.add(sample.sample_id)
                if len(pending) >= workers:
                    commit_next()
                pending.append((sample, None if sample.sample_id in skip_ids
                                else pool.submit(evaluate, sample)))
            while pending:
                commit_next()
        finally:
            for _, future in pending:
                if future is not None:
                    future.cancel()
    return count


def run(config_path: str | Path, *, resume_from: str | Path | None = None,
        initial_output_retries: int = 2, stop_after_train: int | None = None) -> dict[str, Any]:
    """Fresh run, or continue a certified prefix into a new immutable-lineage directory."""
    if type(initial_output_retries) is not int or not 0 <= initial_output_retries <= 3:
        raise ConfigError("initial_output_retries must be an integer in [0, 3]")
    if stop_after_train is not None and (type(stop_after_train) is not int or stop_after_train < 1):
        raise ConfigError("stop_after_train must be a positive integer")
    modules = None
    try:
        config = load_config(config_path)
        # Reject an existing run directory before any optional network download.
        run_dir = Path(config["run"]["output_dir"])
        if run_dir.exists():
            raise ValueError(f"run.output_dir: already exists: {run_dir}; use a fresh directory")
        _ensure_prepared_dataset(config, resuming=resume_from is not None)
        dataset, modules, run_dir, store_root, max_attempts = _components(config)
        provenance = _verify_prepared_dataset(config, dataset)
        backend_names = {spec["backend"] for spec in
                         [config["models"]["default"], *config["models"].get("roles", {}).values()]}
        manifest = {"format_version": 1, "config": config, "code_version": _code_version(),
                    "execution": {"test_workers": config["run"].get("test_workers", 4),
                                  "mode": "train_test" if "test" in dataset.paths else "train_only",
                                  "initial_output_retries": initial_output_retries,
                                  "sample_error_policy": "record_and_continue"},
                    "dataset_version": "jsonl/v1",
                    "dataset_sha256": {split: _sha256_file(path)
                                       for split, path in dataset.paths.items()},
                    "module_versions": {name: "v1" for name in _IMPLEMENTATIONS},
                    "backend_versions": {name: "v1" for name in sorted(backend_names)},
                    "prompt_versions": {role: PROMPT_VERSION for role in _ROLES}}
        if stop_after_train is not None:
            manifest["execution"]["stop_after_train"] = stop_after_train
        if provenance is not None:
            manifest["dataset_provenance"] = provenance
        checkpoint = None
        skipped: set[str] = set()
        if resume_from is not None:
            checkpoint, skipped, lineage = _resume_checkpoint(
                Path(resume_from), config, dataset, manifest,
            )
            manifest["resumed_from"] = {**lineage, "initial_output_retries": initial_output_retries}
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (ValueError, OSError, TypeError) as exc:
        if modules is not None:
            _close_modules(modules)
        raise ConfigError(str(exc)) from exc
    random.seed(config["run"]["seed"])
    # 旧目录只读；复制已审计的完整前缀和 Memory 到全新目录，绝不覆盖来源。
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "run_manifest.json").write_text(serialized, encoding="utf-8")
        root = run_dir / store_root
        if checkpoint is not None:
            root.mkdir(parents=True, exist_ok=False)
            for name, digest in manifest["resumed_from"]["checkpoint_sha256"].items():
                shutil.copyfile(checkpoint / name, root / name)
                if _sha256_file(root / name) != digest:
                    raise ValueError(f"resume: copied checkpoint changed: {name}")
        memory = ExperienceMemory(root)
        trajectories = TrajectoryLog(root)
        role_io_file = root / "role_io.jsonl"
        role_io_file.touch(mode=0o600, exist_ok=True)
        role_io_file.chmod(0o600)
        modules.initial_generator.invalid_output_retries = initial_output_retries
        request_context: ContextVar[dict[str, str]] = ContextVar("sample_context", default={})
        log_lock = Lock()

        def save_request(event: dict[str, Any]) -> None:
            # 只记录白名单元数据；不记录消息、偏好文本或 API key。
            with log_lock, (root / "llm_requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**request_context.get(), "backend": "openai_compatible", **event},
                                        ensure_ascii=False, allow_nan=False) + "\n")

        def save_role_io(event: dict[str, Any]) -> None:
            # Raw role inputs/outputs are sensitive research artifacts, never request headers or keys.
            with log_lock, role_io_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**request_context.get(), **event},
                                        ensure_ascii=False, allow_nan=False) + "\n")

        backends = {id(item): item for item in [modules.backend, *(
            getattr(modules, role).backend for role in _ROLES)]
            if isinstance(item, (OpenAICompatibleBackend, FakeLLMBackend))}
        for item in backends.values():
            item.trace_hook = save_role_io
            if isinstance(item, OpenAICompatibleBackend):
                item.metadata_hook = save_request
        embedder = modules.retriever.backend if isinstance(modules.retriever, VectorRetriever) else None
        if isinstance(embedder, OpenAICompatibleEmbedding):
            embedder.metadata_hook = save_request
        runner = PipelineRunner(modules, memory, trajectories, max_attempts=max_attempts,
                                max_revision_output_retries=config["modules"]["revision"]["config"].get(
                                    "invalid_output_retries", 2))
        seen_ids: set[str] = set()
        train_total = _split_total(dataset, "train")
        train_progress = _Progress("train", train_total)
        try:
            train_count = _process_split(runner, dataset, "train", seen_ids,
                                         request_context, train_progress, skipped, stop_after_train)
        finally:
            train_progress.close()
        if train_count < train_total:
            return {"run_dir": str(run_dir), "train_samples": train_count, "test_samples": 0,
                    "train_total": train_total, "status": "paused_after_train_prefix"}
        test_count = 0
        if "test" in dataset.paths:
            # 测试阶段复用训练后的 Memory，完全不写回信用或生成经验。
            evaluation = PipelineRunner(modules, ExperienceMemory(root, read_only=True),
                                        trajectories, training=False, max_attempts=max_attempts)
            test_progress = _Progress("test", _split_total(dataset, "test"))
            try:
                workers = config["run"].get("test_workers", 4)
                if workers == 1:
                    test_count = _process_split(evaluation, dataset, "test", seen_ids,
                                                request_context, test_progress, skipped)
                else:
                    test_count = _process_test_parallel(evaluation, dataset, seen_ids,
                                                        request_context, test_progress, skipped, workers)
            finally:
                test_progress.close()
        return {"run_dir": str(run_dir), "train_samples": train_count, "test_samples": test_count}
    finally:
        _close_modules(modules)


def _close_modules(modules: Any) -> None:
    backends = {id(item): item for item in [modules.backend, *(
        getattr(modules, role).backend for role in _ROLES)]
        if isinstance(item, OpenAICompatibleBackend)}
    for item in backends.values():
        item.close()
    embedder = modules.retriever.backend if isinstance(modules.retriever, VectorRetriever) else None
    if isinstance(embedder, OpenAICompatibleEmbedding):
        embedder.close()


def inspect_sample(run_dir: str | Path, sample_id: str) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest_file = directory / "run_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"{manifest_file}: unsupported manifest format")
    store_root = Path(manifest["config"]["store"]["config"].get("root_dir", "artifacts"))
    if store_root.is_absolute() or ".." in store_root.parts or store_root == Path("."):
        raise ValueError(f"{manifest_file}: invalid store root_dir")
    record = TrajectoryLog(directory / store_root).get(sample_id)
    if record is None:
        raise ValueError(f"sample {sample_id!r} not found in {directory}")
    return record.to_dict()


def summarize_rewardbench(run_dir: str | Path) -> dict[str, Any]:
    """汇总本次 Chat Hard 内部分割的 test 初始判别率，不宣称官方榜单成绩。"""
    directory = Path(run_dir)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    paths = manifest["config"]["dataset"]["config"]["paths"]
    if "test" not in paths:
        raise ValueError("RewardBench summary requires a test split")
    test_hash = manifest.get("dataset_sha256", {}).get("test")
    if not isinstance(test_hash, str) or _sha256_file(Path(paths["test"])) != test_hash:
        raise ValueError("test dataset has changed since run or lacks a recorded SHA-256")
    dataset = JsonlDataset({"test": paths["test"]})
    expected = {sample.sample_id: sample.metadata.get("subset")
                for sample in dataset.load("test")}
    counts = {subset: list(expected.values()).count(subset) for subset in CHAT_HARD_SUBSETS}
    if (not expected or any(subset not in CHAT_HARD_SUBSETS for subset in expected.values())
            or len(set(counts.values())) != 1 or 0 in counts.values()):
        raise ValueError("test split must cover all six Chat Hard subsets equally")
    records = {}
    for record in TrajectoryLog(
            directory / manifest["config"]["store"]["config"]["root_dir"]).all():
        if record.sample_id in records:
            raise ValueError(f"duplicate trajectory sample_id {record.sample_id!r}")
        records[record.sample_id] = record
    missing = expected.keys() - records.keys()
    if missing:
        raise ValueError(f"incomplete test evaluation: {len(missing)} sample(s) have no trajectory")
    per_subset = {subset: {"total": 0, "correct": 0, "judge_errors": 0, "sample_errors": 0}
                  for subset in CHAT_HARD_SUBSETS}
    for sample_id, subset in expected.items():
        record = records[sample_id]
        judgment = record.initial_judgment
        per_subset[subset]["total"] += 1
        if judgment is None:
            if record.state == "sample_error" and record.sample_error:
                per_subset[subset]["sample_errors"] += 1
            elif record.state == "judge_error_initial" and record.judge_error:
                per_subset[subset]["judge_errors"] += 1
            else:
                raise ValueError(f"test sample {sample_id}: missing initial judgment")
        else:
            per_subset[subset]["correct"] += int(judgment.success)
    for counts in per_subset.values():
        valid = counts["total"] - counts["judge_errors"] - counts["sample_errors"]
        counts["accuracy"] = (counts["correct"] / counts["total"]
                              if counts["total"] and not counts["judge_errors"] and not counts["sample_errors"] else None)
        counts["accuracy_on_valid"] = counts["correct"] / valid if valid else None
    total = sum(item["total"] for item in per_subset.values())
    correct = sum(item["correct"] for item in per_subset.values())
    judge_errors = sum(item["judge_errors"] for item in per_subset.values())
    sample_errors = sum(item["sample_errors"] for item in per_subset.values())
    invalid_revisions = sum(len(record.invalid_revision_attempts) for record in records.values())
    return {"run_dir": str(directory), "scope": "internal Chat Hard split (not official leaderboard)",
            "test_total": total, "test_correct": correct,
            "test_judge_errors": judge_errors, "test_sample_errors": sample_errors,
            "test_accuracy": correct / total if not (judge_errors + sample_errors) else None,
            "test_accuracy_on_valid": correct / (total - judge_errors - sample_errors) if total > judge_errors + sample_errors else None,
            "per_subset": per_subset, "invalid_revision_attempts": invalid_revisions}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VERGE experiment CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="run train and optional held-out test")
    run_command.add_argument("--config", required=True)
    run_command.add_argument("--resume-from", help="copy a certified incomplete run to a new output directory")
    run_command.add_argument("--initial-output-retries", type=int,
                             help="semantic R0 validation retries (default: 2)")
    run_command.add_argument("--stop-after-train", type=int,
                             help="cleanly stop at this absolute training-prefix length, before test")
    inspect_command = commands.add_parser("inspect-sample", help="inspect a saved trajectory")
    inspect_command.add_argument("--run-dir", required=True)
    inspect_command.add_argument("--sample-id", required=True)
    summary_command = commands.add_parser("summarize-rewardbench", help="summarize Chat Hard test judgments")
    summary_command.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.config, resume_from=args.resume_from, stop_after_train=args.stop_after_train,
                         initial_output_retries=(args.initial_output_retries if
                                                 args.initial_output_retries is not None else
                                                 2))
        elif args.command == "summarize-rewardbench":
            result = summarize_rewardbench(args.run_dir)
        else:
            result = inspect_sample(args.run_dir, args.sample_id)
    except ConfigError as exc:
        print(f"error: config: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
