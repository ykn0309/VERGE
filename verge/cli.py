"""最小命令行入口：运行离线实验，并按样本检查轨迹。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from verge.config import load_config
from verge.dataset import JsonlDataset
from verge.llm import FakeLLMBackend
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import make_fake_modules
from verge.pipeline import PipelineRunner


_ROLES = (
    "selector", "initial_generator", "judge", "critic", "revision_generator",
    "reflector", "revision_credit_assigner", "admission_checker", "fusion_judge",
)
_FAKE_SCENARIOS = {"initial_success", "attempt_1_success", "attempt_2_success", "final_failure"}
_IMPLEMENTATIONS = {
    "retrieval": "fake", "selection": "llm", "revision": "two_attempt_same_base",
    "revision_credit": "operation_bound", "reliability": "beta",
}


class ConfigError(ValueError):
    """配置或组件组合无效；CLI 使用退出码 2。"""


def _only_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path}: unsupported fields: {', '.join(sorted(unknown))}")


def _fake_backend(spec: dict[str, Any], path: str) -> FakeLLMBackend:
    _only_keys(spec, {"backend", "model"}, path)
    if spec["backend"] != "fake" or spec["model"] not in _FAKE_SCENARIOS:
        raise ValueError(f"{path}: only fake backend with a supported scenario is available in stage 07")
    return FakeLLMBackend(spec["model"])


def _components(config: dict[str, Any]) -> tuple[JsonlDataset, Any, Path, Path, int]:
    """运行前校验所有当前可装配的组件，不把未实现的配置静默降级。"""
    _only_keys(config, {"run", "dataset", "models", "modules", "store"}, "config")
    _only_keys(config["run"], {"seed", "output_dir"}, "run")
    dataset_config = config["dataset"]
    _only_keys(dataset_config, {"implementation", "config"}, "dataset")
    if dataset_config["implementation"] != "jsonl":
        raise ValueError("dataset.implementation: stage 07 supports only jsonl")
    options = dataset_config["config"]
    if set(options) - {"paths", "field_map"} or "paths" not in options:
        raise ValueError("dataset.config: expected paths and optional field_map")
    dataset = JsonlDataset(options["paths"], options.get("field_map"))
    if "train" not in dataset.paths:
        raise ValueError("dataset.config.paths.train: required for run")
    if set(dataset.paths) - {"train", "test"}:
        raise ValueError("dataset.config.paths: stage 07 supports only train and optional test")
    for split, path in dataset.paths.items():
        if not path.is_file():
            raise ValueError(f"dataset.config.paths.{split}: file not found: {path}")

    models = config["models"]
    _only_keys(models, {"default", "roles"}, "models")
    backend = _fake_backend(models["default"], "models.default")
    if set(models.get("roles", {})) - set(_ROLES):
        raise ValueError("models.roles: unknown LLM role")

    configured = config["modules"]
    if set(configured) != set(_IMPLEMENTATIONS):
        raise ValueError(f"modules: expected exactly {', '.join(_IMPLEMENTATIONS)}")
    for name, implementation in _IMPLEMENTATIONS.items():
        _only_keys(configured[name], {"implementation", "config"}, f"modules.{name}")
        if configured[name]["implementation"] != implementation:
            raise ValueError(f"modules.{name}.implementation: stage 07 supports only {implementation}")
    retrieval = configured["retrieval"]["config"]
    selection = configured["selection"]["config"]
    revision = configured["revision"]["config"]
    credit = configured["revision_credit"]["config"]
    if (set(retrieval) - {"top_l"} or set(selection) - {"max_selected", "exploration_slots"}
            or set(revision) - {"max_attempts"} or set(credit) - {"max_budget"}
            or configured["reliability"]["config"]):
        raise ValueError("modules: unsupported component options for stage 07")
    if selection.get("exploration_slots", 0) != 0:
        raise ValueError("modules.selection.config.exploration_slots: fake selector does not explore")
    if credit.get("max_budget", 1.0) != 1.0:
        raise ValueError("modules.revision_credit.config.max_budget: stage 07 requires 1.0")

    modules = make_fake_modules(backend.scenario)
    modules.backend = backend
    for role in _ROLES:
        getattr(modules, role).backend = backend
    for role, spec in models.get("roles", {}).items():
        getattr(modules, role).backend = _fake_backend(spec, f"models.roles.{role}")
    modules.retriever.top_l = retrieval.get("top_l", 20)
    modules.selector.max_selected = selection.get("max_selected", 5)

    store = config["store"]
    _only_keys(store, {"implementation", "config"}, "store")
    if store["implementation"] != "local" or set(store["config"]) - {"root_dir"}:
        raise ValueError("store: stage 07 supports only local with root_dir")
    relative_store = Path(store["config"].get("root_dir", "artifacts"))
    if relative_store.is_absolute() or ".." in relative_store.parts or relative_store == Path("."):
        raise ValueError("store.config.root_dir: must be a relative subdirectory of run.output_dir")
    run_dir = Path(config["run"]["output_dir"])
    if run_dir.exists():
        raise ValueError(f"run.output_dir: already exists: {run_dir}; use a fresh directory")
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


def _process_split(runner: PipelineRunner, dataset: JsonlDataset, split: str,
                   seen_ids: set[str]) -> int:
    count = 0
    iterator = iter(dataset.load(split))
    while True:
        try:
            sample = next(iterator)
        except StopIteration:
            break
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"split {split!r}, stage dataset: {exc}") from exc
        if sample.sample_id in seen_ids:
            raise RuntimeError(f"sample {sample.sample_id!r}, stage dataset: duplicate ID across splits")
        seen_ids.add(sample.sample_id)
        try:
            runner.run_sample(sample)
        except Exception as exc:
            raise RuntimeError(
                f"sample {sample.sample_id!r}, stage {runner.current_stage}: {exc}"
            ) from exc
        count += 1
    return count


def run(config_path: str | Path) -> dict[str, Any]:
    try:
        config = load_config(config_path)
        dataset, modules, run_dir, store_root, max_attempts = _components(config)
        manifest = {"format_version": 1, "config": config, "code_version": _code_version()}
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (ValueError, OSError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    random.seed(config["run"]["seed"])
    # 配置验证完成后才建目录；同一目录不会重复写入或覆盖已有实验。
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_manifest.json").write_text(serialized, encoding="utf-8")
    root = run_dir / store_root
    memory = ExperienceMemory(root)
    trajectories = TrajectoryLog(root)
    runner = PipelineRunner(modules, memory, trajectories, max_attempts=max_attempts)
    seen_ids: set[str] = set()
    train_count = _process_split(runner, dataset, "train", seen_ids)
    test_count = 0
    if "test" in dataset.paths:
        # 测试阶段复用训练后的 Memory，完全不写回信用或生成经验。
        evaluation = PipelineRunner(modules, ExperienceMemory(root, read_only=True),
                                    trajectories, training=False, max_attempts=max_attempts)
        test_count = _process_split(evaluation, dataset, "test", seen_ids)
    return {"run_dir": str(run_dir), "train_samples": train_count, "test_samples": test_count}


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VERGE experiment CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="run train and optional held-out test")
    run_command.add_argument("--config", required=True)
    inspect_command = commands.add_parser("inspect-sample", help="inspect a saved trajectory")
    inspect_command.add_argument("--run-dir", required=True)
    inspect_command.add_argument("--sample-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.config)
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
