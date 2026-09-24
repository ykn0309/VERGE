"""命令行入口：按配置运行 Fake/真实 LLM 实验，并按样本检查轨迹。"""

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
from verge.llm import FakeLLMBackend, LLMBackend, OpenAICompatibleBackend
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.modules import make_llm_modules
from verge.pipeline import PipelineRunner
from verge.prompts import PROMPT_VERSION


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


def _backend(spec: dict[str, Any], path: str) -> LLMBackend:
    if spec["backend"] == "fake":
        _only_keys(spec, {"backend", "model"}, path)
        if spec["model"] not in _FAKE_SCENARIOS:
            raise ValueError(f"{path}.model: unsupported fake scenario")
        return FakeLLMBackend(spec["model"])
    if spec["backend"] == "openai_compatible":
        _only_keys(spec, {"backend", "model", "base_url", "api_key_env",
                          "timeout_seconds", "max_retries"}, path)
        options = {key: spec[key] for key in ("base_url", "api_key_env", "timeout_seconds", "max_retries")
                   if key in spec}
        try:
            return OpenAICompatibleBackend(spec["model"], **options)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    raise ValueError(f"{path}.backend: supported backends are fake and openai_compatible")


def _components(config: dict[str, Any]) -> tuple[JsonlDataset, Any, Path, Path, int]:
    """运行前校验所有当前可装配的组件，不把未实现的配置静默降级。"""
    _only_keys(config, {"run", "dataset", "models", "modules", "store"}, "config")
    _only_keys(config["run"], {"seed", "output_dir"}, "run")
    dataset_config = config["dataset"]
    _only_keys(dataset_config, {"implementation", "config"}, "dataset")
    if dataset_config["implementation"] != "jsonl":
        raise ValueError("dataset.implementation: V1 supports only jsonl")
    options = dataset_config["config"]
    if set(options) - {"paths", "field_map"} or "paths" not in options:
        raise ValueError("dataset.config: expected paths and optional field_map")
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
        if configured[name]["implementation"] != implementation:
            raise ValueError(f"modules.{name}.implementation: V1 supports only {implementation}")
    retrieval = configured["retrieval"]["config"]
    selection = configured["selection"]["config"]
    revision = configured["revision"]["config"]
    credit = configured["revision_credit"]["config"]
    if (set(retrieval) - {"top_l"} or set(selection) - {"max_selected", "exploration_slots"}
            or set(revision) - {"max_attempts"} or set(credit) - {"max_budget"}
            or configured["reliability"]["config"]):
        raise ValueError("modules: unsupported V1 component options")
    if selection.get("exploration_slots", 0) != 0:
        raise ValueError("modules.selection.config.exploration_slots: V1 selector does not explore")
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

    created: list[OpenAICompatibleBackend] = []
    try:
        backend = _backend(models["default"], "models.default")
        if isinstance(backend, OpenAICompatibleBackend):
            created.append(backend)
        modules = make_llm_modules(backend, top_l=retrieval.get("top_l", 20),
                                   max_selected=selection.get("max_selected", 5))
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


def _process_split(runner: PipelineRunner, dataset: JsonlDataset, split: str,
                   seen_ids: set[str], request_context: dict[str, str] | None = None) -> int:
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
        if request_context is not None:
            request_context.update(sample_id=sample.sample_id, split=split)
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
        backend_names = {spec["backend"] for spec in
                         [config["models"]["default"], *config["models"].get("roles", {}).values()]}
        manifest = {"format_version": 1, "config": config, "code_version": _code_version(),
                    "dataset_version": "jsonl/v1",
                    "module_versions": {name: "v1" for name in _IMPLEMENTATIONS},
                    "backend_versions": {name: "v1" for name in sorted(backend_names)},
                    "prompt_versions": {role: PROMPT_VERSION for role in _ROLES}}
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
    request_context: dict[str, str] = {}

    def save_request(event: dict[str, Any]) -> None:
        # 只记录白名单元数据；不记录消息、偏好文本或 API key。
        with (root / "llm_requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**request_context, "backend": "openai_compatible", **event},
                                    ensure_ascii=False, allow_nan=False) + "\n")

    backends = {id(item): item for item in [modules.backend, *(
        getattr(modules, role).backend for role in _ROLES)]
        if isinstance(item, OpenAICompatibleBackend)}
    for item in backends.values():
        item.metadata_hook = save_request
    try:
        runner = PipelineRunner(modules, memory, trajectories, max_attempts=max_attempts)
        seen_ids: set[str] = set()
        train_count = _process_split(runner, dataset, "train", seen_ids, request_context)
        test_count = 0
        if "test" in dataset.paths:
            # 测试阶段复用训练后的 Memory，完全不写回信用或生成经验。
            evaluation = PipelineRunner(modules, ExperienceMemory(root, read_only=True),
                                        trajectories, training=False, max_attempts=max_attempts)
            test_count = _process_split(evaluation, dataset, "test", seen_ids, request_context)
        return {"run_dir": str(run_dir), "train_samples": train_count, "test_samples": test_count}
    finally:
        for item in backends.values():
            item.close()


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
