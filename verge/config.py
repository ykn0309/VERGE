"""最小实验 YAML 配置加载；组件实现是否合法由后续 Registry 校验。"""

from __future__ import annotations

from math import isfinite
import re
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a mapping with string keys")
    return value


def _required(data: dict[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ValueError(f"{path}.{key}: required field is missing")
    return data[key]


def _text(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected a non-empty string")


def _integer(value: Any, path: str, lower: int, upper: int | None = None) -> None:
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        bound = f"{lower}..{upper}" if upper is not None else f">= {lower}"
        raise ValueError(f"{path}: expected an integer in {bound}")


def _unit(value: Any, path: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{path}: expected a finite number in [0, 1]")


def _component(value: Any, path: str) -> dict[str, Any]:
    component = _mapping(value, path)
    _text(_required(component, "implementation", path), f"{path}.implementation")
    _mapping(_required(component, "config", path), f"{path}.config")
    return component


def _model(value: Any, path: str) -> None:
    model = _mapping(value, path)
    _text(_required(model, "backend", path), f"{path}.backend")
    _text(_required(model, "model", path), f"{path}.model")


def load_config(path: str | Path) -> dict[str, Any]:
    """读取配置并校验公共形状；保留组件私有 config 供对应实现检查。"""
    file = Path(path)
    try:
        with file.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"{file}: invalid YAML: {exc}") from exc
    root = _mapping(data, str(file))

    run = _mapping(_required(root, "run", "config"), "run")
    _integer(_required(run, "seed", "run"), "run.seed", 0)
    _text(_required(run, "output_dir", "run"), "run.output_dir")
    if "experience_enabled" in run:
        raise ValueError(
            "run.experience_enabled was removed; use the candidate-validation protocol"
        )
    if "stop_after_train" in run:
        _integer(run["stop_after_train"], "run.stop_after_train", 1)

    dataset = _component(_required(root, "dataset", "config"), "dataset")
    if "prepare" in dataset["config"]:
        prepare = _mapping(dataset["config"]["prepare"], "dataset.config.prepare")
        expected = {
            "implementation",
            "output_dir",
            "revision",
            "seed",
            "per_subset",
            "train_per_subset",
            "preserve_splits_from",
            "mode",
        }
        if set(prepare) - expected:
            raise ValueError("dataset.config.prepare: unsupported fields")
        if prepare.get("implementation") != "rewardbench_chat_hard":
            raise ValueError(
                "dataset.config.prepare.implementation: expected rewardbench_chat_hard"
            )
        for name in ("output_dir", "revision"):
            _text(
                _required(prepare, name, "dataset.config.prepare"),
                f"dataset.config.prepare.{name}",
            )
        if not re.fullmatch(r"[0-9a-f]{40}", prepare["revision"]):
            raise ValueError(
                "dataset.config.prepare.revision: expected a 40-character commit SHA"
            )
        if "preserve_splits_from" in prepare:
            _text(
                prepare["preserve_splits_from"],
                "dataset.config.prepare.preserve_splits_from",
            )
        mode = prepare.get("mode", "split")
        if mode not in ("split", "full_train"):
            raise ValueError(
                "dataset.config.prepare.mode: expected split or full_train"
            )
        _integer(
            _required(prepare, "seed", "dataset.config.prepare"),
            "dataset.config.prepare.seed",
            0,
        )
        if mode == "full_train":
            if set(prepare) & {
                "per_subset",
                "train_per_subset",
                "preserve_splits_from",
            }:
                raise ValueError("full_train does not accept split sampling fields")
        else:
            for name, minimum in (("per_subset", 2), ("train_per_subset", 1)):
                _integer(
                    _required(prepare, name, "dataset.config.prepare"),
                    f"dataset.config.prepare.{name}",
                    minimum,
                )
            if prepare["train_per_subset"] >= prepare["per_subset"]:
                raise ValueError(
                    "dataset.config.prepare.train_per_subset: must be less than per_subset"
                )
    models = _mapping(_required(root, "models", "config"), "models")
    _model(_required(models, "default", "models"), "models.default")
    for role, value in _mapping(models.get("roles", {}), "models.roles").items():
        _text(role, "models.roles key")
        _model(value, f"models.roles.{role}")

    modules = _mapping(_required(root, "modules", "config"), "modules")
    for name, value in modules.items():
        _text(name, "modules key")
        component = _component(value, f"modules.{name}")
        options = component["config"]
        if name == "retrieval" and "top_l" in options:
            _integer(options["top_l"], "modules.retrieval.config.top_l", 1)
            if options["top_l"] != 20:
                raise ValueError("modules.retrieval.config.top_l: V1 requires 20")
        elif name == "selection":
            if "max_selected" in options:
                _integer(
                    options["max_selected"],
                    "modules.selection.config.max_selected",
                    0,
                    5,
                )
            if "exploration_slots" in options:
                _integer(
                    options["exploration_slots"],
                    "modules.selection.config.exploration_slots",
                    0,
                    1,
                )
                if (
                    "max_selected" in options
                    and options["exploration_slots"] > options["max_selected"]
                ):
                    raise ValueError(
                        "modules.selection.config.exploration_slots: cannot exceed max_selected"
                    )
        elif name == "application" and "max_rounds" in options:
            _integer(
                options["max_rounds"], "modules.application.config.max_rounds", 1, 2
            )
    if set(modules) != {"retrieval", "selection", "application", "reliability"}:
        raise ValueError(
            "modules: expected retrieval, selection, application, reliability"
        )
    if modules["application"]["implementation"] != "critic_candidate_validation":
        raise ValueError("modules.application: expected critic_candidate_validation")
    _component(_required(root, "store", "config"), "store")
    store_config = root["store"]["config"]
    if "root_dir" in store_config:
        _text(store_config["root_dir"], "store.config.root_dir")
    return root
