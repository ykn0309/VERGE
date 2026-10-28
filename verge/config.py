"""Strict V2 experiment configuration; no legacy protocol fallback."""

from pathlib import Path
import yaml


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {
        "run",
        "dataset",
        "models",
        "modules",
        "store",
    }:
        raise ValueError("V2 requires run/dataset/models/modules/store")
    run = data["run"]
    if (
        set(run) - {"seed", "output_dir", "stop_after_train", "test_workers", "train_workers", "mode"}
        or not {"seed", "output_dir"} <= run.keys()
    ):
        raise ValueError("invalid run configuration")
    if (
        type(run["seed"]) is not int
        or not isinstance(run["output_dir"], str)
        or not run["output_dir"]
    ):
        raise ValueError("invalid seed/output directory")
    if "stop_after_train" in run and (
        type(run["stop_after_train"]) is not int or run["stop_after_train"] < 1
    ):
        raise ValueError("invalid stop_after_train")
    if type(run.get("train_workers", 1)) is not int or run.get("train_workers", 1) < 1:
        raise ValueError("train_workers must be a positive integer")
    if run.get("test_workers", 1) != 1:
        raise ValueError(
            "V2 currently evaluates frozen tasks sequentially; use test_workers=1"
        )
    if run.get("mode", "learning") not in ("learning", "collection"):
        raise ValueError("run.mode must be learning or collection")
    mods = data["modules"]
    if set(mods) != {"retrieval", "selection", "application", "reliability"}:
        raise ValueError("invalid V2 modules")
    impls = {
        "retrieval": {"fake", "hybrid"},
        "selection": {"llm"},
        "application": {"semistructured_v2"},
        "reliability": {"bayesian"},
    }
    keys = {
        "retrieval": {"top_l", "rrf_k", "embedding_model", "dimensions"},
        "selection": {"max_selected"},
        "application": {
            "max_rounds",
            "max_candidates",
            "invalid_output_retries",
            "input_bytes",
            "check_interval",
            "error_window",
            "error_threshold",
            "q_threshold",
            "min_replay",
            "revision_enabled",
        },
        "reliability": {"parameter_file"},
    }
    for name, value in mods.items():
        if (
            set(value) != {"implementation", "config"}
            or value["implementation"] not in impls[name]
            or not isinstance(value["config"], dict)
            or set(value["config"]) - keys[name]
        ):
            raise ValueError("unsupported V2 component: " + name)
    opts = mods["application"]["config"]
    for key, low, high in [
        ("max_rounds", 1, 2),
        ("max_candidates", 1, 1),
        ("invalid_output_retries", 0, 3),
        ("input_bytes", 1000, 1000000),
        ("check_interval", 1, 100),
        ("error_window", 1, 100),
        ("error_threshold", 1, 100),
        ("min_replay", 1, 6),
    ]:
        if key in opts and (type(opts[key]) is not int or not low <= opts[key] <= high):
            raise ValueError("invalid " + key)
    if "revision_enabled" in opts and type(opts["revision_enabled"]) is not bool:
        raise ValueError("invalid revision_enabled")
    if "q_threshold" in opts and (
        type(opts["q_threshold"]) not in (int, float)
        or not 0 <= opts["q_threshold"] <= 1
    ):
        raise ValueError("invalid q_threshold")
    selected = mods["selection"]["config"].get("max_selected", 5)
    if type(selected) is not int or not 0 <= selected <= 5:
        raise ValueError("invalid max_selected")
    if mods["retrieval"]["config"].get("top_l", 20) != 20:
        raise ValueError("V2 Top20 retrieval required")
    if data["store"].get("implementation") != "local" or set(
        data["store"].get("config", {})
    ) - {"root_dir"}:
        raise ValueError("only local store supported")
    if data["dataset"].get("implementation") != "jsonl" or set(
        data["dataset"].get("config", {})
    ) - {"paths", "field_map"}:
        raise ValueError("prepare datasets separately, then use JSONL")
    if run.get("mode") == "collection":
        if mods["reliability"]["config"].get("parameter_file"):
            raise ValueError("collection mode requires no probability parameters")
        if opts.get("revision_enabled", True):
            raise ValueError("collection mode requires revision_enabled=false")
        if "test" in data["dataset"]["config"].get("paths", {}):
            raise ValueError("collection mode must not inspect a held-out split")
    return data
