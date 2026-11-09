"""Explicit configuration for the history-learning protocol; no V2 adapters."""

from copy import deepcopy
from pathlib import Path
import yaml

_DEFAULTS = {
    "run": {
        "output_dir": "runs/learning-demo",
        "train_workers": 5,
        "test_workers": 5,
        "max_inflight_requests": 16,
    },
    "dataset": {"paths": {"train": "configs/data/demo_train.jsonl"}},
    "model": {"backend": "fake", "model": "repairable"},
    "embedding": {"backend": "fake"},
    "roles": {},
    "clustering": {
        "reference_weight": 0.5,
        "task_threshold": 0.75,
        "experience_threshold": 0.75,
        "max_rounds": 5,
    },
    "retrieval": {"task_group_limit": 3, "historical": 6, "semantic": 4},
    "selection": {"max_selected": 2},
    "critic": {"max_candidates": 2, "max_rounds": 2},
    "requests": {"retries": 2, "input_bytes": 100000},
}


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - _DEFAULTS.keys():
        raise ValueError("unsupported learning configuration sections")
    config = deepcopy(_DEFAULTS)
    for section, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be an object")
        if (
            section not in ("model", "embedding", "roles")
            and set(value) - config[section].keys()
        ):
            raise ValueError(f"unknown {section} settings")
        config[section].update(value)
    for section, key in [
        ("run", "train_workers"),
        ("run", "test_workers"),
        ("run", "max_inflight_requests"),
        ("critic", "max_candidates"),
        ("critic", "max_rounds"),
        ("retrieval", "task_group_limit"),
    ]:
        if type(config[section][key]) is not int or config[section][key] < 1:
            raise ValueError(f"{section}.{key} must be positive integer")
    if config["critic"]["max_candidates"] > 2 or config["critic"]["max_rounds"] > 2:
        raise ValueError("Critic is limited to two candidates and two rounds")
    if (
        type(config["selection"]["max_selected"]) is not int
        or not 0 <= config["selection"]["max_selected"] <= 2
    ):
        raise ValueError("selection.max_selected must be 0..2")
    if (
        type(config["requests"]["retries"]) is not int
        or not 0 <= config["requests"]["retries"] <= 5
    ):
        raise ValueError("requests.retries must be 0..5")
    if (
        type(config["requests"]["input_bytes"]) is not int
        or config["requests"]["input_bytes"] < 1000
    ):
        raise ValueError("requests.input_bytes must be at least 1000")
    from verge.clustering import OnlineClustering
    from verge.retrieval import CandidateQuota

    OnlineClustering(**config["clustering"])
    CandidateQuota(config["retrieval"]["historical"], config["retrieval"]["semantic"])
    if not config["dataset"]["paths"] or set(config["dataset"]["paths"]) - {
        "train",
        "test",
    }:
        raise ValueError("dataset needs train and optional test paths")
    if "train" not in config["dataset"]["paths"]:
        raise ValueError("training path required")
    for spec in [config["model"], config["embedding"], *config["roles"].values()]:
        if not isinstance(spec, dict) or "api_key" in spec:
            raise ValueError(
                "model settings use api_key_env, never plaintext credentials"
            )
    if set(config["roles"]) - {"selector", "generator", "judge", "critic"}:
        raise ValueError("unknown role override")
    return config
