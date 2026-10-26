"""Stratified General-only pilot sampling; no downloads, validation access or LLM calls."""

from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.dataset import JsonlDataset


def stratified_general(samples, count=30, seed=42):
    if count < 1 or count > len(samples):
        raise ValueError("sample count must be positive and no larger than population")
    if any(s.metadata.get("domain") != "general" for s in samples):
        raise ValueError("General-only population required")
    if len({s.task_id for s in samples}) != len(samples):
        raise ValueError("population tasks must be unique")
    lengths = sorted(len(s.task) for s in samples)
    thresholds = [lengths[(len(lengths) - 1) * i // 3] for i in (1, 2)]
    groups = defaultdict(list)

    def group(s):
        strength = abs(s.metadata["overall_preference"])
        if strength not in (2, 3):
            raise ValueError("strong preference population required")
        length = len(s.task)
        bucket = (
            "short"
            if length <= thresholds[0]
            else "medium" if length <= thresholds[1] else "long"
        )
        return f"{strength}:{bucket}"

    for s in samples:
        groups[group(s)].append(s)
    total = len(samples)
    allocation = {k: count * len(v) // total for k, v in groups.items()}
    order = sorted(groups, key=lambda k: (-(count * len(groups[k]) % total), k))
    for k in order[: count - sum(allocation.values())]:
        allocation[k] += 1
    rng = random.Random(seed)
    selected = []
    for k in sorted(groups):
        selected.extend(
            rng.sample(sorted(groups[k], key=lambda s: s.sample_id), allocation[k])
        )
    rng.shuffle(selected)
    summary = {
        "seed": seed,
        "count": count,
        "population": total,
        "strata": "absolute preference strength × task character length tertiles",
        "length_thresholds": thresholds,
        "population_counts": {k: len(groups[k]) for k in sorted(groups)},
        "sample_counts": allocation,
        "sample_ids": [s.sample_id for s in selected],
    }
    return selected, summary


def prepare(source, output, count=30, seed=42):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("output must be a new directory")
    samples = list(JsonlDataset({"train": source}).load("train"))
    selected, summary = stratified_general(samples, count, seed)
    summary.update(
        source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="filtered General/English/single-turn/strong-consensus train; not full HelpSteer3 or user traffic",
        held_out_accessed=False,
    )
    output.mkdir(parents=True)
    dest = output / "train.jsonl"
    dest.write_text("".join(s.to_json() + "\n" for s in selected))
    summary["sample_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (output / "sampling.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", default="runs/helpsteer3-general-consensus-data/train.jsonl"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    r = prepare(a.source, a.output, a.count, a.seed)
    print(
        json.dumps(
            {k: v for k, v in r.items() if k != "sample_ids"},
            ensure_ascii=False,
            indent=2,
        )
    )
