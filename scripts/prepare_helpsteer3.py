"""Prepare a pinned, strong-preference HelpSteer3 train set without model calls.

Run from the repository root: python -m scripts.prepare_helpsteer3
Validation data is deliberately left untouched. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from verge.dataset import JsonlDataset

REVISION = "f6d145777bcbde96137596340fab89793acd1031"
SOURCE_SHA256 = "32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb"
SOURCE_URL = f"https://huggingface.co/datasets/nvidia/HelpSteer3/resolve/{REVISION}/preference/train.jsonl.gz"
RULES = {
    "domain": "general", "language": "english", "single_user_turn": True,
    "minimum_absolute_preference": 2, "all_individual_scores_same_direction": True,
    "minimum_individual_strength": 2,
    "distinct_nonempty_responses": True, "task_deduplication": "whitespace_normalized",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(rows, *, seed: int = 42):
    """Return a deterministic, task-deduplicated list plus mutually exclusive counts."""
    selected, counts = [], Counter()
    for index, row in enumerate(rows):
        counts["source_rows"] += 1
        if row.get("domain") != "general":
            counts["other_domain"] += 1
            continue
        if row.get("language") != "english":
            counts["other_language"] += 1
            continue
        context = row.get("context")
        if (not isinstance(context, list) or len(context) != 1
                or not isinstance(context[0], dict) or context[0].get("role") != "user"):
            counts["not_single_user_turn"] += 1
            continue
        task = context[0].get("content")
        responses = [row.get("response1"), row.get("response2")]
        if any(not isinstance(s, str) or not s.strip() for s in [task, *responses]):
            raise ValueError(f"train row {index}: invalid task/response text")
        score = row.get("overall_preference")
        if type(score) is not int or score not in range(-3, 4):
            raise ValueError(f"train row {index}: invalid overall preference")
        if abs(score) < 2:
            counts["weak_or_tied_preference"] += 1
            continue
        annotations = row.get("individual_preference")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"train row {index}: missing individual annotations")
        for item in annotations:
            if (not isinstance(item, dict) or type(item.get("score")) is not int
                    or item["score"] not in range(-3, 4)
                    or not isinstance(item.get("reasoning"), str) or not item["reasoning"].strip()):
                raise ValueError(f"train row {index}: invalid individual annotation")
        if any(item["score"] * score <= 0 for item in annotations):
            counts["individual_direction_disagreement"] += 1
            continue
        if any(abs(item["score"]) < RULES["minimum_individual_strength"] for item in annotations):
            counts["weak_individual_preference"] += 1
            continue
        if responses[0].strip() == responses[1].strip():
            counts["identical_responses"] += 1
            continue
        identity = json.dumps([context, *responses], ensure_ascii=False, sort_keys=True).encode()
        sid = "helpsteer3:train:" + hashlib.sha256(identity).hexdigest()
        chosen_index = 0 if score < 0 else 1
        selected.append({
            "sample_id": sid, "task": task, "chosen": responses[chosen_index],
            "rejected": responses[1 - chosen_index],
            "metadata": {
                "source": "nvidia/HelpSteer3", "source_config": "preference",
                "source_split": "train", "source_index": index, "revision": REVISION,
                "domain": row["domain"], "language": row["language"],
                "overall_preference": score, "chosen_response": chosen_index + 1,
                "individual_preference": annotations,
            },
        })
    selected.sort(key=lambda s: (hashlib.sha256(f"{seed}:{s['sample_id']}".encode()).hexdigest(), s["sample_id"]))
    unique, seen = [], set()
    for sample in selected:
        task_key = re.sub(r"\s+", " ", sample["task"]).strip()
        if task_key in seen:
            counts["duplicate_task"] += 1
            continue
        seen.add(task_key)
        unique.append(sample)
    counts["retained"] = len(unique)
    assert sum(v for k, v in counts.items() if k != "source_rows") == counts["source_rows"]
    return unique, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("runs/helpsteer3-source"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/helpsteer3-general-consensus-data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite")
    args.source_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_dir / f"preference-train-{REVISION}.jsonl.gz"
    if not source.exists():
        partial = source.with_suffix(".download")
        if partial.exists():
            parser.error("partial source download exists; inspect it before retrying")
        print("Downloading pinned official train archive; validation remains untouched.", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, partial.open("xb") as dest:
            while chunk := response.read(1024 * 1024):
                dest.write(chunk)
        if sha256_file(partial) != SOURCE_SHA256:
            raise ValueError("source archive differs from Hugging Face LFS SHA256")
        partial.rename(source)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("cached source archive differs from pinned SHA256")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        samples, counts = select_samples((json.loads(line) for line in stream), seed=args.seed)
    if len(samples) < 30:
        raise ValueError("fewer than 30 eligible tasks")
    args.output_dir.mkdir()
    train = args.output_dir / "train.jsonl"
    with train.open("x", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    assert len(list(JsonlDataset({"train": train}).load("train"))) == len(samples)
    provenance = {
        "format_version": 1, "dataset": "nvidia/HelpSteer3", "source_config": "preference",
        "source_split": "train", "revision": REVISION, "source_url": SOURCE_URL,
        "source_archive": str(source), "source_sha256": SOURCE_SHA256,
        "rules": RULES, "seed": args.seed, "counts": counts,
        "samples": {"train": len(samples)}, "sha256": {"train": sha256_file(train)},
        "prefix_ids": [s["sample_id"] for s in samples[:30]],
        "validation_status": "not_downloaded_or_inspected",
        "annotation_policy": "metadata only; never included in LLM inputs",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
