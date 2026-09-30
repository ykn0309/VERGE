"""逐行读取 JSONL 偏好样本，并转换为统一 PreferenceSample。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from verge.models import PreferenceSample


_FIELDS = ("sample_id", "task", "chosen", "rejected")

# 官方 RewardBench (allenai/reward-bench) constants.py 中的 Chat Hard 分类。
CHAT_HARD_SUBSETS = (
    "mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
    "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual",
)


class JsonlDataset:
    """使用 paths 按 split 选文件；field_map 将标准字段映射到 JSONL 字段名。"""

    def __init__(self, paths: Mapping[str, str | Path],
                 field_map: Mapping[str, str] | None = None) -> None:
        if not isinstance(paths, Mapping) or not paths:
            raise ValueError("dataset.paths: expected a non-empty split-to-path mapping")
        self.paths: dict[str, Path] = {}
        for split, path in paths.items():
            if not isinstance(split, str) or not split.strip():
                raise ValueError("dataset.paths: split must be a non-empty string")
            if not isinstance(path, (str, Path)) or not str(path).strip():
                raise ValueError(f"dataset.paths.{split}: expected a non-empty path")
            self.paths[split] = Path(path)

        names = {field: field for field in (*_FIELDS, "metadata")}
        if field_map is not None:
            if not isinstance(field_map, Mapping):
                raise ValueError("dataset.field_map: expected a mapping")
            for key, value in field_map.items():
                if key not in names or not isinstance(value, str) or not value.strip():
                    raise ValueError(f"dataset.field_map.{key}: unknown field or invalid column name")
                names[key] = value
        self.field_map = names
        if len(set(self.field_map.values())) != len(self.field_map):
            raise ValueError("dataset.field_map: source column names must be distinct")

    def load(self, split: str) -> Iterator[PreferenceSample]:
        """空行略过；非法行报文件名和物理行号；ID 在当前 split 内唯一。"""
        if split not in self.paths:
            raise ValueError(f"dataset.paths.{split}: unknown split")
        path = self.paths[split]
        seen: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                position = f"{path}:{line_number}"
                try:
                    row = json.loads(line, parse_constant=_invalid_constant)
                except ValueError as exc:
                    raise ValueError(f"{position}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{position}: expected a JSON object")
                row_id = row.get(self.field_map["sample_id"])
                location = (f"{position}: sample_id={row_id!r}"
                            if isinstance(row_id, str) and row_id.strip() else position)
                data: dict[str, Any] = {}
                for field in _FIELDS:
                    column = self.field_map[field]
                    if column not in row:
                        raise ValueError(f"{location}: missing {field} (column {column!r})")
                    data[field] = row[column]
                data["metadata"] = row.get(self.field_map["metadata"], {})
                try:
                    sample = PreferenceSample.from_dict(data)
                except ValueError as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                if sample.sample_id in seen:
                    raise ValueError(f"{position}: duplicate sample_id {sample.sample_id!r}")
                seen.add(sample.sample_id)
                yield sample


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def split_rewardbench_chat_hard(rows: Iterable[Mapping[str, Any]], *, seed: int = 42,
                                per_subset: int = 2, train_per_subset: int = 1,
                                fixed_splits: Mapping[tuple[str, int], str] | None = None
                                ) -> dict[str, list[dict[str, Any]]]:
    """按官方六个子集确定性采样；可保留先前的 split 归属，且题面不跨 split。"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (type(per_subset) is not int or type(train_per_subset) is not int
            or not 0 < train_per_subset < per_subset):
        raise ValueError("per_subset and train_per_subset must leave samples for both splits")
    fixed = dict(fixed_splits or {})
    if any(split not in ("train", "test") for split in fixed.values()):
        raise ValueError("fixed_splits must assign each sample to train or test")
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in CHAT_HARD_SUBSETS}
    for row in rows:
        subset = row.get("subset")
        if not isinstance(subset, str) or subset not in grouped:
            continue
        if (type(row.get("id")) is not int
                or any(not isinstance(row.get(field), str) or not row[field].strip()
                       for field in ("prompt", "chosen", "rejected"))):
            raise ValueError("RewardBench Chat Hard row has invalid id, prompt, chosen or rejected")
        grouped[subset].append(row)

    def prompt_key(row: Mapping[str, Any]) -> str:
        return re.sub(r"\s+", " ", row["prompt"]).strip()

    reserved_prompts: set[str] = set()
    found_fixed: set[tuple[str, int]] = set()
    for subset, candidates in grouped.items():
        for row in candidates:
            key = (subset, row["id"])
            if key in fixed:
                normalized = prompt_key(row)
                if key in found_fixed or normalized in reserved_prompts:
                    raise ValueError("fixed_splits contains duplicate ids or prompts")
                found_fixed.add(key)
                reserved_prompts.add(normalized)
    if found_fixed != fixed.keys():
        raise ValueError("fixed_splits contains samples absent from Chat Hard")

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    used_prompts: set[str] = set()
    used_ids: set[int] = set()
    for subset, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda row: (
            hashlib.sha256(f"{seed}:{subset}:{row['id']}".encode()).hexdigest(), row["id"]))
        counts = {"train": 0, "test": 0}

        def append(row: Mapping[str, Any], split: str) -> None:
            normalized = prompt_key(row)
            if normalized in used_prompts or row["id"] in used_ids:
                raise ValueError("fixed_splits conflicts with another selected sample")
            metadata = {"source": "allenai/reward-bench", "subset": subset,
                        "source_id": row["id"]}
            for field in ("chosen_model", "rejected_model"):
                if isinstance(row.get(field), str):
                    metadata[field] = row[field]
            selected[split].append({"sample_id": f"rewardbench:{subset}:{row['id']}",
                                    "task": row["prompt"], "chosen": row["chosen"],
                                    "rejected": row["rejected"], "metadata": metadata})
            used_prompts.add(normalized)
            used_ids.add(row["id"])
            counts[split] += 1

        for split, quota in (("train", train_per_subset),
                             ("test", per_subset - train_per_subset)):
            pinned = [row for row in ordered if fixed.get((subset, row["id"])) == split]
            if len(pinned) > quota:
                raise ValueError(f"RewardBench subset {subset}: too many fixed {split} samples")
            for row in pinned:
                append(row, split)
        for row in ordered:
            if (counts["train"] >= train_per_subset
                    and counts["test"] >= per_subset - train_per_subset):
                break
            if (subset, row["id"]) in fixed:
                continue
            normalized = prompt_key(row)
            if normalized in reserved_prompts or normalized in used_prompts or row["id"] in used_ids:
                continue
            split = "train" if counts["train"] < train_per_subset else "test"
            append(row, split)
        if counts != {"train": train_per_subset, "test": per_subset - train_per_subset}:
            raise ValueError(f"RewardBench subset {subset}: too few distinct prompts for requested split")
    return selected


def full_rewardbench_chat_hard(rows: Iterable[Mapping[str, Any]], *, seed: int = 42
                               ) -> dict[str, list[dict[str, Any]]]:
    """全部合法 Chat Hard 记录；全局哈希打散，不按题面去重。"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    samples = []
    seen: set[int] = set()
    for row in rows:
        subset = row.get("subset")
        if subset not in CHAT_HARD_SUBSETS:
            continue
        if (type(row.get("id")) is not int or row["id"] in seen or
                any(not isinstance(row.get(key), str) or not row[key].strip()
                    for key in ("prompt", "chosen", "rejected"))):
            raise ValueError("RewardBench full train has invalid fields or duplicate source ID")
        seen.add(row["id"])
        metadata = {"source": "allenai/reward-bench", "subset": subset, "source_id": row["id"]}
        for key in ("chosen_model", "rejected_model"):
            if isinstance(row.get(key), str):
                metadata[key] = row[key]
        samples.append({"sample_id": f"rewardbench:{subset}:{row['id']}",
                        "task": row["prompt"], "chosen": row["chosen"],
                        "rejected": row["rejected"], "metadata": metadata})
    if {item["metadata"]["subset"] for item in samples} != set(CHAT_HARD_SUBSETS):
        raise ValueError("RewardBench full train requires all six Chat Hard subsets")
    samples.sort(key=lambda item: (hashlib.sha256(
        f"{seed}:{item['sample_id']}".encode()).hexdigest(), item["sample_id"]))
    return {"train": samples}


def _preserved_chat_hard_splits(directory: Path, revision: str
                                ) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    """从同一上游版本的已校验 JSONL 读取样本 split 归属。"""
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if (not isinstance(provenance, dict)
            or provenance.get("dataset") != "allenai/reward-bench"
            or provenance.get("revision") != revision):
        raise ValueError("preserved split must come from the same RewardBench revision")
    original_hashes = provenance.get("sha256")
    declared_samples = provenance.get("samples")
    per_subset = provenance.get("per_subset")
    train_per_subset = provenance.get("train_per_subset")
    if (not isinstance(original_hashes, dict) or not isinstance(declared_samples, dict)
            or type(per_subset) is not int or type(train_per_subset) is not int
            or not 0 < train_per_subset < per_subset):
        raise ValueError("preserved split has invalid provenance counts or hashes")
    fixed: dict[tuple[str, int], str] = {}
    hashes: dict[str, str] = {}
    expected_counts = {"train": train_per_subset, "test": per_subset - train_per_subset}
    for split in ("train", "test"):
        path = directory / f"{split}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != original_hashes.get(split):
            raise ValueError(f"preserved split {split} does not match its provenance hash")
        hashes[split] = digest
        subset_counts = {subset: 0 for subset in CHAT_HARD_SUBSETS}
        for sample in JsonlDataset({split: path}).load(split):
            subset = sample.metadata.get("subset")
            source_id = sample.metadata.get("source_id")
            if (subset not in CHAT_HARD_SUBSETS or type(source_id) is not int
                    or sample.sample_id != f"rewardbench:{subset}:{source_id}"
                    or sample.metadata.get("revision") != revision):
                raise ValueError("preserved split has invalid RewardBench sample metadata")
            key = (subset, source_id)
            if key in fixed:
                raise ValueError("preserved splits have overlapping sample IDs")
            fixed[key] = split
            subset_counts[subset] += 1
        if (any(count != expected_counts[split] for count in subset_counts.values())
                or declared_samples.get(split) != sum(subset_counts.values())):
            raise ValueError("preserved split does not match its provenance counts")
    return fixed, hashes


def prepare_rewardbench_chat_hard(output_dir: str | Path, *, seed: int = 42,
                                  per_subset: int = 2, train_per_subset: int = 1,
                                  revision: str | None = None,
                                  preserve_splits_from: str | Path | None = None,
                                  mode: str = "split"
                                  ) -> dict[str, Any]:
    """固定上游 SHA 下载 filtered split；按模式采样或全量训练，写入新目录。"""
    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, hf_hub_download

    if mode not in ("split", "full_train") or (mode == "full_train" and preserve_splits_from is not None):
        raise ValueError("invalid preparation mode or full_train preserve_splits_from")
    dataset_id = "allenai/reward-bench"
    if revision is not None and (not isinstance(revision, str)
                                 or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("RewardBench revision must be a 40-character commit SHA")
    target = Path(output_dir)
    if target.exists():
        raise ValueError(f"RewardBench output_dir already exists: {target}")
    info = HfApi().dataset_info(dataset_id, revision=revision)
    if revision is not None and info.sha != revision:
        raise ValueError("RewardBench upstream revision differs from requested commit")
    revision = info.sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("RewardBench upstream did not return a valid commit SHA")
    files = sorted(item.rfilename for item in info.siblings
                   if re.fullmatch(r"data/filtered-.*\.parquet", item.rfilename))
    if not files:
        raise ValueError("RewardBench filtered Parquet files not found")

    def filtered_rows() -> Iterator[dict[str, Any]]:
        for filename in files:
            path = hf_hub_download(dataset_id, filename=filename,
                                   repo_type="dataset", revision=revision)
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=1024):
                yield from batch.to_pylist()

    fixed, previous_hashes = ({}, {}) if preserve_splits_from is None else (
        _preserved_chat_hard_splits(Path(preserve_splits_from), revision))
    selected = (full_rewardbench_chat_hard(filtered_rows(), seed=seed) if mode == "full_train"
                else split_rewardbench_chat_hard(filtered_rows(), seed=seed, per_subset=per_subset,
                                                train_per_subset=train_per_subset, fixed_splits=fixed))
    target.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for split, samples in selected.items():
        for sample in samples:
            sample["metadata"]["revision"] = revision
        content = "".join(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n"
                          for sample in samples)
        content_bytes = content.encode("utf-8")
        (target / f"{split}.jsonl").write_bytes(content_bytes)
        hashes[split] = hashlib.sha256(content_bytes).hexdigest()
    provenance = {"dataset": dataset_id, "revision": revision, "source_split": "filtered",
                  "category": "Chat Hard", "seed": seed, "per_subset": per_subset,
                  "train_per_subset": train_per_subset,
                  "samples": {split: len(samples) for split, samples in selected.items()},
                  "sha256": hashes}
    if mode == "full_train":
        provenance.pop("per_subset")
        provenance.pop("train_per_subset")
        provenance["mode"] = mode
        provenance["subset_counts"] = {subset: sum(item["metadata"]["subset"] == subset
            for item in selected["train"]) for subset in CHAT_HARD_SUBSETS}
    if previous_hashes:
        provenance["preserved_split_sha256"] = previous_hashes
        provenance["fixed_assignments"] = {
            split: [{"subset": subset, "source_id": source_id}
                    for (subset, source_id), assigned in sorted(fixed.items()) if assigned == split]
            for split in ("train", "test")}
    (target / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(target), **provenance}
