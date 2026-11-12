"""逐行读取 JSONL 偏好样本，并转换为统一 PreferenceSample。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from verge.models import PreferenceSample


_FIELDS = ("sample_id", "task", "chosen", "rejected")


class JsonlDataset:
    """使用 paths 按 split 选文件；field_map 将标准字段映射到 JSONL 字段名。"""

    def __init__(
        self,
        paths: Mapping[str, str | Path],
        field_map: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(paths, Mapping) or not paths:
            raise ValueError(
                "dataset.paths: expected a non-empty split-to-path mapping"
            )
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
                    raise ValueError(
                        f"dataset.field_map.{key}: unknown field or invalid column name"
                    )
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
        tasks: set[str] = set()
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
                location = (
                    f"{position}: sample_id={row_id!r}"
                    if isinstance(row_id, str) and row_id.strip()
                    else position
                )
                data: dict[str, Any] = {}
                for field in _FIELDS:
                    column = self.field_map[field]
                    if column not in row:
                        raise ValueError(
                            f"{location}: missing {field} (column {column!r})"
                        )
                    data[field] = row[column]
                data["metadata"] = row.get(self.field_map["metadata"], {})
                try:
                    sample = PreferenceSample.from_dict(data)
                except ValueError as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                if sample.sample_id in seen:
                    raise ValueError(
                        f"{position}: duplicate sample_id {sample.sample_id!r}"
                    )
                if sample.task_id in tasks:
                    raise ValueError(f"{position}: duplicate normalized task")
                tasks.add(sample.task_id)
                seen.add(sample.sample_id)
                yield sample


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")
