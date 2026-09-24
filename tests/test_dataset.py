"""JSONL 样本输入、split、字段映射和错误定位。"""

import pytest

from verge.dataset import JsonlDataset
from verge.models import PreferenceSample


def test_jsonl_loads_lazily_skips_blank_lines_and_preserves_metadata(tmp_path):
    file = tmp_path / "train.jsonl"
    file.write_text('\n{"sample_id":"a","task":"任务","chosen":"好","rejected":"差",'
                    '"metadata":{"source":"x"}}\n   \n'
                    '{"sample_id":"b","task":"task","chosen":"yes","rejected":"no"}\n',
                    encoding="utf-8")
    adapter = JsonlDataset({"train": file})
    records = adapter.load("train")
    assert next(records) == PreferenceSample("a", "任务", "好", "差", {"source": "x"})
    assert next(records) == PreferenceSample("b", "task", "yes", "no", {})
    with pytest.raises(StopIteration):
        next(records)


def test_field_mapping_and_split_selection(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"id":"a","prompt":"p","winner":"c","loser":"r",'
                     '"info":{"kind":"training"}}\n', encoding="utf-8")
    test.write_text('{"id":"b","prompt":"q","winner":"x","loser":"y"}\n',
                    encoding="utf-8")
    adapter = JsonlDataset({"train": train, "test": test},
                           {"sample_id": "id", "task": "prompt", "chosen": "winner",
                            "rejected": "loser", "metadata": "info"})
    assert list(adapter.load("train"))[0].metadata == {"kind": "training"}
    assert [record.sample_id for record in adapter.load("test")] == ["b"]
    with pytest.raises(ValueError, match="unknown split"):
        list(adapter.load("validation"))


@pytest.mark.parametrize("lines,location,detail", [
    ('{"task":"q","chosen":"a","rejected":"b"}\n', 1, "missing sample_id"),
    ('{"sample_id":"x","chosen":"a","rejected":"b"}\n', 1, "missing task"),
    ('{"sample_id":"x","task":"q","chosen":"a"}\n', 1, "missing rejected"),
    ('{"sample_id":"x","task":7,"chosen":"a","rejected":"b"}\n', 1,
     "PreferenceSample"),
    ('{"sample_id":" ","task":"q","chosen":"a","rejected":"b"}\n', 1,
     "sample_id"),
    ('{"sample_id":"x","task":"q","chosen":"a","rejected":"b",'
     '"metadata":null}\n', 1, "PreferenceSample"),
    ('{"sample_id":"x","task":"q","chosen":"a","rejected":"b"}\n'
     '\n{"sample_id":"x","task":"q","chosen":"a","rejected":"b"}\n', 3,
     "duplicate sample_id"),
    ('\n{bad json}\n', 2, "invalid JSON"),
    ('{"sample_id":"x","task":"q","chosen":"a","rejected":"b",'
     '"metadata":{"bad":NaN}}\n', 1, "invalid JSON"),
    ('[1,2,3]\n', 1, "expected a JSON object"),
])
def test_invalid_sample_reports_physical_line(tmp_path, lines, location, detail):
    file = tmp_path / "bad.jsonl"
    file.write_text(lines, encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        list(JsonlDataset({"train": file}).load("train"))
    assert f"{file}:{location}:" in str(exc.value)
    assert detail in str(exc.value)


def test_invalid_adapter_configuration(tmp_path):
    with pytest.raises(ValueError, match="dataset.paths"):
        JsonlDataset({})
    with pytest.raises(ValueError, match="dataset.paths.train"):
        JsonlDataset({"train": ""})
    with pytest.raises(ValueError, match="dataset.field_map"):
        JsonlDataset({"train": tmp_path / "train.jsonl"}, {"chosen": "task"})
    with pytest.raises(ValueError, match="dataset.field_map"):
        JsonlDataset({"train": tmp_path / "train.jsonl"}, {"unknown": "col"})
