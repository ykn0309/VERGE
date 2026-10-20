"""RewardBench Chat Hard 只用伪造行做离线 contract/integration 测试。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from verge.cli import main
from verge.dataset import (CHAT_HARD_SUBSETS, JsonlDataset,
                           prepare_rewardbench_chat_hard, split_rewardbench_chat_hard)


PROJECT = Path(__file__).resolve().parents[1]


def synthetic_rows():
    return [{"subset": subset, "id": section * 10 + number,
             "prompt": f"unique prompt {subset} {number}", "chosen": "good answer",
             "rejected": "bad answer", "chosen_model": "a", "rejected_model": "b"}
            for section, subset in enumerate(CHAT_HARD_SUBSETS)
            for number in range(4)]


def test_chat_hard_split_is_deterministic_balanced_and_prompt_disjoint():
    rows = synthetic_rows()
    rows.append({**rows[0], "id": 999, "subset": "not-chat-hard"})
    first = split_rewardbench_chat_hard(rows, seed=42)
    assert first == split_rewardbench_chat_hard(list(reversed(rows)), seed=42)
    assert len(first["train"]) == len(first["test"]) == len(CHAT_HARD_SUBSETS)
    for name in ("train", "test"):
        assert Counter(item["metadata"]["subset"] for item in first[name]) == {
            subset: 1 for subset in CHAT_HARD_SUBSETS}
    assert not ({item["task"] for item in first["train"]}
                & {item["task"] for item in first["test"]})
    assert len({item["sample_id"] for split in first.values() for item in split}) == 12
    assert first["train"][0]["metadata"]["chosen_model"] == "a"


def test_chat_hard_preserves_previous_split_membership():
    rows = synthetic_rows()
    earlier = split_rewardbench_chat_hard(rows)
    fixed = {(sample["metadata"]["subset"], sample["metadata"]["source_id"]): split
             for split, samples in earlier.items() for sample in samples}
    result = split_rewardbench_chat_hard(rows, per_subset=4, train_per_subset=3,
                                         fixed_splits=fixed)
    assert len(result["train"]) == 18 and len(result["test"]) == 6
    for split, samples in earlier.items():
        assert {sample["sample_id"] for sample in samples} <= {
            sample["sample_id"] for sample in result[split]}
    assert not ({sample["task"] for sample in result["train"]}
                & {sample["task"] for sample in result["test"]})
    with pytest.raises(ValueError, match="absent from Chat Hard"):
        split_rewardbench_chat_hard(rows, fixed_splits={("mt-bench-hard", 999): "train"})
    with pytest.raises(ValueError, match="too many fixed"):
        split_rewardbench_chat_hard(rows, fixed_splits={
            ("mt-bench-hard", number): "train" for number in range(2)})


def test_chat_hard_cross_subset_prompt_leakage_is_prevented():
    rows = synthetic_rows()
    # 除了共享题面，每个子集还有足够独立 prompt；顺序不影响拆分。
    for row in rows:
        if row["id"] % 10 == 0:
            row["prompt"] = "same task across subsets"
    selected = split_rewardbench_chat_hard(rows)
    all_prompts = [item["task"] for split in selected.values() for item in split]
    assert len(all_prompts) == len(set(all_prompts)) == 12


def test_chat_hard_rejects_invalid_counts_rows_and_shortage():
    for options in ({"per_subset": 1}, {"train_per_subset": 2}, {"seed": -1},
                    {"per_subset": True}):
        with pytest.raises(ValueError):
            split_rewardbench_chat_hard(synthetic_rows(), **options)
    with pytest.raises(ValueError, match="invalid id"):
        split_rewardbench_chat_hard([{**synthetic_rows()[0], "id": "wrong"}])
    with pytest.raises(ValueError, match="too few distinct prompts"):
        split_rewardbench_chat_hard(synthetic_rows()[:1])


def test_prepare_chat_hard_pins_revision_and_never_overwrites(tmp_path, monkeypatch):
    revision = "a" * 40
    observed = []
    source = tmp_path / "filtered.parquet"
    pq.write_table(pa.Table.from_pylist(synthetic_rows()), source)
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info",
                        lambda self, name, revision=None: SimpleNamespace(
                            sha="a" * 40, siblings=[SimpleNamespace(
                                rfilename="data/filtered-00000-of-00001.parquet")]))

    def download(*args, **kwargs):
        observed.append((args, kwargs))
        return source

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    output = tmp_path / "data"
    info = prepare_rewardbench_chat_hard(output)
    assert observed == [(("allenai/reward-bench",),
                         {"filename": "data/filtered-00000-of-00001.parquet",
                          "repo_type": "dataset", "revision": revision})]
    assert info["samples"] == {"train": 6, "test": 6}
    assert json.loads((output / "provenance.json").read_text())["revision"] == revision
    for split in ("train", "test"):
        content = (output / f"{split}.jsonl").read_bytes()
        assert info["sha256"][split] == hashlib.sha256(content).hexdigest()
        samples = list(JsonlDataset({split: output / f"{split}.jsonl"}).load(split))
        assert len(samples) == 6
        assert all(item.metadata["revision"] == revision for item in samples)
    with pytest.raises(ValueError, match="already exists"):
        prepare_rewardbench_chat_hard(output, revision=revision)
    with pytest.raises(ValueError, match="commit SHA"):
        prepare_rewardbench_chat_hard(tmp_path / "other", revision="main")
    expanded = tmp_path / "expanded"
    expanded_info = prepare_rewardbench_chat_hard(
        expanded, revision=revision, per_subset=4, train_per_subset=3,
        preserve_splits_from=output)
    assert expanded_info["samples"] == {"train": 18, "test": 6}
    assert expanded_info["preserved_split_sha256"] == info["sha256"]
    assert {split: len(samples) for split, samples in
            expanded_info["fixed_assignments"].items()} == {"train": 6, "test": 6}
    for split in ("train", "test"):
        old_ids = {item.sample_id for item in JsonlDataset(
            {split: output / f"{split}.jsonl"}).load(split)}
        new_ids = {item.sample_id for item in JsonlDataset(
            {split: expanded / f"{split}.jsonl"}).load(split)}
        assert old_ids <= new_ids
    (output / "test.jsonl").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hash"):
        prepare_rewardbench_chat_hard(tmp_path / "tampered", revision=revision,
                                      preserve_splits_from=output)
    (output / "test.jsonl").write_bytes((expanded / "test.jsonl").read_bytes())
    info_file = output / "provenance.json"
    older_info = json.loads(info_file.read_text(encoding="utf-8"))
    older_info["sha256"]["test"] = hashlib.sha256((output / "test.jsonl").read_bytes()).hexdigest()
    older_info["samples"]["test"] = 5
    info_file.write_text(json.dumps(older_info), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance counts"):
        prepare_rewardbench_chat_hard(tmp_path / "wrong-count", revision=revision,
                                      preserve_splits_from=output)


def test_prepare_from_experiment_config_and_refuse_mismatch(
    tmp_path, monkeypatch, capsys
):
    revision = "b" * 40
    source = tmp_path / "filtered.parquet"
    pq.write_table(pa.Table.from_pylist(synthetic_rows()), source)
    observed = []

    def info(self, name, revision=None):
        observed.append(revision)
        return SimpleNamespace(
            sha="b" * 40,
            siblings=[
                SimpleNamespace(rfilename="data/filtered-00000-of-00001.parquet")
            ],
        )

    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", info)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *args, **kwargs: source
    )
    output = tmp_path / "prepared"
    config = yaml.safe_load(
        (PROJECT / "configs/rewardbench-chat-hard-qwen.yaml").read_text()
    )
    config["modules"] = yaml.safe_load((PROJECT / "configs/default.yaml").read_text())[
        "modules"
    ]
    config["dataset"]["config"]["prepare"].update(
        output_dir=str(output), revision=revision, seed=7
    )
    config["dataset"]["config"]["paths"] = {
        split: str(output / f"{split}.jsonl") for split in ("train", "test")
    }
    config["run"]["output_dir"] = str(tmp_path / "experiment")
    config["models"]["default"] = {"backend": "fake", "model": "initial_success"}
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config), encoding="utf-8")
    wrong_paths = yaml.safe_load(file.read_text(encoding="utf-8"))
    wrong_paths["dataset"]["config"]["paths"]["test"] = str(tmp_path / "wrong.jsonl")
    file.write_text(yaml.safe_dump(wrong_paths), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 2
    assert "paths must point" in capsys.readouterr().err
    assert not observed and not output.exists()
    with pytest.raises(SystemExit, match="2"):
        main(["prepare-rewardbench", "--config", str(file)])
    assert not observed and not output.exists()
    capsys.readouterr()
    file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 0
    assert observed == [revision]
    capsys.readouterr()
    result = json.loads((output / "provenance.json").read_text())
    assert result["seed"] == 7 and result["samples"] == {"train": 6, "test": 6}
    manifest = json.loads((tmp_path / "experiment/run_manifest.json").read_text())
    assert manifest["dataset_provenance"]["sha256"] == result["sha256"]
    # A second fresh run reuses the same verified bytes, without another download.
    config["run"]["output_dir"] = str(tmp_path / "reused")
    file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 0
    assert observed == [revision]
    capsys.readouterr()
    # Mismatched preparation settings or tampered files fail before a run directory is created.
    for change in ("seed", "hash", "path", "missing-provenance"):
        config["run"]["output_dir"] = str(tmp_path / f"refused-{change}")
        if change == "seed":
            config["dataset"]["config"]["prepare"]["seed"] = 8
        elif change == "hash":
            (output / "test.jsonl").write_text("tampered\n", encoding="utf-8")
        elif change == "path":
            config["dataset"]["config"]["paths"]["test"] = str(tmp_path / "other.jsonl")
        else:
            (output / "provenance.json").unlink()
        file.write_text(yaml.safe_dump(config), encoding="utf-8")
        assert main(["run", "--config", str(file)]) == 2
        error = capsys.readouterr().err
        assert (
            "provenance"
            if change not in ("path", "missing-provenance")
            else ("paths must point" if change == "path" else "incomplete")
        ) in error
        assert not (tmp_path / f"refused-{change}").exists()
        config["dataset"]["config"]["prepare"]["seed"] = 7
        config["dataset"]["config"]["paths"]["test"] = str(output / "test.jsonl")
    # An existing but incomplete directory is never re-downloaded or repaired.
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    config["dataset"]["config"]["prepare"]["output_dir"] = str(incomplete)
    config["dataset"]["config"]["paths"] = {
        split: str(incomplete / f"{split}.jsonl") for split in ("train", "test")
    }
    config["run"]["output_dir"] = str(tmp_path / "no-partial-repair")
    file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 2
    assert "incomplete" in capsys.readouterr().err
    assert list(incomplete.iterdir()) == [] and observed == [revision]
    # A resume must never recreate absent data, even if the config can prepare it.
    config["dataset"]["config"]["prepare"]["output_dir"] = str(tmp_path / "missing")
    config["dataset"]["config"]["paths"] = {
        split: str(tmp_path / "missing" / f"{split}.jsonl")
        for split in ("train", "test")
    }
    file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert (
        main(
            [
                "run",
                "--config",
                str(file),
                "--resume-from",
                str(tmp_path / "experiment"),
            ]
        )
        == 2
    )
    assert "resume requires the original prepared dataset" in capsys.readouterr().err
    assert observed == [revision]


def test_chat_hard_config_runs_existing_pipeline_offline(tmp_path, monkeypatch, capsys):
    revision = "f" * 40
    source = tmp_path / "filtered.parquet"
    pq.write_table(pa.Table.from_pylist(synthetic_rows()), source)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.dataset_info",
        lambda self, name, revision=None: SimpleNamespace(
            sha="f" * 40,
            siblings=[
                SimpleNamespace(rfilename="data/filtered-00000-of-00001.parquet")
            ],
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *args, **kwargs: source
    )
    data_dir = tmp_path / "data"
    prepare_rewardbench_chat_hard(data_dir, revision=revision)
    config = yaml.safe_load(
        (PROJECT / "configs/rewardbench-chat-hard-qwen.yaml").read_text()
    )
    config["modules"] = yaml.safe_load((PROJECT / "configs/default.yaml").read_text())[
        "modules"
    ]
    config["dataset"]["config"]["paths"] = {
        split: str(data_dir / f"{split}.jsonl") for split in ("train", "test")
    }
    config["dataset"]["config"]["prepare"]["output_dir"] = str(data_dir)
    config["dataset"]["config"]["prepare"]["revision"] = revision
    config["run"]["output_dir"] = str(tmp_path / "experiment")
    config["models"]["default"] = {"backend": "fake", "model": "initial_success"}
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "--config", str(file)]) == 0, capsys.readouterr().err
    capsys.readouterr()
    trajectories = (
        (tmp_path / "experiment/artifacts/trajectories.jsonl").read_text().splitlines()
    )
    assert len(trajectories) == 12
    assert all(json.loads(row)["initial_judgment"]["success"] for row in trajectories)
    assert (
        main(["summarize-rewardbench", "--run-dir", str(tmp_path / "experiment")]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["test_total"] == summary["test_correct"] == 6
    assert summary["test_accuracy"] == 1.0
    assert summary["invalid_application_attempts"] == 0
    assert all(item["accuracy"] == 1.0 for item in summary["per_subset"].values())
    trajectory_file = tmp_path / "experiment/artifacts/trajectories.jsonl"
    original_trajectories = trajectory_file.read_text(encoding="utf-8")
    audited = [json.loads(row) for row in trajectories]
    audited[0]["proposal_attempts"] = [
        {
            "round_number": 1,
            "analysis": "分析",
            "candidate": {
                "candidate_id": "draft_invalid",
                "content": "指导",
                "trigger": "条件",
            },
            "atomic_candidates": [{"candidate_id":"candidate_invalid","content":"指导","trigger":"条件"}],
        }
    ]
    audited[0]["application_attempts"] = [
        {
            "round_number": 1,
            "candidate_id": "candidate_invalid",
            "error": "invalid structure",
        }
    ]
    trajectory_file.write_text(
        "\n".join(json.dumps(row) for row in audited) + "\n", encoding="utf-8"
    )
    assert (
        main(["summarize-rewardbench", "--run-dir", str(tmp_path / "experiment")]) == 0
    )
    assert json.loads(capsys.readouterr().out)["invalid_application_attempts"] == 1
    audited = [json.loads(row) for row in trajectories]
    audited[-1]["state"] = "judge_error_initial"
    audited[-1]["initial_judgment"] = None
    audited[-1]["judge_error"] = "numbered criterion mismatch"
    trajectory_file.write_text(
        "\n".join(json.dumps(row) for row in audited) + "\n", encoding="utf-8"
    )
    assert (
        main(["summarize-rewardbench", "--run-dir", str(tmp_path / "experiment")]) == 0
    )
    partial = json.loads(capsys.readouterr().out)
    assert partial["test_total"] == 6 and partial["test_correct"] == 5
    assert partial["test_judge_errors"] == 1 and partial["test_accuracy"] is None
    assert partial["test_accuracy_on_valid"] == 1.0
    trajectory_file.write_text(
        original_trajectories + trajectories[-1] + "\n", encoding="utf-8"
    )
    assert (
        main(["summarize-rewardbench", "--run-dir", str(tmp_path / "experiment")]) == 1
    )
    assert "duplicate trajectory sample_id" in capsys.readouterr().err
    trajectory_file.write_text(original_trajectories, encoding="utf-8")
    test_file = data_dir / "test.jsonl"
    test_file.write_bytes(test_file.read_bytes().splitlines(keepends=True)[0])
    assert (
        main(["summarize-rewardbench", "--run-dir", str(tmp_path / "experiment")]) == 1
    )
    assert "changed since run" in capsys.readouterr().err
    manifest_file = tmp_path / "experiment/run_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert (
        manifest["dataset_sha256"]["test"]
        != hashlib.sha256(test_file.read_bytes()).hexdigest()
    )
    manifest["dataset_sha256"]["test"] = hashlib.sha256(
        test_file.read_bytes()
    ).hexdigest()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        main(["summarize-rewardbench", "--run-dir", str(tmp_path / "experiment")]) == 1
    )
    assert "all six Chat Hard subsets" in capsys.readouterr().err


def test_full_train_keeps_same_prompt_and_globally_orders_all_rows():
    from verge.dataset import full_rewardbench_chat_hard
    rows = synthetic_rows()
    for row in rows:
        row['prompt'] = 'same prompt'
    first = full_rewardbench_chat_hard(rows)
    assert first == full_rewardbench_chat_hard(list(reversed(rows)))
    assert set(first) == {'train'} and len(first['train']) == 24
    assert len({s['sample_id'] for s in first['train']}) == 24
    assert first != full_rewardbench_chat_hard(rows, seed=43)
    with pytest.raises(ValueError, match='duplicate'):
        full_rewardbench_chat_hard(rows + [rows[0]])
    with pytest.raises(ValueError, match='all six'):
        full_rewardbench_chat_hard(rows[:4])
    with pytest.raises(ValueError, match='invalid'):
        full_rewardbench_chat_hard([{**rows[0], 'chosen': ''}] + rows[1:])


def test_full_train_prepare_and_cli_verification(tmp_path, monkeypatch):
    from verge.cli import _verify_prepared_dataset, _ensure_prepared_dataset
    from verge.config import load_config
    import huggingface_hub
    revision = 'a' * 40
    parquet = tmp_path / 'filtered.parquet'
    pq.write_table(pa.Table.from_pylist(synthetic_rows()), parquet)
    monkeypatch.setattr(huggingface_hub.HfApi, 'dataset_info',
                        lambda *a, **k: SimpleNamespace(sha=revision, siblings=[
                            SimpleNamespace(rfilename='data/filtered-00000.parquet')]))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *a, **k: str(parquet))
    directory = tmp_path / 'full'
    provenance = prepare_rewardbench_chat_hard(directory, mode='full_train', revision=revision)
    assert provenance['samples'] == {'train': 24}
    assert not (directory / 'test.jsonl').exists()
    config = yaml.safe_load((PROJECT / 'configs/default.yaml').read_text())
    config['dataset']['config'] = {'paths': {'train': str(directory / 'train.jsonl')},
        'prepare': {'implementation': 'rewardbench_chat_hard', 'mode': 'full_train',
                    'output_dir': str(directory), 'revision': revision, 'seed': 42}}
    path = tmp_path / 'full.yaml'
    path.write_text(yaml.safe_dump(config))
    checked = load_config(path)
    _ensure_prepared_dataset(checked, resuming=False)
    assert _verify_prepared_dataset(checked, JsonlDataset(checked['dataset']['config']['paths']))['mode'] == 'full_train'
    config['dataset']['config']['prepare']['per_subset'] = 2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='split sampling'):
        load_config(path)
    checked['dataset']['config']['prepare']['seed'] = 99
    with pytest.raises(ValueError, match='differs'):
        _verify_prepared_dataset(checked, JsonlDataset(checked['dataset']['config']['paths']))
