"""Independent initial-only baseline and read-only paired comparison.

Run explicitly with ``python -m verge.baseline run``. Importing this module or
running ``compare`` never starts model requests or changes the source run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from threading import Lock
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from verge.cli import _backend, _code_version, _Progress
from verge.dataset import JsonlDataset
from verge.llm import OpenAICompatibleBackend, LLMRefusalError
from verge.memory import TrajectoryLog
from verge.models import TrajectoryRecord
from verge.modules import JudgeOutputError, LLMInitialRubricGenerator, LLMJudge
from verge.prompts import PROMPT_VERSION


ROLES = ("initial_generator", "judge")
PROJECT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(directory: str | Path) -> tuple[Path, dict[str, Any], JsonlDataset]:
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported source manifest format")
    config = manifest["config"]["dataset"]
    if config["implementation"] != "jsonl":
        raise ValueError("baseline requires a JSONL source dataset")
    original = Path(config["config"]["paths"]["train"])
    candidates = [original] if original.is_absolute() else [
        Path.cwd() / original, PROJECT / original, directory / original,
    ]
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None or _sha(path) != manifest.get("dataset_sha256", {}).get("train"):
        raise ValueError("source train dataset is missing or has changed")
    dataset = JsonlDataset({"train": path}, config["config"].get("field_map"))
    return directory, manifest, dataset


def _root(directory: Path, manifest: dict[str, Any]) -> Path:
    relative = Path(manifest["config"]["store"]["config"].get("root_dir", "artifacts"))
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("invalid store root_dir")
    return directory / relative


def run_baseline(source_run: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Generate and judge once per source training task; no memory/module pipeline."""
    source, original, dataset = _source(source_run)
    output = Path(output_dir).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("baseline output must be separate from the source run")
    if output.exists():
        raise ValueError("baseline output already exists; use a fresh directory")
    for role in ROLES:
        if original.get("prompt_versions", {}).get(role) != PROMPT_VERSION:
            raise ValueError(f"source {role} prompt version differs from current code")
    # Validate the entire input before creating output or issuing requests.
    samples = list(dataset.load("train"))
    config = deepcopy(original["config"])
    config["run"] = {**config["run"], "output_dir": str(output), "test_workers": 1}
    config["dataset"]["config"]["paths"] = {"train": str(dataset.paths["train"])}
    # Data preparation belongs to the source. This entry never downloads or rebuilds it.
    config["dataset"]["config"].pop("prepare", None)
    config["models"]["roles"] = {
        role: config["models"].get("roles", {})[role]
        for role in ROLES if role in config["models"].get("roles", {})
    }
    root = _root(output, {"config": config})
    retries = original.get("resumed_from", {}).get("initial_output_retries", 0)
    # Resolve only active roles: never instantiate retriever/embedding/Selector/Critic.
    backends: dict[str, Any] = {}
    try:
        for role in ROLES:
            spec = config["models"]["roles"].get(role, config["models"]["default"])
            backends[role] = _backend(spec, f"models.{role}")
        generator = LLMInitialRubricGenerator(backends["initial_generator"],
                                              invalid_output_retries=retries)
        judge = LLMJudge(backends["judge"])
        manifest = {
            "format_version": 1, "config": config, "code_version": _code_version(),
            "execution": {"mode": "initial_only_baseline", "test_workers": 1,
                          "roles": list(ROLES), "initial_output_retries": retries},
            "dataset_version": original.get("dataset_version"),
            "dataset_sha256": {"train": original["dataset_sha256"]["train"]},
            "prompt_versions": {role: PROMPT_VERSION for role in ROLES},
            "baseline_source": {"run_dir": str(source),
                                "manifest_sha256": _sha(source / "run_manifest.json"),
                                "code_version": original.get("code_version")},
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8")
        root.mkdir(parents=True, exist_ok=False)
        # Static empty view for existing read-only dashboard, not an ExperienceMemory instance.
        (root / "experience_memory.json").write_text('{"experiences": []}\n', encoding="utf-8")
        role_file = root / "role_io.jsonl"
        role_file.touch(mode=0o600)
        role_file.chmod(0o600)
        context: dict[str, str] = {}
        log_lock = Lock()

        def save_role(event: dict[str, Any]) -> None:
            with log_lock, role_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**context, **event}, ensure_ascii=False,
                                        allow_nan=False) + "\n")

        def save_request(event: dict[str, Any]) -> None:
            with log_lock, (root / "llm_requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**context, "backend": "openai_compatible", **event},
                                        ensure_ascii=False, allow_nan=False) + "\n")

        for backend in backends.values():
            backend.trace_hook = save_role
            if isinstance(backend, OpenAICompatibleBackend):
                backend.metadata_hook = save_request
        random.seed(config["run"]["seed"])
        log = TrajectoryLog(root)
        correct = valid = errors = sample_errors = 0
        progress = _Progress("baseline", len(samples))
        try:
            for sample in samples:
                context.update(sample_id=sample.sample_id, split="train")
                record = TrajectoryRecord(sample.sample_id, "in_progress", [], [],
                                          None, None, [], [], {})
                stage = "initial_generator"
                try:
                    record.initial_rubric = generator.generate(sample.task, [])
                    stage = "initial_judge"
                    record.initial_judgment = judge.judge(record.initial_rubric, sample.chosen,
                                                         sample.rejected, task=sample.task)
                except JudgeOutputError as exc:
                    record.state = "judge_error_initial"
                    record.judge_error = str(exc)
                    errors += 1
                except (ValueError, LLMRefusalError) as exc:
                    record.state = "sample_error"
                    record.sample_error = {"stage": stage, "type": type(exc).__name__,
                                           "message": str(exc)[:500] or type(exc).__name__}
                    sample_errors += 1
                else:
                    valid += 1
                    correct += int(record.initial_judgment.success)
                    record.state = "initial_success" if record.initial_judgment.success else "initial_failure"
                log.append(record)
                progress.advance()
        finally:
            progress.close()
        result = {"run_dir": str(output), "samples": len(samples), "valid": valid,
                  "correct": correct, "judge_errors": errors, "sample_errors": sample_errors,
                  "valid_accuracy": correct / valid if valid else None,
                  "accuracy": correct / len(samples) if samples and not (errors + sample_errors) else None}
        (output / "baseline_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        for backend in backends.values():
            if isinstance(backend, OpenAICompatibleBackend):
                backend.close()


def _records(directory: Path, manifest: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    """Accept a committed ordered prefix; ignore only an unfinished final line."""
    path = _root(directory, manifest) / "trajectories.jsonl"
    if not path.exists():
        return {}
    records = {}
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            break
        record = TrajectoryRecord.from_json(line.decode("utf-8"))
        if index >= len(ids) or record.sample_id != ids[index]:
            raise ValueError("trajectories are not an ordered training prefix")
        records[record.sample_id] = record
    return records


def compare_runs(source_run: str | Path, baseline_run: str | Path) -> dict[str, Any]:
    """Compare valid initial judgments by sample_id; incomplete runs are provisional."""
    source, main_manifest, dataset = _source(source_run)
    baseline, base_manifest, base_dataset = _source(baseline_run)
    if base_manifest.get("execution", {}).get("mode") != "initial_only_baseline":
        raise ValueError("comparison requires an initial-only baseline")
    provenance = base_manifest.get("baseline_source", {})
    if (provenance.get("run_dir") != str(source)
            or provenance.get("manifest_sha256") != _sha(source / "run_manifest.json")
            or base_manifest["dataset_sha256"]["train"] != main_manifest["dataset_sha256"]["train"]):
        raise ValueError("baseline does not match this source run")
    samples = list(dataset.load("train"))
    ids = [s.sample_id for s in samples]
    if ids != [s.sample_id for s in base_dataset.load("train")]:
        raise ValueError("baseline task IDs or order differ")
    main = _records(source, main_manifest, ids)
    base = _records(baseline, base_manifest, ids)

    def counts(keys: list[str]) -> dict[str, Any]:
        result = {"tasks": len(keys), "paired_valid": 0, "both_correct": 0,
                  "experience_only_correct": 0, "baseline_only_correct": 0, "both_wrong": 0}
        for key in keys:
            if key not in main or key not in base:
                continue
            a, b = main[key].initial_judgment, base[key].initial_judgment
            if a is None or b is None:
                continue
            result["paired_valid"] += 1
            category = ("both_correct" if a.success and b.success else
                        "experience_only_correct" if a.success else
                        "baseline_only_correct" if b.success else "both_wrong")
            result[category] += 1
        n = result["paired_valid"]
        result["excluded_or_pending"] = len(keys) - n
        result["net_accuracy_gain"] = ((result["experience_only_correct"] -
                                        result["baseline_only_correct"]) / n if n else None)
        result["experience_accuracy"] = ((result["both_correct"] +
                                           result["experience_only_correct"]) / n if n else None)
        result["baseline_accuracy"] = ((result["both_correct"] +
                                        result["baseline_only_correct"]) / n if n else None)
        return result

    result = counts(ids)
    result.update(source_completed=len(main), baseline_completed=len(base),
                  provisional=len(main) != len(ids) or len(base) != len(ids),
                  source_initial_errors=sum(r.initial_judgment is None for r in main.values()),
                  baseline_initial_errors=sum(r.initial_judgment is None for r in base.values()))
    result["segments"] = [{"start": start + 1, "end": min(start + 50, len(ids)),
                           **counts(ids[start:start + 50])} for start in range(0, len(ids), 50)]
    subsets = sorted({str(s.metadata.get("subset", "unknown")) for s in samples})
    result["by_subset"] = {subset: counts([s.sample_id for s in samples
                                          if str(s.metadata.get("subset", "unknown")) == subset])
                           for subset in subsets}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="starts model requests only when explicitly invoked")
    runner.add_argument("--source-run", required=True)
    runner.add_argument("--output-dir", required=True)
    compare = commands.add_parser("compare", help="read-only; no model requests")
    compare.add_argument("--source-run", required=True)
    compare.add_argument("--baseline-run", required=True)
    args = parser.parse_args()
    try:
        result = (run_baseline(args.source_run, args.output_dir) if args.command == "run" else
                  compare_runs(args.source_run, args.baseline_run))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"baseline: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
