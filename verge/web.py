"""Read-only, loopback-only dashboard for completed or in-progress VERGE runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlsplit

from verge.models import Experience, TrajectoryRecord
from verge.prompts import PROMPT_VERSION

ASSETS = Path(__file__).with_name("web_assets")
_MAX_ROLE_IO_BYTES = 1024 * 1024 * 1024  # Do not scan multi-gigabyte artifacts per HTTP request.
_MAX_ROLE_LINE_BYTES = 4 * 1024 * 1024
_MAX_ROLE_ENTRIES = 256
_MAX_ROLE_WARNINGS = 12
_TRANSLATOR_SOURCE = Path(__file__).with_name("apple_translate.swift")
_translator_lock = Lock()
_translator_dir: tempfile.TemporaryDirectory[str] | None = None


class TranslationUnavailable(RuntimeError):
    """Displayable error from the on-device translator; never contains source text."""


def _translate_apple(text: str) -> str:
    """Compile the native bridge once, then translate via Apple without uploading text."""
    global _translator_dir
    version = platform.mac_ver()[0]
    if sys.platform != "darwin" or not version or int(version.split(".")[0]) < 26:
        raise TranslationUnavailable("苹果设备端翻译需要 macOS 26 或更新版本。")
    if not shutil.which("swiftc"):
        raise TranslationUnavailable("未找到 Swift 编译器；请安装 Xcode Command Line Tools。")
    with _translator_lock:
        if _translator_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="verge-apple-translation-")
            binary = Path(temporary.name) / "translate"
            try:
                result = subprocess.run(
                    ["swiftc", "-parse-as-library", "-O", "-o", str(binary), str(_TRANSLATOR_SOURCE)],
                    capture_output=True, timeout=120, check=False,
                )
                if result.returncode:
                    raise TranslationUnavailable("苹果翻译组件编译失败，请检查 Swift/macOS SDK 版本。")
            except (OSError, subprocess.TimeoutExpired) as exc:
                temporary.cleanup()
                raise TranslationUnavailable("无法启动或编译苹果翻译组件。") from exc
            except TranslationUnavailable:
                temporary.cleanup()
                raise
            _translator_dir = temporary
        binary = Path(_translator_dir.name) / "translate"
    try:
        result = subprocess.run([str(binary)], input=text, text=True,
                                capture_output=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranslationUnavailable("苹果翻译服务未响应，请稍后重试。") from exc
    if result.returncode:
        if "LANGUAGE_PACK_NOT_INSTALLED" in result.stderr:
            raise TranslationUnavailable("未安装所需语言包：请在系统设置 → 通用 → 语言与地区 → 翻译语言中下载英语和简体中文。")
        raise TranslationUnavailable("苹果设备端翻译失败，请确认原文语言受支持并已安装相应语言包。")
    return result.stdout.rstrip("\n")


def _dataset_path(value: str, run_dir: Path) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path if path.is_file() else None
    # Dataset paths in manifests are relative to the original working directory.
    for base in (Path.cwd(), Path(__file__).resolve().parents[1], run_dir):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return None


def _sample_context(manifest: dict[str, Any], run_dir: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    config = manifest.get("config", {}).get("dataset", {}).get("config", {})
    paths = config.get("paths", {})
    field_map = config.get("field_map", {})
    names = {key: field_map.get(key, key) for key in ("sample_id", "task", "chosen", "rejected", "metadata")}
    for split in ("train", "test"):
        source = paths.get(split)
        if not isinstance(source, str):
            continue
        file = _dataset_path(source, run_dir)
        if file is None:
            warnings.append(f"{split} 数据文件不可用：仅展示轨迹，无法显示任务原文或确认 split。")
            continue
        digest = manifest.get("dataset_sha256", {}).get(split)
        hasher = sha256()
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        if not isinstance(digest, str) or hasher.hexdigest() != digest:
            warnings.append(f"{split} 数据文件哈希不匹配：已跳过任务原文和 split 标注。")
            continue
        with file.open(encoding="utf-8") as source_file:
            for number, line in enumerate(source_file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                except ValueError:
                    warnings.append(f"{split} 数据文件第 {number} 行无效，已跳过。")
                    continue
                sample_id = row.get(names["sample_id"])
                if isinstance(sample_id, str):
                    if sample_id in result:
                        # A repeated ID cannot be joined safely to a particular trajectory.
                        ambiguous.add(sample_id)
                    else:
                        result[sample_id] = {"split": split, "task": row.get(names["task"]),
                                             "chosen": row.get(names["chosen"]),
                                             "rejected": row.get(names["rejected"]),
                                             "metadata": row.get(names["metadata"], {})}
    for sample_id in ambiguous:
        result.pop(sample_id, None)
    if ambiguous:
        warnings.append(f"{len(ambiguous)} 个重复样本 ID 无法安全关联数据，已隐藏任务正文和 split。")
    return result


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                rows.append(value)
            except ValueError:
                warnings.append(f"{path.name} 第 {number} 行不完整或无效，已跳过。")
    return rows


def _evolution(experience: dict[str, Any], records: list[dict[str, Any]],
               order: dict[str, int], warnings: list[str]) -> list[dict[str, Any]]:
    """Replay canonical evidence; add zero-credit fusion events absent from evidence_history."""
    exp_id = experience["experience_id"]
    fusion_events = []
    for record in records:
        for candidate in record.get("reflected_candidates", []):
            if candidate.get("experience_id") == exp_id and candidate.get("fusion") in ("ADD", "MERGE"):
                fusion_events.append({"sample_id": record["sample_id"], "fusion": candidate["fusion"],
                                      "candidate_id": candidate["candidate_id"], "credit": candidate["credit"],
                                      "candidate_content": candidate["content"],
                                      "candidate_trigger": candidate["trigger"],
                                      "fusion_record": candidate.get("fusion_record")})
    # Memory may have a committed fusion before the complete trajectory is appended.
    for history in experience.get("fusion_history", []):
        key = (history["sample_id"], history["candidate_id"])
        match = next((ev for ev in fusion_events if (ev["sample_id"], ev["candidate_id"]) == key), None)
        if match is None:
            fusion_events.append({"sample_id": history["sample_id"], "candidate_id": history["candidate_id"],
                                  "fusion": history["fusion"], "credit": None, "fusion_record": history})
        else:
            match["fusion_record"] = history
    used: set[int] = set()
    events = []
    for index, evidence in enumerate(experience["evidence_history"]):
        item = dict(evidence)
        item["kind"] = evidence["evidence_type"]
        item["_sequence"] = index
        if item["kind"] == "fusion_credit":
            matches = [n for n, fusion in enumerate(fusion_events)
                       if n not in used and fusion["sample_id"] == item["sample_id"]
                       and fusion["candidate_id"] == item.get("candidate_id")]
            if len(matches) == 1:
                n = matches[0]
                item.update({key: value for key, value in fusion_events[n].items() if key != "credit"})
                used.add(n)
            else:
                warnings.append(f"经验 {exp_id} 的融合证据缺少唯一文本版本。")
        elif item["kind"] == "revision_credit":
            matches = [n for n, fusion in enumerate(fusion_events)
                       if n not in used and fusion["sample_id"] == item["sample_id"]
                       and fusion["credit"] is not None
                       and abs(fusion["credit"] - item["credit"]) < 1e-8]
            if len(matches) == 1:
                n = matches[0]
                item.update({key: value for key, value in fusion_events[n].items() if key != "credit"})
                used.add(n)
            elif len(matches) > 1:
                warnings.append(f"经验 {exp_id} 在样本 {item['sample_id']} 中有同额候选，无法唯一绑定证据。")
        events.append(item)
    for n, fusion in enumerate(fusion_events):
        if n not in used:
            events.append({**fusion, "kind": "fusion", "evidence_weight": 0,
                           "_sequence": len(events) + n})
    events.sort(key=lambda event: (order.get(event["sample_id"], len(order)),
                                   0 if event["kind"] == "downstream_credit" else 1,
                                   event["_sequence"]))
    alpha = beta = 1.0
    # Beta(1,1) has no evidence yet, but is instantiated when the first ADD
    # admits the experience. Prefer that explicit origin over the source list.
    add_origin = next((item["sample_id"] for item in fusion_events if item["fusion"] == "ADD"), None)
    source_origin = next(iter(experience.get("source_trajectories", [])), None)
    origin = add_origin or source_origin
    timeline = [{"kind": "prior", "alpha": alpha, "beta": beta, "confidence": 0.5,
                 "sample_id": origin, "origin_kind": "add" if add_origin else "source" if origin else None,
                 "has_trajectory": origin in order if origin else False}]
    for event in events:
        credit = event.get("credit", 0) or 0
        change = abs(credit) * event.get("evidence_weight", 1)
        if event["kind"] != "fusion":
            if credit > 0:
                alpha += change
            elif credit < 0:
                beta += change
        timeline.append({key: value for key, value in event.items() if not key.startswith("_")}
                        | {"alpha": alpha, "beta": beta, "confidence": alpha / (alpha + beta),
                           "has_trajectory": event["sample_id"] in order})
    if abs(alpha - experience["alpha"]) > 1e-7 or abs(beta - experience["beta"]) > 1e-7:
        warnings.append(f"经验 {exp_id} 的证据与最终 α/β 不一致；图中仅展示可重建的证据。")
    return timeline


def _run_artifacts(directory: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported run manifest format")
    versions = manifest.get("prompt_versions")
    if not isinstance(versions, dict) or not versions or set(versions.values()) not in (
            {PROMPT_VERSION}, {"generator-dialogue-initial-design-experience-v2"},
            {"generator-dialogue-initial-design-experience"},
            {"generator-dialogue-task-grounded-experience-compact"},
            {"generator-dialogue-task-grounded-experience"},
            {"generator-dialogue-fusion-revision-evidence"},
            {"generator-dialogue-critic-feedback"}):
        raise ValueError("Web only supports runs using the current prompt version or known previous dialogue versions")
    root_name = Path(manifest["config"]["store"]["config"]["root_dir"])
    if root_name.is_absolute() or ".." in root_name.parts or root_name == Path("."):
        raise ValueError("invalid store root_dir in manifest")
    return manifest, directory / root_name


def load_role_io(run_dir: str | Path, sample_id: str) -> dict[str, Any]:
    """Lazy-load only the selected sample's original role calls, not the whole run into the page."""
    directory = Path(run_dir).expanduser().resolve()
    _, root = _run_artifacts(directory)
    path = root / "role_io.jsonl"
    if not path.is_file():
        return {"available": False, "entries": [], "warnings": []}
    if path.stat().st_size > _MAX_ROLE_IO_BYTES:
        return {"available": True, "entries": [], "warnings": [],
                "error": "角色原始日志超过 1 GiB，请在本机按样本筛选或归档后查看。"}
    warnings: list[str] = []
    entries: list[dict[str, Any]] = []
    bad_lines = 0
    skipped_entries = 0
    with path.open("rb") as source:
        number = 0
        while line := source.readline(_MAX_ROLE_LINE_BYTES + 1):
            number += 1
            if len(line) > _MAX_ROLE_LINE_BYTES:
                # Skip the rest of this single oversized JSONL record without reading it all.
                while line and not line.endswith(b"\n"):
                    line = source.readline(_MAX_ROLE_LINE_BYTES + 1)
                bad_lines += 1
                continue
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("expected object")
            except (ValueError, UnicodeDecodeError):
                bad_lines += 1
                if len(warnings) < _MAX_ROLE_WARNINGS:
                    warnings.append(f"role_io.jsonl 第 {number} 行不完整或无效，已跳过。")
                continue
            if item.get("sample_id") == sample_id:
                if len(entries) < _MAX_ROLE_ENTRIES:
                    entries.append(item)
                else:
                    skipped_entries += 1
    if bad_lines > len(warnings):
        warnings.append(f"另有 {bad_lines-len(warnings)} 行超长或损坏，已跳过。")
    if skipped_entries:
        warnings.append(f"该样本仅展示前 {_MAX_ROLE_ENTRIES} 次调用，另有 {skipped_entries} 条未加载。")
    # Do not forward unknown fields from external artifacts (e.g. accidentally saved headers).
    allowed = ("role", "backend", "model", "prompt_version", "attempt", "status",
               "input", "messages", "raw_output", "output", "error")
    return {"available": True, "entries": [{key: item[key] for key in allowed if key in item}
                                           for item in entries if isinstance(item.get("role"), str)],
            "warnings": warnings}


def _overview_metrics(samples: list[dict[str, Any]], experiences: list[dict[str, Any]],
                      context: dict[str, dict[str, Any]], manifest: dict[str, Any],
                      warnings: list[str]) -> dict[str, Any]:
    """只读复算完整训练轨迹；使用前 Beta 不包含当前任务证据。"""
    from verge.calculator import calculate_existing_experience_credit
    from verge.models import Rubric, RubricJudgment

    train = [s for s in samples if s.get("split") == "train" and s["state"] != "interrupted"]
    ids = [s["sample_id"] for s in train]
    known = set(ids)
    pool: dict[str, list[float]] = {}
    evidence: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    replay_ok = len(known) == len(ids)
    adds = {c["experience_id"] for s in train for c in s.get("reflected_candidates", [])
            if c.get("fusion") == "ADD"}
    if adds != {e["experience_id"] for e in experiences}:
        replay_ok = False
    for e in experiences:
        alpha = beta = 1.0
        for ev in e["evidence_history"]:
            if ev["sample_id"] not in known:
                replay_ok = False
            evidence[ev["sample_id"]].append((e["experience_id"], ev))
            change = abs(ev["credit"]) * ev["evidence_weight"]
            alpha += change if ev["credit"] > 0 else 0
            beta += change if ev["credit"] < 0 else 0
        if abs(alpha - e["alpha"]) > 1e-7 or abs(beta - e["beta"]) > 1e-7:
            replay_ok = False
    # 初始判断的缺失包括 initial Judge 错误；修订 Judge 错误仍有有效 R0。
    def rate(rows):
        valid = [r for r in rows if r.get("initial_judgment") is not None]
        correct = sum(r["initial_judgment"]["success"] for r in valid)
        return {"total": len(rows), "valid": len(valid), "correct": correct,
                "errors": len(rows) - len(valid),
                "accuracy": correct / len(valid) if valid else None}

    usage = {e["experience_id"]: {"experience_id": e["experience_id"], "recalled": 0,
        "selected": 0, "referenced": 0, "positive": 0, "negative": 0, "zero": 0}
        for e in experiences}
    bins = [{"label": f"{i / 10:.1f}–{(i + 1) / 10:.1f}", "uses": 0,
             "positive": 0, "negative": 0, "zero": 0, "credit_sum": 0.0} for i in range(10)]
    scale = [{"label": label, "tasks": 0, "recalled": 0, "selected": 0, "referenced": 0,
              "requests": 0, "tokens": 0, "token_reports": 0}
             for label in ("0", "1–20", "21–50", "51–100", "101+")]
    curve, merges = [], []
    add_count = merge_count = candidate_count = passed = failed = 0
    for index, s in enumerate(train):
        before = len(pool)
        recalled = set(s.get("retrieved_experience_ids", []))
        selected = set(s.get("selected_experience_ids", []))
        referenced = {eid for c in (s.get("initial_rubric") or {}).get("criteria", []) for eid in c["source_experiences"]}
        for field, keys in (("recalled", recalled), ("selected", selected), ("referenced", referenced)):
            for eid in keys:
                if eid in usage:
                    usage[eid][field] += 1
        credits = (calculate_existing_experience_credit(Rubric.from_dict(s["initial_rubric"]),
                    RubricJudgment.from_dict(s["initial_judgment"]))
                   if s.get("initial_judgment") else {})
        if replay_ok and s.get("initial_judgment"):
            for eid in selected:
                if eid not in pool:
                    replay_ok = False
                    break
                alpha, beta = pool[eid]
                bucket = bins[min(9, int(10 * alpha / (alpha + beta)))]
                credit = credits.get(eid, 0.0)
                kind = "positive" if credit > 0 else "negative" if credit < 0 else "zero"
                bucket["uses"] += 1
                bucket[kind] += 1
                bucket["credit_sum"] += credit
                if eid in usage:
                    usage[eid][kind] += 1
        bucket = scale[0 if before == 0 else 1 if before <= 20 else 2 if before <= 50
                       else 3 if before <= 100 else 4]
        bucket["tasks"] += 1
        bucket["recalled"] += len(recalled)
        bucket["selected"] += len(selected)
        bucket["referenced"] += len(selected & referenced)
        bucket["requests"] += len(s.get("requests", []))
        for request in s.get("requests", []):
            total = request.get("usage", {}).get("total_tokens")
            if type(total) is int:
                bucket["tokens"] += total
                bucket["token_reports"] += 1
        candidates = s.get("reflected_candidates", [])
        candidate_count += len(candidates)
        passed += sum(c["admission"] == "PASS" for c in candidates)
        failed += sum(c["admission"] == "FAIL" for c in candidates)
        added_here = set()
        for c in candidates:
            eid = c.get("experience_id")
            if c.get("fusion") == "ADD":
                pool[eid] = [1.0, 1.0]
                added_here.add(eid)
                add_count += 1
            elif c.get("fusion") == "MERGE":
                merge_count += 1
                if eid not in pool:
                    replay_ok = False
                merges.append({"sample_id": s["sample_id"], "candidate_id": c["candidate_id"],
                    "content": c["content"], "trigger": c["trigger"], "experience_id": eid,
                    "credit": c["credit"], "same_task": eid in added_here})
        for eid, ev in evidence[s["sample_id"]]:
            if eid not in pool:
                replay_ok = False
                continue
            change = abs(ev["credit"]) * ev["evidence_weight"]
            pool[eid][0 if ev["credit"] > 0 else 1] += change
        curve.append({"index": index + 1, "pool": len(pool), "adds": add_count, "merges": merge_count,
                      "cumulative": rate(train[:index + 1]), "rolling": rate(train[max(0, index - 49):index + 1])})
    if not replay_ok and train:
        warnings.append("经验或证据无法与完整训练轨迹对齐：使用前置信度和池规模分桶不可用；池曲线仅累计已记录的融合事件。")
    segments = [{"start": i + 1, "end": min(i + 50, len(train)), **rate(train[i:i + 50]),
                 "subsets": dict(Counter(r.get("metadata", {}).get("subset", "unknown")
                                         for r in train[i:i + 50]))} for i in range(0, len(train), 50)]
    subsets = {name: rate([s for s in train if s.get("metadata", {}).get("subset", "unknown") == name])
               for name in sorted({s.get("metadata", {}).get("subset", "unknown") for s in train})}
    for b in bins:
        b["mean_credit"] = b["credit_sum"] / b["uses"] if b["uses"] else None
    for b in scale:
        b["selection_rate"] = b["selected"] / b["recalled"] if b["recalled"] else None
        b["reference_rate"] = b["referenced"] / b["selected"] if b["selected"] else None
    initial_failures = sum(s.get("initial_judgment") is not None and
                           not s["initial_judgment"]["success"] for s in train)
    first = sum(any(a["attempt_number"] == 1 and a["judgment"]["success"]
                    for a in s.get("revision_attempts", [])) for s in train)
    second = sum(any(a["attempt_number"] == 2 and a["judgment"]["success"]
                     for a in s.get("revision_attempts", [])) for s in train)
    attempted_second = sum(any(a["attempt_number"] == 2 for a in
                              s.get("revision_attempts", []) + s.get("invalid_revision_attempts", []))
                           for s in train)
    return {"mode": "train_test" if "test" in manifest["config"]["dataset"]["config"]["paths"] else "train_only",
            "train_total": sum(c.get("split") == "train" for c in context.values()),
            "train_completed": len(train), "sample_errors": sum(bool(s.get("sample_error")) for s in train),
            "initial": rate(train), "curve": curve,
            "segments": segments, "subsets": subsets, "adds": add_count, "merges": merge_count,
            "candidates": candidate_count, "admission_pass": passed,
            "admission_fail": failed, "merge_rate": merge_count / (add_count + merge_count)
                if add_count + merge_count else None,
            "merge_events": merges, "confidence_bins": bins if replay_ok else None,
            "scale_bins": scale if replay_ok else None, "usage": list(usage.values()),
            "replay_available": replay_ok, "initial_failures": initial_failures,
            "repair_successes": first + second,
            "repair_rate": (first + second) / initial_failures if initial_failures else None,
            "first_rate": first / initial_failures if initial_failures else None,
            "second_successes": second, "second_attempted": attempted_second,
            "second_rate": second / attempted_second if attempted_second else None}


def _generation_summaries(root: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Stream only Generator analysis and Reflector counts; keep full calls lazy-loaded."""
    path = root / "role_io.jsonl"
    summaries: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return summaries
    if path.stat().st_size > _MAX_ROLE_IO_BYTES:
        warnings.append("角色日志超过 1 GiB，生成分析与原始候选数未加载。")
        return summaries
    bad = 0
    with path.open("rb") as source:
        while line := source.readline(_MAX_ROLE_LINE_BYTES + 1):
            if len(line) > _MAX_ROLE_LINE_BYTES:
                while line and not line.endswith(b"\n"):
                    line = source.readline(_MAX_ROLE_LINE_BYTES + 1)
                bad += 1
                continue
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                bad += 1
                continue
            if not isinstance(entry, dict):
                bad += 1
                continue
            sid, output = entry.get("sample_id"), entry.get("output")
            if not isinstance(sid, str) or not isinstance(output, dict) or entry.get("error"):
                continue
            if (entry.get("role") == "initial_generator"
                    and not (entry.get("input") or {}).get("revision_number")):
                analysis = output.get("analysis")
                # The latest structured generation supersedes any previous retry's analysis.
                summaries.setdefault(sid, {})["generator_analysis"] = (
                    analysis if isinstance(analysis, str) and analysis.strip() else None)
            elif entry.get("role") == "reflector" and isinstance(output.get("candidates"), list):
                summaries.setdefault(sid, {})["generated_candidates"] = len(output["candidates"])
    if bad:
        warnings.append(f"角色摘要跳过 {bad} 条损坏或超长记录；部分生成状态可能不可确认。")
    return summaries


def _sample_experience_status(sample: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    candidates = sample.get("reflected_candidates", [])
    referenced = {eid for c in (sample.get("initial_rubric") or {}).get("criteria", [])
                  for eid in c.get("source_experiences", [])}
    generated = summary.get("generated_candidates")
    if generated is None and sample.get("state") in ("initial_success", "initial_failure"):
        generated = 0  # Completed without the reflection branch.
    return {"generated_candidates": generated,
            "added": sum(c.get("fusion") == "ADD" for c in candidates),
            "merged": sum(c.get("fusion") == "MERGE" for c in candidates),
            "selected": len(set(sample.get("selected_experience_ids", []))),
            "referenced": len(referenced) if sample.get("initial_rubric") else None}


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load persisted artifacts without changing the run; tolerate truncated JSONL lines."""
    directory = Path(run_dir).expanduser().resolve()
    manifest, root = _run_artifacts(directory)
    warnings: list[str] = []
    context = _sample_context(manifest, directory, warnings)
    raw_records = _read_jsonl(root / "trajectories.jsonl", warnings)
    records = [TrajectoryRecord.from_dict(item).to_dict() for item in raw_records]
    # A sample ID can occur more than once in a damaged/replayed log; retain all entries.
    order = {item["sample_id"]: index for index, item in enumerate(records)}
    if len(order) != len(records):
        warnings.append("轨迹含重复样本 ID：事件时间线只能近似按样本排序，部分数据关联不确定。")
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _read_jsonl(root / "llm_requests.jsonl", warnings):
        sample_id = item.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            requests[sample_id].append({key: item[key] for key in (
                "role", "model", "status", "attempt", "finish_reason", "usage", "split",
            ) if key in item})
    summaries = _generation_summaries(root, warnings)
    samples = []
    for item in records:
        sample_id = item["sample_id"]
        samples.append({**item, **context.get(sample_id, {}), "requests": requests.get(sample_id, [])})
    for sample_id, calls in requests.items():
        if sample_id not in order:
            samples.append({"sample_id": sample_id, "state": "interrupted",
                            **context.get(sample_id, {}), "requests": calls})
    for sample in samples:
        summary = summaries.get(sample["sample_id"], {})
        sample["generator_analysis"] = summary.get("generator_analysis")
        sample["experience_status"] = _sample_experience_status(sample, summary)
    memory = json.loads((root / "experience_memory.json").read_text(encoding="utf-8"))
    experiences = [Experience.from_dict(item).to_dict() for item in memory["experiences"]]
    for item in experiences:
        item["evolution"] = _evolution(item, records, order, warnings)
    overview = _overview_metrics(samples, experiences, context, manifest, warnings)
    counts = Counter(item["state"] for item in samples)
    return {"run_name": directory.name, "run_dir": str(directory), "code_version": manifest.get("code_version"),
            "resumed_from": manifest.get("resumed_from", {}).get("source_run"),
            "samples": samples, "experiences": experiences, "counts": dict(counts),
            "warnings": warnings, "overview": overview}


def discover_runs(root: Path) -> list[str]:
    """Only immediate, non-symlink children containing a run manifest are selectable."""
    return sorted(child.name for child in root.iterdir()
                  if child.is_dir() and not child.is_symlink()
                  and (child / "run_manifest.json").is_file()
                  and not (child / "run_manifest.json").is_symlink())


class DashboardHandler(BaseHTTPRequestHandler):
    run_dir: Path
    runs_root: Path | None = None

    def do_POST(self) -> None:
        """Translate only a verified sample field; do not accept arbitrary text from the browser."""
        host = self.headers.get("Host", "").split(":")[0]
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1", "localhost") or (origin and origin != f"http://{self.headers['Host']}"):
            self.send_error(403, "invalid origin")
            return
        if urlsplit(self.path).path != "/api/translate" or urlsplit(self.path).query:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if (self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json"
                or not 0 < size <= 4096):
            self._json_error(400, "请求格式无效。")
            return
        try:
            request = json.loads(self.rfile.read(size))
        except (UnicodeError, ValueError):
            self._json_error(400, "请求 JSON 无效。")
            return
        if not isinstance(request, dict) or set(request) != {"run", "sample_id", "field"}:
            self._json_error(400, "请求字段无效。")
            return
        name, sample_id, field = (request.get(key) for key in ("run", "sample_id", "field"))
        root = self.runs_root or self.run_dir.parent
        if (not isinstance(name, str) or name not in discover_runs(root)
                or not isinstance(sample_id, str) or field not in ("task", "chosen", "rejected")):
            self._json_error(404, "未找到运行或样本。")
            return
        try:
            sample = next((item for item in load_run(root / name)["samples"]
                           if item["sample_id"] == sample_id), None)
            text = sample.get(field) if sample else None
            if not isinstance(text, str) or not text.strip():
                self._json_error(404, "该样本原文不可用或尚未验证。")
                return
            if len(text) > 20000:
                self._json_error(413, "原文超过单次翻译的 20000 字符上限。")
                return
            translated = _translate_apple(text)
        except TranslationUnavailable as exc:
            self._json_error(503, str(exc))
            return
        except (OSError, ValueError, KeyError, TypeError):
            self._json_error(500, "运行数据暂时无法读取。")
            return
        body = json.dumps({"translation": translated}, ensure_ascii=False).encode("utf-8")
        self._send(200, "application/json; charset=utf-8", body)

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:
        host = self.headers.get("Host", "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403, "invalid host")
            return
        parsed = urlsplit(self.path)
        route = parsed.path
        root = self.runs_root or self.run_dir.parent
        if route == "/api/role-io":
            values = parse_qs(parsed.query, keep_blank_values=True)
            if (set(values) != {"name", "sample_id"}
                    or any(len(values[key]) != 1 for key in values)):
                self._json_error(400, "需要唯一的运行名称与样本 ID。")
                return
            name, sample_id = values["name"][0], values["sample_id"][0]
            if name not in discover_runs(root) or not sample_id or len(sample_id) > 512:
                self._json_error(404, "未找到运行或样本。")
                return
            try:
                directory = root / name
                if not any(item["sample_id"] == sample_id for item in load_run(directory)["samples"]):
                    self._json_error(404, "未找到该样本。")
                    return
                result = load_role_io(directory, sample_id)
            except (OSError, ValueError, KeyError, TypeError):
                self._json_error(500, "运行数据暂时无法读取。")
                return
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        elif route == "/api/runs":
            names = discover_runs(root)
            self._send(200, "application/json; charset=utf-8",
                       json.dumps({"runs": names, "default": self.run_dir.name},
                                  ensure_ascii=False).encode("utf-8"))
        elif route == "/api/run":
            values = parse_qs(parsed.query).get("name", [])
            if len(values) > 1:
                self.send_error(400, "only one run name allowed")
                return
            name = values[0] if values else self.run_dir.name
            if name not in discover_runs(root):
                self.send_error(404, "run not found")
                return
            try:
                data = json.dumps(load_run(root / name), ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                data = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(500, "application/json; charset=utf-8", data)
                return
            self._send(200, "application/json; charset=utf-8", data)
        elif route == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        elif route in ("/", "/index.html", "/app.js", "/style.css"):
            name = "index.html" if route == "/" else route[1:]
            content_type = ("text/html" if name.endswith(".html") else
                            "text/javascript" if name.endswith(".js") else "text/css")
            self._send(200, f"{content_type}; charset=utf-8", (ASSETS / name).read_bytes())
        else:
            self.send_error(404)

    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only VERGE run dashboard")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="initial run; also discover sibling runs")
    source.add_argument("--runs-dir", help="directory containing multiple run directories")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else Path(args.run_dir).expanduser().resolve().parent
    if not root.is_dir():
        parser.error(f"runs directory does not exist: {root}")
    names = discover_runs(root)
    directory = (Path(args.run_dir).expanduser().resolve() if args.run_dir
                 else root / (names[0] if names else "__pending__"))
    # Fail early, before accepting requests; --run-dir must be a direct child.
    if args.run_dir and (directory.parent != root or directory.name not in names):
        parser.error("--run-dir must contain a manifest and be an immediate child of the runs directory")
    if names:
        load_run(directory)
    handler = type("RunDashboardHandler", (DashboardHandler,), {"run_dir": directory, "runs_root": root})
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"VERGE dashboard: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
