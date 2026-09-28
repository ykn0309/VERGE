"""Read-only, loopback-only dashboard for completed or in-progress VERGE runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from verge.models import Experience, TrajectoryRecord

ASSETS = Path(__file__).with_name("web_assets")


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
                                      "candidate_trigger": candidate["trigger"]})
    used: set[int] = set()
    events = []
    for index, evidence in enumerate(experience["evidence_history"]):
        item = dict(evidence)
        item["kind"] = evidence["evidence_type"]
        item["_sequence"] = index
        if item["kind"] == "revision_credit":
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
    timeline = [{"kind": "prior", "alpha": alpha, "beta": beta, "confidence": 0.5}]
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


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load persisted artifacts without changing the run; tolerate truncated JSONL lines."""
    directory = Path(run_dir).expanduser().resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported run manifest format")
    root_name = Path(manifest["config"]["store"]["config"].get("root_dir", "artifacts"))
    if root_name.is_absolute() or ".." in root_name.parts or root_name == Path("."):
        raise ValueError("invalid store root_dir in manifest")
    root = directory / root_name
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
    samples = []
    for item in records:
        sample_id = item["sample_id"]
        samples.append({**item, **context.get(sample_id, {}), "requests": requests.get(sample_id, [])})
    for sample_id, calls in requests.items():
        if sample_id not in order:
            samples.append({"sample_id": sample_id, "state": "interrupted",
                            **context.get(sample_id, {}), "requests": calls})
    memory = json.loads((root / "experience_memory.json").read_text(encoding="utf-8"))
    experiences = [Experience.from_dict(item).to_dict() for item in memory["experiences"]]
    for item in experiences:
        item["evolution"] = _evolution(item, records, order, warnings)
    counts = Counter(item["state"] for item in samples)
    return {"run_name": directory.name, "run_dir": str(directory), "code_version": manifest.get("code_version"),
            "resumed_from": manifest.get("resumed_from", {}).get("source_run"),
            "samples": samples, "experiences": experiences, "counts": dict(counts),
            "warnings": warnings}


def discover_runs(root: Path) -> list[str]:
    """Only immediate, non-symlink children containing a run manifest are selectable."""
    return sorted(child.name for child in root.iterdir()
                  if child.is_dir() and not child.is_symlink()
                  and (child / "run_manifest.json").is_file()
                  and not (child / "run_manifest.json").is_symlink())


class DashboardHandler(BaseHTTPRequestHandler):
    run_dir: Path
    runs_root: Path | None = None

    def do_GET(self) -> None:
        host = self.headers.get("Host", "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403, "invalid host")
            return
        parsed = urlsplit(self.path)
        route = parsed.path
        root = self.runs_root or self.run_dir.parent
        if route == "/api/runs":
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
    if not names:
        parser.error(f"no runs with run_manifest.json found in {root}")
    directory = Path(args.run_dir).expanduser().resolve() if args.run_dir else root / names[0]
    # Fail early, before accepting requests; --run-dir must be a direct child.
    if directory.parent != root or directory.name not in names:
        parser.error("--run-dir must contain a manifest and be an immediate child of the runs directory")
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
