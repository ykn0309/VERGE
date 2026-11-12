"""Read-only summaries of saved requests; never infer an exact in-flight stage."""

from collections import Counter, OrderedDict
import json
from pathlib import Path
from threading import Lock
import time

_CACHE = OrderedDict()
_LOCK = Lock()


def _json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _rows(path):
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # A growing file may end in an unfinished write.
            if isinstance(row, dict):
                yield row


def _settings_and_tasks(directory):
    config = _json(directory / "config.json", {})
    model = config.get("model", {})
    settings = {
        "model": model.get("model"),
        "enable_thinking": model.get("enable_thinking"),
        "train_workers": config.get("run", {}).get("train_workers"),
        "retrieval": config.get("retrieval", {}),
        "max_selected": config.get("selection", {}).get("max_selected"),
        "critic": config.get("critic", {}),
        "clustering": config.get("clustering", {}),
    }
    path = config.get("dataset", {}).get("paths", {}).get("train")
    tasks = {}
    if isinstance(path, str):
        for row in _rows(Path(path)):
            if isinstance(row.get("sample_id"), str) and isinstance(
                row.get("task"), str
            ):
                tasks[row["sample_id"]] = row["task"][:250]
    return settings, tasks


def _scan(directory):
    settings, tasks = _settings_and_tasks(directory)
    roles, last_roles = Counter(), {}
    total = retries = input_tokens = output_tokens = invalid = 0
    usage_seen = False
    for row in _rows(directory / "llm_requests.jsonl"):
        total += 1
        roles[row.get("role", "unknown")] += 1
        retries += isinstance(row.get("attempt"), int) and row["attempt"] > 1
        sid = row.get("sample_id")
        if sid:
            last_roles[sid] = row.get("role", "未知")
        usage = row.get("usage") or {}
        if isinstance(usage, dict):
            for field, alternate, is_input in [
                ("prompt_tokens", "input_tokens", True),
                ("completion_tokens", "output_tokens", False),
            ]:
                value = usage.get(field, usage.get(alternate))
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage_seen = True
                    if is_input:
                        input_tokens += value
                    else:
                        output_tokens += value
    recent, by_call, timings = [], {}, {}
    for row in _rows(directory / "role_io.jsonl"):
        cid = row.get("call_id")
        if row.get("event_type") == "request_timing" and cid:
            timings.setdefault(cid, []).append(row)
        elif row.get("event_type") == "contract_validation":
            if row.get("validation_status") == "invalid":
                invalid += 1
                if cid in by_call:
                    by_call[cid]["error"] = row.get("error")
        elif row.get("messages"):
            timing = timings.get(cid, [])
            index = (row.get("attempt") or 1) - 1
            timing = timing[index] if 0 <= index < len(timing) else {}
            call = {
                "sample_id": row.get("sample_id"),
                "role": row.get("role"),
                "model": row.get("model"),
                "error": row.get("error"),
                "attempt": row.get("attempt"),
                "seconds": timing.get("request_seconds"),
                "finished_at": timing.get("finished_at"),
            }
            recent.append(call)
            by_call[cid] = call
    return {
        "settings": settings,
        "tasks": tasks,
        "last_roles": last_roles,
        "requests": {
            "total": total,
            "transport_retries": retries,
            "invalid_contracts": invalid,
            "input_tokens": input_tokens if usage_seen else None,
            "output_tokens": output_tokens if usage_seen else None,
            "by_role": dict(roles),
        },
        "recent_calls": recent[-12:][::-1],
    }


def load_observation(directory):
    directory = Path(directory)
    watched = [
        directory / name
        for name in ("config.json", "role_io.jsonl", "llm_requests.jsonl")
    ]
    stamps = tuple(
        (p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None
        for p in watched
    )
    key = str(directory.resolve())
    with _LOCK:
        cached = _CACHE.get(key)
        if not cached or cached[0] != stamps:
            cached = (stamps, _scan(directory))
            _CACHE[key] = cached
            while len(_CACHE) > 8:
                _CACHE.popitem(last=False)
        else:
            _CACHE.move_to_end(key)
        snapshot = cached[1]
    manifest = _json(directory / "manifest.json", {})
    # Use the committed checkpoint for membership, not the last returned request.
    checkpoint = _json(directory / "checkpoint.json", {})
    committed = {r["sample"]["sample_id"] for r in checkpoint.get("trajectories", [])}
    pending = [
        {"sample_id": sid, "task": snapshot["tasks"].get(sid, sid), "last_role": role}
        for sid, role in snapshot["last_roles"].items()
        if sid not in committed
    ]
    return {
        "manifest": manifest,
        "observed_at": time.time(),
        "target_tasks": len(snapshot["tasks"]) or None,
        "settings": snapshot["settings"],
        "requests": snapshot["requests"],
        "recent_calls": snapshot["recent_calls"],
        "uncommitted": pending,
    }
