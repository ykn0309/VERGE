"""Plan65 isolated module checks from frozen real histories; never write a live pool."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_module_model_replay import RecordedClient
from verge.llm import OpenAICompatibleBackend, LLMFormatError, LLMSampleError
from verge.models import CategoryOrganization, ExperienceUpdate
from verge.prompts import llm_messages

SOURCE = Path("runs/helpsteer3-general-delayed-experience-qwen-flash-200-resume5")
ROLES = ("critic", "experience_updater", "selector", "category_organizer")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    path.chmod(0o600)


def grouped(candidates):
    """Controlled category fixtures over actual candidate text; no new experience text."""
    groups = {}
    singles = []
    for e in candidates:
        text = e["content"] + e["trigger"]
        kind = ("角色与归属" if any(w in text for w in ("角色", "身份", "人物")) else
                "信息依据" if any(w in text for w in ("信息", "材料", "事实", "虚构")) else None)
        if kind is None:
            singles.append(deepcopy(e))
        else:
            groups.setdefault(kind, []).append(deepcopy(e))
    return [{"category_id": str(i), "name": name,
             "description": "组织" + name + "相关的核验方法，须逐条核对成员的具体适用条件。",
             "members": members} for i, (name, members) in enumerate(groups.items(), 1)] + singles


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    events = [json.loads(x) for x in (SOURCE / "artifacts/role_io.jsonl").read_text().splitlines()]
    trajectories = [json.loads(x) for x in (SOURCE / "artifacts/trajectories.jsonl").read_text().splitlines()]
    order = {r["sample_id"]: i + 1 for i, r in enumerate(trajectories)}
    priorities = [57, 60, 63, 73, 84, 105, 148, 154, 161, 186, 198]
    rows = []
    for role in ROLES[:3]:
        pool = [e for e in events if e["role"] == role and e.get("output") is not None]
        chosen = []
        seen = set()
        ranked = sorted(pool, key=lambda e: (order[e["sample_id"]] not in priorities,
                                            order[e["sample_id"]]))
        # Spread remaining normal/boundary tasks over the whole run, not an early prefix.
        ranked = ([e for e in ranked if order[e["sample_id"]] in priorities]
                  + pool[::max(1, len(pool)//20)] + pool)
        for event in ranked:
            payload = deepcopy(event["input"])
            task = payload.get("task", payload.get("cases", [{}])[-1].get("task"))
            if task in seen:
                continue
            conversion = "system atomicity instruction only; complete real message history preserved"
            if role == "selector":
                if not payload["candidates"]:
                    continue
                payload["candidates"] = grouped(payload["candidates"])
                conversion = "same recalled leaves/order within each group; controlled category fixture; no rate added"
            elif role == "experience_updater":
                cards = []
                for card in payload["cases"]:
                    card["criteria"] = [c for c in card["criteria"] if c["chosen"] != c["rejected"]]
                    if card["criteria"]:
                        cards.append(card)
                if len(cards) < 2:
                    continue
                payload["cases"] = cards
                conversion = "remove same-verdict criteria and resulting empty cases; preserve full tasks/answers and used text"
            messages = llm_messages(role, payload)
            if role == "critic":
                messages = [messages[0], *deepcopy(event["messages"][1:])]
            chosen.append({"role": role, "key": f"{role}-{len(chosen)+1:02}",
                           "sample_id": event["sample_id"], "sample_index": order[event["sample_id"]],
                           "task": task, "input": payload, "messages": messages,
                           "historical_input": event["input"], "historical_output": event["output"],
                           "historical_messages": event["messages"], "conversion": conversion})
            seen.add(task)
            if len(chosen) == 20:
                break
        if len(chosen) != 20:
            raise ValueError(f"insufficient unique task inputs for {role}: {len(chosen)}")
        rows.extend(chosen)
    # The new role has no historical requests. Freeze 20 independent, real candidate-pool snapshots.
    for index, selector in enumerate([r for r in rows if r["role"] == "selector"], 1):
        units = selector["input"]["candidates"]
        categories = [{k: u[k] for k in ("category_id", "name", "description")} for u in units if "members" in u]
        leaves = [{**{k: e[k] for k in ("experience_id", "content", "trigger")},
                   "category_id": u.get("category_id")} for u in units for e in u.get("members", [u])]
        if index <= 5:
            categories = []
            for e in leaves:
                e["category_id"] = None
        payload = {"categories": categories, "experiences": leaves,
                   "pending_experience_ids": [e["experience_id"] for e in leaves[-5:]]}
        rows.append({"role": "category_organizer", "key": f"category_organizer-{index:02}",
                     "sample_id": selector["sample_id"], "task": selector["task"],
                     "sample_index": selector["sample_index"], "input": payload,
                     "messages": llm_messages("category_organizer", payload),
                     "conversion": "new role: actual prior recalled leaf texts; controlled existing-category fixtures, first five ungrouped; no actual historical classification or task/answers supplied to organizer"})
    save(root / "prepared_inputs.json", rows)
    protected = [SOURCE / "run_manifest.json", *(SOURCE / "artifacts" / n for n in
                 ("experience_memory.json", "trajectories.jsonl", "role_io.jsonl", "llm_requests.jsonl")),
                 Path(manifest["config"]["dataset"]["config"]["paths"]["train"])]
    save(root / "replay_manifest.json", {
        "model": manifest["config"]["models"]["default"], "planned_outputs": 80,
        "max_http_requests": 720, "per_role_unique_tasks": {role: 20 for role in ROLES},
        "source_sha256": {str(p): digest(p) for p in protected},
        "input_sha256": digest(root / "prepared_inputs.json"),
        "code_sha256": {str(p): digest(p) for p in [*Path("verge").glob("*.py"), Path(__file__)]},
        "limitations": "isolated acceptance replay; historical outputs are references, not a fresh paired baseline; controlled category fixtures cannot prove live retrieval gains; no chaining, pool writes or complete pipeline"})
    snapshot = root / "code_snapshot"
    snapshot.mkdir()
    for p in [*Path("verge").glob("*.py"), Path(__file__)]:
        (snapshot / p.name).write_bytes(p.read_bytes())


def verify(root):
    m = json.loads((root / "replay_manifest.json").read_text())
    if digest(root / "prepared_inputs.json") != m["input_sha256"]:
        raise ValueError("replay inputs changed")
    for key in ("source_sha256", "code_sha256"):
        for name, value in m[key].items():
            if digest(Path(name)) != value:
                raise ValueError(f"protected file changed: {name}")
    return m


def validate(role, payload, output):
    if role == "experience_updater":
        ExperienceUpdate.from_dict(output)
    elif role == "category_organizer":
        CategoryOrganization.from_dict(output).validate_partition(
            {e["experience_id"] for e in payload["experiences"]},
            {c["category_id"] for c in payload["categories"]})
    elif role == "selector":
        ids = output["selected_experience_ids"]
        allowed = {e["experience_id"] for u in payload["candidates"] for e in u.get("members", [u])}
        if len(ids) != len(set(ids)) or len(ids) > payload["max_selected"] or set(ids) - allowed:
            raise ValueError("invalid leaf selection")
    else:
        for c in output["candidates"]:
            ExperienceUpdate.from_dict(c)


def run_role(root_text, role):
    root = Path(root_text)
    spec = verify(root)["model"]
    directory = root / role
    directory.mkdir(exist_ok=False)
    directory.chmod(0o700)
    context = {}
    def append(name, event):
        path = directory / name
        with path.open("a") as stream:
            path.chmod(0o600)
            stream.write(json.dumps({**context, **event}, ensure_ascii=False) + "\n")
    client = RecordedClient(spec, append)
    backend = OpenAICompatibleBackend(spec["model"], **{k: spec[k] for k in
        ("base_url", "api_key_env", "timeout_seconds", "max_retries", "enable_thinking")}, client=client,
        trace_hook=lambda e: append("role_io.jsonl", e), metadata_hook=lambda e: append("llm_requests.jsonl", e))
    try:
        for case in json.loads((root / "prepared_inputs.json").read_text()):
            if case["role"] != role:
                continue
            context["case_key"] = case["key"]
            messages = deepcopy(case["messages"])
            output, error = None, None
            for attempt in range(3):
                try:
                    with patch("verge.llm.llm_messages", lambda *a, **kw: deepcopy(messages)):
                        raw = backend.generate(role, case["input"])
                    validate(role, case["input"], raw)
                    output, error = raw, None
                    break
                except (LLMFormatError, LLMSampleError) as exc:
                    error = str(exc)
                    break
                except ValueError as exc:
                    error = str(exc)
                    messages.append({"role": "user", "content": "结构校验失败，请完整重写合法输出。"})
            save(directory / (case["key"] + ".json"), {"key": case["key"], "output": output,
                 "error": error, "structure_retries": attempt, "actual_messages": messages})
    finally:
        backend.close()
        client.close()
    return {"role": role, "completed": 20}


def summarize(root):
    verify(root)
    results = {}
    for role in ROLES:
        rows = [json.loads(p.read_text()) for p in (root / role).glob(f"{role}-*.json")]
        http = [json.loads(x) for x in (root / role / "http_io.jsonl").read_text().splitlines()]
        requests = [e["request_body"] for e in http if "request_body" in e]
        if not all(r["model"] == "qwen3.5-flash" and r["enable_thinking"] is False for r in requests):
            raise ValueError("unexpected real model/thinking setting")
        results[role] = {"outputs": len(rows), "valid": sum(r["error"] is None for r in rows),
                         "http_requests": len(requests), "structure_retries": sum(r["structure_retries"] for r in rows)}
    save(root / "summary.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "summarize"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "run":
        with ProcessPoolExecutor(max_workers=2) as pool:
            print(list(pool.map(run_role, [str(args.root)] * len(ROLES), ROLES)), flush=True)
        print(summarize(args.root))
    else:
        print(summarize(args.root))
