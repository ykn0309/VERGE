"""Twenty frozen historical tasks: independently exercise category embeddings/retrieval."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.replay_atomic_categories import SOURCE, grouped, save, verify
from verge.llm import OpenAICompatibleEmbedding, LLMSampleError, LLMFormatError
from verge.models import Experience, ExperienceCategory
from verge.modules import VectorRetriever, retrieval_units


def run(root):
    spec = verify(root)["model"]
    output = root / "retrieval"
    output.mkdir(exist_ok=False)
    output.chmod(0o700)
    targets = {r["sample_id"]: r for r in json.loads((root / "prepared_inputs.json").read_text())
               if r["role"] == "selector"}
    records = [json.loads(x) for x in (SOURCE / "artifacts/trajectories.jsonl").read_text().splitlines()]
    data_manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    data = {r["sample_id"]: r for r in [json.loads(x) for x in
            Path(data_manifest["config"]["dataset"]["config"]["paths"]["train"]).read_text().splitlines()]}
    state = {}
    cases = []
    for record in records:
        if record["sample_id"] in targets:
            target = targets[record["sample_id"]]
            units = grouped(list(state.values()))
            categories = [{k: u[k] for k in ("category_id", "name", "description")} for u in units if "members" in u]
            leaves = [{**e, "category_id": u.get("category_id")} for u in units for e in u.get("members", [u])]
            cases.append({"key": f"retrieval-{len(cases)+1:02}", "sample_id": record["sample_id"],
                          "task": data[record["sample_id"]]["task"], "experiences": leaves, "categories": categories,
                          "conversion": "full actual pool before target task, reconstructed from prior admissions/text updates only; controlled category fixtures, no future text or credit"})
        if record["admitted_experience_id"]:
            c = next(p["candidate"] for p in record["proposal_attempts"] if p["candidate"]
                     and p["candidate"]["candidate_id"] == record["admitted_candidate_id"])
            key = record["admitted_experience_id"]
            state[key] = {"experience_id": key, "content": c["content"], "trigger": c["trigger"]}
        for key, update in record["experience_updates"].items():
            if update["error"] is None:
                state[key].update(content=update["content"], trigger=update["trigger"])
    if len(cases) != 20 or len({c["task"] for c in cases}) != 20:
        raise ValueError("retrieval replay requires twenty distinct historical tasks")
    save(output / "prepared_inputs.json", cases)
    context = {}
    def event(row):
        with (output / "embedding_requests.jsonl").open("a") as stream:
            (output / "embedding_requests.jsonl").chmod(0o600)
            stream.write(json.dumps({**context, **row}, ensure_ascii=False) + "\n")
    embedding = OpenAICompatibleEmbedding("text-embedding-v4", base_url=spec["base_url"],
        api_key_env=spec["api_key_env"], dimensions=1024, timeout_seconds=spec["timeout_seconds"],
        max_retries=spec["max_retries"], metadata_hook=event)
    retriever = VectorRetriever(embedding)
    outputs = []
    try:
        for case in cases:
            context["key"] = case["key"]
            experiences = [Experience(e["experience_id"], e["content"], e["trigger"], 1, 1, [], [],
                                      category_id=e["category_id"]) for e in case["experiences"]]
            categories = [ExperienceCategory.from_dict(c) for c in case["categories"]]
            units = retrieval_units(experiences, categories)
            result, error = None, None
            try:
                recalled = retriever.retrieve(case["task"], experiences, categories=categories)
                ids = [e.experience_id for e in recalled]
                if len(ids) != len(set(ids)):
                    raise ValueError("duplicate recalled member")
                recalled_units = retrieval_units(recalled, categories)
                if any(set(e.experience_id for e in original.members) != set(e.experience_id for e in actual.members)
                       for actual in recalled_units for original in units if actual.unit_id == original.unit_id):
                    raise ValueError("truncated category")
                result = {"recalled_experience_ids": ids,
                          "recalled_units": [{"unit_id": u.unit_id, "category": bool(u.category),
                                              "member_count": len(u.members)} for u in recalled_units]}
            except (LLMSampleError, LLMFormatError) as exc:
                error = str(exc)
            row = {"key": case["key"], "sample_id": case["sample_id"], "pool_size": len(experiences),
                   "input_units": [{"unit_id": u.unit_id, "text": u.text,
                                    "experience_ids": [e.experience_id for e in u.members]} for u in units],
                   "output": result, "error": error}
            save(output / (case["key"] + ".json"), row)
            outputs.append(row)
    finally:
        embedding.close()
    verify(root)
    save(output / "summary.json", {"tasks": len(outputs), "valid": sum(o["error"] is None for o in outputs),
         "embedding_model": "text-embedding-v4", "dimensions": 1024,
         "case_sha256": __import__("hashlib").sha256((output / "prepared_inputs.json").read_bytes()).hexdigest(),
         "limitations": "controlled category assignments, no Selector/Generator/Judge calls; historical full pools small, category compression often recalls every unit; verifies transport/full expansion, not relevance or improvement"})


if __name__ == "__main__":
    run(Path("runs/atomic-categories-module-replay-20"))
