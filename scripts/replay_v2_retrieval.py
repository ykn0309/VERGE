"""Twenty frozen historical task queries against converted active methods, no pipeline writes."""

from pathlib import Path
import json, hashlib, sys, shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verge.models import Experience, Method
from verge.modules import HybridRetriever
from verge.llm import OpenAICompatibleEmbedding, LLMSampleError

base = Path("artifacts/v2-module-replay")
out = base / "retrieval"
out.mkdir(exist_ok=True)
rows = json.loads((base / "inputs.json").read_text())
rows = [r for r in rows if r["role"] == "selector_checker"]
pool = json.loads((base / "source" / "experience_memory.json").read_text())[
    "experiences"
]
experiences = [
    Experience(e["experience_id"], Method(e["trigger"], e["content"]), "historical")
    for e in pool
    if e["status"] == "active"
]
manifest = {
    "tasks": 20,
    "active_versions": len(experiences),
    "embedding_model": "text-embedding-v4",
    "dimensions": 1024,
    "rrf_k": 60,
    "conversion": "active old versions only; content→principle; no categories or inherited quality; optional templates absent",
    "model_scope": "embedding only; no Generator/Judge/Checker calls",
    "source_sha256": hashlib.sha256(
        (base / "source" / "experience_memory.json").read_bytes()
    ).hexdigest(),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
shutil.copy2(__file__, out / "executed_script.py")
b = OpenAICompatibleEmbedding(
    "text-embedding-v4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="DASHSCOPE_API_KEY_2",
    dimensions=1024,
    timeout_seconds=120,
)
r = HybridRetriever(b)
results = []
try:
    for i, row in enumerate(rows, 1):
        task = row["input"]["application"]["sample"]["task"]
        result = {"sample_id": row["sample_id"], "task": task}
        try:
            found = r.retrieve(task, experiences)
            assert len(found) == min(20, len(experiences)) and len(
                {e.experience_id for e in found}
            ) == len(found)
            assert all(e.status == "active" for e in found)
            result.update(
                valid=True,
                selected_ids=[e.experience_id for e in found],
                trace=r.last_trace,
            )
        except LLMSampleError as exc:
            result.update(valid=False, error=str(exc))
        results.append(result)
        (out / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
    print(
        "retrieval",
        len(results),
        "valid",
        sum(x["valid"] for x in results),
        "cached vectors",
        len(r.cache),
        flush=True,
    )
finally:
    b.close()
