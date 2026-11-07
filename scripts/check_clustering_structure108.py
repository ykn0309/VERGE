"""Deterministic counterexamples; synthetic data, not VERGE efficacy tests."""

from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.csgraph import connected_components

out = Path(__file__).resolve().parents[1] / "artifacts/clustering108"
out.mkdir(parents=True, exist_ok=True)
A = np.repeat(np.eye(4), 3, axis=0)
P = A / A.sum()
checks = []
for e_groups in ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]):
    e = np.array(e_groups)
    t = np.repeat(e, 3)
    block = np.zeros((2, 2))
    np.add.at(block, (t[:, None], e[None, :]), P)
    rm = P.sum(1)
    cm = P.sum(0)
    rt = np.bincount(t, weights=rm)
    ce = np.bincount(e, weights=cm)
    Q = block[t[:, None], e[None, :]] * (rm / rt[t])[:, None] * (cm / ce[e])[None, :]
    mask = P > 0
    checks.append(
        {
            "experience_partition": e_groups,
            "KL_loss": float(np.sum(P[mask] * np.log(P[mask] / Q[mask]))),
        }
    )
C = A.T @ A
G = bmat([[None, csr_matrix(A)], [csr_matrix(A.T), None]], format="csr")
ncomp, _ = connected_components(G, directed=False)
# Add an alternative experience application on an existing task: a bridge.
bridge = A.copy()
bridge[0, 1] = 1
GB = bmat([[None, csr_matrix(bridge)], [csr_matrix(bridge.T), None]], format="csr")
bridge_components, _ = connected_components(GB, directed=False)
# Same citation graph, opposite rewards: relation alone cannot rank efficacy.
rewards_a = np.tile([1.0, 0.0, -1.0], 4)
rewards_b = -rewards_a
# A clearer arbitrary per-experience effect pattern on the identical A.
ra = np.repeat([1.0, 0.0, -1.0, 1.0], 3)
rb = -ra
means_a = (A * ra[:, None]).sum(0) / A.sum(0)
means_b = (A * rb[:, None]).sum(0) / A.sum(0)
# A hard task partition and a hard experience partition can have dense block counts.
t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
e = np.array([0, 1, 0, 1, 0, 1, 0, 1])
counts = np.zeros((2, 2), int)
np.add.at(counts, (t, e), 1)
result = {
    "notice": "Synthetic structural checks only, not algorithm/LLM quality or speed evidence",
    "degree_one": {
        "tasks": 12,
        "experiences": 4,
        "components": int(ncomp),
        "experience_shared_task_offdiagonal": (C - np.diag(np.diag(C))).tolist(),
        "alternative_partitions": checks,
    },
    "one_bridge": {
        "components": int(bridge_components),
        "meaning": "source/development retries can supply bridges; they must remain distinguished from downstream reward evidence",
    },
    "same_references_opposite_rewards": {
        "means_a": means_a.tolist(),
        "means_b": means_b.tolist(),
    },
    "hard_memberships_many_to_many": {"block_counts": counts.tolist()},
}
assert ncomp == 4 and bridge_components == 3
assert np.allclose([r["KL_loss"] for r in checks], np.log(2))
assert np.all(counts > 0)
(out / "structure_checks.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
