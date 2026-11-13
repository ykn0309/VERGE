import assert from "node:assert/strict";
import { stages } from "../verge/web_assets/trace.mjs";
import { criticCandidates } from "../verge/web_assets/metrics.mjs";

const first = {
  experience: { experience_id: "a" },
  path: { rubric: { rubric_id: "r1" } },
};
const second = {
  experience: { experience_id: "b" },
  path: { error: "judge timeout", rubric: { rubric_id: "r2" } },
};
const third = { experience: { experience_id: "c" }, path: null };
const record = {
  critic_rounds: [
    { round_number: 1, attempts: [first] },
    { round_number: 2, attempts: [second, third] },
    { round_number: 3, attempts: [], error: "proposal failed" },
  ],
};
const before = JSON.stringify(record);
const candidates = criticCandidates(record);
assert.deepEqual(
  candidates.map((c) => [
    c.round,
    c.number,
    c.attempt.experience.experience_id,
  ]),
  [
    [1, 1, "a"],
    [2, 1, "b"],
    [2, 2, "c"],
  ],
);
assert.equal(
  candidates.find((c) => c.attempt.experience.experience_id === "b").attempt
    .path.rubric.rubric_id,
  "r2",
);
assert.equal(candidates[1].attempt.path.error, "judge timeout");
assert.equal(JSON.stringify(record), before);
assert.deepEqual(criticCandidates({ critic_rounds: [] }), []);
assert.deepEqual(
  stages.map(([id]) => id),
  [
    "selection",
    "draft",
    "rubric",
    "judge",
    "critic",
    "candidate_rubric",
    "candidate_judge",
    "calls",
  ],
);
console.log(
  "Web trace: ordered candidate paths, empty/failed proposal rounds and generation/judgment stages passed.",
);
const { traceStageGroups } = await import("../verge/web_assets/trace.mjs");
const { resolveCriticRound } = await import("../verge/web_assets/metrics.mjs");
assert.deepEqual(
  traceStageGroups(record).map((g) => [g.label, g.round, g.stages.length]),
  [
    ["初始流程", null, 4],
    ["第1轮", 1, 3],
    ["第2轮", 2, 3],
    ["第3轮", 3, 3],
  ],
);
assert.equal(resolveCriticRound(record, 1, "b"), 1); // An explicit round overrides a stale candidate from another round.
assert.equal(resolveCriticRound(record, null, "b"), 2); // Existing candidate deep links retain their round.
assert.equal(resolveCriticRound(record, 99, "missing"), 1);
assert.equal(resolveCriticRound({ critic_rounds: [] }, null, "b"), null);
assert.equal(traceStageGroups({ critic_rounds: [] }).length, 1);
console.log(
  "Round navigation: separate round groups, stale candidate isolation and direct-link recovery passed.",
);
