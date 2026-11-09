import assert from "node:assert/strict";
import {
  citedIds,
  sampleListSummary,
  experienceListSummary,
  admissionPath,
  subsetLabel,
} from "../verge/web_assets/metrics.mjs";
const path = (ids, success) => ({
  provided_experience_ids: ids,
  rubric: { criteria: [{ question: "condition?", weight: 1 }] },
  judgment: { success },
});
const empty = path([], false),
  left = path(["a"], false),
  right = path(["b"], false),
  pair = path(["a", "b"], true);
const attempts = ["a", "b"].map((id, i) => ({
  experience: { experience_id: id },
  path: i ? right : left,
  admitted: true,
  admission_subset: ["a", "b"],
}));
const source = {
  sample: { sample_id: "source" },
  baseline: empty,
  actual: null,
  selection_reasons: {},
  subset_paths: [empty],
  critic_rounds: [
    { round_number: 1, attempts, subset_paths: [empty, left, right, pair] },
  ],
};
const data = {
  schema: "verge-subset-learning-v2",
  samples: [source],
  task_index: { source: "source" },
  feedback: [
    {
      record_id: "later",
      task_id: "later",
      phase: "downstream",
      experience_ids: ["a", "b"],
      baseline_correct: false,
      actual_correct: false,
      credits: { a: 0.5, b: -0.5 },
    },
  ],
};
assert.deepEqual(citedIds(pair), []);
assert.equal(subsetLabel(pair, ["a", "b"]), "经验1＋经验2");
assert.equal(admissionPath(source.critic_rounds[0], attempts[0]), pair);
const summary = sampleListSummary(source, data);
assert.equal(summary.pairResults[0].result, "判对");
assert.equal(summary.candidateResults[0].result, "判错");
assert.equal(summary.candidateResults[0].joint, true);
assert.equal(
  experienceListSummary({ experience_id: "a", source_task_id: "source" }, data)
    .meanReward,
  0.5,
);
assert.equal(
  experienceListSummary({ experience_id: "b", source_task_id: "source" }, data)
    .meanReward,
  -0.5,
);
assert.equal(
  experienceListSummary({ experience_id: "a", source_task_id: "source" }, data)
    .sourceResult,
  "组合判对",
);
assert.equal(
  experienceListSummary({ experience_id: "a", source_task_id: "source" }, data)
    .sourceCited,
  null,
);
console.log(
  "Subset UI: joint source evidence, no attribution, per-experience credit distinct from full path outcome passed.",
);
