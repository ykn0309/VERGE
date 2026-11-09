import assert from "node:assert/strict";
import {
  summarize,
  matchesFilter,
  resultOf,
  feedbackOutcome,
} from "../verge/web_assets/metrics.mjs";

const path = (success, ids = [], error = null) => ({
  error,
  judgment: error ? null : { success },
  provided_experience_ids: ids,
  rubric: {
    criteria: ids.map((id) => ({ weight: 1, source_experiences: [id] })),
  },
});
const record = (id, baseline, actual, error = null) => ({
  sample: { sample_id: id },
  baseline,
  actual,
  error,
  critic_rounds: [],
});
const feedback = (id, before, after, ids, error = null) => ({
  record_id: id,
  phase: "downstream",
  baseline_correct: before,
  actual_correct: after,
  experience_ids: ids,
  error,
});
const d = {
  samples: [
    record("gain", path(false), path(true, ["e"])),
    record("unused", path(true), path(false)),
    record("error", path(true), path(null, ["e"], "timeout")),
    record("baseline", path(true), null),
  ],
  feedback: [
    feedback("gain", false, true, ["e"]),
    feedback("unused", true, false, []),
    feedback("error", true, null, ["e"], "timeout"),
  ],
};
const provided = summarize(d);
assert.equal(provided.completed, 4);
assert.equal(provided.correct, 2);
assert.equal(provided.errors, 1);
assert.equal(provided.selected, 3);
assert.equal(provided.cited, 2);
assert.equal(provided.paired, 2);
assert.equal(provided.uncitedPairs, 1);
assert.equal(provided.gain, 0);
const cited = summarize(d, "cited");
assert.equal(cited.paired, 1);
assert.equal(cited.outcomes["退化"], 0);
assert.equal(cited.gain, 1);
assert.equal(cited.invalidPairs, 1);
assert.equal(matchesFilter(d.samples[1], "退化", d), true);
assert.equal(matchesFilter(d.samples[1], "退化", d, "cited"), false);
assert.equal(resultOf(path(null, [], "timeout")), "技术失败");
assert.equal(feedbackOutcome(d.feedback[2]), "无有效对照");
console.log(
  "Web metrics: provided/cited scopes, error accounting, and drill-down filters passed.",
);

const { sampleListSummary } = await import("../verge/web_assets/metrics.mjs");
const repaired = record("repair", path(false), null);
repaired.critic_rounds = [
  {
    attempts: [
      { admitted: false, path: path(false) },
      { admitted: true, path: path(true) },
    ],
  },
];
const r = sampleListSummary(repaired, { feedback: [] });
assert.equal(r.initial, "判错"); // An admitted candidate must not relabel the initial result.
assert.equal(r.actual, null);
assert.equal(r.candidates, 2);
assert.equal(r.admitted, 1);
assert.equal(r.selected, 0);
assert.equal(sampleListSummary(d.samples[0], d).comparison, "改善");
const unused = sampleListSummary(d.samples[1], d);
assert.equal(unused.comparison, "退化");
assert.equal(unused.cited, 0); // Provided-but-uncited does not imply no comparison.
const failed = sampleListSummary(d.samples[2], d);
assert.equal(failed.actual, "技术失败");
assert.equal(failed.comparison, "无有效对照");
const noCandidate = record("empty", path(false), null);
noCandidate.critic_rounds = [{ error: "timeout", attempts: [] }];
const empty = sampleListSummary(noCandidate, { feedback: [] });
assert.equal(empty.candidates, 0);
assert.equal(empty.criticErrors, 1);
assert.equal(empty.admitted, 0);
console.log(
  "Sample summaries: initial vs candidate results, uncited comparisons, missing paths and technical failures passed.",
);
const notStarted = record("not-started", null, null, "selector failed");
assert.equal(sampleListSummary(notStarted, { feedback: [] }).selected, null);
assert.equal(
  sampleListSummary(notStarted, { feedback: [] }).baseline,
  "未运行",
);
const validationRecord = record("candidate-states", path(false), null);
validationRecord.critic_rounds = [
  {
    round_number: 1,
    attempts: [
      {
        experience: { experience_id: "candidate-a" },
        path: path(false, ["candidate-a"]),
        admitted: false,
      },
      {
        experience: { experience_id: "candidate-b" },
        path: path(true, []),
        admitted: false,
      },
    ],
  },
  {
    round_number: 2,
    attempts: [
      {
        experience: { experience_id: "candidate-c" },
        path: path(true, ["candidate-c"]),
        admitted: true,
      },
      {
        experience: { experience_id: "candidate-d" },
        path: path(null, [], "timeout"),
        admitted: false,
      },
    ],
  },
];
const details = sampleListSummary(validationRecord, { feedback: [] });
assert.equal(details.initial, "判错");
assert.equal(details.baseline, "判错");
assert.deepEqual(details.candidateResults, [
  { round: 1, candidate: 1, result: "判错", cited: true, admitted: false },
  { round: 1, candidate: 2, result: "判对", cited: false, admitted: false },
  { round: 2, candidate: 1, result: "判对", cited: true, admitted: true },
  { round: 2, candidate: 2, result: "技术失败", cited: false, admitted: false },
]);
assert.deepEqual(sampleListSummary(noCandidate, { feedback: [] }).roundErrors, [
  { round: 1, error: "timeout" },
]);
console.log(
  "Candidate states: per-round judgments, uncited success, admitted success, and proposal/validation failures passed.",
);
