import assert from "node:assert/strict";
import { experienceListSummary } from "../verge/web_assets/metrics.mjs";
const e = { experience_id: "e", source_task_id: "source-task" };
const feedback = (id, task, before, after, extra = {}) => ({
  record_id: id,
  task_id: task,
  experience_ids: ["e"],
  phase: "downstream",
  baseline_correct: before,
  actual_correct: after,
  error: null,
  ...extra,
});
const data = {
  task_index: { "source-task": "source-sample" },
  samples: [
    {
      sample: { sample_id: "source-sample" },
      critic_rounds: [
        {
          round_number: 1,
          attempts: [
            {
              experience: e,
              admitted: true,
              path: {
                rubric: {
                  criteria: [{ weight: 1, source_experiences: ["e"] }],
                },
                judgment: { success: true },
              },
            },
          ],
        },
      ],
    },
  ],
  feedback: [
    feedback("development", "source-task", false, true, {
      phase: "development",
    }),
    feedback("gain", "t1", false, true),
    feedback("duplicate", "t1", false, true),
    feedback("loss", "t2", true, false),
    feedback("same", "t3", true, true),
    feedback("error", "t4", null, null, { error: "timeout" }),
    feedback("uncited", "t5", false, true, { experience_ids: [] }),
  ],
};
assert.deepEqual(experienceListSummary(e, data), {
  improved: 1,
  regressed: 1,
  bothCorrect: 1,
  bothWrong: 0,
  citedTasks: 4,
  validTasks: 3,
  technicalFailures: 1,
  meanReward: 0,
  sourceResult: "判对",
  sourceCited: true,
});
const fresh = experienceListSummary(e, { ...data, feedback: [] });
assert.equal(fresh.meanReward, null); // No downstream evidence is not zero reward.
assert.equal(fresh.validTasks, 0);
assert.equal(fresh.sourceResult, "判对"); // Source success does not count as downstream validation.
const missing = experienceListSummary(e, { feedback: [], samples: [] });
assert.equal(missing.sourceResult, "记录缺失");
assert.equal(missing.sourceCited, null);
const recovered = experienceListSummary(e, {
  feedback: [
    feedback("bad", "t", null, null, { error: "timeout" }),
    feedback("ok", "t", false, true),
  ],
});
assert.equal(recovered.citedTasks, 1);
assert.equal(recovered.validTasks, 1);
assert.equal(recovered.meanReward, 1);
console.log(
  "Experience states: source/downstream separation, unknown vs zero, task deduplication and failures passed.",
);
