import assert from "node:assert/strict";
import { callRecords } from "../verge/web_assets/calls.mjs";
const messages = [{ role: "user", content: "line 1\nline 2 <script>" }];
const entries = [
  { event_type: "request_timing", call_id: "a", request_seconds: 1 },
  { call_id: "a", attempt: 1, messages, error: "timeout", raw_output: null },
  { event_type: "request_timing", call_id: "a", request_seconds: 2 },
  {
    call_id: "a",
    attempt: 2,
    messages,
    raw_output: "{ broken",
    output: null,
    error: "invalid JSON",
  },
  { event_type: "request_timing", call_id: "b", request_seconds: 3 },
  {
    call_id: "b",
    attempt: 1,
    messages,
    raw_output: "",
    output: { parsed: true },
  },
  {
    call_id: "b",
    event_type: "contract_validation",
    validation_status: "invalid",
    contract_attempt: 2,
    error: "unknown ID",
  },
  { messages, output: { legacy: true } },
  { role: "embedding", input: { text: "task" }, output: [0.1, 0.2] },
];
const before = JSON.stringify(entries);
const calls = callRecords(entries);
assert.equal(calls.length, 5);
assert.deepEqual(
  calls.slice(0, 3).map((c) => c.timing.request_seconds),
  [1, 2, 3],
);
assert.equal(calls[0].rawAvailable, false);
assert.equal(calls[1].entry.raw_output, "{ broken");
assert.equal(calls[2].rawAvailable, true); // Empty original text is not missing.
assert.equal(calls[2].entry.raw_output, "");
assert.equal(calls[2].error, "unknown ID");
assert.equal(calls[2].validation.contract_attempt, 2);
assert.equal(calls[3].rawAvailable, false); // Never pretend parsed JSON is original text.
assert.deepEqual(calls[0].entry.messages, messages);
assert.equal(JSON.stringify(entries), before);
console.log(
  "Web calls: preserved raw messages, missing/empty replies, retry order, timing and validation passed.",
);

assert.equal(calls[4].entry.role, "embedding");
assert.equal(calls[4].rawAvailable, false);
