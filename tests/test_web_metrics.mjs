import assert from "node:assert/strict";
import { summarize, sampleListSummary, matchesFilter, initialStatus } from "../verge/web_assets/metrics.mjs";
const path=(value,ids=[])=>({provided_experience_ids:ids,error:value===null?"timeout":null,judgment:value===null?null:{success:value}});
const record={sample:{sample_id:"s"},mode:"train",paths:[path(true),path(false,["A"]),path(true,["B"]),path(null,["C"])],
  selection_reasons:{A:"reason",B:"reason",C:"reason"},critic_rounds:[],error:null};
const data={schema:"verge-v3.1",samples:[record],feedback:[
  {record_id:"s:A",experience_id:"A",baseline_correct:true,experience_correct:false,error:null},
  {record_id:"s:B",experience_id:"B",baseline_correct:true,experience_correct:true,error:null},
  {record_id:"s:C",experience_id:"C",baseline_correct:true,experience_correct:null,error:"timeout"},
]};
const before=JSON.stringify(data),summary=summarize(data);
assert.equal(summary.correct,1); // Baseline count, not best training branch.
assert.equal(summary.paired,2); assert.equal(summary.invalidPairs,1);
assert.equal(summary.gain,-1); assert.equal(summary.outcomes["退化"],1);
assert.equal(initialStatus(record),"2路成功·含未知");
const list=sampleListSummary(record,data);
assert.equal(list.initialTrials.length,4); assert.equal(list.initialTrials[3].label,"经验3");
assert.equal(list.baseline,"判对"); assert.equal(list.selected,3);
assert.equal(matchesFilter(record,"退化",data),true); assert.equal(matchesFilter(record,"error",data),true);
assert.equal(JSON.stringify(data),before);
console.log("v3.1: equal trials, independent pairs, unknowns and baseline-only aggregate passed.");
