// Isolated DOM interaction contracts; real browser checks supplement this fixture.
const fs = require("node:fs"),
  vm = require("node:vm"),
  assert = require("node:assert/strict");
function matches(e, selector) {
  if (selector.startsWith("."))
    return selector
      .slice(1)
      .split(".")
      .every((c) => e.className.split(" ").includes(c));
  return e.tagName === selector.toUpperCase();
}
function walk(root, predicate) {
  if (predicate(root)) return root;
  for (const x of root.children || []) {
    const hit = walk(x, predicate);
    if (hit) return hit;
  }
  return null;
}
function all(root, predicate) {
  const rows = [];
  if (predicate(root)) rows.push(root);
  for (const x of root.children || []) rows.push(...all(x, predicate));
  return rows;
}
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.className = "";
    this._text = "";
    this.style = {};
    this.scrollTop = 0;
    this.value = "";
    this.disabled = false;
    this.open = false;
    this.offsetTop = 0;
    this.offsetHeight = 24;
    this.clientHeight = 700;
    this.classList = {
      toggle: (name, force) => {
        const classes = new Set(this.className.split(" ").filter(Boolean));
        const enabled = force ?? !classes.has(name);
        if (enabled) classes.add(name);
        else classes.delete(name);
        this.className = [...classes].join(" ");
      },
    };
  }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(text) {
    this._text = String(text ?? "");
    this.children = [];
  }
  append(...children) {
    for (let child of children) {
      if (typeof child === "string") child = new TextNode(child);
      child.parentNode = this;
      this.children.push(child);
    }
  }
  replaceChildren(...children) {
    this._text = "";
    this.children = [];
    this.append(...children);
  }
  setAttribute(key, value) {
    this.attributes[key] = String(value);
  }
  querySelector(selector) {
    return walk(this, (e) => e !== this && matches(e, selector));
  }
  remove() {
    if (this.parentNode)
      this.parentNode.children = this.parentNode.children.filter(
        (c) => c !== this,
      );
  }
}
class TextNode extends Element {
  constructor(text) {
    super("#text");
    this._text = text;
  }
}
const nodes = {
  "#app": new Element("main"),
  "#run": new Element("select"),
  "#refresh": new Element("button"),
};
const tabs = ["overview", "samples", "experiences"].map((page) => {
  const x = new Element("button");
  x.dataset.page = page;
  return x;
});
const document = {
  createElement: (t) => new Element(t),
  createTextNode: (t) => new TextNode(t),
  querySelector: (s) => nodes[s] || walk(nodes["#app"], (e) => matches(e, s)),
  querySelectorAll: (s) => (s === "[data-page]" ? tabs : []),
};
const method = {
  trigger: "任务包含可核验事实",
  principle: "核验数字是否有依据",
  template: null,
  parameters: {},
};
const r = {
  rubric_id: "rubric-long",
  analysis: "设计思路",
  criteria: [
    {
      criterion_id: "c1-long",
      question: "事实有依据吗？",
      weight: 1,
      source_experiences: ["e1"],
      bindings: {},
    },
    {
      criterion_id: "c2-long",
      question: "覆盖要求吗？",
      weight: 1,
      source_experiences: [],
      bindings: {},
    },
  ],
};
const j = {
  chosen_score: 0.5,
  rejected_score: 0,
  success: true,
  normalized_weights: { "c1-long": 0.5, "c2-long": 0.5 },
  criterion_judgments: [
    {
      criterion_id: "c1-long",
      chosen: true,
      rejected: false,
      chosen_reason: "优选有依据",
      rejected_reason: "非优选无依据",
    },
    {
      criterion_id: "c2-long",
      chosen: false,
      rejected: false,
      chosen_reason: "未覆盖",
      rejected_reason: "未覆盖",
    },
  ],
};
const ss = Array.from({ length: 50 }, (_, i) => ({
  sample_id: "s" + i,
  task: "任务" + i,
  chosen:
    i === 20
      ? "**标题**\n\n<img src=x onerror=evil()>\n\n" + "长回答\n".repeat(300)
      : "优选",
  rejected: "非优选",
  state: "initial_success",
  initial_rubric: r,
  initial_judgment: j,
  experience_updates: [],
  applications: [],
  proposal_attempts: [],
  selected_experiences: [],
  retrieved_experience_ids: [],
  retrieval_trace: [],
  admitted_experience_ids: [],
}));
ss[20].proposal_attempts = [{round_number: 1, review: {approved: true, reason: "方法原子且无特定答案泄漏"}, proposal: {analysis: "初始诊断", candidates: [{experience_id: "e1", method}]}, applications: [], error: null}];
const a = {
  sample: { sample_id: "s20", task: "任务20" },
  version_id: "e1",
  method,
  phase: "downstream",
  rubric: r,
  judgment: j,
  feedback: "positive",
  positive_weight: 0.5,
  reverse_weight: 0,
  outcomes: { positive: 1, reverse: 0, both_true: 0, both_false: 0 },
  recalled: true,
  selected: true,
  cited: true,
  retrieval_rank: 1,
  observation: "error",
  update_reason: "parameters_unavailable",
  q_before: null,
  q_after: null,
  reports: [
    { stage: "selector", status: "correct", evidence: "适用" },
    {
      stage: "generator",
      status: "error",
      evidence: "未忠实落实",
      details: [
        { criterion_id: "c1-long", status: "error", evidence: "新增要求" },
      ],
    },
    { stage: "judge", status: "correct", evidence: "原文判断一致" },
  ],
};
ss[20].applications = [a];
ss[20].selected_experiences = [{ experience_id: "e1", ...method }];
ss[20].retrieved_experience_ids = ["e1"];
ss[20].retrieval_trace = [
  { experience_id: "e1", vector_rank: 1, keyword_rank: null },
];
ss[2].state = "application_success";
ss[2].proposal_attempts = [
  {
    round_number: 1,
    proposal: {
      analysis: "先核对原文",
      candidates: [{ experience_id: "candidate1", method }],
    },
    applications: [{ version_id: "candidate1", feedback: "positive" }],
    rubric: r,
    judgment: j,
  },
];
ss[2].admitted_experience_ids = ["candidate1"];
const ee = Array.from({ length: 51 }, (_, i) => ({
  experience_id: "e" + i,
  method: { ...method, principle: "方法" + i },
  source_sample_id: "s0",
  status: i === 50 ? "retired" : "active",
  q_posterior: i === 0 ? null : 0.7,
  parent_id: null,
  q_prior: null,
  parameter_version: null,
  applications: i === 1 ? [a] : [],
}));
const data = {
  name: "fixture",
  manifest: { run_mode: "collection" },
  metrics: { tasks: 50, initial_correct: 50, active: 50, retired: 1 },
  revisions: [],
  experiences: ee,
  samples: ss,
};
let currentURL = "";
const fetches = [];
const context = vm.createContext({
  document,
  URLSearchParams,
  location: { search: "?run=fixture&page=samples&sample=s20&stage=checks" },
  history: {
    replaceState: (_a, _b, url) => {
      currentURL = url;
    },
  },
  fetch: async (path) => {
    fetches.push(path);
    return {
      ok: true,
      json: async () =>
        path === "/api/runs"
          ? { runs: ["fixture"], default: "fixture" }
          : path.startsWith("/api/role-io")
            ? {
                entries: [
                  {
                    role: "judge_checker",
                    model: "test",
                    attempt: 1,
                    raw_output: "原始回复",
                    validation_status: "invalid",
                    contract_error: "Criterion.bindings: expected a dictionary",
                    messages: [
                      { role: "system", content: "真实系统提示词" },
                      { role: "user", content: "真实消息" },
                    ],
                    input: { task: "任务20" },
                  },
                ],
                warnings: [],
              }
            : data,
    };
  },
  console,
});
vm.runInContext(fs.readFileSync("verge/web_assets/app.js", "utf8"), context);
function run(code) {
  return vm.runInContext(code, context);
}
function findButton(text) {
  return walk(
    nodes["#app"],
    (e) => e.tagName === "BUTTON" && e.textContent === text,
  );
}
function stage(key) {
  walk(nodes["#app"], (e) => e.dataset.stage === key).onclick();
}
function content() {
  return document.querySelector(".detail-content").textContent;
}
function sampleButton(id) {
  return walk(nodes["#app"], (e) => e.dataset.sample === id);
}
function expButton(id) {
  return walk(nodes["#app"], (e) => e.dataset.experience === id);
}
function field(label) {
  return walk(nodes["#app"], (e) => e.attributes["aria-label"] === label);
}
setImmediate(async () => {
  try {
    assert.equal(run("state.sample"), "s20");
    assert.equal(run("state.sampleTab"), "checks");
    for (const text of [
      "选择核对",
      "应用核对",
      "判断核对",
      "未忠实落实",
      "参数尚未估计",
    ])
      assert.ok(content().includes(text), text);
    assert.ok(currentURL.includes("stage=checks"));
    let sidebar = document.querySelector(".sidebar");
    sidebar.scrollTop = 430;
    sampleButton("s21").onclick();
    assert.equal(
      document.querySelector(".sidebar").scrollTop,
      430,
      "sample click preserves scroll",
    );
    assert.equal(run("state.sample"), "s21");
    const filter = field("样本筛选");
    filter.value = "checker";
    filter.onchange();
    assert.equal(
      all(document.querySelector(".list-items"), (e) => e.dataset.sample)
        .length,
      1,
    );
    sampleButton("s20").onclick();
    stage("rubric");
    assert.ok(content().includes("权重 50.0%"));
    assert.ok(content().includes("优选回答"));
    assert.ok(content().includes("非优选回答"));
    assert.ok(content().includes("优选有依据"));
    stage("critic");
    assert.ok(content().includes("经验核对通过 · 允许入池"));
    assert.ok(content().includes("方法原子且无特定答案泄漏"));
    stage("task");
    assert.ok(content().includes("<img src=x onerror=evil()>"));
    assert.equal(
      walk(nodes["#app"], (e) => e.tagName === "IMG"),
      null,
      "dataset HTML never executes",
    );
    const expand = findButton("展开全文");
    assert.ok(expand);
    expand.onclick();
    assert.equal(expand.textContent, "收起全文");
    expand.onclick();
    assert.equal(expand.textContent, "展开全文");
    stage("io");
    const load = findButton("加载原始角色请求／回复");
    await load.onclick();
    assert.ok(
      content().includes("原始回复") &&
        content().includes("发送给模型的提示词"),
    );
    assert.ok(content().includes("系统提示词 · System"));
    assert.ok(content().includes("用户输入 · User"));
    assert.ok(content().includes("真实系统提示词"));
    const failedCall = walk(nodes["#app"], (e) => e.tagName === "SUMMARY" && e.textContent.includes("结构校验失败"));
    assert.ok(failedCall && failedCall.textContent.includes("Criterion.bindings"));
    const prompt = walk(
      nodes["#app"],
      (e) =>
        e.tagName === "DETAILS" &&
        e.children[0]?.textContent.startsWith("发送给模型的提示词"),
    );
    assert.ok(prompt?.open, "input prompts open by default");
    assert.ok(
      content().indexOf("发送给模型的提示词") <
        content().indexOf("模型回复 · 原始文本"),
      "input appears before reply",
    );
    const number = fetches.filter((p) => p.startsWith("/api/role-io")).length;
    stage("task");
    stage("io");
    assert.equal(
      fetches.filter((p) => p.startsWith("/api/role-io")).length,
      number,
      "role IO cached",
    );
    run("state.sampleFilter='all';selectSample('s2','critic')");
    assert.ok(content().includes("第1轮纠错"));
    assert.ok(content().includes("最终已入池"));
    assert.ok(content().includes("先核对原文"));
    tabs[2].onclick();
    sidebar = document.querySelector(".sidebar");
    sidebar.scrollTop = 250;
    expButton("e15").onclick();
    assert.equal(
      document.querySelector(".sidebar").scrollTop,
      250,
      "experience click preserves scroll",
    );
    let sort = field("经验排序");
    sort.value = "references";
    sort.onchange();
    assert.equal(run("state.sort"), "references");
    assert.equal(
      all(document.querySelector(".list-items"), (e) => e.dataset.experience)[0]
        .dataset.experience,
      "e1",
    );
    const status = field("经验状态");
    status.value = "retired";
    status.onchange();
    assert.equal(run("state.experience"), "e50");
    assert.equal(
      all(document.querySelector(".list-items"), (e) => e.dataset.experience)
        .length,
      1,
    );
    run(
      "state.status='active';state.experience='e1';state.experienceTab='method';render()",
    );
    for (const text of [
      "核验方法",
      "适用条件",
      "质量与证据",
      "原始逐标准反馈",
      "双方满足",
    ])
      assert.ok(content().includes(text), text);
    stage("history");
    assert.ok(content().includes("样本 21"));
    stage("evolution");
    assert.ok(content().includes("不是独立泛化验证"));
    assert.match(
      run("quality({...state.data.experiences[2],applications:[]})"),
      /基础先验/,
    );
    tabs[0].onclick();
    assert.ok(nodes["#app"].textContent.includes("Checker核对意见"));
    run("state.sampleFilter='all';selectSample('s20','task')");
    const sampleSearch = field("搜索样本");
    sampleSearch.value = "#01";
    sampleSearch.oninput();
    assert.equal(
      all(document.querySelector(".list-items"), (e) => e.dataset.sample)
        .length,
      1,
    );
    assert.ok(sampleButton("s0"));
    const safeTable = run(
      'rich("| 字段 | 数值 |\\n| --- | --- |\\n| **测试** | 12 |")',
    );
    assert.ok(walk(safeTable, (e) => e.tagName === "TABLE"));
    run(
      "state.data.samples[49].state='sample_error';state.data.samples[49].initial_judgment=null;state.data.samples[49].sample_error={stage:'initial',message:'有限重试耗尽'};selectSample('s49','critic')",
    );
    assert.ok(content().includes("有限重试耗尽"));
    assert.ok(nodes["#app"].textContent.includes("首次：无有效结果"));
    run("selectExperience('e1')");
    const experienceSearch = field("搜索经验");
    experienceSearch.value = "E02";
    experienceSearch.oninput();
    assert.equal(
      all(document.querySelector(".list-items"), (e) => e.dataset.experience)
        .length,
      1,
    );
    assert.ok(expButton("e1"));
    run(
      "state.data.samples[49].experience_updates=[{parent_id:'e1',trigger_sample_id:'s49',action:'keep',accepted:false,update:{reason:'保留原方法'},replays:[],case_refs:[]}];selectSample('s49','critic')",
    );
    assert.ok(content().includes("本题触发的版本修订"));
    assert.ok(content().includes("保留原方法"));
    console.log(
      "DOM contract checks passed: deep links, structured criteria/checkers, rich text safety, long text, filters, scroll preservation, history and lazy role IO.",
    );
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
});
