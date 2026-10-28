"use strict";
const state = {
  page: "overview",
  data: null,
  run: "",
  sample: null,
  experience: null,
  sampleTab: "task",
  experienceTab: "method",
  sampleSearch: "",
  sampleFilter: "all",
  status: "active",
  sort: "quality",
  search: "",
  appPhase: "all",
  scrolls: {},
  io: {},
  role: "all",
  revealList: false,
  loadSequence: 0,
};
const $ = (selector) => document.querySelector(selector);
const list = (value) => (Array.isArray(value) ? value : []);
const samples = () => list(state.data?.samples);
const experiences = () => list(state.data?.experiences);
const SAMPLE_STATES = {
  initial_failure: ["首次未成功", "warn"],
  candidate_rejected: ["经验核对未通过", "warn"],
  candidate_review_failed: ["经验核对失败", "bad"],
  source_validation_failed: ["来源验证失败（已入池）", "bad"],
  initial_success: ["首次正确", "good"],
  application_success: ["纠错成功", "good"],
  validation_exhausted: ["纠错未成功", "warn"],
  no_candidate: ["未提出候选", "warn"],
  invalid_application_exhausted: ["候选格式失败", "bad"],
  sample_error: ["样本异常", "bad"],
  in_progress: ["处理中", "neutral"],
};
const FEEDBACK = {
  positive: ["正向区分", "good"],
  reverse: ["反向区分", "bad"],
  tie: ["净平局", "neutral"],
  unused: ["未参与评分", "neutral"],
};
const REPORT = {
  correct: ["报告：执行正确", "good"],
  error: ["报告：存在错误", "bad"],
  unknown: ["无法确定", "warn"],
  failed: ["调用失败", "bad"],
};
const PHASE = {
  admission: "历史来源准入",
  source_validation: "来源验证",
  downstream: "下游应用",
  replay: "修订回放",
};
const STAGES = {
  selector: ["选择核对", "Selector Checker"],
  generator: ["应用核对", "Generator Checker"],
  judge: ["判断核对", "Judge Checker"],
};
const ROLES = {
  experience_checker: "经验核对器 · 候选质量",
  selector: "Selector · 经验选择",
  initial_generator: "Generator · 标准生成",
  judge: "Judge · 逐项判断",
  critic: "Critic · 经验提案",
  experience_updater: "Updater · 经验修订",
  selector_checker: "Selector Checker · 适用性",
  generator_checker: "Generator Checker · 方法应用",
  judge_checker: "Judge Checker · 判断执行",
  embedding: "Embedding · 检索向量",
};
const REASONS = {
  parameters_unavailable: "概率参数尚未估计，本次不更新Q",
  admission_prior_only: "来源准入不重复更新Q",
  updated: "已更新Q",
  unused: "没有正权重引用，本次不更新Q",
  duplicate_task: "同一任务已记录，不重复更新",
  duplicate: "重复记录，不更新",
  observation_unknown: "核对意见无法确定，不更新Q",
  observation_unavailable: "核对报告不完整，不更新Q",
  parent_replay_no_update: "旧版本仅作比较，不更新Q",
};
function el(tag, text = "", className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function button(text, action, className = "") {
  const node = el("button", text, className);
  node.type = "button";
  node.onclick = action;
  return node;
}
function badge(text, tone = "neutral") {
  return el("span", text, "badge " + tone);
}
function stateBadge(key) {
  const [text, tone] = SAMPLE_STATES[key] || [key || "未记录", "neutral"];
  return badge(text, tone);
}
function feedbackBadge(key) {
  const [text, tone] = FEEDBACK[key] || [key || "无反馈", "neutral"];
  return badge(text, tone);
}
function reportBadge(key) {
  const [text, tone] = REPORT[key] || ["未执行", "neutral"];
  return badge(text, tone);
}
function percent(n) {
  return typeof n === "number" ? (n * 100).toFixed(1) + "%" : "—";
}
function compact(text, n = 80) {
  const clean = String(text || "")
    .replace(/\s+/g, " ")
    .trim();
  return clean.length > n ? clean.slice(0, n) + "…" : clean;
}
function sampleNumber(id) {
  const i = samples().findIndex((s) => s.sample_id === id);
  return i < 0 ? "来源样本" : "样本 " + String(i + 1).padStart(2, "0");
}
function experienceNumber(id) {
  const i = experiences().findIndex((e) => e.experience_id === id);
  return i < 0 ? "候选经验" : "E" + String(i + 1).padStart(2, "0");
}
function quality(e) {
  const updates = list(e.applications).filter(
    (a) => a.update_reason === "updated",
  ).length;
  return e.q_posterior == null
    ? "Q 未估计"
    : `Q ${percent(e.q_posterior)}${updates ? "" : " · 基础先验／证据不足"}`;
}
function qualityBadge(e) {
  return badge(quality(e), e.q_posterior == null ? "neutral" : "blue");
}
function referenceCount(e) {
  return new Set(
    list(e.applications)
      .filter((a) => a.phase === "downstream" && a.cited)
      .map((a) => a.sample.task.replace(/\s+/g, " ").trim()),
  ).size;
}
function finalJudgment(s) {
  return (
    [...list(s.proposal_attempts)].reverse().find((a) => a.judgment)
      ?.judgment || s.initial_judgment
  );
}
function hasCheckerError(s) {
  return list(s.applications).some((a) =>
    list(a.reports).some((r) => r.status === "error"),
  );
}
function hasOutputError(s) {
  return (
    Boolean(s.sample_error) || list(s.proposal_attempts).some((a) => a.error)
  );
}
function raw(parent, title, value) {
  const d = el("details", "", "advanced");
  d.append(el("summary", title), el("pre", JSON.stringify(value, null, 2)));
  parent.append(d);
  return d;
}
function disclosure(parent, title, build, open = false) {
  const d = el("details");
  d.open = open;
  d.append(el("summary", title));
  const body = el("div");
  build(body);
  d.append(body);
  parent.append(d);
  return d;
}
function section(parent, title, note = "") {
  const node = el("div", "", "section-title");
  node.append(el("h2", title));
  if (note) node.append(el("small", note));
  parent.append(node);
}
function notice(parent, text, tone = "") {
  parent.append(el("p", text, "notice " + tone));
}
function empty(parent, title, text = "") {
  const node = el("div", "", "empty");
  node.append(el("h3", title));
  if (text) node.append(el("p", text));
  parent.append(node);
}
function card(parent, title = "") {
  const node = el("section", "", "card");
  if (title) node.append(el("h3", title));
  parent.append(node);
  return node;
}
function inline(parent, text) {
  for (const part of String(text).split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g)) {
    if (part.startsWith("**") && part.endsWith("**"))
      parent.append(el("strong", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`"))
      parent.append(el("code", part.slice(1, -1)));
    else parent.append(document.createTextNode(part));
  }
}
// Render data as text nodes only. Dataset text never becomes executable HTML.
function rich(text) {
  const node = el("div", "", "rich");
  let paragraph = [],
    items = null,
    code = null,
    codeLines = [];
  const flush = () => {
    if (paragraph.length) {
      const p = el("p");
      inline(p, paragraph.join("\n"));
      node.append(p);
      paragraph = [];
    }
    items = null;
  };
  const lines = String(text || "").split("\n");
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (/^\s*```/.test(line)) {
      if (code) {
        code.textContent = codeLines.join("\n");
        node.append(code);
        code = null;
        codeLines = [];
      } else {
        flush();
        code = el("pre");
      }
      continue;
    }
    if (code) {
      codeLines.push(line);
      continue;
    }
    if (!line.trim()) {
      flush();
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)/);
    if (heading) {
      flush();
      const h = el("h4");
      inline(h, heading[1]);
      node.append(h);
      continue;
    }
    if (
      line.includes("|") &&
      index + 1 < lines.length &&
      /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(
        lines[index + 1],
      )
    ) {
      flush();
      const cells = (row) =>
        row
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((x) => x.trim());
      const headers = cells(line),
        rows = [];
      index++;
      while (
        index + 1 < lines.length &&
        lines[index + 1].includes("|") &&
        lines[index + 1].trim()
      ) {
        index++;
        rows.push(
          cells(lines[index]).map((text) => {
            const cell = el("span");
            inline(cell, text);
            return cell;
          }),
        );
      }
      table(node, headers, rows);
      continue;
    }
    const match = line.match(/^\s*(?:([-*+])\s+|\d+[.)]\s+)(.*)/);
    if (match) {
      if (paragraph.length) flush();
      const type = match[1] ? "ul" : "ol";
      if (!items || items.tagName.toLowerCase() !== type) {
        items = el(type);
        node.append(items);
      }
      const li = el("li");
      inline(li, match[2]);
      items.append(li);
      continue;
    }
    if (/^>\s?/.test(line)) {
      flush();
      const q = el("blockquote");
      inline(q, line.replace(/^>\s?/, ""));
      node.append(q);
      continue;
    }
    items = null;
    paragraph.push(line);
  }
  flush();
  if (code) {
    code.textContent = codeLines.join("\n");
    node.append(code);
  }
  return node;
}
function bodyText(parent, text, { collapsible = false } = {}) {
  const content = rich(text);
  parent.append(content);
  if (collapsible && String(text || "").length > 850) {
    content.className += " collapsed";
    let expanded = false;
    const toggle = button(
      "展开全文",
      () => {
        expanded = !expanded;
        content.classList.toggle("collapsed", !expanded);
        toggle.textContent = expanded ? "收起全文" : "展开全文";
        toggle.setAttribute("aria-expanded", String(expanded));
      },
      "text-button",
    );
    toggle.setAttribute("aria-expanded", "false");
    parent.append(toggle);
  }
  return content;
}
function table(parent, headers, rows) {
  const wrap = el("div", "", "table-wrap"),
    t = el("table", "", "data-table"),
    thead = el("thead"),
    tr = el("tr");
  for (const h of headers) tr.append(el("th", h));
  thead.append(tr);
  const tbody = el("tbody");
  for (const row of rows) {
    const r = el("tr");
    for (const cell of row) {
      const td = el("td");
      td.append(
        typeof cell === "string" || typeof cell === "number"
          ? document.createTextNode(String(cell))
          : cell,
      );
      r.append(td);
    }
    tbody.append(r);
  }
  t.append(thead, tbody);
  wrap.append(t);
  parent.append(wrap);
  return t;
}
function updateURL() {
  const p = new URLSearchParams();
  if (state.run) p.set("run", state.run);
  p.set("page", state.page);
  if (state.page === "samples" && state.sample) {
    p.set("sample", state.sample);
    p.set("stage", state.sampleTab);
  }
  if (state.page === "experiences" && state.experience) {
    p.set("experience", state.experience);
    p.set("view", state.experienceTab);
  }
  history.replaceState(null, "", "?" + p.toString());
}
function preserveScroll() {
  const old = $(".sidebar");
  if (old) state.scrolls[old.dataset.page || state.page] = old.scrollTop;
}
function selectSample(id, stage) {
  const target = samples().find((s) => s.sample_id === id);
  if (target && !sampleMatches(target)) {
    state.sampleSearch = "";
    state.sampleFilter = "all";
  }
  preserveScroll();
  state.revealList = state.page !== "samples" || Boolean(stage);
  state.page = "samples";
  state.sample = id;
  if (stage) state.sampleTab = stage;
  render();
}
function selectExperience(id) {
  const e = experiences().find((e) => e.experience_id === id);
  if (!e) return;
  if (
    state.search &&
    !`${experienceNumber(id)} ${id} ${e.method.trigger} ${e.method.principle}`
      .toLowerCase()
      .includes(state.search.toLowerCase())
  )
    state.search = "";
  preserveScroll();
  state.revealList = state.page !== "experiences";
  state.page = "experiences";
  state.experience = id;
  state.status = e.status === "retired" ? "retired" : "active";
  state.experienceTab = "method";
  render();
}
function caseLink(id, stage = "task") {
  return button(sampleNumber(id), () => selectSample(id, stage), "text-button");
}
function sourceLink(id) {
  const e = experiences().find((e) => e.experience_id === id);
  return e
    ? button(experienceNumber(id), () => selectExperience(id), "source-button")
    : badge("本轮候选", "blue");
}
function tabs(parent, entries, current, change) {
  const nav = el("nav", "", "detail-tabs");
  nav.setAttribute("aria-label", "详情导航");
  for (const [key, label] of entries) {
    const b = button(label, () => change(key), current === key ? "active" : "");
    b.setAttribute("aria-current", current === key ? "page" : "false");
    b.dataset.stage = key;
    nav.append(b);
  }
  parent.append(nav);
}
function setupDetail(panel, title, subtitle, badges = [], pager = null) {
  const head = el("div", "", "detail-head"),
    row = el("div", "", "detail-heading"),
    left = el("div");
  left.append(el("h1", title));
  if (subtitle) left.append(el("p", subtitle, "title-text"));
  row.append(left);
  if (pager) row.append(pager);
  head.append(row);
  if (badges.length) {
    const strip = el("div", "", "badge-row");
    strip.style.marginTop = "12px";
    strip.append(...badges);
    head.append(strip);
  }
  panel.append(head);
}
function detailBody(panel) {
  const body = el("div", "", "detail-content");
  panel.append(body);
  return body;
}
function render() {
  preserveScroll();
  const root = $("#app");
  root.replaceChildren();
  document.querySelectorAll("[data-page]").forEach((b) => {
    b.classList.toggle("active", b.dataset.page === state.page);
    b.setAttribute(
      "aria-current",
      b.dataset.page === state.page ? "page" : "false",
    );
  });
  if (!state.data) {
    empty(root, "暂无运行数据", "先选择一个新版运行，或刷新数据。");
    return;
  }
  if (state.page === "overview") {
    overview(root);
    updateURL();
    return;
  }
  const layout = el("div", "", "layout"),
    sidebar = el("aside", "", "sidebar"),
    panel = el("article", "", "detail");
  sidebar.dataset.page = state.page;
  layout.append(sidebar, panel);
  root.append(layout);
  if (state.page === "samples") renderSamples(sidebar, panel);
  else renderExperiences(sidebar, panel);
  sidebar.scrollTop = state.scrolls[state.page] || 0;
  if (state.revealList) {
    const active = sidebar.querySelector(".item.active"),
      filters = sidebar.querySelector(".filters");
    if (active) {
      const top = active.offsetTop || 0,
        height = filters?.offsetHeight || 0;
      if (
        top < sidebar.scrollTop + height ||
        top + (active.offsetHeight || 0) >
          sidebar.scrollTop + (sidebar.clientHeight || 0)
      )
        sidebar.scrollTop = Math.max(0, top - height - 10);
    }
    state.revealList = false;
  }
  updateURL();
}
function metric(parent, label, value, note) {
  const node = el("div", "", "metric");
  node.append(
    el("span", label),
    el("strong", String(value)),
    el("small", note),
  );
  parent.append(node);
}
function overview(root) {
  const ss = samples(),
    ee = experiences(),
    apps = ss.flatMap((s) => list(s.applications)).filter((a) => a.cited),
    reports = apps.flatMap((a) => list(a.reports));
  const initial = ss.filter((s) => s.initial_judgment?.success).length,
    final = ss.filter((s) => finalJudgment(s)?.success).length;
  const mode =
    state.data.manifest.run_mode || state.data.manifest.config?.run?.mode;
  const heading = el("div", "", "page-heading"),
    left = el("div");
  left.append(
    el("div", "RUN OVERVIEW", "eyebrow"),
    el("h1", mode === "collection" ? "在线采集 · 运行概览" : "实验运行概览"),
    el("p", state.data.name, "subtitle"),
  );
  heading.append(
    left,
    badge(
      state.data.manifest.state === "running" ? "运行中" : "运行已记录",
      state.data.manifest.state === "running" ? "blue" : "neutral",
    ),
  );
  root.append(heading);
  const metrics = el("div", "", "metrics");
  metric(
    metrics,
    "已处理样本",
    ss.length,
    mode === "collection" ? "仅训练数据 · 采集模式" : "按当前运行轨迹统计",
  );
  metric(
    metrics,
    "首次判断正确",
    `${initial} / ${ss.length}`,
    ss.length ? percent(initial / ss.length) : "暂无判断",
  );
  metric(
    metrics,
    "纠错后判断正确",
    `${final} / ${ss.length}`,
    "取最后一次有效判断",
  );
  metric(
    metrics,
    "活跃经验",
    ee.filter((e) => e.status === "active").length,
    `${ee.filter((e) => referenceCount(e) > 0).length}条获得下游引用`,
  );
  metric(
    metrics,
    "下游经验引用",
    apps.length,
    `${new Set(apps.map((a) => a.sample.sample_id)).size}个样本实际引用`,
  );
  metric(
    metrics,
    "Checker报告",
    reports.length,
    `${reports.filter((r) => r.status === "error").length}份报告存在错误`,
  );
  root.append(metrics);
  if (mode === "collection")
    notice(
      root,
      "当前为采集模式：Q未估计，Selector仅判断适用性，自动修订关闭。Checker意见是模型报告，不代表人工真值。",
    );
  const grid = el("div", "", "overview-grid"),
    feedback = el("section", "", "surface"),
    checks = el("section", "", "surface");
  feedback.append(
    el("h2", "经验在下游任务中的反馈"),
    el("p", "每个样本、每条实际引用经验汇总一次；不包含来源准入。", "muted"),
  );
  const counts = { positive: 0, reverse: 0, tie: 0 };
  for (const a of apps) if (a.feedback in counts) counts[a.feedback]++;
  const bar = el("div", "", "stacked-bar");
  for (const [key, tone] of [
    ["positive", "good"],
    ["reverse", "bad"],
    ["tie", "neutral"],
  ]) {
    const segment = el("div", "", "segment " + tone);
    segment.style.width =
      (apps.length ? (counts[key] / apps.length) * 100 : 0) + "%";
    bar.append(segment);
  }
  feedback.append(bar);
  const legend = el("div", "", "legend");
  for (const key of Object.keys(counts)) {
    const x = el("span", FEEDBACK[key][0]);
    x.append(el("b", String(counts[key])));
    legend.append(x);
  }
  feedback.append(legend);
  checks.append(el("h2", "Checker核对意见"));
  table(
    checks,
    ["核对环节", "执行正确", "存在错误", "未知／失败"],
    Object.keys(STAGES).map((stage) => [
      STAGES[stage][0],
      String(
        reports.filter((r) => r.stage === stage && r.status === "correct")
          .length,
      ),
      String(
        reports.filter((r) => r.stage === stage && r.status === "error").length,
      ),
      String(
        reports.filter(
          (r) => r.stage === stage && ["unknown", "failed"].includes(r.status),
        ).length,
      ),
    ]),
  );
  grid.append(feedback, checks);
  root.append(grid);
  section(root, "待查看的样本", "按执行顺序 · 包含Checker报错或格式／调用问题");
  const review = el("section", "", "surface"),
    flagged = ss.filter((s) => hasCheckerError(s) || hasOutputError(s));
  if (!flagged.length) empty(review, "暂无已记录的异常意见");
  for (const s of flagged) {
    const row = el("div", "", "review-item"),
      content = el("div", "", "grow");
    content.append(
      el("p", compact(s.task, 120)),
      el(
        "small",
        hasOutputError(s)
          ? "包含格式纠错或调用问题"
          : "Checker报告存在执行错误，需对照原文核实",
      ),
    );
    row.append(
      badge(sampleNumber(s.sample_id), "blue"),
      content,
      button("查看过程", () =>
        selectSample(s.sample_id, hasOutputError(s) ? "critic" : "checks"),
      ),
    );
    review.append(row);
  }
  root.append(review);
  raw(root, "运行配置与概率参数来源", state.data.manifest);
}
function listFilters(sidebar, title) {
  const filters = el("div", "", "filters"),
    head = el("div", "", "filter-heading"),
    count = el("small");
  head.append(el("h2", title), count);
  filters.append(head);
  const items = el("div", "", "list-items");
  sidebar.append(filters, items);
  return { filters, items, count };
}
function selectOptions(options, label) {
  const node = el("select");
  node.setAttribute("aria-label", label);
  for (const [value, text] of options) {
    const o = el("option", text);
    o.value = value;
    node.append(o);
  }
  return node;
}
function sampleMatches(s) {
  const query = state.sampleSearch.trim().toLowerCase(),
    index = samples().indexOf(s) + 1;
  if (/^#\d+$/.test(query) && index !== Number(query.slice(1))) return false;
  if (
    query &&
    !/^#\d+$/.test(query) &&
    !`${index} ${String(index).padStart(2, "0")} ${s.sample_id} ${s.task}`
      .toLowerCase()
      .includes(query)
  )
    return false;
  switch (state.sampleFilter) {
    case "failed":
      return Boolean(s.initial_judgment && !s.initial_judgment.success);
    case "cited":
      return list(s.applications).some((a) => a.cited);
    case "checker":
      return hasCheckerError(s);
    case "errors":
      return hasOutputError(s);
    default:
      return true;
  }
}
function renderSamples(sidebar, panel) {
  const ss = samples();
  if (!ss.some((s) => s.sample_id === state.sample))
    state.sample = ss[0]?.sample_id;
  const ui = listFilters(sidebar, "样本流水线"),
    search = el("input");
  search.placeholder = "搜索任务、样本编号或ID";
  search.setAttribute("aria-label", "搜索样本");
  search.value = state.sampleSearch;
  const filter = selectOptions(
    [
      ["all", "全部样本"],
      ["failed", "首次判断未成功"],
      ["cited", "实际引用了经验"],
      ["checker", "Checker报告错误"],
      ["errors", "格式或调用问题"],
    ],
    "样本筛选",
  );
  filter.value = state.sampleFilter;
  const fill = () => {
    const rows = ss.filter(sampleMatches);
    if (!rows.some((s) => s.sample_id === state.sample))
      state.sample = rows[0]?.sample_id;
    ui.items.replaceChildren();
    ui.count.textContent = `${rows.length} / ${ss.length}`;
    for (const s of rows) {
      const n = ss.indexOf(s) + 1,
        b = button(
          "",
          () => selectSample(s.sample_id),
          "item" + (s.sample_id === state.sample ? " active" : ""),
        );
      b.dataset.sample = s.sample_id;
      const top = el("div", "", "item-top");
      top.append(
        el("span", "#" + String(n).padStart(2, "0"), "item-number"),
        stateBadge(s.state),
      );
      b.append(top, el("span", compact(s.task, 135), "item-title"));
      const foot = el("div", "", "item-foot");
      const cited = list(s.applications).filter((a) => a.cited).length;
      foot.append(
        el("span", `${list(s.initial_rubric?.criteria).length}条标准`),
        el("span", cited ? `引用${cited}条经验` : "无经验引用"),
      );
      if (hasCheckerError(s)) foot.append(el("span", "核对报错", "error"));
      if (list(s.experience_updates).length)
        foot.append(el("span", "触发修订"));
      b.append(foot);
      ui.items.append(b);
    }
    if (!rows.length)
      empty(ui.items, "没有匹配样本", "尝试修改搜索词或筛选条件。");
  };
  search.oninput = () => {
    state.sampleSearch = search.value;
    fill();
    sampleDetail(
      panel,
      ss.find((s) => s.sample_id === state.sample),
    );
    updateURL();
  };
  filter.onchange = () => {
    state.sampleFilter = filter.value;
    sidebar.scrollTop = 0;
    fill();
    sampleDetail(
      panel,
      ss.find((s) => s.sample_id === state.sample),
    );
    updateURL();
  };
  ui.filters.append(search, filter);
  fill();
  sampleDetail(
    panel,
    ss.find((s) => s.sample_id === state.sample),
  );
}
function sampleDetail(panel, s) {
  panel.replaceChildren();
  const ss = samples();
  if (!s) {
    empty(panel, "暂无样本");
    return;
  }
  const i = ss.indexOf(s),
    pager = el("div", "", "pager"),
    previous = button("← 上一题", () => selectSample(ss[i - 1].sample_id)),
    next = button("下一题 →", () => selectSample(ss[i + 1].sample_id));
  previous.disabled = i === 0;
  next.disabled = i === ss.length - 1;
  pager.append(previous, next);
  setupDetail(
    panel,
    `${sampleNumber(s.sample_id)} / ${ss.length}`,
    compact(s.task, 210),
    [
      stateBadge(s.state),
      badge(
        "首次：" +
          (s.initial_judgment
            ? s.initial_judgment.success
              ? "正确"
              : "未成功"
            : "无有效结果"),
        s.initial_judgment?.success ? "good" : "neutral",
      ),
      badge(`${list(s.proposal_attempts).length}轮纠错`),
    ],
    pager,
  );
  tabs(
    panel,
    [
      ["task", "任务与回答"],
      ["rubric", "评分标准"],
      ["checks", "经验与核对"],
      ["critic", "Critic纠错"],
      ["io", "调用记录"],
    ],
    state.sampleTab,
    (key) => {
      state.sampleTab = key;
      render();
    },
  );
  const body = detailBody(panel);
  if (state.sampleTab === "task") taskView(body, s);
  else if (state.sampleTab === "rubric")
    rubricView(body, s.initial_rubric, s.initial_judgment);
  else if (state.sampleTab === "checks") checksView(body, s);
  else if (state.sampleTab === "critic") criticView(body, s);
  else ioView(body, s);
  raw(body, "样本标识与协议元数据", {
    sample_id: s.sample_id,
    state: s.state,
    split: s.split,
    schema_version: s.schema_version,
  });
}
function taskView(parent, s) {
  const task = card(parent),
    head = el("div", "", "card-head");
  head.append(
    el("h3", "任务"),
    el("small", `${(s.task || "").length.toLocaleString()}字符`),
  );
  task.append(head);
  bodyText(task, s.task, { collapsible: true });
  translationButton(task, s, "task");
  section(parent, "回答对比", "优选与非优选来自数据集标签，不代表Judge的判断");
  const pair = el("div", "", "answers");
  for (const [key, title] of [
    ["chosen", "优选回答"],
    ["rejected", "非优选回答"],
  ]) {
    const node = card(pair);
    node.className += " answer-card " + key;
    const h = el("div", "", "card-head");
    h.append(
      el("h3", title),
      el("small", `${(s[key] || "").length.toLocaleString()}字符`),
    );
    node.append(h);
    bodyText(node, s[key], { collapsible: true });
    translationButton(node, s, key);
  }
  parent.append(pair);
}
function translationButton(parent, s, field) {
  const action = el("div", "", "content-actions"),
    b = button(
      "翻译为中文",
      async () => {
        b.disabled = true;
        b.textContent = "正在翻译…";
        const result = el("div", "", "translation");
        try {
          const response = await fetch("/api/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              run: state.run,
              sample_id: s.sample_id,
              field,
            }),
          });
          const data = await response.json();
          result.append(rich(data.translation || data.error || "翻译暂不可用"));
          if (!data.translation) result.className += " error";
        } catch (error) {
          result.append(el("p", String(error)));
          result.className += " error";
        } finally {
          b.disabled = false;
          b.textContent = "翻译为中文";
        }
        const previous = parent.querySelector(".translation");
        if (previous) previous.remove();
        parent.append(result);
      },
      "text-button",
    );
  action.append(b);
  parent.append(action);
}
function rubricView(parent, r, j, onlySource = null) {
  if (!r) {
    empty(
      parent,
      "尚无有效标准",
      "本阶段没有生成有效rubric，错误详情见纠错或调用记录。",
    );
    return;
  }
  if (j && !onlySource) {
    const strip = el("div", "", "score-strip");
    for (const [label, value] of [
      ["优选回答得分", j.chosen_score],
      ["非优选回答得分", j.rejected_score],
    ]) {
      const block = el("div");
      block.append(
        el("span", label, "score-label"),
        el("div", percent(value), "score"),
      );
      strip.append(block);
    }
    strip.append(
      el("div", "", "separator"),
      badge(
        j.success ? "正确区分数据集偏好" : "未正确区分数据集偏好",
        j.success ? "good" : "warn",
      ),
    );
    parent.append(strip);
  }
  if (r.analysis && !onlySource)
    disclosure(parent, "Generator的设计思路", (body) =>
      bodyText(body, r.analysis),
    );
  const criteria = list(r.criteria).filter(
    (c) => !onlySource || list(c.source_experiences).includes(onlySource),
  );
  section(
    parent,
    onlySource ? "该经验对应的评分标准" : "逐条评分标准",
    `${criteria.length}条 · 权重按整套标准归一化`,
  );
  const judgments = new Map(
    list(j?.criterion_judgments).map((x) => [x.criterion_id, x]),
  );
  const total = list(r.criteria).reduce((sum, c) => sum + c.weight, 0);
  for (const c of criteria) {
    const index = r.criteria.indexOf(c) + 1,
      node = card(parent),
      top = el("div", "", "criterion-head"),
      left = el("div", "", "badge-row");
    left.append(
      el("span", "标准 " + String(index).padStart(2, "0"), "criterion-index"),
      badge(
        "权重 " +
          percent(
            j?.normalized_weights?.[c.criterion_id] ??
              (total ? c.weight / total : 0),
          ),
      ),
    );
    for (const id of list(c.source_experiences)) left.append(sourceLink(id));
    if (!list(c.source_experiences).length) left.append(badge("任务基础标准"));
    top.append(left);
    node.append(top, el("p", c.question, "criterion-question"));
    const result = judgments.get(c.criterion_id),
      row = el("div", "", "criterion-result");
    for (const [key, title] of [
      ["chosen", "优选回答"],
      ["rejected", "非优选回答"],
    ]) {
      const cell = el("div", "", "result-pair");
      cell.append(
        el("span", title),
        badge(
          result ? (result[key] ? "满足" : "未满足") : "未返回",
          result ? (result[key] ? "good" : "neutral") : "warn",
        ),
      );
      row.append(cell);
    }
    node.append(row);
    if (result)
      disclosure(node, "查看两侧判断依据", (body) => {
        const pair = el("div", "", "judgment-grid");
        for (const [key, title] of [
          ["chosen_reason", "优选回答 · Judge依据"],
          ["rejected_reason", "非优选回答 · Judge依据"],
        ]) {
          const block = el("div", "", "judgment-reason");
          block.append(el("h4", title), el("p", result[key] || "未提供"));
          pair.append(block);
        }
        body.append(pair);
      });

  }
}
function checksView(parent, s) {
  const apps = [...list(s.applications), ...list(s.proposal_attempts).flatMap((p) => list(p.applications).filter((a) => a.phase === "source_validation"))],
    cited = apps.filter((a) => a.cited);
  section(
    parent,
    "经验应用与核对",
    `选中${list(s.selected_experiences).length}条 · 实际引用${cited.length}条`,
  );
  notice(
    parent,
    "三个Checker分别核对选择、方法应用和Judge执行。以下是模型意见，报告错误不等于已证实的经验缺陷。",
  );
  if (!apps.length)
    empty(
      parent,
      "本题没有已有经验的应用记录",
      list(s.proposal_attempts).length
        ? "候选准入过程请看Critic纠错；准入阶段不运行Checker。"
        : "未选中已有经验，因此不运行Checker。",
    );
  for (const a of apps) applicationView(parent, a);
  disclosure(parent, "检索、选择与排名", (body) => {
    const selected = new Set(
      list(s.selected_experiences).map((e) => e.experience_id),
    );
    if (!list(s.retrieved_experience_ids).length) {
      empty(body, "没有召回经验", "空经验池或未返回候选。");
      return;
    }
    table(
      body,
      ["排序", "经验", "向量排名", "关键词排名", "是否选中"],
      list(s.retrieved_experience_ids).map((id, index) => {
        const trace = list(s.retrieval_trace).find(
          (x) => x.experience_id === id,
        );
        return [
          String(index + 1),
          sourceLink(id),
          String(trace?.vector_rank ?? "—"),
          String(trace?.keyword_rank ?? "—"),
          selected.has(id) ? "已选中" : "未选中",
        ];
      }),
    );
    raw(body, "实际召回快照", s.selected_experiences);
  });
}
function applicationView(parent, a, { showMethod = true } = {}) {
  const node = card(parent),
    head = el("div", "", "card-head"),
    left = el("div", "", "badge-row");
  left.append(
    sourceLink(a.version_id),
    badge(PHASE[a.phase] || a.phase),
    feedbackBadge(a.feedback),
  );
  head.append(left);
  node.append(head);
  if (showMethod) {
    node.append(el("p", a.method.principle, "application-title"));
    const context = el("div", "", "context-block");
    context.append(el("strong", "适用条件"), el("p", a.method.trigger));
    node.append(context);
  }
  const flow = el("div", "", "flow");
  for (const [key, label] of [
    ["recalled", "召回"],
    ["selected", "选中"],
    ["cited", "引用"],
  ]) {
    if (flow.children.length) flow.append(el("span", "→", "arrow"));
    flow.append(el("span", (a[key] ? "✓ " : "— ") + label, a[key] ? "on" : ""));
  }
  if (a.retrieval_rank)
    flow.append(el("span", "召回第" + a.retrieval_rank + "位"));
  node.append(flow);
  if (!a.cited) {
    notice(
      node,
      "已选中，但Generator没有引用这条经验；不运行Checker，也不更新Q。",
    );
    return;
  }
  const reports = new Map(list(a.reports).map((r) => [r.stage, r]));
  if (a.phase === "admission" && !reports.size)
    notice(node, "来源准入案例不运行Checker，不重复更新Q。");
  else {
    const grid = el("div", "", "checker-grid");
    for (const stage of Object.keys(STAGES)) {
      const report = reports.get(stage),
        tone = REPORT[report?.status]?.[1] || "neutral",
        block = el("section", "", "checker " + tone);
      block.append(
        el("h4", STAGES[stage][0]),
        el("small", STAGES[stage][1]),
        reportBadge(report?.status),
      );
      if (report) block.append(el("p", report.evidence));
      else block.append(el("p", "本次未产生核对报告。"));
      if (list(report?.details).length)
        disclosure(block, `逐标准检查 · ${report.details.length}项`, (body) => {
          for (const d of report.details) {
            const idx = list(a.rubric?.criteria).findIndex(
                (c) => c.criterion_id === d.criterion_id,
              ),
              detail = el("div", "", "checker-detail");
            detail.append(
              el(
                "strong",
                idx >= 0
                  ? "标准 " + String(idx + 1).padStart(2, "0")
                  : "标准编号未匹配",
              ),
              el("span", " "),
              reportBadge(d.status),
              el("p", d.evidence),
            );
            body.append(detail);
          }
        });
      grid.append(block);
    }
    node.append(grid);
  }
  const meta = el("div", "", "metadata");
  meta.append(
    el(
      "span",
      "综合核对：" +
        ({
          correct: "执行正确",
          error: "报告存在错误",
          unknown: "无法确定",
          unavailable: "报告不可用",
        }[a.observation] || "未记录"),
    ),
    el("span", REASONS[a.update_reason] || a.update_reason || "未更新Q"),
    el(
      "span",
      `正向权重 ${percent(a.positive_weight)} / 反向权重 ${percent(a.reverse_weight)}`,
    ),
  );
  if (a.q_before != null)
    meta.append(el("span", `Q ${percent(a.q_before)} → ${percent(a.q_after)}`));
  node.append(meta);
  disclosure(node, "查看引用标准与两侧判断", (body) =>
    rubricView(body, a.rubric, a.judgment, a.version_id),
  );
  raw(node, "应用原始数据", a);
}
function methodView(parent, m) {
  section(parent, "核验方法");
  const node = card(parent);
  bodyText(node, m.principle);
  section(parent, "适用条件");
  const trigger = card(parent);
  bodyText(trigger, m.trigger);

}
function criticView(parent, s) {
  if (s.sample_error)
    notice(parent, `${s.sample_error.stage}：${s.sample_error.message}`, "bad");
  const attempts = list(s.proposal_attempts);
  if (!attempts.length) {
    empty(
      parent,
      "没有触发Critic纠错",
      s.initial_judgment?.success
        ? "首次判断已正确，本题不再提出候选经验。"
        : "本题尚无候选纠错记录。",
    );
    if (list(s.experience_updates).length) {
      section(parent, "本题触发的版本修订");
      for (const event of s.experience_updates) revisionView(parent, event);
    }
    return;
  }
  notice(
    parent,
    "经验核对未通过才反馈Critic；通过即入池，再仅用task＋C做来源验证。来源验证结果不再触发重提。历史实验按当时协议查看。",
  );
  const admitted = new Set(list(s.admitted_experience_ids));
  for (const attempt of attempts) {
    const round = el("section", "", "round"),
      head = el("div", "", "round-header");
    head.append(
      el("h2", "第" + attempt.round_number + "轮纠错"),
      badge(`${list(attempt.proposal?.candidates).length}条候选`),
    );
    if (attempt.judgment)
      head.append(
        badge(
          attempt.judgment.success ? "整体判断正确" : "整体判断未成功",
          attempt.judgment.success ? "good" : "warn",
        ),
      );
    round.append(head);
    if (attempt.proposal?.analysis) {
      const analysis = el("div", "", "analysis");
      analysis.append(el("h4", "Critic诊断"), rich(attempt.proposal.analysis));
      round.append(analysis);
    }
    if (attempt.review) {
      const review = card(round);
      review.append(badge(attempt.review.approved ? "经验核对通过 · 允许入池" : "经验核对未通过 · 反馈Critic", attempt.review.approved ? "good" : "warn"), el("p", attempt.review.reason));
    }
    if (attempt.error) notice(round, attempt.error, "bad");
    const grid = el("div", "", "candidate-grid");
    list(attempt.proposal?.candidates).forEach((e, index) => {
      const c = el("section", "", "candidate"),
        top = el("div", "", "card-head"),
        row = el("div", "", "badge-row"),
        a = list(attempt.applications).find(
          (a) => a.version_id === e.experience_id,
        );
      row.append(
        badge("候选 C" + (index + 1), "blue"),
        feedbackBadge(a?.feedback),
      );
      if (admitted.has(e.experience_id))
        row.append(badge("最终已入池", "good"));
      top.append(row);
      c.append(
        top,
        el("p", e.method.principle),
        el("p", "适用条件：" + e.method.trigger, "trigger"),
      );
      grid.append(c);
    });
    round.append(grid);
    if (attempt.rubric)
      disclosure(round, "本轮重新生成的标准与判断", (body) =>
        rubricView(body, attempt.rubric, attempt.judgment),
      );
    parent.append(round);
  }
  if (list(s.experience_updates).length) {
    section(parent, "本题触发的版本修订");
    for (const event of s.experience_updates) revisionView(parent, event);
  }
}
function renderExperiences(sidebar, panel) {
  const ee = experiences(),
    ui = listFilters(sidebar, "经验池"),
    search = el("input");
  search.placeholder = "搜索方法、适用条件或ID";
  search.setAttribute("aria-label", "搜索经验");
  search.value = state.search;
  const controls = el("div", "", "filter-pair"),
    status = selectOptions(
      [
        ["active", "活跃经验"],
        ["retired", "退役经验"],
      ],
      "经验状态",
    ),
    sort = selectOptions(
      [
        ["quality", "按质量后验"],
        ["references", "按下游引用次数"],
      ],
      "经验排序",
    );
  status.value = state.status;
  sort.value = state.sort;
  controls.append(status, sort);
  ui.filters.append(search, controls);
  const rows = () =>
    ee
      .filter(
        (e) =>
          e.status === state.status &&
          `${experienceNumber(e.experience_id)} ${e.experience_id} ${e.method.trigger} ${e.method.principle}`
            .toLowerCase()
            .includes(state.search.toLowerCase()),
      )
      .sort((a, b) =>
        state.sort === "references"
          ? referenceCount(b) - referenceCount(a)
          : (b.q_posterior ?? -1) - (a.q_posterior ?? -1),
      );
  if (
    !ee.some(
      (e) => e.experience_id === state.experience && e.status === state.status,
    )
  )
    state.experience = rows()[0]?.experience_id;
  const fill = () => {
    const filtered = rows();
    if (!filtered.some((e) => e.experience_id === state.experience))
      state.experience = filtered[0]?.experience_id;
    ui.items.replaceChildren();
    ui.count.textContent = `${filtered.length} / ${ee.filter((e) => e.status === state.status).length}`;
    for (const e of filtered) {
      const b = button(
        "",
        () => {
          state.experience = e.experience_id;
          render();
        },
        "item" + (e.experience_id === state.experience ? " active" : ""),
      );
      b.dataset.experience = e.experience_id;
      const top = el("div", "", "item-top");
      top.append(
        el("span", experienceNumber(e.experience_id), "item-number"),
        badge(
          e.q_posterior == null ? "未估计Q" : percent(e.q_posterior),
          e.q_posterior == null ? "neutral" : "blue",
        ),
      );
      const foot = el("div", "", "item-foot");
      foot.append(
        el("span", `下游引用 ${referenceCount(e)}题`),
        el("span", `记录 ${list(e.applications).length}条`),
      );
      b.append(top, el("span", e.method.principle, "item-title"), foot);
      ui.items.append(b);
    }
    if (!filtered.length)
      empty(ui.items, "没有匹配经验", "尝试修改搜索词或状态。");
  };
  search.oninput = () => {
    state.search = search.value;
    fill();
    experienceDetail(
      panel,
      ee.find((e) => e.experience_id === state.experience),
    );
    updateURL();
  };
  status.onchange = () => {
    state.status = status.value;
    state.experience = null;
    render();
    const current = $(".sidebar");
    if (current) current.scrollTop = 0;
  };
  sort.onchange = () => {
    state.sort = sort.value;
    render();
  };
  fill();
  experienceDetail(
    panel,
    ee.find((e) => e.experience_id === state.experience),
  );
}
function experienceDetail(panel, e) {
  panel.replaceChildren();
  if (!e) {
    empty(panel, "没有匹配的经验");
    return;
  }
  setupDetail(
    panel,
    "经验 " + experienceNumber(e.experience_id),
    compact(e.method.principle, 220),
    [
      badge(
        e.status === "active" ? "活跃版本" : "退役版本",
        e.status === "active" ? "good" : "neutral",
      ),
      qualityBadge(e),
      badge(`下游引用 ${referenceCount(e)}题`),
    ],
  );
  tabs(
    panel,
    [
      ["method", "方法与适用条件"],
      ["history", "应用与核对"],
      ["evolution", "版本演化"],
    ],
    state.experienceTab,
    (key) => {
      state.experienceTab = key;
      render();
    },
  );
  const body = detailBody(panel);
  if (state.experienceTab === "method") {
    methodView(body, e.method);
    if (e.admission_review) {
      section(body, "入池核对");
      card(body).append(badge(e.admission_review.approved ? "经验核对通过" : "未通过", e.admission_review.approved ? "good" : "warn"), el("p", e.admission_review.reason));
    }
    section(body, "质量与证据");
    if (e.q_posterior == null)
      notice(body, "Q未估计：尚未加载可用的概率参数，不表示该经验质量低。");
    else if (!list(e.applications).some((a) => a.update_reason === "updated"))
      notice(body, "当前Q仅为基础先验，尚无实际质量更新，证据不足。");
    table(
      body,
      ["证据来源", "记录数", "实际更新Q"],
      ["source_validation", "downstream", "replay"].map((phase) => [
        PHASE[phase],
        String(list(e.applications).filter((a) => a.phase === phase).length),
        String(
          list(e.applications).filter(
            (a) => a.phase === phase && a.update_reason === "updated",
          ).length,
        ),
      ]),
    );
    const counts = { positive: 0, reverse: 0, both_true: 0, both_false: 0 };
    for (const a of list(e.applications))
      for (const k in counts) counts[k] += a.outcomes?.[k] || 0;
    section(body, "原始逐标准反馈", "包含来源、下游、回放；不是独立任务数");
    table(
      body,
      ["正向区分", "反向区分", "双方满足", "双方不满足"],
      [[counts.positive, counts.reverse, counts.both_true, counts.both_false]],
    );
    const links = el("div", "", "badge-row");
    links.append(el("span", "来源：", "muted"), caseLink(e.source_sample_id));
    if (e.parent_id)
      links.append(el("span", "父版本：", "muted"), sourceLink(e.parent_id));
    body.append(links);
    raw(body, "版本标识与概率参数来源", {
      experience_id: e.experience_id,
      parent_id: e.parent_id,
      q_prior: e.q_prior,
      q_posterior: e.q_posterior,
      parameter_version: e.parameter_version,
      parameter_digest: e.parameter_digest,
    });
  } else if (state.experienceTab === "history") {
    const filters = el("div", "", "inline-filter"),
      phase = selectOptions(
        [
          ["all", "全部应用"],
          ["downstream", "仅下游应用"],
          ["source_validation", "来源验证"],
          ["replay", "仅修订回放"],
        ],
        "应用阶段",
      );
    phase.value = state.appPhase;
    filters.append(el("span", "证据来源"), phase);
    body.append(filters);
    const container = el("div");
    body.append(container);
    const fillHistory = () => {
      container.replaceChildren();
      const apps = list(e.applications).filter(
        (a) => state.appPhase === "all" || a.phase === state.appPhase,
      );
      if (!apps.length) {
        empty(container, "暂无此类应用记录");
        return;
      }
      for (const a of apps) {
        const d = el("details", "", "history-item"),
          summary = el("summary"),
          description = el("div", "", "grow");
        description.append(
          el("strong", sampleNumber(a.sample.sample_id)),
          el("p", compact(a.sample.task, 130)),
        );
        summary.append(
          description,
          badge(PHASE[a.phase] || a.phase),
          feedbackBadge(a.feedback),
        );
        d.append(summary);
        const content = el("div");
        content.append(caseLink(a.sample.sample_id, "checks"));
        applicationView(content, a, { showMethod: false });
        d.append(content);
        container.append(d);
      }
    };
    phase.onchange = () => {
      state.appPhase = phase.value;
      fillHistory();
    };
    fillHistory();
  } else {
    notice(body, "回放使用归纳看过的案例，只检验局部修复，不是独立泛化验证。");
    const events = list(state.data.revisions).filter(
      (r) =>
        r.parent_id === e.experience_id ||
        r.candidate?.experience_id === e.experience_id,
    );
    if (!events.length)
      empty(body, "该版本尚无修订记录", "采集模式关闭自动修订。");
    for (const event of events) revisionView(body, event);
  }
}
function revisionView(parent, event) {
  const node = card(parent),
    head = el("div", "", "card-head");
  head.append(
    el("h3", event.accepted ? "已接受新版本" : "保留旧版本"),
    badge(
      event.action === "keep"
        ? "Keep · 保持"
        : event.action === "revise"
          ? "Revise · 修订"
          : event.action || "未完成",
    ),
  );
  node.append(head);
  if (event.update?.reason) bodyText(node, event.update.reason);
  if (event.update?.method)
    disclosure(node, "本次提出的方法修订", (body) =>
      methodView(body, event.update.method),
    );
  if (event.error) notice(node, event.error, "bad");
  const links = el("div", "", "badge-row");
  links.append(
    el("span", "触发样本：", "muted"),
    caseLink(event.trigger_sample_id),
    sourceLink(event.parent_id),
  );
  if (event.candidate) links.append(sourceLink(event.candidate.experience_id));
  node.append(links);
  if (list(event.replays).length)
    table(
      node,
      ["回放样本", "旧版得分", "新版得分", "状态"],
      event.replays.map((p) => [
        caseLink(p.sample_id),
        String(p.old_score ?? "缺失"),
        String(p.new_score ?? "缺失"),
        Object.keys(p.errors || {}).length ? "含调用失败" : "有效结果",
      ]),
    );
  if (list(event.case_refs).length)
    disclosure(node, "归纳案例的版本与用途", (body) =>
      table(
        body,
        ["样本", "实际版本", "阶段／轮次", "用途"],
        event.case_refs.map((c) => [
          caseLink(c.sample_id),
          sourceLink(c.version_id),
          `${PHASE[c.phase] || c.phase} / ${c.round_number}`,
          {
            problem: "问题案例",
            source: "来源案例",
            positive: "历史正向",
            applicability_dispute: "适用性争议",
            recent: "近期补充",
          }[c.kind] || c.kind,
        ]),
      ),
    );
  raw(node, "原始修订判据", event);
}
function ioView(parent, s) {
  section(parent, "原始角色调用", "按实际记录顺序 · 包含有限纠错");
  const key = state.run + ":" + s.sample_id,
    cache = state.io[key];
  if (!cache) {
    const prompt = card(parent);
    prompt.append(
      el(
        "p",
        "按需加载本题的原始输入、回复及完整消息，方便对照格式纠错过程。",
        "muted",
      ),
      button("加载原始角色请求／回复", async () => {
        state.io[key] = { loading: true };
        render();
        try {
          const response = await fetch(
            "/api/role-io?" +
              new URLSearchParams({ name: state.run, sample_id: s.sample_id }),
          );
          if (!response.ok) throw new Error(await response.text());
          state.io[key] = await response.json();
        } catch (error) {
          state.io[key] = { error: String(error) };
        }
        if (
          state.page === "samples" &&
          state.sample === s.sample_id &&
          state.sampleTab === "io"
        )
          render();
      }),
    );
    return;
  }
  if (cache.loading) {
    empty(parent, "正在加载调用记录…");
    return;
  }
  if (cache.error) {
    notice(parent, cache.error, "bad");
    parent.append(
      button("重试加载", () => {
        delete state.io[key];
        render();
      }),
    );
    return;
  }
  for (const warning of list(cache.warnings)) notice(parent, warning, "warn");
  if (
    state.role !== "all" &&
    !list(cache.entries).some((r) => r.role === state.role)
  )
    state.role = "all";
  const toolbar = el("div", "", "role-toolbar"),
    role = selectOptions(
      [
        ["all", "全部LLM角色"],
        ...Object.entries(ROLES)
          .filter(([key]) => list(cache.entries).some((r) => r.role === key))
          .map(([key, name]) => [key, name]),
      ],
      "调用角色筛选",
    );
  role.value = state.role;
  const callCount = el("small", "", "muted");
  toolbar.append(role, callCount);
  parent.append(toolbar);
  const calls = el("div");
  parent.append(calls);
  const fill = () => {
    calls.replaceChildren();
    const rows = list(cache.entries)
      .map((entry, index) => ({ entry, index }))
      .filter(({ entry }) =>
        state.role === "all"
          ? entry.role !== "embedding"
          : entry.role === state.role,
      );
    callCount.textContent = `${rows.length} / ${list(cache.entries).length}条记录`;
    if (!rows.length) empty(calls, "暂无该角色的调用记录");
    for (const { entry: r, index } of rows) {
      const d = el("details", "", "call-item"),
        summary = el("summary"),
        label = el("span", ROLES[r.role] || r.role, "grow");
      summary.append(
        el("span", String(index + 1).padStart(2, "0"), "item-number"),
        label,
      );
      if (r.response_side)
        summary.append(
          badge(r.response_side === "chosen" ? "优选回答" : "非优选回答"),
        );
      const callError = r.contract_error || r.error;
      summary.append(badge(
        r.validation_status === "invalid" ? "结构校验失败" : r.error ? "请求失败" : r.validation_status === "valid" ? "校验通过" : "模型已返回",
        callError ? "bad" : r.validation_status === "valid" ? "good" : "neutral",
      ));
      if (callError) summary.append(el("span", String(callError), "call-error"));
      d.append(summary);
      const content = el("div");
      content.append(
        el(
          "p",
          `${r.model || "未记录模型"} · 网络尝试 ${r.attempt ?? "未记录"} · ${r.prompt_version || ""}`,
          "subtitle",
        ),
      );
      if (callError) notice(content, String(callError), "bad");
      if (list(r.messages).length) {
        disclosure(
          content,
          `发送给模型的提示词 · ${r.messages.length}条消息`,
          (body) => {
            list(r.messages).forEach((message, messageIndex) => {
              const block = el("section", "", "message");
              const names = {
                system: "系统提示词 · System",
                developer: "开发者指令 · Developer",
                user: "用户输入 · User",
                assistant: "历史模型回复 · Assistant",
                tool: "工具返回 · Tool",
              };
              block.append(
                el(
                  "div",
                  `${messageIndex + 1}. ${names[message.role] || message.role}`,
                  "role-name",
                ),
                el(
                  "pre",
                  typeof message.content === "string"
                    ? message.content
                    : JSON.stringify(message.content, null, 2),
                ),
              );
              body.append(block);
            });
          },
          true,
        );
      } else {
        notice(
          content,
          "此记录没有保存完整消息，无法还原实际发送的系统提示词。下方仅提供已记录的结构化输入。",
          "warn",
        );
      }
      if (r.raw_output != null)
        disclosure(
          content,
          "模型回复 · 原始文本",
          (body) => body.append(el("pre", String(r.raw_output))),
          true,
        );
      else if (r.output != null)
        raw(content, "模型回复 · 结构化结果", r.output);
      raw(content, "结构化输入数据（非完整提示词）", r.input);
      raw(content, "完整调用事件（含编号映射）", r);
      d.append(content);
      calls.append(d);
    }
  };
  role.onchange = () => {
    state.role = role.value;
    fill();
  };
  fill();
}
async function load() {
  const sequence = ++state.loadSequence,
    requestRun = state.run;
  const refresh = $("#refresh");
  refresh.disabled = true;
  refresh.textContent = "读取中…";
  try {
    const response = await fetch(
      "/api/run?name=" + encodeURIComponent(state.run),
    );
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    if (sequence !== state.loadSequence || requestRun !== state.run) return;
    state.data = data;
    render();
  } catch (error) {
    if (sequence !== state.loadSequence || requestRun !== state.run) return;
    $("#app").replaceChildren();
    empty($("#app"), "读取运行失败", String(error));
  } finally {
    if (sequence === state.loadSequence) {
      refresh.disabled = false;
      refresh.textContent = "刷新数据";
    }
  }
}
document.querySelectorAll("[data-page]").forEach(
  (b) =>
    (b.onclick = () => {
      preserveScroll();
      state.page = b.dataset.page;
      render();
    }),
);
$("#refresh").onclick = () => {
  state.io = {};
  return load();
};
$("#run").onchange = () => {
  preserveScroll();
  state.run = $("#run").value;
  state.experience = state.sample = null;
  state.scrolls = {};
  state.io = {};
  state.data = null;
  load();
};
(async () => {
  try {
    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    for (const name of list(data.runs)) {
      const option = el("option", name);
      option.value = name;
      $("#run").append(option);
    }
    const params = new URLSearchParams(location.search);
    state.run = params.get("run") || data.default || data.runs?.[0];
    if (!list(data.runs).includes(state.run)) state.run = data.runs?.[0];
    if (["overview", "samples", "experiences"].includes(params.get("page")))
      state.page = params.get("page");
    if (
      ["task", "rubric", "checks", "critic", "io"].includes(params.get("stage"))
    )
      state.sampleTab = params.get("stage");
    if (["method", "history", "evolution"].includes(params.get("view")))
      state.experienceTab = params.get("view");
    state.sample = params.get("sample");
    state.experience = params.get("experience");
    state.revealList = Boolean(state.sample || state.experience);
    if (state.run) {
      $("#run").value = state.run;
      await load();
    } else render();
  } catch (error) {
    empty($("#app"), "无法读取运行列表", String(error));
  }
})();
