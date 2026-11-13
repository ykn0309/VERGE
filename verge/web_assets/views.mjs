import {
  node,
  add,
  button,
  badge,
  tone,
  short,
  number,
  signed,
  duration,
  textBox,
  raw,
  fold,
  heading,
  table,
  metric,
  empty,
  groupName,
} from "./ui.mjs";
import {
  learningProtocol,
  summarize,
  resultOf,
  feedbackOutcome,
  sampleOutcome,
  matchesFilter,
  sampleListSummary,
  experienceListSummary,
} from "./metrics.mjs";
import { traceView, methodBlock } from "./trace.mjs";

function expName(id, data) {
  const i = data.experiences.findIndex((e) => e.experience_id === id);
  return i < 0 ? short(id) : `E${String(i + 1).padStart(2, "0")}`;
}

function sourceSample(taskId, data) {
  return data.task_index?.[taskId];
}

export function overview(state, actions) {
  const d = state.data,
    s = summarize(d, state.pairScope),
    o = state.observation || {};
  const root = node("div", undefined, "overview");
  const progress = node("section", undefined, "panel run-progress");
  const target = o.target_tasks;
  add(
    progress,
    heading(
      "运行进度",
      `${state.run} · ${learningProtocol(d) ? "v3.1单经验试用" : "不支持的协议"}`,
      badge(
        statusLabel(d.manifest.status),
        d.manifest.status === "running" ? "live" : "neutral",
      ),
    ),
  );
  add(
    progress,
    add(
      node("div", undefined, "progress-title"),
      add(
        node("div"),
        node("strong", s.completed, "hero-number"),
        node("span", target ? ` / ${target}题已提交` : " 题已提交", "muted"),
      ),
      node(
        "span",
        `运行 ${duration((d.manifest.finished || Date.now() / 1000) - d.manifest.started)}`,
        "mono muted",
      ),
    ),
  );
  if (target) {
    const bar = node("progress");
    bar.max = target;
    bar.value = s.completed;
    bar.setAttribute("aria-label", `已提交${s.completed}题，共${target}题`);
    add(progress, bar);
  }
  add(
    progress,
    node(
      "p",
      `并发上限 ${o.settings?.train_workers ?? "—"} · ${o.settings?.model || "模型信息待载入"} · ${o.settings?.enable_thinking === false ? "思考关闭" : o.settings?.enable_thinking === true ? "思考开启" : "思考设置未记录"}`,
      "muted",
    ),
  );
  add(root, progress);
  add(
    root,
    add(
      node("section", undefined, "metric-grid"),
      metric(
        "无经验基线判对",
        `${s.correct} / ${s.completed}`,
        "各经验路独立试用，不汇总为推理准确率",
        () => actions.samples("all"),
      ),
      metric(
        "正常路径技术失败",
        s.errors,
        `候选验证另有 ${s.candidateErrors} 次`,
        () => actions.samples("error"),
      ),
      metric(
        "经验池",
        d.experiences.length,
        `${s.repaired}道来源题有候选通过`,
        () => actions.page("experiences"),
      ),
      metric(
        "下游提供经验",
        s.selected,
        `选择了经验的任务共 ${s.selected} 道`,
        () => actions.samples("selected"),
      ),
    ),
  );
  const paired = node("section", undefined, "panel paired-panel");
  add(
    paired,
    heading(
      "提供经验后，结果如何变化",
      `${s.paired}条有效配对 · ${s.invalidPairs}条因技术错误无法比较`,
      node("span", `净改善 ${signed(s.gain, 0)}条配对`, "net-gain"),
    ),
  );
  const outcomes = node("div", undefined, "outcome-grid");
  for (const [label, hint] of [
    ["改善", "无经验错 → 有经验对"],
    ["退化", "无经验对 → 有经验错"],
    ["均对", "两路都判对"],
    ["均错", "两路都判错"],
  ]) {
    const card = button("", () => actions.samples(label), "outcome-card");
    add(
      card,
      badge(label, tone(label)),
      node("strong", s.outcomes[label]),
      node("small", hint, "muted"),
    );
    add(outcomes, card);
  }
  add(
    paired,
    outcomes,
    node(
      "p",
      "逐条经验与同题基线比较；同题多条配对不代表多个独立任务。来源验证不计收益。",
      "caption",
    ),
  );
  add(root, paired);
  const lower = node("div", undefined, "overview-columns");
  const stages = node("section", undefined, "panel");
  add(
    stages,
    heading("流程执行", "从选择到候选验证，分开查看结果"),
    table(
      ["环节", "已记录", "查看"],
      [
        [
          "Selector选中经验",
          `${s.selected}题`,
          button("查看样本", () => actions.samples("selected"), "link"),
        ],
        [
          "Critic提出候选",
          `${s.critic}题 · ${s.attempts}条候选 · ${s.validations}路验证`,
          button("查看验证", () => actions.samples("critic"), "link"),
        ],
        [
          "候选准入",
          `${s.admitted}条`,
          button("查看经验", () => actions.page("experiences"), "link"),
        ],
        [
          "分组情况",
          `${new Set(Object.values(d.task_groups)).size}个任务组 · ${new Set(Object.values(d.experience_groups)).size}个经验组`,
          button("查看分组", () => actions.page("clusters"), "link"),
        ],
      ],
    ),
  );
  const requests = node("section", undefined, "panel");
  add(
    requests,
    heading("请求与资源", "请求出错可能已经重试恢复，不等于样本失败"),
    table(
      ["项目", "数量"],
      [
        ["模型与嵌入请求", o.requests?.total ?? "待载入"],
        ["传输层重试请求", o.requests?.transport_retries ?? "待载入"],
        ["输出校验未通过", o.requests?.invalid_contracts ?? "待载入"],
        [
          "输入 / 输出 tokens",
          o.requests
            ? `${o.requests.input_tokens?.toLocaleString() ?? "未报告"} / ${o.requests.output_tokens?.toLocaleString() ?? "未报告"}`
            : "待载入",
        ],
      ],
    ),
  );
  add(lower, stages, requests);
  add(root, lower);
  const pending = o.uncommitted || [];
  if (pending.length && d.manifest.status === "running")
    add(
      root,
      add(
        node("section", undefined, "panel"),
        heading(
          "日志中尚未提交的样本",
          "最近返回阶段供定位使用，不表示精确的在途请求状态",
        ),
        table(
          ["样本", "任务", "最近返回"],
          pending.map((p) => [
            short(p.sample_id),
            p.task,
            p.last_role || "请求中",
          ]),
        ),
      ),
    );
  if (o.recent_calls?.length)
    add(
      root,
      add(
        node("section", undefined, "panel"),
        heading(
          "最近返回的调用",
          "包含已恢复的重试；进入单题可查看原始输入与回复",
        ),
        table(
          ["样本", "角色", "耗时", "返回情况", "原始记录"],
          o.recent_calls.map((c) => [
            short(c.sample_id, 8),
            c.role,
            duration(c.seconds),
            c.error ? badge("发生错误", "warn") : badge("已返回", "good"),
            d.samples.some((s) => s.sample.sample_id === c.sample_id)
              ? button(
                  "查看调用",
                  () => actions.sample(c.sample_id, "calls"),
                  "link",
                )
              : "尚未提交",
          ]),
        ),
      ),
    );
  if (d.manifest.test)
    add(
      root,
      add(
        node("section", undefined, "panel"),
        heading("冻结评价结果"),
        table(
          ["任务数", "判对", "技术失败"],
          [
            [
              d.manifest.test.tasks,
              d.manifest.test.correct,
              d.manifest.test.errors,
            ],
          ],
        ),
      ),
    );
  add(
    root,
    fold(
      "运行设置与原始manifest",
      raw({ settings: o.settings, manifest: d.manifest }),
      "manifest",
    ),
  );
  return root;
}

export function statusLabel(status) {
  return (
    {
      running: "运行中",
      training_completed: "训练已完成",
      completed: "已完成",
      paused: "已暂停",
      failed: "执行失败",
      stopped_on_problem: "已停止",
    }[status] ||
    status ||
    "未知"
  );
}

export function samples(state, actions) {
  const d = state.data;
  const layout = node("div", undefined, "workbench samples-workbench");
  const list = node("aside", undefined, "panel browser-list");
  add(list, heading("样本", `${d.samples.length}条已提交轨迹`));
  const search = node("input");
  search.type = "search";
  search.placeholder = "搜索任务或编号";
  search.value = state.search;
  search.setAttribute("aria-label", "搜索样本");
  const filter = node("select");
  filter.setAttribute("aria-label", "筛选样本");
  for (const [value, label] of [
    ["all", "全部样本"],
    ["wrong", "初始判错"],
    ["error", "技术失败"],
    ["selected", "使用了经验"],
    ["critic", "触发Critic"],
    ["改善", "下游改善"],
    ["退化", "下游退化"],
    ["均对", "两路均对"],
    ["均错", "两路均错"],
  ]) {
    const option = node("option", label);
    option.value = value;
    add(filter, option);
  }
  filter.value = state.filter;
  add(list, search, filter);
  const rows = node("div", undefined, "list-rows");
  rows.id = "sample-list";
  const draw = () => {
    rows.replaceChildren();
    const q = state.search.toLowerCase();
    const visible = d.samples.filter(
      (r) =>
        matchesFilter(r, state.filter, d, state.pairScope) &&
        (r.sample.task + r.sample.sample_id).toLowerCase().includes(q),
    );
    list.querySelector(".section-heading p").textContent =
      `显示 ${visible.length} / ${d.samples.length} 条轨迹`;
    visible.forEach((r) => {
      const b = button(
        "",
        () => actions.sample(r.sample.sample_id),
        `sample-row ${state.sample === r.sample.sample_id ? "active" : ""}`,
      );
      b.dataset.sample = r.sample.sample_id;
      const info = sampleListSummary(r, d);
      const label = info.initial;
      b.title = r.sample.sample_id;
      add(
        b,
        add(
          node("div", undefined, "row-between"),
          node(
            "small",
            `${String(d.samples.indexOf(r) + 1).padStart(2, "0")} · ${short(r.sample.sample_id, 8)}`,
            "mono",
          ),
          badge(
            ["判对", "判错"].includes(label) ? `初始${label}` : label,
            tone(label),
          ),
        ),
      );
      const statusRow = (label, status) =>
        add(
          node("div", undefined, "sample-status-row"),
          node("span", label, "status-label"),
          badge(status, tone(status)),
        );
      const paths = node("div", undefined, "sample-status-paths");
      for (const item of info.initialTrials) add(paths, statusRow(item.label, item.result));
      add(b, paths);
      const use = node("div", undefined, "sample-use");
      add(
        use,
        node(
          "span",
          `提供${info.selected ?? "—"}条`,
          "muted",
        ),
      );
      if (info.comparison)
        add(use, badge(info.comparison, tone(info.comparison)));
      add(
        use,
        node(
          "span",
          info.rounds
            ? `Critic ${info.rounds}轮 · 准入${info.admitted}/${info.candidates}`
            : "Critic未运行",
          "muted",
        ),
      );
      add(b, use);
      if (info.rounds) {
        const criticBox = node("div", undefined, "sample-critic-status");
        for (const candidate of info.candidateResults) {
          const item = node("div", undefined, "sample-candidate-result");
          add(
            item,
            node(
              "span",
              `${candidate.round}轮·候选${candidate.candidate}`,
              "status-label",
            ),
            badge(candidate.result, tone(candidate.result)),
          );
          const notes = [];
          notes.push(
            candidate.admitted
              ? "准入"
              : "未准入",
          );
          add(
            item,
            node(
              "span",
              notes.join(" · "),
              candidate.admitted
                ? "candidate-note admission"
                : "candidate-note muted",
            ),
          );
          add(criticBox, item);
        }
        for (const failure of info.roundErrors) {
          const line = statusRow(`${failure.round}轮提案`, "技术失败");
          line.title = failure.error;
          add(criticBox, line);
        }
        if (!info.candidates && !info.roundErrors.length)
          add(
            criticBox,
            node("div", "未提出候选，未进行验证", "muted candidate-note"),
          );
        add(b, criticBox);
      }
      add(rows, b);
    });
    if (!visible.length)
      add(rows, empty("没有匹配样本", "换一个关键词或筛选条件。"));
  };
  search.oninput = () => {
    state.search = search.value;
    draw();
  };
  filter.onchange = () => {
    state.filter = filter.value;
    actions.samples(state.filter);
  };
  const matching = d.samples.filter(
    (r) =>
      matchesFilter(r, state.filter, d, state.pairScope) &&
      (r.sample.task + r.sample.sample_id)
        .toLowerCase()
        .includes(state.search.toLowerCase()),
  );
  const selected =
    matching.find((r) => r.sample.sample_id === state.sample) || matching[0];
  if (selected) state.sample = selected.sample.sample_id;
  draw();
  add(list, rows);
  add(layout, list);
  if (selected) {
    add(layout, traceView(selected, state, actions));
  } else
    add(
      layout,
      add(
        node("section", undefined, "panel"),
        empty("还没有提交样本", "自动刷新会在轨迹提交后显示。"),
      ),
    );
  return layout;
}

export function experiences(state, actions) {
  const d = state.data,
    layout = node("div", undefined, "workbench experiences-workbench");
  const list = node("aside", undefined, "panel browser-list");
  add(list, heading("经验池", `${d.experiences.length}条独立经验`));
  const search = node("input");
  search.placeholder = "搜索适用范围或方法";
  search.type = "search";
  search.value = state.expSearch;
  search.setAttribute("aria-label", "搜索经验");
  add(list, search);
  const rows = node("div", undefined, "list-rows");
  rows.id = "experience-list";
  function draw() {
    rows.replaceChildren();
    d.experiences
      .filter((e) =>
        JSON.stringify(e.method)
          .toLowerCase()
          .includes(state.expSearch.toLowerCase()),
      )
      .forEach((e) => {
        const b = button(
          "",
          () => actions.experience(e.experience_id),
          `sample-row ${e.experience_id === state.experience ? "active" : ""}`,
        );
        const status = experienceListSummary(e, d);
        const reward = status.meanReward;
        const stateText =
          reward === null
            ? "待下游验证"
            : reward > 0
              ? "净收益为正"
              : reward < 0
                ? "净收益为负"
                : "收益持平";
        add(
          b,
          add(
            node("div", undefined, "row-between"),
            node("small", expName(e.experience_id, d), "mono"),
            badge(
              stateText,
              reward === null
                ? "neutral"
                : reward > 0
                  ? "good"
                  : reward < 0
                    ? "bad"
                    : "neutral",
            ),
          ),
          node("p", e.method.trigger, "sample-title"),
        );
        add(
          b,
          add(
            node("div", undefined, "experience-status-row"),
            node("span", `来源验证：${status.sourceResult}`),
          ),
          add(
            node("div", undefined, "experience-status-row"),
            node(
              "span",
              `${"下游提供"} ${status.providedTasks}题`,
              "muted",
            ),
            node("span", `有效 ${status.validTasks}题`, "muted"),
          ),
          add(
            node("div", undefined, "experience-outcomes"),
            node("span", `改善 ${status.improved}`),
            node("span", `退化 ${status.regressed}`),
            node("span", `均对 ${status.bothCorrect}`),
            node("span", `均错 ${status.bothWrong}`),
          ),
          add(
            node("div", undefined, "experience-status-row"),
            node(
              "span",
              `平均${"信用"} ${signed(status.meanReward)}`,
              "muted",
            ),
            status.technicalFailures
              ? node("span", `技术失败 ${status.technicalFailures}`, "muted")
              : null,
          ),
        );
        add(rows, b);
      });
  }
  search.oninput = () => {
    state.expSearch = search.value;
    draw();
  };
  const e =
    d.experiences.find((e) => e.experience_id === state.experience) ||
    d.experiences[0];
  if (e) state.experience = e.experience_id;
  draw();
  add(list, rows);
  add(layout, list);
  if (!e)
    return add(
      layout,
      add(
        node("section", undefined, "panel"),
        empty("暂无准入经验", "Critic候选通过即时验证后会显示在这里。"),
      ),
    );
  state.experience = e.experience_id;
  const detail = node("section", undefined, "panel");
  const feedback = d.feedback.filter((f) =>
    f.experience_id === e.experience_id,
  );
  const valid = feedback.filter((f) => !f.error && f.phase === "downstream");
  const gain = valid.reduce(
    (sum, f) =>
      sum +
      (feedbackOutcome(f) === "改善"
        ? 1
        : feedbackOutcome(f) === "退化"
          ? -1
          : 0),
    0,
  );
  add(
    detail,
    heading(`${expName(e.experience_id, d)} · 经验详情`, e.experience_id),
    methodBlock(e, d, actions),
  );
  add(
    detail,
    add(
      node("div", undefined, "mini-metrics"),
      metric("有效下游任务", valid.length, "不包含来源即时修复"),
      metric(
        "平均信用",
        signed(experienceListSummary(e, d).meanReward),
        "单经验路减无经验路；不含来源验证",
      ),
    ),
  );
  const source = sourceSample(e.source_task_id, d);
  add(
    detail,
    heading(
      "从哪里产生",
      null,
      source
        ? button(
            "查看来源验证 →",
            () => actions.sample(source, "critic"),
            "link",
          )
        : null,
    ),
  );
  add(
    detail,
    node(
      "p",
      `所属经验组：${groupName(d.experience_groups[e.experience_id])}`,
      "muted",
    ),
  );
  if (e.parent_ids.length)
    add(
      detail,
      node(
        "p",
        `参考旧经验：${e.parent_ids.map((id) => expName(id, d)).join("、")}`,
        "muted",
      ),
    );
  add(
    detail,
    heading(
      "后续在哪些任务中被提供",
    ),
    table(
      ["任务", "配对结果", "信用", "跳转"],
      feedback.map((f) => {
        const id = sourceSample(f.task_id, d),
          sample = d.samples.find((s) => s.sample.sample_id === id);
        return [
          sample?.sample.task.slice(0, 140) || short(f.task_id),
          badge(feedbackOutcome(f), tone(feedbackOutcome(f))),
          signed(f.credits?.[e.experience_id]),
          id ? button("查看", () => actions.sample(id, "judge"), "link") : "—",
        ];
      }),
    ),
  );
  add(
    detail,
    heading("分任务组的历史收益", "无记录的任务组—经验配对保持未知"),
    table(
      [
        "任务组",
        "有效任务",
        "平均信用",
      ],
      d.group_statistics
        .filter((g) => g.experience_id === e.experience_id)
        .map((g) => [groupName(g.task_group), g.tasks, signed(g.mean_reward)]),
    ),
  );
  return add(layout, detail);
}

export function clusters(state, actions) {
  const d = state.data,
    kind = state.clusterKind,
    assignments = kind === "task" ? d.task_groups : d.experience_groups;
  const groups = new Map();
  Object.entries(assignments).forEach(([id, group]) => {
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(id);
  });
  const root = node("div");
  add(
    root,
    node(
      "p",
      "文本与对侧类别分布共同影响分组；成员单一归属，查询可参考多个任务组。",
      "page-intro",
    ),
  );
  const switcher = node("div", undefined, "segmented");
  for (const [key, label] of [
    ["task", "任务分组"],
    ["experience", "经验分组"],
  ])
    add(
      switcher,
      button(
        label,
        () => actions.cluster(key, ""),
        kind === key ? "active" : "",
      ),
    );
  add(
    root,
    switcher,
    add(
      node("div", undefined, "metric-grid compact"),
      metric(
        "任务组",
        new Set(Object.values(d.task_groups)).size,
        `${Object.keys(d.task_groups).length}个已分组任务`,
      ),
      metric(
        "经验组",
        new Set(Object.values(d.experience_groups)).size,
        `${Object.keys(d.experience_groups).length}条经验`,
      ),
      metric(
        "当前类型的单成员组",
        [...groups.values()].filter((m) => m.length === 1).length,
        "单成员不等于分组错误",
      ),
      metric(
        "提供关系",
        Object.values(d.references || {}).reduce((n, es) => n + es.length, 0),
        "仅后续独立任务关联；来源验证不计，按任务—经验去重",
      ),
    ),
  );
  add(
    root,
    fold(
      "当前聚类参数",
      raw(state.observation?.settings?.clustering || {}),
      "cluster-settings",
    ),
  );
  const layout = node("div", undefined, "workbench cluster-workbench");
  const list = node("aside", undefined, "panel browser-list");
  add(list, heading(kind === "task" ? "任务组" : "经验组", `${groups.size}组`));
  const rows = node("div", undefined, "list-rows");
  rows.id = "cluster-list";
  const id = groups.has(state.group)
    ? state.group
    : [...groups].sort((a, b) => b[1].length - a[1].length)[0]?.[0];
  if (id) state.group = id;
  for (const [id, members] of [...groups].sort(
    (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]),
  )) {
    const b = button(
      "",
      () => actions.cluster(kind, id),
      `sample-row ${state.group === id ? "active" : ""}`,
    );
    add(
      b,
      node("strong", groupName(id)),
      node(
        "small",
        `${members.length}${kind === "task" ? "个任务" : "条经验"}`,
        "muted",
      ),
    );
    add(rows, b);
  }
  add(list, rows);
  add(layout, list);

  if (!id)
    return add(
      root,
      add(
        layout,
        add(
          node("section", undefined, "panel"),
          empty(
            "暂未形成分组",
            "轨迹提交后，已加入状态的任务与经验会出现在这里。",
          ),
        ),
      ),
    );
  state.group = id;
  const detail = node("section", undefined, "panel");
  add(
    detail,
    heading(
      groupName(id),
      kind === "task"
        ? "查看实际成员，以及它们提供了哪些经验"
        : "经验组仅作分析，不参与收益排序，也不自动合并经验正文",
    ),
  );
  add(detail, node("p", `原始编号：${id}`, "muted"));
  const members = groups.get(id);
  if (kind === "task") {
    add(
      detail,
      table(
        ["任务成员", "提供经验", "查看轨迹"],
        members.map((tid) => {
          const sid = sourceSample(tid, d),
            rec = d.samples.find((s) => s.sample.sample_id === sid);
          return [
            rec?.sample.task.slice(0, 180) || short(tid),
            (d.references?.[tid] || []).map((e) => expName(e, d)).join("、") ||
              "无提供",
            sid ? button("查看", () => actions.sample(sid), "link") : "—",
          ];
        }),
      ),
    );
    const linked = new Map();
    members.forEach((tid) =>
      (d.references?.[tid] || []).forEach((eid) => {
        const group = d.experience_groups[eid];
        if (!group) return;
        if (!linked.has(group)) linked.set(group, new Set());
        linked.get(group).add(tid);
      }),
    );
    add(
      detail,
      heading("关联的经验组", "提供任务数按任务去重，不表示收益"),
      table(
        ["经验组", "提供任务数", "查看成员"],
        [...linked].map(([group, ts]) => [
          groupName(group),
          ts.size,
          button("查看", () => actions.cluster("experience", group), "link"),
        ]),
      ),
    );
    add(
      detail,
      heading(
        "本任务组内各条经验的收益",
        "排序实际使用单条经验收益，经验组不参与评分",
      ),
      table(
        ["经验", "有效任务", "平均参与收益"],
        d.group_statistics
          .filter((s) => s.task_group === id)
          .map((s) => [
            button(
              expName(s.experience_id, d),
              () => actions.experience(s.experience_id),
              "link",
            ),
            s.tasks,
            signed(s.mean_reward),
          ]),
      ),
    );
  } else {
    add(
      detail,
      table(
        ["经验成员", "适用范围", "详情"],
        members.map((eid) => {
          const e = d.experiences.find((e) => e.experience_id === eid);
          return [
            expName(eid, d),
            e?.method.trigger || "—",
            button("查看", () => actions.experience(eid), "link"),
          ];
        }),
      ),
    );
    const linked = new Map();
    Object.entries(d.references || {}).forEach(([tid, es]) => {
      if (!es.some((eid) => members.includes(eid))) return;
      const group = d.task_groups[tid];
      if (!group) return;
      if (!linked.has(group)) linked.set(group, new Set());
      linked.get(group).add(tid);
    });
    add(
      detail,
      heading("被哪些任务组提供", "共同提供可能表示相似或互补，不证明内容等价"),
      table(
        ["任务组", "提供任务数", "查看成员"],
        [...linked].map(([group, ts]) => [
          groupName(group),
          ts.size,
          button("查看", () => actions.cluster("task", group), "link"),
        ]),
      ),
    );
  }
  add(root, add(layout, detail));
  return root;
}
