import {
  node,
  add,
  button,
  badge,
  tone,
  short,
  number,
  duration,
  textBox,
  raw,
  fold,
  heading,
  table,
  empty,
} from "./ui.mjs";
import { callRecords } from "./calls.mjs";
import {
  trialLabel,
  initialStatus,
  resultOf,
  sampleOutcome,
  criticCandidates,
  resolveCriticRound,
} from "./metrics.mjs";

export const stages = [
  ["selection", "经验选择"],
  ["draft", "共享初稿"],
  ["rubric", "各路修订标准"],
  ["judge", "初始评判"],
  ["critic", "Critic提经验"],
  ["candidate_rubric", "候选生成标准"],
  ["candidate_judge", "候选评判"],
  ["calls", "原始输入／输出"],
];

export function traceStageGroups(record) {
  return [
    { label: "初始流程", round: null, stages: stages.slice(0, 4) },
    ...(record.critic_rounds || []).map((r) => ({
      label: `第${r.round_number}轮`,
      round: r.round_number,
      stages: stages.slice(4, 7),
    })),
  ];
}

function eLink(id, data, actions) {
  const index = data.experiences.findIndex((e) => e.experience_id === id);
  if (index < 0) {
    const label = node("span", `候选 ${short(id)}`, "mono candidate-id");
    label.title = id;
    return label;
  }
  const b = button(
    `E${String(index + 1).padStart(2, "0")}`,
    () => actions.experience(id),
    "link chip",
  );
  b.title = id;
  return b;
}

export function methodBlock(experience, data, actions) {
  const box = node("div", undefined, "method-block");
  if (
    data.experiences.some((e) => e.experience_id === experience.experience_id)
  )
    add(box, eLink(experience.experience_id, data, actions));
  for (const [field, label] of [
    ["trigger", "适用范围"],
    ["principle", "核心核验方法"],
    ["application_guide", "生成指南"],
  ]) {
    add(box, node("h4", label), textBox(experience.method[field]));
  }
  return box;
}

function verdict(value) {
  return value === true
    ? badge("满足", "good")
    : value === false
      ? badge("不满足", "bad")
      : badge("未判定");
}

function pathSection(path, title, data, actions, judgments = false) {
  const section = node("section", undefined, "path-panel");
  add(
    section,
    heading(
      title,
      null,
      judgments
        ? badge(resultOf(path), tone(resultOf(path)))
        : badge(
            path?.rubric ? "已生成" : path?.error ? "生成失败" : "未生成",
            path?.rubric ? "good" : path?.error ? "bad" : "neutral",
          ),
    ),
  );
  if (!path)
    return add(section, empty("这一路未运行", "未选经验时只运行无经验路径。"));
  if (path.error && (judgments || !path.rubric))
    add(section, textBox(path.error, "error-message"));
  if (path.error && !judgments && path.rubric)
    add(
      section,
      node(
        "p",
        "标准已生成；后续评判发生技术失败，可在评判阶段查看。",
        "muted",
      ),
    );
  if (path.judgment && judgments) {
    const j = path.judgment;
    add(
      section,
      add(
        node("div", undefined, "scoreline"),
        node("span", `chosen ${number(j.chosen_score)}`),
        node("span", `rejected ${number(j.rejected_score)}`),
        node("strong", `分差 ${number(j.margin)}`),
      ),
    );
  }
  if (!path.rubric)
    return add(
      section,
      empty("没有有效Rubric", "查看错误或模型调用，了解执行停在哪一步。"),
    );
  const js = new Map(
    (path.judgment?.criterion_judgments || []).map((j) => [j.criterion_id, j]),
  );
  path.rubric.criteria.forEach((c, i) => {
    const item = node("article", undefined, "criterion");
    add(
      item,
      add(
        node("div", undefined, "criterion-top"),
        node("span", `标准 ${i + 1}`, "eyebrow"),
        node("span", `权重 ${number(c.weight)}`, "weight"),
      ),
      node("p", c.question, "question"),
    );
    if (judgments) {
      const j = js.get(c.criterion_id);
      const pair = node("div", undefined, "judgment-pair");
      for (const side of ["chosen", "rejected"])
        add(
          pair,
          add(
            node("div", undefined, "judgment-side"),
            add(
              node("div", undefined, "row-between"),
              node("strong", side),
              verdict(j?.[side]),
            ),
            textBox(j?.[`${side}_reason`] || "暂无有效判定"),
          ),
        );
      add(item, pair);
    }
    add(section, item);
  });
  if (!judgments) {
    add(
      section,
      fold(
        "设计依据",
        textBox(path.rubric.analysis),
        `analysis-${path.rubric.rubric_id}`,
      ),
    );
  }
  return section;
}

function selection(record, data, actions) {
  const box = node("div");
  const candidates = record.retrieval.candidates || [];
  add(
    box,
    heading(
      "哪些经验进入候选，最终选了哪些",
      `${candidates.length}条候选 · ${Object.keys(record.selection_reasons).length}条选中`,
    ),
  );
  add(
    box,
    table(
      ["经验与适用范围", "候选来源", "历史收益", "是否选中"],
      candidates.map((c) => {
        const e = data.experiences.find(
          (e) => e.experience_id === c.experience_id,
        );
        return [
          add(
            node("div"),
            eLink(c.experience_id, data, actions),
            node("p", e?.method.trigger || "暂无经验文本", "table-excerpt"),
          ),
          c.source === "historical" ? "历史收益" : "全池语义",
          c.historical_score == null ? "未知" : number(c.historical_score),
          record.selection_reasons[c.experience_id]
            ? badge("选中", "good")
            : badge("未选"),
        ];
      }),
    ),
  );
  for (const [id, reason] of Object.entries(record.selection_reasons))
    add(
      box,
      add(
        node("article", undefined, "reason-card"),
        eLink(id, data, actions),
        textBox(reason),
      ),
    );
  if (!candidates.length)
    add(box, empty("没有候选经验", "本题从任务本身生成评分标准。"));
  add(
    box,
    fold("检索依据与相近任务组", raw(record.retrieval), "retrieval-raw"),
  );
  return box;
}

function critic(record, data, actions, roundNumber) {
  const box = node("div");
  add(
    box,
    heading(
      `${roundNumber ? `第${roundNumber}轮：` : ""}Critic为什么提出这些候选经验`,
      "这里展示分析与经验正文；后续Generator标准、Judge判定分别见右侧两个阶段。",
    ),
  );
  if (!record.critic_rounds.length)
    return add(
      box,
      empty(
        "本题没有Critic记录",
        "初始路径判对或出现技术错误时，不一定会提出候选。",
      ),
    );
  for (const round of record.critic_rounds.filter(
    (r) => r.round_number === roundNumber,
  )) {
    add(box, textBox(round.analysis));
    if (round.error) add(box, textBox(round.error, "error-message"));
    if (!round.attempts.length)
      add(box, node("p", "本轮没有形成可验证候选。", "muted"));
    round.attempts.forEach((a, index) => {
      const card = node("article", undefined, "candidate-card");
      const reasons = {
        source_correct: "来源验证判对",
        embedding_failure: "向量生成技术失败",
        incorrect: "仍未判对",
        technical_failure: "验证技术失败",
      };
      add(
        card,
        heading(
          `候选 ${index + 1}`,
          reasons[a.reason] || a.reason,
          badge(a.admitted ? "已准入" : "未准入", a.admitted ? "good" : "warn"),
        ),
        methodBlock(a.experience, data, actions),
      );
      add(
        card,
        add(
          node("div", undefined, "candidate-links"),
          button(
            "查看生成标准 →",
            () =>
              actions.candidate(
                a.experience.experience_id,
                "candidate_rubric",
                round.round_number,
              ),
            "link",
          ),
          button(
            "查看Judge判定 →",
            () =>
              actions.candidate(
                a.experience.experience_id,
                "candidate_judge",
                round.round_number,
              ),
            "link",
          ),
        ),
      );
      add(box, card);
    });
  }
  return box;
}

function candidateValidation(record, state, actions) {
  const box = node("div");
  const candidates = criticCandidates(record).filter(
    (c) => c.round === state.round,
  );
  const judging = state.stage === "candidate_judge";
  add(
    box,
    heading(
      judging
        ? `${state.round ? `第${state.round}轮：` : ""}Judge评判`
        : `${state.round ? `第${state.round}轮：` : ""}Generator应用候选经验`,
      "本轮候选复用共享初稿，单独修订后评判；无经验基线不重跑。",
    ),
  );
  if (!candidates.length)
    return add(
      box,
      empty(
        "没有候选验证轨迹",
        record.critic_rounds.length
          ? "本轮Critic未提出可验证候选，或提案调用失败；详见本轮Critic阶段。"
          : "本题没有运行Critic。",
      ),
    );
  const round = record.critic_rounds.find(
    (r) => r.round_number === state.round,
  );
  const options = candidates.map((c) => ({
    key: c.attempt.experience.experience_id,
    label: `候选${c.number}`,
    path: c.attempt.path,
    attempts: [c.attempt],
  }));
  const selected = options.find((c) => c.key === state.candidate) || options[0];
  state.candidate = selected.key;
  const control = node("div", undefined, "candidate-picker");
  const label = node("label", `第${state.round}轮验证路径`);
  label.htmlFor = "trace-candidate";
  const select = node("select");
  select.id = "trace-candidate";
  select.setAttribute("aria-label", `选择第${state.round}轮候选`);
  for (const option of options) {
    const element = node(
      "option",
      `第${state.round}轮 · ${option.label} · ${resultOf(option.path)}`,
    );
    element.value = option.key;
    add(select, element);
  }
  select.value = selected.key;
  select.onchange = () => actions.candidate(select.value);
  add(control, label, select);
  add(box, control);
  for (const a of selected.attempts)
    add(
      box,
      fold(
        "查看候选经验",
        methodBlock(a.experience, state.data, actions),
        `candidate-method-${a.experience.experience_id}`,
      ),
    );
  add(
    box,
    pathSection(
      selected.path,
      `第${state.round}轮 · ${selected.label}`,
      state.data,
      actions,
      judging,
    ),
  );
  if (judging)
    add(
      box,
      node(
        "p",
        "来源验证只决定准入，不累计信用或建立来源关联。",
        "muted admission-note",
      ),
    );
  return box;
}

function calls(record, state, actions) {
  const box = node("div");
  const value = state.ioCache.get(`${state.run}:${record.sample.sample_id}`);
  add(
    box,
    heading(
      "模型实际收到了什么、返回了什么",
      "包含原始提示、回复、格式重试和请求耗时。",
      button(
        value ? "重新读取" : "读取调用记录",
        () => actions.loadIO(record),
        "secondary",
      ),
    ),
  );
  if (!value)
    return add(
      box,
      empty("按需读取原始调用", "点击上方按钮，展开本题的全部已保存调用。"),
    );
  if (value.loading) return add(box, node("p", "正在读取…", "muted"));
  if (value.error) return add(box, textBox(value.error, "error-message"));
  const requests = callRecords(value.entries);
  add(
    box,
    node(
      "p",
      `共${requests.length}次已保存请求，包含失败及重试。下方的消息和回复原文来自日志；解析结果另列，不替代原文。日志不包含完整HTTP报文。`,
      "muted",
    ),
  );
  if (!requests.length)
    add(
      box,
      empty("没有已保存的模型消息", "尚未返回的请求不一定出现在日志中。"),
    );
  requests.forEach(({ entry, validation, timing, rawAvailable, error }, i) => {
    const body = node("div", undefined, "call-content");
    const side = entry.response_side ? ` · ${entry.response_side}侧` : "";
    const status = error
      ? "失败／触发重试"
      : validation?.validation_status === "valid"
        ? "通过结构校验"
        : entry.output == null
          ? "未取得解析结果"
          : "已返回";
    add(
      body,
      node(
        "p",
        `${entry.model || "模型未记录"}${side} · HTTP ${entry.status ?? "未记录"} · 传输尝试${entry.attempt || 1} · 格式尝试${validation?.contract_attempt ?? "未记录"}`,
        "muted",
      ),
    );
    add(
      body,
      node(
        "p",
        `调用ID：${entry.call_id || "未记录"} · 提示版本：${entry.prompt_version || "未记录"}`,
        "muted call-identity",
      ),
    );
    if (error) add(body, textBox(error, "error-message"));
    const messages = node("div", undefined, "message-list");
    (entry.messages || []).forEach((message, index) => {
      const item = node("section", undefined, "message-block");
      add(
        item,
        node("h4", `${index + 1}. ${message.role}`),
        raw(message.content),
      );
      add(messages, item);
    });
    if (!entry.messages) add(messages, raw(entry.input));
    const input = fold(
      entry.messages
        ? "原始输入 · 按顺序展示完整消息"
        : "已保存输入 · 嵌入请求",
      messages,
      `call-input-${i}`,
    );
    const output = fold(
      "原始输出 · 模型回复原文",
      rawAvailable
        ? raw(entry.raw_output)
        : node(
            "p",
            "日志未保存回复原文；下方解析结果不能还原原文。嵌入调用仅保存返回向量。",
            "muted",
          ),
      `call-output-${i}`,
    );
    input.open = true;
    output.open = true;
    add(
      body,
      input,
      output,
      fold(
        "解析后的JSON结果",
        entry.output != null
          ? raw(entry.output)
          : node("p", "没有解析结果", "muted"),
        `call-parsed-${i}`,
      ),
      fold(
        "完整日志 · 输入JSON、编号映射与校验记录",
        raw({
          request: entry,
          timing: timing ?? null,
          validation: validation ?? null,
        }),
        `call-log-${i}`,
      ),
    );
    add(
      box,
      fold(
        `${String(i + 1).padStart(2, "0")}  ${entry.role}${side} · ${status}${timing ? " · " + duration(timing.request_seconds) : ""}`,
        body,
        `call-${i}`,
      ),
    );
  });
  value.warnings.forEach((w) => add(box, node("p", w, "error-message")));
  return box;
}

export function traceView(record, state, actions) {
  const data = state.data;
  const box = node("section", undefined, "trace-detail");
  const status = record.error ? "技术失败" : initialStatus(record);
  add(
    box,
    heading(
      `样本 ${short(record.sample.sample_id, 12)}`,
      "已提交的完整轨迹",
      badge(status, tone(status)),
    ),
  );
  add(box, textBox(record.sample.task, "task-text"));
  const tools = node("div", undefined, "inline-tools");
  add(
    tools,
    button(
      "翻译任务",
      async () => actions.translate(record.sample.task, tools),
      "link",
    ),
    node("span", sampleOutcome(record, data), "muted"),
  );
  add(
    box,
    tools,
    fold(
      "查看chosen与rejected回答",
      add(
        node("div", undefined, "answer-pair"),
        add(
          node("section"),
          node("h4", "chosen · 数据集偏好回答"),
          textBox(record.sample.chosen),
        ),
        add(
          node("section"),
          node("h4", "rejected · 数据集另一回答"),
          textBox(record.sample.rejected),
        ),
      ),
      "answers",
    ),
  );
  if (record.error) add(box, textBox(record.error, "error-message"));
  if (!stages.some(([key]) => key === state.stage)) state.stage = "selection";
  state.round = resolveCriticRound(record, state.round, state.candidate);
  const tabs = node("nav", undefined, "stage-tabs grouped-stages");
  tabs.setAttribute("aria-label", "实际执行阶段");
  let step = 0;
  for (const group of traceStageGroups(record)) {
    const row = node("div", undefined, "stage-group");
    add(row, node("span", group.label, "stage-group-title"));
    const controls = node("div", undefined, "stage-group-controls");
    for (const [key, name] of group.stages) {
      const active =
        state.stage === key &&
        (group.round === null || state.round === group.round);
      const b = button(
        `${++step} ${name}`,
        () => actions.stage(key, group.round ?? state.round),
        active ? "active" : "",
      );
      b.dataset.stage = key;
      if (group.round !== null) b.dataset.round = group.round;
      b.setAttribute("aria-label", `${group.label} · ${name}`);
      b.setAttribute("aria-current", active ? "step" : "false");
      add(controls, b);
    }
    add(row, controls);
    add(tabs, row);
  }
  add(
    tabs,
    add(
      node("div", undefined, "stage-utilities"),
      button(
        "原始输入／输出",
        () => actions.stage("calls"),
        state.stage === "calls" ? "active" : "",
      ),
    ),
  );
  add(box, tabs);
  if (state.stage === "selection") add(box, selection(record, data, actions));
  else if (state.stage === "critic")
    add(box, critic(record, data, actions, state.round));
  else if (["candidate_rubric", "candidate_judge"].includes(state.stage))
    add(box, candidateValidation(record, state, actions));
  else if (state.stage === "calls") add(box, calls(record, state, actions));
  else {
    const pair = node("div", undefined, "paths-grid");
    if (state.stage === "draft") {
      add(pair, pathSection({rubric: record.draft}, "共享任务初稿R0", data, actions));
    } else {
      for (const path of record.paths || []) {
        add(pair, pathSection(path, trialLabel(path, Object.keys(record.selection_reasons)),
          data, actions, state.stage === "judge"));
      }
    }
    add(box, pair);
  }
  return box;
}
