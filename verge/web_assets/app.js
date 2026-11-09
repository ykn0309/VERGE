import { node, add, button, textBox } from "./ui.mjs";
import {
  overview,
  samples,
  experiences,
  clusters,
  statusLabel,
} from "./views.mjs";

const query = new URLSearchParams(location.search);
const state = {
  run: query.get("run") || "",
  page: query.get("page") || "overview",
  sample: query.get("sample") || "",
  experience: query.get("experience") || "",
  stage: query.get("stage") || "selection",
  candidate: query.get("candidate") || "",
  round: Number(query.get("round")) || null,
  filter: query.get("filter") || "all",
  clusterKind: query.get("kind") || "task",
  group: query.get("group") || "",
  pairScope: query.get("scope") || "provided",
  search: "",
  expSearch: "",
  data: null,
  observation: null,
  ioCache: new Map(),
  auto: true,
  loading: false,
  generation: 0,
};
const app = document.getElementById("app");
const runSelect = document.getElementById("run");
const views = { overview, samples, experiences, clusters };
const names = {
  overview: "运行总览",
  samples: "单题追踪",
  experiences: "经验池",
  clusters: "任务与经验分组",
};

async function api(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "读取失败");
  return result;
}

function save(push = false) {
  const p = new URLSearchParams({ run: state.run, page: state.page });
  if (state.pairScope === "cited") p.set("scope", "cited");
  if (state.page === "samples") {
    if (state.sample) p.set("sample", state.sample);
    p.set("stage", state.stage);
    if (state.candidate) p.set("candidate", state.candidate);
    if (state.round) p.set("round", state.round);
    if (state.filter !== "all") p.set("filter", state.filter);
  }
  if (state.page === "experiences" && state.experience)
    p.set("experience", state.experience);
  if (state.page === "clusters") {
    p.set("kind", state.clusterKind);
    if (state.group) p.set("group", state.group);
  }
  const url = "?" + p;
  if (push && location.search !== url) history.pushState(null, "", url);
  else history.replaceState(null, "", url);
}

const actions = {
  save,
  pairScope(scope) {
    state.pairScope = scope;
    render(true);
    save();
  },
  page(page) {
    state.page = page;
    render();
    save(true);
    window.scrollTo(0, 0);
  },
  samples(filter) {
    state.page = "samples";
    state.filter = filter;
    state.sample = "";
    render();
    save(true);
    window.scrollTo(0, 0);
  },
  sample(id, stage = "selection") {
    state.page = "samples";
    state.sample = id;
    state.stage = stage;
    state.candidate = "";
    state.round = null;
    render(true);
    save(true);
  },
  stage(stage, round = state.round) {
    if (round !== state.round) state.candidate = "";
    state.round = round;
    state.stage = stage;
    render(true);
    save(true);
  },
  candidate(id, stage = state.stage, round = state.round) {
    state.round = round;
    state.candidate = id;
    state.stage = stage;
    render(true);
    save(true);
  },
  experience(id) {
    state.page = "experiences";
    state.experience = id;
    render(true);
    save(true);
    window.scrollTo(0, 0);
  },
  cluster(kind, id) {
    state.page = "clusters";
    state.clusterKind = kind;
    state.group = id;
    render();
    save(true);
  },
  async translate(text, target) {
    const b = button("翻译中…", () => {}, "link");
    b.disabled = true;
    add(target, b);
    try {
      const value = await api("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      add(target, textBox(value.translation, "translation"));
    } catch (error) {
      add(target, textBox(error.message, "error-message"));
    } finally {
      b.remove();
    }
  },
  async loadIO(record) {
    const run = state.run,
      key = `${run}:${record.sample.sample_id}`;
    if (state.ioCache.get(key)?.loading) return;
    state.ioCache.set(key, { loading: true });
    render(true);
    try {
      const result = await api(
        `/api/role-io?run=${encodeURIComponent(run)}&sample_id=${encodeURIComponent(record.sample.sample_id)}`,
      );
      state.ioCache.set(key, result);
    } catch (error) {
      state.ioCache.set(key, { error: error.message });
    }
    if (run === state.run) render(true);
  },
};

function render(preserve = false) {
  if (!state.data) return;
  const open = preserve
    ? [...app.querySelectorAll("details[open][data-key]")].map(
        (d) => d.dataset.key,
      )
    : [];
  const scrolls = preserve
    ? Object.fromEntries(
        [...app.querySelectorAll(".list-rows")].map((e) => [e.id, e.scrollTop]),
      )
    : {};
  const y = window.scrollY;
  document.getElementById("page-title").textContent =
    names[state.page] || names.overview;
  document.getElementById("run-status").textContent = statusLabel(
    state.data.manifest.status,
  );
  document
    .getElementById("run-status")
    .classList.toggle("is-live", state.data.manifest.status === "running");
  document.querySelectorAll("[data-page]").forEach((b) => {
    const active = b.dataset.page === state.page;
    b.classList.toggle("active", active);
    b.setAttribute("aria-current", active ? "page" : "false");
  });
  app.replaceChildren((views[state.page] || views.overview)(state, actions));
  if (state.page === "samples" && state.stage === "calls") {
    const record = state.data.samples.find(
      (r) => r.sample.sample_id === state.sample,
    );
    if (record && !state.ioCache.has(`${state.run}:${state.sample}`))
      queueMicrotask(() => actions.loadIO(record));
  }
  if (preserve) {
    app.querySelectorAll("details[data-key]").forEach((d) => {
      if (open.includes(d.dataset.key)) d.open = true;
    });
    app.querySelectorAll(".list-rows").forEach((e) => {
      e.scrollTop = scrolls[e.id] || 0;
    });
    window.scrollTo(0, y);
  }
}

async function loadRun(quiet = false) {
  if (!state.run || state.loading) return;
  const generation = ++state.generation;
  const run = state.run;
  state.loading = true;
  const status = document.getElementById("freshness");
  if (!quiet) status.textContent = "正在读取…";
  try {
    const observation = await api(
      `/api/observation?run=${encodeURIComponent(run)}`,
    );
    if (generation !== state.generation) return;
    if (
      !state.data ||
      state.data.name !== state.run ||
      state.data.manifest.checkpoint_sha256 !==
        observation.manifest.checkpoint_sha256 ||
      state.data.manifest.status !== observation.manifest.status
    ) {
      const data = await api(`/api/run?run=${encodeURIComponent(run)}`);
      if (generation !== state.generation) return;
      state.data = data;
    }
    state.observation = observation;
    const reading =
      quiet &&
      ((app.contains(document.activeElement) &&
        ["INPUT", "SELECT"].includes(document.activeElement.tagName)) ||
        String(window.getSelection()).length > 0);
    if (!reading) render(quiet);
    save();
    status.textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
  } catch (error) {
    if (generation !== state.generation) return;
    status.textContent = `读取失败：${error.message}`;
    if (!state.data)
      app.replaceChildren(textBox(error.message, "error-message"));
  } finally {
    if (generation === state.generation) state.loading = false;
  }
}

async function initialize() {
  try {
    const value = await api("/api/runs");
    runSelect.replaceChildren();
    value.runs.forEach((run) => {
      const option = node("option", `${run.name} · ${statusLabel(run.status)}`);
      option.value = run.name;
      add(runSelect, option);
    });
    if (!value.runs.some((r) => r.name === state.run))
      state.run = value.runs[0]?.name || "";
    runSelect.value = state.run;
    if (!state.run) {
      app.replaceChildren(textBox("暂无可读取的新格式实验。"));
      return;
    }
    await loadRun();
  } catch (error) {
    app.replaceChildren(textBox(error.message, "error-message"));
  }
}

document.querySelectorAll("[data-page]").forEach((b) => {
  b.onclick = () => actions.page(b.dataset.page);
});
runSelect.onchange = () => {
  state.run = runSelect.value;
  state.sample = "";
  state.experience = "";
  state.candidate = "";
  state.round = null;
  state.group = "";
  state.data = null;
  state.loading = false;
  ++state.generation;
  loadRun();
};
document.getElementById("refresh").onclick = () => loadRun(true);
document.getElementById("auto-refresh").onclick = (event) => {
  state.auto = !state.auto;
  event.currentTarget.textContent = state.auto
    ? "自动刷新 · 5秒"
    : "自动刷新已暂停";
  event.currentTarget.setAttribute("aria-pressed", String(state.auto));
};
window.addEventListener("popstate", () => {
  const p = new URLSearchParams(location.search);
  state.page = p.get("page") || "overview";
  state.sample = p.get("sample") || "";
  state.experience = p.get("experience") || "";
  state.stage = p.get("stage") || "selection";
  state.candidate = p.get("candidate") || "";
  state.round = Number(p.get("round")) || null;
  state.filter = p.get("filter") || "all";
  state.pairScope = p.get("scope") || "provided";
  state.clusterKind = p.get("kind") || "task";
  state.group = p.get("group") || "";
  if (state.run !== p.get("run")) {
    state.run = p.get("run") || "";
    state.data = null;
    state.loading = false;
    ++state.generation;
    loadRun();
  } else render();
});
setInterval(() => {
  if (
    state.auto &&
    !document.hidden &&
    state.data?.manifest.status === "running"
  )
    loadRun(true);
}, 5000);
initialize();
