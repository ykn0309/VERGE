export function node(tag, text, cls = "") {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
}

export function add(parent, ...children) {
  children.filter(Boolean).forEach((c) => parent.appendChild(c));
  return parent;
}

export function button(text, action, cls = "") {
  const e = node("button", text, cls);
  e.type = "button";
  e.onclick = action;
  return e;
}

export function badge(text, tone = "neutral") {
  const b = node("span", undefined, `badge ${tone}`);
  add(b, node("i", undefined, "status-dot"), node("span", text));
  return b;
}

export function tone(text) {
  if (["判对", "改善", "已准入", "已完成", "均对"].includes(text))
    return "good";
  if (["技术失败", "退化", "失败", "未准入"].includes(text)) return "bad";
  if (["判错", "均错", "人工暂停"].includes(text)) return "warn";
  return "neutral";
}

export function short(id, size = 10) {
  return String(id || "—")
    .split(":")
    .at(-1)
    .slice(0, size);
}

export function number(value, digits = 3) {
  return Number.isFinite(value) ? String(Number(value.toFixed(digits))) : "—";
}

export function signed(value, digits = 2) {
  return Number.isFinite(value)
    ? (value > 0 ? "+" : "") + number(value, digits)
    : "未知";
}

export function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${number(seconds, 1)}秒`;
  const minutes = Math.floor(seconds / 60);
  return minutes < 60
    ? `${minutes}分${Math.floor(seconds % 60)}秒`
    : `${Math.floor(minutes / 60)}小时${minutes % 60}分`;
}

export function textBox(text, cls = "") {
  return node("div", text || "暂无记录", `readable ${cls}`);
}

export function raw(value) {
  return node(
    "pre",
    typeof value === "string" ? value : JSON.stringify(value, null, 2),
    "raw",
  );
}

export function fold(title, body, key = "") {
  const d = node("details", undefined, "fold");
  if (key) d.dataset.key = key;
  return add(d, node("summary", title), body);
}

export function heading(title, subtitle, extra) {
  return add(
    node("div", undefined, "section-heading"),
    add(
      node("div"),
      node("h2", title),
      subtitle ? node("p", subtitle, "muted") : null,
    ),
    extra,
  );
}

export function table(headers, rows, cls = "") {
  const t = node("table", undefined, cls);
  add(
    t,
    add(node("thead"), add(node("tr"), ...headers.map((h) => node("th", h)))),
  );
  const body = node("tbody");
  rows.forEach((row) =>
    add(
      body,
      add(
        node("tr"),
        ...row.map((v) =>
          add(node("td"), v instanceof Node ? v : node("span", v ?? "—")),
        ),
      ),
    ),
  );
  if (!rows.length)
    add(
      body,
      add(
        node("tr"),
        Object.assign(node("td", "暂无记录", "empty-cell"), {
          colSpan: headers.length,
        }),
      ),
    );
  return add(node("div", undefined, "table-scroll"), add(t, body));
}

export function metric(label, value, hint, action) {
  const box = node(action ? "button" : "div", undefined, "metric");
  if (action) {
    box.type = "button";
    box.onclick = action;
  }
  box.dataset.metric = label;
  return add(
    box,
    node("span", label, "metric-label"),
    node("strong", value, "metric-value"),
    node("small", hint, "muted"),
  );
}

export function empty(title, hint) {
  return add(
    node("div", undefined, "empty"),
    node("h3", title),
    node("p", hint, "muted"),
  );
}

export function groupName(id) {
  if (!id) return "未分组";
  const match = /^(task|experience)_(\d+)$/.exec(id);
  return match
    ? `${match[1] === "task" ? "任务组 T" : "经验组 G"}${Number(match[2])}`
    : id;
}
