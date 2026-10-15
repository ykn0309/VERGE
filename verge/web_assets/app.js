/* VERGE run dashboard — no third-party runtime; every artifact string is escaped. */
const app = document.getElementById('app');
let data = null, page = 'overview', sampleIndex = 0, expIndex = 0, query = '', splitFilter = 'train', resultFilter = 'all', expQuery = '';
let currentRun = null, requestSequence = 0, memoryTab = 'pool', candidateIndex = 0, candidateQuery = '', expSort = 'original', categoryFilter = 'all';
const experienceSortOptions = [
  ['original', '默认顺序'], ['confidence_desc', '置信度：高 → 低'],
  ['confidence_asc', '置信度：低 → 高'], ['usage_desc', '引用次数：多 → 少'],
  ['usage_asc', '引用次数：少 → 多'],
];
const translations = new Map(), translatedFields = new Set();
const roleIoCache = new Map(), roleIoPending = new Map();
let sampleTab = 'pipeline';
const roleTabs = [
  ['pipeline','流水线'],['selector','经验筛选'],['initial_generator','Generator'],
  ['judge','Judge'],['critic','Critic · 分析与候选'],
  ['experience_updater','经验重新归纳'], ['category_organizer','类别整理'],
];
const translationKey = (run, sampleId, field) => JSON.stringify([run, sampleId, field]);
const h = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n = (value, digits = 3) => typeof value === 'number' ? value.toFixed(digits) : '—';
const pct = value => typeof value === 'number' ? (value * 100).toFixed(1) + '%' : '—';
const short = (s, max = 48) => {
  const text = String(s || '—').replace(/\s+/g, ' ').trim();
  return text.length > max ? text.slice(0, max) + '…' : text;
};
const sampleLabel = s => {
  const parts = String(s.sample_id || '').split(':');
  const subset = parts.length > 2 ? parts.slice(1,-1).join(' · ').replace('llmbar-adver-', '').replace('llmbar-', '') : parts[0];
  return {subset: subset || '样本', number: parts.at(-1) || '—'};
};
const sampleTitle = s => { const label=sampleLabel(s); return `${label.subset} · ${label.number}`; };
const stateName = {sample_error:'样本处理错误',judge_error_initial:'初始判断缺失',judge_error_application:'应用 Judge 错误',initial_success:'初始成功',application_success:'采用候选后成功',success_without_candidate:'成功但未采用候选',validation_exhausted:'应用预算耗尽',invalid_application_exhausted:'技术无效轮耗尽',no_candidate:'未提出候选',initial_failure:'测试未通过',interrupted:'未完成轨迹'};
const successful = state => ['initial_success','application_success','success_without_candidate'].includes(state);
const statusClass = state => successful(state) ? 'ok' : ['sample_error','interrupted'].includes(state) ? 'bad' : 'warn';
const sampleCandidates = sample => (sample.proposal_attempts || []).filter(p=>p.candidate).map(p=>p.candidate);
const badge = (label, type = 'neutral') => `<span class="badge ${type}"><i class="dot"></i>${h(label)}</span>`;
const warnBlock = () => data.warnings.length ? `<div class="warning">⚠ ${data.warnings.map(h).join('<br>')}</div>` : '';
const lineage = () => data.resumed_from ? `<div class="callout">此运行从 ${h(data.resumed_from)} 的已验证轨迹前缀恢复；当前统计已包含复制的样本，不应与父运行相加。</div>` : '';
const heading = (overline, title, description) => `<div class="page-heading"><div><span class="eyebrow">${h(overline)}</span><h1>${h(title)}</h1><p class="subheading">${h(description)}</p></div><span class="run-chip" title="${h(data.run_dir)}">${h(data.run_name)}</span></div>`;
const sampleButton = (id, text = id) => { const idx = data.samples.findIndex(s => s.sample_id === id); return idx < 0 ? h(text) : `<button class="event-link" data-action="sample-preview" data-index="${idx}" aria-haspopup="dialog" title="点击预览 · Cmd+点击查看流水线">${h(text)}</button>`; };
// The persisted experience model has no name field: use a readable, derived title.
const experienceName = e => short(String(e.content || e.trigger || '未命名经验').replace(/\s+/g,' ').trim().split(/(?<=[.!?。！？])\s+/)[0], 44);
const expButton = id => {
  const idx = data.experiences.findIndex(e => e.experience_id === id);
  const rows = candidateEntries(); const ci=rows.findIndex(r=>r.candidate.candidate_id===id);
  return idx < 0 ? (ci<0 ? h(id) : `<button class="event-link" data-action="candidate" data-index="${ci}">本轮候选经验</button>`) :
    `<button class="experience-link" title="${h(id)}" data-action="experience" data-index="${idx}" data-exp-preview="${idx}" aria-label="查看经验：${h(experienceName(data.experiences[idx]))}">${h(experienceName(data.experiences[idx]))}</button>`;
};
const candidateEntries = () => data.samples.flatMap((sample, sampleIndex) =>
  sampleCandidates(sample).map((candidate, localIndex) => ({sample, sampleIndex, localIndex, candidate})));
const candidateButton = (sampleIndex, localIndex, id) => {
  const idx = candidateEntries().findIndex(row => row.sampleIndex === sampleIndex && row.localIndex === localIndex);
  return idx < 0 ? h(id) : `<button class="event-link" data-action="candidate" data-index="${idx}">${h(id)}</button>`;
};

async function refresh(nextName = null) {
  const sequence = ++requestSequence;
  roleIoPending.clear();
  const selector = document.getElementById('run-select');
  selector.disabled = true;
  try {
    const listResponse = await fetch('/api/runs', {cache:'no-store'});
    if (!listResponse.ok) throw Error('无法读取运行目录');
    const listing = await listResponse.json();
    if (sequence !== requestSequence) return;
    selector.innerHTML = listing.runs.map(name => `<option value="${h(name)}">${h(name)}</option>`).join('');
    if (!listing.runs.length) {
      data = null; currentRun = null; roleIoCache.clear();
      selector.innerHTML = '<option>暂无实验</option>';
      document.getElementById('nav-sample-count').textContent = '0';
      document.getElementById('nav-exp-count').textContent = '0';
      app.innerHTML = '<div class="empty">暂无实验运行。启动新版实验后，点击“刷新数据”查看结果。</div>';
      return;
    }
    const requested = nextName ?? currentRun ?? new URLSearchParams(location.search).get('run');
    const name = listing.runs.includes(requested) ? requested : listing.default;
    selector.value = name;
    const response = await fetch(`/api/run?name=${encodeURIComponent(name)}`, {cache:'no-store'});
    const payload = await response.json();
    if (sequence !== requestSequence) return;
    if (!response.ok) throw Error(payload.error || '无法读取运行结果');
    const sameRun = currentRun === name;
    if (!sameRun) {
      sampleIndex = 0; expIndex = 0; candidateIndex = 0; memoryTab = 'pool'; query = ''; expQuery = ''; candidateQuery = ''; splitFilter = 'train'; resultFilter = 'all';
    }
    currentRun = name; if (!sameRun) categoryFilter = 'all';
    translations.clear(); translatedFields.clear(); roleIoCache.clear();
    data = payload;
    if (!['all','ungrouped'].includes(categoryFilter) && !(data.categories||[]).some(c=>c.category_id===categoryFilter)) categoryFilter='all';
    if (!data.samples.some(s=>s.split===splitFilter)) {
      splitFilter = data.samples.some(s=>s.split==='train') ? 'train' : data.samples.some(s=>s.split==='test') ? 'test' : 'unknown';
      resultFilter = 'all';
    }
    history.replaceState(null, '', `?run=${encodeURIComponent(name)}`);
    sampleIndex = Math.min(sampleIndex, Math.max(0, data.samples.length - 1));
    expIndex = Math.min(expIndex, Math.max(0, data.experiences.length - 1));
    candidateIndex = Math.min(candidateIndex, Math.max(0, candidateEntries().length - 1));
    document.getElementById('nav-sample-count').textContent = data.samples.length;
    document.getElementById('nav-exp-count').textContent = data.experiences.length;
    render({preserveListScroll:sameRun});
  } catch (error) {
    if (sequence === requestSequence) app.innerHTML = `<div class="warning">读取失败：${h(error.message)}。可尝试切换到其他运行日志。</div>`;
  } finally { if (sequence === requestSequence) selector.disabled = false; }
}
function render({preserveListScroll = false} = {}) {
  if (!data) return;
  const listScrollTop = preserveListScroll ? app.querySelector('.list-scroll')?.scrollTop : undefined;
  hideExperiencePreview();
  hideSamplePreview();
  document.querySelectorAll('.nav').forEach(button => {
    const active = button.dataset.page === page;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  app.innerHTML = page === 'overview' ? overview() : page === 'samples' ? samplesPage() : memoryPage();
  if (listScrollTop !== undefined) {
    const list = app.querySelector('.list-scroll');
    if (list) list.scrollTop = listScrollTop;
  }
}
function evolutionChart(title, rows, series, ceiling = null) {
  if (!rows.length) return `<section class="panel"><h2>${h(title)}</h2><p class="muted">暂无已完成训练任务</p></section>`;
  const ymax = ceiling || Math.max(1, ...rows.flatMap(r=>series.map(s=>s.value(r) ?? 0)));
  const x = i => 48 + (rows.length === 1 ? 0 : i * 700 / (rows.length - 1));
  const y = v => 160 - 130 * v / ymax;
  const lines = series.map(s=>{
    // 缺失值中断曲线，不将未知正确率画成 0。
    let path='', previous=false;
    rows.forEach((r,i)=>{const v=s.value(r); if(v==null){previous=false;return;}
      path+=`${previous?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)} `;previous=true;});
    return `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"/><circle cx="${x(rows.length-1)}" cy="${y(s.value(rows.at(-1))??0)}" r="${s.value(rows.at(-1))==null?0:3}" fill="${s.color}"/>`;
  }).join('');
  return `<section class="panel evolution-chart"><h2>${h(title)}</h2><p class="legend">${series.map((s,i)=>`<span class="chart-legend-${i}">${h(s.label)}</span>`).join(' · ')} · 横轴：已完成训练任务序号</p><svg viewBox="0 0 780 190" role="img" aria-label="${h(title)}"><path d="M48,30 V160 H748" fill="none" stroke="#ccc"/><text x="4" y="35">${ceiling===1?'100%':ymax}</text><text x="10" y="164">0</text><text x="48" y="182">1</text><text x="715" y="182">${rows.length}</text>${lines}</svg></section>`;
}
function overviewTable(title, headers, rows, foot='') {
  return `<section class="panel"><h2>${h(title)}</h2><div class="overview-table"><table><thead><tr>${headers.map(v=>`<th>${h(v)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v && typeof v==='object' ? (v.sample ? sampleButton(v.sample) : expButton(v.experience)) : h(v)}</td>`).join('')}</tr>`).join('')||`<tr><td colspan="${headers.length}">暂无数据</td></tr>`}</tbody></table></div>${foot?`<p class="legend">${h(foot)}</p>`:''}</section>`;
}
function trainingOverview() {
  const o=data.overview;
  if (!o) return '';
  return `<div class="metrics">${metric('训练进度',`${o.train_completed} / ${o.train_total}`,`样本错误 ${o.sample_errors} · Judge错误 ${o.judge_errors}`)}${metric('首次正确率',pct(o.initial.accuracy),`${o.initial.correct}/${o.initial.valid} 有效`)}${metric('任务救回率',pct(o.repair_rate),`${o.repair_successes}/${o.initial_failures} 初始失败`)}${metric('已声明采用',o.adopted,`${o.applications} 次实际应用`)}${metric('成功但未采用',o.success_without_candidate,'不验证候选、不写入Memory')}</div>` +
    evolutionChart('首次正确率趋势',o.curve,[{label:'累计',color:'#27664f',value:r=>r.cumulative.accuracy},{label:'最近50条',color:'#b47b36',value:r=>r.rolling.accuracy}],1) +
    overviewTable('候选、应用与入池',['候选提出','保留候选','应用次数','入池 / 文本更新','应用耗尽'],[[o.proposal_calls,o.candidates,o.applications,`${o.adds} / ${o.text_updates}`,o.validation_exhausted]],'Critic候选直接进入独立Generator应用，自主采用。候选P>N即具资格；样本结束最多新增一条，优先C2否则回退C1。整体失败也可入池。');
}
function overview() {
  const samples = data.samples, exps = data.experiences, count = key => data.counts[key] || 0;
  const trained = samples.filter(s => s.split === 'train').length;
  const tested = samples.filter(s => s.split === 'test').length;
  const success = count('initial_success') + count('application_success') + count('success_without_candidate');
  const candidates = samples.flatMap(sampleCandidates);
  const added = candidates.filter(c => c.experience_id).length;
  const max = Math.max(1, ...Object.values(data.counts));
  return heading('EXPERIMENT / OVERVIEW', '运行概览', '从一次运行到每个样本，再到经验池的累积变化。') + warnBlock() + lineage() + trainingOverview() +
    `<div class="metrics">
      ${metric('已记录样本',samples.length,`训练 ${trained} · 测试 ${tested}${trained + tested < samples.length ? ' · 未标注 ' + (samples.length - trained - tested) : ''}`)}
      ${metric('正向判别',success,`初始 ${count('initial_success')} · 应用 ${count('application_success') + count('success_without_candidate')}`)}
      ${metric('候选经验',candidates.length,`新增 ${added} · 未入池 ${candidates.length-added}`)}
      ${metric('当前经验',exps.length,data.overview?.mode==='train_only'?'在线训练持续更新经验池':'冻结测试阶段不更新经验池')}
      ${metric('证据事件',exps.reduce((sum,e)=>sum+e.evidence_history.length,0),'R0下游 · 候选应用')}
    </div><section class="panel overview-states"><div class="panel-title"><h2>结果分布</h2><small>SAMPLE STATES</small></div>${Object.entries(data.counts).map(([state,value])=>`<div class="bar-row ${successful(state)?'success':'fail'}"><span>${h(stateName[state]||state)}</span><div class="bar-track"><svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true"><rect class="bar-fill" width="${Math.round(100*value/max)}" height="8"/></svg></div><b>${value}</b></div>`).join('') || '<p class="muted">暂无样本轨迹</p>'}</section>`;
}
function metric(label,value,foot) {return `<div class="metric"><div class="metric-label">${h(label)}</div><div class="metric-value">${value}</div><div class="metric-foot">${h(foot)}</div></div>`;}
function sampleExperienceBadges(s) {
  const e=s.experience_status||{};
  const items=[`候选 ${sampleCandidates(s).length}`,
    `选中 ${e.selected??0}`, e.referenced==null?'引用待确认':`引用 ${e.referenced}`];
  if(e.added) items.push(`入池 ${e.added}`);
  return `<span class="sample-experience-status" aria-label="经验生成与利用状态">${items.map(text=>`<span>${h(text)}</span>`).join('')}</span>`;
}
function generatorAnalysis(s) {
  return s.generator_analysis ? `<details class="generator-analysis"><summary>Generator · 任务分析 <span>展开分析</span></summary><div class="body-text">${h(s.generator_analysis)}</div></details>` : '';
}
function samplesPage() {
  const matchesSplit = s => splitFilter === 'unknown' ? s.split !== 'train' && s.split !== 'test' : s.split === splitFilter;
  const group = data.samples.map((s,i)=>({s,i})).filter(({s})=>matchesSplit(s));
  const matchesResult = s => resultFilter === 'all' || (resultFilter === 'application' ? ['application_success','success_without_candidate'].includes(s.state) : resultFilter === 'failure' ? !successful(s.state) : s.state === resultFilter);
  const visible = group.filter(({s})=>matchesResult(s) && (!query || `${s.sample_id} ${s.task||''} ${s.state}`.toLowerCase().includes(query.toLowerCase())));
  // Never show a test detail beside a training list (or a filtered-out detail).
  const current = visible.find(({i})=>i===sampleIndex) || visible[0];
  const selected = current?.s;
  const splits = [['train','训练',data.samples.filter(s=>s.split==='train').length],['test','测试',data.samples.filter(s=>s.split==='test').length]];
  const unknown = data.samples.length - splits[0][2] - splits[1][2];
  if (unknown) splits.push(['unknown','未标注',unknown]);
  const results = splitFilter === 'train' ? [['all','全部'],['initial_success','初始成功'],['application','应用成功'],['failure','失败 / 中断']] : splitFilter === 'test' ? [['all','全部'],['initial_success','测试通过'],['failure','测试未通过 / 中断']] : [['all','全部'],['failure','失败 / 中断']];
  return warnBlock() +
    `<div class="sample-split-tabs" role="tablist" aria-label="样本阶段">${splits.map(([key,label,count])=>`<button role="tab" aria-selected="${splitFilter===key}" class="split-tab ${splitFilter===key?'active':''}" data-action="split-filter" data-split="${key}">${label}<span class="mono">${count}</span></button>`).join('')}</div>
    <div class="workbench sample-workbench"><section class="panel list-panel"><div class="list-head"><h2>${splitFilter==='train'?'训练':splitFilter==='test'?'测试':'未标注'}样本 <span class="muted mono">${visible.length} / ${group.length}</span></h2><input class="search" data-search="sample" placeholder="搜索 ID 或任务内容…" value="${h(query)}" aria-label="搜索样本"><div class="filters">${results.map(([key,label])=>`<button class="filter ${resultFilter===key?'active':''}" data-action="result-filter" data-filter="${key}">${label}</button>`).join('')}</div></div>
    <div class="list-scroll">${visible.map(({s,i})=>`<button class="list-item ${i===current?.i?'active':''}" data-action="sample" data-index="${i}"><span class="row-top"><strong title="${h(s.sample_id)}">${h(sampleTitle(s))}</strong><span class="list-state ${statusClass(s.state)}" title="${h(stateName[s.state]||s.state)}">${h(s.split==='test'&&s.state==='initial_success'?'测试通过':stateName[s.state]||s.state)}</span></span><p>${h(short(s.task||s.split||'未提供任务文本',130))}</p>${sampleExperienceBadges(s)}</button>`).join('')||'<div class="empty">没有匹配的样本</div>'}</div></section>
    <section class="panel detail">${selected ? sampleDetail(selected) : '<div class="empty">当前阶段没有匹配的样本</div>'}</section></div>`;
}
function sampleInput(s, field, label) {
  if (typeof s[field] !== 'string') return '';
  const key = translationKey(currentRun, s.sample_id, field);
  const translated = translatedFields.has(key) && translations.has(key);
  const action = translated ? '显示原文' : '翻译成中文';
  const scrollable = field !== 'task' ? `role="region" tabindex="0" aria-label="${h(label)} 回答，可滚动"` : '';
  return `<div class="input-card field-${field}"><div class="input-heading"><strong>${h(label)}</strong><button class="translate-button" data-action="translate" data-index="${data.samples.indexOf(s)}" data-field="${field}" aria-label="${h(label)}：${action}">${action}</button></div><div class="input-content" ${scrollable}>${h(translated ? translations.get(key) : s[field])}</div><div class="translation-feedback" role="status" aria-live="polite"></div></div>`;
}
function sampleDetail(s) {
  const tabs = `<div class="role-tab-strip" role="tablist" aria-label="样本流程与角色原始输入输出">${roleTabs.map(([key,label])=>`<button class="role-tab ${sampleTab===key?'active':''}" role="tab" aria-selected="${sampleTab===key}" data-action="sample-tab" data-tab="${key}">${label}</button>`).join('')}</div>`;
  return tabs + (sampleTab === 'pipeline' ? pipelineDetail(s) : roleDetail(s, sampleTab));
}
function roleIoKey(run, sampleId) { return JSON.stringify([run, sampleId]); }
async function fetchRoleIo(s) {
  const run = currentRun, sequence = requestSequence, key = roleIoKey(run, s.sample_id);
  if (roleIoPending.has(key)) return;
  roleIoPending.set(key, sequence);
  try {
    const url = `/api/role-io?name=${encodeURIComponent(run)}&sample_id=${encodeURIComponent(s.sample_id)}`;
    const response = await fetch(url, {cache:'no-store'});
    const result = await response.json();
    if (sequence !== requestSequence) return;
    if (!response.ok) throw Error(result.error || '角色记录读取失败');
    roleIoCache.set(key, result);
  } catch (error) {
    if (sequence === requestSequence) roleIoCache.set(key, {error:error.message});
  } finally {
    if (roleIoPending.get(key) === sequence) roleIoPending.delete(key);
    if (sequence === requestSequence && page === 'samples' && sampleTab !== 'pipeline') render({preserveListScroll:true});
  }
}
const pretty = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? '—';
function roleDetail(s, role) {
  const key = roleIoKey(currentRun, s.sample_id);
  if (!roleIoCache.has(key)) fetchRoleIo(s);
  const log = roleIoCache.get(key);
  const head = `<div class="detail-header"><p>样本 ${h(s.sample_id)}</p></div>`;
  if (!log) return head + '<p class="empty">正在读取角色调用记录…</p>';
  if (log.error) return head + `<div class="warning">${h(log.error)}</div>`;
  if (!log.available) return head + '<div class="role-empty">当前运行尚无角色输入输出记录。</div>';
  const entries = log.entries.filter(item => item.role === role);
  const warnings = log.warnings?.length ? `<div class="warning">${log.warnings.map(h).join('<br>')}</div>` : '';
  if (!entries.length) return head + warnings + '<div class="role-empty">这个样本未调用该角色。可能是流程在此前结束，或尚未完成该阶段。</div>';
  return head + warnings + `<div class="role-call-count">${entries.length} 次调用</div>` +
    entries.map((entry,index)=>`<section class="role-call"><div class="role-call-head"><strong>调用 ${index+1}${entry.attempt>1?` · HTTP 尝试 ${h(entry.attempt)}`:''}</strong>${badge(entry.error ? entry.error : entry.status ? `HTTP ${entry.status}` : entry.backend==='fake'?'Fake':'成功', entry.error?'bad':'ok')}</div>
    <div class="io-grid"><div class="io-side"><h3>输入 <span>INPUT</span></h3>${Array.isArray(entry.messages)?entry.messages.map(m=>`<div class="io-message"><div class="io-label">${h(m.role||'消息')} · 实际发送</div><pre>${h(pretty(m.content))}</pre></div>`).join(''):`<div class="io-message"><div class="io-label">${entry.backend==='fake'?'Fake 角色输入':'角色参数'}</div><pre>${h(pretty(entry.input))}</pre></div>`}</div>
    <div class="io-side"><h3>输出 <span>OUTPUT</span></h3>${entry.raw_output!=null?`<div class="io-message"><div class="io-label">模型原文</div><pre>${h(pretty(entry.raw_output))}</pre></div>`:entry.output!=null?`<div class="io-message"><div class="io-label">${entry.backend==='fake'?'Fake 返回':'结构化返回'}</div><pre>${h(pretty(entry.output))}</pre></div>`:'<div class="io-message"><div class="io-label">无输出</div><p>该尝试在获得有效模型文本前失败。</p></div>'}</div></div></section>`).join('');
}
function traceSection(number,title,content,extra='') {
  return `<section class="detail-section ${extra}"><div class="section-title"><span class="section-number">${number}</span><h3>${title}</h3></div>${content}</section>`;
}
function creditStats(credit) {
  return `<div class="credit-summary"><span>候选证据</span><span class="credit-positive">P <b>${n(credit.positive_credit)}</b></span><span class="credit-negative">N <b>${n(credit.negative_credit)}</b></span><span class="muted">净方向 ${n(credit.positive_credit-credit.negative_credit)}</span></div>`;
}
function proposalCard(p) {
  const c=p.candidate;
  return `<article class="candidate proposal-card">
    <div class="candidate-head"><span class="section-kicker">CRITIC / ${String(p.round_number).padStart(2,'0')}</span><strong>本轮候选</strong>${c?badge('直接应用候选','neutral'):''}</div>
    ${p.error?`<div class="warning">${h(p.error)}</div>`:''}
    ${c?`<p class="candidate-content">${h(c.content)}</p><div class="candidate-trigger"><span>适用条件</span><p>${h(c.trigger)}</p></div>
      ${c.experience_id?`<div class="candidate-admission">${badge('已入池','ok')}${expButton(c.experience_id)}</div>`:''}`:'<p class="muted">未取得候选</p>'}
    <details class="private-analysis"><summary>Critic私有分析</summary><div class="body-text">${h(p.analysis)}</div></details>
  </article>`;
}
function pipelineDetail(s) {
  const selected=s.selected_experiences||[], proposals=s.proposal_attempts||[], applications=s.application_attempts||[];
  const label=sampleLabel(s);
  const recorded=value=>value?'done':'skip';
  const steps=[['输入',recorded(typeof s.task==='string')],['R0',s.initial_judgment?(s.initial_judgment.success?'done':'error'):recorded(s.initial_rubric)],['候选',recorded(proposals.length)],['应用',applications.some(a=>a.judgment?.success)?'done':applications.length?'error':'skip'],['入池',recorded(!!s.admitted_experience_id)]];
  const header=`<div class="detail-header"><div><span class="section-kicker">${h(label.subset)} / SAMPLE</span><h2>样本 ${h(label.number)}</h2><p class="artifact-id" title="${h(s.sample_id)}">${h(s.sample_id)}</p></div>${badge(stateName[s.state]||s.state,statusClass(s.state))}</div>
    <div class="trace-progress" aria-label="已记录流程">${steps.map(([title,status],i)=>`<div class="trace-step ${status}" title="${h(title)} · ${status==='done'?'已完成':status==='error'?'未通过或缺少有效结果':'未进入'}"><span>${status==='done'?'✓':status==='error'?'!':String(i+1).padStart(2,'0')}</span><b>${title}</b></div>`).join('')}</div>`;
  const inputs=traceSection('01','任务与偏好对',`<div class="sample-inputs">${sampleInput(s,'task','TASK · 任务')}${sampleInput(s,'chosen','CHOSEN · 偏好回答')}${sampleInput(s,'rejected','REJECTED · 对照回答')}</div>`,'input-section');
  const memory=traceSection('02','固定经验 E',`<div class="retrieval-summary"><span>召回 <b>${s.retrieved_experience_ids?.length||0}</b></span><span>选中 <b>${selected.length}</b></span><small>R0与两轮应用共享</small></div>${selected.length?`<div class="selected-experiences">${selected.map(e=>`<div>${expButton(e.experience_id)}</div>`).join('')}</div>`:'<p class="muted empty-note">本次未选用已有经验。</p>'}<details class="audit-details"><summary>查看固定经验文本快照</summary><pre>${h(pretty(selected))}</pre></details>`);
  const initial=s.initial_rubric?traceSection('03','初始标准 R0',`${judgmentStats(s.initial_judgment)}${generatorAnalysis(s)}${criteria(s.initial_rubric,s.initial_judgment)}<details class="audit-details"><summary>查看R0正负信用</summary><pre>${h(pretty(s.downstream_credits))}</pre></details>`):'';
  const rounds=[1,2].map(round=>{
    const ps=proposals.filter(p=>p.round_number===round), a=applications.find(a=>a.round_number===round);
    if(!ps.length&&!a) return '';
    const application=a?`<div class="attempt"><div class="attempt-top"><strong>新Generator · 应用结果</strong>${badge(a.adopted===null?'采用未知':a.adopted?'声明采用':'未采用候选',a.adopted?'ok':'neutral')}</div><p class="adoption-meta">声明采用：${a.adopted===null?'未知':a.adopted?'是':'否'} · 引用 ${a.adopted_criterion_ids.length} 条标准</p>${a.adopted_criterion_ids.length?`<details class="audit-details"><summary>查看引用标准ID</summary><pre>${h(a.adopted_criterion_ids.join('\n'))}</pre></details>`:''}${a.error?`<div class="warning">${h(a.error)}</div>`:''}${a.rubric?`${judgmentStats(a.judgment)}${criteria(a.rubric,a.judgment)}`:''}${a.credit?creditStats(a.credit):''}</div>`:'<div class="round-end"><span class="round-end-dot"></span>没有进入Generator应用</div>';
    return traceSection(String(round+3).padStart(2,'0'),`应用轮 ${round}`,`<div class="round-route"><span>Critic</span><i>→</i><span>新Generator</span><i>→</i><span>Judge</span></div>${ps.map(proposalCard).join('')}${application}`,'application-section');
  }).join('');
  return header+`${s.sample_error?`<div class="warning">${h(s.sample_error.stage)} · ${h(s.sample_error.message)}</div>`:''}${s.judge_error?`<div class="warning">Judge判断未知：${h(s.judge_error)}</div>`:''}`+inputs+memory+initial+rounds;
}
function judgmentStats(j) {
  if(!j) return '';
  return `<div class="judgment-summary ${j.success?'judgment-positive':'judgment-negative'}"><div class="margin-score"><span>MARGIN</span><strong>${j.margin>0?'+':''}${n(j.margin)}</strong></div><div class="answer-score"><span>CHOSEN</span><b>${n(j.chosen_score)}</b></div><div class="answer-score"><span>REJECTED</span><b>${n(j.rejected_score)}</b></div>${badge(j.success?'判别成功':'判别未通过',j.success?'ok':'warn')}</div>`;
}
function criteria(rubric, judgment) {
  return `<div class="rubric-list" role="region" tabindex="0" aria-label="Rubric 标准，可滚动">${rubric.criteria.map((c,index)=>{
    const verdict=judgment?.criterion_judgments?.find(j=>j.criterion_id===c.criterion_id);
    return `<article class="criterion"><div class="criterion-head"><strong title="${h(c.criterion_id)}">标准 ${index+1}</strong><span class="criterion-meta">分配权重 ${h(c.weight)} · 权重 ${pct(judgment?.normalized_weights?.[c.criterion_id])}</span>${verdict?`<span class="judgment-pills"><span class="truth ${verdict.chosen?'yes':'no'}" title="Chosen：${verdict.chosen?'是':'否'}">C ${verdict.chosen?'✓':'✕'}</span><span class="truth ${verdict.rejected?'yes':'no'}" title="Rejected：${verdict.rejected?'是':'否'}">R ${verdict.rejected?'✓':'✕'}</span></span>`:''}</div><div class="criterion-body"><p>${h(c.question)}</p>${c.source_experiences?.length?`<div class="sources">来源经验：${c.source_experiences.map(expButton).join(' · ')}</div>`:''}</div>${verdict?`<details class="criterion-audit"><summary>Judge 判断依据</summary><p><strong>Chosen：</strong>${h(verdict.chosen_reason)}</p><p><strong>Rejected：</strong>${h(verdict.rejected_reason)}</p></details>`:''}<details class="criterion-audit"><summary>标准ID</summary><code>${h(c.criterion_id)}</code></details></article>`;
  }).join('')}</div><p class="rubric-legend">C / Chosen · R / Rejected · ✓ 满足标准 · ✕ 不满足</p>`;
}
function experienceUsage(e) {
  // Count distinct samples whose R0 cites this experience, not merely selector choices.
  return new Set(data.samples.filter(s => s.initial_rubric?.criteria?.some(c =>
    c.source_experiences?.includes(e.experience_id))).map(s => s.sample_id)).size;
}
function experienceUpdates(e) {return e.update_history.length;}
function experienceEntries() {
  const entries = data.experiences.map((e,i)=>({e,i,usage:experienceUsage(e)}));
  if (expSort === 'original') return entries;
  const field = expSort.startsWith('confidence_') ? 'confidence' : 'usage';
  const direction = expSort.endsWith('_asc') ? 1 : -1;
  return entries.sort((a,b)=>direction*((field==='confidence'?a.e.confidence:a.usage)-(field==='confidence'?b.e.confidence:b.usage)) || a.i-b.i);
}
function candidateStatus(c) {
  return c.experience_id ? badge('已新增','ok') :
    badge('未入池','neutral');
}
function memoryPage() {
  const rows = candidateEntries();
  const pool = memoryTab !== 'candidates';
  const group = experienceEntries();
  const title = pool ? '经验池' : '候选经验';
  const tabs = [['pool','经验池',data.experiences.length],['candidates','候选经验',rows.length]];
  const visible = pool ? group.filter(({e})=>(categoryFilter==='all'||(e.category_id||'ungrouped')===categoryFilter) && (!expQuery||`${e.experience_id} ${e.content} ${e.trigger} ${e.category?.name||''} ${e.category?.description||''}`.toLowerCase().includes(expQuery.toLowerCase()))) :
    rows.map((entry,i)=>({...entry,i})).filter(({candidate:c,sample:s})=>!candidateQuery||`${c.candidate_id} ${c.content} ${c.trigger} ${s.sample_id}`.toLowerCase().includes(candidateQuery.toLowerCase()));
  const current = visible.find(({i})=>i===(pool?expIndex:candidateIndex)) || visible[0];
  return warnBlock() + `<div class="sample-split-tabs" role="tablist" aria-label="经验分类">
    ${tabs.map(([key,label,count])=>`<button role="tab" aria-selected="${memoryTab===key}" class="split-tab ${memoryTab===key?'active':''}" data-action="memory-tab" data-tab="${key}">${label} <span class="mono">${count}</span></button>`).join('')}</div>
    <div class="workbench memory-workbench"><section class="panel list-panel memory-list"><div class="list-head"><h2>${title} <span class="muted mono">${visible.length} / ${pool?group.length:rows.length}</span></h2><input class="search" data-search="${pool?'experience':'candidate'}" placeholder="搜索内容或 ID…" value="${h(pool?expQuery:candidateQuery)}" aria-label="搜索${title}">${pool?`<label class="experience-sort">类别<select data-category-filter aria-label="经验类别">${[['all','全部类别'],['ungrouped','未分类'],...(data.categories||[]).map(c=>[c.category_id,c.name])].map(([key,label])=>`<option value="${h(key)}"${categoryFilter===key?' selected':''}>${h(label)}</option>`).join('')}</select></label><label class="experience-sort">排序<select data-experience-sort aria-label="经验排序">${experienceSortOptions.map(([value,label])=>`<option value="${value}"${expSort===value?' selected':''}>${label}</option>`).join('')}</select></label>`:''}</div>
    <div class="list-scroll">${pool ? visible.map(({e,i,usage})=>`<button class="list-item ${i===current?.i?'active':''}" data-action="experience" data-index="${i}"><span class="row-top"><strong title="${h(experienceName(e))}">${h(experienceName(e))}</strong>${badge(pct(e.confidence),'ok')}</span><p class="mono" title="${h(e.experience_id)}">${h(e.experience_id)}</p><div class="list-metrics"><span title="训练及测试样本 R0 中引用该经验的次数（每个样本最多计一次）">引用 ${usage} 次</span><span title="经验文本重新归纳次数">归纳 ${experienceUpdates(e)} 次 · 案例 ${e.case_count}/10 · v${e.text_version}${e.text_frozen?' · 已冻结':''}</span></div></button>`).join('') : visible.map(({candidate:c,sample:s,i})=>`<button class="list-item ${i===current?.i?'active':''}" data-action="candidate" data-index="${i}"><span class="row-top"><strong title="${h(c.candidate_id)}">${h(short(c.content,58))}</strong>${candidateStatus(c)}</span><p>${h(short(c.trigger,100))}</p><span class="candidate-origin">${h(short(s.sample_id,36))}</span></button>`).join('')}${visible.length?'':'<div class="empty">没有匹配的经验</div>'}</div></section>
    <section class="panel detail">${current ? pool?experienceDetail(current.e):candidateDetail(current) : `<div class="empty">${visible.length===0 && (pool?expQuery:candidateQuery)?'没有匹配的经验。':pool?'暂无经验。':'当前轨迹没有保留候选经验。'}</div>`}</section></div>`;
}
function candidateDetail({candidate:c,sample:s}) {
  const a=(s.application_attempts||[]).find(a=>a.candidate_id===c.candidate_id);
  return `<div class="detail-header"><div><span class="section-kicker">CRITIC / CANDIDATE</span><h2>候选经验</h2><p class="artifact-id">${h(c.candidate_id)}</p><p>来源样本：${sampleButton(s.sample_id)} · ${h(s.split || '未标注')}</p></div>${candidateStatus(c)}</div>
    <div class="detail-section"><div class="section-kicker">01 / CANDIDATE</div><h3>候选内容</h3><p class="exp-intro">${h(c.content)}</p><div class="exp-trigger"><strong>TRIGGER · 适用条件</strong><br>${h(c.trigger)}</div></div>
    <div class="detail-section"><div class="section-kicker">02 / DECISION</div><h3>提供、采用与入池</h3><div class="mini-stats"><span class="mini-stat">入池 <b>${c.experience_id?'是':'否'}</b></span></div>${a?`<p class="body-text">已提供给Generator · 声明采用：${a.adopted===null?'未知':a.adopted?'是':'否'} · 净正向资格：${a.eligible?'是':'否'} · Judge：${a.judgment?(a.judgment.success?'成功':'失败'):'无有效判断'}${a.credit?` · P ${n(a.credit.positive_credit)} / N ${n(a.credit.negative_credit)}`:''}</p>`:'<p class="body-text">未提供给Generator</p>'}${c.experience_id?`<p class="body-text">目标经验：${expButton(c.experience_id)}</p>`:''}</div>
    <div class="detail-section"><div class="section-kicker">03 / SOURCE</div><h3>来源轨迹</h3><p class="body-text">${sampleButton(s.sample_id)}</p><p class="legend">这里只列出完整轨迹中保留的候选；Critic 原始输出见对应样本的角色页签。</p></div>`;
}
function experienceDetail(e) {
  const events = experienceTimeline(e);
  return `<div class="detail-header"><div><span class="section-kicker">CANONICAL EXPERIENCE / ${String(data.experiences.indexOf(e)+1).padStart(2,'0')}</span><h2>${h(experienceName(e))}</h2><p>名称由正文摘要生成 · 文本归纳历史见时间线</p></div>${badge(pct(e.confidence),'ok')}</div>
    <div class="detail-section"><div class="section-kicker">01 / PRINCIPLE</div><h3>经验内容</h3><p class="exp-intro">${h(e.content)}</p><div class="exp-trigger"><strong>TRIGGER · 适用条件</strong><br>${h(e.trigger)}</div></div>
    <div class="detail-section"><h3>所属类别 · ${h(e.category?.name||'未分类')}</h3><p class="body-text">${h(e.category?.description||'该经验作为独立检索单位，仍可直接被召回。')}</p></div>
    <div class="detail-section experience-cases"><div class="section-kicker">02 / CASES</div><h3>归纳案例 · ${e.cases.length} 个</h3><p class="legend">点击任务预览两个回答；Cmd/Ctrl＋点击查看完整流水线。案例保留当时的经验和判断，历史反馈不代表改写后的文本已直接验证。</p><div class="case-list">${e.cases.map((c,index)=>experienceCase(c,e,index)).join('')}</div></div>
    <div class="detail-section"><div class="section-kicker">03 / RELIABILITY</div><h3>可靠性与证据</h3><div class="mini-stats"><span class="mini-stat">使用 <b>${experienceUsage(e)} 次</b></span><span class="mini-stat">归纳 <b>${experienceUpdates(e)} 次</b></span><span class="mini-stat">α <b>${n(e.alpha)}</b></span><span class="mini-stat">β <b>${n(e.beta)}</b></span><span class="mini-stat">CONFIDENCE <b>${pct(e.confidence)}</b></span><span class="mini-stat">证据量 <b>${n(e.alpha+e.beta-2)}</b></span><span class="mini-stat">下游任务 <b>${e.downstream_task_count||0}</b></span><span class="mini-stat">待归纳反馈 <b>${e.pending_feedback_count||0}/3</b></span></div>${lineChart(events)}<p class="body-text">正确区分 ${e.usage_counts?.positive||0} · 反向区分 ${e.usage_counts?.negative||0} · 双方满足 ${e.usage_counts?.both_true||0} · 双方未满足 ${e.usage_counts?.both_false||0} · 区分率 ${pct(e.discrimination_rate)}</p><p class="legend">以上按被引用标准计次，同判不改信用、不参与归纳；区分率仅展示，不提供给Selector。</p><p class="legend">从 Beta(1,1) 先验开始，正负证据分别增加 α/β，不先抵消。置信度是 α/(α+β)，并非校准后的成功概率；历史信用不证明新文本版本已验证。</p></div>
    <div class="detail-section"><div class="section-kicker">04 / ORIGIN & EVOLUTION</div><h3>演化时间线 · ${events.length} 个事件 · 案例 ${e.case_count}/10 · v${e.text_version}${e.text_frozen?' · 文本冻结':''}</h3><div class="timeline">${events.map(ev=>`<div class="event"><div class="event-top"><strong>${ev.kind==='prior'?'初始先验':ev.kind==='text_update'?'文本重新归纳':ev.kind==='application_credit'?'入池应用证据':'R0下游证据'}</strong><span>${pct(ev.confidence)}</span></div><p>${ev.sample_id?sampleButton(ev.sample_id):'无来源'} · α ${n(ev.alpha)} / β ${n(ev.beta)}<br>P ${n(ev.positive_credit||0)} / N ${n(ev.negative_credit||0)}${ev.text_version?` · 使用版本 v${ev.text_version}`:''}${ev.text_update?`<br>文本 v${ev.text_update.previous_version} → v${ev.text_update.text_version}<br>${h(ev.text_update.content)}<br>适用条件：${h(ev.text_update.trigger)}<br>归纳案例：${ev.text_update.case_ids.map(id=>sampleButton(id)).join('、')}${ev.text_update.error?`<br>更新失败：${h(ev.text_update.error)}`:''}<details><summary>查看原文本</summary><p>${h(ev.text_update.previous_content)}</p><p>${h(ev.text_update.previous_trigger)}</p></details>`:''}</p></div>`).join('')}</div></div>`;

}
function experienceCase(c,e,index) {
  const cited = {...c.rubric, criteria:c.rubric.criteria.filter(item=>item.source_experiences.includes(c.source_id))};
  return `<article class="experience-case"><div class="case-heading"><span class="mono">${String(index+1).padStart(2,'0')}</span>${sampleButton(c.sample_id,short(c.task.replace(/\s+/g,' '),110))}</div><p class="case-meta mono">${h(c.sample_id)} · P ${n(c.credit.positive_credit)} / N ${n(c.credit.negative_credit)}${c.source_id!==e.experience_id?' · 入池原题':''}</p><details class="audit-details"><summary>查看当时经验与引用标准</summary><p class="legend">当时引用：${expButton(c.source_id)} · v${c.text_version}</p><p class="exp-intro">${h(c.content)}</p><p class="legend">适用条件：${h(c.trigger)}</p>${criteria(cited,c.judgment)}</details></article>`;
}
function experienceTimeline(e) {return e.evolution;}

function lineChart(events) {
  const w=600, height=150, left=34, right=14, top=13, bottom=24;
  const x=i=>left+(w-left-right)*(events.length===1?.5:i/(events.length-1));
  const y=value=>top+(height-top-bottom)*(1-value);
  const points=events.map((e,i)=>`${x(i)},${y(e.confidence)}`).join(' ');
  return `<svg class="chart" viewBox="0 0 ${w} ${height}" role="img" aria-label="经验置信度变化折线图">${[0,.25,.5,.75,1].map(v=>`<line class="grid" x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}"/><text x="0" y="${y(v)+3}">${Math.round(v*100)}%</text>`).join('')}<polyline class="line" points="${points}"/>${events.map((e,i)=>`<circle cx="${x(i)}" cy="${y(e.confidence)}" r="3.2"><title>${h(i===0?(e.kind==='prior'?'先验':'入池'):e.sample_id)}: ${pct(e.confidence)}</title></circle>`).join('')}<text x="${left}" y="${height-3}">${events[0]?.kind==='prior'?'先验':'入池'}</text><text x="${w-right-37}" y="${height-3}">当前</text></svg>`;
}
const experiencePreview = document.getElementById('experience-preview');
let previewCloseTimer = null, activePreviewButton = null;
function hideExperiencePreview() {
  clearTimeout(previewCloseTimer);
  experiencePreview.hidden = true;
  activePreviewButton?.removeAttribute('aria-describedby');
  activePreviewButton = null;
}
function positionExperiencePreview(button) {
  const rect = button.getBoundingClientRect();
  const viewport = button.closest('.retrieval-list')?.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top > innerHeight ||
      (viewport && (rect.bottom < viewport.top || rect.top > viewport.bottom))) {
    hideExperiencePreview();
    return;
  }
  const width = experiencePreview.offsetWidth, height = experiencePreview.offsetHeight;
  const left = rect.right + width + 12 <= innerWidth ? rect.right + 12 : rect.left - width - 12;
  experiencePreview.style.left = `${Math.max(12, Math.min(left, innerWidth - width - 12))}px`;
  experiencePreview.style.top = `${Math.max(12, Math.min(rect.top, innerHeight - height - 12))}px`;
}
function showExperiencePreview(button) {
  const exp = data?.experiences[Number(button.dataset.expPreview)];
  if (!exp) return;
  clearTimeout(previewCloseTimer);
  document.querySelector('[aria-describedby="experience-preview"]')?.removeAttribute('aria-describedby');
  experiencePreview.innerHTML = `<div class="preview-label">经验预览 · ${pct(exp.confidence)} 可信度</div><strong>${h(experienceName(exp))}</strong><p>${h(exp.content)}</p><div class="preview-trigger"><b>适用条件</b><br>${h(exp.trigger)}</div><small>点击名称查看证据及演化 →</small>`;
  experiencePreview.hidden = false;
  activePreviewButton = button;
  button.setAttribute('aria-describedby', 'experience-preview');
  positionExperiencePreview(button);
}
function schedulePreviewClose() {
  clearTimeout(previewCloseTimer);
  previewCloseTimer = setTimeout(hideExperiencePreview, 180);
}
document.addEventListener('mouseover', event => {
  const button = event.target.closest('[data-exp-preview]');
  if (button && !button.contains(event.relatedTarget)) showExperiencePreview(button);
});
document.addEventListener('mouseout', event => {
  const button = event.target.closest('[data-exp-preview]');
  if (button && !button.contains(event.relatedTarget)) schedulePreviewClose();
});
experiencePreview.addEventListener('mouseenter', () => clearTimeout(previewCloseTimer));
experiencePreview.addEventListener('mouseleave', schedulePreviewClose);
document.addEventListener('focusin', event => {
  const button = event.target.closest('[data-exp-preview]');
  if (button) showExperiencePreview(button);
});
document.addEventListener('focusout', event => {
  if (event.target.closest('[data-exp-preview]')) schedulePreviewClose();
});
document.addEventListener('scroll', event => {
  if (activePreviewButton && !experiencePreview.contains(event.target))
    positionExperiencePreview(activePreviewButton);
}, true);
document.addEventListener('keydown', event => { if (event.key === 'Escape') hideExperiencePreview(); });
const samplePreview = document.getElementById('sample-preview');
let samplePreviewOpener = null;
function hideSamplePreview() {
  if (!samplePreview.open) return;
  samplePreview.close();
  if (samplePreviewOpener?.isConnected) samplePreviewOpener.focus();
  samplePreviewOpener = null;
}
function showSamplePreview(button) {
  const sample = data?.samples[Number(button.dataset.index)];
  if (!sample) return;
  hideExperiencePreview();
  samplePreviewOpener = button;
  samplePreview.innerHTML = `<div class="sample-preview-heading"><h2 id="sample-preview-title">样本预览</h2><button data-action="close-sample-preview" aria-label="关闭样本预览" autofocus>×</button></div>${['task','chosen','rejected'].map(field=>sampleInput(sample, field, field.toUpperCase())).join('')}`;
  samplePreview.showModal();
}
samplePreview.addEventListener('click', event => {
  if (event.target !== samplePreview) return;
  const rect = samplePreview.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) hideSamplePreview();
});
samplePreview.addEventListener('cancel', event => { event.preventDefault(); hideSamplePreview(); });
async function translateSampleField(button) {
  const field = button.dataset.field;
  const sample = data?.samples[Number(button.dataset.index)];
  if (!['task', 'chosen', 'rejected'].includes(field) || typeof sample?.[field] !== 'string') return;
  const run = currentRun, sequence = requestSequence;
  const key = translationKey(run, sample.sample_id, field);
  const card = button.closest('.input-card');
  const content = card.querySelector('.input-content'), feedback = card.querySelector('.translation-feedback');
  if (translations.has(key)) {
    const showTranslation = !translatedFields.has(key);
    if (showTranslation) translatedFields.add(key); else translatedFields.delete(key);
    content.textContent = showTranslation ? translations.get(key) : sample[field];
    content.scrollTop = 0;
    button.textContent = showTranslation ? '显示原文' : '翻译成中文';
    button.setAttribute('aria-label', `${field.toUpperCase()}：${button.textContent}`);
    feedback.textContent = '';
    return;
  }
  button.disabled = true;
  button.textContent = '翻译中…';
  feedback.textContent = '';
  try {
    const response = await fetch('/api/translate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({run, sample_id:sample.sample_id, field}),
    });
    const result = await response.json();
    if (sequence !== requestSequence) return;
    if (!response.ok) throw Error(result.error || '翻译失败，请稍后重试。');
    if (typeof result.translation !== 'string') throw Error('翻译服务返回格式无效。');
    translations.set(key, result.translation);
    translatedFields.add(key);
    if (button.isConnected && currentRun === run) {
      content.textContent = result.translation;
      content.scrollTop = 0;
      button.textContent = '显示原文';
      button.setAttribute('aria-label', `${field.toUpperCase()}：显示原文`);
    }
  } catch (error) {
    if (button.isConnected && currentRun === run && sequence === requestSequence) feedback.textContent = error.message;
  } finally {
    if (button.isConnected) {
      button.disabled = false;
      if (button.textContent === '翻译中…') button.textContent = '翻译成中文';
    }
  }
}
// Delegated click navigation keeps all dynamic artifacts inert (no inline JS).
document.addEventListener('click', event => {
  const button=event.target.closest('[data-page],[data-action]');
  if(!button) return;
  if(button.dataset.page){page=button.dataset.page;render();window.scrollTo(0,0);return;}
  const action=button.dataset.action;
  if(action==='close-sample-preview'){hideSamplePreview();return;}
  if(action==='sample-preview'){
    if(!event.metaKey && !event.ctrlKey){showSamplePreview(button);return;}
    sampleTab='pipeline';
  }
  if(action==='translate'){translateSampleField(button);return;}
  if(action==='sample-tab' && roleTabs.some(([key])=>key===button.dataset.tab)){
    sampleTab=button.dataset.tab;render({preserveListScroll:true});return;
  }
  if(action==='memory-tab' && ['pool','candidates'].includes(button.dataset.tab)){
    const preserveListScroll=page==='memory'&&memoryTab===button.dataset.tab;
    memoryTab=button.dataset.tab;render({preserveListScroll});return;
  }
  if(action==='refresh'){refresh(currentRun);return;}
  if(action==='split-filter'){splitFilter=button.dataset.split;resultFilter='all';query='';sampleIndex=data.samples.findIndex(s=>splitFilter==='unknown'?s.split!=='train'&&s.split!=='test':s.split===splitFilter);render();return;}
  if(action==='result-filter'){resultFilter=button.dataset.filter;render();return;}
  if(action==='sample'||action==='sample-preview'){sampleIndex=Number(button.dataset.index);const sample=data.samples[sampleIndex];const nextSplit=sample?.split==='train'||sample?.split==='test'?sample.split:'unknown';const preserveListScroll=page==='samples'&&splitFilter===nextSplit;if(!preserveListScroll){resultFilter='all';query='';}splitFilter=nextSplit;page='samples';render({preserveListScroll});window.scrollTo(0,0);return;}
  if(action==='experience'){
    const index=Number(button.dataset.index), experience=data.experiences[index];
    if(!experience) return;
    const nextTab='pool';
    const preserveListScroll=page==='memory'&&memoryTab===nextTab;
    if(!preserveListScroll) expQuery='';
    expIndex=index;
    memoryTab=nextTab;page='memory';render({preserveListScroll});
    if(!preserveListScroll) window.scrollTo(0,0);
    return;
  }
  if(action==='candidate'){
    const preserveListScroll=page==='memory'&&memoryTab==='candidates';
    if(!preserveListScroll) candidateQuery='';
    candidateIndex=Number(button.dataset.index);memoryTab='candidates';page='memory';render({preserveListScroll});
    if(!preserveListScroll) window.scrollTo(0,0);
  }
});
document.getElementById('run-select').addEventListener('change', event => refresh(event.target.value));
document.addEventListener('change', event => {
  if (!event.target.matches('[data-experience-sort]') || !experienceSortOptions.some(([key])=>key===event.target.value)) return;
  expSort=event.target.value;
  render({preserveListScroll:true});
  document.querySelector('[data-experience-sort]')?.focus();
});
document.addEventListener('input', event=>{
  if(!event.target.dataset.search) return;
  const focus=event.target.dataset.search;
  if(focus==='sample') query=event.target.value;
  else if(focus==='candidate') candidateQuery=event.target.value;
  else expQuery=event.target.value;
  const position=event.target.selectionStart;
  render();
  const replacement=document.querySelector(`[data-search="${focus}"]`);
  if(replacement){replacement.focus();if(position!==null) replacement.setSelectionRange(position,position);}
});
document.addEventListener('change', event => {
  if (!event.target.matches('[data-category-filter]')) return;
  if (!['all','ungrouped',...(data?.categories||[]).map(c=>c.category_id)].includes(event.target.value)) return;
  categoryFilter=event.target.value;
  render();
  document.querySelector('[data-category-filter]')?.focus();
});

refresh();
