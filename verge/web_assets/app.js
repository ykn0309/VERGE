/* VERGE run dashboard — no third-party runtime; every artifact string is escaped. */
const app = document.getElementById('app');
let data = null, page = 'overview', sampleIndex = 0, expIndex = 0, query = '', splitFilter = 'train', resultFilter = 'all', expQuery = '';
let currentRun = null, requestSequence = 0;
const translations = new Map(), translatedFields = new Set();
const roleIoCache = new Map(), roleIoPending = new Map();
let sampleTab = 'pipeline';
const roleTabs = [
  ['pipeline','流水线'],['selector','经验筛选'],['initial_generator','初始生成'],
  ['judge','Judge'],['critic','Critic · 分析与操作'],['revision_generator','修订器（仅旧运行）'],
  ['reflector','候选经验生成'],['revision_credit_assigner','信用绑定'],
  ['admission_checker','经验准入'],['fusion_judge','经验融合'],
];
const translationKey = (run, sampleId, field) => JSON.stringify([run, sampleId, field]);
const h = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n = (value, digits = 3) => typeof value === 'number' ? value.toFixed(digits) : '—';
const pct = value => typeof value === 'number' ? (value * 100).toFixed(1) + '%' : '—';
const short = (s, max = 48) => String(s || '—').length > max ? String(s).slice(0, max) + '…' : String(s || '—');
const stateName = {initial_success:'初始成功',revision_success_1:'修订 1 成功',revision_success_2:'修订 2 成功',final_failure:'最终失败',initial_failure:'测试未通过',interrupted:'未完成轨迹'};
const statusClass = state => state === 'initial_success' || state?.startsWith('revision_success') ? 'ok' : state === 'interrupted' ? 'bad' : 'warn';
const badge = (label, type = 'neutral') => `<span class="badge ${type}"><i class="dot"></i>${h(label)}</span>`;
const warnBlock = () => data.warnings.length ? `<div class="warning">⚠ ${data.warnings.map(h).join('<br>')}</div>` : '';
const lineage = () => data.resumed_from ? `<div class="callout">此运行从 ${h(data.resumed_from)} 的已验证轨迹前缀恢复；当前统计已包含复制的样本，不应与父运行相加。</div>` : '';
const heading = (overline, title, description) => `<div class="page-heading"><div><span class="eyebrow">${h(overline)}</span><h1>${h(title)}</h1><p class="subheading">${h(description)}</p></div><span class="run-chip" title="${h(data.run_dir)}">${h(data.run_name)}</span></div>`;
const sampleButton = (id, text = id) => { const idx = data.samples.findIndex(s => s.sample_id === id); return idx < 0 ? h(text) : `<button class="event-link" data-action="sample" data-index="${idx}">${h(text)}</button>`; };
// The persisted experience model has no name field: use a readable, derived title.
const experienceName = e => short(String(e.content || e.trigger || '未命名经验').replace(/\s+/g,' ').trim().split(/(?<=[.!?。！？])\s+/)[0], 68);
const expButton = id => {
  const idx = data.experiences.findIndex(e => e.experience_id === id);
  return idx < 0 ? '<span class="muted">历史经验（当前快照无详情）</span>' :
    `<button class="experience-link" data-action="experience" data-index="${idx}" data-exp-preview="${idx}" aria-label="查看经验：${h(experienceName(data.experiences[idx]))}">${h(experienceName(data.experiences[idx]))}</button>`;
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
    if (!listing.runs.length) throw Error('没有可用的运行日志');
    const requested = nextName ?? currentRun ?? new URLSearchParams(location.search).get('run');
    const name = listing.runs.includes(requested) ? requested : listing.default;
    selector.value = name;
    const response = await fetch(`/api/run?name=${encodeURIComponent(name)}`, {cache:'no-store'});
    const payload = await response.json();
    if (sequence !== requestSequence) return;
    if (!response.ok) throw Error(payload.error || '无法读取运行结果');
    if (currentRun !== name) {
      sampleIndex = 0; expIndex = 0; query = ''; expQuery = ''; splitFilter = 'train'; resultFilter = 'all';
    }
    currentRun = name;
    translations.clear(); translatedFields.clear(); roleIoCache.clear();
    data = payload;
    if (!data.samples.some(s=>s.split===splitFilter)) {
      splitFilter = data.samples.some(s=>s.split==='train') ? 'train' : data.samples.some(s=>s.split==='test') ? 'test' : 'unknown';
      resultFilter = 'all';
    }
    history.replaceState(null, '', `?run=${encodeURIComponent(name)}`);
    sampleIndex = Math.min(sampleIndex, Math.max(0, data.samples.length - 1));
    expIndex = Math.min(expIndex, Math.max(0, data.experiences.length - 1));
    document.getElementById('nav-sample-count').textContent = data.samples.length;
    document.getElementById('nav-exp-count').textContent = data.experiences.length;
    render();
  } catch (error) {
    if (sequence === requestSequence) app.innerHTML = `<div class="warning">读取失败：${h(error.message)}。可尝试切换到其他运行日志。</div>`;
  } finally { if (sequence === requestSequence) selector.disabled = false; }
}
function render() {
  if (!data) return;
  hideExperiencePreview();
  document.querySelectorAll('.nav').forEach(button => button.classList.toggle('active', button.dataset.page === page));
  document.getElementById('crumb').textContent = {overview:'运行概览',samples:'样本流水线',memory:'经验池'}[page];
  app.innerHTML = page === 'overview' ? overview() : page === 'samples' ? samplesPage() : memoryPage();
}
function overview() {
  const samples = data.samples, exps = data.experiences, count = key => data.counts[key] || 0;
  const trained = samples.filter(s => s.split === 'train').length;
  const tested = samples.filter(s => s.split === 'test').length;
  const success = count('initial_success') + count('revision_success_1') + count('revision_success_2');
  const candidates = samples.flatMap(s => s.reflected_candidates || []);
  const added = candidates.filter(c => c.fusion === 'ADD').length;
  const merged = candidates.filter(c => c.fusion === 'MERGE').length;
  const max = Math.max(1, ...Object.values(data.counts));
  return heading('EXPERIMENT / OVERVIEW', '运行概览', '从一次运行到每个样本，再到经验池的累积变化。') + warnBlock() + lineage() +
    `<div class="metrics">
      ${metric('已记录样本',samples.length,`训练 ${trained} · 测试 ${tested}${trained + tested < samples.length ? ' · 未标注 ' + (samples.length - trained - tested) : ''}`)}
      ${metric('正向判别',success,`初始 ${count('initial_success')} · 修订 ${count('revision_success_1') + count('revision_success_2')}`)}
      ${metric('候选经验',candidates.length,`新增 ${added} · 合并 ${merged} · 未入池 ${candidates.length-added-merged}`)}
      ${metric('当前经验',exps.length,'冻结测试阶段不更新经验池')}
      ${metric('证据事件',exps.reduce((sum,e)=>sum+e.evidence_history.length,0),'revision + downstream credit')}
    </div><section class="panel overview-states"><div class="panel-title"><h2>结果分布</h2><small>SAMPLE STATES</small></div>${Object.entries(data.counts).map(([state,value])=>`<div class="bar-row ${state.includes('failure')?'fail':state==='initial_failure'?'test':''}"><span>${h(stateName[state]||state)}</span><div class="bar-track"><svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true"><rect class="bar-fill" width="${Math.round(100*value/max)}" height="8"/></svg></div><b>${value}</b></div>`).join('') || '<p class="muted">暂无样本轨迹</p>'}</section>`;
}
function metric(label,value,foot) {return `<div class="metric"><div class="metric-label">${h(label)}</div><div class="metric-value">${value}</div><div class="metric-foot">${h(foot)}</div></div>`;}
function samplesPage() {
  const matchesSplit = s => splitFilter === 'unknown' ? s.split !== 'train' && s.split !== 'test' : s.split === splitFilter;
  const group = data.samples.map((s,i)=>({s,i})).filter(({s})=>matchesSplit(s));
  const matchesResult = s => resultFilter === 'all' || (resultFilter === 'revision' ? s.state.startsWith('revision_success') : resultFilter === 'failure' ? ['final_failure','initial_failure','interrupted'].includes(s.state) : s.state === resultFilter);
  const visible = group.filter(({s})=>matchesResult(s) && (!query || `${s.sample_id} ${s.task||''} ${s.state}`.toLowerCase().includes(query.toLowerCase())));
  // Never show a test detail beside a training list (or a filtered-out detail).
  const current = visible.find(({i})=>i===sampleIndex) || visible[0];
  const selected = current?.s;
  const splits = [['train','训练',data.samples.filter(s=>s.split==='train').length],['test','测试',data.samples.filter(s=>s.split==='test').length]];
  const unknown = data.samples.length - splits[0][2] - splits[1][2];
  if (unknown) splits.push(['unknown','未标注',unknown]);
  const results = splitFilter === 'train' ? [['all','全部'],['initial_success','初始成功'],['revision','修订成功'],['failure','失败 / 中断']] : splitFilter === 'test' ? [['all','全部'],['initial_success','测试通过'],['failure','测试未通过 / 中断']] : [['all','全部'],['failure','失败 / 中断']];
  return warnBlock() +
    `<div class="sample-split-tabs" role="tablist" aria-label="样本阶段">${splits.map(([key,label,count])=>`<button role="tab" aria-selected="${splitFilter===key}" class="split-tab ${splitFilter===key?'active':''}" data-action="split-filter" data-split="${key}">${label}<span class="mono">${count}</span></button>`).join('')}</div>
    <div class="workbench"><section class="panel list-panel"><div class="list-head"><h2>${splitFilter==='train'?'训练':splitFilter==='test'?'测试':'未标注'}样本 <span class="muted mono">${visible.length} / ${group.length}</span></h2><input class="search" data-search="sample" placeholder="搜索 ID 或任务内容…" value="${h(query)}" aria-label="搜索样本"><div class="filters">${results.map(([key,label])=>`<button class="filter ${resultFilter===key?'active':''}" data-action="result-filter" data-filter="${key}">${label}</button>`).join('')}</div></div>
    <div class="list-scroll">${visible.map(({s,i})=>`<button class="list-item ${i===current?.i?'active':''}" data-action="sample" data-index="${i}"><span class="row-top"><strong title="${h(s.sample_id)}">${h(short(s.sample_id,36))}</strong>${badge(s.split==='test'&&s.state==='initial_success'?'测试通过':stateName[s.state]||s.state,statusClass(s.state))}</span><p>${h(short(s.task||s.split||'未提供任务文本',70))}</p></button>`).join('')||'<div class="empty">没有匹配的样本</div>'}</div></section>
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
    if (sequence === requestSequence && page === 'samples' && sampleTab !== 'pipeline') render();
  }
}
const pretty = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? '—';
function roleDetail(s, role) {
  const key = roleIoKey(currentRun, s.sample_id);
  if (!roleIoCache.has(key)) fetchRoleIo(s);
  const log = roleIoCache.get(key);
  const label = roleTabs.find(([id])=>id===role)?.[1] || role;
  const head = `<div class="detail-header"><div><span class="section-kicker">ROLE I/O / ${h(role.toUpperCase())}</span><h2>${h(label)}</h2><p>样本 ${h(s.sample_id)} · 实际调用的原始输入与输出，按记录顺序展示</p></div></div>`;
  if (!log) return head + '<p class="empty">正在读取角色调用记录…</p>';
  if (log.error) return head + `<div class="warning">${h(log.error)}</div>`;
  if (!log.available) return head + '<div class="role-empty">该运行未保存角色原始输入输出。旧轨迹只能展示流程结果，不能可靠还原当时的提示词与模型回复；重新运行后会生成 role_io.jsonl。</div>';
  const entries = log.entries.filter(item => item.role === role);
  const warnings = log.warnings?.length ? `<div class="warning">${log.warnings.map(h).join('<br>')}</div>` : '';
  if (!entries.length) return head + warnings + '<div class="role-empty">这个样本未调用该角色。可能是流程在此前结束，或来自未记录原始 I/O 的续跑前缀。</div>';
  return head + warnings + `<div class="role-call-count">${entries.length} 次调用记录 · 不同修订 / 候选 / 格式重试均独立保留。输入为角色实际看到的投影，不包含其他角色的私有上下文。</div>` +
    entries.map((entry,index)=>`<section class="role-call"><div class="role-call-head"><strong>调用 ${index+1}${entry.attempt>1?` · HTTP 尝试 ${h(entry.attempt)}`:''}</strong>${badge(entry.error ? entry.error : entry.status ? `HTTP ${entry.status}` : entry.backend==='fake'?'Fake':'成功', entry.error?'bad':'ok')}</div>
    <div class="io-grid"><div class="io-side"><h3>输入 <span>INPUT</span></h3>${Array.isArray(entry.messages)?entry.messages.map(m=>`<div class="io-message"><div class="io-label">${h(m.role||'消息')} · 实际发送</div><pre>${h(pretty(m.content))}</pre></div>`).join(''):`<div class="io-message"><div class="io-label">${entry.backend==='fake'?'Fake 角色输入':'角色参数'}</div><pre>${h(pretty(entry.input))}</pre></div>`}</div>
    <div class="io-side"><h3>输出 <span>OUTPUT</span></h3>${entry.raw_output!=null?`<div class="io-message"><div class="io-label">模型原文</div><pre>${h(pretty(entry.raw_output))}</pre></div>`:entry.output!=null?`<div class="io-message"><div class="io-label">${entry.backend==='fake'?'Fake 返回':'结构化返回'}</div><pre>${h(pretty(entry.output))}</pre></div>`:'<div class="io-message"><div class="io-label">无输出</div><p>该尝试在获得有效模型文本前失败。</p></div>'}${entry.raw_output!=null&&entry.output!=null?`<details><summary>解析后的结构化输出</summary><pre>${h(pretty(entry.output))}</pre></details>`:''}</div></div></section>`).join('');
}
function pipelineDetail(s) {
  const initial = s.initial_judgment, attempts = s.revision_attempts || [], candidates = s.reflected_candidates || [], invalid = s.invalid_revision_attempts || [];
  return `<div class="detail-header"><div><span class="section-kicker">SAMPLE TRACE / ${h(s.split?.toUpperCase()||'UNKNOWN')}</span><h2>${h(s.sample_id)}</h2>${s.metadata?.subset?`<p>子集 ${h(s.metadata.subset)}</p>`:''}</div>${badge(stateName[s.state]||s.state,statusClass(s.state))}</div>
  ${['task','chosen','rejected'].some(field => typeof s[field] === 'string') ? `<div class="detail-section"><div class="section-kicker">01 / INPUT</div><h3>任务与偏好对</h3>${sampleInput(s,'task','TASK')}${sampleInput(s,'chosen','CHOSEN')}${sampleInput(s,'rejected','REJECTED')}</div>`:''}
  <div class="detail-section"><div class="section-kicker">02 / EXPERIENCE RETRIEVAL</div><h3>召回与筛选</h3><div class="mini-stats"><span class="mini-stat">召回 <b>${s.retrieved_experience_ids?.length ?? 0}</b></span><span class="mini-stat">选中 <b>${s.selected_experience_ids?.length ?? 0}</b></span></div>${(s.retrieved_experience_ids||[]).length?`<div class="retrieval-list" aria-label="召回经验，可滚动">${s.retrieved_experience_ids.map(id=>`<div class="retrieval-item">${expButton(id)} ${s.selected_experience_ids?.includes(id)?badge('已选中','ok'):badge('未选中')}</div>`).join('')}</div><p class="legend">悬浮或聚焦经验名称可预览；点击查看完整演化。列表可独立滚动。</p>`:'<p class="muted body-text">经验池为空，或当前任务没有召回经验。</p>'}</div>
  ${s.initial_rubric ? `<div class="detail-section"><div class="section-kicker">03 / INITIAL RUBRIC & JUDGE</div><h3>初始标准 R0 ${badge(initial?.success?'判别成功':'判别失败',initial?.success?'ok':'warn')}</h3>${judgmentStats(initial)}${criteria(s.initial_rubric,initial)}</div>` : '<div class="detail-section"><div class="section-kicker">03 / INITIAL RUBRIC</div><p class="body-text muted">没有完整 R0 记录。模型请求可能已经发生，但该样本尚未写入轨迹。</p></div>'}
  ${(attempts.length||invalid.length) ? `<div class="detail-section"><div class="section-kicker">04 / REVISION BRANCH</div><h3>同基线修订尝试</h3><p class="legend">每次修订都从 R0 出发，而不是以前一次失败修订为基线。</p>${[...attempts.map(a=>({number:a.attempt_number,attempt:a})),...invalid.map(a=>({number:a.attempt_number,invalid:a}))].sort((a,b)=>a.number-b.number).map(({number,attempt,invalid:bad})=>bad?`<div class="attempt"><div class="attempt-top"><strong>Attempt ${number} · 结构校验未通过</strong>${badge('未进入 Judge','bad')}</div><p class="body-text">${bad.analysis!=null?`Critic 私有分析：${h(bad.analysis)}`:`旧版 Critic 反馈：${h(bad.feedback||'未记录')}`}</p><div class="callout warn">${bad.validation_failures.map(h).join('<br>')}</div></div>`:`<div class="attempt"><div class="attempt-top"><strong>Attempt ${number} · ${h(attempt.rubric.rubric_id)}</strong>${badge(attempt.judgment.success?'成功':'失败',attempt.judgment.success?'ok':'warn')}</div><p class="body-text">${attempt.analysis!=null?`Critic 私有分析：${h(attempt.analysis)}`:`旧版 Critic 反馈：${h(attempt.feedback||'未记录')}`}</p>${attempt.validation_failures?.length?`<div class="callout warn">修复过程：${attempt.validation_failures.map(h).join('；')}</div>`:''}${attempt.operations.map(op=>`<div class="operation"><strong>${h(op.operation_type)} · ${h(op.operation_id)}</strong> ${op.operation_id in (s.operation_contributions||{})?`<span class="mono">δ ${n(s.operation_contributions[op.operation_id])}</span>`:''}<p>${op.before_criterion?`原：${h(op.before_criterion.question)}<br>`:''}${op.new_or_updated_criterion?`新：${h(op.new_or_updated_criterion.question)}`:'删除标准 '+h(op.target_criterion_id)}</p></div>`).join('')}${judgmentStats(attempt.judgment)}<details><summary class="inline-link">查看修订后完整 Rubric</summary>${criteria(attempt.rubric,attempt.judgment)}</details></div>`).join('')}</div>` : ''}
  ${candidates.length ? `<div class="detail-section"><div class="section-kicker">05 / REFLECTION → CREDIT → ADMISSION → FUSION</div><h3>候选经验与入池决策</h3>${candidates.map(c=>`<div class="candidate"><div class="candidate-head"><strong class="mono">${h(c.candidate_id)}</strong>${badge(c.credit===null?'CREDIT_FAIL':c.admission==='FAIL'?'准入拒绝':c.fusion?c.fusion:'待处理',c.fusion?'ok':c.admission==='FAIL'?'bad':'warn')}</div><p>${h(c.content)}</p><small>触发条件：${h(c.trigger)}</small>${c.admission ? `<p class="admission-reason"><strong>准入理由：</strong>${h(c.admission_reason || '旧轨迹未记录判断理由')}</p>` : ''}<div class="mini-stats"><span class="mini-stat">绑定操作 <b>${h((c.linked_operation_ids||[]).join(', ')||'无')}</b></span><span class="mini-stat">暂定信用 <b>${n(c.credit)}</b></span></div>${c.experience_id?`<span class="sources">入池经验：${expButton(c.experience_id)}</span>`:''}</div>`).join('')}</div>` : ''}
  `;
}
function judgmentStats(j) {return j ? `<div class="mini-stats"><span class="mini-stat">MARGIN <b>${n(j.margin)}</b></span><span class="mini-stat">CHOSEN <b>${n(j.chosen_score)}</b></span><span class="mini-stat">REJECTED <b>${n(j.rejected_score)}</b></span></div>` : '';}
function criteria(rubric, judgment) {return `<div class="rubric-list" role="region" tabindex="0" aria-label="Rubric 标准，可滚动">${rubric.criteria.map(c=>{const verdict=judgment?.criterion_judgments?.find(j=>j.criterion_id===c.criterion_id);return `<div class="criterion"><div class="criterion-head"><strong>${h(c.criterion_id)}</strong><span class="meta">重要性 ${c.importance_score} · 权重 ${pct(judgment?.normalized_weights?.[c.criterion_id])}</span></div><div class="criterion-body"><p>${h(c.question)}</p><div class="judgment-pills"><span class="truth ${verdict?.chosen?'yes':'no'}">Chosen ${verdict?.chosen?'✓':'✕'}</span><span class="truth ${verdict?.rejected?'yes':'no'}">Rejected ${verdict?.rejected?'✓':'✕'}</span></div>${c.source_experiences?.length?`<div class="sources">来源经验：${c.source_experiences.map(expButton).join(' · ')}</div>`:''}</div></div>`;}).join('')}</div>`;}
function memoryPage() {
  const visible=data.experiences.map((e,i)=>({e,i})).filter(({e})=>!expQuery||`${e.experience_id} ${e.content} ${e.trigger}`.toLowerCase().includes(expQuery.toLowerCase()));
  return warnBlock() +
    `<div class="workbench"><section class="panel list-panel memory-list"><div class="list-head"><h2>经验目录 <span class="muted mono">${visible.length} / ${data.experiences.length}</span></h2><input class="search" data-search="experience" placeholder="搜索经验内容或 ID…" value="${h(expQuery)}" aria-label="搜索经验"></div><div class="list-scroll">${visible.map(({e,i})=>`<button class="list-item ${i===expIndex?'active':''}" data-action="experience" data-index="${i}"><span class="row-top"><strong title="${h(experienceName(e))}">${h(experienceName(e))}</strong>${badge(pct(e.confidence),'ok')}</span><p>${h(short(e.content,74))}</p></button>`).join('')||'<div class="empty">没有匹配的经验</div>'}</div></section><section class="panel detail">${data.experiences[expIndex]?experienceDetail(data.experiences[expIndex]):'<div class="empty">经验池为空；成功修订并通过准入后才会出现经验。</div>'}</section></div>`;
}
function experienceDetail(e) {
  const t=e.evolution;
  const sourceCount=e.source_trajectories.length;
  return `<div class="detail-header"><div><span class="section-kicker">CANONICAL EXPERIENCE / ${String(expIndex+1).padStart(2,'0')}</span><h2>${h(experienceName(e))}</h2><p>名称由正文摘要生成 · 合并时保留原始内容和触发条件</p></div>${badge(pct(e.confidence),'ok')}</div>
    <div class="detail-section"><div class="section-kicker">01 / PRINCIPLE</div><h3>经验内容</h3><p class="exp-intro">${h(e.content)}</p><div class="exp-trigger"><strong>TRIGGER · 适用条件</strong><br>${h(e.trigger)}</div></div>
    <div class="detail-section"><div class="section-kicker">02 / RELIABILITY</div><h3>可靠性与证据</h3><div class="mini-stats"><span class="mini-stat">α <b>${n(e.alpha)}</b></span><span class="mini-stat">β <b>${n(e.beta)}</b></span><span class="mini-stat">CONFIDENCE <b>${pct(e.confidence)}</b></span><span class="mini-stat">证据量 <b>${n(e.alpha+e.beta-2)}</b></span></div>${lineChart(t)}<p class="legend">从 Beta(1,1) 先验开始，正信用增加 α、负信用增加 β。置信度是 α/(α+β)，并非校准后的成功概率。</p></div>
    <div class="detail-section"><div class="section-kicker">03 / ORIGIN & EVOLUTION</div><h3>演化时间线 · ${t.length-1} 个事件</h3><p class="legend">按记录顺序重建，非真实时间戳。${sourceCount} 条来源轨迹 · ${e.evidence_history.length} 条有效证据；无法唯一匹配的候选融合单独标记，不重复计入信用。无完整轨迹的证据可能来自中断样本。</p><div class="timeline">${t.map((ev,i)=>`<div class="event ${i===0?'prior':''}"><div class="event-top"><strong>${i===0?'初始先验':ev.kind==='fusion'?ev.fusion+' · 候选融合（无独立证据）':ev.kind==='revision_credit'?`${ev.fusion||'修订'} · 修订信用`:'下游信用'}</strong><span class="mono">${pct(ev.confidence)}</span></div>${i ? `<p>${sampleButton(ev.sample_id)} ${ev.has_trajectory?'':'· 无完整轨迹'}<br>credit ${n(ev.credit)} × weight ${n(ev.evidence_weight)} · α ${n(ev.alpha)} / β ${n(ev.beta)}${ev.candidate_id?`<br>候选 ${h(ev.candidate_id)}`:''}${ev.fusion==='MERGE'&&ev.candidate_content&&ev.candidate_content!==e.content?`<br>合并候选：${h(ev.candidate_content)}`:''}</p>`:`<p>${ev.sample_id?`${ev.origin_kind==='add'?'入池样本':'首条来源样本'}：${sampleButton(ev.sample_id)}${ev.has_trajectory?'':' · 无完整轨迹'}<br>`:'来源样本未记录<br>'}α 1.000 / β 1.000 · 尚无样本信用</p>`}</div>`).join('')}</div></div>
    <div class="detail-section"><div class="section-kicker">04 / SOURCE TRAJECTORIES</div><h3>产生或再次发现它的样本</h3><p class="body-text">${e.source_trajectories.map(id=>sampleButton(id)).join(' · ')||'没有记录来源样本'}</p></div>`;
}
function lineChart(events) {
  const w=600, height=150, left=34, right=14, top=13, bottom=24;
  const x=i=>left+(w-left-right)*(events.length===1?.5:i/(events.length-1));
  const y=value=>top+(height-top-bottom)*(1-value);
  const points=events.map((e,i)=>`${x(i)},${y(e.confidence)}`).join(' ');
  return `<svg class="chart" viewBox="0 0 ${w} ${height}" role="img" aria-label="经验置信度变化折线图">${[0,.25,.5,.75,1].map(v=>`<line class="grid" x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}"/><text x="0" y="${y(v)+3}">${Math.round(v*100)}%</text>`).join('')}<polyline class="line" points="${points}"/>${events.map((e,i)=>`<circle cx="${x(i)}" cy="${y(e.confidence)}" r="3.2"><title>${h(i===0?'先验':e.sample_id)}: ${pct(e.confidence)}</title></circle>`).join('')}<text x="${left}" y="${height-3}">先验</text><text x="${w-right-37}" y="${height-3}">当前</text></svg>`;
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
  if(action==='translate'){translateSampleField(button);return;}
  if(action==='sample-tab' && roleTabs.some(([key])=>key===button.dataset.tab)){
    sampleTab=button.dataset.tab;render();return;
  }
  if(action==='refresh'){refresh(currentRun);return;}
  if(action==='split-filter'){splitFilter=button.dataset.split;resultFilter='all';query='';sampleIndex=data.samples.findIndex(s=>splitFilter==='unknown'?s.split!=='train'&&s.split!=='test':s.split===splitFilter);render();return;}
  if(action==='result-filter'){resultFilter=button.dataset.filter;render();return;}
  if(action==='sample'){sampleIndex=Number(button.dataset.index);const sample=data.samples[sampleIndex];const nextSplit=sample?.split==='train'||sample?.split==='test'?sample.split:'unknown';if(page!=='samples'||splitFilter!==nextSplit){resultFilter='all';query='';}splitFilter=nextSplit;page='samples';render();window.scrollTo(0,0);return;}
  if(action==='experience'){expIndex=Number(button.dataset.index);page='memory';render();window.scrollTo(0,0);}
});
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
function setSidebarCollapsed(collapsed) {
  if (collapsed && sidebar.contains(document.activeElement)) sidebarToggle.focus();
  document.querySelector('.shell').classList.toggle('sidebar-collapsed', collapsed);
  sidebar.inert = collapsed;
  sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
  sidebarToggle.textContent = collapsed ? '显示侧栏' : '收起侧栏';
  try { localStorage.setItem('verge.sidebar.collapsed', collapsed ? '1' : '0'); } catch (_) { /* private browsing */ }
}
sidebarToggle.addEventListener('click', () => setSidebarCollapsed(!sidebar.inert));
try { if (localStorage.getItem('verge.sidebar.collapsed') === '1') setSidebarCollapsed(true); } catch (_) { /* storage unavailable */ }
document.getElementById('run-select').addEventListener('change', event => refresh(event.target.value));
document.addEventListener('input', event=>{
  if(!event.target.dataset.search) return;
  const focus=event.target.dataset.search;
  if(focus==='sample') query=event.target.value; else expQuery=event.target.value;
  const position=event.target.selectionStart;
  render();
  const replacement=document.querySelector(`[data-search="${focus}"]`);
  if(replacement){replacement.focus();if(position!==null) replacement.setSelectionRange(position,position);}
});
refresh();
