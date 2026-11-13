export function learningProtocol(data) { return data.schema === "verge-v3.1"; }
export function trialLabel(path, allIds) {
  const id = path?.provided_experience_ids?.[0];
  return id ? `经验${allIds.indexOf(id) + 1}` : "无经验";
}
export function admissionPath(round, attempt) { return attempt.path; }
export function baselineOf(record) { return record.paths?.find(p => !p.provided_experience_ids.length); }
export function resultOf(path) {
  if (!path) return "未运行";
  if (path.error) return "技术失败";
  if (!path.judgment) return "未形成判定";
  return path.judgment.success ? "判对" : "判错";
}
export function initialStatus(record) {
  if (record.error) return "技术失败";
  const paths = record.paths || [];
  const successes = paths.filter(p => p.judgment?.success && !p.error).length;
  if (!paths.length || paths.some(p => p.error || !p.judgment)) return `${successes}路成功·含未知`;
  return `${successes}/${paths.length}路成功`;
}
export function providedIds(path) { return path?.provided_experience_ids || []; }
export function feedbackOutcome(record) {
  if (record.error) return "无有效对照";
  if (record.experience_correct && !record.baseline_correct) return "改善";
  if (!record.experience_correct && record.baseline_correct) return "退化";
  return record.experience_correct ? "均对" : "均错";
}
export function summarize(data) {
  const samples = data.samples || [];
  const attempts = samples.flatMap(s => (s.critic_rounds || []).flatMap(r => r.attempts));
  const pairs = data.feedback || [], paired = pairs.filter(f => !f.error);
  const outcomes = Object.fromEntries(["改善", "退化", "均对", "均错"].map(k => [k, 0]));
  paired.forEach(f => outcomes[feedbackOutcome(f)]++);
  return {
    completed: samples.length,
    correct: samples.filter(s => baselineOf(s)?.judgment?.success && !baselineOf(s)?.error).length,
    errors: samples.filter(s => s.error || s.paths.some(p => p.error)).length,
    selected: samples.filter(s => Object.keys(s.selection_reasons).length).length,
    critic: samples.filter(s => s.critic_rounds.length).length,
    repaired: samples.filter(s => s.critic_rounds.some(r => r.attempts.some(a => a.admitted))).length,
    attempts: attempts.length, admitted: attempts.filter(a => a.admitted).length,
    validations: attempts.length, candidateErrors: attempts.filter(a => a.path.error).length,
    paired: paired.length, invalidPairs: pairs.filter(f => f.error).length,
    outcomes, gain: outcomes["改善"] - outcomes["退化"],
  };
}
export function sampleOutcome(record, data) {
  const outcomes = [...new Set(data.feedback.filter(f => f.record_id.startsWith(record.sample.sample_id + ":")).map(feedbackOutcome))];
  return outcomes.join("／") || "未选经验";
}
export function matchesFilter(record, filter, data) {
  if (!filter || filter === "all") return true;
  if (filter === "error") return Boolean(record.error || record.paths.some(p => p.error));
  if (filter === "wrong") return record.paths.length > 0 && record.paths.every(p => resultOf(p) === "判错");
  if (filter === "selected") return Object.keys(record.selection_reasons).length > 0;
  if (filter === "critic") return record.critic_rounds.length > 0;
  return sampleOutcome(record, data).split("／").includes(filter);
}
export function sampleListSummary(record, data) {
  const rounds = record.critic_rounds || [], attempts = rounds.flatMap(r => r.attempts);
  const ids = Object.keys(record.selection_reasons);
  return {
    initial: initialStatus(record), baseline: resultOf(baselineOf(record)),
    selected: ids.length, comparison: sampleOutcome(record, data),
    initialTrials: record.paths.map(p => ({label: trialLabel(p, ids), result: resultOf(p)})),
    candidateResults: rounds.flatMap(r => r.attempts.map(a => ({round: r.round_number,
      candidate: 1, result: resultOf(a.path), admitted: a.admitted}))),
    roundErrors: rounds.filter(r => r.error).map(r => ({round: r.round_number, error: r.error})),
    rounds: rounds.length, candidates: attempts.length, admitted: attempts.filter(a => a.admitted).length,
    criticErrors: rounds.filter(r => r.error).length + attempts.filter(a => a.path.error).length,
  };
}

export function criticCandidates(record) {
  return (record.critic_rounds || []).flatMap((round, roundIndex) =>
    (round.attempts || []).map((attempt, index) => ({
      round: round.round_number ?? roundIndex + 1,
      number: index + 1,
      attempt,
    })),
  );
}

export function resolveCriticRound(record, requested, candidateId) {
  const rounds = record.critic_rounds || [];
  const explicit = rounds.find((r) => r.round_number === requested);
  if (explicit) return explicit.round_number;
  return (
    criticCandidates(record).find(
      (c) => c.attempt.experience.experience_id === candidateId,
    )?.round ??
    rounds[0]?.round_number ??
    null
  );
}

export function experienceListSummary(experience, data) {
  const rows = (data.feedback || []).filter(
    (f) =>
      f.phase === "downstream" &&
      f.experience_id === experience.experience_id,
  );
  const valid = new Map();
  const attemptedTasks = new Set();
  for (const row of rows) {
    const key = row.task_id || row.record_id;
    attemptedTasks.add(key);
    if (
      !row.error &&
      typeof row.baseline_correct === "boolean" &&
      typeof row.experience_correct === "boolean" &&
      !valid.has(key)
    )
      valid.set(key, row);
  }
  const outcomes = { improved: 0, regressed: 0, bothCorrect: 0, bothWrong: 0 };
  for (const row of valid.values()) {
    const key = {
      改善: "improved",
      退化: "regressed",
      均对: "bothCorrect",
      均错: "bothWrong",
    }[feedbackOutcome(row)];
    outcomes[key]++;
  }
  const sourceId = data.task_index?.[experience.source_task_id];
  const source = (data.samples || []).find(
    (r) => r.sample.sample_id === sourceId,
  );
  const candidate = criticCandidates(source || {}).find(
    (c) => c.attempt.experience.experience_id === experience.experience_id,
  );
  const attempt = candidate?.attempt;
  return {
    ...outcomes,
    providedTasks: attemptedTasks.size,
    validTasks: valid.size,
    technicalFailures: rows.filter((r) => r.error).length,
    meanReward: valid.size
      ? [...valid.values()].reduce(
          (sum, row) =>
            sum +
            (row.credits
              ? row.credits[experience.experience_id]
              : Number(row.experience_correct) - Number(row.baseline_correct)),
          0,
        ) / valid.size
      : null,
    sourceResult: attempt
      ? resultOf(attempt.path)
      : "记录缺失",

  };
}
