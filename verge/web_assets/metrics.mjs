export function subsetProtocol(data) {
  return data.schema === "verge-subset-learning-v2";
}

export function subsetLabel(path, allIds) {
  const ids = path?.provided_experience_ids || [];
  if (!ids.length) return "无经验";
  return ids.map((id) => `经验${allIds.indexOf(id) + 1}`).join("＋");
}

export function admissionPath(round, attempt) {
  const ids = attempt.admission_subset || [];
  return ids.length
    ? (round.subset_paths || []).find(
        (p) =>
          p.provided_experience_ids.length === ids.length &&
          ids.every((id) => p.provided_experience_ids.includes(id)),
      )
    : attempt.path;
}

export function usedPath(record) {
  return record.actual || record.baseline;
}

export function resultOf(path) {
  if (!path) return "未运行";
  if (path.error) return "技术失败";
  if (!path.judgment) return "未形成判定";
  return path.judgment.success ? "判对" : "判错";
}

export function citedIds(path) {
  return [
    ...new Set(
      (path?.rubric?.criteria || [])
        .filter((c) => c.weight > 0)
        .flatMap((c) => c.source_experiences || []),
    ),
  ];
}

export function feedbackOutcome(record) {
  if (record.error || record.phase !== "downstream") return "无有效对照";
  if (record.actual_correct && !record.baseline_correct) return "改善";
  if (!record.actual_correct && record.baseline_correct) return "退化";
  return record.actual_correct ? "均对" : "均错";
}

export function summarize(data, scope = "provided") {
  if (subsetProtocol(data)) scope = "provided";
  const samples = data.samples || [];
  const attempts = samples.flatMap((s) =>
    s.critic_rounds.flatMap((r) => r.attempts),
  );
  const validationPaths = samples.flatMap((s) =>
    (s.critic_rounds || []).flatMap((r) =>
      r.subset_paths?.length
        ? r.subset_paths.filter((p) => p.provided_experience_ids.length)
        : (r.attempts || []).map((a) => a.path),
    ),
  );
  const scoped = (data.feedback || []).filter(
    (f) => scope !== "cited" || f.experience_ids.length > 0,
  );
  const paired = scoped.filter((f) => !f.error && f.phase === "downstream");
  const outcomes = Object.fromEntries(
    ["改善", "退化", "均对", "均错"].map((k) => [k, 0]),
  );
  paired.forEach((f) => outcomes[feedbackOutcome(f)]++);
  const errors = samples.filter(
    (s) => s.error || !usedPath(s) || usedPath(s).error,
  );
  const correct = samples.filter(
    (s) => usedPath(s)?.judgment?.success && !s.error,
  );
  return {
    completed: samples.length,
    correct: correct.length,
    errors: errors.length,
    selected: samples.filter((s) => s.actual).length,
    cited: samples.filter((s) => citedIds(s.actual).length).length,
    critic: samples.filter((s) => s.critic_rounds.length).length,
    repaired: samples.filter((s) =>
      s.critic_rounds.some((r) => r.attempts.some((a) => a.admitted)),
    ).length,
    attempts: attempts.length,
    admitted: attempts.filter((a) => a.admitted).length,
    validations: validationPaths.length,
    candidateErrors: validationPaths.filter((p) => p?.error).length,
    paired: paired.length,
    invalidPairs: scoped.filter((f) => f.error).length,
    uncitedPairs: (data.feedback || []).filter(
      (f) => !f.error && f.phase === "downstream" && !f.experience_ids.length,
    ).length,
    outcomes,
    gain: outcomes["改善"] - outcomes["退化"],
  };
}

export function sampleOutcome(record, data) {
  const feedback = data.feedback.find(
    (f) => f.record_id === record.sample.sample_id,
  );
  return feedback
    ? feedbackOutcome(feedback)
    : record.actual
      ? "未引用经验"
      : "未选经验";
}

export function matchesFilter(record, filter, data, scope = "provided") {
  if (subsetProtocol(data)) scope = "provided";
  if (!filter || filter === "all") return true;
  if (filter === "error")
    return Boolean(record.error || usedPath(record)?.error);
  if (filter === "wrong") return resultOf(usedPath(record)) === "判错";
  if (filter === "selected") return Boolean(record.actual);
  if (filter === "critic") return record.critic_rounds.length > 0;
  if (
    scope === "cited" &&
    !data.feedback.find((f) => f.record_id === record.sample.sample_id)
      ?.experience_ids.length
  )
    return false;
  return sampleOutcome(record, data) === filter;
}

// Candidate admission is reported separately from the original path's result.
export function sampleListSummary(record, data) {
  const rounds = record.critic_rounds || [];
  const attempts = rounds.flatMap((r) => r.attempts || []);
  const feedback = (data.feedback || []).find(
    (f) => f.record_id === record.sample.sample_id,
  );
  return {
    initial: record.error ? "技术失败" : resultOf(usedPath(record)),
    baseline: resultOf(record.baseline),
    actual: record.actual ? resultOf(record.actual) : null,
    selected:
      record.error &&
      !record.baseline &&
      !record.actual &&
      !Object.keys(record.selection_reasons || {}).length
        ? null
        : (record.actual?.provided_experience_ids?.length ??
          Object.keys(record.selection_reasons || {}).length),
    cited: citedIds(record.actual).length,
    comparison: record.actual
      ? feedback &&
        typeof feedback.baseline_correct === "boolean" &&
        typeof feedback.actual_correct === "boolean"
        ? feedbackOutcome(feedback)
        : "无有效对照"
      : null,
    initialSubsets: (record.subset_paths || []).map((p) => ({
      label: subsetLabel(p, record.actual?.provided_experience_ids || []),
      result: resultOf(p),
    })),
    pairResults: rounds.flatMap((round) =>
      (round.subset_paths || [])
        .filter((p) => p.provided_experience_ids.length === 2)
        .map((p) => ({ round: round.round_number, result: resultOf(p) })),
    ),
    candidateResults: rounds.flatMap((round, roundIndex) =>
      (round.attempts || []).map((attempt, index) => ({
        round: round.round_number ?? roundIndex + 1,
        candidate: index + 1,
        result: resultOf(attempt.path),
        cited: citedIds(attempt.path).includes(
          attempt.experience?.experience_id,
        ),
        admitted: Boolean(attempt.admitted),
        ...(attempt.admission_subset?.length === 2 ? { joint: true } : {}),
      })),
    ),
    roundErrors: rounds.flatMap((round, index) =>
      round.error
        ? [{ round: round.round_number ?? index + 1, error: round.error }]
        : [],
    ),
    rounds: rounds.length,
    candidates: attempts.length,
    admitted: attempts.filter((a) => a.admitted).length,
    criticErrors:
      rounds.filter((r) => r.error).length +
      attempts.filter((a) => a.path?.error).length,
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
      f.experience_ids.includes(experience.experience_id),
  );
  const valid = new Map();
  const attemptedTasks = new Set();
  for (const row of rows) {
    const key = row.task_id || row.record_id;
    attemptedTasks.add(key);
    if (
      !row.error &&
      typeof row.baseline_correct === "boolean" &&
      typeof row.actual_correct === "boolean" &&
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
  const round = source?.critic_rounds.find(
    (r) => r.round_number === candidate?.round,
  );
  return {
    ...outcomes,
    citedTasks: attemptedTasks.size,
    validTasks: valid.size,
    technicalFailures: rows.filter((r) => r.error).length,
    meanReward: valid.size
      ? [...valid.values()].reduce(
          (sum, row) =>
            sum +
            (row.credits
              ? row.credits[experience.experience_id]
              : Number(row.actual_correct) - Number(row.baseline_correct)),
          0,
        ) / valid.size
      : null,
    sourceResult: attempt
      ? `${attempt.admission_subset?.length === 2 ? "组合" : ""}${resultOf(admissionPath(round, attempt))}`
      : "记录缺失",
    sourceCited:
      attempt && !subsetProtocol(data)
        ? citedIds(attempt.path).includes(experience.experience_id)
        : null,
  };
}
