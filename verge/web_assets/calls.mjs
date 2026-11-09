// Preserve log order, including failed transport attempts and format retries.
export function callRecords(entries) {
  const validations = new Map();
  const timings = new Map();
  for (const entry of entries) {
    if (!entry.call_id) continue;
    if (entry.event_type === "contract_validation")
      validations.set(entry.call_id, entry);
    if (entry.event_type === "request_timing") {
      const rows = timings.get(entry.call_id) || [];
      rows.push(entry);
      timings.set(entry.call_id, rows);
    }
  }
  return entries
    .filter(
      (entry) =>
        Array.isArray(entry.messages) ||
        (entry.role && entry.input != null && !entry.event_type),
    )
    .map((entry) => {
      const validation = validations.get(entry.call_id);
      const timing = timings.get(entry.call_id)?.[(entry.attempt || 1) - 1];
      const rawAvailable = typeof entry.raw_output === "string";
      const error =
        entry.error ||
        (validation?.validation_status === "invalid" ? validation.error : null);
      return { entry, validation, timing, rawAvailable, error };
    });
}
