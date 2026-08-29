/** Fail-closed decoders for the v3 read boundary. A malformed response is a contract state, never an empty list. */

export class ContractError extends Error {
  readonly name = "ContractError";

  constructor(readonly endpoint: string, readonly detail: string) {
    super(`Invalid response from ${endpoint}: ${detail}`);
  }
}

type JsonRecord = Record<string, unknown>;

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(endpoint, path, "must be an object");
  return value as JsonRecord;
}

function array(value: unknown, endpoint: string, path: string): unknown[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function optional<T>(row: JsonRecord, key: string, parser: (value: unknown, path: string) => T, endpoint: string, path: string): T | undefined {
  return key in row ? parser(row[key], `${path}.${key}`) : undefined;
}

function closed(row: JsonRecord, keys: readonly string[], endpoint: string, path: string): void {
  const allowed = new Set(keys);
  for (const key of Object.keys(row)) if (!allowed.has(key)) fail(endpoint, `${path}.${key}`, "is not part of the contract");
}

function stringValue(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nonEmptyString(value: unknown, endpoint: string, path: string): string {
  const parsed = stringValue(value, endpoint, path);
  if (!parsed.trim()) fail(endpoint, path, "must not be empty");
  return parsed;
}

function finiteNumber(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be a finite number");
  return value;
}

function nonNegativeNumber(value: unknown, endpoint: string, path: string): number {
  const parsed = finiteNumber(value, endpoint, path);
  if (parsed < 0) fail(endpoint, path, "must be non-negative");
  return parsed;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = nonNegativeNumber(value, endpoint, path);
  if (!Number.isSafeInteger(parsed)) fail(endpoint, path, "must be an integer");
  return parsed;
}

function positiveInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = nonNegativeInteger(value, endpoint, path);
  if (parsed < 1) fail(endpoint, path, "must be at least 1");
  return parsed;
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function nullableString(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : stringValue(value, endpoint, path);
}

function enumValue<T extends string>(value: unknown, values: readonly T[], endpoint: string, path: string): T {
  if (typeof value !== "string" || !values.includes(value as T)) fail(endpoint, path, `must be one of ${values.join(", ")}`);
  return value as T;
}

const SESSION_ID = /^session_[0-9a-f]{24}$/;
const CLIENT_KEY = /^client_[0-9a-f]{24}$/;

export interface SessionPreview {
  runId: string;
  promptSummary: string;
  responseSummary: string;
}

export interface SessionRecord {
  schemaVersion: "clozn.session.v1";
  id: string;
  createdTs: number;
  createdAt: string;
  clientKey?: string;
  title?: string;
  visibility: "visible" | "hidden";
  materializedFrom: "explicit" | "lazy_backfill";
  firstActivityTs?: number;
  lastActivityTs?: number;
  runCount?: number;
  turnCount?: number;
  preview?: SessionPreview;
}

function decodePreview(value: unknown, endpoint: string, path: string): SessionPreview {
  const row = object(value, endpoint, path);
  closed(row, ["run_id", "prompt_summary", "response_summary"], endpoint, path);
  return {
    runId: nonEmptyString(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`),
    promptSummary: stringValue(required(row, "prompt_summary", endpoint, path), endpoint, `${path}.prompt_summary`),
    responseSummary: stringValue(required(row, "response_summary", endpoint, path), endpoint, `${path}.response_summary`),
  };
}

export function decodeSession(value: unknown, endpoint = "/sessions", path = "$"): SessionRecord {
  const row = object(value, endpoint, path);
  closed(row, [
    "schema_version", "id", "created_ts", "created_at", "client_key", "title", "privacy",
    "materialized_from", "first_activity_ts", "last_activity_ts", "run_count", "turn_count", "preview",
  ], endpoint, path);
  const schemaVersion = enumValue(required(row, "schema_version", endpoint, path), ["clozn.session.v1"], endpoint, `${path}.schema_version`);
  const id = nonEmptyString(required(row, "id", endpoint, path), endpoint, `${path}.id`);
  if (!SESSION_ID.test(id)) fail(endpoint, `${path}.id`, "must be an opaque session id");
  const privacy = object(required(row, "privacy", endpoint, path), endpoint, `${path}.privacy`);
  closed(privacy, ["visibility"], endpoint, `${path}.privacy`);
  const title = optional(row, "title", (v, p) => {
    const parsed = nonEmptyString(v, endpoint, p);
    if (parsed.length > 200) fail(endpoint, p, "must be at most 200 characters");
    return parsed;
  }, endpoint, path);
  const clientKey = optional(row, "client_key", (v, p) => {
    const parsed = nonEmptyString(v, endpoint, p);
    if (!CLIENT_KEY.test(parsed)) fail(endpoint, p, "must be an opaque client key");
    return parsed;
  }, endpoint, path);
  const firstActivityTs = optional(row, "first_activity_ts", (v, p) => nonNegativeNumber(v, endpoint, p), endpoint, path);
  const lastActivityTs = optional(row, "last_activity_ts", (v, p) => nonNegativeNumber(v, endpoint, p), endpoint, path);
  return {
    schemaVersion,
    id,
    createdTs: nonNegativeNumber(required(row, "created_ts", endpoint, path), endpoint, `${path}.created_ts`),
    createdAt: nonEmptyString(required(row, "created_at", endpoint, path), endpoint, `${path}.created_at`),
    clientKey,
    title,
    visibility: enumValue(required(privacy, "visibility", endpoint, `${path}.privacy`), ["visible", "hidden"], endpoint, `${path}.privacy.visibility`),
    materializedFrom: enumValue(required(row, "materialized_from", endpoint, path), ["explicit", "lazy_backfill"], endpoint, `${path}.materialized_from`),
    firstActivityTs,
    lastActivityTs,
    runCount: optional(row, "run_count", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    turnCount: optional(row, "turn_count", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    preview: optional(row, "preview", (v, p) => decodePreview(v, endpoint, p), endpoint, path),
  };
}

export function decodeSessionListDocument(value: unknown, endpoint = "/sessions"): readonly SessionRecord[] {
  const row = object(value, endpoint, "$");
  closed(row, ["sessions"], endpoint, "$");
  return array(required(row, "sessions", endpoint, "$") , endpoint, "$.sessions")
    .map((entry, index) => decodeSession(entry, endpoint, `$.sessions[${index}]`));
}

export interface TraceSession {
  id: string;
  title?: string;
  clientKey?: string;
  visibility?: string;
}

export interface TracePage {
  cursor: string | null;
  nextCursor: string | null;
  limit: number;
  count: number;
}

export interface TraceTotals {
  turnCount: number;
  durationMsTotal: number;
  promptTokensTotal: number;
  generatedTokensTotal: number;
}

export interface TraceContextUsage {
  promptTokens?: number;
  contextWindowTokens?: number;
  requestedMaxTokens?: number;
  generatedTokens?: number;
}

export interface TraceDifference {
  dimension: string;
  kind: string;
}

export interface TraceClassification {
  classification: string;
  status: string;
  summary: string;
  dimensions: string[];
}

export interface TraceContextChanges {
  newSegments: TraceDifference[];
  droppedSegments: TraceDifference[];
  carriedForwardSegmentCount: number;
  otherContextDifferences: TraceDifference[];
}

export type TraceTurnComparison =
  | { available: false; comparedToRunId?: string; reason?: string }
  | { available: true; comparedToRunId?: string; settingsChanges: TraceDifference[]; contextChanges: TraceContextChanges; classifications: TraceClassification[] };

export interface TraceDiagnosticHighlights {
  findingRuleIds: string[];
  statusCounts: Record<"finding" | "notObserved" | "unavailable" | "pending" | "suppressed", number>;
}

export interface TraceTurn {
  runId: string;
  recordedTs: number;
  createdAt: string;
  source: string;
  client: string;
  model: string;
  promptSummary: string;
  responseSummary: string;
  redacted: boolean;
  finishReason?: string;
  error?: string;
  contextUsage?: TraceContextUsage;
  durationMs?: number;
  cumulative: TraceTotals;
  diagnosticHighlights: TraceDiagnosticHighlights;
  turnComparison?: TraceTurnComparison;
}

export interface TraceBranchChild {
  id: string;
  createdAt?: string;
  source?: string;
  promptSummary?: string;
  responseSummary?: string;
  finishReason?: string;
}

export interface TraceBranch {
  parentRunId: string;
  children: TraceBranchChild[];
}

export interface SessionTrace {
  schemaVersion: "clozn.session-trace.v1";
  generatedAt: string;
  sessionId: string;
  session: TraceSession;
  page: TracePage;
  turns: TraceTurn[];
  branches: TraceBranch[];
  totalsThroughThisPage: TraceTotals;
  diagnosticRuleRegistry: Array<{ ruleId: string; ruleName: string }>;
  firstWentWrongCandidates: Array<{ kind: string; runId: string; recordedTs: number; summary: string; comparedToRunId?: string; ruleIds?: string[] }>;
}

function decodeTotals(value: unknown, endpoint: string, path: string): TraceTotals {
  const row = object(value, endpoint, path);
  closed(row, ["turn_count", "duration_ms_total", "prompt_tokens_total", "generated_tokens_total"], endpoint, path);
  return {
    turnCount: nonNegativeInteger(required(row, "turn_count", endpoint, path), endpoint, `${path}.turn_count`),
    durationMsTotal: nonNegativeInteger(required(row, "duration_ms_total", endpoint, path), endpoint, `${path}.duration_ms_total`),
    promptTokensTotal: nonNegativeInteger(required(row, "prompt_tokens_total", endpoint, path), endpoint, `${path}.prompt_tokens_total`),
    generatedTokensTotal: nonNegativeInteger(required(row, "generated_tokens_total", endpoint, path), endpoint, `${path}.generated_tokens_total`),
  };
}

function decodeContextUsage(value: unknown, endpoint: string, path: string): TraceContextUsage {
  const row = object(value, endpoint, path);
  closed(row, ["prompt_tokens", "context_window_tokens", "requested_max_tokens", "generated_tokens"], endpoint, path);
  return {
    promptTokens: optional(row, "prompt_tokens", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    contextWindowTokens: optional(row, "context_window_tokens", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    requestedMaxTokens: optional(row, "requested_max_tokens", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    generatedTokens: optional(row, "generated_tokens", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
  };
}

function decodeDifference(value: unknown, endpoint: string, path: string): TraceDifference {
  const row = object(value, endpoint, path);
  if (!("dimension" in row) || !("kind" in row)) fail(endpoint, path, "must contain dimension and kind");
  return {
    dimension: nonEmptyString(row.dimension, endpoint, `${path}.dimension`),
    kind: nonEmptyString(row.kind, endpoint, `${path}.kind`),
  };
}

function decodeDifferenceList(value: unknown, endpoint: string, path: string): TraceDifference[] {
  return array(value, endpoint, path).map((entry, index) => decodeDifference(entry, endpoint, `${path}[${index}]`));
}

function decodeClassification(value: unknown, endpoint: string, path: string): TraceClassification {
  const row = object(value, endpoint, path);
  return {
    classification: nonEmptyString(required(row, "classification", endpoint, path), endpoint, `${path}.classification`),
    status: nonEmptyString(required(row, "status", endpoint, path), endpoint, `${path}.status`),
    summary: nonEmptyString(required(row, "summary", endpoint, path), endpoint, `${path}.summary`),
    dimensions: optional(row, "dimensions", (v, p) => array(v, endpoint, p).map((item, index) => nonEmptyString(item, endpoint, `${p}[${index}]`)), endpoint, path) ?? [],
  };
}

function decodeContextChanges(value: unknown, endpoint: string, path: string): TraceContextChanges {
  const row = object(value, endpoint, path);
  closed(row, ["new_segments", "dropped_segments", "carried_forward_segment_count", "other_context_differences"], endpoint, path);
  return {
    newSegments: decodeDifferenceList(required(row, "new_segments", endpoint, path), endpoint, `${path}.new_segments`),
    droppedSegments: decodeDifferenceList(required(row, "dropped_segments", endpoint, path), endpoint, `${path}.dropped_segments`),
    carriedForwardSegmentCount: nonNegativeInteger(required(row, "carried_forward_segment_count", endpoint, path), endpoint, `${path}.carried_forward_segment_count`),
    otherContextDifferences: optional(row, "other_context_differences", (v, p) => decodeDifferenceList(v, endpoint, p), endpoint, path) ?? [],
  };
}

function decodeTurnComparison(value: unknown, endpoint: string, path: string): TraceTurnComparison {
  const row = object(value, endpoint, path);
  closed(row, ["available", "compared_to_run_id", "reason", "settings_changes", "context_changes", "classifications"], endpoint, path);
  const available = booleanValue(required(row, "available", endpoint, path), endpoint, `${path}.available`);
  const comparedToRunId = optional(row, "compared_to_run_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path);
  if (!available) {
    if ("settings_changes" in row || "context_changes" in row || "classifications" in row) fail(endpoint, path, "must not include available comparison fields when available is false");
    return { available, comparedToRunId, reason: optional(row, "reason", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path) };
  }
  return {
    available,
    comparedToRunId,
    settingsChanges: decodeDifferenceList(required(row, "settings_changes", endpoint, path), endpoint, `${path}.settings_changes`),
    contextChanges: decodeContextChanges(required(row, "context_changes", endpoint, path), endpoint, `${path}.context_changes`),
    classifications: array(required(row, "classifications", endpoint, path), endpoint, `${path}.classifications`).map((entry, index) => decodeClassification(entry, endpoint, `${path}.classifications[${index}]`)),
  };
}

function decodeStatusCounts(value: unknown, endpoint: string, path: string): TraceDiagnosticHighlights["statusCounts"] {
  const row = object(value, endpoint, path);
  const wireKeys = ["finding", "not_observed", "unavailable", "pending", "suppressed"] as const;
  closed(row, wireKeys, endpoint, path);
  return {
    finding: nonNegativeInteger(required(row, "finding", endpoint, path), endpoint, `${path}.finding`),
    notObserved: nonNegativeInteger(required(row, "not_observed", endpoint, path), endpoint, `${path}.not_observed`),
    unavailable: nonNegativeInteger(required(row, "unavailable", endpoint, path), endpoint, `${path}.unavailable`),
    pending: nonNegativeInteger(required(row, "pending", endpoint, path), endpoint, `${path}.pending`),
    suppressed: nonNegativeInteger(required(row, "suppressed", endpoint, path), endpoint, `${path}.suppressed`),
  };
}

function decodeDiagnosticHighlights(value: unknown, endpoint: string, path: string): TraceDiagnosticHighlights {
  const row = object(value, endpoint, path);
  closed(row, ["findings", "status_counts"], endpoint, path);
  const findings = array(required(row, "findings", endpoint, path), endpoint, `${path}.findings`).map((entry, index) => {
    const finding = object(entry, endpoint, `${path}.findings[${index}]`);
    const ruleId = nonEmptyString(required(finding, "rule_id", endpoint, `${path}.findings[${index}]`), endpoint, `${path}.findings[${index}].rule_id`);
    enumValue(required(finding, "status", endpoint, `${path}.findings[${index}]`), ["finding"], endpoint, `${path}.findings[${index}].status`);
    return ruleId;
  });
  return { findingRuleIds: findings, statusCounts: decodeStatusCounts(required(row, "status_counts", endpoint, path), endpoint, `${path}.status_counts`) };
}

function decodeTurn(value: unknown, endpoint: string, path: string): TraceTurn {
  const row = object(value, endpoint, path);
  closed(row, [
    "run_id", "recorded_ts", "created_at", "source", "client", "model", "prompt_summary", "response_summary",
    "redacted", "finish_reason", "error", "context_usage", "timing", "cumulative", "diagnostic_highlights", "turn_comparison",
  ], endpoint, path);
  const timing = optional(row, "timing", (v, p) => {
    const timingRow = object(v, endpoint, p);
    closed(timingRow, ["duration_ms"], endpoint, p);
    return nonNegativeInteger(required(timingRow, "duration_ms", endpoint, p), endpoint, `${p}.duration_ms`);
  }, endpoint, path);
  return {
    runId: nonEmptyString(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`),
    recordedTs: finiteNumber(required(row, "recorded_ts", endpoint, path), endpoint, `${path}.recorded_ts`),
    createdAt: stringValue(required(row, "created_at", endpoint, path), endpoint, `${path}.created_at`),
    source: stringValue(required(row, "source", endpoint, path), endpoint, `${path}.source`),
    client: stringValue(required(row, "client", endpoint, path), endpoint, `${path}.client`),
    model: stringValue(required(row, "model", endpoint, path), endpoint, `${path}.model`),
    promptSummary: stringValue(required(row, "prompt_summary", endpoint, path), endpoint, `${path}.prompt_summary`),
    responseSummary: stringValue(required(row, "response_summary", endpoint, path), endpoint, `${path}.response_summary`),
    redacted: booleanValue(required(row, "redacted", endpoint, path), endpoint, `${path}.redacted`),
    finishReason: optional(row, "finish_reason", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path),
    error: optional(row, "error", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path),
    contextUsage: optional(row, "context_usage", (v, p) => decodeContextUsage(v, endpoint, p), endpoint, path),
    durationMs: timing,
    cumulative: decodeTotals(required(row, "cumulative", endpoint, path), endpoint, `${path}.cumulative`),
    diagnosticHighlights: decodeDiagnosticHighlights(required(row, "diagnostic_highlights", endpoint, path), endpoint, `${path}.diagnostic_highlights`),
    turnComparison: optional(row, "turn_comparison", (v, p) => decodeTurnComparison(v, endpoint, p), endpoint, path),
  };
}

function decodeBranchChild(value: unknown, endpoint: string, path: string): TraceBranchChild {
  const row = object(value, endpoint, path);
  return {
    id: nonEmptyString(required(row, "id", endpoint, path), endpoint, `${path}.id`),
    createdAt: optional(row, "created_at", (v, p) => stringValue(v, endpoint, p), endpoint, path),
    source: optional(row, "source", (v, p) => stringValue(v, endpoint, p), endpoint, path),
    promptSummary: optional(row, "prompt_summary", (v, p) => stringValue(v, endpoint, p), endpoint, path),
    responseSummary: optional(row, "response_summary", (v, p) => stringValue(v, endpoint, p), endpoint, path),
    finishReason: optional(row, "finish_reason", (v, p) => stringValue(v, endpoint, p), endpoint, path),
  };
}

function decodeBranch(value: unknown, endpoint: string, path: string): TraceBranch {
  const row = object(value, endpoint, path);
  closed(row, ["parent_run_id", "children"], endpoint, path);
  return {
    parentRunId: nonEmptyString(required(row, "parent_run_id", endpoint, path), endpoint, `${path}.parent_run_id`),
    children: array(required(row, "children", endpoint, path), endpoint, `${path}.children`).map((entry, index) => decodeBranchChild(entry, endpoint, `${path}.children[${index}]`)),
  };
}

function decodeTraceSession(value: unknown, endpoint: string, path: string): TraceSession {
  const row = object(value, endpoint, path);
  closed(row, ["id", "title", "client_key", "privacy"], endpoint, path);
  const privacy = object(required(row, "privacy", endpoint, path), endpoint, `${path}.privacy`);
  return {
    id: nonEmptyString(required(row, "id", endpoint, path), endpoint, `${path}.id`),
    title: optional(row, "title", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path),
    clientKey: optional(row, "client_key", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path),
    visibility: optional(privacy, "visibility", (v, p) => stringValue(v, endpoint, p), endpoint, `${path}.privacy`),
  };
}

function decodeCandidate(value: unknown, endpoint: string, path: string): SessionTrace["firstWentWrongCandidates"][number] {
  const row = object(value, endpoint, path);
  closed(row, ["kind", "run_id", "recorded_ts", "summary", "compared_to_run_id", "rule_ids"], endpoint, path);
  const comparedToRunId = optional(row, "compared_to_run_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path);
  const ruleIds = optional(row, "rule_ids", (v, p) => array(v, endpoint, p).map((item, index) => nonEmptyString(item, endpoint, `${p}[${index}]`)), endpoint, path);
  return {
    kind: enumValue(required(row, "kind", endpoint, path), ["first_finding", "first_settings_drift", "first_failed_run"], endpoint, `${path}.kind`),
    runId: nonEmptyString(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`),
    recordedTs: finiteNumber(required(row, "recorded_ts", endpoint, path), endpoint, `${path}.recorded_ts`),
    summary: nonEmptyString(required(row, "summary", endpoint, path), endpoint, `${path}.summary`),
    ...(comparedToRunId ? { comparedToRunId } : {}),
    ...(ruleIds ? { ruleIds } : {}),
  };
}

export function decodeSessionTrace(value: unknown, endpoint = "/sessions/<id>/trace"): SessionTrace {
  const row = object(value, endpoint, "$");
  closed(row, [
    "schema_version", "generated_at", "session_id", "session", "page", "turns", "branches", "totals_through_this_page",
    "diagnostic_rule_registry", "first_went_wrong_candidates",
  ], endpoint, "$");
  const page = object(required(row, "page", endpoint, "$") , endpoint, "$.page");
  closed(page, ["cursor", "next_cursor", "limit", "count"], endpoint, "$.page");
  const traceSession = decodeTraceSession(required(row, "session", endpoint, "$") , endpoint, "$.session");
  const sessionId = nonEmptyString(required(row, "session_id", endpoint, "$") , endpoint, "$.session_id");
  if (traceSession.id !== sessionId) fail(endpoint, "$.session.id", "must match $.session_id");
  const registry = array(required(row, "diagnostic_rule_registry", endpoint, "$") , endpoint, "$.diagnostic_rule_registry").map((entry, index) => {
    const item = object(entry, endpoint, `$.diagnostic_rule_registry[${index}]`);
    closed(item, ["rule_id", "rule_name"], endpoint, `$.diagnostic_rule_registry[${index}]`);
    return {
      ruleId: nonEmptyString(required(item, "rule_id", endpoint, `$.diagnostic_rule_registry[${index}]`), endpoint, `$.diagnostic_rule_registry[${index}].rule_id`),
      ruleName: nonEmptyString(required(item, "rule_name", endpoint, `$.diagnostic_rule_registry[${index}]`), endpoint, `$.diagnostic_rule_registry[${index}].rule_name`),
    };
  });
  return {
    schemaVersion: enumValue(required(row, "schema_version", endpoint, "$") , ["clozn.session-trace.v1"], endpoint, "$.schema_version"),
    generatedAt: nonEmptyString(required(row, "generated_at", endpoint, "$") , endpoint, "$.generated_at"),
    sessionId,
    session: traceSession,
    page: {
      cursor: required(page, "cursor", endpoint, "$.page") === null ? null : nonEmptyString(required(page, "cursor", endpoint, "$.page"), endpoint, "$.page.cursor"),
      nextCursor: required(page, "next_cursor", endpoint, "$.page") === null ? null : nonEmptyString(required(page, "next_cursor", endpoint, "$.page"), endpoint, "$.page.next_cursor"),
      limit: positiveInteger(required(page, "limit", endpoint, "$.page"), endpoint, "$.page.limit"),
      count: nonNegativeInteger(required(page, "count", endpoint, "$.page"), endpoint, "$.page.count"),
    },
    turns: array(required(row, "turns", endpoint, "$") , endpoint, "$.turns").map((entry, index) => decodeTurn(entry, endpoint, `$.turns[${index}]`)),
    branches: array(required(row, "branches", endpoint, "$") , endpoint, "$.branches").map((entry, index) => decodeBranch(entry, endpoint, `$.branches[${index}]`)),
    totalsThroughThisPage: decodeTotals(required(row, "totals_through_this_page", endpoint, "$") , endpoint, "$.totals_through_this_page"),
    diagnosticRuleRegistry: registry,
    firstWentWrongCandidates: array(required(row, "first_went_wrong_candidates", endpoint, "$") , endpoint, "$.first_went_wrong_candidates").map((entry, index) => decodeCandidate(entry, endpoint, `$.first_went_wrong_candidates[${index}]`)),
  };
}

export interface StandaloneRun {
  id: string;
  sessionKey?: string | null;
  createdAt?: string | null;
  createdTs?: number | null;
  model?: string | null;
  promptSummary?: string | null;
  responseSummary?: string | null;
}

export function decodeStandaloneRunsDocument(value: unknown, endpoint = "/runs"): readonly StandaloneRun[] {
  const row = object(value, endpoint, "$");
  closed(row, ["runs"], endpoint, "$");
  return array(required(row, "runs", endpoint, "$") , endpoint, "$.runs").map((value, index) => {
    const item = object(value, endpoint, `$.runs[${index}]`);
    const stringOrNull = (key: string): string | null | undefined => optional(item, key, (v, p) => nullableString(v, endpoint, p), endpoint, `$.runs[${index}]`);
    const numberOrNull = (key: string): number | null | undefined => optional(item, key, (v, p) => v === null ? null : finiteNumber(v, endpoint, p), endpoint, `$.runs[${index}]`);
    return {
      id: nonEmptyString(required(item, "id", endpoint, `$.runs[${index}]`), endpoint, `$.runs[${index}].id`),
      sessionKey: stringOrNull("session_key"),
      createdAt: stringOrNull("created_at"),
      createdTs: numberOrNull("created_ts"),
      model: stringOrNull("model"),
      promptSummary: stringOrNull("prompt_summary"),
      responseSummary: stringOrNull("response_summary"),
    };
  });
}
