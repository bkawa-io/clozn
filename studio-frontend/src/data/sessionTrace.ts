/**
 * F3 -- client for F2's `GET /sessions/<id>/trace` (`clozn.session-trace.v1`,
 * `clozn/server/routes/session_trace.py` / `clozn/runs/session_trace.py`) and F1's `GET /sessions`
 * (`clozn.session.v1` list, `clozn/server/routes/sessions.py`).
 *
 * PAGINATION IS A FIRST-CLASS PART OF THIS CLIENT, NOT AN AFTERTHOUGHT
 * -----------------------------------------------------------------------
 * `build_trace()` is cursor-paginated (`page.cursor`/`page.next_cursor`) precisely because a session can
 * hold thousands of turns -- see session_trace.py's own module docstring. This module exposes ONE page
 * per call (`loadSessionTracePage`) and never fetches "the whole session" on a caller's behalf; the
 * feature layer (`features/investigation/ConversationInvestigation.tsx`) decides when to fetch another
 * page, and always states plainly when what is rendered is a PARTIAL trace. Silently walking every page
 * here would be exactly the kind of "looks complete, isn't" the F3 brief calls out as the worst failure
 * mode for this surface.
 *
 * NO CAUSAL VOCABULARY, NO INVENTED STATUS
 * --------------------------------------------
 * `firstWentWrongCandidates` is parsed verbatim from the backend's own three-kind enum
 * (`first_finding` / `first_settings_drift` / `first_failed_run`) -- this module never infers a fourth
 * kind, never ranks candidates against each other, and never fabricates a candidate when the array is
 * empty (an empty array IS the honest answer "no candidate of any kind was found in this session").
 *
 * `diagnosticHighlights.findings` reuses `data/diagnosisRepair.ts`'s own `parseFinding` -- see that
 * export's doc comment for why this is composition, not a parallel implementation of D1's finding shape.
 * `diagnosticHighlights.statusCounts` reuses D1's own five-value vocabulary (`RepairFindingStatus`)
 * directly, because `clozn.session-trace.v1` embeds `clozn.diagnosis-findings.v1`'s OWN status tally
 * verbatim -- this is the same artifact, not a sibling one, so importing rather than redeclaring is the
 * correct call here (contrast with `data/types.ts`'s deliberately separate `DiagnosisStatus`/
 * `PerformanceRuleStatus`, which really are different rule engines).
 */
import { parseFinding, type RepairFinding, type RepairFindingStatus } from "./diagnosisRepair";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function bool(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function strArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

// ------------------------------------------------------------------------------------------- session list

export interface SessionSummary {
  id: string;
  title?: string;
  clientKey?: string;
  visibility: "visible" | "hidden";
  createdAt: string;
  createdTs: number;
  materializedFrom?: string;
  /** Omitted (never zeroed) when the session has no member runs at all -- see `sessions.get_session`'s
   * own "exists, zero runs yet" vs "no such session" distinction. A session in `loadSessionList`'s
   * result always has at least one run OR an explicit row, but the derived trio is still only present
   * when at least one run exists. */
  firstActivityTs?: number;
  lastActivityTs?: number;
  runCount?: number;
}

function parseSessionSummary(raw: unknown): SessionSummary {
  const doc = record(raw);
  const privacy = record(doc.privacy);
  return {
    id: str(doc.id) ?? "",
    title: str(doc.title),
    clientKey: str(doc.client_key),
    visibility: privacy.visibility === "hidden" ? "hidden" : "visible",
    createdAt: str(doc.created_at) ?? "",
    createdTs: num(doc.created_ts) ?? 0,
    materializedFrom: str(doc.materialized_from),
    firstActivityTs: num(doc.first_activity_ts),
    lastActivityTs: num(doc.last_activity_ts),
    runCount: num(doc.run_count),
  };
}

export class SessionListLoadError extends Error {}

export async function loadSessionList(
  options: { limit?: number; includeHidden?: boolean; signal?: AbortSignal } = {},
): Promise<SessionSummary[]> {
  const params = new URLSearchParams();
  if (options.limit != null) params.set("limit", String(options.limit));
  if (options.includeHidden) params.set("include_hidden", "1");
  const qs = params.toString();
  const response = await fetch(`/sessions${qs ? `?${qs}` : ""}`, { signal: options.signal });
  const body = record(await response.json().catch(() => ({})));
  if (!response.ok) {
    throw new SessionListLoadError(str(body.error) ?? `sessions request failed (${response.status})`);
  }
  return records(body.sessions).map(parseSessionSummary);
}

// -------------------------------------------------------------------------------------------- trace page

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

/** Every key omitted, never zero-padded, mirroring `clozn.session-trace.v1`'s own `ContextUsage`
 * contract -- an absent key means the context receipt never recorded that figure for this turn, which is
 * a different fact from "recorded as zero". */
export interface TraceContextUsage {
  promptTokens?: number;
  contextWindowTokens?: number;
  requestedMaxTokens?: number;
  generatedTokens?: number;
}

/** A loose passthrough of one `run_diff.compare_runs()` `differences[]` entry -- only `dimension`/`kind`
 * are asserted by the schema; everything else is rendered generically (see
 * `ConversationInvestigation.tsx`'s `differenceDetail`). */
export interface TraceDifference {
  dimension: string;
  kind: string;
  valueA?: unknown;
  valueB?: unknown;
  note?: string;
}

/** A loose passthrough of one `run_diff.compare_runs()` `findings[]` entry -- run_diff's OWN "observed"
 * vocabulary (`model_changed`, `context_omission`, ...), deliberately never merged with D1's five-value
 * finding-status vocabulary (`RepairFindingStatus`) even though both are named "status" on the wire. */
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

/** Mirrors the schema's own shape exactly: `available: false` means `run_diff.compare_runs()` itself
 * could not compare the pair -- `reason` names why, and the rest is ABSENT rather than fabricated as
 * empty. Only turns after the session's first carry this at all. */
export type TraceTurnComparison =
  | { available: false; comparedToRunId?: string; reason?: string }
  | {
      available: true;
      comparedToRunId?: string;
      settingsChanges: TraceDifference[];
      contextChanges: TraceContextChanges;
      classifications: TraceClassification[];
    };

export interface TraceDiagnosticHighlights {
  /** Only `status === "finding"` entries -- see this module's own doc comment. Narrowed to the
   * "finding" branch of `RepairFinding`'s own union so callers get `severity`/`confidence`/
   * `suggestedActions` without re-checking `status` themselves. */
  findings: Extract<RepairFinding, { status: "finding" }>[];
  /** D1's own five-key tally (`finding`/`not_observed`/`unavailable`/`pending`/`suppressed`), always
   * all five keys -- `evaluate()` always produces exactly one entry per registered rule. */
  statusCounts: Record<RepairFindingStatus, number>;
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
  /** Absent only for the session's first linear turn. */
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

/** A run whose `parent_run_id` points at one of the page's own linear turns -- never folded into
 * `turns`. See session_trace.py's "BRANCHES ARE NEVER FLATTENED" section: no run ever appears in both
 * `turns` and `branches`. */
export interface TraceBranch {
  parentRunId: string;
  children: TraceBranchChild[];
}

export type TraceCandidateKind = "first_finding" | "first_settings_drift" | "first_failed_run";

/** A DETERMINISTIC heuristic candidate for where a session's evidence trail first shows a problem --
 * never a causal claim. See session_trace.py's "NO CAUSAL VOCABULARY" section: nothing in this artifact,
 * or this client, ever states or implies that one candidate EXPLAINS another turn's behavior. */
export interface TraceCandidate {
  kind: TraceCandidateKind;
  runId: string;
  recordedTs: number;
  summary: string;
  comparedToRunId?: string;
  ruleIds?: string[];
}

export interface TraceRuleRegistryEntry {
  ruleId: string;
  ruleName: string;
}

export interface SessionTracePage {
  schemaVersion: string;
  generatedAt: string;
  sessionId: string;
  session: TraceSession;
  page: TracePage;
  turns: TraceTurn[];
  branches: TraceBranch[];
  totalsThroughThisPage: TraceTotals;
  diagnosticRuleRegistry: TraceRuleRegistryEntry[];
  firstWentWrongCandidates: TraceCandidate[];
}

function parseTotals(raw: unknown): TraceTotals {
  const item = record(raw);
  return {
    turnCount: num(item.turn_count) ?? 0,
    durationMsTotal: num(item.duration_ms_total) ?? 0,
    promptTokensTotal: num(item.prompt_tokens_total) ?? 0,
    generatedTokensTotal: num(item.generated_tokens_total) ?? 0,
  };
}

function parseContextUsage(raw: unknown): TraceContextUsage | undefined {
  const item = record(raw);
  const out: TraceContextUsage = {
    promptTokens: num(item.prompt_tokens),
    contextWindowTokens: num(item.context_window_tokens),
    requestedMaxTokens: num(item.requested_max_tokens),
    generatedTokens: num(item.generated_tokens),
  };
  const hasAny = Object.values(out).some((value) => value !== undefined);
  return hasAny ? out : undefined;
}

function parseDifference(raw: unknown): TraceDifference | null {
  const item = record(raw);
  const dimension = str(item.dimension);
  const kind = str(item.kind);
  if (!dimension || !kind) return null;
  return { dimension, kind, valueA: item.value_a, valueB: item.value_b, note: str(item.note) };
}

function parseDifferenceList(raw: unknown): TraceDifference[] {
  return records(raw).map(parseDifference).filter((item): item is TraceDifference => item != null);
}

function parseClassification(raw: unknown): TraceClassification | null {
  const item = record(raw);
  const classification = str(item.classification);
  const status = str(item.status);
  const summary = str(item.summary);
  if (!classification || !status || !summary) return null;
  return { classification, status, summary, dimensions: strArray(item.dimensions) };
}

function parseContextChanges(raw: unknown): TraceContextChanges {
  const item = record(raw);
  return {
    newSegments: parseDifferenceList(item.new_segments),
    droppedSegments: parseDifferenceList(item.dropped_segments),
    carriedForwardSegmentCount: num(item.carried_forward_segment_count) ?? 0,
    otherContextDifferences: parseDifferenceList(item.other_context_differences),
  };
}

function parseTurnComparison(raw: unknown): TraceTurnComparison | undefined {
  if (raw == null) return undefined;
  const item = record(raw);
  const comparedToRunId = str(item.compared_to_run_id);
  if (item.available !== true) {
    return { available: false, comparedToRunId, reason: str(item.reason) };
  }
  return {
    available: true,
    comparedToRunId,
    settingsChanges: parseDifferenceList(item.settings_changes),
    contextChanges: parseContextChanges(item.context_changes),
    classifications: records(item.classifications).map(parseClassification).filter(
      (entry): entry is TraceClassification => entry != null,
    ),
  };
}

const FINDING_STATUSES: readonly RepairFindingStatus[] =
  ["finding", "not_observed", "unavailable", "pending", "suppressed"];

function parseStatusCounts(raw: unknown): Record<RepairFindingStatus, number> {
  const item = record(raw);
  const out = {} as Record<RepairFindingStatus, number>;
  for (const status of FINDING_STATUSES) out[status] = num(item[status]) ?? 0;
  return out;
}

function parseDiagnosticHighlights(raw: unknown): TraceDiagnosticHighlights {
  const item = record(raw);
  const findings = records(item.findings).map(parseFinding).filter(
    (entry): entry is RepairFinding & { status: "finding" } => entry.status === "finding",
  );
  return { findings, statusCounts: parseStatusCounts(item.status_counts) };
}

function parseTurn(raw: unknown): TraceTurn | null {
  const item = record(raw);
  const runId = str(item.run_id);
  if (!runId) return null;
  return {
    runId,
    recordedTs: num(item.recorded_ts) ?? 0,
    createdAt: str(item.created_at) ?? "",
    source: str(item.source) ?? "—",
    client: str(item.client) ?? "—",
    model: str(item.model) ?? "—",
    promptSummary: str(item.prompt_summary) ?? "",
    responseSummary: str(item.response_summary) ?? "",
    redacted: bool(item.redacted),
    finishReason: str(item.finish_reason),
    error: str(item.error),
    contextUsage: parseContextUsage(item.context_usage),
    durationMs: num(record(item.timing).duration_ms),
    cumulative: parseTotals(item.cumulative),
    diagnosticHighlights: parseDiagnosticHighlights(item.diagnostic_highlights),
    turnComparison: parseTurnComparison(item.turn_comparison),
  };
}

function parseBranchChild(raw: unknown): TraceBranchChild | null {
  const item = record(raw);
  const id = str(item.id);
  if (!id) return null;
  return {
    id,
    createdAt: str(item.created_at),
    source: str(item.source),
    promptSummary: str(item.prompt_summary),
    responseSummary: str(item.response_summary),
    finishReason: str(item.finish_reason),
  };
}

function parseBranch(raw: unknown): TraceBranch | null {
  const item = record(raw);
  const parentRunId = str(item.parent_run_id);
  if (!parentRunId) return null;
  const children = records(item.children).map(parseBranchChild).filter(
    (child): child is TraceBranchChild => child != null,
  );
  return { parentRunId, children };
}

const CANDIDATE_KINDS: readonly TraceCandidateKind[] =
  ["first_finding", "first_settings_drift", "first_failed_run"];

function parseCandidate(raw: unknown): TraceCandidate | null {
  const item = record(raw);
  const kind = item.kind;
  const runId = str(item.run_id);
  const summary = str(item.summary);
  if (typeof kind !== "string" || !CANDIDATE_KINDS.includes(kind as TraceCandidateKind) || !runId || !summary) {
    return null;
  }
  return {
    kind: kind as TraceCandidateKind,
    runId,
    recordedTs: num(item.recorded_ts) ?? 0,
    summary,
    comparedToRunId: str(item.compared_to_run_id),
    ruleIds: item.rule_ids == null ? undefined : strArray(item.rule_ids),
  };
}

function parseSession(raw: unknown): TraceSession {
  const item = record(raw);
  const privacy = record(item.privacy);
  return {
    id: str(item.id) ?? "",
    title: str(item.title),
    clientKey: str(item.client_key),
    visibility: str(privacy.visibility),
  };
}

function parseSessionTracePage(raw: unknown): SessionTracePage {
  const doc = record(raw);
  const page = record(doc.page);
  return {
    schemaVersion: str(doc.schema_version) ?? "",
    generatedAt: str(doc.generated_at) ?? "",
    sessionId: str(doc.session_id) ?? "",
    session: parseSession(doc.session),
    page: {
      cursor: str(page.cursor) ?? null,
      nextCursor: str(page.next_cursor) ?? null,
      limit: num(page.limit) ?? 0,
      count: num(page.count) ?? 0,
    },
    turns: records(doc.turns).map(parseTurn).filter((turn): turn is TraceTurn => turn != null),
    branches: records(doc.branches).map(parseBranch).filter((branch): branch is TraceBranch => branch != null),
    totalsThroughThisPage: parseTotals(doc.totals_through_this_page),
    diagnosticRuleRegistry: records(doc.diagnostic_rule_registry).map((entry) => ({
      ruleId: str(entry.rule_id) ?? "",
      ruleName: str(entry.rule_name) ?? "",
    })),
    firstWentWrongCandidates: records(doc.first_went_wrong_candidates).map(parseCandidate).filter(
      (candidate): candidate is TraceCandidate => candidate != null,
    ),
  };
}

export class SessionTraceLoadError extends Error {}
export class SessionTraceNotFoundError extends SessionTraceLoadError {}

/** The ONE fetch this client makes per call -- exactly one page. `cursor` omitted fetches the first
 * page. Callers own accumulation across pages (see `ConversationInvestigation.tsx`); this function never
 * follows `next_cursor` on its own. */
export async function loadSessionTracePage(
  sessionId: string,
  options: { cursor?: string; limit?: number; signal?: AbortSignal } = {},
): Promise<SessionTracePage> {
  const params = new URLSearchParams();
  if (options.cursor) params.set("cursor", options.cursor);
  if (options.limit != null) params.set("limit", String(options.limit));
  const qs = params.toString();
  const response = await fetch(
    `/sessions/${encodeURIComponent(sessionId)}/trace${qs ? `?${qs}` : ""}`,
    { signal: options.signal },
  );
  if (response.status === 404) throw new SessionTraceNotFoundError("session not found");
  const body = record(await response.json().catch(() => ({})));
  if (!response.ok) {
    throw new SessionTraceLoadError(str(body.error) ?? `session trace request failed (${response.status})`);
  }
  return parseSessionTracePage(body);
}

export function describeSessionTraceError(error: unknown): string {
  return error instanceof Error ? error.message : "the request failed";
}
