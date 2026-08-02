/** Small typed client for F5/F6's explicit, scoped Teach Once store. */

export type CorrectionScopeKind = "session" | "client" | "model" | "project" | "global_local";
export type CorrectionType = "output_format" | "source_requirement" | "style" | "forbidden_behavior";

export interface CorrectionScope {
  kind: CorrectionScopeKind;
  value?: string;
}

export interface Correction {
  schema_version: string;
  id: string;
  scope: CorrectionScope;
  type: CorrectionType;
  content?: string;
  content_hash: string;
  enabled: boolean;
  created_ts: number;
  created_at: string;
  confirmed_ts?: number;
  disabled_ts?: number;
  deleted_ts?: number;
}

export interface CorrectionList {
  schema_version: string;
  corrections: Correction[];
}

export interface CorrectionResolution {
  schema_version: string;
  applied: Array<{ correction_id: string; type: CorrectionType; scope: CorrectionScope; content_hash: string }>;
  conflicts: Array<{ type: CorrectionType; winner_id: string; losing_ids: string[]; rule: string }>;
}

export type CorrectionMatchCriterion = "exact_output" | "tool_parse" | "finish_reason" | "token_budget";

export interface CorrectionVerification {
  schema_version: string;
  correction_id: string;
  target_run_id: string;
  child_run_id: string;
  match_criterion: CorrectionMatchCriterion;
  comparison: { available: boolean; matched?: boolean; reason?: string };
  verification: "passed" | "failed";
  promoted: boolean;
  reason: string;
  correction: Correction;
}

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function errorText(body: JsonRecord, status: number) {
  const error = body.error;
  if (typeof error === "string") return error;
  const nested = record(error);
  if (typeof nested.message === "string") return nested.message;
  return `Request failed (${status})`;
}

async function request(url: string, options: RequestInit = {}): Promise<JsonRecord> {
  const response = await fetch(url, options);
  let body: JsonRecord = {};
  try { body = record(await response.json()); } catch { /* status remains authoritative */ }
  if (!response.ok || body.error) throw new Error(errorText(body, response.status));
  return body;
}

async function post(url: string, body: JsonRecord, signal?: AbortSignal) {
  return request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export async function loadCorrections(signal?: AbortSignal): Promise<CorrectionList> {
  return await request("/corrections", { signal }) as unknown as CorrectionList;
}

export async function draftCorrection(
  scopeKind: CorrectionScopeKind,
  scopeValue: string | undefined,
  type: CorrectionType,
  content: string,
  signal?: AbortSignal,
): Promise<Correction> {
  return await post("/corrections", {
    scope_kind: scopeKind,
    ...(scopeValue ? { scope_value: scopeValue } : {}),
    type,
    content,
  }, signal) as unknown as Correction;
}

export async function confirmCorrection(id: string, signal?: AbortSignal): Promise<Correction> {
  const body = await post(`/corrections/${encodeURIComponent(id)}/confirm`, {}, signal);
  return (body.correction ?? body) as unknown as Correction;
}

export async function disableCorrection(id: string, signal?: AbortSignal): Promise<Correction> {
  return await post(`/corrections/${encodeURIComponent(id)}/disable`, {}, signal) as unknown as Correction;
}

export async function enableCorrection(id: string, signal?: AbortSignal): Promise<Correction> {
  return await post(`/corrections/${encodeURIComponent(id)}/enable`, {}, signal) as unknown as Correction;
}

export async function undoCorrection(id: string, signal?: AbortSignal): Promise<Correction> {
  return await post(`/corrections/${encodeURIComponent(id)}/undo`, {}, signal) as unknown as Correction;
}

export async function verifyCorrection(
  id: string,
  targetRunId: string,
  childRunId: string,
  matchCriterion: CorrectionMatchCriterion = "exact_output",
  signal?: AbortSignal,
): Promise<CorrectionVerification> {
  return await post(`/corrections/${encodeURIComponent(id)}/verify`, {
    target_run_id: targetRunId,
    child_run_id: childRunId,
    match_criterion: matchCriterion,
  }, signal) as unknown as CorrectionVerification;
}
