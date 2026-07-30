/**
 * Client for D3's `clozn.corrective-flow.v1` preview -> confirm -> keep HTTP surface
 * (clozn/server/routes/corrective_actions.py, clozn/behavior/corrective_flow.py) -- the registry of
 * bounded corrective retries (`less-verbose`, `more-concrete`, `use-context`, `ask-before-guessing`,
 * `preserve-formatting`, `stop-repeating`; see clozn/replay/corrective.py's `CORRECTION_PRESETS`, the
 * one place these ids and their exact injected instruction text are defined) plus the reversible
 * scoped-keep transaction that can later promote a corrected child to `once` (this run's own selected
 * revision), `session`, or `profile` scope.
 *
 * MOVED HERE FROM `features/behavior/api.ts` (D5, guided-repair UI): this client was originally written
 * for the Behavior panel's own free-standing "ANSWER FIXES" run picker, but D5 needs the identical
 * preview/confirm/keep mechanics wired to whichever run a Lens diagnosis panel is already showing.
 * Per this codebase's own convention (every OTHER cross-feature-reusable API client lives under
 * `src/data/`, e.g. `received-context.ts`, `tokenWorkbench.ts`; no feature imports another feature's
 * internals anywhere in this tree), the honest fix is to relocate the client here rather than duplicate
 * it or add a first-ever feature-to-feature import. `features/behavior/api.ts` re-exports every name
 * below verbatim, so `Behavior.tsx` needed no changes at all.
 *
 * NOT the same vocabulary as `data/diagnosisRepair.ts`'s `RepairSuggestedActionKind` -- see that
 * module's own doc comment for why no bridge exists between D1's provisional suggested-action kinds and
 * these six real action ids.
 */

export type CorrectiveScope = "once" | "session" | "profile";
export type CorrectiveBackend = "prompt_policy" | "control_vector";

export interface CorrectiveBackendEntry {
  type: string;
  requested_type?: string;
  available?: boolean;
  qualification?: string;
  qualification_id?: string;
  reason?: string;
  unavailability_reason?: string;
}

export interface CorrectiveScopeEligibility {
  scope: CorrectiveScope;
  available: boolean;
  target?: string;
  prior_hash?: string;
  before?: string[];
  after?: string[];
  note?: string;
  unavailability_reason?: string;
}

export interface CorrectiveAction {
  id: string;
  label: string;
  description: string;
  scopes: CorrectiveScope[];
  backends: CorrectiveBackendEntry[];
  scope_eligibility: CorrectiveScopeEligibility[];
}

export interface CorrectiveRegistry {
  schema_version: string;
  version: string;
  run_id: string;
  actions: CorrectiveAction[];
}

export interface CorrectivePreviewReceipt {
  preview_id: string;
  status: string;
  parent_run_id: string;
  action: Pick<CorrectiveAction, "id" | "label" | "description">;
  execution: {
    requested_backend: CorrectiveBackend;
    expected_executed_backend: CorrectiveBackend;
    expected_fallback: boolean;
    qualification?: string;
    qualification_id?: string;
    unavailability_reason?: string;
  };
  scope_eligibility: CorrectiveScopeEligibility[];
  /** Plain-language description of the two-armed comparison a confirm will run
   * (clozn.behavior.corrective_flow.create_preview's own `comparison_contract`) -- "the EXACT change"
   * a guided-repair preview must show before a caller commits to running it. Optional only because the
   * older Behavior-panel client never read it; every live preview response carries it. */
  comparison_contract?: {
    baseline: string;
    corrected: string;
    stored_original: string;
  };
}

export interface CorrectiveChildOutcome {
  status: "success" | "error" | "not_run";
  run_id?: string;
  error?: { code: string; message: string };
}

export interface CorrectiveResult {
  schema_version: string;
  result_id: string;
  parent_run_id: string;
  preview_id: string;
  action: Pick<CorrectiveAction, "id" | "label" | "description">;
  user_intent: { action_id: string; kept_scope?: CorrectiveScope };
  execution: {
    requested_backend: CorrectiveBackend;
    executed_backend?: CorrectiveBackend | null;
    fallback: boolean;
    qualification?: string | null;
    qualification_id?: string | null;
  };
  children: {
    baseline: CorrectiveChildOutcome;
    corrected: CorrectiveChildOutcome;
  };
  comparison: {
    stored_original_reply: string;
    baseline_reply: string;
    corrected_reply: string;
    note: string;
    changed: boolean;
  };
  metrics: Record<string, unknown>;
  coherence?: { degenerate?: boolean; reasons?: string[] };
  intervention_observed?: boolean;
  scope_eligibility: CorrectiveScopeEligibility[];
  outcome: { status: "succeeded" | "execution_error" | "cancelled"; note?: string };
  transaction?: {
    id: string;
    scope: CorrectiveScope;
    target: string;
    undone_ts?: number | null;
  };
  source_use_comparison?: SourceUseComparison;
}

export interface SourceUseComparison {
  status: "available";
  baseline: {
    answer_span_count: number;
    answer_spans_with_clear_source: number;
    observed_source_dependence_ratio: number;
  };
  corrected: {
    answer_span_count: number;
    answer_spans_with_clear_source: number;
    observed_source_dependence_ratio: number;
  };
  delta_observed_source_dependence_ratio: number;
  caveat: string;
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
  if (typeof body.blocked === "string") return body.blocked;
  if (typeof body.reason === "string") return body.reason;
  if (typeof body.note === "string") return body.note;
  return `Request failed (${status})`;
}

async function request(url: string, options: RequestInit = {}): Promise<JsonRecord> {
  const response = await fetch(url, options);
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The HTTP status remains authoritative when the route returns no JSON.
  }
  if (!response.ok || body.error) throw new Error(errorText(body, response.status));
  return body;
}

async function get(url: string, signal?: AbortSignal) {
  return request(url, { signal });
}

async function post(url: string, body: JsonRecord, signal?: AbortSignal) {
  return request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export async function loadCorrectiveActions(
  runId: string,
  signal?: AbortSignal,
): Promise<CorrectiveRegistry> {
  return await get(
    `/runs/${encodeURIComponent(runId)}/corrective-actions`,
    signal,
  ) as unknown as CorrectiveRegistry;
}

export async function previewCorrectiveAction(
  runId: string,
  actionId: string,
  requestedBackend: CorrectiveBackend,
  signal?: AbortSignal,
): Promise<CorrectivePreviewReceipt> {
  return await post(
    `/runs/${encodeURIComponent(runId)}/corrective-actions/preview`,
    { action_id: actionId, requested_backend: requestedBackend },
    signal,
  ) as unknown as CorrectivePreviewReceipt;
}

export async function confirmCorrectivePreview(
  previewId: string,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CorrectiveResult> {
  return await post(
    `/corrective-previews/${encodeURIComponent(previewId)}/confirm`,
    { idempotency_key: idempotencyKey },
    signal,
  ) as unknown as CorrectiveResult;
}

export async function cancelCorrectivePreview(previewId: string, signal?: AbortSignal) {
  return post(`/corrective-previews/${encodeURIComponent(previewId)}/cancel`, {}, signal);
}

export async function keepCorrectiveResult(
  resultId: string,
  scope: CorrectiveScope,
  expectedPriorHash: string,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CorrectiveResult> {
  return await post(
    `/corrective-results/${encodeURIComponent(resultId)}/keep`,
    {
      scope,
      expected_prior_hash: expectedPriorHash,
      idempotency_key: idempotencyKey,
    },
    signal,
  ) as unknown as CorrectiveResult;
}

export async function undoCorrectiveKeep(transactionId: string, signal?: AbortSignal) {
  return post(`/corrective-actions/${encodeURIComponent(transactionId)}/undo`, {}, signal);
}

export async function measureCorrectiveSourceUse(
  result: CorrectiveResult,
): Promise<SourceUseComparison> {
  const baseline = result.children.baseline.run_id;
  const corrected = result.children.corrected.run_id;
  if (!baseline || !corrected) throw new Error("Both child runs are required.");
  // Explicit, separately-triggered expensive work. The result comparison route itself is a pure
  // read of maps produced here and refuses mismatched method/version/threshold contracts.
  await Promise.all([
    post(`/runs/${encodeURIComponent(baseline)}/influence-map`, {}),
    post(`/runs/${encodeURIComponent(corrected)}/influence-map`, {}),
  ]);
  return await post(
    `/corrective-results/${encodeURIComponent(result.result_id)}/source-use`,
    {},
  ) as unknown as SourceUseComparison;
}

/** Shared by every panel that drives this preview/confirm flow (DiagnosisRepair.tsx, ClaimVerification.tsx)
 * -- one copy of each small generic helper rather than one per caller. */
export function describeCorrectiveFlowError(error: unknown): string {
  return error instanceof Error ? error.message : "the request failed";
}

export function correctiveIdempotencyKey(prefix: string): string {
  const suffix = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}:${suffix}`;
}
