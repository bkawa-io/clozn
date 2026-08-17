export type MinimalContextCertificate = "BEST_VERIFIED" | "INCLUSION_MINIMUM";
export type MinimalContextClassification = "preserves" | "diverged" | "unknown";
export type MinimalContextDisposition = "reused" | "executed" | "not_executed";

export interface MinimalContextProgress {
  phase: string;
  completedUnits: number;
  totalUnits: number;
  percent: number;
  bestRetainedSourceCount?: number;
  certificateCandidateKind?: MinimalContextCertificate;
}

export interface MinimalContextEvidenceRef {
  disposition: MinimalContextDisposition;
  experiment_id?: string;
  arm_id?: string;
  observation_id?: string;
  observation_status?: string;
}

export interface MinimalContextTrial {
  ordinal: number;
  stage: string;
  retained_ids: string[];
  cost: number;
  classification: MinimalContextClassification;
  disposition: MinimalContextDisposition;
  experiment_id?: string | null;
  arm_id?: string | null;
  observation_id?: string | null;
  observation_status?: string | null;
  evidence?: MinimalContextEvidenceRef | null;
  batch_id?: number | null;
  parent_retained_ids?: string[];
}

export interface MinimalContextTrajectoryEntry {
  counterfactual_probe_count: number;
  retained_ids: string[];
  cost: number;
  retained_unit_count: number;
  stage: string;
}

export interface MinimalContextBudget {
  max_new_executions: number;
  used_new_executions: number;
  reused_observation_count: number;
  exhausted: boolean;
}

export interface MinimalContextInclusionCheck {
  attempted: boolean;
  complete: boolean;
  tested_child_count: number;
  total_child_count: number;
  all_children_failed: boolean;
}

export interface MinimalContextSourceUnit {
  source_id: string;
  message_index: number;
  role: string;
  unicode_range: [number, number];
  byte_range?: [number, number];
  source_kind?: string;
  derivation?: string;
  provenance_kind?: string;
  client_source_id?: string;
  source_label?: string;
  parent_source_id?: string;
}

export interface MinimalContextResult {
  schema_version: "clozn.minimal-context-search-result.v1";
  search_id: string;
  status: string;
  search_status?: string | null;
  reason?: string | null;
  reason_code?: string | null;
  base_execution_fingerprint: string;
  universe: {
    universe_id?: string;
    source_ids: string[];
    source_count?: number;
    [key: string]: unknown;
  };
  objective: {
    kind: string;
    version: string;
    [key: string]: unknown;
  };
  control_observation_id?: string | null;
  trials: MinimalContextTrial[];
  trajectory: MinimalContextTrajectoryEntry[];
  best?: {
    retained_source_ids: string[];
    removed_source_ids: string[];
    rendered_prompt_token_cost: number;
    experiment_id?: string | null;
    arm_id?: string | null;
    observation_id?: string | null;
    observation_status?: string | null;
  } | null;
  certificate?: MinimalContextCertificate | null;
  policy: {
    kind: string;
    version: string;
    attempt_inclusion_check: boolean;
    [key: string]: unknown;
  };
  budget: MinimalContextBudget;
  inclusion_check: MinimalContextInclusionCheck;
}

export interface MinimalContextJob {
  schemaVersion: string;
  jobId: string;
  runId: string;
  kind: string;
  state: "queued" | "running" | "persisting" | "cancelling" | "completed" | "failed" | "cancelled";
  progress: MinimalContextProgress;
  cancelRequested: boolean;
  cancellable: boolean;
  cached: boolean;
  cancelAccepted?: boolean;
  error?: { code?: string; message: string };
  result?: MinimalContextResult;
}

export interface MinimalContextRunDetail {
  id: string;
  messages?: Array<{ role?: string; content?: string }>;
  context_units?: {
    protected_message_indices?: number[];
    units?: MinimalContextSourceUnit[];
    default_source_ids?: string[];
  };
  context_receipt?: {
    delivered?: Array<{ segment_id?: string; source_label?: string; sources?: MinimalContextSourceUnit[] }>;
    assembled?: Array<{ segment_id?: string; source_label?: string; sources?: MinimalContextSourceUnit[] }>;
  };
  trace?: { tokens?: unknown[]; steps?: unknown[] };
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function parseResult(value: unknown): MinimalContextResult | undefined {
  const body = record(value);
  if (body.schema_version !== "clozn.minimal-context-search-result.v1"
      || typeof body.search_id !== "string" || !body.search_id) return undefined;
  return value as MinimalContextResult;
}

function parseJob(value: unknown): MinimalContextJob {
  const body = record(value);
  const progress = record(body.progress);
  const error = record(body.error);
  return {
    schemaVersion: String(body.schema_version ?? "clozn.influence-map-job.v1"),
    jobId: String(body.job_id ?? ""),
    runId: String(body.run_id ?? ""),
    kind: String(body.kind ?? "minimal_context"),
    state: String(body.state ?? "failed") as MinimalContextJob["state"],
    progress: {
      phase: String(progress.phase ?? "queued"),
      completedUnits: Number(progress.completed_units) || 0,
      totalUnits: Number(progress.total_units) || 0,
      percent: Number(progress.percent) || 0,
      bestRetainedSourceCount: typeof progress.best_retained_source_count === "number"
        ? progress.best_retained_source_count : undefined,
      certificateCandidateKind: typeof progress.certificate_candidate_kind === "string"
        ? progress.certificate_candidate_kind as MinimalContextCertificate : undefined,
    },
    cancelRequested: body.cancel_requested === true,
    cancellable: body.cancellable === true,
    cached: body.cached === true,
    cancelAccepted: typeof body.cancel_accepted === "boolean" ? body.cancel_accepted : undefined,
    error: typeof error.message === "string" ? {
      code: typeof error.code === "string" ? error.code : undefined,
      message: error.message,
    } : undefined,
    result: parseResult(body.result),
  };
}

async function requestJson(url: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(url, init);
  let body: unknown = {};
  try { body = await response.json(); } catch { /* preserve HTTP error below */ }
  if (!response.ok) {
    const error = record(body);
    const failure = new Error(String(error.error ?? `Minimal Context request failed (${response.status})`));
    (failure as Error & { code?: string; status?: number }).code = typeof error.code === "string" ? error.code : undefined;
    (failure as Error & { code?: string; status?: number }).status = response.status;
    throw failure;
  }
  return body;
}

export async function loadMinimalContextRun(runId: string, signal?: AbortSignal): Promise<MinimalContextRunDetail | null> {
  try { return await requestJson(`/runs/${encodeURIComponent(runId)}`, { signal }) as MinimalContextRunDetail; }
  catch { return null; }
}

export async function startMinimalContextJob(
  runId: string,
  request: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<MinimalContextJob> {
  return parseJob(await requestJson(`/runs/${encodeURIComponent(runId)}/minimal-context/jobs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
    signal,
  }));
}

export async function pollMinimalContextJob(runId: string, jobId: string, signal?: AbortSignal): Promise<MinimalContextJob> {
  return parseJob(await requestJson(`/runs/${encodeURIComponent(runId)}/minimal-context/jobs/${encodeURIComponent(jobId)}`, { signal }));
}

export async function cancelMinimalContextJob(runId: string, jobId: string, signal?: AbortSignal): Promise<MinimalContextJob> {
  return parseJob(await requestJson(`/runs/${encodeURIComponent(runId)}/minimal-context/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
    signal,
  }));
}

export async function branchMinimalContextWinner(
  runId: string,
  references: { experiment_id: string; arm_id: string; observation_id: string },
  signal?: AbortSignal,
): Promise<unknown> {
  return requestJson(`/runs/${encodeURIComponent(runId)}/minimal-context/branch`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(references),
    signal,
  });
}
