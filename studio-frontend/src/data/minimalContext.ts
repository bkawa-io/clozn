export type MinimalContextCriterion = "exact_recorded_output";
export type MinimalContextCertificate = "inclusion_minimum" | "best_verified";

export interface MinimalContextProgress {
  phase: string;
  completedUnits: number;
  totalUnits: number;
  percent: number;
  bestRetainedSourceCount?: number;
  certificateCandidateKind?: MinimalContextCertificate;
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

export interface MinimalContextSummary {
  result_id: string;
  preservation_kind: MinimalContextCriterion;
  source_count: number;
  retained_source_count?: number;
  certificate_kind?: MinimalContextCertificate;
  status: string;
  universe_id?: string;
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
  schema_version: string;
  run_id: string;
  result_id: string;
  status: string;
  source_universe: {
    source_ids: string[];
    source_count: number;
    context_units_manifest_sha256?: string;
    search_universe_id?: string;
  };
  preservation: { kind: MinimalContextCriterion; tolerance_nats?: number };
  candidate?: {
    retained_source_ids: string[];
    removed_source_ids: string[];
    retained_source_count: number;
    within_tolerance: boolean;
  };
  certificate?: {
    kind: MinimalContextCertificate;
    candidate_retained_source_count: number;
    global_minimality: "proven" | "not_proven";
    inclusion_minimality: "proven" | "not_proven";
  };
  coverage?: {
    lower_cardinalities?: Array<{
      retained_source_count: number;
      candidate_count: number;
      tested_count: number;
      preserving_count: number;
      complete: boolean;
    }>;
    smaller_candidate_count: number;
    smaller_tested_count: number;
    smaller_remaining_count: number;
  };
  budget?: { total_new_probes?: number; search_probe_budget?: number; certification_probe_budget?: number };
  cache_binding?: Record<string, unknown>;
  error?: string;
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
    result: record(body.result).result_id ? body.result as MinimalContextResult : undefined,
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

export async function listMinimalContextResults(runId: string, signal?: AbortSignal): Promise<MinimalContextSummary[]> {
  try {
    const body = record(await requestJson(`/runs/${encodeURIComponent(runId)}/minimal-context`, { signal }));
    return Array.isArray(body.results) ? body.results as MinimalContextSummary[] : [];
  } catch { return []; }
}

export async function loadMinimalContextResult(runId: string, resultId: string, signal?: AbortSignal): Promise<MinimalContextResult | null> {
  try {
    return await requestJson(`/runs/${encodeURIComponent(runId)}/minimal-context/${encodeURIComponent(resultId)}`, { signal }) as MinimalContextResult;
  } catch { return null; }
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
