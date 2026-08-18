/** Client for the durable checkpoint ledger and run-scoped pin action.
 *
 * The gateway never returns checkpoint bytes here. Studio only needs the manifest metadata and the
 * preview byte estimate; replay/import remains a CLI/advanced API operation.
 */

export interface SnapshotManifest {
  schemaVersion: "clozn.pinned-checkpoint.v1";
  pinId: string;
  runId: string;
  pinnedAt: string;
  note?: string;
  state: {
    nTokens?: number;
    nPast?: number;
    promptTokens?: number;
    causal?: boolean;
    hasSampler?: boolean;
  };
  identity: {
    modelSha256?: string;
    architecture?: string;
    nCtx?: number;
    backend?: string;
  };
  blob: {
    sha256?: string;
    kvBytes?: number;
    envelopeBytes?: number;
  };
}

export interface SnapshotListDocument {
  schemaVersion: "clozn.pinned-checkpoint-list.v1";
  snapshots: SnapshotManifest[];
}

export interface SnapshotPinPreview {
  ok: true;
  preview: true;
  runId: string;
  sizeBytes?: number;
  envelopeBytes?: number;
}

export interface SnapshotPinResult {
  ok: true;
  manifest: SnapshotManifest;
}

export interface SnapshotUnpinResult {
  ok: boolean;
  action: "unpin";
  runId: string;
  cascade: boolean;
  blobCleanup?: { status?: string; sha256?: string };
}

export class SnapshotRequestError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly details?: Record<string, unknown>;

  constructor(message: string, status: number, code?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "SnapshotRequestError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function parseManifest(value: unknown): SnapshotManifest {
  const raw = record(value);
  const state = record(raw.state);
  const identity = record(raw.identity);
  const blob = record(raw.blob);
  const schemaVersion = raw.schema_version;
  const pinId = raw.pin_id;
  const runId = raw.run_id;
  if (schemaVersion !== "clozn.pinned-checkpoint.v1" || typeof pinId !== "string" || !pinId
      || typeof runId !== "string" || !runId) {
    throw new SnapshotRequestError("the gateway returned an invalid pinned checkpoint manifest", 502, "invalid_snapshot_manifest");
  }
  return {
    schemaVersion,
    pinId,
    runId,
    pinnedAt: typeof raw.pinned_at === "string" ? raw.pinned_at : "unknown",
    note: optionalString(raw.note),
    state: {
      nTokens: optionalNumber(state.n_tokens),
      nPast: optionalNumber(state.n_past),
      promptTokens: optionalNumber(state.prompt_tokens),
      causal: typeof state.causal === "boolean" ? state.causal : undefined,
      hasSampler: typeof state.has_sampler === "boolean" ? state.has_sampler : undefined,
    },
    identity: {
      modelSha256: optionalString(identity.model_sha256),
      architecture: optionalString(identity.architecture),
      nCtx: optionalNumber(identity.n_ctx),
      backend: optionalString(identity.backend),
    },
    blob: {
      sha256: optionalString(blob.sha256),
      kvBytes: optionalNumber(blob.kv_bytes),
      envelopeBytes: optionalNumber(blob.envelope_bytes),
    },
  };
}

async function jsonRequest(url: string, init?: RequestInit): Promise<Record<string, unknown>> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    throw new SnapshotRequestError(error instanceof Error ? error.message : "snapshot request failed", 0);
  }
  let body: Record<string, unknown> = {};
  try { body = record(await response.json()); } catch { /* status below remains authoritative */ }
  if (!response.ok) {
    throw new SnapshotRequestError(
      typeof body.error === "string" ? body.error : `snapshot request failed (${response.status})`,
      response.status,
      optionalString(body.code),
      body,
    );
  }
  return body;
}

export async function loadSnapshots(signal?: AbortSignal): Promise<SnapshotListDocument> {
  const body = await jsonRequest("/snapshots", { signal });
  if (body.schema_version !== "clozn.pinned-checkpoint-list.v1" || !Array.isArray(body.snapshots)) {
    throw new SnapshotRequestError("the gateway returned an invalid snapshot list", 502, "invalid_snapshot_list");
  }
  return {
    schemaVersion: "clozn.pinned-checkpoint-list.v1",
    snapshots: body.snapshots.map(parseManifest),
  };
}

async function pinRequest(runId: string, note: string, preview: boolean, signal?: AbortSignal): Promise<Record<string, unknown>> {
  return jsonRequest(`/runs/${encodeURIComponent(runId)}/snapshot/pin`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ note: note || undefined, preview }),
    signal,
  });
}

export async function previewSnapshot(runId: string, note = "", signal?: AbortSignal): Promise<SnapshotPinPreview> {
  const body = await pinRequest(runId, note, true, signal);
  return {
    ok: true,
    preview: true,
    runId,
    sizeBytes: optionalNumber(body.size_bytes),
    envelopeBytes: optionalNumber(body.envelope_bytes),
  };
}

export async function pinSnapshot(runId: string, note = "", signal?: AbortSignal): Promise<SnapshotPinResult> {
  const body = await pinRequest(runId, note, false, signal);
  const manifest = parseManifest(body.manifest);
  return { ok: true, manifest };
}

export async function unpinSnapshot(runId: string, cascade = false, signal?: AbortSignal): Promise<SnapshotUnpinResult> {
  const body = await jsonRequest(`/snapshots/${encodeURIComponent(runId)}/unpin`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ cascade }),
    signal,
  });
  return {
    ok: body.ok === true,
    action: "unpin",
    runId,
    cascade: body.cascade === true,
    blobCleanup: record(body.blob_cleanup) as SnapshotUnpinResult["blobCleanup"],
  };
}

export function formatSnapshotBytes(value?: number): string {
  if (value == null || value < 0) return "unknown size";
  if (value < 1024) return `${Math.round(value)} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(1)} GB`;
}
