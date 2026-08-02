// clozn.token-workbench.v1 (GET) and its Milestone F action surface (POST fork / causal-trace /
// source-measure / mechanistic-diff, plus generic job status/cancel). See
// clozn/runs/token_workbench.py and clozn/runs/token_workbench_actions.py -- this module never
// re-derives their logic, only decodes what they already return.
//
// THE ONE RULE THIS FILE EXISTS TO ENFORCE: `loadTokenWorkbench` performs exactly one GET and computes
// nothing. Every other function here is an explicit POST/GET a caller chooses to make -- selecting a
// token must never reach any of them on its own (see useTokenWorkbench.ts).
//
// WHY THE FOUR CAPABILITIES STAY FOUR SEPARATE TYPES
// ----------------------------------------------------
// clozn.token-workbench.v1's own docstring is explicit: exact_fork/source_measurement/causal_trace/
// mechanistic_diff are never flattened into one shared status enum, because each rides a DIFFERENT
// artifact family's own vocabulary (a fork snapshot_state, an investigation section's persisted status,
// a plain readiness word, a binary structural gate). This module mirrors that: four capability
// interfaces, four action-result parsers, never a shared "status: string" grab-bag.
import type { CausalTraceEvidence } from "../features/observatory/layerApi";
import { parseCausalTraceEvidence } from "../features/observatory/layerApi";
import { parseForkArtifact, type ForkArtifact } from "./api";
import type { InfluenceMapJobState } from "./types";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function finiteNumber(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

// -------------------------------------------------------------------------------- mechanistic diff
/** The observational clozn.mechanistic-diff.v1 artifact. Its nested metric objects intentionally stay
 * JSON-shaped: the backend omits measurements it could not honestly compute, so the UI must not turn
 * an absent metric into a zero or a generic confidence score. */
export interface MechanisticDiffArtifact {
  schemaVersion: string;
  generatedAt: string;
  referenceModel: JsonRecord;
  candidateModel: JsonRecord;
  pairCompatibility: JsonRecord;
  continuation: { nPrompt: number; nCont: number };
  layersRequested: number[];
  positionsRequested: number[];
  layerCapture: JsonRecord[];
  positionMetrics: JsonRecord[];
  residualPoints: JsonRecord[];
  layerChange?: JsonRecord[];
}

export function parseMechanisticDiffArtifact(value: unknown): MechanisticDiffArtifact {
  const item = record(value);
  if (item.schema_version !== "clozn.mechanistic-diff.v1") {
    throw new Error("mechanistic diff result was not a clozn.mechanistic-diff.v1 artifact");
  }
  const continuation = record(item.continuation);
  const nPrompt = finiteNumber(continuation.n_prompt);
  const nCont = finiteNumber(continuation.n_cont);
  if (nPrompt == null || nCont == null) throw new Error("mechanistic diff omitted continuation counts");
  const integerArray = (raw: unknown): number[] => Array.isArray(raw)
    ? raw.flatMap((entry) => {
      const number = finiteNumber(entry);
      return number == null ? [] : [Math.trunc(number)];
    })
    : [];
  return {
    schemaVersion: String(item.schema_version),
    generatedAt: nonEmptyString(item.generated_at) ?? "",
    referenceModel: record(item.reference_model),
    candidateModel: record(item.candidate_model),
    pairCompatibility: record(item.pair_compatibility),
    continuation: { nPrompt: Math.trunc(nPrompt), nCont: Math.trunc(nCont) },
    layersRequested: integerArray(item.layers_requested),
    positionsRequested: integerArray(item.positions_requested),
    layerCapture: records(item.layer_capture),
    positionMetrics: records(item.position_metrics),
    residualPoints: records(item.residual_points),
    layerChange: item.layer_change === undefined ? undefined : records(item.layer_change),
  };
}

// ------------------------------------------------------------------------------------- workbench document
export interface WorkbenchRunSection {
  id: string;
  model?: string;
  substrate?: string;
  source?: string;
  parentRunId?: string;
  tokenCount?: number;
  modelSha256?: string;
  href?: string;
}

export interface WorkbenchTokenAlternative {
  piece: string;
  tokenId?: number;
  prob?: number;
}

export interface WorkbenchTokenSection {
  index: number;
  piece: string;
  tokenId?: number;
  prefixKept?: string;
  alternatives: WorkbenchTokenAlternative[];
}

/** The outer wrapper every one of context/comparison/readouts shares -- `state` is that SECTION's own
 * vocabulary (context: unavailable|failed|delivered_not_measured; comparison: unavailable|failed|
 * supported; readouts: unavailable|supported), never a value compared across sections. `raw` carries the
 * rest of that section's own fields verbatim (never reshaped) for the few call sites that need them --
 * see workbenchComparisonFindings / workbenchReadoutMeasurements below for the ones this Studio renders. */
export interface WorkbenchEvidenceSection {
  state: string;
  reason?: string;
  raw: JsonRecord;
}

export interface WorkbenchAction {
  method: "GET" | "POST" | "CLI";
  href: string;
  requestBody?: JsonRecord;
}

function workbenchAction(value: unknown): WorkbenchAction | undefined {
  const item = record(value);
  const method = item.method;
  if ((method !== "GET" && method !== "POST" && method !== "CLI") || typeof item.href !== "string" || !item.href) {
    return undefined;
  }
  return {
    method,
    href: item.href,
    requestBody: item.request_body && typeof item.request_body === "object"
      ? record(item.request_body)
      : undefined,
  };
}

export interface ExactForkCapability {
  available: boolean;
  snapshotState: string;
  reason?: string;
  action?: WorkbenchAction;
}

export interface SourceMeasurementCapability {
  available: boolean;
  status: string;
  reason?: string;
  action?: WorkbenchAction;
}

export interface CausalTraceCapability {
  available: boolean;
  status: string;
  reason?: string;
  action?: WorkbenchAction;
}

export interface MechanisticDiffCapability {
  available: boolean;
  reason: string;
  action?: WorkbenchAction;
}

export interface WorkbenchCapabilities {
  exactFork: ExactForkCapability;
  sourceMeasurement: SourceMeasurementCapability;
  causalTrace: CausalTraceCapability;
  mechanisticDiff: MechanisticDiffCapability;
}

export interface WorkbenchDocument {
  schemaVersion: "clozn.token-workbench.v1";
  runId: string;
  index: number;
  referenceRunId?: string;
  run: WorkbenchRunSection;
  token: WorkbenchTokenSection;
  context: WorkbenchEvidenceSection;
  comparison: WorkbenchEvidenceSection;
  readouts: WorkbenchEvidenceSection;
  capabilities: WorkbenchCapabilities;
}

function evidenceSection(value: unknown): WorkbenchEvidenceSection {
  const item = record(value);
  const state = typeof item.state === "string" && item.state ? item.state : "unavailable";
  return {
    state,
    reason: nonEmptyString(item.reason),
    raw: item,
  };
}

function runSection(value: unknown): WorkbenchRunSection {
  const item = record(value);
  return {
    id: String(item.id || ""),
    model: nonEmptyString(item.model),
    substrate: nonEmptyString(item.substrate),
    source: nonEmptyString(item.source),
    parentRunId: nonEmptyString(item.parent_run_id),
    tokenCount: finiteNumber(item.token_count),
    modelSha256: nonEmptyString(item.model_sha256),
    href: nonEmptyString(item.href),
  };
}

function tokenSection(value: unknown): WorkbenchTokenSection {
  const item = record(value);
  return {
    index: finiteNumber(item.index) ?? 0,
    piece: typeof item.piece === "string" ? item.piece : "",
    tokenId: finiteNumber(item.token_id),
    prefixKept: typeof item.prefix_kept === "string" ? item.prefix_kept : undefined,
    alternatives: records(item.alternatives).map((alt) => ({
      piece: typeof alt.piece === "string" ? alt.piece : "",
      tokenId: finiteNumber(alt.token_id),
      prob: finiteNumber(alt.prob),
    })),
  };
}

function capabilities(value: unknown): WorkbenchCapabilities {
  const item = record(value);
  const fork = record(item.exact_fork);
  const source = record(item.source_measurement);
  const causal = record(item.causal_trace);
  const diff = record(item.mechanistic_diff);
  return {
    exactFork: {
      available: fork.available === true,
      snapshotState: nonEmptyString(fork.snapshot_state) ?? "unknown",
      reason: nonEmptyString(fork.reason),
      action: workbenchAction(fork.action),
    },
    sourceMeasurement: {
      available: source.available === true,
      status: nonEmptyString(source.status) ?? "unknown",
      reason: nonEmptyString(source.reason),
      action: workbenchAction(source.action),
    },
    causalTrace: {
      available: causal.available === true,
      status: nonEmptyString(causal.status) ?? "unknown",
      reason: nonEmptyString(causal.reason),
      action: workbenchAction(causal.action),
    },
    mechanisticDiff: {
      available: diff.available === true,
      reason: nonEmptyString(diff.reason) ?? "mechanistic diff availability was not explained",
      action: workbenchAction(diff.action),
    },
  };
}

export class WorkbenchLoadError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "WorkbenchLoadError";
    this.status = status;
  }
}

/** GET /runs/<id>/tokens/<index>/workbench[?reference_run_id=]. THE central rule this whole module
 * exists to protect: this call triggers no computation on the server, and is the ONLY network call
 * `useTokenWorkbench` is allowed to fire when a token is selected (see that hook's own docstring). */
export async function loadTokenWorkbench(
  runId: string,
  index: number,
  referenceRunId?: string,
  signal?: AbortSignal,
): Promise<WorkbenchDocument> {
  const query = referenceRunId ? `?reference_run_id=${encodeURIComponent(referenceRunId)}` : "";
  const response = await fetch(
    `/runs/${encodeURIComponent(runId)}/tokens/${index}/workbench${query}`,
    { signal },
  );
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The HTTP status below remains authoritative for a non-JSON body.
  }
  if (!response.ok) {
    throw new WorkbenchLoadError(
      String(body.detail || body.error || `token workbench request failed (${response.status})`),
      response.status,
    );
  }
  return {
    schemaVersion: "clozn.token-workbench.v1",
    runId: String(body.run_id || runId),
    index: finiteNumber(body.index) ?? index,
    referenceRunId: nonEmptyString(body.reference_run_id),
    run: runSection(record(body.sections).run),
    token: tokenSection(record(body.sections).token),
    context: evidenceSection(record(body.sections).context),
    comparison: evidenceSection(record(body.sections).comparison),
    readouts: evidenceSection(record(body.sections).readouts),
    capabilities: capabilities(body.capabilities),
  };
}

/** `comparison.raw.comparison.findings[].summary` -- clozn.run-diff.v1's own plain-English findings,
 * shown verbatim (never re-derived; see clozn/analysis/run_diff.py's `_findings_from` docstring: each is
 * independently classified from a named dimension, capped at "observed" without a replay). */
export function workbenchComparisonFindings(section: WorkbenchEvidenceSection): string[] {
  const comparison = record(section.raw.comparison);
  return records(comparison.findings).flatMap((finding) => {
    const summary = finding.summary;
    return typeof summary === "string" && summary ? [summary] : [];
  });
}

export function workbenchComparisonDifferenceCount(section: WorkbenchEvidenceSection): number | undefined {
  const comparison = record(section.raw.comparison);
  return Array.isArray(comparison.differences) ? comparison.differences.length : undefined;
}

export interface WorkbenchReadoutMeasurements {
  confidence?: number;
  logprob?: number;
  topkEntropy?: number;
}

export function workbenchReadoutMeasurements(section: WorkbenchEvidenceSection): WorkbenchReadoutMeasurements {
  const measurements = record(section.raw.measurements);
  return {
    confidence: finiteNumber(measurements.confidence),
    logprob: finiteNumber(measurements.logprob),
    topkEntropy: finiteNumber(measurements.topk_entropy),
  };
}

export interface WorkbenchReadout {
  provider: string;
  providerType?: string;
  readoutKind?: string;
  layer?: number;
  topReadouts: Array<{ label: string; score: number }>;
}

/** Already position-filtered by the server to this exact token index -- unlike the full run's
 * `trace.workspace_readouts`, no client-side filtering is needed here. */
export function workbenchReadouts(section: WorkbenchEvidenceSection): WorkbenchReadout[] {
  return records(section.raw.workspace_readouts).flatMap((item) => {
    if (typeof item.provider !== "string" || !item.provider) return [];
    const topReadouts = records(item.top_readouts).flatMap((readout) => {
      const score = finiteNumber(readout.score);
      return typeof readout.label === "string" && score != null ? [{ label: readout.label, score }] : [];
    });
    return [{
      provider: item.provider,
      providerType: nonEmptyString(item.provider_type),
      readoutKind: nonEmptyString(item.readout_kind),
      layer: finiteNumber(item.layer),
      topReadouts,
    }];
  });
}

// ------------------------------------------------------------------------------------------ action jobs
// clozn.influence-map-job.v1, additively generalized this milestone with `kind` and an optional `result`
// (see clozn/server/influence_jobs.py's module docstring) -- ONE job lifecycle vocabulary shared by every
// bounded async action in the gateway on purpose (unlike the four capabilities above, which each carry a
// genuinely different evidence claim, "queued/running/persisting/cancelling/completed/failed/cancelled"
// describes the SAME thing -- a background call's own progress -- no matter which action started it).
export interface WorkbenchJob {
  schemaVersion: string;
  jobId: string;
  runId: string;
  kind: string;
  state: InfluenceMapJobState;
  progress: { phase: string; completedUnits: number; totalUnits: number; percent: number };
  cancelRequested: boolean;
  cancellable: boolean;
  cached: boolean;
  cancelAccepted?: boolean;
  error?: { code?: string; message: string };
  /** The action's own artifact, when its worker chose to attach one (fork/causal-trace do; source-
   * measure persists to the run itself instead -- see clozn/runs/token_workbench_actions.py). Raw and
   * unparsed here on purpose: each action's own result parser (below) decodes it, never this module. */
  result?: unknown;
}

const JOB_STATES: readonly InfluenceMapJobState[] =
  ["queued", "running", "persisting", "cancelling", "completed", "failed", "cancelled"];

export const JOB_TERMINAL_STATES: readonly InfluenceMapJobState[] = ["completed", "failed", "cancelled"];

function parseJob(value: unknown): WorkbenchJob {
  const body = record(value);
  const progress = record(body.progress);
  const error = record(body.error);
  const state = JOB_STATES.includes(body.state as InfluenceMapJobState)
    ? (body.state as InfluenceMapJobState)
    : "failed";
  return {
    schemaVersion: nonEmptyString(body.schema_version) ?? "clozn.influence-map-job.v1",
    jobId: String(body.job_id || ""),
    runId: String(body.run_id || ""),
    kind: nonEmptyString(body.kind) ?? "unknown",
    state,
    progress: {
      phase: nonEmptyString(progress.phase) ?? "unknown",
      completedUnits: finiteNumber(progress.completed_units) ?? 0,
      totalUnits: finiteNumber(progress.total_units) ?? 0,
      percent: finiteNumber(progress.percent) ?? 0,
    },
    cancelRequested: body.cancel_requested === true,
    cancellable: body.cancellable === true,
    cached: body.cached === true,
    cancelAccepted: typeof body.cancel_accepted === "boolean" ? body.cancel_accepted : undefined,
    error: typeof error.message === "string"
      ? { code: nonEmptyString(error.code), message: error.message }
      : undefined,
    result: "result" in body ? body.result : undefined,
  };
}

// ------------------------------------------------------------------------------------------- the actions
/** Every one of the four POST actions resolves to exactly this three-shape envelope (see
 * clozn/runs/token_workbench_actions.py's module docstring) -- `TArtifact` is that action's OWN artifact
 * type, decoded by its own parser below, never a shared shape. */
export type WorkbenchActionResult<TArtifact> =
  | { outcome: "cached"; artifact: TArtifact }
  | { outcome: "job"; job: WorkbenchJob }
  | {
      outcome: "unavailable";
      reason: { code: string; message: string };
      /** Present only for mechanistic-diff's typed refusal -- clozn.pair-compatibility.v1, verbatim. */
      pairCompatibility?: JsonRecord;
    };

export class WorkbenchActionError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "WorkbenchActionError";
    this.status = status;
  }
}

async function postAction(
  runId: string,
  index: number,
  path: string,
  body: JsonRecord,
  signal?: AbortSignal,
): Promise<JsonRecord & { __status: number }> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/tokens/${index}/${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  let parsed: JsonRecord = {};
  try {
    parsed = record(await response.json());
  } catch {
    // The HTTP status remains authoritative for a non-JSON body.
  }
  const isTypedEnvelope = parsed.outcome === "cached" || parsed.outcome === "job" || parsed.outcome === "unavailable";
  if (!response.ok && !isTypedEnvelope) {
    throw new WorkbenchActionError(
      String(parsed.error || `action request failed (${response.status})`),
      response.status,
    );
  }
  return { ...parsed, __status: response.status };
}

function parseEnvelope<TArtifact>(
  body: JsonRecord,
  parseArtifact: (value: unknown) => TArtifact,
): WorkbenchActionResult<TArtifact> {
  if (body.outcome === "cached") return { outcome: "cached", artifact: parseArtifact(body.artifact) };
  if (body.outcome === "job") return { outcome: "job", job: parseJob(body.job) };
  const reason = record(body.reason);
  return {
    outcome: "unavailable",
    reason: {
      code: nonEmptyString(reason.code) ?? "unavailable",
      message: nonEmptyString(reason.message) ?? "this action is unavailable",
    },
    pairCompatibility: body.pair_compatibility ? record(body.pair_compatibility) : undefined,
  };
}

export function postForkAction(
  runId: string,
  index: number,
  token: string,
  tokenId: number | undefined,
  signal?: AbortSignal,
): Promise<WorkbenchActionResult<ForkArtifact>> {
  const body = tokenId == null ? { token } : { token, token_id: tokenId };
  return postAction(runId, index, "fork", body, signal).then((raw) => parseEnvelope(raw, parseForkArtifact));
}

export function postCausalTraceAction(
  runId: string,
  index: number,
  options: { refresh?: boolean } = {},
  signal?: AbortSignal,
): Promise<WorkbenchActionResult<CausalTraceEvidence>> {
  const body = options.refresh ? { refresh: true } : {};
  return postAction(runId, index, "causal-trace", body, signal).then((raw) =>
    parseEnvelope(raw, (value) => parseCausalTraceEvidence(record(record(value).result))));
}

/** source-measure's artifact is the run's own `influence_map` document (or, for a job, nothing -- the
 * worker persists straight to the run rather than attaching a job result; see
 * clozn/runs/token_workbench_actions.py's `source_measure_job_worker`). This module intentionally does
 * NOT re-parse it into `SourceReading`/`TokenSourceReading` -- that whole pipeline already lives in
 * data/api.ts's `loadRunInspection`, and a completed source-measure action's caller reloads the run
 * through that same path rather than this module growing a second copy of it. */
export function postSourceMeasureAction(
  runId: string,
  index: number,
  options: { refresh?: boolean } = {},
  signal?: AbortSignal,
): Promise<WorkbenchActionResult<JsonRecord>> {
  const body = options.refresh ? { refresh: true } : {};
  return postAction(runId, index, "source-measure", body, signal).then((raw) => parseEnvelope(raw, record));
}

/** The mechanistic action returns the observational diff artifact after a bounded managed job. Cached
 * responses carry the token-workbench cache entry around that artifact, just like causal-trace. */
export function postMechanisticDiffAction(
  runId: string,
  index: number,
  referenceRunId: string,
  signal?: AbortSignal,
): Promise<WorkbenchActionResult<MechanisticDiffArtifact>> {
  return postAction(runId, index, "mechanistic-diff", { reference_run_id: referenceRunId }, signal)
    .then((raw) => parseEnvelope(raw, (value) => {
      const entry = record(value);
      return parseMechanisticDiffArtifact(
        entry.action === "mechanistic_diff" && entry.result ? entry.result : value,
      );
    }));
}

export async function loadWorkbenchJob(
  runId: string,
  index: number,
  jobId: string,
  signal?: AbortSignal,
): Promise<WorkbenchJob> {
  const response = await fetch(
    `/runs/${encodeURIComponent(runId)}/tokens/${index}/jobs/${encodeURIComponent(jobId)}`,
    { signal },
  );
  const body = record(await response.json().catch(() => ({})));
  if (!response.ok) {
    throw new WorkbenchActionError(String(body.error || `job status request failed (${response.status})`), response.status);
  }
  return parseJob(body);
}

export async function cancelWorkbenchJob(
  runId: string,
  index: number,
  jobId: string,
  signal?: AbortSignal,
): Promise<WorkbenchJob> {
  const response = await fetch(
    `/runs/${encodeURIComponent(runId)}/tokens/${index}/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST", headers: { "content-type": "application/json" }, body: "{}", signal },
  );
  const body = record(await response.json().catch(() => ({})));
  if (!response.ok) {
    throw new WorkbenchActionError(String(body.error || `job cancel request failed (${response.status})`), response.status);
  }
  return parseJob(body);
}
