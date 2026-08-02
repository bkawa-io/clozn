import { num, record, records, str, type JsonRecord } from "./received-context";

/** Client for C3's plan -> execute -> poll controlled investigation experiment. Planning is a pure
 * POST and never generates. Execution is a separate explicit POST that returns a bounded job. */

export type ExperimentInterventionKind =
  | "remove_span"
  | "replace_span_neutral"
  | "omit_source"
  | "sampler_change";

export interface ExperimentIntervention {
  kind: ExperimentInterventionKind;
  spanAddressId?: string;
  sourceId?: string;
}

export type ExperimentPhase = "refused" | "planned" | "completed" | "failed";

export interface ExperimentReason {
  code: string;
  message: string;
}

export interface ExperimentDocument {
  schemaVersion: string;
  experimentId: string;
  runId: string;
  generatedAt: string;
  phase: ExperimentPhase;
  intervention: ExperimentIntervention;
  eligibility: { state: "eligible" | "refused"; reason?: ExperimentReason };
  plan?: {
    armOrder: string[];
    resolvedKind: string;
    spans: Array<{ messageIndex?: number; start?: number; end?: number; spanAddressId?: string }>;
  };
  arms?: {
    baseline?: ExperimentArm;
    noOpReplay?: ExperimentArm;
    treatment?: ExperimentArm;
    randomEqualEffectControl?: ExperimentArm | { available: false; reason: string };
  };
  analysis?: {
    instrumentSane?: boolean;
    effectSpecific?: boolean;
    reasons: string[];
  };
  observed?: {
    treatmentReplyDiffersFromBaseline?: boolean;
    randomControlReplyDiffersFromBaseline?: boolean;
    note?: string;
  };
  causalClaim?: { licensed: boolean; statement: string };
  error?: { stage?: string; message: string };
}

export interface ExperimentArm {
  available?: boolean;
  reason?: string;
  runId?: string;
  replySha256?: string;
  matchesBaseline?: boolean;
}

export type ExperimentJobState =
  | "queued"
  | "running"
  | "persisting"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export interface ExperimentJob {
  schemaVersion: string;
  jobId: string;
  runId: string;
  state: ExperimentJobState;
  progress: { phase: string; completedUnits: number; totalUnits: number; percent: number };
  cancellable: boolean;
  error?: { code?: string; message?: string };
  result?: ExperimentDocument;
}

export class InvestigationExperimentError extends Error {
  document?: ExperimentDocument;
}

const KINDS = new Set<ExperimentInterventionKind>([
  "remove_span", "replace_span_neutral", "omit_source", "sampler_change",
]);
const PHASES = new Set<ExperimentPhase>(["refused", "planned", "completed", "failed"]);
const JOB_STATES = new Set<ExperimentJobState>([
  "queued", "running", "persisting", "cancelling", "completed", "failed", "cancelled",
]);

function kind(value: unknown): ExperimentInterventionKind {
  return typeof value === "string" && KINDS.has(value as ExperimentInterventionKind)
    ? value as ExperimentInterventionKind
    : "remove_span";
}

function phase(value: unknown): ExperimentPhase {
  return typeof value === "string" && PHASES.has(value as ExperimentPhase)
    ? value as ExperimentPhase
    : "failed";
}

function jobState(value: unknown): ExperimentJobState {
  return typeof value === "string" && JOB_STATES.has(value as ExperimentJobState)
    ? value as ExperimentJobState
    : "failed";
}

function bool(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

function intervention(raw: unknown): ExperimentIntervention {
  const item = record(raw);
  const result: ExperimentIntervention = { kind: kind(item.kind) };
  const span = str(item.span_address_id);
  const source = str(item.source_id);
  if (span) result.spanAddressId = span;
  if (source) result.sourceId = source;
  return result;
}

function reason(raw: unknown): ExperimentReason | undefined {
  const item = record(raw);
  const message = str(item.message);
  return message ? { code: str(item.code) ?? "experiment_refused", message } : undefined;
}

function parseArm(raw: unknown): ExperimentArm | undefined {
  const item = record(raw);
  if (bool(item.available) === false) return { available: false, reason: str(item.reason), matchesBaseline: undefined };
  if (!Object.keys(item).length) return undefined;
  return {
    runId: str(item.run_id),
    replySha256: str(item.reply_sha256),
    matchesBaseline: bool(item.matches_baseline),
  };
}

function parseDocument(raw: unknown, requestedRunId: string): ExperimentDocument {
  const item = record(raw);
  const eligibility = record(item.eligibility);
  const plan = record(item.plan);
  const resolved = record(plan.resolved);
  const analysis = record(item.analysis);
  const observed = record(item.observed);
  const causal = record(item.causal_claim);
  const arms = record(item.arms);
  return {
    schemaVersion: str(item.schema_version) ?? "",
    experimentId: str(item.experiment_id) ?? "",
    runId: str(item.run_id) ?? requestedRunId,
    generatedAt: str(item.generated_at) ?? "",
    phase: phase(item.phase),
    intervention: intervention(item.intervention),
    eligibility: {
      state: eligibility.state === "eligible" ? "eligible" : "refused",
      reason: reason(eligibility.reason),
    },
    plan: Object.keys(plan).length ? {
      armOrder: stringArray(plan.arm_order),
      resolvedKind: str(resolved.kind) ?? "",
      spans: records(resolved.spans).map((span) => ({
        messageIndex: num(span.message_index),
        start: num(span.start),
        end: num(span.end),
        spanAddressId: str(span.span_address_id),
      })),
    } : undefined,
    arms: Object.keys(arms).length ? {
      baseline: parseArm(arms.baseline),
      noOpReplay: parseArm(arms.no_op_replay),
      treatment: parseArm(arms.treatment),
      randomEqualEffectControl: parseArm(arms.random_equal_effect_control),
    } : undefined,
    analysis: Object.keys(analysis).length ? {
      instrumentSane: bool(analysis.instrument_sane),
      effectSpecific: bool(analysis.effect_specific),
      reasons: stringArray(analysis.reasons),
    } : undefined,
    observed: Object.keys(observed).length ? {
      treatmentReplyDiffersFromBaseline: bool(observed.treatment_reply_differs_from_baseline),
      randomControlReplyDiffersFromBaseline: bool(observed.random_control_reply_differs_from_baseline),
      note: str(observed.note),
    } : undefined,
    causalClaim: causal.licensed === undefined && !str(causal.statement) ? undefined : {
      licensed: bool(causal.licensed) ?? false,
      statement: str(causal.statement) ?? "No causal statement was returned.",
    },
    error: str(record(item.error).message) ? {
      stage: str(record(item.error).stage),
      message: str(record(item.error).message) as string,
    } : undefined,
  };
}

function parseProgress(raw: unknown) {
  const item = record(raw);
  return {
    phase: str(item.phase) ?? "queued",
    completedUnits: num(item.completed_units) ?? 0,
    totalUnits: num(item.total_units) ?? 0,
    percent: num(item.percent) ?? 0,
  };
}

function parseJob(raw: unknown, requestedRunId: string): ExperimentJob {
  const item = record(raw);
  return {
    schemaVersion: str(item.schema_version) ?? "",
    jobId: str(item.job_id) ?? "",
    runId: str(item.run_id) ?? requestedRunId,
    state: jobState(item.state),
    progress: parseProgress(item.progress),
    cancellable: bool(item.cancellable) ?? false,
    error: str(record(item.error).message) ? {
      code: str(record(item.error).code), message: str(record(item.error).message),
    } : undefined,
    result: item.result ? parseDocument(item.result, requestedRunId) : undefined,
  };
}

async function request(path: string, init: RequestInit | undefined, runId: string, signal?: AbortSignal) {
  const response = await fetch(path, { ...init, signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new InvestigationExperimentError(
      typeof record(body).error === "string" ? record(body).error as string : `Request failed (${response.status})`,
    );
    if (record(body).schema_version === "clozn.investigation-experiment.v1") error.document = parseDocument(body, runId);
    throw error;
  }
  return body;
}

export async function planInvestigationExperiment(
  runId: string,
  change: ExperimentIntervention,
  signal?: AbortSignal,
): Promise<ExperimentDocument> {
  return parseDocument(await request(`/runs/${encodeURIComponent(runId)}/investigation-experiment/plan`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ intervention: toWire(change) }),
  }, runId, signal), runId);
}

export async function startInvestigationExperiment(
  runId: string,
  change: ExperimentIntervention,
  signal?: AbortSignal,
): Promise<ExperimentJob> {
  return parseJob(await request(`/runs/${encodeURIComponent(runId)}/investigation-experiment`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ intervention: toWire(change) }),
  }, runId, signal), runId);
}

export async function loadInvestigationExperimentJob(runId: string, jobId: string, signal?: AbortSignal): Promise<ExperimentJob> {
  return parseJob(await request(`/runs/${encodeURIComponent(runId)}/investigation-experiment/jobs/${encodeURIComponent(jobId)}`, undefined, runId, signal), runId);
}

export async function cancelInvestigationExperiment(runId: string, jobId: string, signal?: AbortSignal): Promise<ExperimentJob> {
  return parseJob(await request(`/runs/${encodeURIComponent(runId)}/investigation-experiment/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST", headers: { "content-type": "application/json" }, body: "{}",
  }, runId, signal), runId);
}

function toWire(change: ExperimentIntervention): JsonRecord {
  const result: JsonRecord = { kind: change.kind };
  if (change.spanAddressId) result.span_address_id = change.spanAddressId;
  if (change.sourceId) result.source_id = change.sourceId;
  return result;
}

export function describeInvestigationExperimentError(error: unknown): string {
  return error instanceof Error ? error.message : "the investigation experiment request failed";
}
