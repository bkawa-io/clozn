import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type EffectsSearchExecutionPolicy =
  | "trusted_scalar_only"
  | "native_screen_then_scalar_refine";
export type EffectsSearchEvidenceStatus =
  | "trusted"
  | "screening_only"
  | "unavailable"
  | "skipped";
export type EffectsSearchJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "persisting"
  | "completed"
  | "failed"
  | "cancelled";

export interface EffectsSearchBudget {
  maxCoarseArms: number;
  maxFineArms: number;
  maxJointArms: number;
}

export interface EffectsSearchArm {
  phase: "coarse" | "fine" | "joint";
  sourceIds: readonly string[];
  selectionReason: string;
  executionState: string;
  executionDisposition: string;
  evidenceStatus: EffectsSearchEvidenceStatus;
  deltaNats: number | null;
  observationId: string | null;
  proofGrade: string | null;
  numericalRegime: string | null;
  executionIdentity: Readonly<Record<string, unknown>> | null;
}

export interface EffectsSearchActualWork {
  modelMeasurementsExecutedTotal: number;
  modelMeasurementsReusedTotal: number;
  modelMeasurementsSkippedTotal: number;
  savedModelMeasurementsTotal: number;
}

export interface EffectsSearchResult {
  schemaVersion: "clozn.context-investigation-effects-search-job-result.v1";
  runId: string;
  nativeScreeningRequested: boolean;
  arms: readonly EffectsSearchArm[];
  actualWork: EffectsSearchActualWork;
  readerProjection: {
    experimentId: string;
    status: string;
    controlObservationId: string | null;
    trustedSingletonSourceIds: readonly string[];
  };
}

export interface EffectsSearchJob {
  schemaVersion: "clozn.influence-map-job.v1";
  jobId: string;
  runId: string;
  kind: "context_investigation_effects_search";
  state: EffectsSearchJobState;
  progress: {
    phase: string;
    completedUnits: number;
    totalUnits: number;
    percent: number;
  };
  cancelRequested: boolean;
  cancellable: boolean;
  cached: boolean;
  error?: { code: string; message: string };
  result?: EffectsSearchResult;
}

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function record(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(endpoint, path, "must be an object");
  }
  return value as JsonRecord;
}

function nullableRecord(value: unknown, endpoint: string, path: string): JsonRecord | null {
  return value === null ? null : record(value, endpoint, path);
}

function list(value: unknown, endpoint: string, path: string): unknown[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string" || !value.trim()) fail(endpoint, path, "must be a non-empty string");
  return value;
}

function nullableText(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : text(value, endpoint, path);
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function finite(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be finite");
  return value;
}

function count(value: unknown, endpoint: string, path: string): number {
  const parsed = finite(value, endpoint, path);
  if (!Number.isSafeInteger(parsed) || parsed < 0) fail(endpoint, path, "must be a non-negative integer");
  return parsed;
}

function oneOf<const T extends readonly string[]>(
  value: unknown,
  values: T,
  endpoint: string,
  path: string,
): T[number] {
  if (typeof value !== "string" || !values.includes(value)) {
    fail(endpoint, path, `must be one of ${values.join(", ")}`);
  }
  return value as T[number];
}

function strings(value: unknown, endpoint: string, path: string): string[] {
  const parsed = list(value, endpoint, path).map((item, index) =>
    text(item, endpoint, `${path}[${index}]`));
  if (new Set(parsed).size !== parsed.length) fail(endpoint, path, "must not contain duplicates");
  return parsed;
}

function decodeArm(value: unknown, endpoint: string, path: string): EffectsSearchArm {
  const row = record(value, endpoint, path);
  const sourceIds = strings(required(row, "source_ids", endpoint, path), endpoint, `${path}.source_ids`);
  if (!sourceIds.length) fail(endpoint, `${path}.source_ids`, "must not be empty");
  const rawDelta = required(row, "delta_nats", endpoint, path);
  return {
    phase: oneOf(required(row, "phase", endpoint, path), ["coarse", "fine", "joint"] as const, endpoint, `${path}.phase`),
    sourceIds,
    selectionReason: text(required(row, "selection_reason", endpoint, path), endpoint, `${path}.selection_reason`),
    executionState: text(required(row, "execution_state", endpoint, path), endpoint, `${path}.execution_state`),
    executionDisposition: text(required(row, "execution_disposition", endpoint, path), endpoint, `${path}.execution_disposition`),
    evidenceStatus: oneOf(required(row, "evidence_status", endpoint, path), ["trusted", "screening_only", "unavailable", "skipped"] as const, endpoint, `${path}.evidence_status`),
    deltaNats: rawDelta === null ? null : finite(rawDelta, endpoint, `${path}.delta_nats`),
    observationId: nullableText(required(row, "observation_id", endpoint, path), endpoint, `${path}.observation_id`),
    proofGrade: nullableText(required(row, "proof_grade", endpoint, path), endpoint, `${path}.proof_grade`),
    numericalRegime: nullableText(required(row, "numerical_regime", endpoint, path), endpoint, `${path}.numerical_regime`),
    executionIdentity: nullableRecord(required(row, "execution_identity", endpoint, path), endpoint, `${path}.execution_identity`),
  };
}

export function decodeEffectsSearchResult(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
): EffectsSearchResult {
  const row = record(value, endpoint, "$.result");
  const schemaVersion = oneOf(
    required(row, "schema_version", endpoint, "$.result"),
    ["clozn.context-investigation-effects-search-job-result.v1"] as const,
    endpoint,
    "$.result.schema_version",
  );
  const runId = text(required(row, "run_id", endpoint, "$.result"), endpoint, "$.result.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.result.run_id", "must match the requested Run");
  const budgetUsage = record(required(row, "budget_usage", endpoint, "$.result"), endpoint, "$.result.budget_usage");
  const actual = record(required(budgetUsage, "actual_work", endpoint, "$.result.budget_usage"), endpoint, "$.result.budget_usage.actual_work");
  const projection = record(required(row, "reader_projection", endpoint, "$.result"), endpoint, "$.result.reader_projection");
  return {
    schemaVersion,
    runId,
    nativeScreeningRequested: booleanValue(required(row, "native_screening_requested", endpoint, "$.result"), endpoint, "$.result.native_screening_requested"),
    arms: list(required(row, "arms", endpoint, "$.result"), endpoint, "$.result.arms")
      .map((item, index) => decodeArm(item, endpoint, `$.result.arms[${index}]`)),
    actualWork: {
      modelMeasurementsExecutedTotal: count(required(actual, "model_measurements_executed_total", endpoint, "$.result.budget_usage.actual_work"), endpoint, "$.result.budget_usage.actual_work.model_measurements_executed_total"),
      modelMeasurementsReusedTotal: count(required(actual, "model_measurements_reused_total", endpoint, "$.result.budget_usage.actual_work"), endpoint, "$.result.budget_usage.actual_work.model_measurements_reused_total"),
      modelMeasurementsSkippedTotal: count(required(actual, "model_measurements_skipped_total", endpoint, "$.result.budget_usage.actual_work"), endpoint, "$.result.budget_usage.actual_work.model_measurements_skipped_total"),
      savedModelMeasurementsTotal: count(required(actual, "saved_model_measurements_total", endpoint, "$.result.budget_usage.actual_work"), endpoint, "$.result.budget_usage.actual_work.saved_model_measurements_total"),
    },
    readerProjection: {
      experimentId: text(required(projection, "experiment_id", endpoint, "$.result.reader_projection"), endpoint, "$.result.reader_projection.experiment_id"),
      status: text(required(projection, "status", endpoint, "$.result.reader_projection"), endpoint, "$.result.reader_projection.status"),
      controlObservationId: nullableText(required(projection, "control_observation_id", endpoint, "$.result.reader_projection"), endpoint, "$.result.reader_projection.control_observation_id"),
      trustedSingletonSourceIds: strings(required(projection, "trusted_singleton_source_ids", endpoint, "$.result.reader_projection"), endpoint, "$.result.reader_projection.trusted_singleton_source_ids"),
    },
  };
}

export function decodeEffectsSearchJob(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
): EffectsSearchJob {
  const row = record(value, endpoint, "$");
  const schemaVersion = oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.influence-map-job.v1"] as const, endpoint, "$.schema_version");
  const runId = text(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.run_id", "must match the requested Run");
  const kind = oneOf(required(row, "kind", endpoint, "$"), ["context_investigation_effects_search"] as const, endpoint, "$.kind");
  const state = oneOf(required(row, "state", endpoint, "$"), ["queued", "running", "cancelling", "persisting", "completed", "failed", "cancelled"] as const, endpoint, "$.state");
  const progress = record(required(row, "progress", endpoint, "$"), endpoint, "$.progress");
  const percent = finite(required(progress, "percent", endpoint, "$.progress"), endpoint, "$.progress.percent");
  if (percent < 0 || percent > 100) fail(endpoint, "$.progress.percent", "must be between 0 and 100");
  const rawError = row.error;
  const error = rawError === undefined ? undefined : record(rawError, endpoint, "$.error");
  const result = row.result === undefined
    ? undefined
    : decodeEffectsSearchResult(row.result, endpoint, expectedRunId);
  if (state === "completed" && !result) fail(endpoint, "$.result", "is required for a completed search job");
  return {
    schemaVersion,
    jobId: text(required(row, "job_id", endpoint, "$"), endpoint, "$.job_id"),
    runId,
    kind,
    state,
    progress: {
      phase: text(required(progress, "phase", endpoint, "$.progress"), endpoint, "$.progress.phase"),
      completedUnits: count(required(progress, "completed_units", endpoint, "$.progress"), endpoint, "$.progress.completed_units"),
      totalUnits: count(required(progress, "total_units", endpoint, "$.progress"), endpoint, "$.progress.total_units"),
      percent,
    },
    cancelRequested: booleanValue(required(row, "cancel_requested", endpoint, "$"), endpoint, "$.cancel_requested"),
    cancellable: booleanValue(required(row, "cancellable", endpoint, "$"), endpoint, "$.cancellable"),
    cached: booleanValue(required(row, "cached", endpoint, "$"), endpoint, "$.cached"),
    error: error ? {
      code: text(required(error, "code", endpoint, "$.error"), endpoint, "$.error.code"),
      message: text(required(error, "message", endpoint, "$.error"), endpoint, "$.error.message"),
    } : undefined,
    result,
  };
}

export function isTerminalEffectsSearchJob(job: EffectsSearchJob): boolean {
  return job.state === "completed" || job.state === "failed" || job.state === "cancelled";
}
