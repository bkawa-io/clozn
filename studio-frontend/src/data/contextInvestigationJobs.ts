import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type ContextEffectsJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "persisting"
  | "completed"
  | "failed"
  | "cancelled";

export interface ContextEffectsResult {
  schemaVersion: "clozn.context-investigation-effects-job-result.v1";
  experimentId: string;
  sourceCount: number;
  completedSourceCount: number;
  unavailableSourceCount: number;
  failedSourceCount: number;
  notMeasuredSourceCount: number;
  state: string;
}

export interface ContextEffectsJob {
  schemaVersion: "clozn.analysis-job.v1";
  jobId: string;
  runId: string;
  kind: "context_investigation_effects";
  state: ContextEffectsJobState;
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
  result?: ContextEffectsResult;
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

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string" || !value.trim()) fail(endpoint, path, "must be a non-empty string");
  return value;
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

export function decodeContextEffectsResult(
  value: unknown,
  endpoint: string,
): ContextEffectsResult {
  const row = record(value, endpoint, "$.result");
  const sourceCount = count(required(row, "source_count", endpoint, "$.result"), endpoint, "$.result.source_count");
  const completedSourceCount = count(required(row, "completed_source_count", endpoint, "$.result"), endpoint, "$.result.completed_source_count");
  const unavailableSourceCount = count(required(row, "unavailable_source_count", endpoint, "$.result"), endpoint, "$.result.unavailable_source_count");
  const failedSourceCount = count(required(row, "failed_source_count", endpoint, "$.result"), endpoint, "$.result.failed_source_count");
  const notMeasuredSourceCount = count(required(row, "not_measured_source_count", endpoint, "$.result"), endpoint, "$.result.not_measured_source_count");
  if (completedSourceCount + unavailableSourceCount + failedSourceCount + notMeasuredSourceCount !== sourceCount) {
    fail(endpoint, "$.result", "source status counts must add up to source_count");
  }
  return {
    schemaVersion: oneOf(
      required(row, "schema_version", endpoint, "$.result"),
      ["clozn.context-investigation-effects-job-result.v1"] as const,
      endpoint,
      "$.result.schema_version",
    ),
    experimentId: text(required(row, "experiment_id", endpoint, "$.result"), endpoint, "$.result.experiment_id"),
    sourceCount,
    completedSourceCount,
    unavailableSourceCount,
    failedSourceCount,
    notMeasuredSourceCount,
    state: text(required(row, "state", endpoint, "$.result"), endpoint, "$.result.state"),
  };
}

export function decodeContextEffectsJob(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
): ContextEffectsJob {
  const row = record(value, endpoint, "$");
  const runId = text(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.run_id", "must match the requested Run");
  const state = oneOf(
    required(row, "state", endpoint, "$"),
    ["queued", "running", "cancelling", "persisting", "completed", "failed", "cancelled"] as const,
    endpoint,
    "$.state",
  );
  const progress = record(required(row, "progress", endpoint, "$"), endpoint, "$.progress");
  const percent = finite(required(progress, "percent", endpoint, "$.progress"), endpoint, "$.progress.percent");
  if (percent < 0 || percent > 100) fail(endpoint, "$.progress.percent", "must be between 0 and 100");
  const rawError = row.error;
  const error = rawError === undefined ? undefined : record(rawError, endpoint, "$.error");
  const result = row.result === undefined
    ? undefined
    : decodeContextEffectsResult(row.result, endpoint);
  if (state === "completed" && !result) fail(endpoint, "$.result", "is required for a completed effects job");
  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.analysis-job.v1"] as const, endpoint, "$.schema_version"),
    jobId: text(required(row, "job_id", endpoint, "$"), endpoint, "$.job_id"),
    runId,
    kind: oneOf(required(row, "kind", endpoint, "$"), ["context_investigation_effects"] as const, endpoint, "$.kind"),
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

export function isTerminalContextEffectsJob(job: ContextEffectsJob): boolean {
  return job.state === "completed" || job.state === "failed" || job.state === "cancelled";
}
