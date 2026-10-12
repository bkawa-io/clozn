import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type ContextActionJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "persisting"
  | "completed"
  | "failed"
  | "cancelled";

export interface ContextActionJobProgress {
  phase: string;
  completedUnits: number;
  totalUnits: number;
  percent: number;
}

export interface ContextActionJobError {
  code: string;
  message: string;
}

export interface RemoveSourceTestResult {
  schemaVersion: "clozn.context-investigation-remove-test-result.v1";
  runId: string;
  sourceId: string;
  experimentId: string;
  armId: string;
  observationId: string | null;
  status: "exact_preserved" | "diverged" | "unavailable" | "failed";
  matchedTokenCount: number | null;
  firstDivergenceIndex: number | null;
  divergenceKind: string | null;
  reason: string | null;
  reasonCode: string | null;
  experimentState: "pending" | "running" | "completed" | "cancelled" | "failed" | "blocked";
}

export interface ContextCounterfactualResult {
  schemaVersion: "clozn.context-investigation-counterfactual-result.v1";
  runId: string;
  sourceId: string;
  experimentId: string;
  armId: string;
  observationId: string | null;
  status: "completed" | "unavailable" | "failed";
  generatedText: string;
  generatedTokenCount: number;
  finishReason: string | null;
  reason: string | null;
  reasonCode: string | null;
  experimentState: "pending" | "running" | "completed" | "cancelled" | "failed" | "blocked";
}

interface ContextActionJob<K extends string, R> {
  schemaVersion: "clozn.influence-map-job.v1";
  jobId: string;
  runId: string;
  kind: K;
  state: ContextActionJobState;
  progress: ContextActionJobProgress;
  cancelRequested: boolean;
  cancellable: boolean;
  cached: boolean;
  createdAt: string;
  updatedAt: string;
  error?: ContextActionJobError;
  result?: R;
}

export type RemoveSourceTestJob = ContextActionJob<
  "context_investigation_remove_test",
  RemoveSourceTestResult
>;

export type ContextCounterfactualJob = ContextActionJob<
  "context_investigation_counterfactual",
  ContextCounterfactualResult
>;

export interface CounterfactualMaterializationRequest {
  experimentId: string;
  armId: string;
  observationId: string;
}

export type CounterfactualMaterializationResult =
  | {
      schemaVersion: "clozn.experiment-materialization.v1";
      state: "completed";
      parentRunId: string;
      childRunId: string;
      experimentId: string;
      armId: string;
      observationId: string;
      realizedFidelity: "direct_context_generation";
      traceState: "available" | "unavailable";
      comparePath: string;
    }
  | {
      schemaVersion: "clozn.experiment-materialization.v1";
      state: "failed";
      parentRunId: string;
      experimentId: string;
      armId: string;
      observationId: string;
      reason: string;
    };

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

function stringValue(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nullableText(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : text(value, endpoint, path);
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function count(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(endpoint, path, "must be a non-negative integer");
  }
  return value;
}

function nullableCount(value: unknown, endpoint: string, path: string): number | null {
  return value === null ? null : count(value, endpoint, path);
}

function finite(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be finite");
  return value;
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

function matchingText(
  row: JsonRecord,
  key: string,
  expected: string | undefined,
  endpoint: string,
  path: string,
): string {
  const value = text(required(row, key, endpoint, path), endpoint, `${path}.${key}`);
  if (expected !== undefined && value !== expected) fail(endpoint, `${path}.${key}`, "does not match the request");
  return value;
}

const experimentStates = ["pending", "running", "completed", "cancelled", "failed", "blocked"] as const;
const jobStates = ["queued", "running", "cancelling", "persisting", "completed", "failed", "cancelled"] as const;

export function decodeRemoveSourceTestResult(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
  expectedSourceId?: string,
): RemoveSourceTestResult {
  const path = "$.result";
  const row = record(value, endpoint, path);
  const status = oneOf(
    required(row, "status", endpoint, path),
    ["exact_preserved", "diverged", "unavailable", "failed"] as const,
    endpoint,
    `${path}.status`,
  );
  const observationId = nullableText(required(row, "observation_id", endpoint, path), endpoint, `${path}.observation_id`);
  const matchedTokenCount = nullableCount(required(row, "matched_token_count", endpoint, path), endpoint, `${path}.matched_token_count`);
  const firstDivergenceIndex = nullableCount(required(row, "first_divergence_index", endpoint, path), endpoint, `${path}.first_divergence_index`);
  const divergenceKind = nullableText(required(row, "divergence_kind", endpoint, path), endpoint, `${path}.divergence_kind`);

  if ((status === "exact_preserved" || status === "diverged") && observationId === null) {
    fail(endpoint, `${path}.observation_id`, `is required for ${status} evidence`);
  }
  if (status === "exact_preserved" && (firstDivergenceIndex !== null || divergenceKind !== null)) {
    fail(endpoint, path, "exact_preserved evidence must not claim a divergence");
  }
  if (status === "diverged" && divergenceKind === null) {
    fail(endpoint, `${path}.divergence_kind`, "is required for diverged evidence");
  }
  if ((status === "unavailable" || status === "failed")
      && (firstDivergenceIndex !== null || divergenceKind !== null)) {
    fail(endpoint, path, `${status} evidence must not claim a divergence`);
  }

  return {
    schemaVersion: oneOf(
      required(row, "schema_version", endpoint, path),
      ["clozn.context-investigation-remove-test-result.v1"] as const,
      endpoint,
      `${path}.schema_version`,
    ),
    runId: matchingText(row, "run_id", expectedRunId, endpoint, path),
    sourceId: matchingText(row, "source_id", expectedSourceId, endpoint, path),
    experimentId: text(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    armId: text(required(row, "arm_id", endpoint, path), endpoint, `${path}.arm_id`),
    observationId,
    status,
    matchedTokenCount,
    firstDivergenceIndex,
    divergenceKind,
    reason: nullableText(required(row, "reason", endpoint, path), endpoint, `${path}.reason`),
    reasonCode: nullableText(required(row, "reason_code", endpoint, path), endpoint, `${path}.reason_code`),
    experimentState: oneOf(required(row, "experiment_state", endpoint, path), experimentStates, endpoint, `${path}.experiment_state`),
  };
}

export function decodeContextCounterfactualResult(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
  expectedSourceId?: string,
): ContextCounterfactualResult {
  const path = "$.result";
  const row = record(value, endpoint, path);
  const status = oneOf(
    required(row, "status", endpoint, path),
    ["completed", "unavailable", "failed"] as const,
    endpoint,
    `${path}.status`,
  );
  const observationId = nullableText(required(row, "observation_id", endpoint, path), endpoint, `${path}.observation_id`);
  const generatedText = stringValue(required(row, "generated_text", endpoint, path), endpoint, `${path}.generated_text`);
  const generatedTokenCount = count(required(row, "generated_token_count", endpoint, path), endpoint, `${path}.generated_token_count`);
  if (status === "completed" && observationId === null) {
    fail(endpoint, `${path}.observation_id`, "is required for completed generation evidence");
  }
  if (status !== "completed" && (generatedText !== "" || generatedTokenCount !== 0)) {
    fail(endpoint, path, `${status} generation must not claim generated output`);
  }
  return {
    schemaVersion: oneOf(
      required(row, "schema_version", endpoint, path),
      ["clozn.context-investigation-counterfactual-result.v1"] as const,
      endpoint,
      `${path}.schema_version`,
    ),
    runId: matchingText(row, "run_id", expectedRunId, endpoint, path),
    sourceId: matchingText(row, "source_id", expectedSourceId, endpoint, path),
    experimentId: text(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    armId: text(required(row, "arm_id", endpoint, path), endpoint, `${path}.arm_id`),
    observationId,
    status,
    generatedText,
    generatedTokenCount,
    finishReason: nullableText(required(row, "finish_reason", endpoint, path), endpoint, `${path}.finish_reason`),
    reason: nullableText(required(row, "reason", endpoint, path), endpoint, `${path}.reason`),
    reasonCode: nullableText(required(row, "reason_code", endpoint, path), endpoint, `${path}.reason_code`),
    experimentState: oneOf(required(row, "experiment_state", endpoint, path), experimentStates, endpoint, `${path}.experiment_state`),
  };
}

function decodeActionJob<K extends string, R>(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
  expectedKind: K,
  decodeResult: (value: unknown) => R,
  expectedJobId?: string,
): ContextActionJob<K, R> {
  const row = record(value, endpoint, "$");
  const state = oneOf(required(row, "state", endpoint, "$"), jobStates, endpoint, "$.state");
  const progress = record(required(row, "progress", endpoint, "$"), endpoint, "$.progress");
  const completedUnits = count(required(progress, "completed_units", endpoint, "$.progress"), endpoint, "$.progress.completed_units");
  const totalUnits = count(required(progress, "total_units", endpoint, "$.progress"), endpoint, "$.progress.total_units");
  const percent = finite(required(progress, "percent", endpoint, "$.progress"), endpoint, "$.progress.percent");
  if (completedUnits > totalUnits) fail(endpoint, "$.progress", "completed_units must not exceed total_units");
  if (percent < 0 || percent > 100) fail(endpoint, "$.progress.percent", "must be between 0 and 100");

  const rawError = row.error;
  const error = rawError === undefined ? undefined : record(rawError, endpoint, "$.error");
  const result = row.result === undefined ? undefined : decodeResult(row.result);
  if (state === "completed" && result === undefined) fail(endpoint, "$.result", "is required for a completed action job");
  if (state === "failed" && error === undefined) fail(endpoint, "$.error", "is required for a failed action job");
  const cancellable = booleanValue(required(row, "cancellable", endpoint, "$"), endpoint, "$.cancellable");
  if ((state === "completed" || state === "failed" || state === "cancelled") && cancellable) {
    fail(endpoint, "$.cancellable", "must be false for a terminal action job");
  }

  return {
    schemaVersion: oneOf(
      required(row, "schema_version", endpoint, "$"),
      ["clozn.influence-map-job.v1"] as const,
      endpoint,
      "$.schema_version",
    ),
    jobId: matchingText(row, "job_id", expectedJobId, endpoint, "$"),
    runId: matchingText(row, "run_id", expectedRunId, endpoint, "$"),
    kind: oneOf(required(row, "kind", endpoint, "$"), [expectedKind] as const, endpoint, "$.kind"),
    state,
    progress: {
      phase: text(required(progress, "phase", endpoint, "$.progress"), endpoint, "$.progress.phase"),
      completedUnits,
      totalUnits,
      percent,
    },
    cancelRequested: booleanValue(required(row, "cancel_requested", endpoint, "$"), endpoint, "$.cancel_requested"),
    cancellable,
    cached: booleanValue(required(row, "cached", endpoint, "$"), endpoint, "$.cached"),
    createdAt: text(required(row, "created_at", endpoint, "$"), endpoint, "$.created_at"),
    updatedAt: text(required(row, "updated_at", endpoint, "$"), endpoint, "$.updated_at"),
    error: error ? {
      code: text(required(error, "code", endpoint, "$.error"), endpoint, "$.error.code"),
      message: text(required(error, "message", endpoint, "$.error"), endpoint, "$.error.message"),
    } : undefined,
    result,
  };
}

export function decodeRemoveSourceTestJob(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
  expectedSourceId?: string,
  expectedJobId?: string,
): RemoveSourceTestJob {
  return decodeActionJob(
    value,
    endpoint,
    expectedRunId,
    "context_investigation_remove_test",
    (result) => decodeRemoveSourceTestResult(result, endpoint, expectedRunId, expectedSourceId),
    expectedJobId,
  );
}

export function decodeContextCounterfactualJob(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
  expectedSourceId?: string,
  expectedJobId?: string,
): ContextCounterfactualJob {
  return decodeActionJob(
    value,
    endpoint,
    expectedRunId,
    "context_investigation_counterfactual",
    (result) => decodeContextCounterfactualResult(result, endpoint, expectedRunId, expectedSourceId),
    expectedJobId,
  );
}

export function decodeCounterfactualMaterializationResult(
  value: unknown,
  endpoint: string,
  expectedParentRunId: string,
  request: CounterfactualMaterializationRequest,
): CounterfactualMaterializationResult {
  const row = record(value, endpoint, "$");
  const schemaVersion = oneOf(
    required(row, "schema_version", endpoint, "$"),
    ["clozn.experiment-materialization.v1"] as const,
    endpoint,
    "$.schema_version",
  );
  const state = oneOf(required(row, "state", endpoint, "$"), ["completed", "failed"] as const, endpoint, "$.state");
  const parentRunId = matchingText(row, "parent_run_id", expectedParentRunId, endpoint, "$"),
    experimentId = matchingText(row, "experiment_id", request.experimentId, endpoint, "$"),
    armId = matchingText(row, "arm_id", request.armId, endpoint, "$"),
    observationId = matchingText(row, "observation_id", request.observationId, endpoint, "$");
  if (state === "failed") {
    return {
      schemaVersion,
      state,
      parentRunId,
      experimentId,
      armId,
      observationId,
      reason: text(required(row, "reason", endpoint, "$"), endpoint, "$.reason"),
    };
  }
  const childRunId = text(required(row, "child_run_id", endpoint, "$"), endpoint, "$.child_run_id");
  if (childRunId === parentRunId) fail(endpoint, "$.child_run_id", "must identify a child, not the source Run");
  const comparePath = text(required(row, "compare_path", endpoint, "$"), endpoint, "$.compare_path");
  if (comparePath !== `#/compare/${parentRunId}/${childRunId}`) {
    fail(endpoint, "$.compare_path", "does not match the returned parent/child lineage");
  }
  return {
    schemaVersion,
    state,
    parentRunId,
    childRunId,
    experimentId,
    armId,
    observationId,
    realizedFidelity: oneOf(
      required(row, "realized_fidelity", endpoint, "$"),
      ["direct_context_generation"] as const,
      endpoint,
      "$.realized_fidelity",
    ),
    traceState: oneOf(required(row, "trace_state", endpoint, "$"), ["available", "unavailable"] as const, endpoint, "$.trace_state"),
    comparePath,
  };
}
