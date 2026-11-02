import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type ContextInteractionsJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "persisting"
  | "completed"
  | "failed"
  | "cancelled";

export interface ContextInteractionEstimate {
  sourceId: string;
  sampleCount: number;
  uniquePairCount: number;
  meanMarginalNats: number | null;
  sampleStddevNats: number | null;
  leaveOneOutNats: number | null;
  meanMinusLeaveOneOutNats: number | null;
}

export interface ContextInteractionLoss {
  answerTokenCount: number;
  positiveLossTokenCount: number;
  meanPositiveSupportLossNatsPerToken: number;
  maxTokenSupportLossNats: number;
}

export interface ContextInteractionsResult {
  schemaVersion: "clozn.context-interactions-result.v1";
  runId: string;
  status: "running" | "completed" | "partial" | "unavailable";
  stopReason: string | null;
  seed: number;
  claim: {
    kind: "exploratory_sampled_marginals";
    redundancyProven: false;
  };
  sourceIds: readonly string[];
  budget: {
    maxNewExecutions: number;
    usedNewExecutions: number;
    reusedObservationCount: number;
    completedPairs: number;
  };
  sourceEstimates: readonly ContextInteractionEstimate[];
  elapsedSeconds: number | null;
  best: {
    retainedSourceIds: readonly string[];
    renderedPromptTokens: number;
    loss: ContextInteractionLoss | null;
  } | null;
}

export interface ContextInteractionsJob {
  schemaVersion: "clozn.analysis-job.v1";
  jobId: string;
  runId: string;
  kind: "context_investigation_interactions";
  state: ContextInteractionsJobState;
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
  result?: ContextInteractionsResult;
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

function nonNegative(value: unknown, endpoint: string, path: string): number {
  const parsed = finite(value, endpoint, path);
  if (parsed < 0) fail(endpoint, path, "must be non-negative");
  return parsed;
}

function count(value: unknown, endpoint: string, path: string): number {
  const parsed = nonNegative(value, endpoint, path);
  if (!Number.isSafeInteger(parsed)) fail(endpoint, path, "must be an integer");
  return parsed;
}

function nullableFinite(value: unknown, endpoint: string, path: string): number | null {
  return value === null ? null : finite(value, endpoint, path);
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

function stringList(value: unknown, endpoint: string, path: string): string[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  const parsed = value.map((item, index) => text(item, endpoint, `${path}[${index}]`));
  if (new Set(parsed).size !== parsed.length) fail(endpoint, path, "must not contain duplicates");
  return parsed;
}

function decodeLoss(value: unknown, endpoint: string, path: string): ContextInteractionLoss | null {
  if (value === null) return null;
  const row = record(value, endpoint, path);
  const answerTokenCount = count(required(row, "answer_token_count", endpoint, path), endpoint, `${path}.answer_token_count`);
  if (!answerTokenCount) fail(endpoint, `${path}.answer_token_count`, "must be positive");
  const positiveLossTokenCount = count(required(row, "positive_loss_token_count", endpoint, path), endpoint, `${path}.positive_loss_token_count`);
  if (positiveLossTokenCount > answerTokenCount) fail(endpoint, `${path}.positive_loss_token_count`, "must not exceed answer_token_count");
  return {
    answerTokenCount,
    positiveLossTokenCount,
    meanPositiveSupportLossNatsPerToken: nonNegative(required(row, "mean_positive_support_loss_nats_per_token", endpoint, path), endpoint, `${path}.mean_positive_support_loss_nats_per_token`),
    maxTokenSupportLossNats: nonNegative(required(row, "max_token_support_loss_nats", endpoint, path), endpoint, `${path}.max_token_support_loss_nats`),
  };
}

export function decodeContextInteractionsResult(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
): ContextInteractionsResult {
  const row = record(value, endpoint, "$.result");
  const runId = text(required(row, "run_id", endpoint, "$.result"), endpoint, "$.result.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.result.run_id", "must match the requested Run");
  const claim = record(required(row, "claim", endpoint, "$.result"), endpoint, "$.result.claim");
  if (required(claim, "redundancy_proven", endpoint, "$.result.claim") !== false) {
    fail(endpoint, "$.result.claim.redundancy_proven", "must be false");
  }
  const sourceIds = stringList(required(row, "source_ids", endpoint, "$.result"), endpoint, "$.result.source_ids");
  if (!sourceIds.length) fail(endpoint, "$.result.source_ids", "must not be empty");
  const rawEstimates = required(row, "source_estimates", endpoint, "$.result");
  if (!Array.isArray(rawEstimates)) fail(endpoint, "$.result.source_estimates", "must be an array");
  const sourceEstimates = rawEstimates.map((value, index): ContextInteractionEstimate => {
    const path = `$.result.source_estimates[${index}]`;
    const estimate = record(value, endpoint, path);
    return {
      sourceId: text(required(estimate, "source_id", endpoint, path), endpoint, `${path}.source_id`),
      sampleCount: count(required(estimate, "sample_count", endpoint, path), endpoint, `${path}.sample_count`),
      uniquePairCount: count(required(estimate, "unique_pair_count", endpoint, path), endpoint, `${path}.unique_pair_count`),
      meanMarginalNats: nullableFinite(required(estimate, "mean_marginal_nats", endpoint, path), endpoint, `${path}.mean_marginal_nats`),
      sampleStddevNats: nullableFinite(required(estimate, "sample_stddev_nats", endpoint, path), endpoint, `${path}.sample_stddev_nats`),
      leaveOneOutNats: nullableFinite(required(estimate, "leave_one_out_nats", endpoint, path), endpoint, `${path}.leave_one_out_nats`),
      meanMinusLeaveOneOutNats: nullableFinite(required(estimate, "mean_minus_leave_one_out_nats", endpoint, path), endpoint, `${path}.mean_minus_leave_one_out_nats`),
    };
  });
  const estimateIds = sourceEstimates.map((estimate) => estimate.sourceId);
  if (
    estimateIds.length !== sourceIds.length
    || new Set(estimateIds).size !== estimateIds.length
    || estimateIds.some((sourceId) => !sourceIds.includes(sourceId))
  ) fail(endpoint, "$.result.source_estimates", "must contain exactly one estimate per source");
  const budget = record(required(row, "budget", endpoint, "$.result"), endpoint, "$.result.budget");
  const maxNewExecutions = count(required(budget, "max_new_executions", endpoint, "$.result.budget"), endpoint, "$.result.budget.max_new_executions");
  const usedNewExecutions = count(required(budget, "used_new_executions", endpoint, "$.result.budget"), endpoint, "$.result.budget.used_new_executions");
  if (usedNewExecutions > maxNewExecutions) fail(endpoint, "$.result.budget.used_new_executions", "must not exceed max_new_executions");
  const rawBest = required(row, "best", endpoint, "$.result");
  const best = rawBest === null ? null : (() => {
    const candidate = record(rawBest, endpoint, "$.result.best");
    const retainedSourceIds = stringList(required(candidate, "retained_source_ids", endpoint, "$.result.best"), endpoint, "$.result.best.retained_source_ids");
    if (retainedSourceIds.some((sourceId) => !sourceIds.includes(sourceId))) {
      fail(endpoint, "$.result.best.retained_source_ids", "must belong to source_ids");
    }
    return {
      retainedSourceIds,
      renderedPromptTokens: count(required(candidate, "rendered_prompt_tokens", endpoint, "$.result.best"), endpoint, "$.result.best.rendered_prompt_tokens"),
      loss: decodeLoss(required(candidate, "loss", endpoint, "$.result.best"), endpoint, "$.result.best.loss"),
    };
  })();
  const elapsed = required(row, "elapsed_seconds", endpoint, "$.result");
  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$.result"), ["clozn.context-interactions-result.v1"] as const, endpoint, "$.result.schema_version"),
    runId,
    status: oneOf(required(row, "status", endpoint, "$.result"), ["running", "completed", "partial", "unavailable"] as const, endpoint, "$.result.status"),
    stopReason: row.stop_reason === null ? null : text(row.stop_reason, endpoint, "$.result.stop_reason"),
    seed: count(required(row, "seed", endpoint, "$.result"), endpoint, "$.result.seed"),
    claim: {
      kind: oneOf(required(claim, "kind", endpoint, "$.result.claim"), ["exploratory_sampled_marginals"] as const, endpoint, "$.result.claim.kind"),
      redundancyProven: false,
    },
    sourceIds,
    budget: {
      maxNewExecutions,
      usedNewExecutions,
      reusedObservationCount: count(required(budget, "reused_observation_count", endpoint, "$.result.budget"), endpoint, "$.result.budget.reused_observation_count"),
      completedPairs: count(required(budget, "completed_pairs", endpoint, "$.result.budget"), endpoint, "$.result.budget.completed_pairs"),
    },
    sourceEstimates,
    elapsedSeconds: elapsed === null ? null : nonNegative(elapsed, endpoint, "$.result.elapsed_seconds"),
    best,
  };
}

export function decodeContextInteractionsJob(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
): ContextInteractionsJob {
  const row = record(value, endpoint, "$");
  const runId = text(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.run_id", "must match the requested Run");
  const state = oneOf(required(row, "state", endpoint, "$"), ["queued", "running", "cancelling", "persisting", "completed", "failed", "cancelled"] as const, endpoint, "$.state");
  const progress = record(required(row, "progress", endpoint, "$"), endpoint, "$.progress");
  const percent = nonNegative(required(progress, "percent", endpoint, "$.progress"), endpoint, "$.progress.percent");
  if (percent > 100) fail(endpoint, "$.progress.percent", "must not exceed 100");
  const rawError = row.error === undefined ? undefined : record(row.error, endpoint, "$.error");
  const result = row.result === undefined
    ? undefined
    : decodeContextInteractionsResult(row.result, endpoint, expectedRunId);
  if (state === "completed" && !result) fail(endpoint, "$.result", "is required for a completed Interactions job");
  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.analysis-job.v1"] as const, endpoint, "$.schema_version"),
    jobId: text(required(row, "job_id", endpoint, "$"), endpoint, "$.job_id"),
    runId,
    kind: oneOf(required(row, "kind", endpoint, "$"), ["context_investigation_interactions"] as const, endpoint, "$.kind"),
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
    error: rawError ? {
      code: text(required(rawError, "code", endpoint, "$.error"), endpoint, "$.error.code"),
      message: text(required(rawError, "message", endpoint, "$.error"), endpoint, "$.error.message"),
    } : undefined,
    result,
  };
}

export function isTerminalContextInteractionsJob(job: ContextInteractionsJob): boolean {
  return job.state === "completed" || job.state === "failed" || job.state === "cancelled";
}
