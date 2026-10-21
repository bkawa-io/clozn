import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type LeanContextJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "persisting"
  | "completed"
  | "failed"
  | "cancelled";

export interface LeanContextLoss {
  answerTokenCount: number;
  positiveLossTokenCount: number;
  meanPositiveSupportLossNatsPerToken: number;
  maxTokenSupportLossNats: number;
}

export interface LeanContextCandidate {
  retainedSourceIds: readonly string[];
  removedSourceIds: readonly string[];
  renderedPromptTokenCost: number;
  toleranceStatus: "within_tolerance" | "outside_tolerance" | null;
  loss: LeanContextLoss | null;
  observationId: string | null;
}

export interface LeanContextTrial {
  ordinal: number;
  stage: string;
  retainedSourceIds: readonly string[];
  removedSourceIds: readonly string[];
  renderedPromptTokenCost: number;
  classification: "preserves" | "diverged" | "unknown";
  disposition: "executed" | "reused" | "not_executed";
  toleranceStatus: "within_tolerance" | "outside_tolerance" | null;
  loss: LeanContextLoss | null;
}

export interface LeanContextResult {
  schemaVersion: "clozn.lean-context-result.v1";
  runId: string;
  searchId: string;
  status: "completed" | "unavailable" | "cancelled";
  reason?: string;
  reasonCode?: string;
  claim: { kind: "best_found_within_budget"; minimumProven: false };
  tolerance: {
    profile: "balanced.v1";
    meanPositiveSupportLossNatsPerToken: number;
    maxTokenSupportLossNats: number;
  };
  sourceIds: readonly string[];
  original: LeanContextCandidate | null;
  best: LeanContextCandidate | null;
  reduction: {
    removedPromptTokens: number;
    retainedPromptTokenFraction: number;
    removedSourceCount: number;
    retainedSourceCount: number;
  } | null;
  trials: readonly LeanContextTrial[];
  budget: {
    maxNewExecutions: number;
    usedNewExecutions: number;
    reusedObservationCount: number;
    exhausted: boolean;
    blockedByBudget: boolean;
  };
  stoppingReason: "search_policy_complete" | "budget_exhausted" | "search_unavailable" | "cancelled";
}

export interface LeanContextJob {
  schemaVersion: "clozn.analysis-job.v1";
  jobId: string;
  runId: string;
  kind: "context_investigation_lean_context";
  state: LeanContextJobState;
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
  result?: LeanContextResult;
}

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function record(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(endpoint, path, "must be an object");
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

function number(value: unknown, endpoint: string, path: string, minimum = 0): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    fail(endpoint, path, `must be a finite number of at least ${minimum}`);
  }
  return value;
}

function count(value: unknown, endpoint: string, path: string): number {
  const parsed = number(value, endpoint, path);
  if (!Number.isSafeInteger(parsed)) fail(endpoint, path, "must be an integer");
  return parsed;
}

function oneOf<const T extends readonly (string | null)[]>(
  value: unknown,
  values: T,
  endpoint: string,
  path: string,
): T[number] {
  if (!values.includes(value as T[number])) fail(endpoint, path, `must be one of ${values.join(", ")}`);
  return value as T[number];
}

function stringList(value: unknown, endpoint: string, path: string): string[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value.map((item, index) => text(item, endpoint, `${path}[${index}]`));
}

function nullableRecord(value: unknown, endpoint: string, path: string): JsonRecord | null {
  return value === null ? null : record(value, endpoint, path);
}

function decodeLoss(value: unknown, endpoint: string, path: string): LeanContextLoss | null {
  const row = nullableRecord(value, endpoint, path);
  if (!row) return null;
  const answerTokenCount = count(required(row, "answer_token_count", endpoint, path), endpoint, `${path}.answer_token_count`);
  if (answerTokenCount < 1) fail(endpoint, `${path}.answer_token_count`, "must be positive");
  const positiveLossTokenCount = count(required(row, "positive_loss_token_count", endpoint, path), endpoint, `${path}.positive_loss_token_count`);
  if (positiveLossTokenCount > answerTokenCount) fail(endpoint, `${path}.positive_loss_token_count`, "must not exceed answer_token_count");
  return {
    answerTokenCount,
    positiveLossTokenCount,
    meanPositiveSupportLossNatsPerToken: number(required(row, "mean_positive_support_loss_nats_per_token", endpoint, path), endpoint, `${path}.mean_positive_support_loss_nats_per_token`),
    maxTokenSupportLossNats: number(required(row, "max_token_support_loss_nats", endpoint, path), endpoint, `${path}.max_token_support_loss_nats`),
  };
}

function decodeCandidate(value: unknown, endpoint: string, path: string): LeanContextCandidate | null {
  const row = nullableRecord(value, endpoint, path);
  if (!row) return null;
  const evidence = nullableRecord(required(row, "evidence", endpoint, path), endpoint, `${path}.evidence`);
  const observationId = evidence?.observation_id;
  return {
    retainedSourceIds: stringList(required(row, "retained_source_ids", endpoint, path), endpoint, `${path}.retained_source_ids`),
    removedSourceIds: stringList(required(row, "removed_source_ids", endpoint, path), endpoint, `${path}.removed_source_ids`),
    renderedPromptTokenCost: count(required(row, "rendered_prompt_token_cost", endpoint, path), endpoint, `${path}.rendered_prompt_token_cost`),
    toleranceStatus: oneOf(required(row, "tolerance_status", endpoint, path), ["within_tolerance", "outside_tolerance", null] as const, endpoint, `${path}.tolerance_status`),
    loss: decodeLoss(required(row, "loss", endpoint, path), endpoint, `${path}.loss`),
    observationId: observationId === undefined || observationId === null
      ? null
      : text(observationId, endpoint, `${path}.evidence.observation_id`),
  };
}

function decodeTrial(value: unknown, endpoint: string, path: string): LeanContextTrial {
  const row = record(value, endpoint, path);
  return {
    ordinal: count(required(row, "ordinal", endpoint, path), endpoint, `${path}.ordinal`),
    stage: text(required(row, "stage", endpoint, path), endpoint, `${path}.stage`),
    retainedSourceIds: stringList(required(row, "retained_source_ids", endpoint, path), endpoint, `${path}.retained_source_ids`),
    removedSourceIds: stringList(required(row, "removed_source_ids", endpoint, path), endpoint, `${path}.removed_source_ids`),
    renderedPromptTokenCost: count(required(row, "rendered_prompt_token_cost", endpoint, path), endpoint, `${path}.rendered_prompt_token_cost`),
    classification: oneOf(required(row, "classification", endpoint, path), ["preserves", "diverged", "unknown"] as const, endpoint, `${path}.classification`),
    disposition: oneOf(required(row, "disposition", endpoint, path), ["executed", "reused", "not_executed"] as const, endpoint, `${path}.disposition`),
    toleranceStatus: oneOf(required(row, "tolerance_status", endpoint, path), ["within_tolerance", "outside_tolerance", null] as const, endpoint, `${path}.tolerance_status`),
    loss: decodeLoss(required(row, "loss", endpoint, path), endpoint, `${path}.loss`),
  };
}

export function decodeLeanContextResult(value: unknown, endpoint: string, expectedRunId: string): LeanContextResult {
  const row = record(value, endpoint, "$.result");
  const runId = text(required(row, "run_id", endpoint, "$.result"), endpoint, "$.result.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.result.run_id", "must match the requested Run");
  const claim = record(required(row, "claim", endpoint, "$.result"), endpoint, "$.result.claim");
  const tolerance = record(required(row, "tolerance", endpoint, "$.result"), endpoint, "$.result.tolerance");
  const universe = record(required(row, "universe", endpoint, "$.result"), endpoint, "$.result.universe");
  const budget = record(required(row, "budget", endpoint, "$.result"), endpoint, "$.result.budget");
  const rawReduction = nullableRecord(required(row, "reduction", endpoint, "$.result"), endpoint, "$.result.reduction");
  const rawTrials = required(row, "trials", endpoint, "$.result");
  if (!Array.isArray(rawTrials)) fail(endpoint, "$.result.trials", "must be an array");
  const minimumProven = required(claim, "minimum_proven", endpoint, "$.result.claim");
  if (minimumProven !== false) fail(endpoint, "$.result.claim.minimum_proven", "must be false");
  const result: LeanContextResult = {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$.result"), ["clozn.lean-context-result.v1"] as const, endpoint, "$.result.schema_version"),
    runId,
    searchId: text(required(row, "search_id", endpoint, "$.result"), endpoint, "$.result.search_id"),
    status: oneOf(required(row, "status", endpoint, "$.result"), ["completed", "unavailable", "cancelled"] as const, endpoint, "$.result.status"),
    ...(typeof row.reason === "string" ? { reason: row.reason } : {}),
    ...(typeof row.reason_code === "string" ? { reasonCode: row.reason_code } : {}),
    claim: {
      kind: oneOf(required(claim, "kind", endpoint, "$.result.claim"), ["best_found_within_budget"] as const, endpoint, "$.result.claim.kind"),
      minimumProven: false,
    },
    tolerance: {
      profile: oneOf(required(tolerance, "profile", endpoint, "$.result.tolerance"), ["balanced.v1"] as const, endpoint, "$.result.tolerance.profile"),
      meanPositiveSupportLossNatsPerToken: number(required(tolerance, "mean_positive_support_loss_nats_per_token", endpoint, "$.result.tolerance"), endpoint, "$.result.tolerance.mean_positive_support_loss_nats_per_token"),
      maxTokenSupportLossNats: number(required(tolerance, "max_token_support_loss_nats", endpoint, "$.result.tolerance"), endpoint, "$.result.tolerance.max_token_support_loss_nats"),
    },
    sourceIds: stringList(required(universe, "source_ids", endpoint, "$.result.universe"), endpoint, "$.result.universe.source_ids"),
    original: decodeCandidate(required(row, "original", endpoint, "$.result"), endpoint, "$.result.original"),
    best: decodeCandidate(required(row, "best", endpoint, "$.result"), endpoint, "$.result.best"),
    reduction: rawReduction ? {
      removedPromptTokens: count(required(rawReduction, "removed_prompt_tokens", endpoint, "$.result.reduction"), endpoint, "$.result.reduction.removed_prompt_tokens"),
      retainedPromptTokenFraction: number(required(rawReduction, "retained_prompt_token_fraction", endpoint, "$.result.reduction"), endpoint, "$.result.reduction.retained_prompt_token_fraction"),
      removedSourceCount: count(required(rawReduction, "removed_source_count", endpoint, "$.result.reduction"), endpoint, "$.result.reduction.removed_source_count"),
      retainedSourceCount: count(required(rawReduction, "retained_source_count", endpoint, "$.result.reduction"), endpoint, "$.result.reduction.retained_source_count"),
    } : null,
    trials: rawTrials.map((trial, index) => decodeTrial(trial, endpoint, `$.result.trials[${index}]`)),
    budget: {
      maxNewExecutions: count(required(budget, "max_new_executions", endpoint, "$.result.budget"), endpoint, "$.result.budget.max_new_executions"),
      usedNewExecutions: count(required(budget, "used_new_executions", endpoint, "$.result.budget"), endpoint, "$.result.budget.used_new_executions"),
      reusedObservationCount: count(required(budget, "reused_observation_count", endpoint, "$.result.budget"), endpoint, "$.result.budget.reused_observation_count"),
      exhausted: booleanValue(required(budget, "exhausted", endpoint, "$.result.budget"), endpoint, "$.result.budget.exhausted"),
      blockedByBudget: booleanValue(required(budget, "blocked_by_budget", endpoint, "$.result.budget"), endpoint, "$.result.budget.blocked_by_budget"),
    },
    stoppingReason: oneOf(required(row, "stopping_reason", endpoint, "$.result"), ["search_policy_complete", "budget_exhausted", "search_unavailable", "cancelled"] as const, endpoint, "$.result.stopping_reason"),
  };
  if (result.status === "completed" && (!result.original || !result.best || !result.reduction)) {
    fail(endpoint, "$.result", "completed result requires original, best, and reduction");
  }
  if (result.best && result.original && result.best.renderedPromptTokenCost > result.original.renderedPromptTokenCost) {
    fail(endpoint, "$.result.best.rendered_prompt_token_cost", "must not exceed the original cost");
  }
  return result;
}

export function decodeLeanContextJob(value: unknown, endpoint: string, expectedRunId: string): LeanContextJob {
  const row = record(value, endpoint, "$");
  const runId = text(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.run_id", "must match the requested Run");
  const state = oneOf(required(row, "state", endpoint, "$"), ["queued", "running", "cancelling", "persisting", "completed", "failed", "cancelled"] as const, endpoint, "$.state");
  const progress = record(required(row, "progress", endpoint, "$"), endpoint, "$.progress");
  const percent = number(required(progress, "percent", endpoint, "$.progress"), endpoint, "$.progress.percent");
  if (percent > 100) fail(endpoint, "$.progress.percent", "must not exceed 100");
  const rawError = row.error === undefined ? undefined : record(row.error, endpoint, "$.error");
  const result = row.result === undefined ? undefined : decodeLeanContextResult(row.result, endpoint, expectedRunId);
  if (state === "completed" && !result) fail(endpoint, "$.result", "is required for a completed Lean Context job");
  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.analysis-job.v1"] as const, endpoint, "$.schema_version"),
    jobId: text(required(row, "job_id", endpoint, "$"), endpoint, "$.job_id"),
    runId,
    kind: oneOf(required(row, "kind", endpoint, "$"), ["context_investigation_lean_context"] as const, endpoint, "$.kind"),
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

export function isTerminalLeanContextJob(job: LeanContextJob): boolean {
  return job.state === "completed" || job.state === "failed" || job.state === "cancelled";
}
