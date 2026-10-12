import { HttpError } from "./client";
import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type MinimalContextCertificate =
  | "BEST_VERIFIED"
  | "INCLUSION_MINIMUM"
  | "EXACT_MINIMUM";
export type MinimalContextStatus = "completed" | "unavailable" | "cancelled";
export type MinimalContextStoppingReason =
  | "exact_minimum_proven"
  | "inclusion_minimum_proven"
  | "budget_exhausted"
  | "search_policy_complete"
  | "control_unavailable"
  | "search_unavailable"
  | "cancelled";
export type MinimalContextJobState =
  | "queued"
  | "running"
  | "cancelling"
  | "persisting"
  | "completed"
  | "failed"
  | "cancelled";

export interface MinimalContextCurrentBinding {
  status: "current" | "stale" | "run_unavailable";
  reason: string | null;
}

export interface MinimalContextUniverse {
  schemaVersion: "clozn.context-search-universe.v1";
  runId: string;
  universeId: string;
  basisContextUnitsDigest: string;
  policy: {
    kind: "bounded_structural_partition";
    version: "bounded_structural_partition.v1";
    maxUnits: number;
    merge: "smallest_adjacent_auto_siblings_leftmost.v1";
    explicitOverCap: "typed_bound_exceeded";
  };
  sourceIds: readonly string[];
  sourceCount: number;
  coverage: {
    protectedMessageIndices: readonly number[];
    removableMessageIndices: readonly number[];
    removableContentCovered: true;
    partitionVerified: true;
    completeCatalogSourceCount: number;
    explicitSourceCount: number;
    selectedSourceCount?: number;
  };
  status: "planned" | "bound_exceeded";
  condition?: { code: string; message: string };
}

export interface MinimalContextCandidate {
  retainedSourceIds: readonly string[];
  removedSourceIds: readonly string[];
  renderedPromptTokenCost: number;
  experimentId: string | null;
  armId: string | null;
  observationId: string | null;
  observationStatus: string | null;
}

export interface MinimalContextEvidenceRef {
  disposition: "reused" | "executed" | "not_executed";
  experimentId: string | null;
  armId: string | null;
  observationId: string | null;
  observationStatus: string | null;
}

export interface MinimalContextTrial {
  ordinal: number;
  stage: string;
  retainedIds: readonly string[];
  cost: number;
  classification: "preserves" | "diverged" | "unknown";
  disposition: "reused" | "executed" | "not_executed";
  experimentId: string | null;
  armId: string | null;
  observationId: string | null;
  observationStatus: string | null;
  evidence: MinimalContextEvidenceRef | null;
  batchId: number | null;
  parentRetainedIds: readonly string[];
}

export interface MinimalContextTrajectoryEntry {
  counterfactualProbeCount: number;
  retainedIds: readonly string[];
  cost: number;
  retainedUnitCount: number;
  stage: string;
}

export interface MinimalContextReduction {
  objective: "rendered_prompt_tokens.v1";
  originalPromptTokenCost: number;
  retainedPromptTokenCost: number;
  removedSourceCount: number;
  retainedSourceCount: number;
  fraction: number;
  percent: number;
}

export interface MinimalContextBudget {
  maxNewExecutions: number;
  usedNewExecutions: number;
  reusedObservationCount: number;
  exhausted: boolean;
  blockedByBudget: boolean;
}

export interface MinimalContextInclusionCheck {
  attempted: boolean;
  complete: boolean;
  testedChildCount: number;
  totalChildCount: number;
  allChildrenFailed: boolean;
}

export interface MinimalContextExperimentAccounting {
  control: { disposition: "reused" | "executed" | "not_executed"; observationId: string | null };
  candidateTrials: number;
  newCounterfactualExecutions: number;
  reusedObservations: number;
  directCandidatesObserved: number;
  preservedCount: number;
  divergedCount: number;
  unknownCount: number;
}

export interface MinimalContextSourceInspection {
  sourceId: string;
  segmentId: string | null;
  messageIndex: number;
  label: string | null;
  provenanceKind: string | null;
  parentSourceId: string | null;
  unicodeRange: readonly [number, number] | null;
  byteRange: readonly [number, number] | null;
  granularity: "exact_span" | "whole_segment";
  text: string | null;
  disposition: "retained" | "removed";
}

export interface MinimalContextProof {
  certificate: MinimalContextCertificate;
  trajectory: readonly MinimalContextTrajectoryEntry[];
  inclusion: {
    attempted: boolean;
    complete: boolean;
    children: readonly Readonly<JsonRecord>[];
  };
  exactMinimum: Readonly<JsonRecord> | null;
}

export interface MinimalContextResult {
  schemaVersion: "clozn.minimal-context-search-result.v2";
  resultId: string;
  runId: string;
  searchId: string;
  status: MinimalContextStatus;
  searchStatus: string | null;
  reason: string | null;
  reasonCode: string | null;
  baseExecutionFingerprint: string;
  universe: MinimalContextUniverse;
  objective: { kind: "rendered_prompt_tokens"; version: "rendered_prompt_tokens.v1" };
  controlObservationId: string | null;
  trials: readonly MinimalContextTrial[];
  trajectory: readonly MinimalContextTrajectoryEntry[];
  original: MinimalContextCandidate | null;
  best: MinimalContextCandidate | null;
  reduction: MinimalContextReduction | null;
  certificate: MinimalContextCertificate | null;
  stoppingReason: MinimalContextStoppingReason;
  experimentAccounting: MinimalContextExperimentAccounting | null;
  sourceInspection: readonly MinimalContextSourceInspection[];
  proof: MinimalContextProof | null;
  policy: {
    kind: "adaptive_bounded_deletion";
    version: "adaptive_bounded_deletion.v1";
    attemptInclusionCheck: boolean;
  };
  budget: MinimalContextBudget;
  inclusionCheck: MinimalContextInclusionCheck;
  currentBinding?: MinimalContextCurrentBinding;
}

/** A result read back from the durable detail route, with its action-gating binding attached. */
export type MinimalContextDurableResult = MinimalContextResult & {
  currentBinding: MinimalContextCurrentBinding;
};

export interface MinimalContextResultSummary {
  schemaVersion: "clozn.minimal-context-search-result.v2";
  resultId: string;
  runId: string;
  searchId: string;
  status: MinimalContextStatus;
  searchStatus: string | null;
  certificate: MinimalContextCertificate | null;
  stoppingReason: MinimalContextStoppingReason;
  best: MinimalContextCandidate | null;
  reduction: MinimalContextReduction | null;
  baseExecutionFingerprint: string;
  currentBinding: MinimalContextCurrentBinding;
}

export interface MinimalContextResultList {
  results: readonly MinimalContextResultSummary[];
}

export interface MinimalContextJob {
  schemaVersion: "clozn.influence-map-job.v1";
  jobId: string;
  runId: string;
  kind: "minimal_context";
  state: MinimalContextJobState;
  progress: {
    phase: string;
    completedUnits: number;
    totalUnits: number;
    percent: number;
    bestRetainedSourceCount?: number;
    certificateCandidateKind?: MinimalContextCertificate;
  };
  cancelRequested: boolean;
  cancellable: boolean;
  cached: boolean;
  cancelAccepted?: boolean;
  createdAt: string;
  updatedAt: string;
  error?: { code: string; message: string };
  result?: MinimalContextResult;
}

export interface MinimalContextStartRequest {
  maxUnits: number;
  maxNewCounterfactualObservations: number;
  attemptInclusionCheck: boolean;
}

export type MinimalContextMaterializationResult =
  | {
      schemaVersion: "clozn.experiment-materialization.v1";
      state: "completed";
      parentRunId: string;
      childRunId: string;
      experimentId: string;
      armId: string;
      removedSourceIds: readonly string[];
      comparePath: string;
    }
  | {
      schemaVersion: "clozn.experiment-materialization.v1";
      state: "failed";
      parentRunId: string;
      experimentId: string;
      armId: string;
      removedSourceIds: readonly string[];
      reason: string;
    };

export class MinimalContextHttpError extends HttpError {
  constructor(
    endpoint: string,
    status: number,
    statusText: string,
    readonly code: string,
    readonly serverMessage: string,
    readonly currentBinding?: MinimalContextCurrentBinding,
  ) {
    super(endpoint, status, statusText);
    this.message = `${serverMessage} (${code}; ${status} ${statusText})`;
  }
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

function list(value: unknown, endpoint: string, path: string): unknown[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function closed(row: JsonRecord, keys: readonly string[], endpoint: string, path: string): void {
  const allowed = new Set(keys);
  for (const key of Object.keys(row)) {
    if (!allowed.has(key)) fail(endpoint, `${path}.${key}`, "is not part of the contract");
  }
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

function nullableString(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : stringValue(value, endpoint, path);
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

function positiveCount(value: unknown, endpoint: string, path: string): number {
  const parsed = count(value, endpoint, path);
  if (parsed < 1) fail(endpoint, path, "must be positive");
  return parsed;
}

function finite(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be finite");
  return value;
}

function fraction(value: unknown, endpoint: string, path: string): number {
  const parsed = finite(value, endpoint, path);
  if (parsed < 0 || parsed > 1) fail(endpoint, path, "must be between 0 and 1");
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

function counts(value: unknown, endpoint: string, path: string): number[] {
  return list(value, endpoint, path).map((item, index) => count(item, endpoint, `${path}[${index}]`));
}

function nullableIdentifier(row: JsonRecord, key: string, endpoint: string, path: string): string | null {
  return nullableText(required(row, key, endpoint, path), endpoint, `${path}.${key}`);
}

function halfOpenRange(value: unknown, endpoint: string, path: string): readonly [number, number] | null {
  if (value === null) return null;
  const parts = list(value, endpoint, path);
  if (parts.length !== 2) fail(endpoint, path, "must contain exactly two offsets");
  const start = count(parts[0], endpoint, `${path}[0]`);
  const end = count(parts[1], endpoint, `${path}[1]`);
  if (end < start) fail(endpoint, path, "must be an ordered half-open range");
  return [start, end];
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function decodeBinding(value: unknown, endpoint: string, path: string): MinimalContextCurrentBinding {
  const row = record(value, endpoint, path);
  closed(row, ["status", "reason"], endpoint, path);
  const status = oneOf(
    required(row, "status", endpoint, path),
    ["current", "stale", "run_unavailable"] as const,
    endpoint,
    `${path}.status`,
  );
  const reason = nullableText(required(row, "reason", endpoint, path), endpoint, `${path}.reason`);
  if (status === "current" && reason !== null) fail(endpoint, `${path}.reason`, "must be null for a current result");
  if (status !== "current" && reason === null) fail(endpoint, `${path}.reason`, `is required for ${status}`);
  return { status, reason };
}

function decodeUniverse(
  value: unknown,
  endpoint: string,
  path: string,
  expectedRunId: string,
): MinimalContextUniverse {
  const row = record(value, endpoint, path);
  closed(row, [
    "schema_version", "run_id", "universe_id", "basis_context_units_digest", "policy",
    "source_ids", "source_count", "coverage", "status", "condition",
  ], endpoint, path);
  const runId = text(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`);
  if (runId !== expectedRunId) fail(endpoint, `${path}.run_id`, "does not match the result Run");
  const universeId = text(required(row, "universe_id", endpoint, path), endpoint, `${path}.universe_id`);
  if (!/^mcu_[0-9a-f]{24}$/.test(universeId)) fail(endpoint, `${path}.universe_id`, "must be an opaque universe id");
  const digest = text(required(row, "basis_context_units_digest", endpoint, path), endpoint, `${path}.basis_context_units_digest`);
  if (!/^[0-9a-f]{64}$/.test(digest)) fail(endpoint, `${path}.basis_context_units_digest`, "must be a lowercase SHA-256 digest");

  const rawPolicy = record(required(row, "policy", endpoint, path), endpoint, `${path}.policy`);
  closed(rawPolicy, ["kind", "version", "max_units", "merge", "explicit_over_cap"], endpoint, `${path}.policy`);
  const sourceIds = strings(required(row, "source_ids", endpoint, path), endpoint, `${path}.source_ids`);
  const sourceIdPattern = /^(?:seg_[0-9a-f]{16}(?:_[0-9]+)?|src_[0-9a-f]{24})$/;
  sourceIds.forEach((sourceId, index) => {
    if (!sourceIdPattern.test(sourceId)) fail(endpoint, `${path}.source_ids[${index}]`, "must be a canonical source id");
  });
  const sourceCount = count(required(row, "source_count", endpoint, path), endpoint, `${path}.source_count`);
  if (sourceCount !== sourceIds.length) fail(endpoint, `${path}.source_count`, "does not match source_ids");

  const rawCoverage = record(required(row, "coverage", endpoint, path), endpoint, `${path}.coverage`);
  closed(rawCoverage, [
    "protected_message_indices", "removable_message_indices", "removable_content_covered",
    "partition_verified", "complete_catalog_source_count", "explicit_source_count",
    "selected_source_count",
  ], endpoint, `${path}.coverage`);
  const protectedMessageIndices = counts(required(rawCoverage, "protected_message_indices", endpoint, `${path}.coverage`), endpoint, `${path}.coverage.protected_message_indices`);
  const removableMessageIndices = counts(required(rawCoverage, "removable_message_indices", endpoint, `${path}.coverage`), endpoint, `${path}.coverage.removable_message_indices`);
  if (new Set(protectedMessageIndices).size !== protectedMessageIndices.length) fail(endpoint, `${path}.coverage.protected_message_indices`, "must not contain duplicates");
  if (new Set(removableMessageIndices).size !== removableMessageIndices.length) fail(endpoint, `${path}.coverage.removable_message_indices`, "must not contain duplicates");
  const removableContentCovered = booleanValue(required(rawCoverage, "removable_content_covered", endpoint, `${path}.coverage`), endpoint, `${path}.coverage.removable_content_covered`);
  const partitionVerified = booleanValue(required(rawCoverage, "partition_verified", endpoint, `${path}.coverage`), endpoint, `${path}.coverage.partition_verified`);
  if (!removableContentCovered || !partitionVerified) fail(endpoint, `${path}.coverage`, "must claim the schema-required verified complete partition");
  const selectedSourceCount = "selected_source_count" in rawCoverage
    ? count(rawCoverage.selected_source_count, endpoint, `${path}.coverage.selected_source_count`)
    : undefined;
  if (selectedSourceCount !== undefined && selectedSourceCount !== sourceCount) {
    fail(endpoint, `${path}.coverage.selected_source_count`, "does not match source_count");
  }

  const status = oneOf(required(row, "status", endpoint, path), ["planned", "bound_exceeded"] as const, endpoint, `${path}.status`);
  const rawCondition = row.condition;
  const condition = rawCondition === undefined ? undefined : (() => {
    const conditionRow = record(rawCondition, endpoint, `${path}.condition`);
    closed(conditionRow, ["code", "message"], endpoint, `${path}.condition`);
    return {
      code: text(required(conditionRow, "code", endpoint, `${path}.condition`), endpoint, `${path}.condition.code`),
      message: text(required(conditionRow, "message", endpoint, `${path}.condition`), endpoint, `${path}.condition.message`),
    };
  })();
  if (status === "planned" && condition !== undefined) fail(endpoint, `${path}.condition`, "must be omitted for a planned universe");
  if (status === "bound_exceeded" && condition === undefined) fail(endpoint, `${path}.condition`, "is required when the bound is exceeded");

  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, path), ["clozn.context-search-universe.v1"] as const, endpoint, `${path}.schema_version`),
    runId,
    universeId,
    basisContextUnitsDigest: digest,
    policy: {
      kind: oneOf(required(rawPolicy, "kind", endpoint, `${path}.policy`), ["bounded_structural_partition"] as const, endpoint, `${path}.policy.kind`),
      version: oneOf(required(rawPolicy, "version", endpoint, `${path}.policy`), ["bounded_structural_partition.v1"] as const, endpoint, `${path}.policy.version`),
      maxUnits: positiveCount(required(rawPolicy, "max_units", endpoint, `${path}.policy`), endpoint, `${path}.policy.max_units`),
      merge: oneOf(required(rawPolicy, "merge", endpoint, `${path}.policy`), ["smallest_adjacent_auto_siblings_leftmost.v1"] as const, endpoint, `${path}.policy.merge`),
      explicitOverCap: oneOf(required(rawPolicy, "explicit_over_cap", endpoint, `${path}.policy`), ["typed_bound_exceeded"] as const, endpoint, `${path}.policy.explicit_over_cap`),
    },
    sourceIds,
    sourceCount,
    coverage: {
      protectedMessageIndices,
      removableMessageIndices,
      removableContentCovered: true,
      partitionVerified: true,
      completeCatalogSourceCount: count(required(rawCoverage, "complete_catalog_source_count", endpoint, `${path}.coverage`), endpoint, `${path}.coverage.complete_catalog_source_count`),
      explicitSourceCount: count(required(rawCoverage, "explicit_source_count", endpoint, `${path}.coverage`), endpoint, `${path}.coverage.explicit_source_count`),
      ...(selectedSourceCount === undefined ? {} : { selectedSourceCount }),
    },
    status,
    condition,
  };
}

function decodeCandidate(value: unknown, endpoint: string, path: string): MinimalContextCandidate {
  const row = record(value, endpoint, path);
  const retainedSourceIds = strings(required(row, "retained_source_ids", endpoint, path), endpoint, `${path}.retained_source_ids`);
  const removedSourceIds = strings(required(row, "removed_source_ids", endpoint, path), endpoint, `${path}.removed_source_ids`);
  if (retainedSourceIds.some((sourceId) => removedSourceIds.includes(sourceId))) {
    fail(endpoint, path, "retained and removed source sets must be disjoint");
  }
  return {
    retainedSourceIds,
    removedSourceIds,
    renderedPromptTokenCost: count(required(row, "rendered_prompt_token_cost", endpoint, path), endpoint, `${path}.rendered_prompt_token_cost`),
    experimentId: nullableIdentifier(row, "experiment_id", endpoint, path),
    armId: nullableIdentifier(row, "arm_id", endpoint, path),
    observationId: nullableIdentifier(row, "observation_id", endpoint, path),
    observationStatus: nullableIdentifier(row, "observation_status", endpoint, path),
  };
}

function decodeOptionalCandidate(value: unknown, endpoint: string, path: string): MinimalContextCandidate | null {
  if (value === null) return null;
  const row = record(value, endpoint, path);
  if (Object.keys(row).length === 0) return null;
  return decodeCandidate(row, endpoint, path);
}

function decodeEvidence(value: unknown, endpoint: string, path: string): MinimalContextEvidenceRef | null {
  if (value === null) return null;
  const row = record(value, endpoint, path);
  return {
    disposition: oneOf(required(row, "disposition", endpoint, path), ["reused", "executed", "not_executed"] as const, endpoint, `${path}.disposition`),
    experimentId: "experiment_id" in row ? nullableText(row.experiment_id, endpoint, `${path}.experiment_id`) : null,
    armId: "arm_id" in row ? nullableText(row.arm_id, endpoint, `${path}.arm_id`) : null,
    observationId: "observation_id" in row ? nullableText(row.observation_id, endpoint, `${path}.observation_id`) : null,
    observationStatus: "observation_status" in row ? nullableText(row.observation_status, endpoint, `${path}.observation_status`) : null,
  };
}

function decodeTrial(value: unknown, endpoint: string, path: string): MinimalContextTrial {
  const row = record(value, endpoint, path);
  const evidence = decodeEvidence(required(row, "evidence", endpoint, path), endpoint, `${path}.evidence`);
  const experimentId = nullableIdentifier(row, "experiment_id", endpoint, path);
  const armId = nullableIdentifier(row, "arm_id", endpoint, path);
  const observationId = nullableIdentifier(row, "observation_id", endpoint, path);
  const observationStatus = nullableIdentifier(row, "observation_status", endpoint, path);
  if (evidence && (
    evidence.experimentId !== experimentId || evidence.armId !== armId
    || evidence.observationId !== observationId || evidence.observationStatus !== observationStatus
  )) fail(endpoint, path, "top-level evidence references disagree with evidence");
  const disposition = oneOf(required(row, "disposition", endpoint, path), ["reused", "executed", "not_executed"] as const, endpoint, `${path}.disposition`);
  if (evidence && evidence.disposition !== disposition) fail(endpoint, path, "evidence disposition disagrees with trial disposition");
  const rawBatchId = required(row, "batch_id", endpoint, path);
  return {
    ordinal: count(required(row, "ordinal", endpoint, path), endpoint, `${path}.ordinal`),
    stage: text(required(row, "stage", endpoint, path), endpoint, `${path}.stage`),
    retainedIds: strings(required(row, "retained_ids", endpoint, path), endpoint, `${path}.retained_ids`),
    cost: count(required(row, "cost", endpoint, path), endpoint, `${path}.cost`),
    classification: oneOf(required(row, "classification", endpoint, path), ["preserves", "diverged", "unknown"] as const, endpoint, `${path}.classification`),
    disposition,
    experimentId,
    armId,
    observationId,
    observationStatus,
    evidence,
    batchId: rawBatchId === null ? null : count(rawBatchId, endpoint, `${path}.batch_id`),
    parentRetainedIds: strings(required(row, "parent_retained_ids", endpoint, path), endpoint, `${path}.parent_retained_ids`),
  };
}

function decodeTrajectory(value: unknown, endpoint: string, path: string): MinimalContextTrajectoryEntry {
  const row = record(value, endpoint, path);
  const retainedIds = strings(required(row, "retained_ids", endpoint, path), endpoint, `${path}.retained_ids`);
  const retainedUnitCount = count(required(row, "retained_unit_count", endpoint, path), endpoint, `${path}.retained_unit_count`);
  if (retainedUnitCount !== retainedIds.length) fail(endpoint, `${path}.retained_unit_count`, "does not match retained_ids");
  return {
    counterfactualProbeCount: count(required(row, "counterfactual_probe_count", endpoint, path), endpoint, `${path}.counterfactual_probe_count`),
    retainedIds,
    cost: count(required(row, "cost", endpoint, path), endpoint, `${path}.cost`),
    retainedUnitCount,
    stage: text(required(row, "stage", endpoint, path), endpoint, `${path}.stage`),
  };
}

function decodeReduction(value: unknown, endpoint: string, path: string): MinimalContextReduction | null {
  const row = record(value, endpoint, path);
  if (Object.keys(row).length === 0) return null;
  const parsedFraction = fraction(required(row, "fraction", endpoint, path), endpoint, `${path}.fraction`);
  const percent = finite(required(row, "percent", endpoint, path), endpoint, `${path}.percent`);
  if (percent < 0 || percent > 100 || Math.abs(percent - parsedFraction * 100) > 1e-8) {
    fail(endpoint, `${path}.percent`, "must match fraction");
  }
  const originalPromptTokenCost = count(required(row, "original_prompt_token_cost", endpoint, path), endpoint, `${path}.original_prompt_token_cost`);
  const retainedPromptTokenCost = count(required(row, "retained_prompt_token_cost", endpoint, path), endpoint, `${path}.retained_prompt_token_cost`);
  if (retainedPromptTokenCost > originalPromptTokenCost) fail(endpoint, path, "retained cost must not exceed original cost");
  return {
    objective: oneOf(required(row, "objective", endpoint, path), ["rendered_prompt_tokens.v1"] as const, endpoint, `${path}.objective`),
    originalPromptTokenCost,
    retainedPromptTokenCost,
    removedSourceCount: count(required(row, "removed_source_count", endpoint, path), endpoint, `${path}.removed_source_count`),
    retainedSourceCount: count(required(row, "retained_source_count", endpoint, path), endpoint, `${path}.retained_source_count`),
    fraction: parsedFraction,
    percent,
  };
}

function decodeBudget(value: unknown, endpoint: string, path: string): MinimalContextBudget {
  const row = record(value, endpoint, path);
  const maxNewExecutions = count(required(row, "max_new_executions", endpoint, path), endpoint, `${path}.max_new_executions`);
  const usedNewExecutions = count(required(row, "used_new_executions", endpoint, path), endpoint, `${path}.used_new_executions`);
  if (usedNewExecutions > maxNewExecutions) fail(endpoint, path, "used executions must not exceed the configured budget");
  return {
    maxNewExecutions,
    usedNewExecutions,
    reusedObservationCount: count(required(row, "reused_observation_count", endpoint, path), endpoint, `${path}.reused_observation_count`),
    exhausted: booleanValue(required(row, "exhausted", endpoint, path), endpoint, `${path}.exhausted`),
    blockedByBudget: booleanValue(required(row, "blocked_by_budget", endpoint, path), endpoint, `${path}.blocked_by_budget`),
  };
}

function decodeInclusion(value: unknown, endpoint: string, path: string): MinimalContextInclusionCheck {
  const row = record(value, endpoint, path);
  const testedChildCount = count(required(row, "tested_child_count", endpoint, path), endpoint, `${path}.tested_child_count`);
  const totalChildCount = count(required(row, "total_child_count", endpoint, path), endpoint, `${path}.total_child_count`);
  if (testedChildCount > totalChildCount) fail(endpoint, path, "tested children must not exceed total children");
  const attempted = booleanValue(required(row, "attempted", endpoint, path), endpoint, `${path}.attempted`);
  const complete = booleanValue(required(row, "complete", endpoint, path), endpoint, `${path}.complete`);
  if (complete && testedChildCount !== totalChildCount) fail(endpoint, path, "a complete check must account for every child");
  if (!attempted && (complete || testedChildCount !== 0)) fail(endpoint, path, "an unattempted check cannot claim tested children");
  return {
    attempted,
    complete,
    testedChildCount,
    totalChildCount,
    allChildrenFailed: booleanValue(required(row, "all_children_failed", endpoint, path), endpoint, `${path}.all_children_failed`),
  };
}

function decodeAccounting(value: unknown, endpoint: string, path: string): MinimalContextExperimentAccounting | null {
  const row = record(value, endpoint, path);
  if (Object.keys(row).length === 0) return null;
  const control = record(required(row, "control", endpoint, path), endpoint, `${path}.control`);
  return {
    control: {
      disposition: oneOf(required(control, "disposition", endpoint, `${path}.control`), ["reused", "executed", "not_executed"] as const, endpoint, `${path}.control.disposition`),
      observationId: nullableText(required(control, "observation_id", endpoint, `${path}.control`), endpoint, `${path}.control.observation_id`),
    },
    candidateTrials: count(required(row, "candidate_trials", endpoint, path), endpoint, `${path}.candidate_trials`),
    newCounterfactualExecutions: count(required(row, "new_counterfactual_executions", endpoint, path), endpoint, `${path}.new_counterfactual_executions`),
    reusedObservations: count(required(row, "reused_observations", endpoint, path), endpoint, `${path}.reused_observations`),
    directCandidatesObserved: count(required(row, "direct_candidates_observed", endpoint, path), endpoint, `${path}.direct_candidates_observed`),
    preservedCount: count(required(row, "preserved_count", endpoint, path), endpoint, `${path}.preserved_count`),
    divergedCount: count(required(row, "diverged_count", endpoint, path), endpoint, `${path}.diverged_count`),
    unknownCount: count(required(row, "unknown_count", endpoint, path), endpoint, `${path}.unknown_count`),
  };
}

function decodeSourceInspection(value: unknown, endpoint: string, path: string): MinimalContextSourceInspection {
  const row = record(value, endpoint, path);
  return {
    sourceId: text(required(row, "source_id", endpoint, path), endpoint, `${path}.source_id`),
    segmentId: nullableText(required(row, "segment_id", endpoint, path), endpoint, `${path}.segment_id`),
    messageIndex: count(required(row, "message_index", endpoint, path), endpoint, `${path}.message_index`),
    label: nullableString(required(row, "label", endpoint, path), endpoint, `${path}.label`),
    provenanceKind: nullableString(required(row, "provenance_kind", endpoint, path), endpoint, `${path}.provenance_kind`),
    parentSourceId: nullableText(required(row, "parent_source_id", endpoint, path), endpoint, `${path}.parent_source_id`),
    unicodeRange: halfOpenRange(required(row, "unicode_range", endpoint, path), endpoint, `${path}.unicode_range`),
    byteRange: halfOpenRange(required(row, "byte_range", endpoint, path), endpoint, `${path}.byte_range`),
    granularity: oneOf(required(row, "granularity", endpoint, path), ["exact_span", "whole_segment"] as const, endpoint, `${path}.granularity`),
    text: nullableString(required(row, "text", endpoint, path), endpoint, `${path}.text`),
    disposition: oneOf(required(row, "disposition", endpoint, path), ["retained", "removed"] as const, endpoint, `${path}.disposition`),
  };
}

function decodeProof(value: unknown, endpoint: string, path: string): MinimalContextProof | null {
  const row = record(value, endpoint, path);
  if (Object.keys(row).length === 0) return null;
  const inclusion = record(required(row, "inclusion", endpoint, path), endpoint, `${path}.inclusion`);
  const rawExact = required(row, "exact_minimum", endpoint, path);
  return {
    certificate: oneOf(required(row, "certificate", endpoint, path), ["BEST_VERIFIED", "INCLUSION_MINIMUM", "EXACT_MINIMUM"] as const, endpoint, `${path}.certificate`),
    trajectory: list(required(row, "trajectory", endpoint, path), endpoint, `${path}.trajectory`)
      .map((entry, index) => decodeTrajectory(entry, endpoint, `${path}.trajectory[${index}]`)),
    inclusion: {
      attempted: booleanValue(required(inclusion, "attempted", endpoint, `${path}.inclusion`), endpoint, `${path}.inclusion.attempted`),
      complete: booleanValue(required(inclusion, "complete", endpoint, `${path}.inclusion`), endpoint, `${path}.inclusion.complete`),
      children: list(required(inclusion, "children", endpoint, `${path}.inclusion`), endpoint, `${path}.inclusion.children`)
        .map((child, index) => record(child, endpoint, `${path}.inclusion.children[${index}]`)),
    },
    exactMinimum: rawExact === null ? null : record(rawExact, endpoint, `${path}.exact_minimum`),
  };
}

const certificates = ["BEST_VERIFIED", "INCLUSION_MINIMUM", "EXACT_MINIMUM"] as const;
const stoppingReasons = [
  "exact_minimum_proven", "inclusion_minimum_proven", "budget_exhausted",
  "search_policy_complete", "control_unavailable", "search_unavailable", "cancelled",
] as const;

export function decodeMinimalContextResult(
  value: unknown,
  endpoint = "/runs/<id>/minimal-context/results/<result-id>",
  expectedRunId?: string,
  requireCurrentBinding = false,
): MinimalContextResult {
  const path = "$";
  const row = record(value, endpoint, path);
  closed(row, [
    "schema_version", "result_id", "run_id", "search_id", "status", "search_status", "reason",
    "reason_code", "base_execution_fingerprint", "universe", "objective", "control_observation_id",
    "trials", "trajectory", "original", "best", "reduction", "certificate", "stopping_reason",
    "experiment_accounting", "source_inspection", "proof", "policy", "budget", "inclusion_check",
    "current_binding",
  ], endpoint, path);
  const runId = text(required(row, "run_id", endpoint, path), endpoint, "$.run_id");
  if (expectedRunId !== undefined && runId !== expectedRunId) fail(endpoint, "$.run_id", "does not match the requested Run");
  const resultId = text(required(row, "result_id", endpoint, path), endpoint, "$.result_id");
  if (!/^mcres_[0-9a-f]{24}$/.test(resultId)) fail(endpoint, "$.result_id", "must be an opaque Minimal Context result id");
  const status = oneOf(required(row, "status", endpoint, path), ["completed", "unavailable", "cancelled"] as const, endpoint, "$.status");
  const universe = decodeUniverse(required(row, "universe", endpoint, path), endpoint, "$.universe", runId);
  const rawObjective = record(required(row, "objective", endpoint, path), endpoint, "$.objective");
  const objective = {
    kind: oneOf(required(rawObjective, "kind", endpoint, "$.objective"), ["rendered_prompt_tokens"] as const, endpoint, "$.objective.kind"),
    version: oneOf(required(rawObjective, "version", endpoint, "$.objective"), ["rendered_prompt_tokens.v1"] as const, endpoint, "$.objective.version"),
  };
  const original = decodeOptionalCandidate(required(row, "original", endpoint, path), endpoint, "$.original");
  const best = decodeOptionalCandidate(required(row, "best", endpoint, path), endpoint, "$.best");
  const reduction = decodeReduction(required(row, "reduction", endpoint, path), endpoint, "$.reduction");
  const rawCertificate = required(row, "certificate", endpoint, path);
  const certificate = rawCertificate === null ? null : oneOf(rawCertificate, certificates, endpoint, "$.certificate");
  const trajectory = list(required(row, "trajectory", endpoint, path), endpoint, "$.trajectory")
    .map((entry, index) => decodeTrajectory(entry, endpoint, `$.trajectory[${index}]`));
  const proof = decodeProof(required(row, "proof", endpoint, path), endpoint, "$.proof");
  const rawPolicy = record(required(row, "policy", endpoint, path), endpoint, "$.policy");
  const currentBinding = "current_binding" in row
    ? decodeBinding(row.current_binding, endpoint, "$.current_binding")
    : undefined;
  if (requireCurrentBinding && currentBinding === undefined) fail(endpoint, "$.current_binding", "is required for a durable result read");

  if (status === "completed" && (best === null || original === null || reduction === null || certificate === null)) {
    fail(endpoint, "$", "a completed result requires original, best, reduction, and certificate evidence");
  }
  if (best) {
    const partition = [...best.retainedSourceIds, ...best.removedSourceIds];
    if (partition.length !== universe.sourceIds.length || new Set(partition).size !== partition.length
        || universe.sourceIds.some((sourceId) => !partition.includes(sourceId))) {
      fail(endpoint, "$.best", "source sets do not partition the recorded universe");
    }
  }
  if (original && (!sameStrings(original.retainedSourceIds, universe.sourceIds) || original.removedSourceIds.length !== 0)) {
    fail(endpoint, "$.original", "must describe the full recorded universe");
  }
  if (best && reduction && (
    reduction.originalPromptTokenCost !== original?.renderedPromptTokenCost
    || reduction.retainedPromptTokenCost !== best.renderedPromptTokenCost
    || reduction.retainedSourceCount !== best.retainedSourceIds.length
    || reduction.removedSourceCount !== best.removedSourceIds.length
  )) fail(endpoint, "$.reduction", "does not match the recorded candidates");
  if (proof && (proof.certificate !== certificate || proof.trajectory.length !== trajectory.length)) {
    fail(endpoint, "$.proof", "does not match the result certificate and trajectory");
  }

  const sourceInspection = list(required(row, "source_inspection", endpoint, path), endpoint, "$.source_inspection")
    .map((entry, index) => decodeSourceInspection(entry, endpoint, `$.source_inspection[${index}]`));
  if (new Set(sourceInspection.map((entry) => entry.sourceId)).size !== sourceInspection.length
      || sourceInspection.some((entry) => !universe.sourceIds.includes(entry.sourceId))) {
    fail(endpoint, "$.source_inspection", "must contain unique sources from the recorded universe");
  }
  if (best && sourceInspection.some((entry) =>
    entry.disposition !== (best.retainedSourceIds.includes(entry.sourceId) ? "retained" : "removed"))) {
    fail(endpoint, "$.source_inspection", "dispositions do not match the recorded winner");
  }

  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, path), ["clozn.minimal-context-search-result.v2"] as const, endpoint, "$.schema_version"),
    resultId,
    runId,
    searchId: text(required(row, "search_id", endpoint, path), endpoint, "$.search_id"),
    status,
    searchStatus: nullableText(required(row, "search_status", endpoint, path), endpoint, "$.search_status"),
    reason: nullableText(required(row, "reason", endpoint, path), endpoint, "$.reason"),
    reasonCode: nullableText(required(row, "reason_code", endpoint, path), endpoint, "$.reason_code"),
    baseExecutionFingerprint: text(required(row, "base_execution_fingerprint", endpoint, path), endpoint, "$.base_execution_fingerprint"),
    universe,
    objective,
    controlObservationId: nullableText(required(row, "control_observation_id", endpoint, path), endpoint, "$.control_observation_id"),
    trials: list(required(row, "trials", endpoint, path), endpoint, "$.trials")
      .map((entry, index) => decodeTrial(entry, endpoint, `$.trials[${index}]`)),
    trajectory,
    original,
    best,
    reduction,
    certificate,
    stoppingReason: oneOf(required(row, "stopping_reason", endpoint, path), stoppingReasons, endpoint, "$.stopping_reason"),
    experimentAccounting: decodeAccounting(required(row, "experiment_accounting", endpoint, path), endpoint, "$.experiment_accounting"),
    sourceInspection,
    proof,
    policy: {
      kind: oneOf(required(rawPolicy, "kind", endpoint, "$.policy"), ["adaptive_bounded_deletion"] as const, endpoint, "$.policy.kind"),
      version: oneOf(required(rawPolicy, "version", endpoint, "$.policy"), ["adaptive_bounded_deletion.v1"] as const, endpoint, "$.policy.version"),
      attemptInclusionCheck: booleanValue(required(rawPolicy, "attempt_inclusion_check", endpoint, "$.policy"), endpoint, "$.policy.attempt_inclusion_check"),
    },
    budget: decodeBudget(required(row, "budget", endpoint, path), endpoint, "$.budget"),
    inclusionCheck: decodeInclusion(required(row, "inclusion_check", endpoint, path), endpoint, "$.inclusion_check"),
    currentBinding,
  };
}

function decodeSummary(
  value: unknown,
  endpoint: string,
  path: string,
  expectedRunId: string,
): MinimalContextResultSummary {
  const row = record(value, endpoint, path);
  closed(row, [
    "schema_version", "result_id", "run_id", "search_id", "status", "search_status", "certificate",
    "stopping_reason", "best", "reduction", "base_execution_fingerprint", "current_binding",
  ], endpoint, path);
  const runId = text(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`);
  if (runId !== expectedRunId) fail(endpoint, `${path}.run_id`, "does not match the requested Run");
  const resultId = text(required(row, "result_id", endpoint, path), endpoint, `${path}.result_id`);
  if (!/^mcres_[0-9a-f]{24}$/.test(resultId)) fail(endpoint, `${path}.result_id`, "must be an opaque Minimal Context result id");
  const rawCertificate = required(row, "certificate", endpoint, path);
  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, path), ["clozn.minimal-context-search-result.v2"] as const, endpoint, `${path}.schema_version`),
    resultId,
    runId,
    searchId: text(required(row, "search_id", endpoint, path), endpoint, `${path}.search_id`),
    status: oneOf(required(row, "status", endpoint, path), ["completed", "unavailable", "cancelled"] as const, endpoint, `${path}.status`),
    searchStatus: nullableText(required(row, "search_status", endpoint, path), endpoint, `${path}.search_status`),
    certificate: rawCertificate === null ? null : oneOf(rawCertificate, certificates, endpoint, `${path}.certificate`),
    stoppingReason: oneOf(required(row, "stopping_reason", endpoint, path), stoppingReasons, endpoint, `${path}.stopping_reason`),
    best: decodeOptionalCandidate(required(row, "best", endpoint, path), endpoint, `${path}.best`),
    reduction: decodeReduction(required(row, "reduction", endpoint, path), endpoint, `${path}.reduction`),
    baseExecutionFingerprint: text(required(row, "base_execution_fingerprint", endpoint, path), endpoint, `${path}.base_execution_fingerprint`),
    currentBinding: decodeBinding(required(row, "current_binding", endpoint, path), endpoint, `${path}.current_binding`),
  };
}

export function decodeMinimalContextResultList(
  value: unknown,
  endpoint = "/runs/<id>/minimal-context/results",
  expectedRunId: string,
): MinimalContextResultList {
  const row = record(value, endpoint, "$ ".trim());
  closed(row, ["results"], endpoint, "$");
  const results = list(required(row, "results", endpoint, "$"), endpoint, "$.results")
    .map((entry, index) => decodeSummary(entry, endpoint, `$.results[${index}]`, expectedRunId));
  if (new Set(results.map((result) => result.resultId)).size !== results.length) {
    fail(endpoint, "$.results", "must not contain duplicate result IDs");
  }
  return { results };
}

export function decodeMinimalContextJob(
  value: unknown,
  endpoint: string,
  expectedRunId: string,
  expectedJobId?: string,
): MinimalContextJob {
  const row = record(value, endpoint, "$ ".trim());
  closed(row, [
    "schema_version", "job_id", "run_id", "kind", "state", "progress", "cancel_requested",
    "cancellable", "cached", "cancel_accepted", "created_at", "updated_at", "error", "result",
  ], endpoint, "$");
  const jobId = text(required(row, "job_id", endpoint, "$"), endpoint, "$.job_id");
  if (expectedJobId !== undefined && jobId !== expectedJobId) fail(endpoint, "$.job_id", "does not match the requested job");
  const runId = text(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.run_id", "does not match the requested Run");
  const state = oneOf(required(row, "state", endpoint, "$"), ["queued", "running", "cancelling", "persisting", "completed", "failed", "cancelled"] as const, endpoint, "$.state");
  const rawProgress = record(required(row, "progress", endpoint, "$"), endpoint, "$.progress");
  closed(rawProgress, ["phase", "completed_units", "total_units", "percent", "best_retained_source_count", "certificate_candidate_kind"], endpoint, "$.progress");
  const completedUnits = count(required(rawProgress, "completed_units", endpoint, "$.progress"), endpoint, "$.progress.completed_units");
  const totalUnits = count(required(rawProgress, "total_units", endpoint, "$.progress"), endpoint, "$.progress.total_units");
  if (completedUnits > totalUnits) fail(endpoint, "$.progress", "completed_units must not exceed total_units");
  const percent = finite(required(rawProgress, "percent", endpoint, "$.progress"), endpoint, "$.progress.percent");
  if (percent < 0 || percent > 100) fail(endpoint, "$.progress.percent", "must be between 0 and 100");
  const result = "result" in row
    ? decodeMinimalContextResult(row.result, endpoint, expectedRunId)
    : undefined;
  const rawError = "error" in row ? record(row.error, endpoint, "$.error") : undefined;
  if (state === "completed" && result === undefined) fail(endpoint, "$.result", "is required for a completed Minimal Context job");
  if (state === "failed" && rawError === undefined) fail(endpoint, "$.error", "is required for a failed Minimal Context job");
  const cancellable = booleanValue(required(row, "cancellable", endpoint, "$"), endpoint, "$.cancellable");
  if (["completed", "failed", "cancelled"].includes(state) && cancellable) fail(endpoint, "$.cancellable", "must be false for a terminal job");
  const rawBestCount = rawProgress.best_retained_source_count;
  const rawCandidate = rawProgress.certificate_candidate_kind;
  return {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.influence-map-job.v1"] as const, endpoint, "$.schema_version"),
    jobId,
    runId,
    kind: oneOf(required(row, "kind", endpoint, "$"), ["minimal_context"] as const, endpoint, "$.kind"),
    state,
    progress: {
      phase: text(required(rawProgress, "phase", endpoint, "$.progress"), endpoint, "$.progress.phase"),
      completedUnits,
      totalUnits,
      percent,
      ...(rawBestCount === undefined ? {} : { bestRetainedSourceCount: count(rawBestCount, endpoint, "$.progress.best_retained_source_count") }),
      ...(rawCandidate === undefined ? {} : { certificateCandidateKind: oneOf(rawCandidate, certificates, endpoint, "$.progress.certificate_candidate_kind") }),
    },
    cancelRequested: booleanValue(required(row, "cancel_requested", endpoint, "$"), endpoint, "$.cancel_requested"),
    cancellable,
    cached: booleanValue(required(row, "cached", endpoint, "$"), endpoint, "$.cached"),
    ...(row.cancel_accepted === undefined ? {} : { cancelAccepted: booleanValue(row.cancel_accepted, endpoint, "$.cancel_accepted") }),
    createdAt: text(required(row, "created_at", endpoint, "$"), endpoint, "$.created_at"),
    updatedAt: text(required(row, "updated_at", endpoint, "$"), endpoint, "$.updated_at"),
    ...(rawError ? {
      error: {
        code: text(required(rawError, "code", endpoint, "$.error"), endpoint, "$.error.code"),
        message: text(required(rawError, "message", endpoint, "$.error"), endpoint, "$.error.message"),
      },
    } : {}),
    ...(result === undefined ? {} : { result }),
  };
}

export function isTerminalMinimalContextJob(job: MinimalContextJob): boolean {
  return job.state === "completed" || job.state === "failed" || job.state === "cancelled";
}

export function minimalContextMaterializationBlocker(result: MinimalContextResult): string | null {
  if (result.currentBinding === undefined) return "Reload this durable result before materializing it.";
  if (result.currentBinding.status !== "current") return result.currentBinding.reason ?? "This result is not current.";
  if (result.status !== "completed") return "Only a completed Minimal Context result can be materialized.";
  if (result.certificate === null || result.best === null) return "This result has no verified winner.";
  if (result.best.removedSourceIds.length === 0) return "The verified winner does not reduce the context.";
  if (!result.best.experimentId || !result.best.armId || !result.best.observationId) {
    return "The winner is missing durable experiment evidence.";
  }
  if (result.best.observationStatus !== "exact_preserved") {
    return "The winner is not bound to an exact-preservation observation.";
  }
  return null;
}

export function decodeMinimalContextMaterializationResult(
  value: unknown,
  endpoint: string,
  expectedParentRunId: string,
  result: MinimalContextResult,
): MinimalContextMaterializationResult {
  const row = record(value, endpoint, "$ ".trim());
  const state = oneOf(required(row, "state", endpoint, "$"), ["completed", "failed"] as const, endpoint, "$.state");
  closed(row, state === "completed" ? [
    "schema_version", "state", "parent_run_id", "child_run_id", "experiment_id", "arm_id",
    "intervention", "comparison", "compare_path",
  ] : [
    "schema_version", "state", "parent_run_id", "experiment_id", "arm_id", "intervention", "reason",
  ], endpoint, "$");
  const parentRunId = text(required(row, "parent_run_id", endpoint, "$"), endpoint, "$.parent_run_id");
  if (parentRunId !== expectedParentRunId) fail(endpoint, "$.parent_run_id", "does not match the requested Run");
  const best = result.best;
  if (!best?.experimentId || !best.armId || !best.observationId) fail(endpoint, "$", "the source result has no materializable winner evidence");
  const experimentId = text(required(row, "experiment_id", endpoint, "$"), endpoint, "$.experiment_id");
  const armId = text(required(row, "arm_id", endpoint, "$"), endpoint, "$.arm_id");
  if (experimentId !== best.experimentId || armId !== best.armId) {
    fail(endpoint, "$", "materialized evidence does not match the persisted result winner");
  }
  const intervention = record(required(row, "intervention", endpoint, "$"), endpoint, "$.intervention");
  closed(intervention, ["kind", "source_ids"], endpoint, "$.intervention");
  oneOf(required(intervention, "kind", endpoint, "$.intervention"), ["delete_source"] as const, endpoint, "$.intervention.kind");
  const removedSourceIds = strings(required(intervention, "source_ids", endpoint, "$.intervention"), endpoint, "$.intervention.source_ids");
  if (!sameStrings(removedSourceIds, best.removedSourceIds)) {
    fail(endpoint, "$.intervention.source_ids", "does not match the persisted result winner");
  }
  const schemaVersion = oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.experiment-materialization.v1"] as const, endpoint, "$.schema_version");
  if (state === "failed") {
    return {
      schemaVersion,
      state,
      parentRunId,
      experimentId,
      armId,
      removedSourceIds,
      reason: text(required(row, "reason", endpoint, "$"), endpoint, "$.reason"),
    };
  }
  const childRunId = text(required(row, "child_run_id", endpoint, "$"), endpoint, "$.child_run_id");
  if (childRunId === parentRunId) fail(endpoint, "$.child_run_id", "must identify a child, not the source Run");
  const rawComparison = required(row, "comparison", endpoint, "$"), comparisonPath = "$.comparison";
  if (rawComparison !== null) record(rawComparison, endpoint, comparisonPath);
  const comparePath = text(required(row, "compare_path", endpoint, "$"), endpoint, "$.compare_path");
  if (comparePath !== `#/compare/${parentRunId}/${childRunId}`) fail(endpoint, "$.compare_path", "does not match the returned lineage");
  return {
    schemaVersion,
    state,
    parentRunId,
    childRunId,
    experimentId,
    armId,
    removedSourceIds,
    comparePath,
  };
}

function identifier(value: string, name: string): string {
  if (!value.trim()) throw new ContractError("client", `${name} must not be blank`);
  return value;
}

function runPath(runId: string, suffix: string): string {
  return `/runs/${encodeURIComponent(identifier(runId, "run id"))}/minimal-context/${suffix}`;
}

async function responseJson(
  response: Response,
  endpoint: string,
  acceptedStatuses: readonly number[] = [200],
): Promise<unknown> {
  const body = await parseResponseBody(response, endpoint, acceptedStatuses.includes(response.status));
  if (!acceptedStatuses.includes(response.status)) throwResponseError(response, endpoint, body);
  return body;
}

async function parseResponseBody(response: Response, endpoint: string, requiredJson: boolean): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    if (requiredJson) throw new ContractError(endpoint, "response is not valid JSON");
    return undefined;
  }
}

function throwResponseError(response: Response, endpoint: string, body: unknown): never {
  if (body !== null && typeof body === "object" && !Array.isArray(body)) {
    const error = (body as JsonRecord).error;
    const code = (body as JsonRecord).code;
    if (typeof error === "string" && error.trim() && typeof code === "string" && code.trim()) {
      const binding = "current_binding" in body
        ? decodeBinding((body as JsonRecord).current_binding, endpoint, "$.current_binding")
        : undefined;
      throw new MinimalContextHttpError(endpoint, response.status, response.statusText, code, error, binding);
    }
  }
  throw new HttpError(endpoint, response.status, response.statusText);
}

function startBody(request: MinimalContextStartRequest): string {
  if (!Number.isSafeInteger(request.maxUnits) || request.maxUnits < 1) {
    throw new ContractError("client", "maxUnits must be a positive integer");
  }
  if (!Number.isSafeInteger(request.maxNewCounterfactualObservations)
      || request.maxNewCounterfactualObservations < 0) {
    throw new ContractError("client", "maxNewCounterfactualObservations must be a non-negative integer");
  }
  if (typeof request.attemptInclusionCheck !== "boolean") {
    throw new ContractError("client", "attemptInclusionCheck must be a boolean");
  }
  return JSON.stringify({
    preservation: { kind: "exact_recorded_output" },
    universe: { max_units: request.maxUnits },
    max_new_counterfactual_observations: request.maxNewCounterfactualObservations,
    attempt_inclusion_check: request.attemptInclusionCheck,
  });
}

export const minimalContextApi = {
  async results(runId: string, signal?: AbortSignal): Promise<MinimalContextResultList> {
    const endpoint = runPath(runId, "results");
    const response = await fetch(endpoint, { method: "GET", signal, headers: { Accept: "application/json" } });
    return decodeMinimalContextResultList(await responseJson(response, endpoint), endpoint, runId);
  },

  async result(runId: string, resultId: string, signal?: AbortSignal): Promise<MinimalContextDurableResult> {
    identifier(resultId, "result id");
    const endpoint = runPath(runId, `results/${encodeURIComponent(resultId)}`);
    const response = await fetch(endpoint, { method: "GET", signal, headers: { Accept: "application/json" } });
    const result = decodeMinimalContextResult(await responseJson(response, endpoint), endpoint, runId, true);
    if (result.resultId !== resultId) throw new ContractError(endpoint, "$.result_id does not match the requested result");
    return result as MinimalContextDurableResult;
  },

  async start(
    runId: string,
    request: MinimalContextStartRequest,
    signal?: AbortSignal,
  ): Promise<MinimalContextJob> {
    const endpoint = runPath(runId, "jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: startBody(request),
    });
    return decodeMinimalContextJob(await responseJson(response, endpoint, [202]), endpoint, runId);
  },

  async job(runId: string, jobId: string, signal?: AbortSignal): Promise<MinimalContextJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, { method: "GET", signal, headers: { Accept: "application/json" } });
    return decodeMinimalContextJob(await responseJson(response, endpoint), endpoint, runId, jobId);
  },

  async cancel(runId: string, jobId: string, signal?: AbortSignal): Promise<MinimalContextJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeMinimalContextJob(await responseJson(response, endpoint), endpoint, runId, jobId);
  },

  async materialize(
    runId: string,
    result: MinimalContextResult,
    signal?: AbortSignal,
  ): Promise<MinimalContextMaterializationResult> {
    if (result.runId !== runId) throw new ContractError("client", "result does not belong to the requested Run");
    const blocker = minimalContextMaterializationBlocker(result);
    if (blocker !== null) throw new ContractError("client", blocker);
    const endpoint = runPath(runId, `results/${encodeURIComponent(result.resultId)}/materialize`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    const body = await parseResponseBody(response, endpoint, response.status === 201 || response.status === 409);
    if (response.status !== 201) {
      if (response.status !== 409
          || body === null || typeof body !== "object" || Array.isArray(body)
          || (body as JsonRecord).state !== "failed") {
        throwResponseError(response, endpoint, body);
      }
    }
    return decodeMinimalContextMaterializationResult(body, endpoint, runId, result);
  },
};
