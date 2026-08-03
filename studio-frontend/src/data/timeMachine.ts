import { num, record, records, str } from "./received-context";

export type ReplayFidelity =
  | "exact_replay_eligible"
  | "same_inputs_environment_changed"
  | "structurally_reproducible"
  | "unavailable";

export interface TimeMachineReason {
  code: string;
  message: string;
}

export interface TimeMachineDurablePin {
  status: "stored" | "unavailable";
  reason: TimeMachineReason;
  pin?: {
    pinId: string;
    pinnedAt: string;
    kvBytes: number;
    envelopeBytes: number;
  };
}

export interface TimeMachineTurnSource {
  status: "available" | "unavailable";
  runId?: string;
  scope?: "full_run_prompt_boundary" | "session_turn_prompt_boundary";
  sourceTurn?: number;
  durablePin: TimeMachineDurablePin;
  reasons: TimeMachineReason[];
}

/** Read-only live-store descriptor. It is only a candidate: the exact action revalidates it. */
export interface TimeMachineSnapshot {
  runId?: string;
  turn?: number;
  hasCache: boolean;
}

export interface TimeMachineTurn {
  turn: number;
  branchEligible: boolean;
  replayFidelity: ReplayFidelity;
  exactReplayEligible: boolean;
  snapshot?: TimeMachineSnapshot | null;
  source: TimeMachineTurnSource;
  reasons: TimeMachineReason[];
  lastVerification?: {
    verificationId: string | null;
    status: "verified" | "unavailable" | "failed" | null;
    fidelity: ReplayFidelity;
    exactReplay: boolean;
    message: string | null;
    scope?: "full_run_prompt_boundary" | "session_turn_prompt_boundary";
    requestedRunId?: string;
    sourceRunId?: string;
    sourceTurn?: number;
  };
}

export interface TimeMachineVerification {
  schemaVersion: string;
  verificationId: string;
  parentRunId: string;
  requestedRunId?: string;
  sourceRunId?: string;
  sourceTurn?: number;
  turn: number;
  status: "verified" | "unavailable" | "failed";
  exactReplay: boolean;
  fidelity: ReplayFidelity;
  exactnessRegime?: string;
  reasons: TimeMachineReason[];
  checkpointReferenceId?: string;
}

export type TimeMachineContinuationStatus = "completed" | "unavailable" | "failed" | "cancelled";

export interface TimeMachineContinuationRequest {
  turn: number;
  user: { content: string };
  maxTokens: number;
}

export interface TimeMachineContinuationSource {
  status: "resolved" | "unavailable" | "ambiguous";
  sourceRunId?: string;
  sourceTurn?: number;
  resolution?: "exact_latest_run" | "exact_organic_session_prefix";
  reasons: TimeMachineReason[];
}

export interface TimeMachineContinuationCheckpoint {
  status: "available" | "unavailable" | "failed";
  provenance?: "live_worker_checkpoint" | "durable_pin_import";
  restartSafe?: boolean;
  sourceRunId?: string;
  checkpointReferenceId?: string;
  pinId?: string;
  reasons: TimeMachineReason[];
}

export interface TimeMachineContinuationFailure {
  stage: "request" | "source_resolution" | "checkpoint" | "identity" | "append_derivation"
    | "worker_restore" | "worker_append" | "generation" | "persistence";
  code: string;
  message: string;
  retryable: boolean;
}

interface TimeMachineContinuationBase {
  schemaVersion: "clozn.time-machine-continuation.v1";
  continuationId: string;
  requestedRunId: string;
  sourceTurn: number;
  status: TimeMachineContinuationStatus;
  request: {
    requestId: string;
    turn: number;
    maxTokens: number;
  };
  source: TimeMachineContinuationSource;
  sourceCheckpoint: TimeMachineContinuationCheckpoint;
  reasons: TimeMachineReason[];
}

export interface TimeMachineContinuationCompleted extends TimeMachineContinuationBase {
  status: "completed";
  exactness: {
    claim: "exact_historical_state_append";
    appendOnlyExecution: true;
    historicalPrefixRecomputed: false;
    historicalPrefixRetokenizedForExecution: false;
    structuralFallbackUsed: false;
  };
  childLineage: {
    requestedParentRunId: string;
    sourceCheckpointRunId: string;
    childRunId: string;
  };
  failure: null;
}

export interface TimeMachineContinuationTerminalFailure extends TimeMachineContinuationBase {
  status: "unavailable" | "failed" | "cancelled";
  exactness: null;
  childLineage: null;
  failure: TimeMachineContinuationFailure;
}

export type TimeMachineContinuationReceipt =
  | TimeMachineContinuationCompleted
  | TimeMachineContinuationTerminalFailure;

/** A terminal receipt is the only accepted response from the append-only route. */
export class TimeMachineContinuationReceiptError extends Error {}

export interface TimeMachineDocument {
  schemaVersion: string;
  runId: string;
  state: ReplayFidelity;
  eligible: boolean;
  exactReplay: { eligible: boolean; reason: TimeMachineReason };
  reasons: TimeMachineReason[];
  turns: TimeMachineTurn[];
}

function fidelity(value: unknown): ReplayFidelity {
  const allowed: ReplayFidelity[] = [
    "exact_replay_eligible",
    "same_inputs_environment_changed",
    "structurally_reproducible",
    "unavailable",
  ];
  return typeof value === "string" && allowed.includes(value as ReplayFidelity)
    ? value as ReplayFidelity
    : "unavailable";
}

function reason(value: unknown): TimeMachineReason {
  const item = record(value);
  return {
    code: str(item.code) ?? "unknown",
    message: str(item.message) ?? "No reason was supplied.",
  };
}

function requiredString(value: unknown, field: string): string {
  const parsed = str(value);
  if (!parsed) throw new TimeMachineContinuationReceiptError(`Continuation receipt omitted ${field}.`);
  return parsed;
}

function requiredInteger(value: unknown, field: string, minimum = 0): number {
  const parsed = num(value);
  if (parsed == null || !Number.isInteger(parsed) || parsed < minimum) {
    throw new TimeMachineContinuationReceiptError(`Continuation receipt has an invalid ${field}.`);
  }
  return parsed;
}

function requiredNumber(value: unknown, field: string, minimum = 0): number {
  const parsed = num(value);
  if (parsed == null || parsed < minimum) {
    throw new TimeMachineContinuationReceiptError(`Continuation receipt has an invalid ${field}.`);
  }
  return parsed;
}

function requiredBoolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") {
    throw new TimeMachineContinuationReceiptError(`Continuation receipt omitted ${field}.`);
  }
  return value;
}

function requiredReasons(value: unknown, field: string): TimeMachineReason[] {
  const raw = records(value);
  if (raw.length === 0 || raw.some((entry) => !str(entry.code) || !str(entry.message))) {
    throw new TimeMachineContinuationReceiptError(`Continuation receipt omitted ${field}.`);
  }
  return raw.map(reason);
}

function parseContinuationSource(value: unknown): TimeMachineContinuationSource {
  const item = record(value);
  const status = item.status;
  if (status === "resolved") {
    const resolution = item.resolution;
    if (resolution !== "exact_latest_run" && resolution !== "exact_organic_session_prefix") {
      throw new TimeMachineContinuationReceiptError("Continuation receipt has an unknown source resolution.");
    }
    return {
      status,
      sourceRunId: requiredString(item.source_run_id, "source.source_run_id"),
      sourceTurn: requiredInteger(item.source_turn, "source.source_turn"),
      resolution,
      reasons: [],
    };
  }
  if (status === "unavailable" || status === "ambiguous") {
    return { status, reasons: requiredReasons(item.reasons, "source.reasons") };
  }
  throw new TimeMachineContinuationReceiptError("Continuation receipt has an unknown source status.");
}

function parseContinuationCheckpoint(value: unknown): TimeMachineContinuationCheckpoint {
  const item = record(value);
  if (item.status === "available") {
    const provenance = item.provenance;
    if (provenance !== "live_worker_checkpoint" && provenance !== "durable_pin_import") {
      throw new TimeMachineContinuationReceiptError("Continuation receipt has an unknown checkpoint provenance.");
    }
    const restartSafe = requiredBoolean(item.restart_safe, "source_checkpoint.restart_safe");
    if ((provenance === "durable_pin_import") !== restartSafe) {
      throw new TimeMachineContinuationReceiptError("Continuation receipt has incompatible checkpoint restart safety.");
    }
    return {
      status: "available",
      provenance,
      restartSafe,
      sourceRunId: requiredString(item.source_run_id, "source_checkpoint.source_run_id"),
      checkpointReferenceId: requiredString(item.checkpoint_reference_id, "source_checkpoint.checkpoint_reference_id"),
      pinId: provenance === "durable_pin_import"
        ? requiredString(item.pin_id, "source_checkpoint.pin_id") : undefined,
      reasons: [],
    };
  }
  if (item.status === "unavailable" || item.status === "failed") {
    return { status: item.status, reasons: requiredReasons(item.reasons, "source_checkpoint.reasons") };
  }
  throw new TimeMachineContinuationReceiptError("Continuation receipt has an unknown checkpoint status.");
}

function parseContinuationFailure(value: unknown): TimeMachineContinuationFailure {
  const item = record(value);
  const stage = item.stage;
  const stages: TimeMachineContinuationFailure["stage"][] = [
    "request", "source_resolution", "checkpoint", "identity", "append_derivation",
    "worker_restore", "worker_append", "generation", "persistence",
  ];
  if (typeof stage !== "string" || !stages.includes(stage as TimeMachineContinuationFailure["stage"])) {
    throw new TimeMachineContinuationReceiptError("Continuation receipt has an unknown failure stage.");
  }
  return {
    stage: stage as TimeMachineContinuationFailure["stage"],
    code: requiredString(item.code, "failure.code"),
    message: requiredString(item.message, "failure.message"),
    retryable: requiredBoolean(item.retryable, "failure.retryable"),
  };
}

function sectionStatus(value: unknown, field: string, allowed: readonly string[]): { status: string; item: Record<string, unknown> } {
  const item = record(value);
  const status = str(item.status);
  if (!status || !allowed.includes(status)) {
    throw new TimeMachineContinuationReceiptError(`Continuation receipt has an invalid ${field}.status.`);
  }
  return { status, item };
}

function requireReasonsForUnavailableSection(
  item: Record<string, unknown>,
  status: string,
  availableStatus: string,
  field: string,
) {
  if (status !== availableStatus) requiredReasons(item.reasons, `${field}.reasons`);
}

function parseContinuationReceipt(value: unknown, runId: string): TimeMachineContinuationReceipt {
  const item = record(value);
  if (item.schema_version !== "clozn.time-machine-continuation.v1") {
    throw new TimeMachineContinuationReceiptError("Gateway returned an unrecognized exact continuation receipt.");
  }
  const status = item.status;
  if (status !== "completed" && status !== "unavailable" && status !== "failed" && status !== "cancelled") {
    throw new TimeMachineContinuationReceiptError("Gateway returned a non-terminal exact continuation receipt.");
  }
  const request = record(item.request);
  if (request.append_kind !== "new_user_turn") {
    throw new TimeMachineContinuationReceiptError("Continuation receipt has an unsupported append kind.");
  }
  requiredNumber(item.created_ts, "created_ts");
  requiredNumber(item.finished_ts, "finished_ts");
  requiredString(request.user_content_sha256, "request.user_content_sha256");
  requiredInteger(request.user_content_bytes, "request.user_content_bytes", 1);
  requiredString(request.generation_config_sha256, "request.generation_config_sha256");
  if (!Array.isArray(item.unavoidable_differences)) {
    throw new TimeMachineContinuationReceiptError("Continuation receipt omitted unavoidable_differences.");
  }
  const base: TimeMachineContinuationBase = {
    schemaVersion: "clozn.time-machine-continuation.v1",
    continuationId: requiredString(item.continuation_id, "continuation_id"),
    requestedRunId: requiredString(item.requested_run_id, "requested_run_id"),
    sourceTurn: requiredInteger(item.source_turn, "source_turn"),
    status,
    request: {
      requestId: requiredString(request.request_id, "request.request_id"),
      turn: requiredInteger(request.turn, "request.turn"),
      maxTokens: requiredInteger(request.max_tokens, "request.max_tokens", 1),
    },
    source: parseContinuationSource(item.source),
    sourceCheckpoint: parseContinuationCheckpoint(item.source_checkpoint),
    reasons: requiredReasons(item.reasons, "reasons"),
  };
  if (base.requestedRunId !== runId || base.request.turn !== base.sourceTurn) {
    throw new TimeMachineContinuationReceiptError("Continuation receipt does not match the requested run or turn.");
  }
  const identity = sectionStatus(item.identity, "identity", ["matched", "unavailable", "mismatch"]);
  requireReasonsForUnavailableSection(identity.item, identity.status, "matched", "identity");
  const append = sectionStatus(item.append, "append", ["validated", "unavailable", "failed"]);
  requireReasonsForUnavailableSection(append.item, append.status, "validated", "append");
  const sampler = sectionStatus(item.sampler, "sampler", ["preserved", "unavailable", "mismatch"]);
  requireReasonsForUnavailableSection(sampler.item, sampler.status, "preserved", "sampler");
  const exactnessStatus = sectionStatus(item.exactness, "exactness", ["confirmed", "not_confirmed", "failed"]);
  requireReasonsForUnavailableSection(exactnessStatus.item, exactnessStatus.status, "confirmed", "exactness");
  const worker = sectionStatus(item.worker, "worker", ["completed", "not_run", "failed", "cancelled"]);
  requireReasonsForUnavailableSection(worker.item, worker.status, "completed", "worker");
  const childLineageStatus = sectionStatus(item.child_lineage, "child_lineage", ["created", "not_created", "failed", "cancelled"]);
  requireReasonsForUnavailableSection(childLineageStatus.item, childLineageStatus.status, "created", "child_lineage");
  if (status !== "completed") {
    if (item.failure === null || item.failure === undefined) {
      throw new TimeMachineContinuationReceiptError("Terminal continuation failure omitted its typed failure.");
    }
    if (childLineageStatus.status === "created") {
      throw new TimeMachineContinuationReceiptError("Terminal continuation failure cannot claim an immutable child.");
    }
    return {
      ...base,
      status,
      exactness: null,
      childLineage: null,
      failure: parseContinuationFailure(item.failure),
    };
  }
  const exactness = record(item.exactness);
  if (
    exactnessStatus.status !== "confirmed"
    || identity.status !== "matched"
    || append.status !== "validated"
    || sampler.status !== "preserved"
    || worker.status !== "completed"
    || childLineageStatus.status !== "created"
    || exactness.claim !== "exact_historical_state_append"
    || exactness.append_only_execution !== true
    || exactness.historical_prefix_recomputed !== false
    || exactness.historical_prefix_retokenized_for_execution !== false
    || exactness.structural_fallback_used !== false
  ) {
    throw new TimeMachineContinuationReceiptError("Completed continuation receipt does not prove append-only exactness.");
  }
  const lineage = record(item.child_lineage);
  if (lineage.status !== "created" || lineage.relation !== "exact_continuation" || lineage.parent_immutable !== true
    || lineage.source_immutable !== true || lineage.receipt_persisted !== true || item.failure !== null) {
    throw new TimeMachineContinuationReceiptError("Completed continuation receipt has invalid immutable child lineage.");
  }
  const childLineage = {
    requestedParentRunId: requiredString(lineage.requested_parent_run_id, "child_lineage.requested_parent_run_id"),
    sourceCheckpointRunId: requiredString(lineage.source_checkpoint_run_id, "child_lineage.source_checkpoint_run_id"),
    childRunId: requiredString(lineage.child_run_id, "child_lineage.child_run_id"),
  };
  if (base.source.status !== "resolved" || base.sourceCheckpoint.status !== "available"
    || base.source.sourceRunId !== base.sourceCheckpoint.sourceRunId
    || childLineage.requestedParentRunId !== base.requestedRunId
    || childLineage.sourceCheckpointRunId !== base.source.sourceRunId) {
    throw new TimeMachineContinuationReceiptError("Completed continuation receipt has inconsistent source provenance.");
  }
  return {
    ...base,
    status: "completed",
    exactness: {
      claim: "exact_historical_state_append",
      appendOnlyExecution: true,
      historicalPrefixRecomputed: false,
      historicalPrefixRetokenizedForExecution: false,
      structuralFallbackUsed: false,
    },
    childLineage,
    failure: null,
  };
}

function parseSource(value: unknown): TimeMachineTurnSource {
  const item = record(value);
  const durablePin = record(item.durable_pin);
  const pin = record(durablePin.pin);
  const stored = durablePin.status === "stored";
  const pinId = str(pin.pin_id);
  const pinnedAt = str(pin.pinned_at);
  const kvBytes = num(pin.kv_bytes);
  const envelopeBytes = num(pin.envelope_bytes);
  const hasStoredPin = stored && Boolean(pinId && pinnedAt && kvBytes != null && envelopeBytes != null);
  return {
    status: item.status === "available" ? "available" : "unavailable",
    runId: str(item.run_id) ?? undefined,
    scope: item.scope === "full_run_prompt_boundary" || item.scope === "session_turn_prompt_boundary"
      ? item.scope : undefined,
    sourceTurn: num(item.source_turn) ?? undefined,
    durablePin: {
      status: hasStoredPin ? "stored" : "unavailable",
      reason: reason(durablePin.reason),
      pin: hasStoredPin && pinId && pinnedAt && kvBytes != null && envelopeBytes != null
        ? { pinId, pinnedAt, kvBytes, envelopeBytes }
        : undefined,
    },
    reasons: records(item.reasons).map(reason),
  };
}

function parseTurn(value: unknown): TimeMachineTurn {
  const item = record(value);
  const snapshot = record(item.snapshot);
  const previous = record(item.last_verification);
  const lastVerification: TimeMachineTurn["lastVerification"] = item.last_verification && typeof item.last_verification === "object"
    ? {
      verificationId: str(previous.verification_id) ?? null,
      status: previous.status === "verified" || previous.status === "unavailable" || previous.status === "failed"
        ? previous.status : null,
      fidelity: fidelity(previous.fidelity),
      exactReplay: previous.exact_replay === true,
      message: str(previous.message) ?? null,
      scope: previous.scope === "full_run_prompt_boundary" || previous.scope === "session_turn_prompt_boundary"
        ? previous.scope : undefined,
      requestedRunId: str(previous.requested_run_id) ?? undefined,
      sourceRunId: str(previous.source_run_id) ?? undefined,
      sourceTurn: num(previous.source_turn) ?? undefined,
    }
    : undefined;
  return {
    turn: num(item.turn) ?? 0,
    branchEligible: item.branch_eligible === true,
    replayFidelity: fidelity(item.replay_fidelity),
    exactReplayEligible: item.exact_replay_eligible === true,
    snapshot: item.snapshot === null ? null : item.snapshot && typeof item.snapshot === "object" && !Array.isArray(item.snapshot)
      ? {
        runId: str(snapshot.run_id) ?? undefined,
        turn: num(snapshot.turn) ?? undefined,
        hasCache: snapshot.has_cache === true,
      }
      : undefined,
    source: parseSource(item.source),
    reasons: records(item.reasons).map(reason),
    lastVerification,
  };
}

function parseDocument(value: unknown, runId: string): TimeMachineDocument {
  const item = record(value);
  const exact = record(item.exact_replay);
  return {
    schemaVersion: str(item.schema_version) ?? "",
    runId: str(item.run_id) ?? runId,
    state: fidelity(item.state),
    eligible: item.eligible === true,
    exactReplay: {
      eligible: exact.eligible === true,
      reason: reason(exact.reason),
    },
    reasons: records(item.reasons).map(reason),
    turns: records(item.turns).map(parseTurn),
  };
}

export async function loadTimeMachine(runId: string, signal?: AbortSignal): Promise<TimeMachineDocument> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/time-machine`, { signal });
  if (!response.ok) throw new Error(`Time Machine request failed (${response.status})`);
  return parseDocument(await response.json(), runId);
}

export async function branchTimeMachine(
  runId: string,
  turn: number,
  altUser?: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  const body: Record<string, unknown> = { turn };
  if (altUser?.trim()) body.alt_user = altUser.trim();
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/branch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const item = record(payload);
    throw new Error(str(item.error) ?? `Branch request failed (${response.status})`);
  }
  return record(payload);
}

export async function exactBranchTimeMachine(
  runId: string,
  turn: number,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  const response = await fetch("/runs/" + encodeURIComponent(runId) + "/time-machine/branch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ turn }),
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  const item = record(payload);
  if (!response.ok) {
    const reasons = records(item.reasons);
    throw new Error(
      str(reasons[0] && record(reasons[0]).message)
      ?? str(item.error)
      ?? ("Exact child replay failed (" + response.status + ")"),
    );
  }
  return item;
}

/**
 * Restore an exact historical checkpoint, append one new user turn, and generate a new immutable
 * child. This deliberately sends the v1's closed request body only: settings remain source-bound.
 * A valid unavailable/failed/cancelled receipt is a normal terminal result, even on a non-2xx HTTP
 * response; malformed or non-terminal payloads fail closed instead of being displayed as evidence.
 */
export async function continueTimeMachine(
  runId: string,
  request: TimeMachineContinuationRequest,
  signal?: AbortSignal,
): Promise<TimeMachineContinuationReceipt> {
  if (!Number.isInteger(request.turn) || request.turn < 0) {
    throw new TimeMachineContinuationReceiptError("A continuation turn must be a non-negative integer.");
  }
  if (!request.user.content.trim()) {
    throw new TimeMachineContinuationReceiptError("Enter a new question before continuing this turn.");
  }
  if (!Number.isInteger(request.maxTokens) || request.maxTokens < 1) {
    throw new TimeMachineContinuationReceiptError("The continuation token limit must be a positive integer.");
  }
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/time-machine/continue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      turn: request.turn,
      user: { content: request.user.content },
      max_tokens: request.maxTokens,
    }),
    signal,
  });
  const payload = await response.json().catch(() => undefined);
  try {
    return parseContinuationReceipt(payload, runId);
  } catch (error) {
    if (error instanceof TimeMachineContinuationReceiptError) throw error;
    throw new TimeMachineContinuationReceiptError(
      `Gateway returned an invalid exact continuation receipt (${response.status}).`,
    );
  }
}

export async function verifyTimeMachine(
  runId: string,
  turn: number,
  signal?: AbortSignal,
): Promise<TimeMachineVerification> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/time-machine/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ turn }),
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  const item = record(payload);
  if (!response.ok && item.schema_version !== "clozn.time-machine-verification.v1") {
    throw new Error(str(item.error) ?? `Time Machine verification failed (${response.status})`);
  }
  return {
    schemaVersion: str(item.schema_version) ?? "",
    verificationId: str(item.verification_id) ?? "",
    parentRunId: str(item.parent_run_id) ?? runId,
    requestedRunId: str(item.requested_run_id) ?? undefined,
    sourceRunId: str(item.source_run_id) ?? undefined,
    sourceTurn: num(item.source_turn) ?? undefined,
    turn: num(item.turn) ?? turn,
    status: item.status === "verified" || item.status === "unavailable" || item.status === "failed"
      ? item.status : "failed",
    exactReplay: item.exact_replay === true,
    fidelity: fidelity(item.fidelity),
    exactnessRegime: str(item.exactness_regime) ?? undefined,
    reasons: records(item.reasons).map(reason),
    checkpointReferenceId: str(item.checkpoint_reference_id) ?? undefined,
  };
}
