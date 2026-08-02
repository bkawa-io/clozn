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

export interface TimeMachineTurn {
  turn: number;
  branchEligible: boolean;
  replayFidelity: ReplayFidelity;
  exactReplayEligible: boolean;
  snapshot?: Record<string, unknown> | null;
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

function parseTurn(value: unknown): TimeMachineTurn {
  const item = record(value);
  const snapshot = item.snapshot;
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
    snapshot: snapshot && typeof snapshot === "object" && !Array.isArray(snapshot)
      ? snapshot as Record<string, unknown>
      : snapshot === null ? null : undefined,
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
