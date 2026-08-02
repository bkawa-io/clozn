import { num, record, records, str, type JsonRecord } from "./received-context";

/** Client for E4's model second-opinion routes.
 *
 * Candidates are a read-only capability probe. Running an opinion is always an explicit POST from the
 * panel's button; selecting the investigation question or mounting the panel never starts generation.
 * The wire document deliberately keeps the anchor and second-opinion arms independently typed: a
 * generation failure for arm B must not hide the original run's evidence in arm A.
 */

export interface SecondOpinionCandidate {
  modelId: string;
  ready: boolean;
}

export interface SecondOpinionCandidates {
  managed: boolean;
  ownModelId?: string;
  candidates: SecondOpinionCandidate[];
}

export interface WorkerIdentity {
  modelSha256?: string;
  templateFingerprint?: string;
  engineBuild?: string;
  backend?: string;
  contextSize?: number;
  workerId?: string;
  workerGeneration?: number;
}

export type AnchorStatus = "ok" | "empty" | "redacted" | "unavailable";
export type SecondOpinionArmStatus = "ok" | "refused" | "generation_error";

export interface AnchorArm {
  role: "anchor";
  runId: string;
  modelId?: string;
  workerIdentity?: WorkerIdentity;
  status: AnchorStatus;
  responseText?: string;
  finishReason?: string;
  latencyMs?: number;
  promptTokens?: number;
  generatedTokens?: number;
}

export interface ArmRefusal {
  code: string;
  message: string;
}

export interface SecondOpinionArm {
  role: "second_opinion";
  requestedModelId: string;
  modelId: string;
  workerIdentity?: WorkerIdentity;
  status: SecondOpinionArmStatus;
  refusal?: ArmRefusal;
  responseText?: string;
  finishReason?: string;
  latencyMs?: number;
  generatedTokens?: number;
}

export type TemplateCompatibilityState = "same" | "differs" | "unknown";
export type ContextCompatibilityState = "within_estimate" | "exceeds_estimate" | "unknown";

export interface Compatibility {
  chatTemplate: {
    state: TemplateCompatibilityState;
    method: string;
    armATemplateFingerprint?: string;
    armBTemplateFingerprint?: string;
    caveat?: string;
  };
  contextLimit: {
    state: ContextCompatibilityState;
    method: string;
    armAPromptTokensEstimate?: number;
    armBContextWindowTokens?: number;
    caveat?: string;
  };
  toolsOrSchema: {
    state: "none_used" | "used_not_replayed";
    requestedMode?: string;
    caveat?: string;
  };
  qualifiedEvidence: {
    state: "anchor_only";
    note: string;
  };
}

export interface SecondOpinionComparison {
  agreement: {
    method: string;
    lexicalDifferencePercent: number;
    caveat: string;
  };
  formatChanged?: boolean;
  length: {
    armAWords: number;
    armBWords: number;
  };
}

export interface SecondOpinionDocument {
  schemaVersion: string;
  generatedAt: string;
  runId: string;
  deliveredInput: {
    messageCount: number;
    sha256: string;
    identicalAcrossArms: boolean;
  };
  armA: AnchorArm;
  armB: SecondOpinionArm;
  compatibility: Compatibility;
  comparison?: SecondOpinionComparison;
}

export class SecondOpinionLoadError extends Error {}

function bool(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function workerIdentity(raw: unknown): WorkerIdentity | undefined {
  const item = record(raw);
  const value: WorkerIdentity = {
    modelSha256: str(item.model_sha256),
    templateFingerprint: str(item.template_fingerprint),
    engineBuild: str(item.engine_build),
    backend: str(item.backend),
    contextSize: num(item.context_size),
    workerId: str(item.worker_id),
    workerGeneration: num(item.worker_generation),
  };
  return Object.values(value).some((entry) => entry !== undefined) ? value : undefined;
}

const ANCHOR_STATUSES = new Set<AnchorStatus>(["ok", "empty", "redacted", "unavailable"]);
const ARM_STATUSES = new Set<SecondOpinionArmStatus>(["ok", "refused", "generation_error"]);
const TEMPLATE_STATES = new Set<TemplateCompatibilityState>(["same", "differs", "unknown"]);
const CONTEXT_STATES = new Set<ContextCompatibilityState>(["within_estimate", "exceeds_estimate", "unknown"]);

function anchorStatus(value: unknown): AnchorStatus {
  return typeof value === "string" && ANCHOR_STATUSES.has(value as AnchorStatus)
    ? value as AnchorStatus
    : "unavailable";
}

function armStatus(value: unknown): SecondOpinionArmStatus {
  return typeof value === "string" && ARM_STATUSES.has(value as SecondOpinionArmStatus)
    ? value as SecondOpinionArmStatus
    : "generation_error";
}

function templateState(value: unknown): TemplateCompatibilityState {
  return typeof value === "string" && TEMPLATE_STATES.has(value as TemplateCompatibilityState)
    ? value as TemplateCompatibilityState
    : "unknown";
}

function contextState(value: unknown): ContextCompatibilityState {
  return typeof value === "string" && CONTEXT_STATES.has(value as ContextCompatibilityState)
    ? value as ContextCompatibilityState
    : "unknown";
}

function parseRefusal(raw: unknown): ArmRefusal | undefined {
  const item = record(raw);
  const code = str(item.code);
  const message = str(item.message);
  return code && message ? { code, message } : undefined;
}

function parseAnchor(raw: unknown, requestedRunId: string): AnchorArm {
  const item = record(raw);
  return {
    role: "anchor",
    runId: str(item.run_id) ?? requestedRunId,
    modelId: str(item.model_id),
    workerIdentity: workerIdentity(item.worker_identity),
    status: anchorStatus(item.status),
    responseText: str(item.response_text),
    finishReason: str(item.finish_reason),
    latencyMs: num(item.latency_ms),
    promptTokens: num(item.prompt_tokens),
    generatedTokens: num(item.generated_tokens),
  };
}

function parseArm(raw: unknown): SecondOpinionArm {
  const item = record(raw);
  const status = armStatus(item.status);
  return {
    role: "second_opinion",
    requestedModelId: str(item.requested_model_id) ?? "unknown",
    modelId: str(item.model_id) ?? "unknown",
    workerIdentity: workerIdentity(item.worker_identity),
    status,
    refusal: parseRefusal(item.refusal) ?? (status === "generation_error"
      ? { code: "invalid_arm_document", message: "the gateway returned an untyped second-opinion failure" }
      : undefined),
    responseText: str(item.response_text),
    finishReason: str(item.finish_reason),
    latencyMs: num(item.latency_ms),
    generatedTokens: num(item.generated_tokens),
  };
}

function parseCompatibility(raw: unknown): Compatibility {
  const item = record(raw);
  const chat = record(item.chat_template);
  const context = record(item.context_limit);
  const tools = record(item.tools_or_schema);
  const evidence = record(item.qualified_evidence);
  return {
    chatTemplate: {
      state: templateState(chat.state),
      method: str(chat.method) ?? "template_fingerprint_compare",
      armATemplateFingerprint: str(chat.arm_a_template_fingerprint),
      armBTemplateFingerprint: str(chat.arm_b_template_fingerprint),
      caveat: str(chat.caveat),
    },
    contextLimit: {
      state: contextState(context.state),
      method: str(context.method) ?? "arm_a_recorded_prompt_tokens_vs_arm_b_context_window",
      armAPromptTokensEstimate: num(context.arm_a_prompt_tokens_estimate),
      armBContextWindowTokens: num(context.arm_b_context_window_tokens),
      caveat: str(context.caveat),
    },
    toolsOrSchema: {
      state: tools.state === "used_not_replayed" ? "used_not_replayed" : "none_used",
      requestedMode: str(tools.requested_mode),
      caveat: str(tools.caveat),
    },
    qualifiedEvidence: {
      state: "anchor_only",
      note: str(evidence.note) ?? "claim-level support is available for the anchor only in this version",
    },
  };
}

function parseDocument(raw: unknown, requestedRunId: string): SecondOpinionDocument {
  const item = record(raw);
  const delivered = record(item.delivered_input);
  const comparison = record(item.comparison);
  const agreement = record(comparison.agreement);
  const length = record(comparison.length);
  const hasComparison = Object.keys(comparison).length > 0;
  return {
    schemaVersion: str(item.schema_version) ?? "",
    generatedAt: str(item.generated_at) ?? "",
    runId: str(item.run_id) ?? requestedRunId,
    deliveredInput: {
      messageCount: num(delivered.message_count) ?? 0,
      sha256: str(delivered.sha256) ?? "",
      identicalAcrossArms: bool(delivered.identical_across_arms) ?? false,
    },
    armA: parseAnchor(item.arm_a, requestedRunId),
    armB: parseArm(item.arm_b),
    compatibility: parseCompatibility(item.compatibility),
    comparison: hasComparison ? {
      agreement: {
        method: str(agreement.method) ?? "lexical_overlap_heuristic",
        lexicalDifferencePercent: num(agreement.lexical_difference_percent) ?? 0,
        caveat: str(agreement.caveat) ?? "this is a lexical proxy, not semantic agreement",
      },
      formatChanged: bool(comparison.format_changed),
      length: {
        armAWords: num(length.arm_a_words) ?? 0,
        armBWords: num(length.arm_b_words) ?? 0,
      },
    } : undefined,
  };
}

async function getJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof record(body).error === "string"
      ? record(body).error as string
      : `Request failed (${response.status})`;
    throw new SecondOpinionLoadError(message);
  }
  return body;
}

export async function loadSecondOpinionCandidates(
  runId: string,
  signal?: AbortSignal,
): Promise<SecondOpinionCandidates> {
  const item = record(await getJson(`/runs/${encodeURIComponent(runId)}/second-opinion/candidates`, signal));
  return {
    managed: bool(item.managed) ?? false,
    ownModelId: str(item.own_model_id),
    candidates: records(item.candidates)
      .map((candidate: JsonRecord) => ({
        modelId: str(candidate.model_id) ?? "",
        ready: bool(candidate.ready) ?? false,
      }))
      .filter((candidate) => candidate.modelId.length > 0),
  };
}

export async function runSecondOpinion(
  runId: string,
  modelId: string,
  signal?: AbortSignal,
): Promise<SecondOpinionDocument> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/second-opinion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: modelId }),
    signal,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof record(body).error === "string"
      ? record(body).error as string
      : `Request failed (${response.status})`;
    throw new SecondOpinionLoadError(message);
  }
  return parseDocument(body, runId);
}

export function describeSecondOpinionError(error: unknown): string {
  return error instanceof Error ? error.message : "the second-opinion request failed";
}
