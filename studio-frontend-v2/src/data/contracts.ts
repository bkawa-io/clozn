/**
 * Narrow, fail-closed decoders for the Studio v2 read boundary.
 *
 * These are deliberately dependency-free: a malformed server document is a contract failure, not an
 * empty result. In particular, absence is represented by an optional property rather than a guessed 0.
 */

export class ContractError extends Error {
  readonly name = "ContractError";

  constructor(readonly endpoint: string, readonly detail: string) {
    super(`Invalid response from ${endpoint}: ${detail}`);
  }
}

type JsonRecord = Record<string, unknown>;
export type JsonObject = Readonly<Record<string, unknown>>;

/** Validate an open, versioned action document while preserving its backend-owned nested contract. */
export function decodeJsonObject(value: unknown, endpoint: string): JsonObject {
  return object(value, endpoint, "$");
}

const spanIdPattern = /^span_[0-9a-f]{24}$/;
const relationIdPattern = /^rel_[0-9a-f]{24}$/;
const tensionIdPattern = /^tension_[0-9a-f]{24}$/;
const breakpointIdPattern = /^breakpoint_[0-9a-f]{24}$/;
const hashPattern = /^[0-9a-f]{64}$/;

function fail(endpoint: string, detail: string): never {
  throw new ContractError(endpoint, detail);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(endpoint, `${path} must be an object`);
  }
  return value as JsonRecord;
}

function array(value: unknown, endpoint: string, path: string): unknown[] {
  if (!Array.isArray(value)) fail(endpoint, `${path} must be an array`);
  return value;
}

function string(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, `${path} must be a string`);
  return value;
}

function nonEmptyString(value: unknown, endpoint: string, path: string): string {
  const parsed = string(value, endpoint, path);
  if (!parsed) fail(endpoint, `${path} must not be empty`);
  return parsed;
}

function boolean(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, `${path} must be a boolean`);
  return value;
}

function number(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, `${path} must be a finite number`);
  return value;
}

function integer(value: unknown, endpoint: string, path: string): number {
  const parsed = number(value, endpoint, path);
  if (!Number.isInteger(parsed)) fail(endpoint, `${path} must be an integer`);
  return parsed;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = integer(value, endpoint, path);
  if (parsed < 0) fail(endpoint, `${path} must be non-negative`);
  return parsed;
}

function enumValue<T extends string>(
  value: unknown,
  values: readonly T[],
  endpoint: string,
  path: string,
): T {
  if (typeof value !== "string" || !values.includes(value as T)) {
    fail(endpoint, `${path} must be one of ${values.join(", ")}`);
  }
  return value as T;
}

function optional<T>(record: JsonRecord, key: string, parse: (value: unknown, path: string) => T, endpoint: string, path: string): T | undefined {
  return record[key] === undefined ? undefined : parse(record[key], `${path}.${key}`);
}

function closed(record: JsonRecord, keys: readonly string[], endpoint: string, path: string): void {
  for (const key of Object.keys(record)) {
    if (!keys.includes(key)) fail(endpoint, `${path}.${key} is not allowed`);
  }
}

function required(record: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in record)) fail(endpoint, `${path}.${key} is required`);
  return record[key];
}

function jsonObject(value: unknown, endpoint: string, path: string): JsonObject {
  return object(value, endpoint, path);
}

function hash(value: unknown, endpoint: string, path: string): string {
  const parsed = string(value, endpoint, path);
  if (!hashPattern.test(parsed)) fail(endpoint, `${path} must be a SHA-256 digest`);
  return parsed;
}

function spanId(value: unknown, endpoint: string, path: string): string {
  const parsed = string(value, endpoint, path);
  if (!spanIdPattern.test(parsed)) fail(endpoint, `${path} must be a span address id`);
  return parsed;
}

export type MeasurementState = "available" | "not_measured" | "unavailable" | "error";
export type EvidenceState = "causally_supported" | "observed";
export type InfluenceEffect = "supports" | "suppresses" | "neutral";

export interface MeasurementAvailability {
  state: MeasurementState;
  /** Defined for every non-available state; never synthesized for available evidence. */
  reason?: string;
  influenceSchema?: "clozn.context_answer_influence.v1";
  artifactSha256?: string;
  method?: JsonObject;
  thresholds?: JsonObject;
}

function decodeMeasurement(value: unknown, endpoint: string, path: string): MeasurementAvailability {
  const item = object(value, endpoint, path);
  closed(item, ["state", "reason", "influence_schema", "artifact_sha256", "method", "thresholds"], endpoint, path);
  const state = enumValue(required(item, "state", endpoint, path), ["available", "not_measured", "unavailable", "error"] as const, endpoint, `${path}.state`);
  const reason = optional(item, "reason", (v, p) => nonEmptyString(v, endpoint, p), endpoint, path);
  if (state === "available") {
    if (reason !== undefined) fail(endpoint, `${path}.reason is not allowed when state is available`);
    return {
      state,
      influenceSchema: optional(item, "influence_schema", (v, p) => {
        if (v !== "clozn.context_answer_influence.v1") fail(endpoint, `${p} has an unsupported value`);
        return v;
      }, endpoint, path),
      artifactSha256: optional(item, "artifact_sha256", (v, p) => hash(v, endpoint, p), endpoint, path),
      method: optional(item, "method", (v, p) => jsonObject(v, endpoint, p), endpoint, path),
      thresholds: optional(item, "thresholds", (v, p) => jsonObject(v, endpoint, p), endpoint, path),
    };
  }
  if (!reason) fail(endpoint, `${path}.reason is required when measurement is ${state}`);
  for (const key of ["influence_schema", "artifact_sha256", "method", "thresholds"]) {
    if (item[key] !== undefined) fail(endpoint, `${path}.${key} is not allowed when measurement is ${state}`);
  }
  return { state, reason };
}

export interface RunRecord {
  id: string;
  createdAt?: string | null;
  createdTs?: number | null;
  source?: string | null;
  client?: string | null;
  sessionKey?: string | null;
  model?: string | null;
  promptSummary?: string | null;
  responseSummary?: string | null;
  /** Full detail only. An absent response stays absent. */
  response?: string | null;
  parentRunId?: string | null;
  finishReason?: string | null;
  tokenCount?: number;
  confidence?: readonly number[];
  confidenceMin?: number;
  confidenceMean?: number;
  lowConfidenceCount?: number;
  flags?: readonly string[];
  warningCount?: number;
  /** Ordered, readable request messages when the immutable run retained them. */
  messages?: readonly RunMessage[];
  /** Ordered context as actually assembled, when the run retained it separately. */
  assembledMessages?: readonly RunMessage[] | null;
  /** Recorded response token pieces when the full run retained a trace. */
  responseTokens?: readonly string[];
  responseTokenIds?: readonly number[];
}

export interface RunMessage {
  role: string;
  content: string;
  sourceId?: string;
  sourceLabel?: string;
}

function optionalStringOrNull(item: JsonRecord, key: string, endpoint: string, path: string): string | null | undefined {
  return optional(item, key, (v, p) => v === null ? null : string(v, endpoint, p), endpoint, path);
}

function optionalFinite(item: JsonRecord, key: string, endpoint: string, path: string): number | undefined {
  return optional(item, key, (v, p) => number(v, endpoint, p), endpoint, path);
}

function decodeRun(value: unknown, endpoint: string, path: string): RunRecord {
  const item = object(value, endpoint, path);
  const confidence = optional(item, "confidence", (v, p) => array(v, endpoint, p).map((entry, index) => number(entry, endpoint, `${p}[${index}]`)), endpoint, path);
  const messages = optional(item, "messages", (v, p) => decodeMessages(v, endpoint, p), endpoint, path);
  const assembledMessages = optional(item, "assembled_messages", (v, p) => v === null ? null : decodeMessages(v, endpoint, p), endpoint, path);
  const trace = item.trace === undefined ? undefined : object(item.trace, endpoint, `${path}.trace`);
  const responseTokens = trace?.tokens === undefined ? undefined : array(trace.tokens, endpoint, `${path}.trace.tokens`).map((entry, index) => string(entry, endpoint, `${path}.trace.tokens[${index}]`));
  const responseTokenIds = trace?.token_ids === undefined ? undefined : array(trace.token_ids, endpoint, `${path}.trace.token_ids`).map((entry, index) => nonNegativeInteger(entry, endpoint, `${path}.trace.token_ids[${index}]`));
  return {
    id: nonEmptyString(required(item, "id", endpoint, path), endpoint, `${path}.id`),
    createdAt: optionalStringOrNull(item, "created_at", endpoint, path),
    createdTs: optional(item, "created_ts", (v, p) => v === null ? null : number(v, endpoint, p), endpoint, path),
    source: optionalStringOrNull(item, "source", endpoint, path),
    client: optionalStringOrNull(item, "client", endpoint, path),
    sessionKey: optionalStringOrNull(item, "session_key", endpoint, path),
    model: optionalStringOrNull(item, "model", endpoint, path),
    promptSummary: optionalStringOrNull(item, "prompt_summary", endpoint, path),
    responseSummary: optionalStringOrNull(item, "response_summary", endpoint, path),
    response: optionalStringOrNull(item, "response", endpoint, path),
    parentRunId: optionalStringOrNull(item, "parent_run_id", endpoint, path),
    finishReason: optionalStringOrNull(item, "finish_reason", endpoint, path),
    tokenCount: optional(item, "token_count", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    confidence,
    confidenceMin: optionalFinite(item, "confidence_min", endpoint, path),
    confidenceMean: optionalFinite(item, "confidence_mean", endpoint, path),
    lowConfidenceCount: optional(item, "low_confidence_count", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    flags: optional(item, "flags", (v, p) => array(v, endpoint, p).map((entry, index) => string(entry, endpoint, `${p}[${index}]`)), endpoint, path),
    warningCount: optional(item, "warnings", (v, p) => array(v, endpoint, p).length, endpoint, path),
    messages,
    assembledMessages,
    responseTokens,
    responseTokenIds,
  };
}

function decodeMessages(value: unknown, endpoint: string, path: string): readonly RunMessage[] {
  return array(value, endpoint, path).map((entry, index) => {
    const item = object(entry, endpoint, `${path}[${index}]`);
    // Source fields are intentionally optional: plain chat messages have neither, and missing values
    // remain missing rather than being relabelled as a source.
    return {
      role: nonEmptyString(required(item, "role", endpoint, `${path}[${index}]`), endpoint, `${path}[${index}].role`),
      content: string(required(item, "content", endpoint, `${path}[${index}]`), endpoint, `${path}[${index}].content`),
      sourceId: optional(item, "source_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `${path}[${index}]`) ?? optional(item, "client_source_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `${path}[${index}]`),
      sourceLabel: optional(item, "source_label", (v, p) => string(v, endpoint, p), endpoint, `${path}[${index}]`),
    };
  });
}

export function decodeRunsDocument(value: unknown, endpoint = "/runs"): readonly RunRecord[] {
  const body = object(value, endpoint, "$" );
  closed(body, ["runs"], endpoint, "$");
  return array(required(body, "runs", endpoint, "$"), endpoint, "$.runs").map((entry, index) => decodeRun(entry, endpoint, `$.runs[${index}]`));
}

export function decodeRunDocument(value: unknown, endpoint: string): RunRecord {
  return decodeRun(value, endpoint, "$");
}

export interface HealthStatus {
  status: "ok" | "not_ready";
  service: "clozn";
  reason?: string;
  /** Gateway-to-worker wire version reported by /readyz; absence is not compatibility. */
  protocolVersion?: string;
  /** The single active worker's model identity when the legacy runtime reports one. */
  model?: string;
  mode?: string;
  worker?: RuntimeWorkerStatus;
  /** Capability documents can contain both booleans and structured values. */
  capabilities?: JsonObject;
  queue?: RuntimeQueueStatus;
}

/** The directly observed subset of the legacy worker health response used by Studio. */
export interface RuntimeWorkerStatus {
  model?: string;
  modelSha256?: string;
  backend?: string;
  device?: string;
  contextSize?: number;
  engineVersion?: string;
  buildId?: string;
  protocolVersion?: string;
  workerGenerationId?: string;
  capabilities?: JsonObject;
}

/** A point-in-time gateway queue snapshot. No absent field is turned into zero. */
export interface RuntimeQueueStatus {
  active?: number;
  waiting?: number;
  capacity?: number;
}

function decodeRuntimeWorker(value: unknown, endpoint: string, path: string): RuntimeWorkerStatus {
  const item = object(value, endpoint, path);
  return {
    model: optional(item, "model", (v, p) => string(v, endpoint, p), endpoint, path),
    modelSha256: optional(item, "model_sha256", (v, p) => hash(v, endpoint, p), endpoint, path),
    backend: optional(item, "backend", (v, p) => string(v, endpoint, p), endpoint, path),
    device: optional(item, "device", (v, p) => string(v, endpoint, p), endpoint, path),
    contextSize: optional(item, "n_ctx", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    engineVersion: optional(item, "engine_version", (v, p) => string(v, endpoint, p), endpoint, path),
    buildId: optional(item, "build_id", (v, p) => string(v, endpoint, p), endpoint, path),
    protocolVersion: optional(item, "protocol_version", (v, p) => string(v, endpoint, p), endpoint, path),
    workerGenerationId: optional(item, "worker_generation_id", (v, p) => string(v, endpoint, p), endpoint, path),
    capabilities: optional(item, "capabilities", (v, p) => jsonObject(v, endpoint, p), endpoint, path),
  };
}

function decodeRuntimeQueue(value: unknown, endpoint: string, path: string): RuntimeQueueStatus {
  const item = object(value, endpoint, path);
  return {
    active: optional(item, "active", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    waiting: optional(item, "waiting", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
    capacity: optional(item, "capacity", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, path),
  };
}

function decodeHealth(value: unknown, endpoint: string, allowedStatuses: readonly HealthStatus["status"][]): HealthStatus {
  const item = object(value, endpoint, "$");
  const status = enumValue(required(item, "status", endpoint, "$"), allowedStatuses, endpoint, "$.status");
  const service = required(item, "service", endpoint, "$");
  if (service !== "clozn") fail(endpoint, "$.service must be clozn");
  return {
    status,
    service,
    reason: optional(item, "reason", (v, p) => string(v, endpoint, p), endpoint, "$"),
    protocolVersion: optional(item, "protocol_version", (v, p) => string(v, endpoint, p), endpoint, "$"),
    model: optional(item, "model", (v, p) => string(v, endpoint, p), endpoint, "$"),
    mode: optional(item, "mode", (v, p) => string(v, endpoint, p), endpoint, "$"),
    worker: optional(item, "worker", (v, p) => decodeRuntimeWorker(v, endpoint, p), endpoint, "$"),
    capabilities: optional(item, "capabilities", (v, p) => jsonObject(v, endpoint, p), endpoint, "$"),
    // A gateway without a request gate reports null. That is unavailable, not an empty queue.
    queue: optional(item, "queue", (v, p) => v === null ? undefined : decodeRuntimeQueue(v, endpoint, p), endpoint, "$"),
  };
}

export function decodeLiveness(value: unknown): HealthStatus {
  return decodeHealth(value, "/healthz", ["ok"]);
}

export function decodeReadiness(value: unknown): HealthStatus {
  return decodeHealth(value, "/readyz", ["ok", "not_ready"]);
}

export interface RuntimeModel {
  modelId: string;
  state: string;
  isDefault: boolean;
  preloaded: boolean;
  runtimeKeySha256: string;
  workerGeneration?: number | null;
  workerId?: string | null;
  failureCode?: string | null;
}

export interface RuntimeModels {
  managed: boolean;
  defaultModelId: string | null;
  preloadModelIds: readonly string[];
  maxLoadedModels: number;
  configuredCount: number;
  residentCount: number;
  models: readonly RuntimeModel[];
}

export function decodeRuntimeModels(value: unknown, endpoint = "/runtime/models"): RuntimeModels {
  const item = object(value, endpoint, "$");
  for (const key of ["managed", "default_model_id", "preload_model_ids", "max_loaded_models", "configured_count", "resident_count", "models"]) required(item, key, endpoint, "$");
  const model = (entry: unknown, index: number): RuntimeModel => {
    const row = object(entry, endpoint, `$.models[${index}]`);
    for (const key of ["model_id", "state", "default", "preloaded", "runtime_key_sha256", "worker_generation", "worker_id", "failure_code"]) required(row, key, endpoint, `$.models[${index}]`);
    return {
      modelId: nonEmptyString(row.model_id, endpoint, `$.models[${index}].model_id`),
      state: nonEmptyString(row.state, endpoint, `$.models[${index}].state`),
      isDefault: boolean(row.default, endpoint, `$.models[${index}].default`),
      preloaded: boolean(row.preloaded, endpoint, `$.models[${index}].preloaded`),
      runtimeKeySha256: hash(row.runtime_key_sha256, endpoint, `$.models[${index}].runtime_key_sha256`),
      workerGeneration: row.worker_generation === null ? null : nonNegativeInteger(row.worker_generation, endpoint, `$.models[${index}].worker_generation`),
      workerId: row.worker_id === null ? null : string(row.worker_id, endpoint, `$.models[${index}].worker_id`),
      failureCode: row.failure_code === null ? null : string(row.failure_code, endpoint, `$.models[${index}].failure_code`),
    };
  };
  return {
    managed: boolean(item.managed, endpoint, "$.managed"),
    defaultModelId: item.default_model_id === null ? null : nonEmptyString(item.default_model_id, endpoint, "$.default_model_id"),
    preloadModelIds: array(item.preload_model_ids, endpoint, "$.preload_model_ids").map((entry, index) => nonEmptyString(entry, endpoint, `$.preload_model_ids[${index}]`)),
    maxLoadedModels: nonNegativeInteger(item.max_loaded_models, endpoint, "$.max_loaded_models"),
    configuredCount: nonNegativeInteger(item.configured_count, endpoint, "$.configured_count"),
    residentCount: nonNegativeInteger(item.resident_count, endpoint, "$.resident_count"),
    models: array(item.models, endpoint, "$.models").map(model),
  };
}

export interface InfluenceQuery {
  runId: string;
  target: { start: number; end: number; basisSha256?: string; answerSpanIds?: readonly string[] };
  measurement: MeasurementAvailability;
  links: readonly {
    sourceSpanId: string; answerSpanId: string; effect: InfluenceEffect; deltaNats: number;
    absDeltaNats: number; clearsFloor: boolean; evidenceState: EvidenceState;
    native?: { contextSpanId?: string; answerSpanId?: string; contextIndex?: number; answerIndex?: number };
  }[];
  summary: { selectedAnswerSpans: number; measuredLinks: number; returnedLinks: number; causallySupportedLinks: number; observedLinks: number; supportingLinks: number; suppressingLinks: number; neutralLinks: number };
}

function decodeInfluenceLink(value: unknown, endpoint: string, path: string): InfluenceQuery["links"][number] {
  const item = object(value, endpoint, path);
  closed(item, ["source_span_id", "answer_span_id", "effect", "delta_nats", "abs_delta_nats", "clears_floor", "evidence_state", "native"], endpoint, path);
  const native = item.native === undefined ? undefined : (() => {
    const value = object(item.native, endpoint, `${path}.native`);
    closed(value, ["context_span_id", "answer_span_id", "context_index", "answer_index"], endpoint, `${path}.native`);
    return {
      contextSpanId: optional(value, "context_span_id", (v, p) => string(v, endpoint, p), endpoint, `${path}.native`),
      answerSpanId: optional(value, "answer_span_id", (v, p) => string(v, endpoint, p), endpoint, `${path}.native`),
      contextIndex: optional(value, "context_index", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, `${path}.native`),
      answerIndex: optional(value, "answer_index", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, `${path}.native`),
    };
  })();
  return {
    sourceSpanId: spanId(required(item, "source_span_id", endpoint, path), endpoint, `${path}.source_span_id`),
    answerSpanId: spanId(required(item, "answer_span_id", endpoint, path), endpoint, `${path}.answer_span_id`),
    effect: enumValue(required(item, "effect", endpoint, path), ["supports", "suppresses", "neutral"] as const, endpoint, `${path}.effect`),
    deltaNats: number(required(item, "delta_nats", endpoint, path), endpoint, `${path}.delta_nats`),
    absDeltaNats: nonNegative(required(item, "abs_delta_nats", endpoint, path), endpoint, `${path}.abs_delta_nats`),
    clearsFloor: boolean(required(item, "clears_floor", endpoint, path), endpoint, `${path}.clears_floor`),
    evidenceState: enumValue(required(item, "evidence_state", endpoint, path), ["causally_supported", "observed"] as const, endpoint, `${path}.evidence_state`),
    native,
  };
}

function nonNegative(value: unknown, endpoint: string, path: string): number {
  const parsed = number(value, endpoint, path);
  if (parsed < 0) fail(endpoint, `${path} must be non-negative`);
  return parsed;
}

export function decodeInfluenceQuery(value: unknown, endpoint: string): InfluenceQuery {
  const item = object(value, endpoint, "$");
  closed(item, ["schema_version", "run_id", "privacy", "target", "measurement", "links", "summary"], endpoint, "$");
  if (required(item, "schema_version", endpoint, "$") !== "clozn.influence-query.v1") fail(endpoint, "$.schema_version is invalid");
  if (required(item, "privacy", endpoint, "$") !== "metadata_only") fail(endpoint, "$.privacy must be metadata_only");
  const target = object(required(item, "target", endpoint, "$"), endpoint, "$.target");
  closed(target, ["basis", "unit", "interval", "start", "end", "basis_sha256", "answer_span_ids"], endpoint, "$.target");
  if (target.basis !== "recorded_answer" || target.unit !== "unicode_code_points" || target.interval !== "half_open") fail(endpoint, "$.target has an unsupported offset contract");
  const summary = object(required(item, "summary", endpoint, "$"), endpoint, "$.summary");
  closed(summary, ["selected_answer_spans", "measured_links", "returned_links", "causally_supported_links", "observed_links", "supporting_links", "suppressing_links", "neutral_links"], endpoint, "$.summary");
  const readSummary = (key: string) => nonNegativeInteger(required(summary, key, endpoint, "$.summary"), endpoint, `$.summary.${key}`);
  const start = nonNegativeInteger(required(target, "start", endpoint, "$.target"), endpoint, "$.target.start");
  const end = nonNegativeInteger(required(target, "end", endpoint, "$.target"), endpoint, "$.target.end");
  if (end <= start) fail(endpoint, "$.target end must be after start");
  return {
    runId: nonEmptyString(required(item, "run_id", endpoint, "$"), endpoint, "$.run_id"),
    target: {
      start,
      end,
      basisSha256: optional(target, "basis_sha256", (v, p) => hash(v, endpoint, p), endpoint, "$.target"),
      answerSpanIds: optional(target, "answer_span_ids", (v, p) => array(v, endpoint, p).map((entry, index) => spanId(entry, endpoint, `${p}[${index}]`)), endpoint, "$.target"),
    },
    measurement: decodeMeasurement(required(item, "measurement", endpoint, "$"), endpoint, "$.measurement"),
    links: array(required(item, "links", endpoint, "$"), endpoint, "$.links").map((entry, index) => decodeInfluenceLink(entry, endpoint, `$.links[${index}]`)),
    summary: { selectedAnswerSpans: readSummary("selected_answer_spans"), measuredLinks: readSummary("measured_links"), returnedLinks: readSummary("returned_links"), causallySupportedLinks: readSummary("causally_supported_links"), observedLinks: readSummary("observed_links"), supportingLinks: readSummary("supporting_links"), suppressingLinks: readSummary("suppressing_links"), neutralLinks: readSummary("neutral_links") },
  };
}

export interface ContextTension {
  runId: string;
  target: { scope: "whole_answer" | "answer_range"; start?: number; end?: number; basisSha256?: string };
  measurement: MeasurementAvailability;
  tensions: readonly { tensionId: string; answerSpanId: string; supporting: TensionSide; suppressing: TensionSide }[];
  summary: { answerSpansExamined: number; answerSpansWithTension: number; tensionPairs: number; returnedTensionPairs: number; distinctSourceSpans: number };
}

interface TensionSide { sourceSpanId: string; deltaNats: number; absDeltaNats: number; effect: "supports" | "suppresses"; evidenceState: "causally_supported"; }

function decodeTensionSide(value: unknown, expectedEffect: TensionSide["effect"], endpoint: string, path: string): TensionSide {
  const item = object(value, endpoint, path);
  closed(item, ["source_span_id", "delta_nats", "abs_delta_nats", "effect", "evidence_state"], endpoint, path);
  if (required(item, "effect", endpoint, path) !== expectedEffect) fail(endpoint, `${path}.effect must be ${expectedEffect}`);
  if (required(item, "evidence_state", endpoint, path) !== "causally_supported") fail(endpoint, `${path}.evidence_state must be causally_supported`);
  return { sourceSpanId: spanId(required(item, "source_span_id", endpoint, path), endpoint, `${path}.source_span_id`), deltaNats: number(required(item, "delta_nats", endpoint, path), endpoint, `${path}.delta_nats`), absDeltaNats: nonNegative(required(item, "abs_delta_nats", endpoint, path), endpoint, `${path}.abs_delta_nats`), effect: expectedEffect, evidenceState: "causally_supported" };
}

export function decodeContextTension(value: unknown, endpoint: string): ContextTension {
  const item = object(value, endpoint, "$");
  closed(item, ["schema_version", "run_id", "privacy", "target", "measurement", "tensions", "summary"], endpoint, "$");
  if (required(item, "schema_version", endpoint, "$") !== "clozn.context-tension.v1") fail(endpoint, "$.schema_version is invalid");
  if (required(item, "privacy", endpoint, "$") !== "metadata_only") fail(endpoint, "$.privacy must be metadata_only");
  const target = object(required(item, "target", endpoint, "$"), endpoint, "$.target");
  closed(target, ["scope", "basis", "unit", "interval", "start", "end", "basis_sha256"], endpoint, "$.target");
  if (target.basis !== "recorded_answer" || target.unit !== "unicode_code_points" || target.interval !== "half_open") fail(endpoint, "$.target has an unsupported offset contract");
  const scope = enumValue(required(target, "scope", endpoint, "$.target"), ["whole_answer", "answer_range"] as const, endpoint, "$.target.scope");
  const start = optional(target, "start", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, "$.target");
  const end = optional(target, "end", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, "$.target");
  if ((scope === "whole_answer" && (start !== undefined || end !== undefined)) || (scope === "answer_range" && (start === undefined || end === undefined))) fail(endpoint, "$.target range does not match scope");
  if (start !== undefined && end !== undefined && end <= start) fail(endpoint, "$.target end must be after start");
  const tensions = array(required(item, "tensions", endpoint, "$"), endpoint, "$.tensions").map((entry, index) => {
    const row = object(entry, endpoint, `$.tensions[${index}]`);
    closed(row, ["tension_id", "answer_span_id", "supporting", "suppressing", "native"], endpoint, `$.tensions[${index}]`);
    const tensionId = string(required(row, "tension_id", endpoint, `$.tensions[${index}]`), endpoint, `$.tensions[${index}].tension_id`);
    if (!tensionIdPattern.test(tensionId)) fail(endpoint, `$.tensions[${index}].tension_id is invalid`);
    return { tensionId, answerSpanId: spanId(required(row, "answer_span_id", endpoint, `$.tensions[${index}]`), endpoint, `$.tensions[${index}].answer_span_id`), supporting: decodeTensionSide(required(row, "supporting", endpoint, `$.tensions[${index}]`), "supports", endpoint, `$.tensions[${index}].supporting`), suppressing: decodeTensionSide(required(row, "suppressing", endpoint, `$.tensions[${index}]`), "suppresses", endpoint, `$.tensions[${index}].suppressing`) };
  });
  const summary = object(required(item, "summary", endpoint, "$"), endpoint, "$.summary");
  closed(summary, ["answer_spans_examined", "answer_spans_with_tension", "tension_pairs", "returned_tension_pairs", "distinct_source_spans"], endpoint, "$.summary");
  const read = (key: string) => nonNegativeInteger(required(summary, key, endpoint, "$.summary"), endpoint, `$.summary.${key}`);
  return { runId: nonEmptyString(required(item, "run_id", endpoint, "$"), endpoint, "$.run_id"), target: { scope, start, end, basisSha256: optional(target, "basis_sha256", (v, p) => hash(v, endpoint, p), endpoint, "$.target") }, measurement: decodeMeasurement(required(item, "measurement", endpoint, "$"), endpoint, "$.measurement"), tensions, summary: { answerSpansExamined: read("answer_spans_examined"), answerSpansWithTension: read("answer_spans_with_tension"), tensionPairs: read("tension_pairs"), returnedTensionPairs: read("returned_tension_pairs"), distinctSourceSpans: read("distinct_source_spans") } };
}

export interface SuggestedBreakpoint {
  breakpointId: string;
  position: number;
  placement: "exact_token_decision" | "answer_span_entry_proxy";
  rankClass: "combined" | "meaningful_close_call" | "context_tension" | "close_call";
  tokenInterval?: { start: number; end: number };
  closeCall?: {
    emittedTokenId?: number;
    rivalTokenId?: number;
    emittedProbability?: number;
    rivalProbability?: number;
    margin?: number;
    meaningful: boolean;
  };
}

export interface SuggestedBreakpoints {
  runId: string;
  analysisState: "available" | "partially_available" | "unavailable";
  breakpoints: readonly SuggestedBreakpoint[];
  summary: {
    candidateState: "detected" | "none_detected" | "unavailable";
    suggestedBreakpoints: number;
    returnedBreakpoints: number;
    ordinaryCloseCalls: number;
    meaningfulCloseCalls: number;
    combinedBreakpoints: number;
  };
}

function decodeTokenInterval(value: unknown, endpoint: string, path: string): { start: number; end: number } {
  const row = object(value, endpoint, path);
  closed(row, ["start", "end", "unit", "interval"], endpoint, path);
  if (row.unit !== "unicode_code_points" || row.interval !== "half_open") fail(endpoint, `${path} has an unsupported offset contract`);
  const start = nonNegativeInteger(required(row, "start", endpoint, path), endpoint, `${path}.start`);
  const end = nonNegativeInteger(required(row, "end", endpoint, path), endpoint, `${path}.end`);
  if (end <= start) fail(endpoint, `${path}.end must be after start`);
  return { start, end };
}

/** Decode the read-only backend locator; the UI never ranks or drops returned close calls. */
export function decodeSuggestedBreakpoints(value: unknown, endpoint: string): SuggestedBreakpoints {
  const item = object(value, endpoint, "$");
  closed(item, ["schema_version", "run_id", "privacy", "coordinates", "analysis", "evidence", "breakpoints", "summary"], endpoint, "$");
  if (item.schema_version !== "clozn.suggested-breakpoints.v1" || item.privacy !== "metadata_only") fail(endpoint, "response is not a metadata-only suggested-breakpoints document");
  const analysis = object(required(item, "analysis", endpoint, "$"), endpoint, "$.analysis");
  closed(analysis, ["state", "reason"], endpoint, "$.analysis");
  const analysisState = enumValue(required(analysis, "state", endpoint, "$.analysis"), ["available", "partially_available", "unavailable"] as const, endpoint, "$.analysis.state");
  optional(analysis, "reason", (v, p) => string(v, endpoint, p), endpoint, "$.analysis");
  // Validate required evidence/coordinate projections even though this surface consumes only locators.
  object(required(item, "coordinates", endpoint, "$"), endpoint, "$.coordinates");
  object(required(item, "evidence", endpoint, "$"), endpoint, "$.evidence");

  const breakpoints = array(required(item, "breakpoints", endpoint, "$"), endpoint, "$.breakpoints").map((raw, index): SuggestedBreakpoint => {
    const path = `$.breakpoints[${index}]`;
    const row = object(raw, endpoint, path);
    closed(row, ["breakpoint_id", "position", "placement", "rank_class", "token_interval", "reasons"], endpoint, path);
    const breakpointId = string(required(row, "breakpoint_id", endpoint, path), endpoint, `${path}.breakpoint_id`);
    if (!breakpointIdPattern.test(breakpointId)) fail(endpoint, `${path}.breakpoint_id is invalid`);
    const reasons = array(required(row, "reasons", endpoint, path), endpoint, `${path}.reasons`);
    if (!reasons.length) fail(endpoint, `${path}.reasons must not be empty`);
    let closeCall: SuggestedBreakpoint["closeCall"];
    for (const [reasonIndex, reasonRaw] of reasons.entries()) {
      const reasonPath = `${path}.reasons[${reasonIndex}]`;
      const reason = object(reasonRaw, endpoint, reasonPath);
      const type = required(reason, "type", endpoint, reasonPath);
      if (type === "close_call") {
        closed(reason, ["type", "emitted_token_id", "rival_token_id", "emitted_probability", "rival_probability", "margin", "meaningful_heuristic"], endpoint, reasonPath);
        const probability = (key: string) => optional(reason, key, (v, p) => {
          const parsed = number(v, endpoint, p);
          if (parsed < 0 || parsed > 1) fail(endpoint, `${p} must be in [0, 1]`);
          return parsed;
        }, endpoint, reasonPath);
        closeCall = {
          emittedTokenId: optional(reason, "emitted_token_id", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, reasonPath),
          rivalTokenId: optional(reason, "rival_token_id", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, reasonPath),
          emittedProbability: probability("emitted_probability"),
          rivalProbability: probability("rival_probability"),
          margin: optional(reason, "margin", (v, p) => number(v, endpoint, p), endpoint, reasonPath),
          meaningful: boolean(required(reason, "meaningful_heuristic", endpoint, reasonPath), endpoint, `${reasonPath}.meaningful_heuristic`),
        };
      } else if (type === "context_tension") {
        // Context-tension detail is displayed by its dedicated surface. Still verify its stable identity.
        array(required(reason, "answer_span_ids", endpoint, reasonPath), endpoint, `${reasonPath}.answer_span_ids`).forEach((id, idIndex) => spanId(id, endpoint, `${reasonPath}.answer_span_ids[${idIndex}]`));
      } else fail(endpoint, `${reasonPath}.type is unsupported`);
    }
    return {
      breakpointId,
      position: nonNegativeInteger(required(row, "position", endpoint, path), endpoint, `${path}.position`),
      placement: enumValue(required(row, "placement", endpoint, path), ["exact_token_decision", "answer_span_entry_proxy"] as const, endpoint, `${path}.placement`),
      rankClass: enumValue(required(row, "rank_class", endpoint, path), ["combined", "meaningful_close_call", "context_tension", "close_call"] as const, endpoint, `${path}.rank_class`),
      tokenInterval: row.token_interval === undefined ? undefined : decodeTokenInterval(row.token_interval, endpoint, `${path}.token_interval`),
      closeCall,
    };
  });
  const summary = object(required(item, "summary", endpoint, "$"), endpoint, "$.summary");
  closed(summary, ["candidate_state", "suggested_breakpoints", "returned_breakpoints", "combined_breakpoints", "meaningful_close_call_breakpoints", "context_tension_breakpoints", "ordinary_close_call_breakpoints"], endpoint, "$.summary");
  const count = (key: string) => nonNegativeInteger(required(summary, key, endpoint, "$.summary"), endpoint, `$.summary.${key}`);
  return {
    runId: nonEmptyString(required(item, "run_id", endpoint, "$"), endpoint, "$.run_id"),
    analysisState,
    breakpoints,
    summary: {
      candidateState: enumValue(required(summary, "candidate_state", endpoint, "$.summary"), ["detected", "none_detected", "unavailable"] as const, endpoint, "$.summary.candidate_state"),
      suggestedBreakpoints: count("suggested_breakpoints"),
      returnedBreakpoints: count("returned_breakpoints"),
      ordinaryCloseCalls: count("ordinary_close_call_breakpoints"),
      meaningfulCloseCalls: count("meaningful_close_call_breakpoints"),
      combinedBreakpoints: count("combined_breakpoints"),
    },
  };
}

export interface SpanAddressDocument {
  runId: string;
  privacy: "full" | "metadata_only";
  addresses: readonly SpanAddress[];
}

export type ComparisonAxisState = "changed" | "unchanged" | "unavailable";
export type ComparisonFindingState = "observed" | "eliminated" | "reproduced" | "correlated" | "causally_supported";

export interface RunComparison {
  runA: string;
  runB: string;
  generatedAt?: string;
  privacyLimited: boolean;
  selection?: { mode: string; referenceRunId: string; candidateRunId: string; reason: string; basis?: JsonObject };
  axes: Readonly<Record<string, { status: ComparisonAxisState; note?: string }>>;
  differences: readonly {
    dimension: string;
    kind: "added" | "removed" | "changed" | "unavailable" | "diff_failed";
    rank: number;
    valueA?: unknown;
    valueB?: unknown;
    evidence: readonly unknown[];
    note?: string;
  }[];
  findings: readonly {
    classification: string;
    status: ComparisonFindingState;
    summary: string;
    dimensions: readonly string[];
  }[];
  ranking?: { order: readonly string[]; note?: string };
}

export function decodeRunComparison(value: unknown, endpoint: string): RunComparison {
  const item = object(value, endpoint, "$");
  closed(item, ["schema_version", "ok", "run_a", "run_b", "comparison_selection", "summary_axes", "generated_at", "differences", "findings", "ranking", "privacy_limited"], endpoint, "$");
  if (required(item, "schema_version", endpoint, "$") !== "clozn.run-diff.v1") fail(endpoint, "$.schema_version is invalid");
  if (item.ok !== undefined && item.ok !== true) fail(endpoint, "$.ok must be true for a comparison document");
  const axesRaw = object(required(item, "summary_axes", endpoint, "$"), endpoint, "$.summary_axes");
  const axes: Record<string, { status: ComparisonAxisState; note?: string }> = {};
  for (const [key, raw] of Object.entries(axesRaw)) {
    const axis = object(raw, endpoint, `$.summary_axes.${key}`);
    closed(axis, ["status", "note"], endpoint, `$.summary_axes.${key}`);
    axes[key] = {
      status: enumValue(required(axis, "status", endpoint, `$.summary_axes.${key}`), ["changed", "unchanged", "unavailable"] as const, endpoint, `$.summary_axes.${key}.status`),
      note: optional(axis, "note", (v, p) => string(v, endpoint, p), endpoint, `$.summary_axes.${key}`),
    };
  }
  const selection = item.comparison_selection === undefined ? undefined : (() => {
    const row = object(item.comparison_selection, endpoint, "$.comparison_selection");
    return {
      mode: nonEmptyString(required(row, "mode", endpoint, "$.comparison_selection"), endpoint, "$.comparison_selection.mode"),
      referenceRunId: nonEmptyString(required(row, "reference_run_id", endpoint, "$.comparison_selection"), endpoint, "$.comparison_selection.reference_run_id"),
      candidateRunId: nonEmptyString(required(row, "candidate_run_id", endpoint, "$.comparison_selection"), endpoint, "$.comparison_selection.candidate_run_id"),
      reason: nonEmptyString(required(row, "reason", endpoint, "$.comparison_selection"), endpoint, "$.comparison_selection.reason"),
      basis: optional(row, "basis", (v, p) => jsonObject(v, endpoint, p), endpoint, "$.comparison_selection"),
    };
  })();
  const differences = array(required(item, "differences", endpoint, "$"), endpoint, "$.differences").map((raw, index) => {
    const row = object(raw, endpoint, `$.differences[${index}]`);
    return {
      dimension: nonEmptyString(required(row, "dimension", endpoint, `$.differences[${index}]`), endpoint, `$.differences[${index}].dimension`),
      kind: enumValue(required(row, "kind", endpoint, `$.differences[${index}]`), ["added", "removed", "changed", "unavailable", "diff_failed"] as const, endpoint, `$.differences[${index}].kind`),
      rank: nonNegativeInteger(required(row, "rank", endpoint, `$.differences[${index}]`), endpoint, `$.differences[${index}].rank`),
      valueA: row.value_a,
      valueB: row.value_b,
      evidence: array(required(row, "evidence", endpoint, `$.differences[${index}]`), endpoint, `$.differences[${index}].evidence`),
      note: optional(row, "note", (v, p) => string(v, endpoint, p), endpoint, `$.differences[${index}]`),
    };
  });
  const findings = array(required(item, "findings", endpoint, "$"), endpoint, "$.findings").map((raw, index) => {
    const row = object(raw, endpoint, `$.findings[${index}]`);
    return {
      classification: nonEmptyString(required(row, "classification", endpoint, `$.findings[${index}]`), endpoint, `$.findings[${index}].classification`),
      status: enumValue(required(row, "status", endpoint, `$.findings[${index}]`), ["observed", "eliminated", "reproduced", "correlated", "causally_supported"] as const, endpoint, `$.findings[${index}].status`),
      summary: nonEmptyString(required(row, "summary", endpoint, `$.findings[${index}]`), endpoint, `$.findings[${index}].summary`),
      dimensions: optional(row, "dimensions", (v, p) => array(v, endpoint, p).map((entry, entryIndex) => string(entry, endpoint, `${p}[${entryIndex}]`)), endpoint, `$.findings[${index}]`) ?? [],
    };
  });
  const ranking = item.ranking === undefined ? undefined : (() => {
    const row = object(item.ranking, endpoint, "$.ranking");
    return {
      order: array(required(row, "order", endpoint, "$.ranking"), endpoint, "$.ranking.order").map((entry, index) => string(entry, endpoint, `$.ranking.order[${index}]`)),
      note: optional(row, "note", (v, p) => string(v, endpoint, p), endpoint, "$.ranking"),
    };
  })();
  return {
    runA: nonEmptyString(required(item, "run_a", endpoint, "$"), endpoint, "$.run_a"),
    runB: nonEmptyString(required(item, "run_b", endpoint, "$"), endpoint, "$.run_b"),
    generatedAt: optional(item, "generated_at", (v, p) => string(v, endpoint, p), endpoint, "$"),
    privacyLimited: optional(item, "privacy_limited", (v, p) => boolean(v, endpoint, p), endpoint, "$") ?? false,
    selection,
    axes,
    differences,
    findings,
    ranking,
  };
}

export interface RewindFidelity {
  runId: string;
  recordedTokenCount?: number;
  reconstructedReplay: {
    state: "available" | "unavailable";
    supportedChangeTypes?: readonly string[];
    unavoidableDifferences?: readonly string[];
    reasons?: readonly string[];
  };
  exactRewind: {
    state: "requires_live_plan" | "static_prerequisites_unavailable";
    staticPrerequisites: Readonly<Record<string, "available" | "unavailable">>;
    liveRequirements?: readonly string[];
    reasons?: readonly string[];
  };
  historicalProof: {
    state: "available" | "partially_unavailable";
    malformedReceiptCount?: number;
    verifiedBoundaries: readonly {
      position: number;
      state: "historically_verified_exact";
      verifiedExecutionCount: number;
      latestExecutionId: string;
      regimes: readonly string[];
    }[];
  };
  liveExecution: { state: "not_checked"; reason: "read_only_projection"; authority: "execution_fork_plan" };
}

export function decodeRewindFidelity(value: unknown, endpoint: string): RewindFidelity {
  const item = object(value, endpoint, "$");
  closed(item, ["schema_version", "run_id", "privacy", "coordinates", "recorded_capability", "historical_proof", "live_execution"], endpoint, "$");
  if (required(item, "schema_version", endpoint, "$") !== "clozn.rewind-fidelity.v1" || item.privacy !== "metadata_only") fail(endpoint, "response is not a metadata-only rewind-fidelity document");
  const capability = object(required(item, "recorded_capability", endpoint, "$"), endpoint, "$.recorded_capability");
  const reconstructed = object(required(capability, "reconstructed_replay", endpoint, "$.recorded_capability"), endpoint, "$.recorded_capability.reconstructed_replay");
  const exact = object(required(capability, "exact_rewind", endpoint, "$.recorded_capability"), endpoint, "$.recorded_capability.exact_rewind");
  const prerequisites = object(required(exact, "static_prerequisites", endpoint, "$.recorded_capability.exact_rewind"), endpoint, "$.recorded_capability.exact_rewind.static_prerequisites");
  const staticPrerequisites: Record<string, "available" | "unavailable"> = {};
  for (const [key, raw] of Object.entries(prerequisites)) staticPrerequisites[key] = enumValue(raw, ["available", "unavailable"] as const, endpoint, `$.recorded_capability.exact_rewind.static_prerequisites.${key}`);
  const historical = object(required(item, "historical_proof", endpoint, "$"), endpoint, "$.historical_proof");
  const verifiedBoundaries = array(required(historical, "verified_boundaries", endpoint, "$.historical_proof"), endpoint, "$.historical_proof.verified_boundaries").map((raw, index) => {
    const row = object(raw, endpoint, `$.historical_proof.verified_boundaries[${index}]`);
    if (row.state !== "historically_verified_exact") fail(endpoint, `$.historical_proof.verified_boundaries[${index}].state is invalid`);
    return {
      position: nonNegativeInteger(required(row, "position", endpoint, `$.historical_proof.verified_boundaries[${index}]`), endpoint, `$.historical_proof.verified_boundaries[${index}].position`),
      state: "historically_verified_exact" as const,
      verifiedExecutionCount: nonNegativeInteger(required(row, "verified_execution_count", endpoint, `$.historical_proof.verified_boundaries[${index}]`), endpoint, `$.historical_proof.verified_boundaries[${index}].verified_execution_count`),
      latestExecutionId: nonEmptyString(required(row, "latest_execution_id", endpoint, `$.historical_proof.verified_boundaries[${index}]`), endpoint, `$.historical_proof.verified_boundaries[${index}].latest_execution_id`),
      regimes: array(required(row, "regimes", endpoint, `$.historical_proof.verified_boundaries[${index}]`), endpoint, `$.historical_proof.verified_boundaries[${index}].regimes`).map((entry, entryIndex) => string(entry, endpoint, `$.historical_proof.verified_boundaries[${index}].regimes[${entryIndex}]`)),
    };
  });
  const live = object(required(item, "live_execution", endpoint, "$"), endpoint, "$.live_execution");
  if (live.state !== "not_checked" || live.reason !== "read_only_projection" || live.authority !== "execution_fork_plan") fail(endpoint, "$.live_execution has an unsupported state");
  const stringList = (row: JsonRecord, key: string, path: string) => optional(row, key, (v, p) => array(v, endpoint, p).map((entry, index) => string(entry, endpoint, `${p}[${index}]`)), endpoint, path);
  const coordinates = item.coordinates === undefined ? undefined : object(item.coordinates, endpoint, "$.coordinates");
  return {
    runId: nonEmptyString(required(item, "run_id", endpoint, "$"), endpoint, "$.run_id"),
    recordedTokenCount: coordinates === undefined ? undefined : nonNegativeInteger(required(coordinates, "recorded_token_count", endpoint, "$.coordinates"), endpoint, "$.coordinates.recorded_token_count"),
    reconstructedReplay: {
      state: enumValue(required(reconstructed, "state", endpoint, "$.recorded_capability.reconstructed_replay"), ["available", "unavailable"] as const, endpoint, "$.recorded_capability.reconstructed_replay.state"),
      supportedChangeTypes: stringList(reconstructed, "supported_change_types", "$.recorded_capability.reconstructed_replay"),
      unavoidableDifferences: stringList(reconstructed, "unavoidable_differences", "$.recorded_capability.reconstructed_replay"),
      reasons: stringList(reconstructed, "reasons", "$.recorded_capability.reconstructed_replay"),
    },
    exactRewind: {
      state: enumValue(required(exact, "state", endpoint, "$.recorded_capability.exact_rewind"), ["requires_live_plan", "static_prerequisites_unavailable"] as const, endpoint, "$.recorded_capability.exact_rewind.state"),
      staticPrerequisites,
      liveRequirements: stringList(exact, "live_requirements", "$.recorded_capability.exact_rewind"),
      reasons: stringList(exact, "reasons", "$.recorded_capability.exact_rewind"),
    },
    historicalProof: {
      state: enumValue(required(historical, "state", endpoint, "$.historical_proof"), ["available", "partially_unavailable"] as const, endpoint, "$.historical_proof.state"),
      malformedReceiptCount: optional(historical, "malformed_receipt_count", (v, p) => nonNegativeInteger(v, endpoint, p), endpoint, "$.historical_proof"),
      verifiedBoundaries,
    },
    liveExecution: { state: "not_checked", reason: "read_only_projection", authority: "execution_fork_plan" },
  };
}
export interface SpanAddress {
  addressId: string; runId: string; kind: "delivered_message" | "rendered_prompt_segment" | "attached_source_span" | "answer_span" | "claim"; relationKey: string;
  nativeRef: { artifactSchema: string; collection: string; id: string; parentId?: string; segmentId?: string; clientSourceId?: string; sourceLabel?: string; selected?: boolean };
  resolution: { state: "exact" | "metadata_only" | "drifted" | "redacted" | "unavailable"; reason?: string; canonical?: { basis: string; start: number; end: number; basisSha256: string; spanSha256: string; text?: string } };
}

export function decodeSpanAddressDocument(value: unknown, endpoint: string): SpanAddressDocument {
  const item = object(value, endpoint, "$");
  closed(item, ["schema_version", "run_id", "privacy", "offset_contract", "source_artifacts", "addresses", "lineage"], endpoint, "$");
  if (required(item, "schema_version", endpoint, "$") !== "clozn.text-span-addresses.v1") fail(endpoint, "$.schema_version is invalid");
  const privacy = enumValue(required(item, "privacy", endpoint, "$"), ["full", "metadata_only"] as const, endpoint, "$.privacy");
  const contract = object(required(item, "offset_contract", endpoint, "$"), endpoint, "$.offset_contract");
  closed(contract, ["unit", "interval", "hash_algorithm", "canonicalization"], endpoint, "$.offset_contract");
  for (const key of ["unit", "interval", "hash_algorithm", "canonicalization"]) required(contract, key, endpoint, "$.offset_contract");
  if (contract.unit !== "unicode_code_points" || contract.interval !== "half_open" || contract.hash_algorithm !== "sha256" || contract.canonicalization !== "exact_string_utf8_v1") fail(endpoint, "$.offset_contract is invalid");
  // Validate even currently-unused required projections so a malformed document never slips through.
  array(required(item, "source_artifacts", endpoint, "$"), endpoint, "$.source_artifacts").forEach((entry, index) => {
    const source = object(entry, endpoint, `$.source_artifacts[${index}]`);
    closed(source, ["schema", "native_status", "available", "privacy", "reason", "artifact_sha256", "method"], endpoint, `$.source_artifacts[${index}]`);
    nonEmptyString(required(source, "schema", endpoint, `$.source_artifacts[${index}]`), endpoint, `$.source_artifacts[${index}].schema`);
    optional(source, "native_status", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `$.source_artifacts[${index}]`);
    optional(source, "available", (v, p) => boolean(v, endpoint, p), endpoint, `$.source_artifacts[${index}]`);
    optional(source, "privacy", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `$.source_artifacts[${index}]`);
    optional(source, "reason", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `$.source_artifacts[${index}]`);
    optional(source, "artifact_sha256", (v, p) => hash(v, endpoint, p), endpoint, `$.source_artifacts[${index}]`);
    optional(source, "method", (v, p) => jsonObject(v, endpoint, p), endpoint, `$.source_artifacts[${index}]`);
  });
  const lineage = object(required(item, "lineage", endpoint, "$"), endpoint, "$.lineage");
  closed(lineage, ["parent_run_id", "mappings"], endpoint, "$.lineage");
  if (required(lineage, "parent_run_id", endpoint, "$.lineage") !== null) nonEmptyString(lineage.parent_run_id, endpoint, "$.lineage.parent_run_id");
  array(required(lineage, "mappings", endpoint, "$.lineage"), endpoint, "$.lineage.mappings");
  const kinds = ["delivered_message", "rendered_prompt_segment", "attached_source_span", "answer_span", "claim"] as const;
  const addresses = array(required(item, "addresses", endpoint, "$"), endpoint, "$.addresses").map((entry, index): SpanAddress => {
    const row = object(entry, endpoint, `$.addresses[${index}]`);
    closed(row, ["address_id", "run_id", "kind", "relation_key", "native_ref", "resolution"], endpoint, `$.addresses[${index}]`);
    const relationKey = string(required(row, "relation_key", endpoint, `$.addresses[${index}]`), endpoint, `$.addresses[${index}].relation_key`);
    if (!relationIdPattern.test(relationKey)) fail(endpoint, `$.addresses[${index}].relation_key is invalid`);
    const native = object(required(row, "native_ref", endpoint, `$.addresses[${index}]`), endpoint, `$.addresses[${index}].native_ref`);
    const resolution = object(required(row, "resolution", endpoint, `$.addresses[${index}]`), endpoint, `$.addresses[${index}].resolution`);
    closed(native, ["artifact_schema", "collection", "id", "parent_id", "segment_id", "client_source_id", "source_label", "selected", "recorded_hash"], endpoint, `$.addresses[${index}].native_ref`);
    closed(resolution, ["state", "canonical", "reason"], endpoint, `$.addresses[${index}].resolution`);
    const state = enumValue(required(resolution, "state", endpoint, `$.addresses[${index}].resolution`), ["exact", "metadata_only", "drifted", "redacted", "unavailable"] as const, endpoint, `$.addresses[${index}].resolution.state`);
    const canonical = resolution.canonical === undefined ? undefined : (() => {
      const c = object(resolution.canonical, endpoint, `$.addresses[${index}].resolution.canonical`);
      closed(c, ["basis", "unit", "interval", "start", "end", "basis_sha256", "span_sha256", "basis_code_points", "span_code_points", "basis_utf8_bytes", "span_utf8_bytes", "text"], endpoint, `$.addresses[${index}].resolution.canonical`);
      if (c.unit !== "unicode_code_points" || c.interval !== "half_open") fail(endpoint, `$.addresses[${index}].resolution.canonical has an invalid offset contract`);
      const parsed = { basis: nonEmptyString(required(c, "basis", endpoint, "canonical"), endpoint, `$.addresses[${index}].resolution.canonical.basis`), start: nonNegativeInteger(required(c, "start", endpoint, "canonical"), endpoint, `$.addresses[${index}].resolution.canonical.start`), end: nonNegativeInteger(required(c, "end", endpoint, "canonical"), endpoint, `$.addresses[${index}].resolution.canonical.end`), basisSha256: hash(required(c, "basis_sha256", endpoint, "canonical"), endpoint, `$.addresses[${index}].resolution.canonical.basis_sha256`), spanSha256: hash(required(c, "span_sha256", endpoint, "canonical"), endpoint, `$.addresses[${index}].resolution.canonical.span_sha256`), text: optional(c, "text", (v, p) => string(v, endpoint, p), endpoint, `$.addresses[${index}].resolution.canonical`) };
      if (parsed.end < parsed.start) fail(endpoint, `$.addresses[${index}].resolution.canonical end precedes start`);
      if (privacy === "metadata_only" && parsed.text !== undefined) fail(endpoint, "metadata_only address leaks canonical text");
      return parsed;
    })();
    const reason = optional(resolution, "reason", (v, p) => string(v, endpoint, p), endpoint, `$.addresses[${index}].resolution`);
    if ((state === "exact" || state === "metadata_only") && (!canonical || reason !== undefined)) fail(endpoint, `$.addresses[${index}].resolution has an invalid ${state} shape`);
    if (state === "exact" && canonical?.text === undefined) fail(endpoint, `$.addresses[${index}].resolution.exact requires canonical text`);
    if (state === "drifted" && (!canonical || !reason)) fail(endpoint, `$.addresses[${index}].resolution has an invalid drifted shape`);
    if ((state === "redacted" || state === "unavailable") && (canonical || !reason)) fail(endpoint, `$.addresses[${index}].resolution has an invalid unavailable shape`);
    return { addressId: spanId(required(row, "address_id", endpoint, `$.addresses[${index}]`), endpoint, `$.addresses[${index}].address_id`), runId: nonEmptyString(required(row, "run_id", endpoint, `$.addresses[${index}]`), endpoint, `$.addresses[${index}].run_id`), kind: enumValue(required(row, "kind", endpoint, `$.addresses[${index}]`), kinds, endpoint, `$.addresses[${index}].kind`), relationKey, nativeRef: { artifactSchema: nonEmptyString(required(native, "artifact_schema", endpoint, "native_ref"), endpoint, `$.addresses[${index}].native_ref.artifact_schema`), collection: nonEmptyString(required(native, "collection", endpoint, "native_ref"), endpoint, `$.addresses[${index}].native_ref.collection`), id: nonEmptyString(required(native, "id", endpoint, "native_ref"), endpoint, `$.addresses[${index}].native_ref.id`), parentId: optional(native, "parent_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `$.addresses[${index}].native_ref`), segmentId: optional(native, "segment_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `$.addresses[${index}].native_ref`), clientSourceId: optional(native, "client_source_id", (v, p) => nonEmptyString(v, endpoint, p), endpoint, `$.addresses[${index}].native_ref`), sourceLabel: optional(native, "source_label", (v, p) => string(v, endpoint, p), endpoint, `$.addresses[${index}].native_ref`), selected: optional(native, "selected", (v, p) => boolean(v, endpoint, p), endpoint, `$.addresses[${index}].native_ref`) }, resolution: { state, reason, canonical } };
  });
  return { runId: nonEmptyString(required(item, "run_id", endpoint, "$"), endpoint, "$.run_id"), privacy, addresses };
}
