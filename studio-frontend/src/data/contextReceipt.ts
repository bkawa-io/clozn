import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type ContextReceiptShape = "recorded" | "absent" | "unrecognized";
export type ReceiptPrivacy = "full" | "metadata_only" | "hashes_only" | "off";
export type ReceiptRedactionState = "full" | "redacted" | "hash_only";
export type ReceiptSegmentReason =
  | "context_budget"
  | "client_not_delivered"
  | "history_policy"
  | "attachment_limit"
  | "deduplicated"
  | "empty_content"
  | "unsupported_role"
  | "tool_result_pruned"
  | "template_transformed"
  | "system_message_merged"
  | "encoding_error";
export type ReceiptTerminationReason =
  | "eos"
  | "stop_sequence"
  | "max_tokens"
  | "context_limit"
  | "client_cancelled"
  | "timeout"
  | "worker_error"
  | "tool_call"
  | "content_filter"
  | "unknown";
export type ReceiptProvenanceKind = "message";

export interface ReceiptSourceSpan {
  sourceId: string;
  segmentId: string;
  messageIndex: number;
  unicodeRange: readonly [number, number];
  byteRange: readonly [number, number];
  contentSha256: string;
  provenanceKind: ReceiptProvenanceKind;
}

export interface ReceiptSegment {
  segmentId: string;
  sourceType: "message";
  originalOrder: number;
  sourceLabel?: string;
  deliveredBytes?: number;
  deliveredTokens?: number;
  included?: boolean;
  contentHash?: string;
  sources?: readonly ReceiptSourceSpan[];
  reason?: ReceiptSegmentReason;
  redactionState?: ReceiptRedactionState;
}

export interface ReceiptRendered {
  sha256?: string;
  bytes?: number;
  tokens?: number;
  templateFingerprint?: string;
  estimated?: boolean;
  specialTokens?: readonly string[];
}

export interface ReceiptOmission {
  segmentId: string;
  reason: ReceiptSegmentReason;
  detail?: string;
}

export interface ReceiptTransformation {
  reason: ReceiptSegmentReason;
  segmentIds: readonly string[];
  detail?: string;
}

export interface ReceiptTermination {
  reason: ReceiptTerminationReason;
  reasonRaw?: string;
  source?: string;
  generatedTokens?: number;
}

export interface ContextReceiptV1 {
  kind: "v1";
  schemaVersion: "clozn.context-receipt.v1";
  runId: string;
  templateFingerprint?: string;
  tokenizerConflatedWithTemplate?: boolean;
  contextWindowTokens?: number;
  reservedOutputTokens?: number;
  delivered?: readonly ReceiptSegment[];
  assembled?: readonly ReceiptSegment[];
  rendered?: ReceiptRendered;
  omissions?: readonly ReceiptOmission[];
  transformations?: readonly ReceiptTransformation[];
  termination?: ReceiptTermination;
  privacy?: ReceiptPrivacy;
}

/** A future schema is recorded evidence, but its facts are not interpreted by this v1 decoder. */
export interface UnsupportedContextReceipt {
  kind: "unsupported";
  schemaVersion: string;
  runId: string;
  rawKeys: readonly string[];
}

export type ContextReceiptView =
  | { shape: "absent"; runId: string; note: string }
  | { shape: "unrecognized"; runId: string; schemaVersionRaw?: string; rawKeys: readonly string[] }
  | { shape: "recorded"; runId: string; receipt: ContextReceiptV1 | UnsupportedContextReceipt };

const SEGMENT_REASONS = [
  "context_budget", "client_not_delivered", "history_policy", "attachment_limit",
  "deduplicated", "empty_content", "unsupported_role", "tool_result_pruned",
  "template_transformed", "system_message_merged", "encoding_error",
] as const;
const TERMINATION_REASONS = [
  "eos", "stop_sequence", "max_tokens", "context_limit", "client_cancelled",
  "timeout", "worker_error", "tool_call", "content_filter", "unknown",
] as const;
const PRIVACY_TIERS = ["full", "metadata_only", "hashes_only", "off"] as const;
const REDACTION_STATES = ["full", "redacted", "hash_only"] as const;
const PROVENANCE_KINDS = ["message"] as const;

const SEGMENT_ID = /^seg_[0-9a-f]{16}(?:_[0-9]+)?$/;
const SOURCE_ID = /^src_[0-9a-f]{24}$/;
const HASH_16 = /^[0-9a-f]{16}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const RUN_ID = /^[A-Za-z0-9_-]+$/;

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(endpoint, path, "must be an object");
  return value as JsonRecord;
}

function array(value: unknown, endpoint: string, path: string): unknown[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function closed(row: JsonRecord, keys: readonly string[], endpoint: string, path: string): void {
  const allowed = new Set(keys);
  for (const key of Object.keys(row)) if (!allowed.has(key)) fail(endpoint, `${path}.${key}`, "is not part of the contract");
}

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nonEmptyText(value: unknown, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!parsed.trim()) fail(endpoint, path, "must not be empty");
  return parsed;
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(endpoint, path, "must be a non-negative integer");
  }
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

function pattern(value: unknown, expression: RegExp, label: string, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!expression.test(parsed)) fail(endpoint, path, `must be ${label}`);
  return parsed;
}

function range(value: unknown, endpoint: string, path: string): readonly [number, number] {
  const items = array(value, endpoint, path);
  if (items.length !== 2) fail(endpoint, path, "must contain exactly two coordinates");
  const start = nonNegativeInteger(items[0], endpoint, `${path}[0]`);
  const end = nonNegativeInteger(items[1], endpoint, `${path}[1]`);
  if (end <= start) fail(endpoint, path, "must be a non-empty half-open range");
  return [start, end];
}

function decodeSourceSpan(
  value: unknown,
  segmentId: string,
  endpoint: string,
  path: string,
): ReceiptSourceSpan {
  const row = object(value, endpoint, path);
  closed(row, [
    "source_id", "segment_id", "message_index", "unicode_range", "byte_range", "content_sha256",
    "provenance_kind",
  ], endpoint, path);
  const decodedSegmentId = pattern(required(row, "segment_id", endpoint, path), SEGMENT_ID, "a segment id", endpoint, `${path}.segment_id`);
  if (decodedSegmentId !== segmentId) fail(endpoint, `${path}.segment_id`, "must match its containing segment");
  const span: ReceiptSourceSpan = {
    sourceId: pattern(required(row, "source_id", endpoint, path), SOURCE_ID, "a canonical source id", endpoint, `${path}.source_id`),
    segmentId: decodedSegmentId,
    messageIndex: nonNegativeInteger(required(row, "message_index", endpoint, path), endpoint, `${path}.message_index`),
    unicodeRange: range(required(row, "unicode_range", endpoint, path), endpoint, `${path}.unicode_range`),
    byteRange: range(required(row, "byte_range", endpoint, path), endpoint, `${path}.byte_range`),
    contentSha256: pattern(required(row, "content_sha256", endpoint, path), SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.content_sha256`),
    provenanceKind: oneOf(required(row, "provenance_kind", endpoint, path), PROVENANCE_KINDS, endpoint, `${path}.provenance_kind`),
  };
  return span;
}

function decodeSegment(value: unknown, endpoint: string, path: string): ReceiptSegment {
  const row = object(value, endpoint, path);
  const segmentId = pattern(required(row, "segment_id", endpoint, path), SEGMENT_ID, "a segment id", endpoint, `${path}.segment_id`);
  const segment: ReceiptSegment = {
    segmentId,
    sourceType: oneOf(required(row, "source_type", endpoint, path), ["message"] as const, endpoint, `${path}.source_type`),
    originalOrder: nonNegativeInteger(required(row, "original_order", endpoint, path), endpoint, `${path}.original_order`),
  };
  if ("source_label" in row) segment.sourceLabel = text(row.source_label, endpoint, `${path}.source_label`);
  if ("delivered_bytes" in row) segment.deliveredBytes = nonNegativeInteger(row.delivered_bytes, endpoint, `${path}.delivered_bytes`);
  if ("delivered_tokens" in row) segment.deliveredTokens = nonNegativeInteger(row.delivered_tokens, endpoint, `${path}.delivered_tokens`);
  if ("included" in row) segment.included = booleanValue(row.included, endpoint, `${path}.included`);
  if ("content_hash" in row) segment.contentHash = pattern(row.content_hash, HASH_16, "a lowercase 16-character content hash", endpoint, `${path}.content_hash`);
  if ("sources" in row) {
    segment.sources = array(row.sources, endpoint, `${path}.sources`).map((item, index) => (
      decodeSourceSpan(item, segmentId, endpoint, `${path}.sources[${index}]`)
    ));
    const ids = segment.sources.map((source) => source.sourceId);
    if (new Set(ids).size !== ids.length) fail(endpoint, `${path}.sources`, "must not contain duplicate source ids");
  }
  if ("reason" in row) segment.reason = oneOf(row.reason, SEGMENT_REASONS, endpoint, `${path}.reason`);
  if ("redaction_state" in row) segment.redactionState = oneOf(row.redaction_state, REDACTION_STATES, endpoint, `${path}.redaction_state`);
  return segment;
}

function decodeSegments(value: unknown, endpoint: string, path: string): readonly ReceiptSegment[] {
  const segments = array(value, endpoint, path).map((item, index) => decodeSegment(item, endpoint, `${path}[${index}]`));
  const ids = segments.map((segment) => segment.segmentId);
  const orders = segments.map((segment) => segment.originalOrder);
  if (new Set(ids).size !== ids.length) fail(endpoint, path, "must not contain duplicate segment ids");
  if (new Set(orders).size !== orders.length) fail(endpoint, path, "must not contain duplicate original_order values");
  return segments;
}

function decodeRendered(value: unknown, endpoint: string, path: string): ReceiptRendered {
  const row = object(value, endpoint, path);
  closed(row, ["sha256", "bytes", "tokens", "template_fingerprint", "estimated", "special_tokens"], endpoint, path);
  const rendered: ReceiptRendered = {};
  if ("sha256" in row) rendered.sha256 = pattern(row.sha256, SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.sha256`);
  if ("bytes" in row) rendered.bytes = nonNegativeInteger(row.bytes, endpoint, `${path}.bytes`);
  if ("tokens" in row) rendered.tokens = nonNegativeInteger(row.tokens, endpoint, `${path}.tokens`);
  if ("template_fingerprint" in row) rendered.templateFingerprint = pattern(row.template_fingerprint, HASH_16, "a lowercase 16-character fingerprint", endpoint, `${path}.template_fingerprint`);
  if ("estimated" in row) rendered.estimated = booleanValue(row.estimated, endpoint, `${path}.estimated`);
  if ("special_tokens" in row) rendered.specialTokens = array(row.special_tokens, endpoint, `${path}.special_tokens`).map((item, index) => text(item, endpoint, `${path}.special_tokens[${index}]`));
  return rendered;
}

function decodeOmission(value: unknown, endpoint: string, path: string): ReceiptOmission {
  const row = object(value, endpoint, path);
  const omission: ReceiptOmission = {
    segmentId: nonEmptyText(required(row, "segment_id", endpoint, path), endpoint, `${path}.segment_id`),
    reason: oneOf(required(row, "reason", endpoint, path), SEGMENT_REASONS, endpoint, `${path}.reason`),
  };
  if ("detail" in row) omission.detail = text(row.detail, endpoint, `${path}.detail`);
  return omission;
}

function decodeTransformation(value: unknown, endpoint: string, path: string): ReceiptTransformation {
  const row = object(value, endpoint, path);
  const transformation: ReceiptTransformation = {
    reason: oneOf(required(row, "reason", endpoint, path), SEGMENT_REASONS, endpoint, `${path}.reason`),
    segmentIds: array(required(row, "segment_ids", endpoint, path), endpoint, `${path}.segment_ids`)
      .map((item, index) => nonEmptyText(item, endpoint, `${path}.segment_ids[${index}]`)),
  };
  if ("detail" in row) transformation.detail = text(row.detail, endpoint, `${path}.detail`);
  return transformation;
}

function decodeTermination(value: unknown, endpoint: string, path: string): ReceiptTermination {
  const row = object(value, endpoint, path);
  closed(row, ["reason", "reason_raw", "source", "generated_tokens"], endpoint, path);
  const termination: ReceiptTermination = {
    reason: oneOf(required(row, "reason", endpoint, path), TERMINATION_REASONS, endpoint, `${path}.reason`),
  };
  if ("reason_raw" in row) termination.reasonRaw = text(row.reason_raw, endpoint, `${path}.reason_raw`);
  if ("source" in row) termination.source = text(row.source, endpoint, `${path}.source`);
  if ("generated_tokens" in row) termination.generatedTokens = nonNegativeInteger(row.generated_tokens, endpoint, `${path}.generated_tokens`);
  return termination;
}

function decodeV1(row: JsonRecord, endpoint: string, expectedRunId: string): ContextReceiptV1 {
  closed(row, [
    "schema_version", "run_id", "template_fingerprint", "tokenizer_conflated_with_template",
    "context_window_tokens", "reserved_output_tokens", "delivered", "assembled", "rendered",
    "omissions", "transformations", "termination", "privacy",
  ], endpoint, "$.context_receipt");
  const runId = pattern(required(row, "run_id", endpoint, "$.context_receipt"), RUN_ID, "a valid run id", endpoint, "$.context_receipt.run_id");
  if (runId !== expectedRunId) fail(endpoint, "$.context_receipt.run_id", "does not match the response run_id");
  const receipt: ContextReceiptV1 = {
    kind: "v1",
    schemaVersion: "clozn.context-receipt.v1",
    runId,
  };
  if ("template_fingerprint" in row) receipt.templateFingerprint = pattern(row.template_fingerprint, HASH_16, "a lowercase 16-character fingerprint", endpoint, "$.context_receipt.template_fingerprint");
  if ("tokenizer_conflated_with_template" in row) receipt.tokenizerConflatedWithTemplate = booleanValue(row.tokenizer_conflated_with_template, endpoint, "$.context_receipt.tokenizer_conflated_with_template");
  if ("context_window_tokens" in row) receipt.contextWindowTokens = nonNegativeInteger(row.context_window_tokens, endpoint, "$.context_receipt.context_window_tokens");
  if ("reserved_output_tokens" in row) receipt.reservedOutputTokens = nonNegativeInteger(row.reserved_output_tokens, endpoint, "$.context_receipt.reserved_output_tokens");
  if ("delivered" in row) receipt.delivered = decodeSegments(row.delivered, endpoint, "$.context_receipt.delivered");
  if ("assembled" in row) receipt.assembled = decodeSegments(row.assembled, endpoint, "$.context_receipt.assembled");
  if ("rendered" in row) receipt.rendered = decodeRendered(row.rendered, endpoint, "$.context_receipt.rendered");
  if ("omissions" in row) receipt.omissions = array(row.omissions, endpoint, "$.context_receipt.omissions").map((item, index) => decodeOmission(item, endpoint, `$.context_receipt.omissions[${index}]`));
  if ("transformations" in row) receipt.transformations = array(row.transformations, endpoint, "$.context_receipt.transformations").map((item, index) => decodeTransformation(item, endpoint, `$.context_receipt.transformations[${index}]`));
  if ("termination" in row) receipt.termination = decodeTermination(row.termination, endpoint, "$.context_receipt.termination");
  if ("privacy" in row) receipt.privacy = oneOf(row.privacy, PRIVACY_TIERS, endpoint, "$.context_receipt.privacy");
  return receipt;
}

/** Decode the exact endpoint envelope and its closed v1 receipt body. */
export function decodeContextReceipt(
  value: unknown,
  endpoint = "/runs/<id>/context-receipt",
  expectedRunId?: string,
): ContextReceiptView {
  const row = object(value, endpoint, "$");
  const shape = oneOf(required(row, "shape", endpoint, "$"), ["recorded", "absent", "unrecognized"] as const, endpoint, "$.shape");
  const runId = nonEmptyText(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (expectedRunId !== undefined && runId !== expectedRunId) fail(endpoint, "$.run_id", "does not match the request");
  const rawReceipt = object(required(row, "context_receipt", endpoint, "$"), endpoint, "$.context_receipt");

  if (shape === "absent") {
    closed(row, ["run_id", "shape", "context_receipt", "note"], endpoint, "$");
    if (Object.keys(rawReceipt).length) fail(endpoint, "$.context_receipt", "must be empty when shape is absent");
    return {
      shape,
      runId,
      note: nonEmptyText(required(row, "note", endpoint, "$"), endpoint, "$.note"),
    };
  }

  closed(row, ["run_id", "shape", "context_receipt"], endpoint, "$");
  if (!Object.keys(rawReceipt).length) fail(endpoint, "$.context_receipt", `must not be empty when shape is ${shape}`);

  if (shape === "unrecognized") {
    const schemaValue = rawReceipt.schema_version ?? rawReceipt.schema;
    return {
      shape,
      runId,
      ...(schemaValue === undefined ? {} : { schemaVersionRaw: text(schemaValue, endpoint, "$.context_receipt schema field") }),
      rawKeys: Object.keys(rawReceipt),
    };
  }

  const schemaVersion = nonEmptyText(required(rawReceipt, "schema_version", endpoint, "$.context_receipt"), endpoint, "$.context_receipt.schema_version");
  if (!schemaVersion.startsWith("clozn.context-receipt.")) {
    fail(endpoint, "$.context_receipt.schema_version", "must identify a clozn context receipt");
  }
  if (schemaVersion !== "clozn.context-receipt.v1") {
    const internalRunId = nonEmptyText(required(rawReceipt, "run_id", endpoint, "$.context_receipt"), endpoint, "$.context_receipt.run_id");
    if (internalRunId !== runId) fail(endpoint, "$.context_receipt.run_id", "does not match the response run_id");
    return {
      shape,
      runId,
      receipt: { kind: "unsupported", schemaVersion, runId: internalRunId, rawKeys: Object.keys(rawReceipt) },
    };
  }
  return { shape, runId, receipt: decodeV1(rawReceipt, endpoint, runId) };
}
