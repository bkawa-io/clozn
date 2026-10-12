import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type InvestigationState =
  | "supported"
  | "measured_effect"
  | "below_measurement_floor"
  | "delivered_not_measured"
  | "omitted"
  | "unavailable"
  | "failed"
  | "inconclusive";

export type SpanResolutionState = "metadata_only" | "drifted" | "redacted" | "unavailable";

export interface InvestigationArtifact {
  schema: string;
  method: string;
  nativeStatus?: string;
}

export type InvestigationActionKind = "measurement" | "navigation" | "corrective";
export type InvestigationActionMethod = "GET" | "POST";
export type InvestigationActionAvailability = "ready" | "conditional" | "unavailable";

export interface InvestigationAction {
  id: string;
  label: string;
  kind: InvestigationActionKind;
  method: InvestigationActionMethod;
  href: string;
  availability: InvestigationActionAvailability;
  reason?: string;
  requestBody?: JsonRecord;
  requires?: readonly string[];
}

export interface UnavailableMeasurement {
  id: string;
  state: "delivered_not_measured" | "unavailable" | "failed" | "inconclusive";
  reason: string;
  actionId?: string;
}

export interface ReceivedSegment {
  segmentId?: string;
  clientSourceId?: string;
  sourceType?: string;
  sourceLabel?: string;
  originalOrder?: number;
  deliveredBytes?: number;
  deliveredTokens?: number;
  included?: boolean;
  contentHash?: string;
  reason?: string;
  detail?: string;
  redactionState?: string;
}

export interface ReceivedContextSection {
  state: InvestigationState;
  artifact?: InvestigationArtifact;
  reason?: string;
  privacy?: string;
  deliveredCount?: number;
  delivered?: readonly ReceivedSegment[];
  assembled?: readonly ReceivedSegment[];
  omitted?: readonly ReceivedSegment[];
  rendered?: {
    sha256?: string;
    bytes?: number;
    tokens?: number;
    estimated?: boolean;
  };
  contextWindowTokens?: number | null;
  reservedOutputTokens?: number | null;
  termination?: { reason?: string; reasonRaw?: string; source?: string; generatedTokens?: number };
}

export interface SpanSectionSummary {
  state: InvestigationState;
  artifact?: InvestigationArtifact;
  reason?: string;
  href?: string;
  privacy?: string;
  addressCount?: number;
  resolutionCounts?: Readonly<Partial<Record<SpanResolutionState | "exact", number>>>;
  influenceNativeStatus?: string;
}

export interface RunInvestigation {
  schemaVersion: "clozn.run-investigation.v1";
  runId: string;
  receivedContext: ReceivedContextSection;
  spanSection?: SpanSectionSummary;
  actions?: readonly InvestigationAction[];
  unavailableMeasurements?: readonly UnavailableMeasurement[];
}

export type SpanKind =
  | "delivered_message"
  | "rendered_prompt_segment"
  | "attached_source_span"
  | "answer_span"
  | "claim";

export type NativeCollection =
  | "context_receipt.delivered"
  | "context_receipt.rendered"
  | "run.messages"
  | "influence.prompt_sources"
  | "influence.prompt_spans"
  | "influence.answer_spans"
  | "derived.claims";

export interface RecordedHash {
  algorithm: "sha256" | "sha256_truncated_64bit";
  value: string;
  scope: "canonical_basis" | "span";
}

export interface SpanAddress {
  addressId: string;
  runId: string;
  kind: SpanKind;
  relationKey: string;
  nativeRef: {
    artifactSchema: string;
    collection: NativeCollection;
    id: string;
    parentId?: string;
    segmentId?: string;
    clientSourceId?: string;
    sourceLabel?: string;
    selected?: boolean;
    recordedHash?: RecordedHash;
  };
  resolution: {
    state: SpanResolutionState;
    reason?: string;
    canonical?: {
      basis: "delivered_message" | "rendered_prompt" | "attached_source" | "scored_answer" | "recorded_answer";
      start: number;
      end: number;
      basisSha256: string;
      spanSha256: string;
      basisCodePoints?: number;
      spanCodePoints?: number;
      basisUtf8Bytes?: number;
      spanUtf8Bytes?: number;
    };
  };
}

export interface SpanSourceArtifact {
  schema: string;
  nativeStatus?: string;
  available?: boolean;
  privacy?: string;
  reason?: string;
  artifactSha256?: string;
  method?: JsonRecord;
}

export interface SpanLineageMapping {
  relationKey: string;
  parentAddressId: string;
  childAddressId?: string;
  state: "inherited" | "drifted" | "unavailable";
  reason:
    | "exact_text_and_hashes_unchanged"
    | "canonical_basis_hash_changed"
    | "span_text_hash_changed"
    | "offset_changed"
    | "kind_changed"
    | "parent_unresolved"
    | "child_unresolved"
    | "missing_in_child"
    | "ambiguous_relation";
}

export interface SpanAddressDocument {
  schemaVersion: "clozn.text-span-addresses.v1";
  runId: string;
  privacy: "metadata_only";
  sourceArtifacts: readonly SpanSourceArtifact[];
  addresses: readonly SpanAddress[];
  parentRunId: string | null;
  lineageMappings?: readonly SpanLineageMapping[];
}

const INVESTIGATION_STATES = [
  "supported", "measured_effect", "below_measurement_floor", "delivered_not_measured",
  "omitted", "unavailable", "failed", "inconclusive",
] as const;
const RESOLUTION_STATES = ["metadata_only", "drifted", "redacted", "unavailable"] as const;
const SPAN_KINDS = ["delivered_message", "rendered_prompt_segment", "attached_source_span", "answer_span", "claim"] as const;
const COLLECTIONS = [
  "context_receipt.delivered", "context_receipt.rendered", "run.messages",
  "influence.prompt_sources", "influence.prompt_spans", "influence.answer_spans", "derived.claims",
] as const;
const BASES = ["delivered_message", "rendered_prompt", "attached_source", "scored_answer", "recorded_answer"] as const;
const REASONS = [
  "native_content_hash_mismatch", "native_rendered_hash_mismatch", "basis_hash_mismatch",
  "span_text_hash_mismatch", "offset_out_of_bounds", "invalid_offsets", "invalid_text_encoding",
  "source_text_redacted", "source_text_unavailable", "canonical_basis_unavailable", "offsets_unavailable",
] as const;
const ACTION_KINDS = ["measurement", "navigation", "corrective"] as const;
const ACTION_METHODS = ["GET", "POST"] as const;
const ACTION_AVAILABILITIES = ["ready", "conditional", "unavailable"] as const;
const UNAVAILABLE_STATES = ["delivered_not_measured", "unavailable", "failed", "inconclusive"] as const;
const LINEAGE_STATES = ["inherited", "drifted", "unavailable"] as const;
const LINEAGE_REASONS = [
  "exact_text_and_hashes_unchanged", "canonical_basis_hash_changed", "span_text_hash_changed",
  "offset_changed", "kind_changed", "parent_unresolved", "child_unresolved", "missing_in_child",
  "ambiguous_relation",
] as const;
const ADDRESS_ID = /^span_[0-9a-f]{24}$/;
const RELATION_KEY = /^rel_[0-9a-f]{24}$/;
const SHA256 = /^[0-9a-f]{64}$/;

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
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) fail(endpoint, path, "must be a non-negative integer");
  return value;
}

function nullableNonNegativeInteger(value: unknown, endpoint: string, path: string): number | null {
  return value === null ? null : nonNegativeInteger(value, endpoint, path);
}

function enumValue<const T extends readonly string[]>(value: unknown, values: T, endpoint: string, path: string): T[number] {
  if (typeof value !== "string" || !values.includes(value)) fail(endpoint, path, `must be one of ${values.join(", ")}`);
  return value as T[number];
}

function patterned(value: unknown, pattern: RegExp, label: string, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!pattern.test(parsed)) fail(endpoint, path, `must be ${label}`);
  return parsed;
}

function optionalString(row: JsonRecord, key: string, endpoint: string, path: string): string | undefined {
  return key in row ? text(row[key], endpoint, `${path}.${key}`) : undefined;
}

function optionalNonEmptyText(row: JsonRecord, key: string, endpoint: string, path: string): string | undefined {
  return key in row ? nonEmptyText(row[key], endpoint, `${path}.${key}`) : undefined;
}

function optionalNonNegativeInteger(row: JsonRecord, key: string, endpoint: string, path: string): number | undefined {
  return key in row ? nonNegativeInteger(row[key], endpoint, `${path}.${key}`) : undefined;
}

function decodeArtifact(value: unknown, endpoint: string, path: string): InvestigationArtifact {
  const row = object(value, endpoint, path);
  closed(row, ["schema", "method", "native_status"], endpoint, path);
  return {
    schema: nonEmptyText(required(row, "schema", endpoint, path), endpoint, `${path}.schema`),
    method: nonEmptyText(required(row, "method", endpoint, path), endpoint, `${path}.method`),
    nativeStatus: optionalNonEmptyText(row, "native_status", endpoint, path),
  };
}

function decodeSegment(value: unknown, endpoint: string, path: string): ReceivedSegment {
  const row = object(value, endpoint, path);
  const segment: ReceivedSegment = {};
  if ("segment_id" in row) segment.segmentId = nonEmptyText(row.segment_id, endpoint, `${path}.segment_id`);
  if ("client_source_id" in row) segment.clientSourceId = nonEmptyText(row.client_source_id, endpoint, `${path}.client_source_id`);
  if ("source_type" in row) segment.sourceType = nonEmptyText(row.source_type, endpoint, `${path}.source_type`);
  if ("source_label" in row) segment.sourceLabel = text(row.source_label, endpoint, `${path}.source_label`);
  if ("original_order" in row) segment.originalOrder = nonNegativeInteger(row.original_order, endpoint, `${path}.original_order`);
  if ("delivered_bytes" in row) segment.deliveredBytes = nonNegativeInteger(row.delivered_bytes, endpoint, `${path}.delivered_bytes`);
  if ("delivered_tokens" in row) segment.deliveredTokens = nonNegativeInteger(row.delivered_tokens, endpoint, `${path}.delivered_tokens`);
  if ("included" in row) segment.included = booleanValue(row.included, endpoint, `${path}.included`);
  if ("content_hash" in row) segment.contentHash = patterned(row.content_hash, /^[0-9a-f]{16}$/, "a 16-character lowercase hash", endpoint, `${path}.content_hash`);
  if ("reason" in row) segment.reason = nonEmptyText(row.reason, endpoint, `${path}.reason`);
  if ("detail" in row) segment.detail = nonEmptyText(row.detail, endpoint, `${path}.detail`);
  if ("redaction_state" in row) segment.redactionState = enumValue(row.redaction_state, ["full", "redacted", "hash_only"] as const, endpoint, `${path}.redaction_state`);
  if (!segment.segmentId && !segment.clientSourceId) fail(endpoint, path, "must carry a segment or client source id");
  return segment;
}

function decodeSegments(value: unknown, endpoint: string, path: string): readonly ReceivedSegment[] {
  return array(value, endpoint, path).map((item, index) => decodeSegment(item, endpoint, `${path}[${index}]`));
}

function decodeReceivedContext(value: unknown, endpoint: string, path: string): ReceivedContextSection {
  const row = object(value, endpoint, path);
  const section: ReceivedContextSection = {
    state: enumValue(required(row, "state", endpoint, path), INVESTIGATION_STATES, endpoint, `${path}.state`),
  };
  if ("artifact" in row) section.artifact = decodeArtifact(row.artifact, endpoint, `${path}.artifact`);
  if ("reason" in row) section.reason = nonEmptyText(row.reason, endpoint, `${path}.reason`);
  if ("privacy" in row) section.privacy = nonEmptyText(row.privacy, endpoint, `${path}.privacy`);
  if ("delivered_count" in row) section.deliveredCount = nonNegativeInteger(row.delivered_count, endpoint, `${path}.delivered_count`);
  if ("delivered" in row) section.delivered = decodeSegments(row.delivered, endpoint, `${path}.delivered`);
  if ("assembled" in row) section.assembled = decodeSegments(row.assembled, endpoint, `${path}.assembled`);
  if ("omitted" in row) section.omitted = decodeSegments(row.omitted, endpoint, `${path}.omitted`);
  if ("rendered" in row) {
    const rendered = object(row.rendered, endpoint, `${path}.rendered`);
    section.rendered = {
      ...(rendered.sha256 === undefined ? {} : { sha256: patterned(rendered.sha256, SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.rendered.sha256`) }),
      ...(rendered.bytes === undefined ? {} : { bytes: nonNegativeInteger(rendered.bytes, endpoint, `${path}.rendered.bytes`) }),
      ...(rendered.tokens === undefined ? {} : { tokens: nonNegativeInteger(rendered.tokens, endpoint, `${path}.rendered.tokens`) }),
      ...(rendered.estimated === undefined ? {} : { estimated: booleanValue(rendered.estimated, endpoint, `${path}.rendered.estimated`) }),
    };
  }
  if ("context_window_tokens" in row) section.contextWindowTokens = nullableNonNegativeInteger(row.context_window_tokens, endpoint, `${path}.context_window_tokens`);
  if ("reserved_output_tokens" in row) section.reservedOutputTokens = nullableNonNegativeInteger(row.reserved_output_tokens, endpoint, `${path}.reserved_output_tokens`);
  if ("termination" in row) {
    const termination = object(row.termination, endpoint, `${path}.termination`);
    section.termination = {
      reason: optionalString(termination, "reason", endpoint, `${path}.termination`),
      reasonRaw: optionalString(termination, "reason_raw", endpoint, `${path}.termination`),
      source: optionalString(termination, "source", endpoint, `${path}.termination`),
      generatedTokens: optionalNonNegativeInteger(termination, "generated_tokens", endpoint, `${path}.termination`),
    };
  }
  return section;
}

function decodeGenericSection(value: unknown, endpoint: string, path: string): InvestigationState {
  const row = object(value, endpoint, path);
  const state = enumValue(required(row, "state", endpoint, path), INVESTIGATION_STATES, endpoint, `${path}.state`);
  if ("artifact" in row) decodeArtifact(row.artifact, endpoint, `${path}.artifact`);
  if ("reason" in row) nonEmptyText(row.reason, endpoint, `${path}.reason`);
  if ("action_id" in row) nonEmptyText(row.action_id, endpoint, `${path}.action_id`);
  return state;
}

function decodeSpanSummary(value: unknown, endpoint: string, path: string): SpanSectionSummary {
  const row = object(value, endpoint, path);
  const summary: SpanSectionSummary = {
    state: enumValue(required(row, "state", endpoint, path), INVESTIGATION_STATES, endpoint, `${path}.state`),
  };
  if ("artifact" in row) summary.artifact = decodeArtifact(row.artifact, endpoint, `${path}.artifact`);
  if ("reason" in row) summary.reason = nonEmptyText(row.reason, endpoint, `${path}.reason`);
  if ("href" in row) summary.href = nonEmptyText(row.href, endpoint, `${path}.href`);
  if ("privacy" in row) summary.privacy = nonEmptyText(row.privacy, endpoint, `${path}.privacy`);
  if ("address_count" in row) summary.addressCount = nonNegativeInteger(row.address_count, endpoint, `${path}.address_count`);
  if ("influence_native_status" in row) summary.influenceNativeStatus = nonEmptyText(row.influence_native_status, endpoint, `${path}.influence_native_status`);
  if ("resolution_counts" in row) {
    const counts = object(row.resolution_counts, endpoint, `${path}.resolution_counts`);
    const decoded: Partial<Record<SpanResolutionState | "exact", number>> = {};
    for (const [key, count] of Object.entries(counts)) {
      if (!["exact", ...RESOLUTION_STATES].includes(key as SpanResolutionState)) fail(endpoint, `${path}.resolution_counts.${key}`, "is not a known resolution state");
      decoded[key as SpanResolutionState | "exact"] = nonNegativeInteger(count, endpoint, `${path}.resolution_counts.${key}`);
    }
    summary.resolutionCounts = decoded;
  }
  return summary;
}

function decodeAction(value: unknown, endpoint: string, path: string): InvestigationAction {
  const row = object(value, endpoint, path);
  closed(row, [
    "id", "label", "kind", "method", "href", "availability", "reason", "request_body", "requires",
  ], endpoint, path);
  const requires = row.requires === undefined
    ? undefined
    : array(row.requires, endpoint, `${path}.requires`).map((item, index) => (
      nonEmptyText(item, endpoint, `${path}.requires[${index}]`)
    ));
  return {
    id: nonEmptyText(required(row, "id", endpoint, path), endpoint, `${path}.id`),
    label: nonEmptyText(required(row, "label", endpoint, path), endpoint, `${path}.label`),
    kind: enumValue(required(row, "kind", endpoint, path), ACTION_KINDS, endpoint, `${path}.kind`),
    method: enumValue(required(row, "method", endpoint, path), ACTION_METHODS, endpoint, `${path}.method`),
    href: nonEmptyText(required(row, "href", endpoint, path), endpoint, `${path}.href`),
    availability: enumValue(
      required(row, "availability", endpoint, path), ACTION_AVAILABILITIES, endpoint, `${path}.availability`,
    ),
    reason: optionalNonEmptyText(row, "reason", endpoint, path),
    ...(row.request_body === undefined
      ? {}
      : { requestBody: object(row.request_body, endpoint, `${path}.request_body`) }),
    ...(requires === undefined ? {} : { requires }),
  };
}

function decodeUnavailableMeasurement(
  value: unknown,
  endpoint: string,
  path: string,
): UnavailableMeasurement {
  const row = object(value, endpoint, path);
  closed(row, ["id", "state", "reason", "action_id"], endpoint, path);
  return {
    id: nonEmptyText(required(row, "id", endpoint, path), endpoint, `${path}.id`),
    state: enumValue(required(row, "state", endpoint, path), UNAVAILABLE_STATES, endpoint, `${path}.state`),
    reason: nonEmptyText(required(row, "reason", endpoint, path), endpoint, `${path}.reason`),
    actionId: optionalNonEmptyText(row, "action_id", endpoint, path),
  };
}

export function decodeRunInvestigation(
  value: unknown,
  endpoint = "/runs/<id>/investigation",
  expectedRunId?: string,
): RunInvestigation {
  const row = object(value, endpoint, "$");
  closed(row, ["schema_version", "run_id", "sections", "actions", "unavailable_measurements"], endpoint, "$");
  const schemaVersion = enumValue(required(row, "schema_version", endpoint, "$"), ["clozn.run-investigation.v1"] as const, endpoint, "$.schema_version");
  const runId = nonEmptyText(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (expectedRunId !== undefined && runId !== expectedRunId) fail(endpoint, "$.run_id", "does not match the request");
  const sections = object(required(row, "sections", endpoint, "$"), endpoint, "$.sections");
  closed(sections, [
    "received_context", "prompt_source_influence", "answer_support", "text_span_addresses",
    "diagnosis", "comparisons", "corrective_actions",
  ], endpoint, "$.sections");
  const requiredSections = ["received_context", "prompt_source_influence", "answer_support", "diagnosis", "comparisons", "corrective_actions"];
  for (const key of requiredSections) required(sections, key, endpoint, "$.sections");
  const receivedContext = decodeReceivedContext(sections.received_context, endpoint, "$.sections.received_context");
  for (const key of requiredSections.slice(1)) decodeGenericSection(sections[key], endpoint, `$.sections.${key}`);
  const actions = array(required(row, "actions", endpoint, "$"), endpoint, "$.actions")
    .map((item, index) => decodeAction(item, endpoint, `$.actions[${index}]`));
  const unavailableMeasurements = array(
    required(row, "unavailable_measurements", endpoint, "$"), endpoint, "$.unavailable_measurements",
  ).map((item, index) => decodeUnavailableMeasurement(item, endpoint, `$.unavailable_measurements[${index}]`));
  return {
    schemaVersion,
    runId,
    receivedContext,
    ...(sections.text_span_addresses === undefined ? {} : {
      spanSection: decodeSpanSummary(sections.text_span_addresses, endpoint, "$.sections.text_span_addresses"),
    }),
    actions,
    unavailableMeasurements,
  };
}

function decodeCanonical(value: unknown, endpoint: string, path: string): NonNullable<SpanAddress["resolution"]["canonical"]> {
  const row = object(value, endpoint, path);
  closed(row, [
    "basis", "unit", "interval", "start", "end", "basis_sha256", "span_sha256",
    "basis_code_points", "span_code_points", "basis_utf8_bytes", "span_utf8_bytes",
  ], endpoint, path);
  const start = nonNegativeInteger(required(row, "start", endpoint, path), endpoint, `${path}.start`);
  const end = nonNegativeInteger(required(row, "end", endpoint, path), endpoint, `${path}.end`);
  if (end < start) fail(endpoint, path, "must describe an ordered half-open range");
  if (required(row, "unit", endpoint, path) !== "unicode_code_points" || required(row, "interval", endpoint, path) !== "half_open") {
    fail(endpoint, path, "must use Unicode code-point half-open coordinates");
  }
  return {
    basis: enumValue(required(row, "basis", endpoint, path), BASES, endpoint, `${path}.basis`),
    start,
    end,
    basisSha256: patterned(required(row, "basis_sha256", endpoint, path), SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.basis_sha256`),
    spanSha256: patterned(required(row, "span_sha256", endpoint, path), SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.span_sha256`),
    basisCodePoints: optionalNonNegativeInteger(row, "basis_code_points", endpoint, path),
    spanCodePoints: optionalNonNegativeInteger(row, "span_code_points", endpoint, path),
    basisUtf8Bytes: optionalNonNegativeInteger(row, "basis_utf8_bytes", endpoint, path),
    spanUtf8Bytes: optionalNonNegativeInteger(row, "span_utf8_bytes", endpoint, path),
  };
}

function decodeRecordedHash(value: unknown, endpoint: string, path: string): RecordedHash {
  const row = object(value, endpoint, path);
  closed(row, ["algorithm", "value", "scope"], endpoint, path);
  const algorithm = enumValue(
    required(row, "algorithm", endpoint, path),
    ["sha256", "sha256_truncated_64bit"] as const,
    endpoint,
    `${path}.algorithm`,
  );
  return {
    algorithm,
    value: patterned(
      required(row, "value", endpoint, path),
      algorithm === "sha256" ? SHA256 : /^[0-9a-f]{16}$/,
      algorithm === "sha256" ? "a lowercase SHA-256 digest" : "a 16-character lowercase hash",
      endpoint,
      `${path}.value`,
    ),
    scope: enumValue(
      required(row, "scope", endpoint, path), ["canonical_basis", "span"] as const, endpoint, `${path}.scope`,
    ),
  };
}

/**
 * Decode the one shared text-span address shape used by every Studio evidence reader.
 * Artifact-specific readers must add their own kind/native-reference invariants after this structural decode.
 */
export function decodeSpanAddress(
  value: unknown,
  expectedRunId: string,
  endpoint = "/runs/<id>/span-addresses",
  path = "$.address",
): SpanAddress {
  const row = object(value, endpoint, path);
  closed(row, ["address_id", "run_id", "kind", "relation_key", "native_ref", "resolution"], endpoint, path);
  const runId = nonEmptyText(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`);
  if (runId !== expectedRunId) fail(endpoint, `${path}.run_id`, "does not match the document run_id");
  const native = object(required(row, "native_ref", endpoint, path), endpoint, `${path}.native_ref`);
  closed(native, ["artifact_schema", "collection", "id", "parent_id", "segment_id", "client_source_id", "source_label", "selected", "recorded_hash"], endpoint, `${path}.native_ref`);
  const resolution = object(required(row, "resolution", endpoint, path), endpoint, `${path}.resolution`);
  closed(resolution, ["state", "canonical", "reason"], endpoint, `${path}.resolution`);
  const state = enumValue(required(resolution, "state", endpoint, `${path}.resolution`), RESOLUTION_STATES, endpoint, `${path}.resolution.state`);
  const reason = resolution.reason === undefined ? undefined : enumValue(resolution.reason, REASONS, endpoint, `${path}.resolution.reason`);
  if (state === "metadata_only" && reason !== undefined) fail(endpoint, `${path}.resolution.reason`, "must be absent for metadata_only resolution");
  if (state !== "metadata_only" && !reason) fail(endpoint, `${path}.resolution.reason`, `is required for ${state} resolution`);
  const canonical = resolution.canonical === undefined ? undefined : decodeCanonical(resolution.canonical, endpoint, `${path}.resolution.canonical`);
  if ((state === "metadata_only" || state === "drifted") && !canonical) fail(endpoint, `${path}.resolution.canonical`, `is required for ${state} resolution`);
  if ((state === "redacted" || state === "unavailable") && canonical) fail(endpoint, `${path}.resolution.canonical`, `must be absent for ${state} resolution`);
  return {
    addressId: patterned(required(row, "address_id", endpoint, path), ADDRESS_ID, "a span address id", endpoint, `${path}.address_id`),
    runId,
    kind: enumValue(required(row, "kind", endpoint, path), SPAN_KINDS, endpoint, `${path}.kind`),
    relationKey: patterned(required(row, "relation_key", endpoint, path), RELATION_KEY, "a relation key", endpoint, `${path}.relation_key`),
    nativeRef: {
      artifactSchema: nonEmptyText(required(native, "artifact_schema", endpoint, `${path}.native_ref`), endpoint, `${path}.native_ref.artifact_schema`),
      collection: enumValue(required(native, "collection", endpoint, `${path}.native_ref`), COLLECTIONS, endpoint, `${path}.native_ref.collection`),
      id: nonEmptyText(required(native, "id", endpoint, `${path}.native_ref`), endpoint, `${path}.native_ref.id`),
      parentId: optionalNonEmptyText(native, "parent_id", endpoint, `${path}.native_ref`),
      segmentId: optionalNonEmptyText(native, "segment_id", endpoint, `${path}.native_ref`),
      clientSourceId: optionalNonEmptyText(native, "client_source_id", endpoint, `${path}.native_ref`),
      sourceLabel: optionalString(native, "source_label", endpoint, `${path}.native_ref`),
      ...(native.selected === undefined ? {} : { selected: booleanValue(native.selected, endpoint, `${path}.native_ref.selected`) }),
      ...(native.recorded_hash === undefined
        ? {}
        : { recordedHash: decodeRecordedHash(native.recorded_hash, endpoint, `${path}.native_ref.recorded_hash`) }),
    },
    resolution: { state, ...(reason ? { reason } : {}), ...(canonical ? { canonical } : {}) },
  };
}

export function decodeSpanAddressDocument(
  value: unknown,
  endpoint = "/runs/<id>/span-addresses",
  expectedRunId?: string,
): SpanAddressDocument {
  const row = object(value, endpoint, "$");
  closed(row, ["schema_version", "run_id", "privacy", "offset_contract", "source_artifacts", "addresses", "lineage"], endpoint, "$");
  const schemaVersion = enumValue(required(row, "schema_version", endpoint, "$"), ["clozn.text-span-addresses.v1"] as const, endpoint, "$.schema_version");
  const runId = nonEmptyText(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (expectedRunId !== undefined && runId !== expectedRunId) fail(endpoint, "$.run_id", "does not match the request");
  if (required(row, "privacy", endpoint, "$") !== "metadata_only") fail(endpoint, "$.privacy", "must be metadata_only for this reader");
  const offsets = object(required(row, "offset_contract", endpoint, "$"), endpoint, "$.offset_contract");
  closed(offsets, ["unit", "interval", "hash_algorithm", "canonicalization"], endpoint, "$.offset_contract");
  if (
    required(offsets, "unit", endpoint, "$.offset_contract") !== "unicode_code_points"
    || required(offsets, "interval", endpoint, "$.offset_contract") !== "half_open"
    || required(offsets, "hash_algorithm", endpoint, "$.offset_contract") !== "sha256"
    || required(offsets, "canonicalization", endpoint, "$.offset_contract") !== "exact_string_utf8_v1"
  ) fail(endpoint, "$.offset_contract", "does not match the stable coordinate contract");
  const sourceArtifacts = array(required(row, "source_artifacts", endpoint, "$"), endpoint, "$.source_artifacts").map((value, index) => {
    const path = `$.source_artifacts[${index}]`;
    const artifact = object(value, endpoint, path);
    closed(artifact, ["schema", "native_status", "available", "privacy", "reason", "artifact_sha256", "method"], endpoint, path);
    return {
      schema: nonEmptyText(required(artifact, "schema", endpoint, path), endpoint, `${path}.schema`),
      nativeStatus: optionalNonEmptyText(artifact, "native_status", endpoint, path),
      ...(artifact.available === undefined ? {} : { available: booleanValue(artifact.available, endpoint, `${path}.available`) }),
      privacy: optionalNonEmptyText(artifact, "privacy", endpoint, path),
      reason: optionalNonEmptyText(artifact, "reason", endpoint, path),
      ...(artifact.artifact_sha256 === undefined ? {} : { artifactSha256: patterned(artifact.artifact_sha256, SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.artifact_sha256`) }),
      ...(artifact.method === undefined ? {} : { method: object(artifact.method, endpoint, `${path}.method`) }),
    };
  });
  const addresses = array(required(row, "addresses", endpoint, "$"), endpoint, "$.addresses")
    .map((item, index) => decodeSpanAddress(item, runId, endpoint, `$.addresses[${index}]`));
  const ids = addresses.map((address) => address.addressId);
  if (new Set(ids).size !== ids.length) fail(endpoint, "$.addresses", "must not contain duplicate address ids");
  const lineage = object(required(row, "lineage", endpoint, "$"), endpoint, "$.lineage");
  closed(lineage, ["parent_run_id", "mappings"], endpoint, "$.lineage");
  const parentRaw = required(lineage, "parent_run_id", endpoint, "$.lineage");
  const parentRunId = parentRaw === null ? null : text(parentRaw, endpoint, "$.lineage.parent_run_id");
  const lineageMappings = array(
    required(lineage, "mappings", endpoint, "$.lineage"), endpoint, "$.lineage.mappings",
  ).map((value, index): SpanLineageMapping => {
    const path = `$.lineage.mappings[${index}]`;
    const mapping = object(value, endpoint, path);
    closed(mapping, ["relation_key", "parent_address_id", "child_address_id", "state", "reason"], endpoint, path);
    return {
      relationKey: patterned(
        required(mapping, "relation_key", endpoint, path), RELATION_KEY, "a relation key", endpoint, `${path}.relation_key`,
      ),
      parentAddressId: patterned(
        required(mapping, "parent_address_id", endpoint, path), ADDRESS_ID, "a span address id", endpoint, `${path}.parent_address_id`,
      ),
      ...(mapping.child_address_id === undefined ? {} : {
        childAddressId: patterned(
          mapping.child_address_id, ADDRESS_ID, "a span address id", endpoint, `${path}.child_address_id`,
        ),
      }),
      state: enumValue(required(mapping, "state", endpoint, path), LINEAGE_STATES, endpoint, `${path}.state`),
      reason: enumValue(required(mapping, "reason", endpoint, path), LINEAGE_REASONS, endpoint, `${path}.reason`),
    };
  });
  return {
    schemaVersion, runId, privacy: "metadata_only", sourceArtifacts, addresses, parentRunId, lineageMappings,
  };
}
