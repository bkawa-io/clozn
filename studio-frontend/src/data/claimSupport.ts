import { ContractError } from "./contracts";
import { decodeSpanAddress, type SpanAddress } from "./receivedContext";

type JsonRecord = Record<string, unknown>;

export type ClaimCategory =
  | "factual_claim"
  | "recommendation"
  | "uncertainty_statement"
  | "instruction_procedure"
  | "non_verifiable_prose";

export type ClaimCategoryReason =
  | "code_fence_block"
  | "interrogative_sentence"
  | "hedge_marker"
  | "recommendation_marker"
  | "list_item_imperative"
  | "imperative_lead"
  | "factual_declarative"
  | "no_deterministic_category_match";

export type ClaimSupportStatus =
  | "supported"
  | "weakly_supported"
  | "contradicted"
  | "unsupported_by_supplied_materials"
  | "unverifiable_from_available_evidence"
  | "measurement_unavailable";

export type InfluenceGate =
  | "ok"
  | "no_influence_map"
  | "influence_measurement_unavailable"
  | "influence_measurement_error"
  | "answer_text_mismatch"
  | "no_resolvable_answer_spans";

export type AnswerSegmentation =
  | { state: "ok"; claimCount: number }
  | { state: "empty"; reason: "answer_text_empty" }
  | { state: "segmentation_limited"; reason: "unsupported_script_density" }
  | { state: "unavailable"; reason: "answer_text_redacted" | "no_answer_text" };

export interface AnswerSource {
  basis: "recorded_answer";
  basisSha256?: string;
  basisCodePoints?: number;
  basisUtf8Bytes?: number;
}

export interface AnswerClaim {
  index: number;
  category: ClaimCategory;
  categoryReason: ClaimCategoryReason;
  textSpan: SpanAddress;
}

export interface AnswerClaimsDocument {
  schemaVersion: "clozn.answer-claims.v1";
  runId: string;
  privacy: "metadata_only";
  segmentation: AnswerSegmentation;
  answerSource: AnswerSource;
  claims: readonly AnswerClaim[];
}

interface ClaimResultBase {
  claimIndex: number;
  claimAddressId: string;
}

export type ClaimSupportResult =
  | (ClaimResultBase & {
    status: "supported";
    method: { name: "forced_score_intervention"; maxAbsDeltaNats: number };
    sourceSpanIds: readonly string[];
  })
  | (ClaimResultBase & {
    status: "weakly_supported";
    method: { name: "textual_overlap"; overlapFraction: number };
    sourceSpanIds: readonly string[];
  })
  | (ClaimResultBase & {
    status: "contradicted";
    method: { name: "numeric_or_date_mismatch" | "direct_negation" };
    sourceSpanIds: readonly string[];
  })
  | (ClaimResultBase & {
    status: "unsupported_by_supplied_materials";
    method: { name: "measured_comparison_no_match" };
  })
  | (ClaimResultBase & {
    status: "unverifiable_from_available_evidence";
    method: { name: "category_rule" };
  })
  | (ClaimResultBase & {
    status: "measurement_unavailable";
    method: {
      name:
        | "no_influence_map"
        | "influence_measurement_unavailable"
        | "influence_measurement_error"
        | "answer_text_mismatch"
        | "no_resolvable_answer_spans";
    };
  });

export interface ClaimSupportResultsDocument {
  schemaVersion: "clozn.claim-support.v1";
  runId: string;
  privacy: "metadata_only";
  claimsSchemaVersion: "clozn.answer-claims.v1";
  influenceGate: InfluenceGate;
  results: readonly ClaimSupportResult[];
}

export interface ClaimSupportDocument {
  claims: AnswerClaimsDocument;
  support: ClaimSupportResultsDocument;
}

export interface RecordedRunResponse {
  runId: string;
  response: string | null;
}

export interface ClaimSupportEvidence extends ClaimSupportDocument {
  recordedResponse: string | null;
  claimTexts: readonly string[];
}

const CLAIM_CATEGORIES = [
  "factual_claim", "recommendation", "uncertainty_statement", "instruction_procedure",
  "non_verifiable_prose",
] as const;
const CLAIM_REASONS = [
  "code_fence_block", "interrogative_sentence", "hedge_marker", "recommendation_marker",
  "list_item_imperative", "imperative_lead", "factual_declarative", "no_deterministic_category_match",
] as const;
const SUPPORT_STATUSES = [
  "supported", "weakly_supported", "contradicted", "unsupported_by_supplied_materials",
  "unverifiable_from_available_evidence", "measurement_unavailable",
] as const;
const INFLUENCE_GATES = [
  "ok", "no_influence_map", "influence_measurement_unavailable", "influence_measurement_error",
  "answer_text_mismatch", "no_resolvable_answer_spans",
] as const;
const MEASUREMENT_UNAVAILABLE_METHODS = [
  "no_influence_map", "influence_measurement_unavailable", "influence_measurement_error",
  "answer_text_mismatch", "no_resolvable_answer_spans",
] as const;
const ADDRESS_ID = /^span_[0-9a-f]{24}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

const CATEGORY_FOR_REASON: Readonly<Record<ClaimCategoryReason, ClaimCategory>> = {
  code_fence_block: "non_verifiable_prose",
  interrogative_sentence: "non_verifiable_prose",
  hedge_marker: "uncertainty_statement",
  recommendation_marker: "recommendation",
  list_item_imperative: "instruction_procedure",
  imperative_lead: "instruction_procedure",
  factual_declarative: "factual_claim",
  no_deterministic_category_match: "non_verifiable_prose",
};

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

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(endpoint, path, "must be a non-negative integer");
  }
  return value;
}

function nonNegativeNumber(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(endpoint, path, "must be a finite non-negative number");
  }
  return value;
}

function patterned(value: unknown, pattern: RegExp, label: string, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!pattern.test(parsed)) fail(endpoint, path, `must be ${label}`);
  return parsed;
}

function decodeOffsetContract(value: unknown, endpoint: string, path: string): void {
  const row = object(value, endpoint, path);
  closed(row, ["unit", "interval", "hash_algorithm", "canonicalization"], endpoint, path);
  if (
    required(row, "unit", endpoint, path) !== "unicode_code_points"
    || required(row, "interval", endpoint, path) !== "half_open"
    || required(row, "hash_algorithm", endpoint, path) !== "sha256"
    || required(row, "canonicalization", endpoint, path) !== "exact_string_utf8_v1"
  ) {
    fail(endpoint, path, "does not match the stable Unicode coordinate contract");
  }
}

function decodeSegmentation(value: unknown, endpoint: string, path: string): AnswerSegmentation {
  const row = object(value, endpoint, path);
  const state = oneOf(
    required(row, "state", endpoint, path),
    ["ok", "empty", "segmentation_limited", "unavailable"] as const,
    endpoint,
    `${path}.state`,
  );
  if (state === "ok") {
    closed(row, ["state", "claim_count"], endpoint, path);
    return { state, claimCount: nonNegativeInteger(required(row, "claim_count", endpoint, path), endpoint, `${path}.claim_count`) };
  }
  closed(row, ["state", "reason"], endpoint, path);
  const reason = nonEmptyText(required(row, "reason", endpoint, path), endpoint, `${path}.reason`);
  if (state === "empty" && reason === "answer_text_empty") return { state, reason };
  if (state === "segmentation_limited" && reason === "unsupported_script_density") return { state, reason };
  if (state === "unavailable" && (reason === "answer_text_redacted" || reason === "no_answer_text")) {
    return { state, reason };
  }
  fail(endpoint, `${path}.reason`, `is not valid for segmentation state ${state}`);
}

function decodeAnswerSource(
  value: unknown,
  segmentation: AnswerSegmentation,
  endpoint: string,
  path: string,
): AnswerSource {
  const row = object(value, endpoint, path);
  if (segmentation.state === "unavailable") {
    closed(row, ["basis"], endpoint, path);
    if (required(row, "basis", endpoint, path) !== "recorded_answer") {
      fail(endpoint, `${path}.basis`, "must be recorded_answer");
    }
    return { basis: "recorded_answer" };
  }
  closed(row, ["basis", "basis_sha256", "basis_code_points", "basis_utf8_bytes"], endpoint, path);
  if (required(row, "basis", endpoint, path) !== "recorded_answer") {
    fail(endpoint, `${path}.basis`, "must be recorded_answer");
  }
  const source: AnswerSource = {
    basis: "recorded_answer",
    basisSha256: patterned(
      required(row, "basis_sha256", endpoint, path), SHA256, "a lowercase SHA-256 digest", endpoint, `${path}.basis_sha256`,
    ),
    basisCodePoints: nonNegativeInteger(
      required(row, "basis_code_points", endpoint, path), endpoint, `${path}.basis_code_points`,
    ),
    basisUtf8Bytes: nonNegativeInteger(
      required(row, "basis_utf8_bytes", endpoint, path), endpoint, `${path}.basis_utf8_bytes`,
    ),
  };
  if (segmentation.state === "empty" && (
    source.basisSha256 !== EMPTY_SHA256 || source.basisCodePoints !== 0 || source.basisUtf8Bytes !== 0
  )) {
    fail(endpoint, path, "must carry the exact empty recorded-answer identity");
  }
  if (segmentation.state !== "empty" && source.basisCodePoints === 0) {
    fail(endpoint, `${path}.basis_code_points`, `must be positive for ${segmentation.state} segmentation`);
  }
  if ((source.basisUtf8Bytes ?? -1) < (source.basisCodePoints ?? 0)) {
    fail(endpoint, `${path}.basis_utf8_bytes`, "cannot be smaller than the Unicode code-point count");
  }
  return source;
}

function decodeClaim(
  value: unknown,
  index: number,
  runId: string,
  answerSource: AnswerSource,
  endpoint: string,
  path: string,
): AnswerClaim {
  const row = object(value, endpoint, path);
  closed(row, ["index", "category", "category_reason", "text_span"], endpoint, path);
  const decodedIndex = nonNegativeInteger(required(row, "index", endpoint, path), endpoint, `${path}.index`);
  if (decodedIndex !== index) fail(endpoint, `${path}.index`, "must equal its zero-based position");
  const category = oneOf(required(row, "category", endpoint, path), CLAIM_CATEGORIES, endpoint, `${path}.category`);
  const categoryReason = oneOf(
    required(row, "category_reason", endpoint, path), CLAIM_REASONS, endpoint, `${path}.category_reason`,
  );
  if (CATEGORY_FOR_REASON[categoryReason] !== category) {
    fail(endpoint, `${path}.category_reason`, `does not produce category ${category}`);
  }
  const textSpan = decodeSpanAddress(required(row, "text_span", endpoint, path), runId, endpoint, `${path}.text_span`);
  if (textSpan.kind !== "claim") fail(endpoint, `${path}.text_span.kind`, "must be claim");
  if (textSpan.nativeRef.artifactSchema !== "clozn.answer-claims.v1") {
    fail(endpoint, `${path}.text_span.native_ref.artifact_schema`, "must be clozn.answer-claims.v1");
  }
  if (textSpan.nativeRef.collection !== "derived.claims") {
    fail(endpoint, `${path}.text_span.native_ref.collection`, "must be derived.claims");
  }
  if (textSpan.nativeRef.id !== `claim-${index}`) {
    fail(endpoint, `${path}.text_span.native_ref.id`, `must be claim-${index}`);
  }
  if (textSpan.resolution.state !== "metadata_only" || !textSpan.resolution.canonical) {
    fail(endpoint, `${path}.text_span.resolution`, "must be a metadata-only canonical claim span");
  }
  const canonical = textSpan.resolution.canonical;
  if (canonical.basis !== "recorded_answer") {
    fail(endpoint, `${path}.text_span.resolution.canonical.basis`, "must be recorded_answer");
  }
  if (canonical.end <= canonical.start) {
    fail(endpoint, `${path}.text_span.resolution.canonical`, "must select at least one Unicode code point");
  }
  if (
    answerSource.basisSha256 === undefined
    || canonical.basisSha256 !== answerSource.basisSha256
  ) {
    fail(endpoint, `${path}.text_span.resolution.canonical.basis_sha256`, "must match answer_source");
  }
  if (answerSource.basisCodePoints === undefined || canonical.end > answerSource.basisCodePoints) {
    fail(endpoint, `${path}.text_span.resolution.canonical.end`, "must be within the recorded answer");
  }
  if (canonical.basisCodePoints !== answerSource.basisCodePoints) {
    fail(endpoint, `${path}.text_span.resolution.canonical.basis_code_points`, "must match answer_source");
  }
  if (canonical.basisUtf8Bytes !== answerSource.basisUtf8Bytes) {
    fail(endpoint, `${path}.text_span.resolution.canonical.basis_utf8_bytes`, "must match answer_source");
  }
  if (canonical.spanCodePoints !== canonical.end - canonical.start) {
    fail(endpoint, `${path}.text_span.resolution.canonical.span_code_points`, "must match the half-open range");
  }
  if (
    canonical.spanUtf8Bytes === undefined
    || canonical.spanUtf8Bytes < canonical.spanCodePoints
    || canonical.spanUtf8Bytes > (answerSource.basisUtf8Bytes ?? -1)
  ) {
    fail(endpoint, `${path}.text_span.resolution.canonical.span_utf8_bytes`, "is inconsistent with the answer span");
  }
  return { index: decodedIndex, category, categoryReason, textSpan };
}

export function decodeAnswerClaims(
  value: unknown,
  endpoint = "/runs/<id>/claim-support",
  expectedRunId?: string,
): AnswerClaimsDocument {
  const row = object(value, endpoint, "$.claims");
  closed(row, [
    "schema_version", "run_id", "privacy", "offset_contract", "segmentation", "answer_source", "claims",
  ], endpoint, "$.claims");
  if (required(row, "schema_version", endpoint, "$.claims") !== "clozn.answer-claims.v1") {
    fail(endpoint, "$.claims.schema_version", "must be clozn.answer-claims.v1");
  }
  const runId = nonEmptyText(required(row, "run_id", endpoint, "$.claims"), endpoint, "$.claims.run_id");
  if (expectedRunId !== undefined && runId !== expectedRunId) {
    fail(endpoint, "$.claims.run_id", "does not match the request");
  }
  if (required(row, "privacy", endpoint, "$.claims") !== "metadata_only") {
    fail(endpoint, "$.claims.privacy", "must be metadata_only for this reader");
  }
  decodeOffsetContract(required(row, "offset_contract", endpoint, "$.claims"), endpoint, "$.claims.offset_contract");
  const segmentation = decodeSegmentation(
    required(row, "segmentation", endpoint, "$.claims"), endpoint, "$.claims.segmentation",
  );
  const answerSource = decodeAnswerSource(
    required(row, "answer_source", endpoint, "$.claims"), segmentation, endpoint, "$.claims.answer_source",
  );
  const claims = array(required(row, "claims", endpoint, "$.claims"), endpoint, "$.claims.claims")
    .map((claim, index) => decodeClaim(
      claim, index, runId, answerSource, endpoint, `$.claims.claims[${index}]`,
    ));
  if (segmentation.state === "ok" && segmentation.claimCount !== claims.length) {
    fail(endpoint, "$.claims.segmentation.claim_count", "must equal the claims array length");
  }
  if (segmentation.state !== "ok" && claims.length !== 0) {
    fail(endpoint, "$.claims.claims", `must be empty for ${segmentation.state} segmentation`);
  }
  const addressIds = claims.map((claim) => claim.textSpan.addressId);
  if (new Set(addressIds).size !== addressIds.length) {
    fail(endpoint, "$.claims.claims", "must not contain duplicate claim address ids");
  }
  const relationKeys = claims.map((claim) => claim.textSpan.relationKey);
  if (new Set(relationKeys).size !== relationKeys.length) {
    fail(endpoint, "$.claims.claims", "must not contain duplicate claim relation keys");
  }
  for (let index = 1; index < claims.length; index += 1) {
    const previous = claims[index - 1].textSpan.resolution.canonical;
    const current = claims[index].textSpan.resolution.canonical;
    if (!previous || !current || current.start < previous.end) {
      fail(endpoint, `$.claims.claims[${index}].text_span`, "must follow the previous claim without overlap");
    }
  }
  return {
    schemaVersion: "clozn.answer-claims.v1",
    runId,
    privacy: "metadata_only",
    segmentation,
    answerSource,
    claims,
  };
}

function decodeSourceSpanIds(
  value: unknown,
  claimAddressIds: ReadonlySet<string>,
  endpoint: string,
  path: string,
): readonly string[] {
  const values = array(value, endpoint, path).map((item, index) => patterned(
    item, ADDRESS_ID, "a span address id", endpoint, `${path}[${index}]`,
  ));
  if (values.length === 0) fail(endpoint, path, "must contain at least one source span id");
  if (new Set(values).size !== values.length) fail(endpoint, path, "must not contain duplicates");
  for (const value of values) {
    if (claimAddressIds.has(value)) fail(endpoint, path, "must reference a supplied-source span, not a claim span");
  }
  return values;
}

function decodeSupportResult(
  value: unknown,
  claim: AnswerClaim,
  gate: InfluenceGate,
  claimAddressIds: ReadonlySet<string>,
  endpoint: string,
  path: string,
): ClaimSupportResult {
  const row = object(value, endpoint, path);
  closed(row, ["claim_index", "claim_address_id", "status", "method", "source_span_ids"], endpoint, path);
  const claimIndex = nonNegativeInteger(required(row, "claim_index", endpoint, path), endpoint, `${path}.claim_index`);
  if (claimIndex !== claim.index) fail(endpoint, `${path}.claim_index`, "must match the claim at this position");
  const claimAddressId = patterned(
    required(row, "claim_address_id", endpoint, path), ADDRESS_ID, "a span address id", endpoint, `${path}.claim_address_id`,
  );
  if (claimAddressId !== claim.textSpan.addressId) {
    fail(endpoint, `${path}.claim_address_id`, "must match the source claim address");
  }
  const status = oneOf(required(row, "status", endpoint, path), SUPPORT_STATUSES, endpoint, `${path}.status`);
  const method = object(required(row, "method", endpoint, path), endpoint, `${path}.method`);
  const hasSources = Object.prototype.hasOwnProperty.call(row, "source_span_ids");

  if (claim.category !== "factual_claim" && status !== "unverifiable_from_available_evidence") {
    fail(endpoint, `${path}.status`, "must be unverifiable_from_available_evidence for a non-factual claim");
  }
  if (claim.category === "factual_claim" && status === "unverifiable_from_available_evidence") {
    fail(endpoint, `${path}.status`, "cannot use the category-rule status for a factual claim");
  }
  if (gate !== "ok" && claim.category === "factual_claim" && status !== "measurement_unavailable") {
    fail(endpoint, `${path}.status`, `must be measurement_unavailable while the influence gate is ${gate}`);
  }

  const base = { claimIndex, claimAddressId };
  if (status === "supported") {
    closed(method, ["name", "max_abs_delta_nats"], endpoint, `${path}.method`);
    if (required(method, "name", endpoint, `${path}.method`) !== "forced_score_intervention") {
      fail(endpoint, `${path}.method.name`, "must be forced_score_intervention for supported status");
    }
    if (!hasSources) fail(endpoint, path, "is missing source_span_ids for supported status");
    return {
      ...base,
      status,
      method: {
        name: "forced_score_intervention",
        maxAbsDeltaNats: nonNegativeNumber(
          required(method, "max_abs_delta_nats", endpoint, `${path}.method`), endpoint, `${path}.method.max_abs_delta_nats`,
        ),
      },
      sourceSpanIds: decodeSourceSpanIds(row.source_span_ids, claimAddressIds, endpoint, `${path}.source_span_ids`),
    };
  }
  if (status === "weakly_supported") {
    closed(method, ["name", "overlap_fraction"], endpoint, `${path}.method`);
    if (required(method, "name", endpoint, `${path}.method`) !== "textual_overlap") {
      fail(endpoint, `${path}.method.name`, "must be textual_overlap for weakly_supported status");
    }
    const overlapFraction = nonNegativeNumber(
      required(method, "overlap_fraction", endpoint, `${path}.method`), endpoint, `${path}.method.overlap_fraction`,
    );
    if (overlapFraction > 1) fail(endpoint, `${path}.method.overlap_fraction`, "must not exceed 1");
    if (!hasSources) fail(endpoint, path, "is missing source_span_ids for weakly_supported status");
    return {
      ...base,
      status,
      method: { name: "textual_overlap", overlapFraction },
      sourceSpanIds: decodeSourceSpanIds(row.source_span_ids, claimAddressIds, endpoint, `${path}.source_span_ids`),
    };
  }
  if (status === "contradicted") {
    closed(method, ["name"], endpoint, `${path}.method`);
    const name = oneOf(
      required(method, "name", endpoint, `${path}.method`),
      ["numeric_or_date_mismatch", "direct_negation"] as const,
      endpoint,
      `${path}.method.name`,
    );
    if (!hasSources) fail(endpoint, path, "is missing source_span_ids for contradicted status");
    return {
      ...base,
      status,
      method: { name },
      sourceSpanIds: decodeSourceSpanIds(row.source_span_ids, claimAddressIds, endpoint, `${path}.source_span_ids`),
    };
  }
  if (hasSources) fail(endpoint, `${path}.source_span_ids`, `must be absent for ${status} status`);
  closed(method, ["name"], endpoint, `${path}.method`);
  const methodName = nonEmptyText(required(method, "name", endpoint, `${path}.method`), endpoint, `${path}.method.name`);
  if (status === "unsupported_by_supplied_materials") {
    if (methodName !== "measured_comparison_no_match") {
      fail(endpoint, `${path}.method.name`, "must be measured_comparison_no_match for unsupported status");
    }
    return { ...base, status, method: { name: "measured_comparison_no_match" } };
  }
  if (status === "unverifiable_from_available_evidence") {
    if (methodName !== "category_rule") {
      fail(endpoint, `${path}.method.name`, "must be category_rule for unverifiable status");
    }
    return { ...base, status, method: { name: "category_rule" } };
  }
  const unavailableMethod = oneOf(methodName, MEASUREMENT_UNAVAILABLE_METHODS, endpoint, `${path}.method.name`);
  if (gate !== "ok" && unavailableMethod !== gate) {
    fail(endpoint, `${path}.method.name`, "must match the failed influence gate");
  }
  if (gate === "ok" && unavailableMethod !== "answer_text_mismatch" && unavailableMethod !== "no_resolvable_answer_spans") {
    fail(endpoint, `${path}.method.name`, "is incompatible with an ok influence gate");
  }
  return { ...base, status, method: { name: unavailableMethod } };
}

export function decodeClaimSupportResults(
  value: unknown,
  claims: AnswerClaimsDocument,
  endpoint = "/runs/<id>/claim-support",
): ClaimSupportResultsDocument {
  const row = object(value, endpoint, "$.support");
  closed(row, ["schema_version", "run_id", "privacy", "offset_contract", "source", "results"], endpoint, "$.support");
  if (required(row, "schema_version", endpoint, "$.support") !== "clozn.claim-support.v1") {
    fail(endpoint, "$.support.schema_version", "must be clozn.claim-support.v1");
  }
  const runId = nonEmptyText(required(row, "run_id", endpoint, "$.support"), endpoint, "$.support.run_id");
  if (runId !== claims.runId) fail(endpoint, "$.support.run_id", "must match the claims run_id");
  if (required(row, "privacy", endpoint, "$.support") !== "metadata_only") {
    fail(endpoint, "$.support.privacy", "must be metadata_only for this reader");
  }
  decodeOffsetContract(required(row, "offset_contract", endpoint, "$.support"), endpoint, "$.support.offset_contract");
  const source = object(required(row, "source", endpoint, "$.support"), endpoint, "$.support.source");
  closed(source, ["claims_schema_version", "influence_map"], endpoint, "$.support.source");
  if (required(source, "claims_schema_version", endpoint, "$.support.source") !== claims.schemaVersion) {
    fail(endpoint, "$.support.source.claims_schema_version", "must match the claims artifact schema");
  }
  const influence = object(
    required(source, "influence_map", endpoint, "$.support.source"), endpoint, "$.support.source.influence_map",
  );
  closed(influence, ["gate"], endpoint, "$.support.source.influence_map");
  const influenceGate = oneOf(
    required(influence, "gate", endpoint, "$.support.source.influence_map"),
    INFLUENCE_GATES,
    endpoint,
    "$.support.source.influence_map.gate",
  );
  const rawResults = array(required(row, "results", endpoint, "$.support"), endpoint, "$.support.results");
  if (rawResults.length !== claims.claims.length) {
    fail(endpoint, "$.support.results", "must contain exactly one ordered result for every claim");
  }
  const claimAddressIds = new Set(claims.claims.map((claim) => claim.textSpan.addressId));
  const results = rawResults.map((result, index) => decodeSupportResult(
    result, claims.claims[index], influenceGate, claimAddressIds, endpoint, `$.support.results[${index}]`,
  ));
  const resultIndexes = results.map((result) => result.claimIndex);
  if (new Set(resultIndexes).size !== resultIndexes.length) {
    fail(endpoint, "$.support.results", "must not contain duplicate claim indexes");
  }
  const resultAddresses = results.map((result) => result.claimAddressId);
  if (new Set(resultAddresses).size !== resultAddresses.length) {
    fail(endpoint, "$.support.results", "must not contain duplicate claim address ids");
  }
  return {
    schemaVersion: "clozn.claim-support.v1",
    runId,
    privacy: "metadata_only",
    claimsSchemaVersion: "clozn.answer-claims.v1",
    influenceGate,
    results,
  };
}

/** Decode the exact metadata-only pair returned by GET /runs/<id>/claim-support. */
export function decodeClaimSupportDocument(
  value: unknown,
  endpoint = "/runs/<id>/claim-support",
  expectedRunId?: string,
): ClaimSupportDocument {
  const row = object(value, endpoint, "$");
  closed(row, ["claims", "support"], endpoint, "$");
  const claims = decodeAnswerClaims(required(row, "claims", endpoint, "$"), endpoint, expectedRunId);
  const support = decodeClaimSupportResults(required(row, "support", endpoint, "$"), claims, endpoint);
  return { claims, support };
}

/** Project and validate the two recorded-run fields this surface consumes. */
export function decodeRecordedRunResponse(
  value: unknown,
  endpoint = "/runs/<id>",
  expectedRunId?: string,
): RecordedRunResponse {
  const row = object(value, endpoint, "$");
  const runId = nonEmptyText(required(row, "id", endpoint, "$"), endpoint, "$.id");
  if (expectedRunId !== undefined && runId !== expectedRunId) fail(endpoint, "$.id", "does not match the request");
  const rawResponse = required(row, "response", endpoint, "$");
  if (rawResponse !== null && typeof rawResponse !== "string") fail(endpoint, "$.response", "must be a string or null");
  return { runId, response: rawResponse as string | null };
}

/** Slice a Python-style Unicode code-point half-open range without using JavaScript UTF-16 indexes. */
export function sliceUnicodeCodePoints(value: string, start: number, end: number): string {
  return Array.from(value).slice(start, end).join("");
}

async function sha256(value: string, endpoint: string, path: string): Promise<string> {
  if (!globalThis.crypto?.subtle) fail(endpoint, path, "cannot verify SHA-256 in this runtime");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Bind metadata-only offsets and hashes to the separately read, already-recorded response. */
export async function bindRecordedResponse(
  document: ClaimSupportDocument,
  recorded: RecordedRunResponse,
  endpoint = "/runs/<id>/claim-support",
): Promise<ClaimSupportEvidence> {
  if (recorded.runId !== document.claims.runId) fail(endpoint, "$.recorded.id", "does not match the claim-support run");
  const { answerSource, segmentation } = document.claims;
  if (segmentation.state === "unavailable") {
    if (recorded.response !== null) fail(endpoint, "$.recorded.response", "must be unavailable with the claims artifact");
    return { ...document, recordedResponse: null, claimTexts: [] };
  }
  if (recorded.response === null) fail(endpoint, "$.recorded.response", "is required by the claims artifact");
  const codePoints = Array.from(recorded.response).length;
  const utf8Bytes = new TextEncoder().encode(recorded.response).byteLength;
  if (codePoints !== answerSource.basisCodePoints) {
    fail(endpoint, "$.recorded.response", "Unicode code-point count does not match answer_source");
  }
  if (utf8Bytes !== answerSource.basisUtf8Bytes) {
    fail(endpoint, "$.recorded.response", "UTF-8 byte count does not match answer_source");
  }
  if (await sha256(recorded.response, endpoint, "$.recorded.response") !== answerSource.basisSha256) {
    fail(endpoint, "$.recorded.response", "SHA-256 does not match answer_source");
  }
  const claimTexts: string[] = [];
  for (const claim of document.claims.claims) {
    const canonical = claim.textSpan.resolution.canonical;
    if (!canonical) fail(endpoint, `$.claims.claims[${claim.index}].text_span`, "has no canonical range");
    const selected = sliceUnicodeCodePoints(recorded.response, canonical.start, canonical.end);
    if (Array.from(selected).length !== canonical.spanCodePoints) {
      fail(endpoint, `$.claims.claims[${claim.index}].text_span`, "does not select the declared code-point count");
    }
    if (new TextEncoder().encode(selected).byteLength !== canonical.spanUtf8Bytes) {
      fail(endpoint, `$.claims.claims[${claim.index}].text_span`, "does not select the declared UTF-8 byte count");
    }
    if (await sha256(selected, endpoint, `$.claims.claims[${claim.index}].text_span`) !== canonical.spanSha256) {
      fail(endpoint, `$.claims.claims[${claim.index}].text_span`, "does not match the declared span SHA-256");
    }
    claimTexts.push(selected);
  }
  return { ...document, recordedResponse: recorded.response, claimTexts };
}
