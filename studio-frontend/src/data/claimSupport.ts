import { num, parseAddress, record, records, str, type SpanAddress } from "./received-context";

/**
 * Client for GET /runs/<id>/claim-support (clozn/server/routes/claim_support.py) -- E1's
 * `clozn.answer-claims.v1` claim segmentation (clozn/runs/claims.py) and E2's `clozn.claim-support.v1`
 * per-claim verification status (clozn/runs/claim_support.py), served together, derived fresh on every
 * request. Both are pure, metadata-only projections -- a claim's `textSpan` carries offsets and hashes,
 * never the literal answer substring (the SAME `clozn.text-span-addresses.v1` address shape
 * `data/received-context.ts` already types as `SpanAddress` and decodes via its own exported
 * `parseAddress` -- reused here verbatim, never a second parser for the one wire shape, for `kind:
 * "claim"`).
 *
 * THE SIX STATUSES, ONE RULE EACH (see clozn/runs/claim_support.py's own docstring, "the three honesty
 * rules"): `unsupported_by_supplied_materials` means exactly "the supplied materials do not support
 * this" -- never a claim about truth. `measurement_unavailable` means the verification measurement
 * itself could not be consulted or reconciled with this run's answer text at all -- a distinct, honest
 * state from "measured and found nothing" (`unsupported_by_supplied_materials`). `contradicted` requires
 * explicit evidence (a disjoint number/date, or an unnegated-claim/negated-source pair) at a high
 * overlap bar, never merely the absence of support. `unverifiable_from_available_evidence` is the
 * category-rule status for every non-`factual_claim` category (recommendation, uncertainty_statement,
 * instruction_procedure, non_verifiable_prose) -- a category rule, never a per-claim judgment call.
 *
 * A caller that already shows this run's recorded answer text elsewhere (Studio's Lens response pane
 * does) recovers each claim's own substring locally from `textSpan.resolution.canonical.start`/`end` --
 * UNICODE CODE POINT offsets, half-open, per `offsetContract` -- never a second offset scheme.
 */

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

export type SegmentationState = "ok" | "empty" | "segmentation_limited" | "unavailable";

export type SegmentationReason =
  | "answer_text_empty"
  | "answer_text_redacted"
  | "no_answer_text"
  | "unsupported_script_density";

export interface Segmentation {
  state: SegmentationState;
  reason?: SegmentationReason;
  claimCount?: number;
}

export interface Claim {
  index: number;
  category: ClaimCategory;
  categoryReason: ClaimCategoryReason;
  textSpan: SpanAddress;
}

export interface OffsetContract {
  unit: string;
  interval: string;
  hashAlgorithm: string;
  canonicalization: string;
}

/** Present whenever `segmentation.state !== "unavailable"`. `basisCodePoints` is the cheap, sufficient
 * cross-check a caller needs before trusting `textSpan.resolution.canonical.start`/`end` against a
 * SEPARATELY fetched copy of this run's answer text (`loadRunAnswerText` below): if the two code-point
 * counts disagree, the two were never the same text and offsets must not be sliced against it. */
export interface AnswerSource {
  basis: string;
  basisSha256?: string;
  basisCodePoints?: number;
  basisUtf8Bytes?: number;
}

export interface AnswerClaimsDocument {
  schemaVersion: string;
  runId: string;
  privacy: string;
  offsetContract: OffsetContract;
  segmentation: Segmentation;
  answerSource: AnswerSource;
  claims: Claim[];
}

export type ClaimSupportStatus =
  | "supported"
  | "weakly_supported"
  | "contradicted"
  | "unsupported_by_supplied_materials"
  | "unverifiable_from_available_evidence"
  | "measurement_unavailable";

export type ClaimSupportMethodName =
  | "forced_score_intervention"
  | "textual_overlap"
  | "numeric_or_date_mismatch"
  | "direct_negation"
  | "measured_comparison_no_match"
  | "category_rule"
  | "no_influence_map"
  | "influence_measurement_unavailable"
  | "influence_measurement_error"
  | "answer_text_mismatch"
  | "no_resolvable_answer_spans";

export interface ClaimSupportMethod {
  name: ClaimSupportMethodName;
  maxAbsDeltaNats?: number;
  overlapFraction?: number;
}

export interface ClaimSupportResult {
  claimIndex: number;
  claimAddressId: string;
  status: ClaimSupportStatus;
  method: ClaimSupportMethod;
  /** Real `clozn.text-span-addresses.v1` address_id values. Present only for
   * supported/weakly_supported/contradicted -- absent, never an empty array, for every other status. */
  sourceSpanIds?: string[];
}

/** Whole-run influence-measurement gate (`source.influence_map.gate`). "ok" means the measurement was
 * consulted and reconciled with this run's answer text -- individual claims may still land on any status.
 * Any other value means the gate failed for the WHOLE run before any per-claim comparison could run. */
export type InfluenceGate =
  | "ok"
  | "no_influence_map"
  | "influence_measurement_unavailable"
  | "influence_measurement_error"
  | "answer_text_mismatch"
  | "no_resolvable_answer_spans";

export interface ClaimSupportDocument {
  schemaVersion: string;
  runId: string;
  privacy: string;
  offsetContract: OffsetContract;
  claimsSchemaVersion: string;
  influenceGate: InfluenceGate;
  results: ClaimSupportResult[];
}

export interface ClaimVerification {
  claims: AnswerClaimsDocument;
  support: ClaimSupportDocument;
}

// ----------------------------------------------------------------------------------------------- parse

function strArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const items = value.filter((item): item is string => typeof item === "string");
  return items.length ? items : undefined;
}

const CLAIM_CATEGORIES = new Set<ClaimCategory>([
  "factual_claim", "recommendation", "uncertainty_statement", "instruction_procedure",
  "non_verifiable_prose",
]);

function claimCategory(value: unknown): ClaimCategory {
  return typeof value === "string" && CLAIM_CATEGORIES.has(value as ClaimCategory)
    ? value as ClaimCategory
    : "non_verifiable_prose";
}

const CATEGORY_REASONS = new Set<ClaimCategoryReason>([
  "code_fence_block", "interrogative_sentence", "hedge_marker", "recommendation_marker",
  "list_item_imperative", "imperative_lead", "factual_declarative", "no_deterministic_category_match",
]);

function categoryReason(value: unknown): ClaimCategoryReason {
  return typeof value === "string" && CATEGORY_REASONS.has(value as ClaimCategoryReason)
    ? value as ClaimCategoryReason
    : "no_deterministic_category_match";
}

const SEGMENTATION_STATES = new Set<SegmentationState>(["ok", "empty", "segmentation_limited", "unavailable"]);

function segmentationState(value: unknown): SegmentationState {
  return typeof value === "string" && SEGMENTATION_STATES.has(value as SegmentationState)
    ? value as SegmentationState
    : "unavailable";
}

const SEGMENTATION_REASONS = new Set<SegmentationReason>([
  "answer_text_empty", "answer_text_redacted", "no_answer_text", "unsupported_script_density",
]);

function segmentationReason(value: unknown): SegmentationReason | undefined {
  return typeof value === "string" && SEGMENTATION_REASONS.has(value as SegmentationReason)
    ? value as SegmentationReason
    : undefined;
}

/** Empty-address fallback -- `parseAddress` (data/received-context.ts) returns `null` only when
 * `address_id` itself is missing, which the `clozn.answer-claims.v1` schema never allows for a present
 * claim; this exists solely so `parseClaim` below stays a total function without a third "maybe absent"
 * branch its caller would have to handle. */
const EMPTY_SPAN_ADDRESS: SpanAddress = {
  addressId: "", kind: "claim", nativeRef: {}, resolution: { state: "unavailable" },
};

function parseClaim(raw: unknown): Claim {
  const item = record(raw);
  return {
    index: num(item.index) ?? 0,
    category: claimCategory(item.category),
    categoryReason: categoryReason(item.category_reason),
    textSpan: parseAddress(item.text_span) ?? EMPTY_SPAN_ADDRESS,
  };
}

function parseOffsetContract(raw: unknown): OffsetContract {
  const item = record(raw);
  return {
    unit: str(item.unit) ?? "",
    interval: str(item.interval) ?? "",
    hashAlgorithm: str(item.hash_algorithm) ?? "",
    canonicalization: str(item.canonicalization) ?? "",
  };
}

function parseSegmentation(raw: unknown): Segmentation {
  const item = record(raw);
  return {
    state: segmentationState(item.state),
    reason: segmentationReason(item.reason),
    claimCount: num(item.claim_count),
  };
}

function parseAnswerSource(raw: unknown): AnswerSource {
  const item = record(raw);
  return {
    basis: str(item.basis) ?? "recorded_answer",
    basisSha256: str(item.basis_sha256),
    basisCodePoints: num(item.basis_code_points),
    basisUtf8Bytes: num(item.basis_utf8_bytes),
  };
}

function parseClaimsDocument(raw: unknown, requestedRunId: string): AnswerClaimsDocument {
  const doc = record(raw);
  return {
    schemaVersion: str(doc.schema_version) ?? "",
    runId: str(doc.run_id) ?? requestedRunId,
    privacy: str(doc.privacy) ?? "metadata_only",
    offsetContract: parseOffsetContract(doc.offset_contract),
    segmentation: parseSegmentation(doc.segmentation),
    answerSource: parseAnswerSource(doc.answer_source),
    claims: records(doc.claims).map(parseClaim),
  };
}

const SUPPORT_STATUSES = new Set<ClaimSupportStatus>([
  "supported", "weakly_supported", "contradicted", "unsupported_by_supplied_materials",
  "unverifiable_from_available_evidence", "measurement_unavailable",
]);

function supportStatus(value: unknown): ClaimSupportStatus {
  return typeof value === "string" && SUPPORT_STATUSES.has(value as ClaimSupportStatus)
    ? value as ClaimSupportStatus
    : "measurement_unavailable";
}

const METHOD_NAMES = new Set<ClaimSupportMethodName>([
  "forced_score_intervention", "textual_overlap", "numeric_or_date_mismatch", "direct_negation",
  "measured_comparison_no_match", "category_rule", "no_influence_map",
  "influence_measurement_unavailable", "influence_measurement_error", "answer_text_mismatch",
  "no_resolvable_answer_spans",
]);

function methodName(value: unknown): ClaimSupportMethodName {
  return typeof value === "string" && METHOD_NAMES.has(value as ClaimSupportMethodName)
    ? value as ClaimSupportMethodName
    : "no_resolvable_answer_spans";
}

function parseMethod(raw: unknown): ClaimSupportMethod {
  const item = record(raw);
  return {
    name: methodName(item.name),
    maxAbsDeltaNats: num(item.max_abs_delta_nats),
    overlapFraction: num(item.overlap_fraction),
  };
}

function parseResult(raw: unknown): ClaimSupportResult {
  const item = record(raw);
  return {
    claimIndex: num(item.claim_index) ?? 0,
    claimAddressId: str(item.claim_address_id) ?? "",
    status: supportStatus(item.status),
    method: parseMethod(item.method),
    sourceSpanIds: strArray(item.source_span_ids),
  };
}

const INFLUENCE_GATES = new Set<InfluenceGate>([
  "ok", "no_influence_map", "influence_measurement_unavailable", "influence_measurement_error",
  "answer_text_mismatch", "no_resolvable_answer_spans",
]);

function influenceGate(value: unknown): InfluenceGate {
  return typeof value === "string" && INFLUENCE_GATES.has(value as InfluenceGate)
    ? value as InfluenceGate
    : "no_influence_map";
}

function parseSupportDocument(raw: unknown, requestedRunId: string): ClaimSupportDocument {
  const doc = record(raw);
  const source = record(doc.source);
  const influenceMap = record(source.influence_map);
  return {
    schemaVersion: str(doc.schema_version) ?? "",
    runId: str(doc.run_id) ?? requestedRunId,
    privacy: str(doc.privacy) ?? "metadata_only",
    offsetContract: parseOffsetContract(doc.offset_contract),
    claimsSchemaVersion: str(source.claims_schema_version) ?? "",
    influenceGate: influenceGate(influenceMap.gate),
    results: records(doc.results).map(parseResult),
  };
}

export class ClaimVerificationLoadError extends Error {}

async function getJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { signal });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof record(body).error === "string"
      ? (record(body).error as string)
      : `Request failed (${response.status})`;
    throw new ClaimVerificationLoadError(message);
  }
  return body;
}

/** The ONE fetch this feature needs for claims + support -- a GET, nothing else. Rendering this panel
 * must never itself start a measurement or mutate a run; see clozn/server/routes/claim_support.py's own
 * "pure composition, nothing more" contract. */
export async function loadClaimVerification(runId: string, signal?: AbortSignal): Promise<ClaimVerification> {
  const raw = record(await getJson(`/runs/${encodeURIComponent(runId)}/claim-support`, signal));
  return {
    claims: parseClaimsDocument(raw.claims, runId),
    support: parseSupportDocument(raw.support, runId),
  };
}

/** The run's recorded answer text, fetched separately from the untouched generic `GET /runs/<id>`
 * (the same route `loadRunInspection` already reads `response` from) -- `claim-support` stays
 * metadata-only and never embeds it, so a caller that wants to slice claim substrings out of the real
 * text using `textSpan.resolution.canonical.start`/`end` gets it here. Unicode CODE POINT offsets, not
 * UTF-16 code units -- callers must index via `Array.from(text)`, never `text.slice`, or a span past the
 * BMP silently misaligns. */
export async function loadRunAnswerText(runId: string, signal?: AbortSignal): Promise<string> {
  const raw = record(await getJson(`/runs/${encodeURIComponent(runId)}`, signal));
  return typeof raw.response === "string" ? raw.response : "";
}
