/**
 * Strict read-boundary decoders for canonical Context Investigation evidence.
 *
 * Wire offsets are Python Unicode code-point offsets. They are deliberately
 * retained as code-point coordinates here; presentation code must translate
 * them before indexing a browser UTF-16 string.
 */
import { ContractError } from "./contracts";

export type UnicodeRange = readonly [start: number, end: number];
export type ReaderStatus = "not_measured" | "partial" | "completed" | "stale" | "unavailable";
export type SourceMeasurementStatus = "measured" | "not_measured" | "unavailable" | "failed";
export type EffectDirection = "support" | "suppression" | "below_floor" | Exclude<SourceMeasurementStatus, "measured">;
export type EffectClassification = "measured_support" | "measured_suppression" | "below_floor" | null;

export interface ContextInvestigationMeasurement {
  status: ReaderStatus;
  experimentId: string | null;
  evaluator: "score_recorded_continuation";
  intervention: "delete_source";
  sourceUniverse: readonly string[];
  sourceUniverseBasis: string | null;
  completedSourceCount: number;
  unavailableSourceCount: number;
  failedSourceCount: number;
  notMeasuredSourceCount: number;
  baselineObservationId?: string | null;
}

export interface ContextInvestigationBlock {
  messageIndex: number;
  role: string;
  text: string;
  segmentId: string;
  displayUnicodeRange: UnicodeRange;
  sourceIds: readonly string[];
}

export interface ContextInvestigationContext {
  coordinateBasis: "context_display_unicode.v1";
  separator: string;
  blocks: readonly ContextInvestigationBlock[];
}

export interface ContextInvestigationAnswerToken {
  index: number;
  tokenId: number;
  piece: string;
  unicodeRange: UnicodeRange;
  logprob?: number;
}

export type ContextInvestigationAnswer =
  | {
      status: "available";
      text: string;
      sha256: string;
      tokens: readonly ContextInvestigationAnswerToken[];
      fidelity: "recorded_run_trace" | "baseline_token_score_observation";
      observationId: string | null;
    }
  | { status: "unavailable" };

export interface DeleteSourceIntervention {
  kind: "delete_source";
  target: {
    kind: "context_selection";
    sourceIds: readonly string[];
  };
}

export interface ContextSourceMeasurement {
  status: SourceMeasurementStatus;
  experimentId: string | null;
  armId: string | null;
  observationId: string | null;
  intervention?: DeleteSourceIntervention;
}

export interface ContextSourceSummary {
  maxAbsTokenDeltaNats: number | null;
  totalContinuationDeltaNats: number | null;
  aboveFloorLocusCount: number;
}

export interface ContextInvestigationSource {
  sourceId: string;
  segmentId: string;
  messageIndex: number;
  label?: string | null;
  provenanceKind?: string | null;
  parentSourceId?: string | null;
  unicodeRange: UnicodeRange;
  byteRange: UnicodeRange;
  displayUnicodeRange: UnicodeRange;
  granularity: "exact_span" | "whole_segment";
  measurement: ContextSourceMeasurement;
  summary: ContextSourceSummary;
  loci: readonly string[];
}

export interface ContextInvestigationLocus {
  locusId: string;
  sourceId: string;
  direction: "support" | "suppression";
  classification: "measured_support" | "measured_suppression";
  answerTokenRange: UnicodeRange;
  answerUnicodeRange: UnicodeRange;
  answerText: string;
  deltaNats: number;
  absoluteDeltaNats: number;
  baselineSelectedLogp: number;
  intervenedSelectedLogp: number;
  experimentId: string;
  armId: string;
  baselineObservationId: string;
  interventionObservationId: string;
  perTokenDeltaNats: readonly number[];
}

export interface ContextInvestigationReader {
  schemaVersion: "clozn.context-investigation-reader.v1";
  runId: string;
  status: ReaderStatus;
  reason?: string;
  reasonCode?: string;
  measurement: ContextInvestigationMeasurement;
  context: ContextInvestigationContext;
  answer: ContextInvestigationAnswer;
  sources: readonly ContextInvestigationSource[];
  loci: readonly ContextInvestigationLocus[];
  presentation: { measurementFloorNats: number };
  capabilities: {
    contextEffects: "measurable" | "unavailable";
    measurement: ReaderStatus;
    answerSelectionProjection: "available" | "unavailable";
  };
}

export interface ContextInvestigationSelection {
  unicodeRange: UnicodeRange;
  tokenRange: UnicodeRange;
  text: string;
  tokenIds: readonly number[];
  tokenPieces: readonly string[];
}

export interface ContextInvestigationEffect {
  sourceId: string;
  deltaNats: number | null;
  direction: EffectDirection;
  classification: EffectClassification;
  measurementStatus: SourceMeasurementStatus;
  experimentId: string;
  armId: string;
  baselineObservationId: string | null;
  interventionObservationId: string | null;
}

export interface ContextInvestigationQuery {
  schemaVersion: "clozn.context-investigation-query.v1";
  status: ReaderStatus;
  reason?: string;
  reasonCode?: string;
  selection: ContextInvestigationSelection | null;
  effects: readonly ContextInvestigationEffect[];
  measurementFloorNats: number;
}

type JsonRecord = Record<string, unknown>;

const READER_STATUSES = ["not_measured", "partial", "completed", "stale", "unavailable"] as const;
const SOURCE_STATUSES = ["measured", "not_measured", "unavailable", "failed"] as const;
const SHA256 = /^[0-9a-f]{64}$/;

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function record(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(endpoint, path, "must be an object");
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

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nonEmptyText(value: unknown, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!parsed.trim()) fail(endpoint, path, "must not be empty");
  return parsed;
}

function nullableNonEmptyText(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : nonEmptyText(value, endpoint, path);
}

function nullableText(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : text(value, endpoint, path);
}

function finite(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be a finite number");
  return value;
}

function nonNegative(value: unknown, endpoint: string, path: string): number {
  const parsed = finite(value, endpoint, path);
  if (parsed < 0) fail(endpoint, path, "must be non-negative");
  return parsed;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = nonNegative(value, endpoint, path);
  if (!Number.isSafeInteger(parsed)) fail(endpoint, path, "must be an integer");
  return parsed;
}

function oneOf<const T extends readonly string[]>(value: unknown, choices: T, endpoint: string, path: string): T[number] {
  if (typeof value !== "string" || !choices.includes(value)) fail(endpoint, path, `must be one of ${choices.join(", ")}`);
  return value as T[number];
}

function optional<T>(row: JsonRecord, key: string, parse: (value: unknown, path: string) => T, path: string): T | undefined {
  return key in row ? parse(row[key], `${path}.${key}`) : undefined;
}

function range(value: unknown, endpoint: string, path: string, nonEmpty = false): UnicodeRange {
  const values = list(value, endpoint, path);
  if (values.length !== 2) fail(endpoint, path, "must be a two-item half-open range");
  const start = nonNegativeInteger(values[0], endpoint, `${path}[0]`);
  const end = nonNegativeInteger(values[1], endpoint, `${path}[1]`);
  if (end < start || (nonEmpty && end === start)) fail(endpoint, path, `must have end ${nonEmpty ? "greater than" : "at least"} start`);
  return [start, end];
}

function nullableFinite(value: unknown, endpoint: string, path: string): number | null {
  return value === null ? null : finite(value, endpoint, path);
}

function stringList(value: unknown, endpoint: string, path: string): string[] {
  return list(value, endpoint, path).map((item, index) => nonEmptyText(item, endpoint, `${path}[${index}]`));
}

function assertUnique(values: readonly string[], endpoint: string, path: string): void {
  if (new Set(values).size !== values.length) fail(endpoint, path, "must not contain duplicate IDs");
}

function codePoints(value: string): string[] {
  return Array.from(value);
}

function codePointLength(value: string): number {
  return codePoints(value).length;
}

function codePointSlice(value: string, start: number, end: number): string {
  return codePoints(value).slice(start, end).join("");
}

function utf8Length(value: string): number {
  return new TextEncoder().encode(value).length;
}

function rangesEqual(left: UnicodeRange, right: UnicodeRange): boolean {
  return left[0] === right[0] && left[1] === right[1];
}

function nearlyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9 * Math.max(1, Math.abs(left), Math.abs(right));
}

function decodeMeasurement(value: unknown, endpoint: string, path: string): ContextInvestigationMeasurement {
  const row = record(value, endpoint, path);
  return {
    status: oneOf(required(row, "status", endpoint, path), READER_STATUSES, endpoint, `${path}.status`),
    experimentId: nullableNonEmptyText(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    evaluator: oneOf(required(row, "evaluator", endpoint, path), ["score_recorded_continuation"] as const, endpoint, `${path}.evaluator`),
    intervention: oneOf(required(row, "intervention", endpoint, path), ["delete_source"] as const, endpoint, `${path}.intervention`),
    sourceUniverse: stringList(required(row, "source_universe", endpoint, path), endpoint, `${path}.source_universe`),
    sourceUniverseBasis: nullableText(required(row, "source_universe_basis", endpoint, path), endpoint, `${path}.source_universe_basis`),
    completedSourceCount: nonNegativeInteger(required(row, "completed_source_count", endpoint, path), endpoint, `${path}.completed_source_count`),
    unavailableSourceCount: nonNegativeInteger(required(row, "unavailable_source_count", endpoint, path), endpoint, `${path}.unavailable_source_count`),
    failedSourceCount: nonNegativeInteger(required(row, "failed_source_count", endpoint, path), endpoint, `${path}.failed_source_count`),
    notMeasuredSourceCount: nonNegativeInteger(required(row, "not_measured_source_count", endpoint, path), endpoint, `${path}.not_measured_source_count`),
    baselineObservationId: optional(row, "baseline_observation_id", (item, itemPath) => nullableNonEmptyText(item, endpoint, itemPath), path),
  };
}

function decodeBlock(value: unknown, endpoint: string, path: string): ContextInvestigationBlock {
  const row = record(value, endpoint, path);
  const sourceIds = stringList(required(row, "source_ids", endpoint, path), endpoint, `${path}.source_ids`);
  assertUnique(sourceIds, endpoint, `${path}.source_ids`);
  return {
    messageIndex: nonNegativeInteger(required(row, "message_index", endpoint, path), endpoint, `${path}.message_index`),
    role: nonEmptyText(required(row, "role", endpoint, path), endpoint, `${path}.role`),
    text: text(required(row, "text", endpoint, path), endpoint, `${path}.text`),
    segmentId: nonEmptyText(required(row, "segment_id", endpoint, path), endpoint, `${path}.segment_id`),
    displayUnicodeRange: range(required(row, "display_unicode_range", endpoint, path), endpoint, `${path}.display_unicode_range`),
    sourceIds,
  };
}

function decodeContext(value: unknown, endpoint: string, path: string): ContextInvestigationContext {
  const row = record(value, endpoint, path);
  const context: ContextInvestigationContext = {
    coordinateBasis: oneOf(required(row, "coordinate_basis", endpoint, path), ["context_display_unicode.v1"] as const, endpoint, `${path}.coordinate_basis`),
    separator: text(required(row, "separator", endpoint, path), endpoint, `${path}.separator`),
    blocks: list(required(row, "blocks", endpoint, path), endpoint, `${path}.blocks`).map((item, index) => decodeBlock(item, endpoint, `${path}.blocks[${index}]`)),
  };

  const segmentIds = context.blocks.map((block) => block.segmentId);
  assertUnique(segmentIds, endpoint, `${path}.blocks[].segment_id`);
  assertUnique(context.blocks.map((block) => String(block.messageIndex)), endpoint, `${path}.blocks[].message_index`);
  const separatorLength = codePointLength(context.separator);
  let cursor = 0;
  context.blocks.forEach((block, index) => {
    if (block.messageIndex !== index) fail(endpoint, `${path}.blocks[${index}].message_index`, "must preserve recorded message order");
    const expected: UnicodeRange = [cursor, cursor + codePointLength(block.text)];
    if (!rangesEqual(block.displayUnicodeRange, expected)) {
      fail(endpoint, `${path}.blocks[${index}].display_unicode_range`, "does not match the continuous context code-point coordinate");
    }
    cursor = expected[1] + (index < context.blocks.length - 1 ? separatorLength : 0);
  });
  return context;
}

function decodeAnswerToken(value: unknown, endpoint: string, path: string): ContextInvestigationAnswerToken {
  const row = record(value, endpoint, path);
  return {
    index: nonNegativeInteger(required(row, "index", endpoint, path), endpoint, `${path}.index`),
    tokenId: nonNegativeInteger(required(row, "token_id", endpoint, path), endpoint, `${path}.token_id`),
    piece: text(required(row, "piece", endpoint, path), endpoint, `${path}.piece`),
    unicodeRange: range(required(row, "unicode_range", endpoint, path), endpoint, `${path}.unicode_range`),
    logprob: optional(row, "logprob", (item, itemPath) => finite(item, endpoint, itemPath), path),
  };
}

function decodeAnswer(value: unknown, endpoint: string, path: string): ContextInvestigationAnswer {
  const row = record(value, endpoint, path);
  const status = oneOf(required(row, "status", endpoint, path), ["available", "unavailable"] as const, endpoint, `${path}.status`);
  if (status === "unavailable") {
    if ("text" in row || "tokens" in row || "observation_id" in row) fail(endpoint, path, "must not carry available answer evidence when unavailable");
    return { status };
  }

  const answerText = text(required(row, "text", endpoint, path), endpoint, `${path}.text`);
  const sha256 = nonEmptyText(required(row, "sha256", endpoint, path), endpoint, `${path}.sha256`);
  if (!SHA256.test(sha256)) fail(endpoint, `${path}.sha256`, "must be a lowercase SHA-256 digest");
  const tokens = list(required(row, "tokens", endpoint, path), endpoint, `${path}.tokens`)
    .map((item, index) => decodeAnswerToken(item, endpoint, `${path}.tokens[${index}]`));
  const fidelity = oneOf(
    required(row, "fidelity", endpoint, path),
    ["recorded_run_trace", "baseline_token_score_observation"] as const,
    endpoint,
    `${path}.fidelity`,
  );
  const observationId = nullableNonEmptyText(required(row, "observation_id", endpoint, path), endpoint, `${path}.observation_id`);

  let cursor = 0;
  tokens.forEach((token, index) => {
    if (token.index !== index) fail(endpoint, `${path}.tokens[${index}].index`, "must match recorded token order");
    const expected: UnicodeRange = [cursor, cursor + codePointLength(token.piece)];
    if (!rangesEqual(token.unicodeRange, expected)) fail(endpoint, `${path}.tokens[${index}].unicode_range`, "must match its token piece in code-point coordinates");
    cursor = expected[1];
  });
  if (tokens.map((token) => token.piece).join("") !== answerText || cursor !== codePointLength(answerText)) {
    fail(endpoint, `${path}.tokens`, "must reconstruct and cover the recorded answer exactly");
  }
  if (fidelity === "recorded_run_trace") {
    if (observationId !== null || tokens.some((token) => token.logprob !== undefined)) {
      fail(endpoint, path, "recorded-run trace fidelity must not claim score-observation evidence");
    }
  } else if (observationId === null || tokens.some((token) => token.logprob === undefined)) {
    fail(endpoint, path, "baseline score fidelity requires one observation and a complete logprob vector");
  }
  return { status, text: answerText, sha256, tokens, fidelity, observationId };
}

function decodeIntervention(value: unknown, endpoint: string, path: string): DeleteSourceIntervention {
  const row = record(value, endpoint, path);
  const targetRow = record(required(row, "target", endpoint, path), endpoint, `${path}.target`);
  const sourceIds = stringList(required(targetRow, "source_ids", endpoint, `${path}.target`), endpoint, `${path}.target.source_ids`);
  assertUnique(sourceIds, endpoint, `${path}.target.source_ids`);
  return {
    kind: oneOf(required(row, "kind", endpoint, path), ["delete_source"] as const, endpoint, `${path}.kind`),
    target: {
      kind: oneOf(required(targetRow, "kind", endpoint, `${path}.target`), ["context_selection"] as const, endpoint, `${path}.target.kind`),
      sourceIds,
    },
  };
}

function decodeSourceMeasurement(value: unknown, endpoint: string, path: string): ContextSourceMeasurement {
  const row = record(value, endpoint, path);
  return {
    status: oneOf(required(row, "status", endpoint, path), SOURCE_STATUSES, endpoint, `${path}.status`),
    experimentId: nullableNonEmptyText(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    armId: nullableNonEmptyText(required(row, "arm_id", endpoint, path), endpoint, `${path}.arm_id`),
    observationId: nullableNonEmptyText(required(row, "observation_id", endpoint, path), endpoint, `${path}.observation_id`),
    intervention: optional(row, "intervention", (item, itemPath) => decodeIntervention(item, endpoint, itemPath), path),
  };
}

function decodeSourceSummary(value: unknown, endpoint: string, path: string): ContextSourceSummary {
  const row = record(value, endpoint, path);
  const maxAbs = nullableFinite(required(row, "max_abs_token_delta_nats", endpoint, path), endpoint, `${path}.max_abs_token_delta_nats`);
  if (maxAbs !== null && maxAbs < 0) fail(endpoint, `${path}.max_abs_token_delta_nats`, "must be non-negative");
  return {
    maxAbsTokenDeltaNats: maxAbs,
    totalContinuationDeltaNats: nullableFinite(required(row, "total_continuation_delta_nats", endpoint, path), endpoint, `${path}.total_continuation_delta_nats`),
    aboveFloorLocusCount: nonNegativeInteger(required(row, "above_floor_locus_count", endpoint, path), endpoint, `${path}.above_floor_locus_count`),
  };
}

function decodeSource(value: unknown, endpoint: string, path: string): ContextInvestigationSource {
  const row = record(value, endpoint, path);
  const loci = stringList(required(row, "loci", endpoint, path), endpoint, `${path}.loci`);
  assertUnique(loci, endpoint, `${path}.loci`);
  return {
    sourceId: nonEmptyText(required(row, "source_id", endpoint, path), endpoint, `${path}.source_id`),
    segmentId: nonEmptyText(required(row, "segment_id", endpoint, path), endpoint, `${path}.segment_id`),
    messageIndex: nonNegativeInteger(required(row, "message_index", endpoint, path), endpoint, `${path}.message_index`),
    label: optional(row, "label", (item, itemPath) => nullableText(item, endpoint, itemPath), path),
    provenanceKind: optional(row, "provenance_kind", (item, itemPath) => nullableText(item, endpoint, itemPath), path),
    parentSourceId: optional(row, "parent_source_id", (item, itemPath) => nullableNonEmptyText(item, endpoint, itemPath), path),
    unicodeRange: range(required(row, "unicode_range", endpoint, path), endpoint, `${path}.unicode_range`),
    byteRange: range(required(row, "byte_range", endpoint, path), endpoint, `${path}.byte_range`),
    displayUnicodeRange: range(required(row, "display_unicode_range", endpoint, path), endpoint, `${path}.display_unicode_range`),
    granularity: oneOf(required(row, "granularity", endpoint, path), ["exact_span", "whole_segment"] as const, endpoint, `${path}.granularity`),
    measurement: decodeSourceMeasurement(required(row, "measurement", endpoint, path), endpoint, `${path}.measurement`),
    summary: decodeSourceSummary(required(row, "summary", endpoint, path), endpoint, `${path}.summary`),
    loci,
  };
}

function decodeLocus(value: unknown, endpoint: string, path: string): ContextInvestigationLocus {
  const row = record(value, endpoint, path);
  return {
    locusId: nonEmptyText(required(row, "locus_id", endpoint, path), endpoint, `${path}.locus_id`),
    sourceId: nonEmptyText(required(row, "source_id", endpoint, path), endpoint, `${path}.source_id`),
    direction: oneOf(required(row, "direction", endpoint, path), ["support", "suppression"] as const, endpoint, `${path}.direction`),
    classification: oneOf(required(row, "classification", endpoint, path), ["measured_support", "measured_suppression"] as const, endpoint, `${path}.classification`),
    answerTokenRange: range(required(row, "answer_token_range", endpoint, path), endpoint, `${path}.answer_token_range`, true),
    answerUnicodeRange: range(required(row, "answer_unicode_range", endpoint, path), endpoint, `${path}.answer_unicode_range`, true),
    answerText: text(required(row, "answer_text", endpoint, path), endpoint, `${path}.answer_text`),
    deltaNats: finite(required(row, "delta_nats", endpoint, path), endpoint, `${path}.delta_nats`),
    absoluteDeltaNats: nonNegative(required(row, "absolute_delta_nats", endpoint, path), endpoint, `${path}.absolute_delta_nats`),
    baselineSelectedLogp: finite(required(row, "baseline_selected_logp", endpoint, path), endpoint, `${path}.baseline_selected_logp`),
    intervenedSelectedLogp: finite(required(row, "intervened_selected_logp", endpoint, path), endpoint, `${path}.intervened_selected_logp`),
    experimentId: nonEmptyText(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    armId: nonEmptyText(required(row, "arm_id", endpoint, path), endpoint, `${path}.arm_id`),
    baselineObservationId: nonEmptyText(required(row, "baseline_observation_id", endpoint, path), endpoint, `${path}.baseline_observation_id`),
    interventionObservationId: nonEmptyText(required(row, "intervention_observation_id", endpoint, path), endpoint, `${path}.intervention_observation_id`),
    perTokenDeltaNats: list(required(row, "per_token_delta_nats", endpoint, path), endpoint, `${path}.per_token_delta_nats`)
      .map((item, index) => finite(item, endpoint, `${path}.per_token_delta_nats[${index}]`)),
  };
}

function decodePresentation(value: unknown, endpoint: string, path: string): ContextInvestigationReader["presentation"] {
  const row = record(value, endpoint, path);
  return { measurementFloorNats: nonNegative(required(row, "measurement_floor_nats", endpoint, path), endpoint, `${path}.measurement_floor_nats`) };
}

function decodeCapabilities(value: unknown, endpoint: string, path: string): ContextInvestigationReader["capabilities"] {
  const row = record(value, endpoint, path);
  return {
    contextEffects: oneOf(required(row, "context_effects", endpoint, path), ["measurable", "unavailable"] as const, endpoint, `${path}.context_effects`),
    measurement: oneOf(required(row, "measurement", endpoint, path), READER_STATUSES, endpoint, `${path}.measurement`),
    answerSelectionProjection: oneOf(required(row, "answer_selection_projection", endpoint, path), ["available", "unavailable"] as const, endpoint, `${path}.answer_selection_projection`),
  };
}

function validateSourceCoordinates(reader: ContextInvestigationReader, endpoint: string): void {
  const blocks = new Map(reader.context.blocks.map((block) => [block.messageIndex, block]));
  const sourceById = new Map(reader.sources.map((source) => [source.sourceId, source]));
  assertUnique(reader.sources.map((source) => source.sourceId), endpoint, "$.sources[].source_id");

  reader.sources.forEach((source, index) => {
    const path = `$.sources[${index}]`;
    const block = blocks.get(source.messageIndex);
    if (!block) fail(endpoint, `${path}.message_index`, "does not reference a context block");
    if (source.segmentId !== block.segmentId) fail(endpoint, `${path}.segment_id`, "does not match its context block");
    if (!block.sourceIds.includes(source.sourceId)) fail(endpoint, `${path}.source_id`, "is not registered by its context block");
    const blockLength = codePointLength(block.text);
    if (source.unicodeRange[1] > blockLength) fail(endpoint, `${path}.unicode_range`, "extends beyond its context block");
    const expectedDisplay: UnicodeRange = [
      block.displayUnicodeRange[0] + source.unicodeRange[0],
      block.displayUnicodeRange[0] + source.unicodeRange[1],
    ];
    if (!rangesEqual(source.displayUnicodeRange, expectedDisplay)) {
      fail(endpoint, `${path}.display_unicode_range`, "does not register the source's block-local Unicode range");
    }
    const expectedBytes: UnicodeRange = [
      utf8Length(codePointSlice(block.text, 0, source.unicodeRange[0])),
      utf8Length(codePointSlice(block.text, 0, source.unicodeRange[1])),
    ];
    if (!rangesEqual(source.byteRange, expectedBytes)) fail(endpoint, `${path}.byte_range`, "does not match the UTF-8 byte range");
    if (source.granularity === "whole_segment" && !rangesEqual(source.unicodeRange, [0, blockLength])) {
      fail(endpoint, `${path}.unicode_range`, "must cover the whole block for whole-segment granularity");
    }
    if (source.granularity === "exact_span" && source.unicodeRange[0] === source.unicodeRange[1]) {
      fail(endpoint, `${path}.unicode_range`, "must be non-empty for exact-span granularity");
    }
    if (source.parentSourceId && !sourceById.has(source.parentSourceId)) {
      fail(endpoint, `${path}.parent_source_id`, "does not reference a projected source");
    }
  });

  const isCompleteUnavailableProjection = reader.status === "unavailable"
    && reader.capabilities.contextEffects === "unavailable"
    && reader.measurement.sourceUniverse.length === 0
    && reader.sources.length > 0;
  if (["not_measured", "partial", "completed"].includes(reader.status) || isCompleteUnavailableProjection) {
    const registeredIds = reader.context.blocks.flatMap((block) => block.sourceIds);
    assertUnique(registeredIds, endpoint, "$.context.blocks[].source_ids");
    if (registeredIds.length !== reader.sources.length || registeredIds.some((id) => !sourceById.has(id))) {
      fail(endpoint, "$.sources", "must exactly represent the context block source registrations");
    }
    reader.context.blocks.forEach((block, index) => {
      const root = sourceById.get(block.segmentId);
      if (!root || root.messageIndex !== block.messageIndex || root.granularity !== "whole_segment") {
        fail(endpoint, `$.context.blocks[${index}].segment_id`, "must resolve to its whole-segment source");
      }
    });
  }
}

function validateReaderEvidence(reader: ContextInvestigationReader, endpoint: string): void {
  const { measurement } = reader;
  const answer = reader.answer;
  assertUnique(measurement.sourceUniverse, endpoint, "$.measurement.source_universe");
  const countTotal = measurement.completedSourceCount + measurement.unavailableSourceCount
    + measurement.failedSourceCount + measurement.notMeasuredSourceCount;
  if (countTotal !== measurement.sourceUniverse.length) fail(endpoint, "$.measurement", "source status counts must partition the source universe");
  if (reader.status !== measurement.status || reader.status !== reader.capabilities.measurement) {
    fail(endpoint, "$.status", "must agree with measurement and capability status");
  }
  const answerAvailable = answer.status === "available";
  if ((reader.capabilities.answerSelectionProjection === "available") !== answerAvailable) {
    fail(endpoint, "$.capabilities.answer_selection_projection", "must agree with answer availability");
  }
  if (["stale", "unavailable"].includes(reader.status) && (!reader.reason || !reader.reasonCode)) {
    fail(endpoint, "$", "must include a reason and reason_code for stale or unavailable evidence");
  }
  if (!answerAvailable && reader.loci.length) fail(endpoint, "$.loci", "must be empty when recorded answer coordinates are unavailable");
  if (["stale", "unavailable"].includes(reader.status) && reader.loci.length) fail(endpoint, "$.loci", "must be empty for stale or unavailable evidence");
  if (reader.status === "not_measured" && (
    measurement.completedSourceCount !== 0 || measurement.unavailableSourceCount !== 0 || measurement.failedSourceCount !== 0
    || reader.loci.length !== 0
  )) fail(endpoint, "$.measurement", "not_measured must not claim measured, unavailable, failed, or locus evidence");
  if (reader.status === "completed" && (
    measurement.completedSourceCount !== measurement.sourceUniverse.length
    || measurement.unavailableSourceCount !== 0 || measurement.failedSourceCount !== 0 || measurement.notMeasuredSourceCount !== 0
    || !measurement.baselineObservationId || !answerAvailable
  )) fail(endpoint, "$.measurement", "completed must have a baseline and every source measured");
  if (reader.status === "partial" && (
    measurement.completedSourceCount + measurement.unavailableSourceCount + measurement.failedSourceCount === 0
    || measurement.completedSourceCount === measurement.sourceUniverse.length
  )) fail(endpoint, "$.measurement", "partial must contain incomplete measurement evidence");

  const sources = new Map(reader.sources.map((source) => [source.sourceId, source]));
  if (["not_measured", "partial", "completed"].includes(reader.status)) {
    for (const sourceId of measurement.sourceUniverse) {
      if (!sources.has(sourceId)) fail(endpoint, "$.measurement.source_universe", `references missing source ${sourceId}`);
    }
    const actual = { measured: 0, unavailable: 0, failed: 0, not_measured: 0 };
    for (const sourceId of measurement.sourceUniverse) actual[sources.get(sourceId)!.measurement.status] += 1;
    if (
      actual.measured !== measurement.completedSourceCount
      || actual.unavailable !== measurement.unavailableSourceCount
      || actual.failed !== measurement.failedSourceCount
      || actual.not_measured !== measurement.notMeasuredSourceCount
    ) fail(endpoint, "$.measurement", "source counts do not match source measurement states");
  }

  const armIds: string[] = [];
  const interventionObservationIds: string[] = [];
  reader.sources.forEach((source, index) => {
    const path = `$.sources[${index}]`;
    const sourceMeasurement = source.measurement;
    if (sourceMeasurement.experimentId !== null && sourceMeasurement.experimentId !== measurement.experimentId) {
      fail(endpoint, `${path}.measurement.experiment_id`, "does not match the reader experiment");
    }
    if (sourceMeasurement.status !== "not_measured" && !measurement.sourceUniverse.includes(source.sourceId)) {
      const isUnmeasurableDisplaySource = reader.status === "unavailable"
        && reader.capabilities.contextEffects === "unavailable"
        && measurement.sourceUniverse.length === 0
        && sourceMeasurement.status === "unavailable";
      if (!isUnmeasurableDisplaySource) {
        fail(endpoint, `${path}.measurement.status`, "claims evidence outside the measurement source universe");
      }
      if (
        sourceMeasurement.experimentId !== null
        || sourceMeasurement.armId !== null
        || sourceMeasurement.observationId !== null
        || sourceMeasurement.intervention !== undefined
      ) fail(endpoint, `${path}.measurement`, "an unmeasurable display source must not claim experiment evidence");
    }
    if (sourceMeasurement.intervention) {
      const ids = sourceMeasurement.intervention.target.sourceIds;
      if (ids.length !== 1 || ids[0] !== source.sourceId) fail(endpoint, `${path}.measurement.intervention`, "must delete exactly this canonical source");
    }
    if (sourceMeasurement.status === "measured") {
      if (!sourceMeasurement.experimentId || !sourceMeasurement.armId || !sourceMeasurement.observationId || !sourceMeasurement.intervention) {
        fail(endpoint, `${path}.measurement`, "measured evidence requires experiment, arm, observation, and intervention identity");
      }
      if (source.summary.maxAbsTokenDeltaNats === null || source.summary.totalContinuationDeltaNats === null) {
        fail(endpoint, `${path}.summary`, "measured evidence requires numeric summaries");
      }
    } else if (
      source.loci.length !== 0 || source.summary.aboveFloorLocusCount !== 0
      || source.summary.maxAbsTokenDeltaNats !== null || source.summary.totalContinuationDeltaNats !== null
    ) fail(endpoint, path, "nonmeasurement states must not carry measured locus summaries");
    if (source.summary.aboveFloorLocusCount !== source.loci.length) {
      fail(endpoint, `${path}.summary.above_floor_locus_count`, "must match the source locus references");
    }
    if (sourceMeasurement.armId) armIds.push(sourceMeasurement.armId);
    if (sourceMeasurement.observationId) interventionObservationIds.push(sourceMeasurement.observationId);
  });
  assertUnique(armIds, endpoint, "$.sources[].measurement.arm_id");
  assertUnique(interventionObservationIds, endpoint, "$.sources[].measurement.observation_id");
  if (measurement.baselineObservationId && interventionObservationIds.includes(measurement.baselineObservationId)) {
    fail(endpoint, "$.measurement.baseline_observation_id", "must be distinct from intervention observations");
  }

  const locusIds = reader.loci.map((locus) => locus.locusId);
  assertUnique(locusIds, endpoint, "$.loci[].locus_id");
  const referencedLoci = reader.sources.flatMap((source) => source.loci);
  assertUnique(referencedLoci, endpoint, "$.sources[].loci");
  if (locusIds.length !== referencedLoci.length || locusIds.some((id) => !referencedLoci.includes(id))) {
    fail(endpoint, "$.sources[].loci", "must reference every projected locus exactly once");
  }

  reader.loci.forEach((locus, index) => {
    const path = `$.loci[${index}]`;
    const source = sources.get(locus.sourceId);
    if (!source || source.measurement.status !== "measured" || !source.loci.includes(locus.locusId)) {
      fail(endpoint, `${path}.source_id`, "does not resolve to its measured source registration");
    }
    const classification = locus.direction === "support" ? "measured_support" : "measured_suppression";
    if (locus.classification !== classification) fail(endpoint, `${path}.classification`, "does not match its direction");
    if (!answerAvailable) fail(endpoint, path, "cannot resolve without available answer evidence");
    if (answer.status !== "available") fail(endpoint, path, "cannot resolve without available answer evidence");
    const [tokenStart, tokenEnd] = locus.answerTokenRange;
    if (tokenEnd > answer.tokens.length) fail(endpoint, `${path}.answer_token_range`, "extends beyond the recorded token sequence");
    const selectedTokens = answer.tokens.slice(tokenStart, tokenEnd);
    const expectedUnicode: UnicodeRange = [selectedTokens[0].unicodeRange[0], selectedTokens[selectedTokens.length - 1].unicodeRange[1]];
    if (!rangesEqual(locus.answerUnicodeRange, expectedUnicode)) fail(endpoint, `${path}.answer_unicode_range`, "does not match its recorded token range");
    if (
      locus.answerText !== codePointSlice(answer.text, expectedUnicode[0], expectedUnicode[1])
      || locus.answerText !== selectedTokens.map((token) => token.piece).join("")
    ) fail(endpoint, `${path}.answer_text`, "does not match its recorded answer coordinates");
    if (locus.perTokenDeltaNats.length !== tokenEnd - tokenStart) fail(endpoint, `${path}.per_token_delta_nats`, "must align one-to-one with the token range");
    const expectedSign = locus.direction === "support" ? 1 : -1;
    if (locus.deltaNats * expectedSign <= 0 || locus.perTokenDeltaNats.some((value) => value * expectedSign <= reader.presentation.measurementFloorNats)) {
      fail(endpoint, path, "direction and per-token deltas must clear the configured measurement floor");
    }
    const sum = locus.perTokenDeltaNats.reduce((total, value) => total + value, 0);
    if (!nearlyEqual(sum, locus.deltaNats)) fail(endpoint, `${path}.delta_nats`, "must equal the per-token delta sum");
    if (!nearlyEqual(locus.absoluteDeltaNats, Math.abs(locus.deltaNats))) fail(endpoint, `${path}.absolute_delta_nats`, "must equal the absolute signed delta");
    if (!nearlyEqual(locus.baselineSelectedLogp - locus.intervenedSelectedLogp, locus.deltaNats)) {
      fail(endpoint, `${path}.delta_nats`, "must agree with the selected baseline and intervention log probabilities");
    }
    if (locus.experimentId !== measurement.experimentId || locus.experimentId !== source.measurement.experimentId) {
      fail(endpoint, `${path}.experiment_id`, "does not match its source experiment");
    }
    if (locus.armId !== source.measurement.armId) fail(endpoint, `${path}.arm_id`, "does not match its source arm");
    if (locus.interventionObservationId !== source.measurement.observationId) {
      fail(endpoint, `${path}.intervention_observation_id`, "does not match its source observation");
    }
    if (!measurement.baselineObservationId || locus.baselineObservationId !== measurement.baselineObservationId) {
      fail(endpoint, `${path}.baseline_observation_id`, "does not match the measurement baseline");
    }
    if (answer.observationId !== locus.baselineObservationId) {
      fail(endpoint, `${path}.baseline_observation_id`, "does not match the answer score evidence");
    }
  });
}

export function decodeContextInvestigationReader(
  value: unknown,
  endpoint = "/runs/<id>/context-investigation/reader",
  expectedRunId?: string,
): ContextInvestigationReader {
  const row = record(value, endpoint, "$");
  const reader: ContextInvestigationReader = {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.context-investigation-reader.v1"] as const, endpoint, "$.schema_version"),
    runId: nonEmptyText(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id"),
    status: oneOf(required(row, "status", endpoint, "$"), READER_STATUSES, endpoint, "$.status"),
    reason: optional(row, "reason", (item, itemPath) => nonEmptyText(item, endpoint, itemPath), "$"),
    reasonCode: optional(row, "reason_code", (item, itemPath) => nonEmptyText(item, endpoint, itemPath), "$"),
    measurement: decodeMeasurement(required(row, "measurement", endpoint, "$"), endpoint, "$.measurement"),
    context: decodeContext(required(row, "context", endpoint, "$"), endpoint, "$.context"),
    answer: decodeAnswer(required(row, "answer", endpoint, "$"), endpoint, "$.answer"),
    sources: list(required(row, "sources", endpoint, "$"), endpoint, "$.sources").map((item, index) => decodeSource(item, endpoint, `$.sources[${index}]`)),
    loci: list(required(row, "loci", endpoint, "$"), endpoint, "$.loci").map((item, index) => decodeLocus(item, endpoint, `$.loci[${index}]`)),
    presentation: decodePresentation(required(row, "presentation", endpoint, "$"), endpoint, "$.presentation"),
    capabilities: decodeCapabilities(required(row, "capabilities", endpoint, "$"), endpoint, "$.capabilities"),
  };
  if (expectedRunId !== undefined && reader.runId !== expectedRunId) fail(endpoint, "$.run_id", `must match requested Run ${expectedRunId}`);
  validateSourceCoordinates(reader, endpoint);
  validateReaderEvidence(reader, endpoint);
  return reader;
}

function decodeSelection(value: unknown, endpoint: string, path: string): ContextInvestigationSelection {
  const row = record(value, endpoint, path);
  const unicodeRange = range(required(row, "unicode_range", endpoint, path), endpoint, `${path}.unicode_range`, true);
  const tokenRange = range(required(row, "token_range", endpoint, path), endpoint, `${path}.token_range`, true);
  const selectedText = text(required(row, "text", endpoint, path), endpoint, `${path}.text`);
  const tokenIds = list(required(row, "token_ids", endpoint, path), endpoint, `${path}.token_ids`)
    .map((item, index) => nonNegativeInteger(item, endpoint, `${path}.token_ids[${index}]`));
  const tokenPieces = list(required(row, "token_pieces", endpoint, path), endpoint, `${path}.token_pieces`)
    .map((item, index) => text(item, endpoint, `${path}.token_pieces[${index}]`));
  if (codePointLength(selectedText) !== unicodeRange[1] - unicodeRange[0]) fail(endpoint, `${path}.text`, "length must match its Unicode code-point range");
  if (tokenIds.length !== tokenPieces.length || tokenIds.length !== tokenRange[1] - tokenRange[0]) {
    fail(endpoint, path, "token IDs and pieces must align with the token range");
  }
  if (codePointLength(tokenPieces.join("")) < codePointLength(selectedText)) fail(endpoint, `${path}.token_pieces`, "cannot cover the selected text");
  return { unicodeRange, tokenRange, text: selectedText, tokenIds, tokenPieces };
}

function decodeEffect(value: unknown, endpoint: string, path: string): ContextInvestigationEffect {
  const row = record(value, endpoint, path);
  return {
    sourceId: nonEmptyText(required(row, "source_id", endpoint, path), endpoint, `${path}.source_id`),
    deltaNats: nullableFinite(required(row, "delta_nats", endpoint, path), endpoint, `${path}.delta_nats`),
    direction: oneOf(
      required(row, "direction", endpoint, path),
      ["support", "suppression", "below_floor", "not_measured", "unavailable", "failed"] as const,
      endpoint,
      `${path}.direction`,
    ),
    classification: required(row, "classification", endpoint, path) === null
      ? null
      : oneOf(row.classification, ["measured_support", "measured_suppression", "below_floor"] as const, endpoint, `${path}.classification`),
    measurementStatus: oneOf(required(row, "measurement_status", endpoint, path), SOURCE_STATUSES, endpoint, `${path}.measurement_status`),
    experimentId: nonEmptyText(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    armId: nonEmptyText(required(row, "arm_id", endpoint, path), endpoint, `${path}.arm_id`),
    baselineObservationId: nullableNonEmptyText(required(row, "baseline_observation_id", endpoint, path), endpoint, `${path}.baseline_observation_id`),
    interventionObservationId: nullableNonEmptyText(required(row, "intervention_observation_id", endpoint, path), endpoint, `${path}.intervention_observation_id`),
  };
}

function validateQuery(query: ContextInvestigationQuery, endpoint: string): void {
  if (["stale", "unavailable"].includes(query.status) && (!query.reason || !query.reasonCode)) {
    fail(endpoint, "$", "must include a reason and reason_code for stale or unavailable evidence");
  }
  if (["not_measured", "stale", "unavailable"].includes(query.status)) {
    if (query.selection !== null || query.effects.length !== 0) fail(endpoint, "$", "nonmeasurement query states must not carry a selection or effects");
    return;
  }
  if (query.selection === null) fail(endpoint, "$.selection", "is required for completed or partial query evidence");

  assertUnique(query.effects.map((effect) => effect.sourceId), endpoint, "$.effects[].source_id");
  assertUnique(query.effects.map((effect) => effect.armId), endpoint, "$.effects[].arm_id");
  const observations = query.effects.flatMap((effect) => effect.interventionObservationId ? [effect.interventionObservationId] : []);
  assertUnique(observations, endpoint, "$.effects[].intervention_observation_id");
  const experiments = new Set(query.effects.map((effect) => effect.experimentId));
  if (experiments.size > 1) fail(endpoint, "$.effects[].experiment_id", "must identify one canonical experiment");
  const baselines = new Set(query.effects.map((effect) => effect.baselineObservationId));
  if (baselines.size > 1 || baselines.has(null)) fail(endpoint, "$.effects[].baseline_observation_id", "must identify one measured baseline");
  if (query.effects.some((effect) => effect.baselineObservationId !== null && effect.baselineObservationId === effect.interventionObservationId)) {
    fail(endpoint, "$.effects[].intervention_observation_id", "must be distinct from the measured baseline");
  }

  query.effects.forEach((effect, index) => {
    const path = `$.effects[${index}]`;
    if (effect.measurementStatus === "measured") {
      if (effect.deltaNats === null || effect.interventionObservationId === null) fail(endpoint, path, "measured effects require a delta and intervention observation");
      const expectedClassification = effect.direction === "support"
        ? "measured_support"
        : effect.direction === "suppression" ? "measured_suppression" : effect.direction === "below_floor" ? "below_floor" : null;
      if (expectedClassification === null || effect.classification !== expectedClassification) fail(endpoint, `${path}.classification`, "does not match the measured direction");
      if (
        (effect.direction === "support" && effect.deltaNats <= query.measurementFloorNats)
        || (effect.direction === "suppression" && effect.deltaNats >= -query.measurementFloorNats)
        || (effect.direction === "below_floor" && Math.abs(effect.deltaNats) > query.measurementFloorNats)
      ) fail(endpoint, `${path}.delta_nats`, "does not match its direction at the configured floor");
    } else if (
      effect.deltaNats !== null || effect.classification !== null || effect.direction !== effect.measurementStatus
    ) fail(endpoint, path, "nonmeasurement effects must not imply a measured direction or delta");
  });
  const allMeasured = query.effects.every((effect) => effect.measurementStatus === "measured");
  if ((query.status === "completed") !== allMeasured) fail(endpoint, "$.status", "must reflect whether every returned effect is measured");
}

export function decodeContextInvestigationQuery(
  value: unknown,
  endpoint = "/runs/<id>/context-investigation/query",
): ContextInvestigationQuery {
  const row = record(value, endpoint, "$");
  const rawSelection = required(row, "selection", endpoint, "$" );
  const query: ContextInvestigationQuery = {
    schemaVersion: oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.context-investigation-query.v1"] as const, endpoint, "$.schema_version"),
    status: oneOf(required(row, "status", endpoint, "$"), READER_STATUSES, endpoint, "$.status"),
    reason: optional(row, "reason", (item, itemPath) => nonEmptyText(item, endpoint, itemPath), "$"),
    reasonCode: optional(row, "reason_code", (item, itemPath) => nonEmptyText(item, endpoint, itemPath), "$"),
    selection: rawSelection === null ? null : decodeSelection(rawSelection, endpoint, "$.selection"),
    effects: list(required(row, "effects", endpoint, "$"), endpoint, "$.effects").map((item, index) => decodeEffect(item, endpoint, `$.effects[${index}]`)),
    measurementFloorNats: nonNegative(required(row, "measurement_floor_nats", endpoint, "$"), endpoint, "$.measurement_floor_nats"),
  };
  validateQuery(query, endpoint);
  return query;
}
