/**
 * Narrow clients for the read-only answer-investigation receipt surface.
 *
 * Both APIs are metadata-only. Raw messages and the exact rendered prompt
 * remain owned by `data/context-receipt.ts` and its explicit disclosure UI.
 */

import type { InfluenceEvidenceState } from "./types";

export type InvestigationState =
  | "supported"
  | "measured_effect"
  | "below_measurement_floor"
  | "delivered_not_measured"
  | "omitted"
  | "unavailable"
  | "failed"
  | "inconclusive";

export type SpanResolutionState =
  | "exact"
  | "metadata_only"
  | "drifted"
  | "redacted"
  | "unavailable";

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
  reason?: string;
  privacy?: string;
  delivered: ReceivedSegment[];
  assembled: ReceivedSegment[];
  omitted: ReceivedSegment[];
  rendered?: {
    sha256?: string;
    bytes?: number;
    tokens?: number;
    tokenCount?: number;
    contentAvailable?: boolean;
    estimated?: boolean;
  };
  limits: {
    promptTokens?: number;
    contextWindowTokens?: number;
    requestedMaxTokens?: number;
    generatedTokens?: number;
  };
}

export interface RunInvestigationReceipt {
  schemaVersion: string;
  runId: string;
  receivedContext: ReceivedContextSection;
  spanSection?: {
    state: InvestigationState;
    reason?: string;
    href?: string;
    privacy?: string;
    addressCount?: number;
    influenceNativeStatus?: string;
  };
  /** `sections.prompt_source_influence` -- the cross-linked prompt-span x answer-span evidence "What
   * mattered?" is built from. Metadata-only, same as everything else this module decodes: spans carry
   * hashes, ids, and offsets, never literal text. */
  promptSourceInfluence: PromptSourceInfluenceSection;
  /** Top-level `actions[]` -- typed action descriptors (href/method/availability), never followed
   * automatically. `measure_prompt_source_influence` is the one "What mattered?" offers as a button; it
   * is read here only to decide whether/why that button is enabled, never to trigger it. */
  actions: InvestigationAction[];
}

/** One canonical prompt/answer span reference from `clozn.context_answer_influence.v1`'s portable
 * export (see `context_answer_influence.portable_export`'s `span()` helper) -- an answer span additionally
 * carries `tokenIndex`/`tokenId` (see `InfluenceAnswerSpan` below). Text is never present at
 * `metadata_only` privacy: only a SHA-256 + byte count, offsets, and native identity survive. */
export interface InfluenceSpanRef {
  id: string;
  parentId?: string;
  level?: string;
  kind?: string;
  messageIndex?: number;
  role?: string;
  name?: string;
  sourceKind?: string;
  segmentId?: string;
  clientSourceId?: string;
  sourceLabel?: string;
  externalSourceId?: string;
  /** Present on `prompt_sources` entries only: whether this source was inside the bounded span budget
   * `context_answer_influence` actually measured. `false` means the source reached the model (it is
   * already in the assembled prompt) but this particular measurement run never scored it -- a real,
   * distinct "not measured" claim, never the same as `received_context.omitted` (never reached the
   * model at all). */
  selected?: boolean;
  start?: number;
  end?: number;
  byteStart?: number;
  byteEnd?: number;
  childUnitCount?: number;
  textSha256?: string;
  textBytes?: number;
}

export interface InfluenceAnswerSpan extends InfluenceSpanRef {
  tokenIndex?: number;
  tokenId?: number;
}

export type InfluenceLinkEffect = "supports" | "suppresses" | "neutral";

/** One measured (context span, answer span) cell -- `clozn.context_answer_influence.v1`'s own `links[]`,
 * verbatim field names translated to camelCase. `evidenceState` is the SAME closed union
 * `data/types.ts` already defines (`causally_supported` | `observed`) -- this module never widens it. */
export interface InfluenceLink {
  contextSpanId: string;
  answerSpanId: string;
  contextIndex?: number;
  answerIndex?: number;
  deltaNats: number;
  absDeltaNats: number;
  effect: InfluenceLinkEffect;
  clearsFloor: boolean;
  evidenceState: InfluenceEvidenceState;
}

export interface InfluenceThresholdsSection {
  cellAbsDeltaNats?: number;
  sourceClearRule?: string;
  calibration?: string;
}

/** `sections.prompt_source_influence` (see `clozn/runs/investigation.py`'s `_influence_section`).
 * `state` never reports `causally_supported`/`observed` itself -- those live per-link in `links[]`; this
 * `state` says whether ANY measurement exists at all for this run (`measured_effect` /
 * `below_measurement_floor`) or why not (`delivered_not_measured` / `unavailable` / `failed` /
 * `inconclusive`). */
export interface PromptSourceInfluenceSection {
  state: InvestigationState;
  reason?: string;
  privacy?: string;
  promptSources: InfluenceSpanRef[];
  promptSpans: InfluenceSpanRef[];
  answerSpans: InfluenceAnswerSpan[];
  links: InfluenceLink[];
  thresholds: InfluenceThresholdsSection;
}

/** One entry of the investigation document's top-level `actions[]` -- a typed descriptor for a
 * measurement/navigation/corrective call a caller MAY make, never one this module (or any reader of it)
 * calls on its own. */
export interface InvestigationAction {
  id: string;
  label: string;
  kind: string;
  method: string;
  href: string;
  availability: string;
  reason?: string;
}

export interface SpanAddress {
  addressId: string;
  kind: string;
  relationKey?: string;
  nativeRef: {
    collection?: string;
    id?: string;
    segmentId?: string;
    clientSourceId?: string;
    sourceLabel?: string;
    selected?: boolean;
  };
  resolution: {
    state: SpanResolutionState;
    reason?: string;
    canonical?: {
      basis?: string;
      start?: number;
      end?: number;
      basisSha256?: string;
      spanSha256?: string;
      spanCodePoints?: number;
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
}

export interface SpanAddressDocument {
  schemaVersion: string;
  runId: string;
  privacy: string;
  sourceArtifacts: SpanSourceArtifact[];
  addresses: SpanAddress[];
}

/** Exported for reuse by data/claimSupport.ts -- the same tiny JSON-decoding primitives this whole
 * `src/data/` layer already repeats per-file; one shared copy for the two modules that decode the
 * closely related `clozn.text-span-addresses.v1` family. */
export type JsonRecord = Record<string, unknown>;

export function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}

export function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

export function str(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

export function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function bool(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

const INVESTIGATION_STATES = new Set<InvestigationState>([
  "supported",
  "measured_effect",
  "below_measurement_floor",
  "delivered_not_measured",
  "omitted",
  "unavailable",
  "failed",
  "inconclusive",
]);

function investigationState(value: unknown): InvestigationState {
  return typeof value === "string" && INVESTIGATION_STATES.has(value as InvestigationState)
    ? value as InvestigationState
    : "failed";
}

const SPAN_STATES = new Set<SpanResolutionState>([
  "exact",
  "metadata_only",
  "drifted",
  "redacted",
  "unavailable",
]);

function spanState(value: unknown): SpanResolutionState {
  return typeof value === "string" && SPAN_STATES.has(value as SpanResolutionState)
    ? value as SpanResolutionState
    : "unavailable";
}

function parseSegment(raw: unknown): ReceivedSegment {
  const segment = record(raw);
  return {
    segmentId: str(segment.segment_id),
    clientSourceId: str(segment.client_source_id),
    sourceType: str(segment.source_type),
    sourceLabel: str(segment.source_label),
    originalOrder: num(segment.original_order),
    deliveredBytes: num(segment.delivered_bytes),
    deliveredTokens: num(segment.delivered_tokens),
    included: bool(segment.included),
    contentHash: str(segment.content_hash),
    reason: str(segment.reason),
    detail: str(segment.detail),
    redactionState: str(segment.redaction_state),
  };
}

function parseReceivedContext(raw: unknown): ReceivedContextSection {
  const section = record(raw);
  const rendered = record(section.rendered);
  const limits = record(section.limits);
  return {
    state: investigationState(section.state),
    reason: str(section.reason),
    privacy: str(section.privacy),
    delivered: records(section.delivered).map(parseSegment),
    assembled: records(section.assembled).map(parseSegment),
    omitted: records(section.omitted).map(parseSegment),
    rendered: Object.keys(rendered).length ? {
      sha256: str(rendered.sha256),
      bytes: num(rendered.bytes),
      tokens: num(rendered.tokens),
      tokenCount: num(rendered.token_count),
      contentAvailable: bool(rendered.content_available),
      estimated: bool(rendered.estimated),
    } : undefined,
    limits: {
      promptTokens: num(limits.prompt_tokens),
      contextWindowTokens: num(limits.context_window_tokens),
      requestedMaxTokens: num(limits.requested_max_tokens),
      generatedTokens: num(limits.generated_tokens),
    },
  };
}

function parseSpanSection(raw: unknown): RunInvestigationReceipt["spanSection"] {
  const section = record(raw);
  if (!Object.keys(section).length) return undefined;
  return {
    state: investigationState(section.state),
    reason: str(section.reason),
    href: str(section.href),
    privacy: str(section.privacy),
    addressCount: num(section.address_count),
    influenceNativeStatus: str(section.influence_native_status),
  };
}

const LINK_EFFECTS = new Set<InfluenceLinkEffect>(["supports", "suppresses", "neutral"]);

function linkEffect(value: unknown): InfluenceLinkEffect | undefined {
  return typeof value === "string" && LINK_EFFECTS.has(value as InfluenceLinkEffect)
    ? value as InfluenceLinkEffect
    : undefined;
}

const INFLUENCE_EVIDENCE_STATES = new Set<InfluenceEvidenceState>(["causally_supported", "observed"]);

function influenceEvidenceState(value: unknown): InfluenceEvidenceState | undefined {
  return typeof value === "string" && INFLUENCE_EVIDENCE_STATES.has(value as InfluenceEvidenceState)
    ? value as InfluenceEvidenceState
    : undefined;
}

function parseSpanRef(raw: unknown): InfluenceSpanRef {
  const item = record(raw);
  return {
    id: str(item.id) ?? "",
    parentId: str(item.parent_id),
    level: str(item.level),
    kind: str(item.kind),
    messageIndex: num(item.message_index),
    role: str(item.role),
    name: str(item.name),
    sourceKind: str(item.source_kind),
    segmentId: str(item.segment_id),
    clientSourceId: str(item.client_source_id),
    sourceLabel: str(item.source_label),
    externalSourceId: str(item.external_source_id),
    selected: bool(item.selected),
    start: num(item.start),
    end: num(item.end),
    byteStart: num(item.byte_start),
    byteEnd: num(item.byte_end),
    childUnitCount: num(item.child_unit_count),
    textSha256: str(item.text_sha256),
    textBytes: num(item.text_bytes),
  };
}

function parseAnswerSpan(raw: unknown): InfluenceAnswerSpan {
  const item = record(raw);
  return {
    ...parseSpanRef(raw),
    tokenIndex: num(item.token_index),
    tokenId: num(item.token_id),
  };
}

/** `null` when a link is missing a field this module's closed unions require -- dropped rather than
 * guessed (a cell with no link renders as NOT MEASURED, which is the honest fallback; see
 * WhatMattered.tsx's cell-state builder). Never upgrades an unrecognized `evidence_state` to a known one. */
function parseLink(raw: unknown): InfluenceLink | null {
  const item = record(raw);
  const contextSpanId = str(item.context_span_id);
  const answerSpanId = str(item.answer_span_id);
  const deltaNats = num(item.delta_nats);
  const absDeltaNats = num(item.abs_delta_nats);
  const effect = linkEffect(item.effect);
  const evidenceState = influenceEvidenceState(item.evidence_state);
  const clearsFloor = bool(item.clears_floor);
  if (
    !contextSpanId || !answerSpanId || deltaNats == null || absDeltaNats == null
    || !effect || !evidenceState || clearsFloor == null
  ) {
    return null;
  }
  return {
    contextSpanId,
    answerSpanId,
    contextIndex: num(item.context_index),
    answerIndex: num(item.answer_index),
    deltaNats,
    absDeltaNats,
    effect,
    clearsFloor,
    evidenceState,
  };
}

function parseThresholds(raw: unknown): InfluenceThresholdsSection {
  const item = record(raw);
  return {
    cellAbsDeltaNats: num(item.cell_abs_delta_nats),
    sourceClearRule: str(item.source_clear_rule),
    calibration: str(item.calibration),
  };
}

function parsePromptSourceInfluence(raw: unknown): PromptSourceInfluenceSection {
  const item = record(raw);
  return {
    state: investigationState(item.state),
    reason: str(item.reason),
    privacy: str(item.privacy),
    promptSources: records(item.prompt_sources).map(parseSpanRef),
    promptSpans: records(item.prompt_spans).map(parseSpanRef),
    answerSpans: records(item.answer_spans).map(parseAnswerSpan),
    links: records(item.links)
      .map(parseLink)
      .filter((link): link is InfluenceLink => link != null),
    thresholds: parseThresholds(item.thresholds),
  };
}

function parseAction(raw: unknown): InvestigationAction | null {
  const item = record(raw);
  const id = str(item.id);
  const href = str(item.href);
  const method = str(item.method);
  const availability = str(item.availability);
  if (!id || !href || !method || !availability) return null;
  return {
    id,
    label: str(item.label) ?? id,
    kind: str(item.kind) ?? "action",
    method,
    href,
    availability,
    reason: str(item.reason),
  };
}

function parseInvestigation(raw: unknown, requestedRunId: string): RunInvestigationReceipt {
  const document = record(raw);
  const sections = record(document.sections);
  return {
    schemaVersion: str(document.schema_version) ?? "",
    runId: str(document.run_id) ?? requestedRunId,
    receivedContext: parseReceivedContext(sections.received_context),
    spanSection: parseSpanSection(sections.text_span_addresses),
    promptSourceInfluence: parsePromptSourceInfluence(sections.prompt_source_influence),
    actions: records(document.actions)
      .map(parseAction)
      .filter((action): action is InvestigationAction => action != null),
  };
}

/** Exported for reuse by data/claimSupport.ts, which decodes the SAME `clozn.text-span-addresses.v1`
 * address shape for `kind: "claim"` -- one parser for the one wire shape, never a second copy per kind. */
export function parseAddress(raw: unknown): SpanAddress | null {
  const address = record(raw);
  const nativeRef = record(address.native_ref);
  const resolution = record(address.resolution);
  const canonical = record(resolution.canonical);
  const addressId = str(address.address_id);
  if (!addressId) return null;
  return {
    addressId,
    kind: str(address.kind) ?? "unknown",
    relationKey: str(address.relation_key),
    nativeRef: {
      collection: str(nativeRef.collection),
      id: str(nativeRef.id),
      segmentId: str(nativeRef.segment_id),
      clientSourceId: str(nativeRef.client_source_id),
      sourceLabel: str(nativeRef.source_label),
      selected: bool(nativeRef.selected),
    },
    resolution: {
      state: spanState(resolution.state),
      reason: str(resolution.reason),
      canonical: Object.keys(canonical).length ? {
        basis: str(canonical.basis),
        start: num(canonical.start),
        end: num(canonical.end),
        basisSha256: str(canonical.basis_sha256),
        spanSha256: str(canonical.span_sha256),
        spanCodePoints: num(canonical.span_code_points),
        spanUtf8Bytes: num(canonical.span_utf8_bytes),
      } : undefined,
    },
  };
}

function parseSourceArtifact(raw: unknown): SpanSourceArtifact | null {
  const artifact = record(raw);
  const schema = str(artifact.schema);
  if (!schema) return null;
  return {
    schema,
    nativeStatus: str(artifact.native_status),
    available: bool(artifact.available),
    privacy: str(artifact.privacy),
    reason: str(artifact.reason),
    artifactSha256: str(artifact.artifact_sha256),
  };
}

function parseSpanDocument(raw: unknown, requestedRunId: string): SpanAddressDocument {
  const document = record(raw);
  return {
    schemaVersion: str(document.schema_version) ?? "",
    runId: str(document.run_id) ?? requestedRunId,
    privacy: str(document.privacy) ?? "metadata_only",
    sourceArtifacts: records(document.source_artifacts)
      .map(parseSourceArtifact)
      .filter((item): item is SpanSourceArtifact => item != null),
    addresses: records(document.addresses)
      .map(parseAddress)
      .filter((item): item is SpanAddress => item != null),
  };
}

async function getJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export async function loadRunInvestigation(
  runId: string,
  signal?: AbortSignal,
): Promise<RunInvestigationReceipt> {
  const raw = await getJson(
    `/runs/${encodeURIComponent(runId)}/investigation`,
    signal,
  );
  return parseInvestigation(raw, runId);
}

export async function loadSpanAddresses(
  runId: string,
  signal?: AbortSignal,
): Promise<SpanAddressDocument> {
  const raw = await getJson(
    `/runs/${encodeURIComponent(runId)}/span-addresses`,
    signal,
  );
  return parseSpanDocument(raw, runId);
}
