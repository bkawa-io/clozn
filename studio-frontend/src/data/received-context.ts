/**
 * Narrow clients for the read-only answer-investigation receipt surface.
 *
 * Both APIs are metadata-only. Raw messages and the exact rendered prompt
 * remain owned by `data/context-receipt.ts` and its explicit disclosure UI.
 */

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

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function num(value: unknown): number | undefined {
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

function parseInvestigation(raw: unknown, requestedRunId: string): RunInvestigationReceipt {
  const document = record(raw);
  const sections = record(document.sections);
  return {
    schemaVersion: str(document.schema_version) ?? "",
    runId: str(document.run_id) ?? requestedRunId,
    receivedContext: parseReceivedContext(sections.received_context),
    spanSection: parseSpanSection(sections.text_span_addresses),
  };
}

function parseAddress(raw: unknown): SpanAddress | null {
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
