/**
 * Client for `GET /runs/<id>/context-receipt` (feature 06). Deliberately its OWN module rather than an
 * addition to `data/api.ts`/`data/types.ts` -- those are shared across every Studio surface, and this
 * feature only needs one endpoint, consumed from exactly one slot panel. Keeping it self-contained means
 * no edit to a file another feature might also be mid-edit on.
 *
 * `clozn.runs.context_receipt.read_receipt()` reports one of four shapes and this module preserves that
 * distinction end to end -- collapsing "legacy" or "unrecognized" into "absent" would hide a real
 * difference (see clozn/runs/context_receipt.py's module docstring, "SHAPE HISTORY").
 */

export type ReceiptShape = "new" | "legacy" | "absent" | "unrecognized";

export interface ReceiptSegment {
  segmentId?: string;
  sourceType?: string;
  sourceLabel?: string;
  originalOrder?: number;
  deliveredBytes?: number;
  deliveredTokens?: number;
  /** Did this segment survive from delivered into assembled. Absent (not `false`) means "not recorded". */
  included?: boolean;
  contentHash?: string;
  reason?: string;
  redactionState?: string;
}

export interface ReceiptOmission {
  segmentId?: string;
  reason?: string;
  detail?: string;
}

export interface ReceiptTransformation {
  reason?: string;
  segmentIds: string[];
  detail?: string;
}

export interface ReceiptTermination {
  reason?: string;
  reasonRaw?: string;
  generatedTokens?: number;
}

export interface ReceiptRendered {
  sha256?: string;
  tokenCount?: number;
  /** Present only when the backend has an explicit opinion. Absent = the backend has not said either way. */
  estimated?: boolean;
  specialTokens?: string[];
}

export interface LegacyLimits {
  promptTokens?: number;
  contextWindowTokens?: number;
  requestedMaxTokens?: number;
  generatedTokens?: number;
}

export interface LegacyWarning {
  code: string;
  severity?: string;
  message?: string;
  requestedMaxTokens?: number;
}

export interface NewReceipt {
  schemaVersion: string;
  runId: string;
  templateFingerprint?: string;
  tokenizerConflatedWithTemplate?: boolean;
  contextWindowTokens?: number;
  reservedOutputTokens?: number;
  delivered: ReceiptSegment[];
  assembled: ReceiptSegment[];
  rendered?: ReceiptRendered;
  omissions: ReceiptOmission[];
  transformations: ReceiptTransformation[];
  termination?: ReceiptTermination;
  privacy?: string;
  schemaValidationError?: string;
  // Additive, legacy-shaped fields carried unchanged onto every new-schema document too (see
  // clozn/runs/context_receipt.py's module docstring). Full message/prompt TEXT, unlike the segment
  // arrays above -- gate these behind an explicit reveal wherever the UI shows them.
  limits: LegacyLimits;
  outputCutOff?: boolean;
  legacyFinalPrompt?: string;
  legacyAssembledMessages?: Array<{ role: string; content?: string }>;
  contentWithheldByPrivacyTier?: string;
  contentWithheldByRequest?: string;
}

export interface LegacyReceipt {
  deliveredMessages: Array<{ role: string; content?: string }>;
  assembledMessages: Array<{ role: string; content?: string }>;
  finalPrompt?: string;
  inputTruncated?: boolean;
  inputPolicy?: string;
  outputCutOff?: boolean;
  limits: LegacyLimits;
  warnings: LegacyWarning[];
  contentWithheldByRequest?: string;
}

export type ReceiptView =
  | { shape: "absent" }
  | { shape: "unrecognized"; rawKeys: string[]; schemaVersionRaw?: string }
  | { shape: "new"; receipt: NewReceipt }
  | { shape: "legacy"; receipt: LegacyReceipt };

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : {};
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

function legacyMessage(raw: unknown): { role: string; content?: string } {
  const message = record(raw);
  return { role: str(message.role) ?? "unknown", content: typeof message.content === "string" ? message.content : undefined };
}

function parseSegment(raw: unknown): ReceiptSegment {
  const segment = record(raw);
  return {
    segmentId: str(segment.segment_id),
    sourceType: str(segment.source_type),
    sourceLabel: str(segment.source_label),
    originalOrder: num(segment.original_order),
    deliveredBytes: num(segment.delivered_bytes),
    deliveredTokens: num(segment.delivered_tokens),
    included: bool(segment.included),
    contentHash: str(segment.content_hash),
    reason: str(segment.reason),
    redactionState: str(segment.redaction_state),
  };
}

function parseOmission(raw: unknown): ReceiptOmission {
  const omission = record(raw);
  return { segmentId: str(omission.segment_id), reason: str(omission.reason), detail: str(omission.detail) };
}

function parseTransformation(raw: unknown): ReceiptTransformation {
  const transformation = record(raw);
  return {
    reason: str(transformation.reason),
    segmentIds: Array.isArray(transformation.segment_ids) ? transformation.segment_ids.map(String) : [],
    detail: str(transformation.detail),
  };
}

function parseTermination(raw: unknown): ReceiptTermination | undefined {
  const termination = record(raw);
  if (!("reason" in termination)) return undefined;
  return {
    reason: str(termination.reason),
    reasonRaw: str(termination.reason_raw),
    generatedTokens: num(termination.generated_tokens),
  };
}

function parseRendered(raw: unknown): ReceiptRendered | undefined {
  const rendered = record(raw);
  if (!Object.keys(rendered).length) return undefined;
  return {
    sha256: str(rendered.sha256),
    tokenCount: num(rendered.token_count),
    estimated: bool(rendered.estimated),
    specialTokens: Array.isArray(rendered.special_tokens) ? rendered.special_tokens.map(String) : undefined,
  };
}

function parseLimits(raw: unknown): LegacyLimits {
  const limits = record(raw);
  return {
    promptTokens: num(limits.prompt_tokens),
    contextWindowTokens: num(limits.context_window_tokens),
    requestedMaxTokens: num(limits.requested_max_tokens),
    generatedTokens: num(limits.generated_tokens),
  };
}

function parseWarning(raw: unknown): LegacyWarning {
  const warning = record(raw);
  return {
    code: str(warning.code) ?? "unknown",
    severity: str(warning.severity),
    message: str(warning.message),
    requestedMaxTokens: num(warning.requested_max_tokens),
  };
}

export async function loadContextReceipt(runId: string, signal?: AbortSignal): Promise<ReceiptView> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/context-receipt`, { signal });
  if (!response.ok) throw new Error(`Context receipt request failed (${response.status})`);
  const body = record(await response.json());
  const shape = body.shape;
  const receiptRaw = record(body.context_receipt);

  if (shape === "absent") return { shape: "absent" };

  if (shape === "legacy") {
    const delivered = record(receiptRaw.delivered);
    const survived = record(receiptRaw.survived);
    return {
      shape: "legacy",
      receipt: {
        deliveredMessages: records(delivered.messages).map(legacyMessage),
        assembledMessages: records(survived.assembled_messages).map(legacyMessage),
        finalPrompt: str(survived.final_prompt),
        inputTruncated: bool(receiptRaw.input_truncated),
        inputPolicy: str(receiptRaw.input_policy),
        outputCutOff: bool(receiptRaw.output_cut_off),
        limits: parseLimits(receiptRaw.limits),
        warnings: records(receiptRaw.warnings).map(parseWarning),
        contentWithheldByRequest: str(delivered.content_withheld_by_request) ?? str(survived.content_withheld_by_request),
      },
    };
  }

  if (shape === "new") {
    const survived = record(receiptRaw.survived);
    return {
      shape: "new",
      receipt: {
        schemaVersion: str(receiptRaw.schema_version) ?? "",
        runId: str(receiptRaw.run_id) ?? runId,
        templateFingerprint: str(receiptRaw.template_fingerprint),
        tokenizerConflatedWithTemplate: bool(receiptRaw.tokenizer_conflated_with_template),
        contextWindowTokens: num(receiptRaw.context_window_tokens),
        reservedOutputTokens: num(receiptRaw.reserved_output_tokens),
        delivered: records(receiptRaw.delivered).map(parseSegment),
        assembled: records(receiptRaw.assembled).map(parseSegment),
        rendered: parseRendered(receiptRaw.rendered),
        omissions: records(receiptRaw.omissions).map(parseOmission),
        transformations: records(receiptRaw.transformations).map(parseTransformation),
        termination: parseTermination(receiptRaw.termination),
        privacy: str(receiptRaw.privacy),
        schemaValidationError: str(receiptRaw.schema_validation_error),
        limits: parseLimits(receiptRaw.limits),
        outputCutOff: bool(receiptRaw.output_cut_off),
        legacyFinalPrompt: str(survived.final_prompt),
        legacyAssembledMessages: Array.isArray(survived.assembled_messages)
          ? records(survived.assembled_messages).map(legacyMessage)
          : undefined,
        contentWithheldByPrivacyTier: str(survived.content_withheld_by_privacy_tier),
        contentWithheldByRequest: str(survived.content_withheld_by_request),
      },
    };
  }

  return {
    shape: "unrecognized",
    rawKeys: Object.keys(receiptRaw),
    schemaVersionRaw: str(receiptRaw.schema_version) ?? str(receiptRaw.schema),
  };
}
