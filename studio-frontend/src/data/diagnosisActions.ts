import { HttpError } from "./client";
import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export const CORRECTIVE_ACTION_IDS = [
  "less-verbose",
  "more-concrete",
  "use-context",
  "ask-before-guessing",
  "preserve-formatting",
  "stop-repeating",
] as const;

export type CorrectiveActionId = (typeof CORRECTIVE_ACTION_IDS)[number];
export type DiagnosisFindingStatus =
  | "finding"
  | "not_observed"
  | "unavailable"
  | "pending"
  | "suppressed";
export type DiagnosisSeverity = "info" | "low" | "medium" | "high";
export type DiagnosisConfidence = "exact" | "pattern_match" | "derived";
export type DiagnosisSuggestedActionKind =
  | "resend_context"
  | "increase_context_budget"
  | "reconcile_conflicting_instructions"
  | "deduplicate_instructions"
  | "deduplicate_source_content"
  | "clarify_output_format"
  | "move_instruction_near_request"
  | "reinforce_low_effect_source"
  | "resupply_below_floor_source"
  | "restate_conversation_instruction"
  | "increase_max_tokens"
  | "reconfirm_run_configuration"
  | "no_action_available";

/** Descriptive diagnosis text only. These kinds are not corrective action IDs. */
export interface DiagnosisSuggestion {
  kind: DiagnosisSuggestedActionKind;
  description: string;
}

export type DiagnosisEvidence =
  | {
      kind: "field";
      path: string;
      value?: unknown;
      note?: string;
    }
  | {
      kind: "text_span";
      addressId: string;
      messageIndex?: number;
      localStart?: number;
      localEnd?: number;
      note?: string;
    };

interface DiagnosisFindingBase {
  ruleId: string;
  ruleName: string;
  summary: string;
  evidence: readonly DiagnosisEvidence[];
  limitations: readonly string[];
}

export type DiagnosisFinding =
  | (DiagnosisFindingBase & {
      status: "finding";
      severity: DiagnosisSeverity;
      confidence: DiagnosisConfidence;
      suggestedActions: readonly DiagnosisSuggestion[];
    })
  | (DiagnosisFindingBase & {
      status: "not_observed" | "unavailable" | "pending" | "suppressed";
    });

export interface DiagnosisFindings {
  schemaVersion: "clozn.diagnosis-findings.v1";
  generatedAt: string;
  runId: string;
  comparisonRunId?: string;
  redacted: boolean;
  ruleRegistry: readonly { ruleId: string; ruleName: string }[];
  suppressedRuleIds: readonly string[];
  findings: readonly DiagnosisFinding[];
  statusCounts: Readonly<Record<DiagnosisFindingStatus, number>>;
}

export type DiagnosisNarrativeEvidence =
  | { kind: "finding"; ruleId: string }
  | { kind: "diff_field"; dimension: string }
  | { kind: "text_span"; addressId: string };

export interface DiagnosisObservedChange {
  dimension: string;
  kind: "changed" | "added" | "removed" | "unavailable" | "diff_failed";
  text: string;
  evidence: readonly DiagnosisNarrativeEvidence[];
}

export interface DiagnosisMeasuredEffect {
  rank: number;
  basis: "intervention" | "rule_finding";
  ruleId: string;
  severity: DiagnosisSeverity;
  confidence: DiagnosisConfidence;
  text: string;
  evidence: readonly DiagnosisNarrativeEvidence[];
}

export interface DiagnosisPlausibleNote {
  text: string;
  note: string;
  evidence: readonly DiagnosisNarrativeEvidence[];
}

export interface DiagnosisNarrative {
  schemaVersion: "clozn.diagnosis-narrative.v1";
  generatedAt: string;
  runId: string;
  comparisonRunId?: string;
  comparisonAvailable: boolean;
  findingsSchemaVersion: "clozn.diagnosis-findings.v1";
  headline: string;
  observedChanges: readonly DiagnosisObservedChange[];
  measuredEffects: readonly DiagnosisMeasuredEffect[];
  plausibleButUnproven: readonly DiagnosisPlausibleNote[];
}

export interface DiagnosisDocument {
  findings: DiagnosisFindings;
  narrative: DiagnosisNarrative;
}

export interface CorrectiveScopeEligibility {
  scope: "once";
  available: boolean;
  target?: string;
  priorHash?: string;
  before?: readonly string[];
  after?: readonly string[];
  note?: string;
  unavailabilityReason?: string;
}

export type CorrectiveBackendDeclaration =
  | {
      type: "prompt_policy";
      recipeVersion: string;
      parameters: Readonly<{ preset: CorrectiveActionId }>;
      qualification: "generic";
      qualificationId: string;
      available: true;
    }
  | {
      type: "unsupported";
      requestedType: "sampling_policy";
      available: false;
      reason: string;
      unavailabilityReason: string;
    };

export interface CorrectiveAction {
  id: CorrectiveActionId;
  label: string;
  description: string;
  conflicts: readonly CorrectiveActionId[];
  scopes: readonly ["once"];
  eligibility: { eligible: boolean; note?: string; reason?: string };
  evaluationMetrics: readonly string[];
  backends: readonly CorrectiveBackendDeclaration[];
  scopeEligibility: readonly [CorrectiveScopeEligibility];
}

export interface CorrectiveRegistry {
  schemaVersion: "clozn.action-registry.v1";
  version: string;
  runId: string;
  runFingerprint: string;
  actions: readonly CorrectiveAction[];
}

export type CorrectivePreviewStatus =
  | "ready"
  | "confirming"
  | "cancel_requested"
  | "cancelled"
  | "consumed"
  | "expired";

export interface CorrectivePreview {
  previewId: string;
  status: CorrectivePreviewStatus;
  createdTs: number;
  expiresTs: number;
  parentRunId: string;
  parentRunFingerprint: string;
  action: { id: CorrectiveActionId; label: string; description: string };
  execution: {
    requestedBackend: "prompt_policy";
    expectedExecutedBackend: "prompt_policy";
    expectedFallback: false;
    qualification: string;
    qualificationId: string;
  };
  scopeEligibility: readonly [CorrectiveScopeEligibility];
  comparisonContract: {
    baseline: string;
    corrected: string;
    storedOriginal: string;
  };
  resultId?: string;
}

export type CorrectiveChild =
  | { status: "success"; runId: string }
  | { status: "error"; runId?: string | null; error: { code: string; message: string } }
  | { status: "not_run" };

export interface CorrectiveRevisionSelection {
  resultId: string;
  childRunId: string;
  actionId: CorrectiveActionId;
  selectedTs: number;
}

export interface CorrectiveTransaction {
  id: string;
  scope: "once";
  target: string;
  before: CorrectiveRevisionSelection | null;
  after: CorrectiveRevisionSelection;
  createdTs: number;
  undoneTs: number | null;
}

export interface CorrectiveSourceUseComparison {
  status: "available";
  method: Readonly<JsonRecord>;
  thresholds: Readonly<JsonRecord>;
  baseline: {
    answerSpanCount: number;
    answerSpansWithClearSource: number;
    observedSourceDependenceRatio: number;
  };
  corrected: {
    answerSpanCount: number;
    answerSpansWithClearSource: number;
    observedSourceDependenceRatio: number;
  };
  deltaObservedSourceDependenceRatio: number;
  caveat: string;
}

export interface CorrectiveResult {
  schemaVersion: "clozn.corrective-retry-result.v1";
  resultId: string;
  createdTs: number;
  parentRunId: string;
  previewId: string;
  action: { id: CorrectiveActionId; label: string; description: string };
  userIntent: { actionId: CorrectiveActionId; keptScope?: "once" };
  execution: {
    requestedBackend: "prompt_policy";
    executedBackend: "prompt_policy" | null;
    fallback: boolean;
    qualification: string | null;
    qualificationId: string | null;
    identity: Readonly<JsonRecord> | null;
  };
  children: { baseline: CorrectiveChild; corrected: CorrectiveChild };
  comparison: {
    storedOriginalReply: string;
    baselineReply: string;
    correctedReply: string;
    note: string;
    changed: boolean;
  };
  metrics: Readonly<JsonRecord>;
  coherence?: { degenerate: boolean; reason: string };
  interventionObserved?: boolean;
  scopeEligibility: readonly [CorrectiveScopeEligibility];
  outcome: { status: "succeeded" | "execution_error" | "cancelled"; note?: string };
  transaction?: CorrectiveTransaction;
  sourceUseComparison?: CorrectiveSourceUseComparison;
}

export interface CorrectiveUndo {
  status: "undone";
  undoId: string;
  scope: "once";
  target: string;
}

const FINDING_STATUSES = ["finding", "not_observed", "unavailable", "pending", "suppressed"] as const;
const SEVERITIES = ["info", "low", "medium", "high"] as const;
const CONFIDENCES = ["exact", "pattern_match", "derived"] as const;
const SUGGESTED_ACTION_KINDS = [
  "resend_context", "increase_context_budget", "reconcile_conflicting_instructions",
  "deduplicate_instructions", "deduplicate_source_content", "clarify_output_format",
  "move_instruction_near_request", "reinforce_low_effect_source", "resupply_below_floor_source",
  "restate_conversation_instruction", "increase_max_tokens", "reconfirm_run_configuration",
  "no_action_available",
] as const;
const OBSERVED_KINDS = ["changed", "added", "removed", "unavailable", "diff_failed"] as const;
const PREVIEW_STATUSES = ["ready", "confirming", "cancel_requested", "cancelled", "consumed", "expired"] as const;
const DIAGNOSIS_RULES = [
  ["R01", "input_omitted_or_rejected"],
  ["R02", "context_budget_pressure"],
  ["R03", "conflicting_instructions"],
  ["R04", "duplicate_instructions"],
  ["R05", "repeated_source_content"],
  ["R06", "requested_format_absent"],
  ["R07", "instruction_far_from_request"],
  ["R08", "source_below_measurement_floor"],
  ["R09", "source_little_effect"],
  ["R10", "request_conflicts_with_earlier_instructions"],
  ["R11", "output_stopped_length"],
  ["R12", "run_to_run_drift"],
] as const;
const SHA256 = /^[0-9a-f]{64}$/;
const RULE_ID = /^R[0-9]{2}$/;
const SPAN_ID = /^span_[0-9a-f]{24}$/;
const PREVIEW_ID = /^fix_preview_[a-f0-9]{20}$/;
const RESULT_ID = /^fix_result_[a-f0-9]{20}$/;
const IDEMPOTENCY_KEY = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/;

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(endpoint, path, "must be an object");
  }
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
  for (const key of Object.keys(row)) {
    if (!allowed.has(key)) fail(endpoint, `${path}.${key}`, "is not part of the contract");
  }
}

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nonEmptyText(value: unknown, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!parsed) fail(endpoint, path, "must not be empty");
  return parsed;
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function finiteNumber(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be a finite number");
  return value;
}

function nonNegativeNumber(value: unknown, endpoint: string, path: string): number {
  const parsed = finiteNumber(value, endpoint, path);
  if (parsed < 0) fail(endpoint, path, "must be non-negative");
  return parsed;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = nonNegativeNumber(value, endpoint, path);
  if (!Number.isSafeInteger(parsed)) fail(endpoint, path, "must be an integer");
  return parsed;
}

function positiveInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = nonNegativeInteger(value, endpoint, path);
  if (parsed < 1) fail(endpoint, path, "must be positive");
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

function matchingText(value: unknown, pattern: RegExp, endpoint: string, path: string): string {
  const parsed = nonEmptyText(value, endpoint, path);
  if (!pattern.test(parsed)) fail(endpoint, path, "has an invalid format");
  return parsed;
}

function optionalText(row: JsonRecord, key: string, endpoint: string, path: string): string | undefined {
  return key in row ? nonEmptyText(row[key], endpoint, `${path}.${key}`) : undefined;
}

function textArray(value: unknown, endpoint: string, path: string): string[] {
  return array(value, endpoint, path).map((entry, index) => text(entry, endpoint, `${path}[${index}]`));
}

function decodeEvidence(value: unknown, endpoint: string, path: string): DiagnosisEvidence {
  const row = object(value, endpoint, path);
  if (row.kind === "field") {
    closed(row, ["kind", "path", "value", "note"], endpoint, path);
    return {
      kind: "field",
      path: nonEmptyText(required(row, "path", endpoint, path), endpoint, `${path}.path`),
      ...("value" in row ? { value: row.value } : {}),
      ...(optionalText(row, "note", endpoint, path) ? { note: optionalText(row, "note", endpoint, path) } : {}),
    };
  }
  if (row.kind === "text_span") {
    closed(row, ["kind", "address_id", "message_index", "local_start", "local_end", "note"], endpoint, path);
    const messageIndex = "message_index" in row
      ? nonNegativeInteger(row.message_index, endpoint, `${path}.message_index`) : undefined;
    const localStart = "local_start" in row
      ? nonNegativeInteger(row.local_start, endpoint, `${path}.local_start`) : undefined;
    const localEnd = "local_end" in row
      ? nonNegativeInteger(row.local_end, endpoint, `${path}.local_end`) : undefined;
    if ((localStart === undefined) !== (localEnd === undefined)) {
      fail(endpoint, path, "must carry local_start and local_end together");
    }
    if (localStart !== undefined && localEnd! < localStart) fail(endpoint, `${path}.local_end`, "must not precede local_start");
    const note = optionalText(row, "note", endpoint, path);
    return {
      kind: "text_span",
      addressId: matchingText(required(row, "address_id", endpoint, path), SPAN_ID, endpoint, `${path}.address_id`),
      ...(messageIndex === undefined ? {} : { messageIndex }),
      ...(localStart === undefined ? {} : { localStart, localEnd }),
      ...(note === undefined ? {} : { note }),
    };
  }
  fail(endpoint, `${path}.kind`, "must be field or text_span");
}

function decodeSuggestion(value: unknown, endpoint: string, path: string): DiagnosisSuggestion {
  const row = object(value, endpoint, path);
  closed(row, ["kind", "description"], endpoint, path);
  return {
    kind: oneOf(required(row, "kind", endpoint, path), SUGGESTED_ACTION_KINDS, endpoint, `${path}.kind`),
    description: nonEmptyText(required(row, "description", endpoint, path), endpoint, `${path}.description`),
  };
}

function decodeFinding(value: unknown, endpoint: string, path: string): DiagnosisFinding {
  const row = object(value, endpoint, path);
  const status = oneOf(required(row, "status", endpoint, path), FINDING_STATUSES, endpoint, `${path}.status`);
  const commonKeys = ["rule_id", "rule_name", "status", "summary", "evidence", "limitations"];
  closed(
    row,
    status === "finding" ? [...commonKeys, "severity", "confidence", "suggested_actions"] : commonKeys,
    endpoint,
    path,
  );
  const common: DiagnosisFindingBase = {
    ruleId: matchingText(required(row, "rule_id", endpoint, path), RULE_ID, endpoint, `${path}.rule_id`),
    ruleName: nonEmptyText(required(row, "rule_name", endpoint, path), endpoint, `${path}.rule_name`),
    summary: nonEmptyText(required(row, "summary", endpoint, path), endpoint, `${path}.summary`),
    evidence: array(required(row, "evidence", endpoint, path), endpoint, `${path}.evidence`)
      .map((entry, index) => decodeEvidence(entry, endpoint, `${path}.evidence[${index}]`)),
    limitations: textArray(required(row, "limitations", endpoint, path), endpoint, `${path}.limitations`),
  };
  if (status !== "finding") return { ...common, status };
  return {
    ...common,
    status,
    severity: oneOf(required(row, "severity", endpoint, path), SEVERITIES, endpoint, `${path}.severity`),
    confidence: oneOf(required(row, "confidence", endpoint, path), CONFIDENCES, endpoint, `${path}.confidence`),
    suggestedActions: array(required(row, "suggested_actions", endpoint, path), endpoint, `${path}.suggested_actions`)
      .map((entry, index) => decodeSuggestion(entry, endpoint, `${path}.suggested_actions[${index}]`)),
  };
}

function decodeFindings(value: unknown, endpoint: string): DiagnosisFindings {
  const path = "$.findings";
  const row = object(value, endpoint, path);
  closed(row, [
    "schema_version", "generated_at", "run_id", "comparison_run_id", "redacted", "rule_registry",
    "suppressed_rule_ids", "findings", "summary",
  ], endpoint, path);
  if (required(row, "schema_version", endpoint, path) !== "clozn.diagnosis-findings.v1") {
    fail(endpoint, `${path}.schema_version`, "must be clozn.diagnosis-findings.v1");
  }
  const ruleRegistry = array(required(row, "rule_registry", endpoint, path), endpoint, `${path}.rule_registry`)
    .map((entry, index) => {
      const itemPath = `${path}.rule_registry[${index}]`;
      const item = object(entry, endpoint, itemPath);
      closed(item, ["rule_id", "rule_name"], endpoint, itemPath);
      return {
        ruleId: matchingText(required(item, "rule_id", endpoint, itemPath), RULE_ID, endpoint, `${itemPath}.rule_id`),
        ruleName: nonEmptyText(required(item, "rule_name", endpoint, itemPath), endpoint, `${itemPath}.rule_name`),
      };
    });
  if (ruleRegistry.length !== DIAGNOSIS_RULES.length) {
    fail(endpoint, `${path}.rule_registry`, `must contain exactly ${DIAGNOSIS_RULES.length} rules`);
  }
  DIAGNOSIS_RULES.forEach(([ruleId, ruleName], index) => {
    const registered = ruleRegistry[index];
    if (registered?.ruleId !== ruleId || registered.ruleName !== ruleName) {
      fail(endpoint, `${path}.rule_registry[${index}]`, `must be ${ruleId} ${ruleName}`);
    }
  });
  if (new Set(ruleRegistry.map((rule) => rule.ruleId)).size !== ruleRegistry.length) {
    fail(endpoint, `${path}.rule_registry`, "must not contain duplicate rule ids");
  }
  const findings = array(required(row, "findings", endpoint, path), endpoint, `${path}.findings`)
    .map((entry, index) => decodeFinding(entry, endpoint, `${path}.findings[${index}]`));
  if (findings.length !== ruleRegistry.length) fail(endpoint, `${path}.findings`, "must contain one entry per registered rule");
  ruleRegistry.forEach((rule, index) => {
    const finding = findings[index]!;
    if (finding.ruleId !== rule.ruleId || finding.ruleName !== rule.ruleName) {
      fail(endpoint, `${path}.findings[${index}]`, "does not match the registry entry at the same position");
    }
  });
  const suppressedRuleIds = textArray(
    required(row, "suppressed_rule_ids", endpoint, path), endpoint, `${path}.suppressed_rule_ids`,
  );
  suppressedRuleIds.forEach((id, index) => {
    if (!RULE_ID.test(id)) fail(endpoint, `${path}.suppressed_rule_ids[${index}]`, "has an invalid rule id");
  });
  if (new Set(suppressedRuleIds).size !== suppressedRuleIds.length) {
    fail(endpoint, `${path}.suppressed_rule_ids`, "must not contain duplicate rule ids");
  }
  const suppressedFindings = findings
    .filter((finding) => finding.status === "suppressed")
    .map((finding) => finding.ruleId);
  if (suppressedRuleIds.length !== suppressedFindings.length
      || suppressedRuleIds.some((ruleId, index) => ruleId !== suppressedFindings[index])) {
    fail(endpoint, `${path}.suppressed_rule_ids`, "must exactly match suppressed findings in registry order");
  }
  const summary = object(required(row, "summary", endpoint, path), endpoint, `${path}.summary`);
  closed(summary, ["status_counts"], endpoint, `${path}.summary`);
  const counts = object(required(summary, "status_counts", endpoint, `${path}.summary`), endpoint, `${path}.summary.status_counts`);
  closed(counts, FINDING_STATUSES, endpoint, `${path}.summary.status_counts`);
  const statusCounts = Object.fromEntries(FINDING_STATUSES.map((status) => [
    status,
    nonNegativeInteger(required(counts, status, endpoint, `${path}.summary.status_counts`), endpoint, `${path}.summary.status_counts.${status}`),
  ])) as unknown as Record<DiagnosisFindingStatus, number>;
  for (const status of FINDING_STATUSES) {
    if (statusCounts[status] !== findings.filter((finding) => finding.status === status).length) {
      fail(endpoint, `${path}.summary.status_counts.${status}`, "does not match the findings array");
    }
  }
  const comparisonRunId = optionalText(row, "comparison_run_id", endpoint, path);
  return {
    schemaVersion: "clozn.diagnosis-findings.v1",
    generatedAt: nonEmptyText(required(row, "generated_at", endpoint, path), endpoint, `${path}.generated_at`),
    runId: nonEmptyText(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`),
    ...(comparisonRunId ? { comparisonRunId } : {}),
    redacted: booleanValue(required(row, "redacted", endpoint, path), endpoint, `${path}.redacted`),
    ruleRegistry,
    suppressedRuleIds,
    findings,
    statusCounts,
  };
}

function decodeNarrativeEvidence(value: unknown, endpoint: string, path: string): DiagnosisNarrativeEvidence {
  const row = object(value, endpoint, path);
  if (row.kind === "finding") {
    closed(row, ["kind", "rule_id"], endpoint, path);
    return { kind: "finding", ruleId: matchingText(required(row, "rule_id", endpoint, path), RULE_ID, endpoint, `${path}.rule_id`) };
  }
  if (row.kind === "diff_field") {
    closed(row, ["kind", "dimension"], endpoint, path);
    return { kind: "diff_field", dimension: nonEmptyText(required(row, "dimension", endpoint, path), endpoint, `${path}.dimension`) };
  }
  if (row.kind === "text_span") {
    closed(row, ["kind", "address_id"], endpoint, path);
    return { kind: "text_span", addressId: matchingText(required(row, "address_id", endpoint, path), SPAN_ID, endpoint, `${path}.address_id`) };
  }
  fail(endpoint, `${path}.kind`, "must be finding, diff_field, or text_span");
}

function decodeNarrativeEvidenceArray(value: unknown, endpoint: string, path: string): DiagnosisNarrativeEvidence[] {
  const result = array(value, endpoint, path)
    .map((entry, index) => decodeNarrativeEvidence(entry, endpoint, `${path}[${index}]`));
  if (result.length === 0) fail(endpoint, path, "must not be empty");
  return result;
}

function decodeNarrative(value: unknown, endpoint: string, findings: DiagnosisFindings): DiagnosisNarrative {
  const path = "$.narrative";
  const row = object(value, endpoint, path);
  closed(row, [
    "schema_version", "generated_at", "run_id", "comparison_run_id", "comparison_available",
    "findings_schema_version", "headline", "registers", "summary",
  ], endpoint, path);
  if (required(row, "schema_version", endpoint, path) !== "clozn.diagnosis-narrative.v1") {
    fail(endpoint, `${path}.schema_version`, "must be clozn.diagnosis-narrative.v1");
  }
  if (required(row, "findings_schema_version", endpoint, path) !== "clozn.diagnosis-findings.v1") {
    fail(endpoint, `${path}.findings_schema_version`, "must be clozn.diagnosis-findings.v1");
  }
  const registers = object(required(row, "registers", endpoint, path), endpoint, `${path}.registers`);
  closed(registers, ["observed_changes", "measured_effects", "plausible_but_unproven"], endpoint, `${path}.registers`);
  const observedChanges = array(required(registers, "observed_changes", endpoint, `${path}.registers`), endpoint, `${path}.registers.observed_changes`)
    .map((entry, index): DiagnosisObservedChange => {
      const itemPath = `${path}.registers.observed_changes[${index}]`;
      const item = object(entry, endpoint, itemPath);
      closed(item, ["dimension", "kind", "text", "evidence"], endpoint, itemPath);
      return {
        dimension: nonEmptyText(required(item, "dimension", endpoint, itemPath), endpoint, `${itemPath}.dimension`),
        kind: oneOf(required(item, "kind", endpoint, itemPath), OBSERVED_KINDS, endpoint, `${itemPath}.kind`),
        text: nonEmptyText(required(item, "text", endpoint, itemPath), endpoint, `${itemPath}.text`),
        evidence: decodeNarrativeEvidenceArray(required(item, "evidence", endpoint, itemPath), endpoint, `${itemPath}.evidence`),
      };
    });
  const measuredEffects = array(required(registers, "measured_effects", endpoint, `${path}.registers`), endpoint, `${path}.registers.measured_effects`)
    .map((entry, index): DiagnosisMeasuredEffect => {
      const itemPath = `${path}.registers.measured_effects[${index}]`;
      const item = object(entry, endpoint, itemPath);
      closed(item, ["rank", "basis", "rule_id", "severity", "confidence", "text", "evidence"], endpoint, itemPath);
      const ruleId = matchingText(required(item, "rule_id", endpoint, itemPath), RULE_ID, endpoint, `${itemPath}.rule_id`);
      if (!findings.findings.some((finding) => finding.ruleId === ruleId && finding.status === "finding")) {
        fail(endpoint, `${itemPath}.rule_id`, "must refer to a diagnosis finding");
      }
      return {
        rank: positiveInteger(required(item, "rank", endpoint, itemPath), endpoint, `${itemPath}.rank`),
        basis: oneOf(required(item, "basis", endpoint, itemPath), ["intervention", "rule_finding"] as const, endpoint, `${itemPath}.basis`),
        ruleId,
        severity: oneOf(required(item, "severity", endpoint, itemPath), SEVERITIES, endpoint, `${itemPath}.severity`),
        confidence: oneOf(required(item, "confidence", endpoint, itemPath), CONFIDENCES, endpoint, `${itemPath}.confidence`),
        text: nonEmptyText(required(item, "text", endpoint, itemPath), endpoint, `${itemPath}.text`),
        evidence: decodeNarrativeEvidenceArray(required(item, "evidence", endpoint, itemPath), endpoint, `${itemPath}.evidence`),
      };
    });
  const plausibleButUnproven = array(
    required(registers, "plausible_but_unproven", endpoint, `${path}.registers`), endpoint,
    `${path}.registers.plausible_but_unproven`,
  ).map((entry, index): DiagnosisPlausibleNote => {
    const itemPath = `${path}.registers.plausible_but_unproven[${index}]`;
    const item = object(entry, endpoint, itemPath);
    closed(item, ["text", "note", "evidence"], endpoint, itemPath);
    return {
      text: nonEmptyText(required(item, "text", endpoint, itemPath), endpoint, `${itemPath}.text`),
      note: nonEmptyText(required(item, "note", endpoint, itemPath), endpoint, `${itemPath}.note`),
      evidence: decodeNarrativeEvidenceArray(required(item, "evidence", endpoint, itemPath), endpoint, `${itemPath}.evidence`),
    };
  });
  const summary = object(required(row, "summary", endpoint, path), endpoint, `${path}.summary`);
  closed(summary, ["counts"], endpoint, `${path}.summary`);
  const counts = object(required(summary, "counts", endpoint, `${path}.summary`), endpoint, `${path}.summary.counts`);
  closed(counts, ["observed_changes", "measured_effects", "plausible_but_unproven"], endpoint, `${path}.summary.counts`);
  const expectedCounts = {
    observed_changes: observedChanges.length,
    measured_effects: measuredEffects.length,
    plausible_but_unproven: plausibleButUnproven.length,
  };
  for (const [key, expected] of Object.entries(expectedCounts)) {
    if (nonNegativeInteger(required(counts, key, endpoint, `${path}.summary.counts`), endpoint, `${path}.summary.counts.${key}`) !== expected) {
      fail(endpoint, `${path}.summary.counts.${key}`, "does not match its register");
    }
  }
  const comparisonRunId = optionalText(row, "comparison_run_id", endpoint, path);
  const comparisonAvailable = booleanValue(
    required(row, "comparison_available", endpoint, path), endpoint, `${path}.comparison_available`,
  );
  if (comparisonAvailable !== (comparisonRunId !== undefined)) {
    fail(endpoint, `${path}.comparison_available`, "must match the presence of comparison_run_id");
  }
  return {
    schemaVersion: "clozn.diagnosis-narrative.v1",
    generatedAt: nonEmptyText(required(row, "generated_at", endpoint, path), endpoint, `${path}.generated_at`),
    runId: nonEmptyText(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`),
    ...(comparisonRunId ? { comparisonRunId } : {}),
    comparisonAvailable,
    findingsSchemaVersion: "clozn.diagnosis-findings.v1",
    headline: nonEmptyText(required(row, "headline", endpoint, path), endpoint, `${path}.headline`),
    observedChanges,
    measuredEffects,
    plausibleButUnproven,
  };
}

export function decodeDiagnosisDocument(
  value: unknown,
  endpoint = "/runs/<id>/diagnosis-findings",
  expectedRunId?: string,
  expectedComparisonRunId?: string,
): DiagnosisDocument {
  const row = object(value, endpoint, "$");
  closed(row, ["findings", "narrative"], endpoint, "$");
  const findings = decodeFindings(required(row, "findings", endpoint, "$"), endpoint);
  const narrative = decodeNarrative(required(row, "narrative", endpoint, "$"), endpoint, findings);
  if (findings.runId !== narrative.runId) fail(endpoint, "$.narrative.run_id", "does not match findings.run_id");
  if (findings.comparisonRunId !== narrative.comparisonRunId) {
    fail(endpoint, "$.narrative.comparison_run_id", "does not match findings.comparison_run_id");
  }
  if (expectedRunId !== undefined && findings.runId !== expectedRunId) {
    fail(endpoint, "$.findings.run_id", "does not match the requested run");
  }
  if (findings.comparisonRunId !== expectedComparisonRunId) {
    fail(endpoint, "$.findings.comparison_run_id", "does not match the requested comparison run");
  }
  return { findings, narrative };
}

function decodeScopeEligibility(value: unknown, endpoint: string, path: string): CorrectiveScopeEligibility {
  const row = object(value, endpoint, path);
  closed(row, ["scope", "available", "target", "prior_hash", "before", "after", "note", "unavailability_reason"], endpoint, path);
  if (required(row, "scope", endpoint, path) !== "once") fail(endpoint, `${path}.scope`, "must be once");
  const target = optionalText(row, "target", endpoint, path);
  const priorHash = "prior_hash" in row
    ? matchingText(row.prior_hash, SHA256, endpoint, `${path}.prior_hash`)
    : undefined;
  const note = optionalText(row, "note", endpoint, path);
  const unavailabilityReason = optionalText(row, "unavailability_reason", endpoint, path);
  const available = booleanValue(required(row, "available", endpoint, path), endpoint, `${path}.available`);
  if (available && (!target || !priorHash)) {
    fail(endpoint, path, "must carry target and prior_hash when available");
  }
  return {
    scope: "once",
    available,
    ...(target ? { target } : {}),
    ...(priorHash ? { priorHash } : {}),
    ...("before" in row ? { before: textArray(row.before, endpoint, `${path}.before`) } : {}),
    ...("after" in row ? { after: textArray(row.after, endpoint, `${path}.after`) } : {}),
    ...(note ? { note } : {}),
    ...(unavailabilityReason ? { unavailabilityReason } : {}),
  };
}

function decodeOneScopeEligibility(value: unknown, endpoint: string, path: string): readonly [CorrectiveScopeEligibility] {
  const items = array(value, endpoint, path);
  if (items.length !== 1) fail(endpoint, path, "must contain exactly the once scope");
  return [decodeScopeEligibility(items[0], endpoint, `${path}[0]`)];
}

function correctiveActionId(value: unknown, endpoint: string, path: string): CorrectiveActionId {
  return oneOf(value, CORRECTIVE_ACTION_IDS, endpoint, path);
}

function decodeActionIdentity(value: unknown, endpoint: string, path: string) {
  const row = object(value, endpoint, path);
  closed(row, ["id", "label", "description"], endpoint, path);
  return {
    id: correctiveActionId(required(row, "id", endpoint, path), endpoint, `${path}.id`),
    label: nonEmptyText(required(row, "label", endpoint, path), endpoint, `${path}.label`),
    description: nonEmptyText(required(row, "description", endpoint, path), endpoint, `${path}.description`),
  };
}

function decodeBackend(value: unknown, endpoint: string, path: string, actionId: CorrectiveActionId): CorrectiveBackendDeclaration {
  const row = object(value, endpoint, path);
  if (row.type === "prompt_policy") {
    closed(row, ["type", "recipe_version", "parameters", "qualification", "qualification_id", "available"], endpoint, path);
    if (required(row, "qualification", endpoint, path) !== "generic") fail(endpoint, `${path}.qualification`, "must be generic");
    if (required(row, "available", endpoint, path) !== true) fail(endpoint, `${path}.available`, "must be true");
    const parameters = object(required(row, "parameters", endpoint, path), endpoint, `${path}.parameters`);
    closed(parameters, ["preset"], endpoint, `${path}.parameters`);
    const preset = correctiveActionId(required(parameters, "preset", endpoint, `${path}.parameters`), endpoint, `${path}.parameters.preset`);
    if (preset !== actionId) fail(endpoint, `${path}.parameters.preset`, "must match the action id");
    return {
      type: "prompt_policy",
      recipeVersion: nonEmptyText(required(row, "recipe_version", endpoint, path), endpoint, `${path}.recipe_version`),
      parameters: { preset },
      qualification: "generic",
      qualificationId: nonEmptyText(required(row, "qualification_id", endpoint, path), endpoint, `${path}.qualification_id`),
      available: true,
    };
  }
  if (row.type === "unsupported") {
    closed(row, ["type", "requested_type", "available", "reason", "unavailability_reason"], endpoint, path);
    if (required(row, "requested_type", endpoint, path) !== "sampling_policy") {
      fail(endpoint, `${path}.requested_type`, "must be sampling_policy");
    }
    if (required(row, "available", endpoint, path) !== false) fail(endpoint, `${path}.available`, "must be false");
    return {
      type: "unsupported",
      requestedType: "sampling_policy",
      available: false,
      reason: nonEmptyText(required(row, "reason", endpoint, path), endpoint, `${path}.reason`),
      unavailabilityReason: nonEmptyText(
        required(row, "unavailability_reason", endpoint, path), endpoint, `${path}.unavailability_reason`,
      ),
    };
  }
  fail(endpoint, `${path}.type`, "must be prompt_policy or the declared unsupported sampling_policy");
}

export function decodeCorrectiveRegistry(
  value: unknown,
  endpoint = "/runs/<id>/corrective-actions",
  expectedRunId?: string,
): CorrectiveRegistry {
  const row = object(value, endpoint, "$");
  closed(row, ["schema_version", "version", "actions", "run_id", "run_fingerprint"], endpoint, "$");
  if (required(row, "schema_version", endpoint, "$") !== "clozn.action-registry.v1") {
    fail(endpoint, "$.schema_version", "must be clozn.action-registry.v1");
  }
  const actions = array(required(row, "actions", endpoint, "$"), endpoint, "$.actions")
    .map((entry, index): CorrectiveAction => {
      const path = `$.actions[${index}]`;
      const item = object(entry, endpoint, path);
      closed(item, [
        "id", "label", "description", "conflicts", "scopes", "eligibility", "evaluation_metrics",
        "backends", "scope_eligibility",
      ], endpoint, path);
      const id = correctiveActionId(required(item, "id", endpoint, path), endpoint, `${path}.id`);
      const conflicts = array(required(item, "conflicts", endpoint, path), endpoint, `${path}.conflicts`)
        .map((conflict, conflictIndex) => correctiveActionId(conflict, endpoint, `${path}.conflicts[${conflictIndex}]`));
      const scopes = array(required(item, "scopes", endpoint, path), endpoint, `${path}.scopes`);
      if (scopes.length !== 1 || scopes[0] !== "once") fail(endpoint, `${path}.scopes`, "must contain only once");
      const eligibility = object(required(item, "eligibility", endpoint, path), endpoint, `${path}.eligibility`);
      closed(eligibility, ["eligible", "note", "reason"], endpoint, `${path}.eligibility`);
      const note = optionalText(eligibility, "note", endpoint, `${path}.eligibility`);
      const reason = optionalText(eligibility, "reason", endpoint, `${path}.eligibility`);
      const backends = array(required(item, "backends", endpoint, path), endpoint, `${path}.backends`)
        .map((backend, backendIndex) => decodeBackend(backend, endpoint, `${path}.backends[${backendIndex}]`, id));
      if (backends.filter((backend) => backend.type === "prompt_policy").length !== 1) {
        fail(endpoint, `${path}.backends`, "must contain exactly one prompt_policy backend");
      }
      return {
        id,
        label: nonEmptyText(required(item, "label", endpoint, path), endpoint, `${path}.label`),
        description: nonEmptyText(required(item, "description", endpoint, path), endpoint, `${path}.description`),
        conflicts,
        scopes: ["once"],
        eligibility: {
          eligible: booleanValue(required(eligibility, "eligible", endpoint, `${path}.eligibility`), endpoint, `${path}.eligibility.eligible`),
          ...(note ? { note } : {}),
          ...(reason ? { reason } : {}),
        },
        evaluationMetrics: textArray(required(item, "evaluation_metrics", endpoint, path), endpoint, `${path}.evaluation_metrics`),
        backends,
        scopeEligibility: decodeOneScopeEligibility(
          required(item, "scope_eligibility", endpoint, path), endpoint, `${path}.scope_eligibility`,
        ),
      };
    });
  if (actions.length !== CORRECTIVE_ACTION_IDS.length
      || new Set(actions.map((action) => action.id)).size !== CORRECTIVE_ACTION_IDS.length
      || CORRECTIVE_ACTION_IDS.some((id) => !actions.some((action) => action.id === id))) {
    fail(endpoint, "$.actions", "must contain each of the six corrective actions exactly once");
  }
  const runId = nonEmptyText(required(row, "run_id", endpoint, "$"), endpoint, "$.run_id");
  if (expectedRunId !== undefined && runId !== expectedRunId) fail(endpoint, "$.run_id", "does not match the requested run");
  if (actions.some((action) => action.scopeEligibility[0].target !== runId)) {
    fail(endpoint, "$.actions", "scope eligibility target must match run_id");
  }
  return {
    schemaVersion: "clozn.action-registry.v1",
    version: nonEmptyText(required(row, "version", endpoint, "$"), endpoint, "$.version"),
    runId,
    runFingerprint: matchingText(required(row, "run_fingerprint", endpoint, "$"), SHA256, endpoint, "$.run_fingerprint"),
    actions,
  };
}

export function decodeCorrectivePreview(
  value: unknown,
  endpoint = "/corrective-previews/<id>",
  expected?: { previewId?: string; parentRunId?: string; actionId?: CorrectiveActionId },
): CorrectivePreview {
  const row = object(value, endpoint, "$");
  closed(row, [
    "preview_id", "status", "created_ts", "expires_ts", "parent_run_id", "parent_run_fingerprint",
    "action", "execution", "scope_eligibility", "comparison_contract", "result_id",
  ], endpoint, "$");
  const previewId = matchingText(required(row, "preview_id", endpoint, "$"), PREVIEW_ID, endpoint, "$.preview_id");
  const parentRunId = nonEmptyText(required(row, "parent_run_id", endpoint, "$"), endpoint, "$.parent_run_id");
  const action = decodeActionIdentity(required(row, "action", endpoint, "$"), endpoint, "$.action");
  const execution = object(required(row, "execution", endpoint, "$"), endpoint, "$.execution");
  closed(execution, [
    "requested_backend", "expected_executed_backend", "expected_fallback", "qualification",
    "qualification_id", "unavailability_reason",
  ], endpoint, "$.execution");
  if (required(execution, "requested_backend", endpoint, "$.execution") !== "prompt_policy") {
    fail(endpoint, "$.execution.requested_backend", "must be prompt_policy");
  }
  if (required(execution, "expected_executed_backend", endpoint, "$.execution") !== "prompt_policy") {
    fail(endpoint, "$.execution.expected_executed_backend", "must be prompt_policy");
  }
  if (required(execution, "expected_fallback", endpoint, "$.execution") !== false) {
    fail(endpoint, "$.execution.expected_fallback", "must be false");
  }
  if ("unavailability_reason" in execution) {
    fail(endpoint, "$.execution.unavailability_reason", "does not apply to prompt_policy");
  }
  const contract = object(required(row, "comparison_contract", endpoint, "$"), endpoint, "$.comparison_contract");
  closed(contract, ["baseline", "corrected", "stored_original"], endpoint, "$.comparison_contract");
  if (expected?.previewId !== undefined && previewId !== expected.previewId) fail(endpoint, "$.preview_id", "does not match the request");
  if (expected?.parentRunId !== undefined && parentRunId !== expected.parentRunId) fail(endpoint, "$.parent_run_id", "does not match the request");
  if (expected?.actionId !== undefined && action.id !== expected.actionId) fail(endpoint, "$.action.id", "does not match the request");
  const resultId = "result_id" in row
    ? matchingText(row.result_id, RESULT_ID, endpoint, "$.result_id") : undefined;
  const scopeEligibility = decodeOneScopeEligibility(
    required(row, "scope_eligibility", endpoint, "$"), endpoint, "$.scope_eligibility",
  );
  if (scopeEligibility[0].target !== parentRunId) {
    fail(endpoint, "$.scope_eligibility[0].target", "must match parent_run_id");
  }
  return {
    previewId,
    status: oneOf(required(row, "status", endpoint, "$"), PREVIEW_STATUSES, endpoint, "$.status"),
    createdTs: nonNegativeNumber(required(row, "created_ts", endpoint, "$"), endpoint, "$.created_ts"),
    expiresTs: nonNegativeNumber(required(row, "expires_ts", endpoint, "$"), endpoint, "$.expires_ts"),
    parentRunId,
    parentRunFingerprint: matchingText(
      required(row, "parent_run_fingerprint", endpoint, "$"), SHA256, endpoint, "$.parent_run_fingerprint",
    ),
    action,
    execution: {
      requestedBackend: "prompt_policy",
      expectedExecutedBackend: "prompt_policy",
      expectedFallback: false,
      qualification: nonEmptyText(required(execution, "qualification", endpoint, "$.execution"), endpoint, "$.execution.qualification"),
      qualificationId: nonEmptyText(required(execution, "qualification_id", endpoint, "$.execution"), endpoint, "$.execution.qualification_id"),
    },
    scopeEligibility,
    comparisonContract: {
      baseline: nonEmptyText(required(contract, "baseline", endpoint, "$.comparison_contract"), endpoint, "$.comparison_contract.baseline"),
      corrected: nonEmptyText(required(contract, "corrected", endpoint, "$.comparison_contract"), endpoint, "$.comparison_contract.corrected"),
      storedOriginal: nonEmptyText(required(contract, "stored_original", endpoint, "$.comparison_contract"), endpoint, "$.comparison_contract.stored_original"),
    },
    ...(resultId ? { resultId } : {}),
  };
}

function decodeChild(value: unknown, endpoint: string, path: string): CorrectiveChild {
  const row = object(value, endpoint, path);
  const status = oneOf(required(row, "status", endpoint, path), ["success", "error", "not_run"] as const, endpoint, `${path}.status`);
  if (status === "success") {
    closed(row, ["status", "run_id"], endpoint, path);
    return { status, runId: nonEmptyText(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`) };
  }
  if (status === "error") {
    closed(row, ["status", "run_id", "error"], endpoint, path);
    const error = object(required(row, "error", endpoint, path), endpoint, `${path}.error`);
    closed(error, ["code", "message"], endpoint, `${path}.error`);
    const runId = "run_id" in row
      ? row.run_id === null ? null : nonEmptyText(row.run_id, endpoint, `${path}.run_id`)
      : undefined;
    return {
      status,
      ...(runId === undefined ? {} : { runId }),
      error: {
        code: nonEmptyText(required(error, "code", endpoint, `${path}.error`), endpoint, `${path}.error.code`),
        message: nonEmptyText(required(error, "message", endpoint, `${path}.error`), endpoint, `${path}.error.message`),
      },
    };
  }
  closed(row, ["status"], endpoint, path);
  return { status };
}

function decodeRevision(value: unknown, endpoint: string, path: string): CorrectiveRevisionSelection {
  const row = object(value, endpoint, path);
  closed(row, ["result_id", "child_run_id", "action_id", "selected_ts"], endpoint, path);
  return {
    resultId: matchingText(required(row, "result_id", endpoint, path), RESULT_ID, endpoint, `${path}.result_id`),
    childRunId: nonEmptyText(required(row, "child_run_id", endpoint, path), endpoint, `${path}.child_run_id`),
    actionId: correctiveActionId(required(row, "action_id", endpoint, path), endpoint, `${path}.action_id`),
    selectedTs: nonNegativeNumber(required(row, "selected_ts", endpoint, path), endpoint, `${path}.selected_ts`),
  };
}

function decodeTransaction(value: unknown, endpoint: string, path: string): CorrectiveTransaction {
  const row = object(value, endpoint, path);
  closed(row, ["id", "scope", "target", "before", "after", "created_ts", "undone_ts"], endpoint, path);
  if (required(row, "scope", endpoint, path) !== "once") fail(endpoint, `${path}.scope`, "must be once");
  return {
    id: nonEmptyText(required(row, "id", endpoint, path), endpoint, `${path}.id`),
    scope: "once",
    target: nonEmptyText(required(row, "target", endpoint, path), endpoint, `${path}.target`),
    before: row.before === null ? null : decodeRevision(required(row, "before", endpoint, path), endpoint, `${path}.before`),
    after: decodeRevision(required(row, "after", endpoint, path), endpoint, `${path}.after`),
    createdTs: nonNegativeNumber(required(row, "created_ts", endpoint, path), endpoint, `${path}.created_ts`),
    undoneTs: row.undone_ts === null
      ? null : nonNegativeNumber(required(row, "undone_ts", endpoint, path), endpoint, `${path}.undone_ts`),
  };
}

function decodeSourceUseArm(value: unknown, endpoint: string, path: string) {
  const row = object(value, endpoint, path);
  closed(row, ["answer_span_count", "answer_spans_with_clear_source", "observed_source_dependence_ratio"], endpoint, path);
  return {
    answerSpanCount: nonNegativeInteger(required(row, "answer_span_count", endpoint, path), endpoint, `${path}.answer_span_count`),
    answerSpansWithClearSource: nonNegativeInteger(
      required(row, "answer_spans_with_clear_source", endpoint, path), endpoint, `${path}.answer_spans_with_clear_source`,
    ),
    observedSourceDependenceRatio: finiteNumber(
      required(row, "observed_source_dependence_ratio", endpoint, path), endpoint, `${path}.observed_source_dependence_ratio`,
    ),
  };
}

function decodeSourceUse(value: unknown, endpoint: string, path: string): CorrectiveSourceUseComparison {
  const row = object(value, endpoint, path);
  closed(row, [
    "status", "method", "thresholds", "baseline", "corrected",
    "delta_observed_source_dependence_ratio", "caveat",
  ], endpoint, path);
  if (required(row, "status", endpoint, path) !== "available") fail(endpoint, `${path}.status`, "must be available");
  return {
    status: "available",
    method: object(required(row, "method", endpoint, path), endpoint, `${path}.method`),
    thresholds: object(required(row, "thresholds", endpoint, path), endpoint, `${path}.thresholds`),
    baseline: decodeSourceUseArm(required(row, "baseline", endpoint, path), endpoint, `${path}.baseline`),
    corrected: decodeSourceUseArm(required(row, "corrected", endpoint, path), endpoint, `${path}.corrected`),
    deltaObservedSourceDependenceRatio: finiteNumber(
      required(row, "delta_observed_source_dependence_ratio", endpoint, path), endpoint,
      `${path}.delta_observed_source_dependence_ratio`,
    ),
    caveat: nonEmptyText(required(row, "caveat", endpoint, path), endpoint, `${path}.caveat`),
  };
}

export function decodeCorrectiveResult(
  value: unknown,
  endpoint = "/corrective-results/<id>",
  expected?: { resultId?: string; previewId?: string; parentRunId?: string },
): CorrectiveResult {
  const row = object(value, endpoint, "$");
  closed(row, [
    "schema_version", "result_id", "created_ts", "parent_run_id", "preview_id", "action", "user_intent",
    "execution", "children", "comparison", "metrics", "coherence", "intervention_observed",
    "scope_eligibility", "outcome", "transaction", "source_use_comparison",
  ], endpoint, "$");
  if (required(row, "schema_version", endpoint, "$") !== "clozn.corrective-retry-result.v1") {
    fail(endpoint, "$.schema_version", "must be clozn.corrective-retry-result.v1");
  }
  const resultId = matchingText(required(row, "result_id", endpoint, "$"), RESULT_ID, endpoint, "$.result_id");
  const previewId = matchingText(required(row, "preview_id", endpoint, "$"), PREVIEW_ID, endpoint, "$.preview_id");
  const parentRunId = nonEmptyText(required(row, "parent_run_id", endpoint, "$"), endpoint, "$.parent_run_id");
  if (expected?.resultId !== undefined && resultId !== expected.resultId) fail(endpoint, "$.result_id", "does not match the request");
  if (expected?.previewId !== undefined && previewId !== expected.previewId) fail(endpoint, "$.preview_id", "does not match the request");
  if (expected?.parentRunId !== undefined && parentRunId !== expected.parentRunId) fail(endpoint, "$.parent_run_id", "does not match the request");
  const action = decodeActionIdentity(required(row, "action", endpoint, "$"), endpoint, "$.action");
  const intent = object(required(row, "user_intent", endpoint, "$"), endpoint, "$.user_intent");
  closed(intent, ["action_id", "kept_scope"], endpoint, "$.user_intent");
  const intentAction = correctiveActionId(required(intent, "action_id", endpoint, "$.user_intent"), endpoint, "$.user_intent.action_id");
  if (intentAction !== action.id) fail(endpoint, "$.user_intent.action_id", "must match action.id");
  if ("kept_scope" in intent && intent.kept_scope !== "once") fail(endpoint, "$.user_intent.kept_scope", "must be once");
  const execution = object(required(row, "execution", endpoint, "$"), endpoint, "$.execution");
  closed(execution, [
    "requested_backend", "executed_backend", "fallback", "qualification", "qualification_id", "identity",
  ], endpoint, "$.execution");
  if (required(execution, "requested_backend", endpoint, "$.execution") !== "prompt_policy") {
    fail(endpoint, "$.execution.requested_backend", "must be prompt_policy");
  }
  const executedBackend = required(execution, "executed_backend", endpoint, "$.execution");
  if (executedBackend !== null && executedBackend !== "prompt_policy") {
    fail(endpoint, "$.execution.executed_backend", "must be prompt_policy or null");
  }
  const qualification = required(execution, "qualification", endpoint, "$.execution");
  if (qualification !== null && typeof qualification !== "string") fail(endpoint, "$.execution.qualification", "must be a string or null");
  const qualificationId = required(execution, "qualification_id", endpoint, "$.execution");
  if (qualificationId !== null && typeof qualificationId !== "string") fail(endpoint, "$.execution.qualification_id", "must be a string or null");
  const identityValue = required(execution, "identity", endpoint, "$.execution");
  const identity = identityValue === null ? null : object(identityValue, endpoint, "$.execution.identity");
  const children = object(required(row, "children", endpoint, "$"), endpoint, "$.children");
  closed(children, ["baseline", "corrected"], endpoint, "$.children");
  const comparison = object(required(row, "comparison", endpoint, "$"), endpoint, "$.comparison");
  closed(comparison, [
    "stored_original_reply", "baseline_reply", "corrected_reply", "note", "changed",
  ], endpoint, "$.comparison");
  const outcome = object(required(row, "outcome", endpoint, "$"), endpoint, "$.outcome");
  closed(outcome, ["status", "note"], endpoint, "$.outcome");
  const outcomeStatus = oneOf(
    required(outcome, "status", endpoint, "$.outcome"),
    ["succeeded", "execution_error", "cancelled"] as const,
    endpoint,
    "$.outcome.status",
  );
  const outcomeNote = optionalText(outcome, "note", endpoint, "$.outcome");
  let coherence: CorrectiveResult["coherence"];
  if ("coherence" in row) {
    const item = object(row.coherence, endpoint, "$.coherence");
    closed(item, ["degenerate", "reason"], endpoint, "$.coherence");
    coherence = {
      degenerate: booleanValue(required(item, "degenerate", endpoint, "$.coherence"), endpoint, "$.coherence.degenerate"),
      reason: text(required(item, "reason", endpoint, "$.coherence"), endpoint, "$.coherence.reason"),
    };
  }
  const transaction = "transaction" in row
    ? decodeTransaction(row.transaction, endpoint, "$.transaction") : undefined;
  if (transaction && (transaction.target !== parentRunId || transaction.after.resultId !== resultId
      || transaction.after.actionId !== action.id)) {
    fail(endpoint, "$.transaction", "does not identify this result, action, and parent run");
  }
  const sourceUseComparison = "source_use_comparison" in row
    ? decodeSourceUse(row.source_use_comparison, endpoint, "$.source_use_comparison") : undefined;
  const scopeEligibility = decodeOneScopeEligibility(
    required(row, "scope_eligibility", endpoint, "$"), endpoint, "$.scope_eligibility",
  );
  if (scopeEligibility[0].target !== parentRunId) {
    fail(endpoint, "$.scope_eligibility[0].target", "must match parent_run_id");
  }
  return {
    schemaVersion: "clozn.corrective-retry-result.v1",
    resultId,
    createdTs: nonNegativeNumber(required(row, "created_ts", endpoint, "$"), endpoint, "$.created_ts"),
    parentRunId,
    previewId,
    action,
    userIntent: { actionId: intentAction, ...("kept_scope" in intent ? { keptScope: "once" as const } : {}) },
    execution: {
      requestedBackend: "prompt_policy",
      executedBackend,
      fallback: booleanValue(required(execution, "fallback", endpoint, "$.execution"), endpoint, "$.execution.fallback"),
      qualification,
      qualificationId,
      identity,
    },
    children: {
      baseline: decodeChild(required(children, "baseline", endpoint, "$.children"), endpoint, "$.children.baseline"),
      corrected: decodeChild(required(children, "corrected", endpoint, "$.children"), endpoint, "$.children.corrected"),
    },
    comparison: {
      storedOriginalReply: text(required(comparison, "stored_original_reply", endpoint, "$.comparison"), endpoint, "$.comparison.stored_original_reply"),
      baselineReply: text(required(comparison, "baseline_reply", endpoint, "$.comparison"), endpoint, "$.comparison.baseline_reply"),
      correctedReply: text(required(comparison, "corrected_reply", endpoint, "$.comparison"), endpoint, "$.comparison.corrected_reply"),
      note: text(required(comparison, "note", endpoint, "$.comparison"), endpoint, "$.comparison.note"),
      changed: booleanValue(required(comparison, "changed", endpoint, "$.comparison"), endpoint, "$.comparison.changed"),
    },
    metrics: object(required(row, "metrics", endpoint, "$"), endpoint, "$.metrics"),
    ...(coherence ? { coherence } : {}),
    ...("intervention_observed" in row
      ? { interventionObserved: booleanValue(row.intervention_observed, endpoint, "$.intervention_observed") }
      : {}),
    scopeEligibility,
    outcome: { status: outcomeStatus, ...(outcomeNote ? { note: outcomeNote } : {}) },
    ...(transaction ? { transaction } : {}),
    ...(sourceUseComparison ? { sourceUseComparison } : {}),
  };
}

export function decodeCorrectiveUndo(value: unknown, endpoint = "/corrective-actions/<id>/undo"): CorrectiveUndo {
  const row = object(value, endpoint, "$");
  closed(row, ["status", "undo_id", "scope", "target"], endpoint, "$");
  if (required(row, "status", endpoint, "$") !== "undone") fail(endpoint, "$.status", "must be undone");
  if (required(row, "scope", endpoint, "$") !== "once") fail(endpoint, "$.scope", "must be once");
  return {
    status: "undone",
    undoId: nonEmptyText(required(row, "undo_id", endpoint, "$"), endpoint, "$.undo_id"),
    scope: "once",
    target: nonEmptyText(required(row, "target", endpoint, "$"), endpoint, "$.target"),
  };
}

function identifier(value: string, label: string): string {
  if (!value.trim()) throw new ContractError("client", `${label} must not be blank`);
  return value;
}

export function preserveCorrectiveIdempotencyKey(value: string): string {
  if (!IDEMPOTENCY_KEY.test(value)) {
    throw new ContractError(
      "client",
      "idempotency key must be 8-128 letters, digits, dots, colons, dashes, or underscores",
    );
  }
  return value;
}

async function responseJson(response: Response, endpoint: string, acceptedStatuses: readonly number[] = [200]): Promise<unknown> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ContractError(endpoint, "response is not valid JSON");
  }
  if (!acceptedStatuses.includes(response.status)) throw new HttpError(endpoint, response.status, response.statusText);
  return body;
}

async function get(endpoint: string, signal?: AbortSignal): Promise<unknown> {
  return responseJson(await fetch(endpoint, {
    method: "GET",
    signal,
    headers: { Accept: "application/json" },
  }), endpoint);
}

async function post(endpoint: string, body: JsonRecord, signal?: AbortSignal, statuses: readonly number[] = [200]): Promise<unknown> {
  return responseJson(await fetch(endpoint, {
    method: "POST",
    signal,
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }), endpoint, statuses);
}

export const diagnosisActionsApi = {
  async diagnosis(
    runIdValue: string,
    options: { compareRunId?: string; suppressedRuleIds?: readonly string[]; signal?: AbortSignal } = {},
  ): Promise<DiagnosisDocument> {
    const runId = identifier(runIdValue, "run id");
    const query = new URLSearchParams();
    let compareRunId: string | undefined;
    if (options.compareRunId !== undefined) {
      compareRunId = identifier(options.compareRunId, "comparison run id");
      query.set("compare", compareRunId);
    }
    if (options.suppressedRuleIds?.length) {
      const unique = [...new Set(options.suppressedRuleIds)];
      unique.forEach((id, index) => {
        if (!RULE_ID.test(id)) throw new ContractError("client", `suppressed rule id ${index} is invalid`);
      });
      query.set("suppress", unique.join(","));
    }
    const suffix = query.size ? `?${query}` : "";
    const endpoint = `/runs/${encodeURIComponent(runId)}/diagnosis-findings${suffix}`;
    return decodeDiagnosisDocument(await get(endpoint, options.signal), endpoint, runId, compareRunId);
  },

  async registry(runIdValue: string, signal?: AbortSignal): Promise<CorrectiveRegistry> {
    const runId = identifier(runIdValue, "run id");
    const endpoint = `/runs/${encodeURIComponent(runId)}/corrective-actions`;
    return decodeCorrectiveRegistry(await get(endpoint, signal), endpoint, runId);
  },

  async preview(
    runIdValue: string,
    actionId: CorrectiveActionId,
    signal?: AbortSignal,
  ): Promise<CorrectivePreview> {
    const runId = identifier(runIdValue, "run id");
    correctiveActionId(actionId, "client", "action id");
    const endpoint = `/runs/${encodeURIComponent(runId)}/corrective-actions/preview`;
    return decodeCorrectivePreview(
      await post(endpoint, { action_id: actionId, requested_backend: "prompt_policy" }, signal, [201]),
      endpoint,
      { parentRunId: runId, actionId },
    );
  },

  async confirm(
    previewIdValue: string,
    request: { idempotencyKey: string },
    signal?: AbortSignal,
  ): Promise<CorrectiveResult> {
    const previewId = matchingText(identifier(previewIdValue, "preview id"), PREVIEW_ID, "client", "preview id");
    const idempotencyKey = preserveCorrectiveIdempotencyKey(request.idempotencyKey);
    const endpoint = `/corrective-previews/${encodeURIComponent(previewId)}/confirm`;
    return decodeCorrectiveResult(
      await post(endpoint, { idempotency_key: idempotencyKey }, signal),
      endpoint,
      { previewId },
    );
  },

  async cancel(previewIdValue: string, signal?: AbortSignal): Promise<CorrectivePreview> {
    const previewId = matchingText(identifier(previewIdValue, "preview id"), PREVIEW_ID, "client", "preview id");
    const endpoint = `/corrective-previews/${encodeURIComponent(previewId)}/cancel`;
    return decodeCorrectivePreview(await post(endpoint, {}, signal), endpoint, { previewId });
  },

  async result(resultIdValue: string, signal?: AbortSignal): Promise<CorrectiveResult> {
    const resultId = matchingText(identifier(resultIdValue, "result id"), RESULT_ID, "client", "result id");
    const endpoint = `/corrective-results/${encodeURIComponent(resultId)}`;
    return decodeCorrectiveResult(await get(endpoint, signal), endpoint, { resultId });
  },

  async keep(
    resultIdValue: string,
    request: { expectedPriorHash: string; idempotencyKey: string },
    signal?: AbortSignal,
  ): Promise<CorrectiveResult> {
    const resultId = matchingText(identifier(resultIdValue, "result id"), RESULT_ID, "client", "result id");
    const expectedPriorHash = matchingText(request.expectedPriorHash, SHA256, "client", "expected prior hash");
    const idempotencyKey = preserveCorrectiveIdempotencyKey(request.idempotencyKey);
    const endpoint = `/corrective-results/${encodeURIComponent(resultId)}/keep`;
    return decodeCorrectiveResult(await post(endpoint, {
      scope: "once",
      expected_prior_hash: expectedPriorHash,
      idempotency_key: idempotencyKey,
    }, signal), endpoint, { resultId });
  },

  async undo(transactionIdValue: string, signal?: AbortSignal): Promise<CorrectiveUndo> {
    const transactionId = identifier(transactionIdValue, "transaction id");
    const endpoint = `/corrective-actions/${encodeURIComponent(transactionId)}/undo`;
    const result = decodeCorrectiveUndo(await post(endpoint, {}, signal), endpoint);
    if (result.undoId !== transactionId) fail(endpoint, "$.undo_id", "does not match the request");
    return result;
  },
};
