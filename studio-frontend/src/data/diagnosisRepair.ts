/**
 * Client for GET /runs/<id>/diagnosis-findings -- D1's `clozn.diagnosis-findings.v1` rule-engine
 * findings AND D2's `clozn.diagnosis-narrative.v1` plain-language narrative, served together
 * (clozn/server/routes/diagnosis_findings.py).
 *
 * A DELIBERATELY SEPARATE vocabulary from data/types.ts's `RunDiagnosis`/`DiagnosisStatus`
 * (clozn.run_diagnosis.v1, the older why-slow/why-cut-off engine) and `PerformanceRuleReport`/
 * `PerformanceRuleStatus` (clozn.performance-trace.v1) -- three different rule engines already coexist
 * in this codebase with three different status enums, on purpose, so a rule-engine entry from one can
 * never be rendered with another's status classes by an accidental type-widening (see data/types.ts's
 * own comment on why `PerformanceRuleStatus` stays distinct from `DiagnosisStatus`). Every type below is
 * prefixed `Repair*` for exactly that reason: this is D1/D2's OWN five-value vocabulary
 * (`finding` / `not_observed` / `unavailable` / `pending` / `suppressed`), never merged with either.
 */

export type RepairFindingStatus = "finding" | "not_observed" | "unavailable" | "pending" | "suppressed";
export type RepairSeverity = "info" | "low" | "medium" | "high";
export type RepairConfidence = "exact" | "pattern_match" | "derived";

/** `clozn.diagnosis-findings.v1`'s own provisional `suggested_actions[].kind` enum
 * (clozn/runs/diagnosis_rules.py's `SUGGESTED_ACTION_KINDS`). The schema's own description says this
 * explicitly: it "anticipates D3's clozn.corrective-flow.v1 action-kind vocabulary, which does not exist
 * as a registered schema yet... callers must not assume D3 will keep them verbatim." Checked against
 * clozn/behavior/registry.py: D3's actual action registry ids (less-verbose, more-concrete, use-context,
 * ask-before-guessing, preserve-formatting, stop-repeating) share NO members with this list. There is no
 * backend bridge from one vocabulary to the other -- DiagnosisRepair.tsx renders these as description
 * text, never as a button wired to a specific corrective-flow action_id (seeing one here is not license
 * to invent that correspondence client-side). */
export type RepairSuggestedActionKind =
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

export interface RepairSuggestedAction {
  kind: RepairSuggestedActionKind;
  description: string;
}

/** One rule's evidence: either a structured field (dotted path + exact value) or a reference into an
 * already-built clozn.text-span-addresses.v1 document -- never a third, invented addressing scheme. */
export type RepairEvidence =
  | { kind: "field"; path: string; value: unknown; note?: string }
  | {
      kind: "text_span";
      addressId: string;
      messageIndex?: number;
      localStart?: number;
      localEnd?: number;
      note?: string;
    };

interface RepairFindingCommon {
  ruleId: string;
  ruleName: string;
  summary: string;
  evidence: RepairEvidence[];
  limitations: string[];
}

/** Mirrors the schema's own `oneOf` exactly: only `status === "finding"` carries
 * severity/confidence/suggestedActions -- the other four statuses never do (omitted, never null-padded,
 * matching clozn.diagnosis-findings.v1's own "roadmap rule: omit, never null-pad" contract). */
export type RepairFinding =
  | (RepairFindingCommon & {
      status: "finding";
      severity: RepairSeverity;
      confidence: RepairConfidence;
      suggestedActions: RepairSuggestedAction[];
    })
  | (RepairFindingCommon & { status: "not_observed" | "unavailable" | "pending" | "suppressed" });

export interface RepairFindingsDocument {
  schemaVersion: string;
  generatedAt: string;
  runId: string;
  comparisonRunId?: string;
  redacted: boolean;
  ruleRegistry: { ruleId: string; ruleName: string }[];
  suppressedRuleIds: string[];
  findings: RepairFinding[];
  statusCounts: Record<RepairFindingStatus, number>;
}

// ------------------------------------------------------------------------------------------- narrative

export type RepairNarrativeEvidenceRef =
  | { kind: "finding"; ruleId: string }
  | { kind: "diff_field"; dimension: string }
  | { kind: "text_span"; addressId: string };

export type RepairObservedKind = "changed" | "added" | "removed" | "unavailable" | "diff_failed";

export interface RepairObservedChange {
  dimension: string;
  kind: RepairObservedKind;
  text: string;
  evidence: RepairNarrativeEvidenceRef[];
}

export interface RepairMeasuredEffect {
  rank: number;
  basis: "intervention" | "rule_finding";
  ruleId: string;
  severity: RepairSeverity;
  confidence: RepairConfidence;
  text: string;
  evidence: RepairNarrativeEvidenceRef[];
}

/** Deliberately has no rank/severity/confidence/ruleId field -- mirrors the schema's own structural ban
 * on ranking an unproven entry (clozn.diagnosis-narrative.v1's `plausible_entry` has no such properties
 * at all). */
export interface RepairPlausibleUnproven {
  text: string;
  note: string;
  evidence: RepairNarrativeEvidenceRef[];
}

export interface RepairNarrative {
  schemaVersion: string;
  generatedAt: string;
  runId: string;
  comparisonRunId?: string;
  comparisonAvailable: boolean;
  findingsSchemaVersion: string;
  headline: string;
  observedChanges: RepairObservedChange[];
  measuredEffects: RepairMeasuredEffect[];
  plausibleButUnproven: RepairPlausibleUnproven[];
}

export interface RepairDiagnosis {
  findings: RepairFindingsDocument;
  narrative: RepairNarrative;
}

// ----------------------------------------------------------------------------------------------- parse

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
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

function bool(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function strArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

const FINDING_STATUSES = new Set<RepairFindingStatus>([
  "finding", "not_observed", "unavailable", "pending", "suppressed",
]);
const SEVERITIES = new Set<RepairSeverity>(["info", "low", "medium", "high"]);
const CONFIDENCES = new Set<RepairConfidence>(["exact", "pattern_match", "derived"]);
const SUGGESTED_ACTION_KINDS = new Set<RepairSuggestedActionKind>([
  "resend_context", "increase_context_budget", "reconcile_conflicting_instructions",
  "deduplicate_instructions", "deduplicate_source_content", "clarify_output_format",
  "move_instruction_near_request", "reinforce_low_effect_source", "resupply_below_floor_source",
  "restate_conversation_instruction", "increase_max_tokens", "reconfirm_run_configuration",
  "no_action_available",
]);
const OBSERVED_KINDS = new Set<RepairObservedKind>(["changed", "added", "removed", "unavailable", "diff_failed"]);

function findingStatus(value: unknown): RepairFindingStatus {
  return typeof value === "string" && FINDING_STATUSES.has(value as RepairFindingStatus)
    ? value as RepairFindingStatus
    : "unavailable";
}

function severity(value: unknown): RepairSeverity {
  return typeof value === "string" && SEVERITIES.has(value as RepairSeverity) ? value as RepairSeverity : "info";
}

function confidence(value: unknown): RepairConfidence {
  return typeof value === "string" && CONFIDENCES.has(value as RepairConfidence)
    ? value as RepairConfidence
    : "derived";
}

function suggestedActionKind(value: unknown): RepairSuggestedActionKind {
  return typeof value === "string" && SUGGESTED_ACTION_KINDS.has(value as RepairSuggestedActionKind)
    ? value as RepairSuggestedActionKind
    : "no_action_available";
}

function observedKind(value: unknown): RepairObservedKind {
  return typeof value === "string" && OBSERVED_KINDS.has(value as RepairObservedKind)
    ? value as RepairObservedKind
    : "unavailable";
}

function parseEvidence(raw: unknown): RepairEvidence | null {
  const item = record(raw);
  if (item.kind === "field" && typeof item.path === "string") {
    return { kind: "field", path: item.path, value: item.value, note: str(item.note) };
  }
  if (item.kind === "text_span" && typeof item.address_id === "string") {
    return {
      kind: "text_span",
      addressId: item.address_id,
      messageIndex: num(item.message_index),
      localStart: num(item.local_start),
      localEnd: num(item.local_end),
      note: str(item.note),
    };
  }
  return null;
}

function parseEvidenceList(raw: unknown): RepairEvidence[] {
  return records(raw).map(parseEvidence).filter((item): item is RepairEvidence => item != null);
}

function parseSuggestedAction(raw: unknown): RepairSuggestedAction {
  const item = record(raw);
  return { kind: suggestedActionKind(item.kind), description: str(item.description) ?? "" };
}

function parseFinding(raw: unknown): RepairFinding {
  const item = record(raw);
  const common: RepairFindingCommon = {
    ruleId: str(item.rule_id) ?? "",
    ruleName: str(item.rule_name) ?? "",
    summary: str(item.summary) ?? "",
    evidence: parseEvidenceList(item.evidence),
    limitations: strArray(item.limitations),
  };
  const status = findingStatus(item.status);
  if (status === "finding") {
    return {
      ...common,
      status,
      severity: severity(item.severity),
      confidence: confidence(item.confidence),
      suggestedActions: records(item.suggested_actions).map(parseSuggestedAction),
    };
  }
  return { ...common, status };
}

function parseFindingsDocument(raw: unknown): RepairFindingsDocument {
  const doc = record(raw);
  const counts = record(record(doc.summary).status_counts);
  return {
    schemaVersion: str(doc.schema_version) ?? "",
    generatedAt: str(doc.generated_at) ?? "",
    runId: str(doc.run_id) ?? "",
    comparisonRunId: str(doc.comparison_run_id),
    redacted: bool(doc.redacted),
    ruleRegistry: records(doc.rule_registry).map((entry) => ({
      ruleId: str(entry.rule_id) ?? "",
      ruleName: str(entry.rule_name) ?? "",
    })),
    suppressedRuleIds: strArray(doc.suppressed_rule_ids),
    findings: records(doc.findings).map(parseFinding),
    statusCounts: {
      finding: num(counts.finding) ?? 0,
      not_observed: num(counts.not_observed) ?? 0,
      unavailable: num(counts.unavailable) ?? 0,
      pending: num(counts.pending) ?? 0,
      suppressed: num(counts.suppressed) ?? 0,
    },
  };
}

function parseNarrativeEvidenceRef(raw: unknown): RepairNarrativeEvidenceRef | null {
  const item = record(raw);
  if (item.kind === "finding" && typeof item.rule_id === "string") return { kind: "finding", ruleId: item.rule_id };
  if (item.kind === "diff_field" && typeof item.dimension === "string") {
    return { kind: "diff_field", dimension: item.dimension };
  }
  if (item.kind === "text_span" && typeof item.address_id === "string") {
    return { kind: "text_span", addressId: item.address_id };
  }
  return null;
}

function parseNarrativeEvidence(raw: unknown): RepairNarrativeEvidenceRef[] {
  return records(raw).map(parseNarrativeEvidenceRef).filter(
    (item): item is RepairNarrativeEvidenceRef => item != null,
  );
}

function parseObservedChange(raw: unknown): RepairObservedChange {
  const item = record(raw);
  return {
    dimension: str(item.dimension) ?? "",
    kind: observedKind(item.kind),
    text: str(item.text) ?? "",
    evidence: parseNarrativeEvidence(item.evidence),
  };
}

function parseMeasuredEffect(raw: unknown): RepairMeasuredEffect {
  const item = record(raw);
  return {
    rank: num(item.rank) ?? 0,
    basis: item.basis === "intervention" ? "intervention" : "rule_finding",
    ruleId: str(item.rule_id) ?? "",
    severity: severity(item.severity),
    confidence: confidence(item.confidence),
    text: str(item.text) ?? "",
    evidence: parseNarrativeEvidence(item.evidence),
  };
}

function parsePlausible(raw: unknown): RepairPlausibleUnproven {
  const item = record(raw);
  return {
    text: str(item.text) ?? "",
    note: str(item.note) ?? "",
    evidence: parseNarrativeEvidence(item.evidence),
  };
}

function parseNarrative(raw: unknown): RepairNarrative {
  const doc = record(raw);
  const registers = record(doc.registers);
  return {
    schemaVersion: str(doc.schema_version) ?? "",
    generatedAt: str(doc.generated_at) ?? "",
    runId: str(doc.run_id) ?? "",
    comparisonRunId: str(doc.comparison_run_id),
    comparisonAvailable: bool(doc.comparison_available),
    findingsSchemaVersion: str(doc.findings_schema_version) ?? "",
    headline: str(doc.headline) ?? "",
    observedChanges: records(registers.observed_changes).map(parseObservedChange),
    measuredEffects: records(registers.measured_effects).map(parseMeasuredEffect),
    plausibleButUnproven: records(registers.plausible_but_unproven).map(parsePlausible),
  };
}

export class DiagnosisRepairLoadError extends Error {}

/** The ONE fetch this whole feature is allowed to make on render -- a GET, nothing else. Rendering the
 * diagnosis panel must never itself run a rule, mutate a run, or start a corrective action. */
export async function loadDiagnosisRepair(
  runId: string,
  options: { compareRunId?: string; signal?: AbortSignal } = {},
): Promise<RepairDiagnosis> {
  const query = options.compareRunId ? `?compare=${encodeURIComponent(options.compareRunId)}` : "";
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/diagnosis-findings${query}`, {
    signal: options.signal,
  });
  const body = record(await response.json().catch(() => ({})));
  if (!response.ok) {
    throw new DiagnosisRepairLoadError(str(body.error) ?? `diagnosis-findings request failed (${response.status})`);
  }
  return { findings: parseFindingsDocument(body.findings), narrative: parseNarrative(body.narrative) };
}
