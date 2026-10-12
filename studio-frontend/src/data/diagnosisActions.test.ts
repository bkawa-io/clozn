import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractError } from "./contracts";
import {
  CORRECTIVE_ACTION_IDS,
  decodeCorrectivePreview,
  decodeCorrectiveRegistry,
  decodeCorrectiveResult,
  decodeDiagnosisDocument,
  diagnosisActionsApi,
  preserveCorrectiveIdempotencyKey,
} from "./diagnosisActions";

const runId = "run_contract";
const previewId = `fix_preview_${"a".repeat(20)}`;
const resultId = `fix_result_${"b".repeat(20)}`;
const priorHash = "c".repeat(64);
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

function diagnosisBody() {
  return {
    findings: {
      schema_version: "clozn.diagnosis-findings.v1",
      generated_at: "2026-10-10T00:00:00Z",
      run_id: runId,
      redacted: false,
      rule_registry: DIAGNOSIS_RULES.map(([rule_id, rule_name]) => ({ rule_id, rule_name })),
      suppressed_rule_ids: ["R05"],
      findings: [
        {
          rule_id: "R01",
          rule_name: "input_omitted_or_rejected",
          status: "finding",
          severity: "medium",
          confidence: "exact",
          summary: "One input was omitted.",
          evidence: [{ kind: "field", path: "receipt.omissions", value: 1 }],
          limitations: [],
          suggested_actions: [{ kind: "resend_context", description: "Resend the omitted input." }],
        },
        {
          rule_id: "R02", rule_name: "context_budget_pressure", status: "not_observed",
          summary: "No budget pressure was observed.", evidence: [], limitations: [],
        },
        {
          rule_id: "R03", rule_name: "conflicting_instructions", status: "unavailable",
          summary: "Conflict evidence is unavailable.", evidence: [], limitations: [],
        },
        {
          rule_id: "R04", rule_name: "duplicate_instructions", status: "pending",
          summary: "Duplicate detection was not measured.", evidence: [], limitations: [],
        },
        {
          rule_id: "R05", rule_name: "repeated_source_content", status: "suppressed",
          summary: "Format drift was excluded.", evidence: [], limitations: [],
        },
        ...DIAGNOSIS_RULES.slice(5).map(([rule_id, rule_name]) => ({
          rule_id, rule_name, status: "not_observed", summary: `${rule_id} was not observed.`,
          evidence: [], limitations: [],
        })),
      ],
      summary: { status_counts: { finding: 1, not_observed: 8, unavailable: 1, pending: 1, suppressed: 1 } },
    },
    narrative: {
      schema_version: "clozn.diagnosis-narrative.v1",
      generated_at: "2026-10-10T00:00:00Z",
      run_id: runId,
      comparison_available: false,
      findings_schema_version: "clozn.diagnosis-findings.v1",
      headline: "One finding was recorded.",
      registers: {
        observed_changes: [],
        measured_effects: [{
          rank: 1,
          basis: "rule_finding",
          rule_id: "R01",
          severity: "medium",
          confidence: "exact",
          text: "The receipt records an omission.",
          evidence: [{ kind: "finding", rule_id: "R01" }],
        }],
        plausible_but_unproven: [],
      },
      summary: { counts: { observed_changes: 0, measured_effects: 1, plausible_but_unproven: 0 } },
    },
  };
}

function scopeEligibility() {
  return [{ scope: "once", available: true, target: runId, prior_hash: priorHash }];
}

function registryBody() {
  return {
    schema_version: "clozn.action-registry.v1",
    version: "1",
    run_id: runId,
    run_fingerprint: "d".repeat(64),
    actions: CORRECTIVE_ACTION_IDS.map((id) => ({
      id,
      label: id,
      description: `Apply ${id}.`,
      conflicts: [],
      scopes: ["once"],
      eligibility: { eligible: true },
      evaluation_metrics: [],
      backends: [{
        type: "prompt_policy",
        recipe_version: "1",
        parameters: { preset: id },
        qualification: "generic",
        qualification_id: "clozn.prompt-policy.generic.v1",
        available: true,
      }],
      scope_eligibility: scopeEligibility(),
    })),
  };
}

function previewBody() {
  return {
    preview_id: previewId,
    status: "ready",
    created_ts: 1,
    expires_ts: 2,
    parent_run_id: runId,
    parent_run_fingerprint: "d".repeat(64),
    action: { id: "less-verbose", label: "More concise", description: "Answer concisely." },
    execution: {
      requested_backend: "prompt_policy",
      expected_executed_backend: "prompt_policy",
      expected_fallback: false,
      qualification: "generic",
      qualification_id: "clozn.prompt-policy.generic.v1",
    },
    scope_eligibility: scopeEligibility(),
    comparison_contract: {
      baseline: "matched baseline replay",
      corrected: "matched corrected replay",
      stored_original: "recorded original",
    },
  };
}

function resultBody() {
  return {
    schema_version: "clozn.corrective-retry-result.v1",
    result_id: resultId,
    created_ts: 3,
    parent_run_id: runId,
    preview_id: previewId,
    action: { id: "less-verbose", label: "More concise", description: "Answer concisely." },
    user_intent: { action_id: "less-verbose" },
    execution: {
      requested_backend: "prompt_policy",
      executed_backend: "prompt_policy",
      fallback: false,
      qualification: "generic",
      qualification_id: "clozn.prompt-policy.generic.v1",
      identity: {},
    },
    children: {
      baseline: { status: "success", run_id: "run_baseline" },
      corrected: { status: "success", run_id: "run_corrected" },
    },
    comparison: {
      stored_original_reply: "Original.",
      baseline_reply: "Baseline.",
      corrected_reply: "Corrected.",
      note: "Matched comparison.",
      changed: true,
    },
    metrics: {},
    scope_eligibility: scopeEligibility(),
    outcome: { status: "succeeded" },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("diagnosis and corrective action contracts", () => {
  it("decodes the five-state diagnosis vocabulary and the closed six-action prompt-policy registry", () => {
    const diagnosis = decodeDiagnosisDocument(diagnosisBody(), "/diagnosis", runId);
    expect(diagnosis.findings.findings.slice(0, 5).map((finding) => finding.status)).toEqual([
      "finding", "not_observed", "unavailable", "pending", "suppressed",
    ]);
    expect(diagnosis.findings.findings).toHaveLength(12);
    expect(diagnosis.findings.findings[0]).toMatchObject({ status: "finding", severity: "medium", confidence: "exact" });
    expect(diagnosis.narrative.measuredEffects[0]).toMatchObject({ rank: 1, ruleId: "R01" });

    const registry = decodeCorrectiveRegistry(registryBody(), "/registry", runId);
    expect(registry.actions.map((action) => action.id)).toEqual(CORRECTIVE_ACTION_IDS);
    expect(registry.actions.every((action) => action.backends.some((backend) => backend.type === "prompt_policy"))).toBe(true);
  });

  it("fails closed on unknown statuses, incomplete registries, non-prompt previews, and mismatched identities", () => {
    const unknownFinding = structuredClone(diagnosisBody());
    unknownFinding.findings.findings[0]!.status = "maybe";
    expect(() => decodeDiagnosisDocument(unknownFinding, "/diagnosis", runId)).toThrow(ContractError);

    const shortRuleRegistry = structuredClone(diagnosisBody());
    shortRuleRegistry.findings.rule_registry.pop();
    expect(() => decodeDiagnosisDocument(shortRuleRegistry, "/diagnosis", runId)).toThrow(/exactly 12 rules/);

    const outOfOrderRules = structuredClone(diagnosisBody());
    outOfOrderRules.findings.rule_registry.reverse();
    expect(() => decodeDiagnosisDocument(outOfOrderRules, "/diagnosis", runId)).toThrow(/must be R01/);

    const duplicateSuppression = structuredClone(diagnosisBody());
    duplicateSuppression.findings.suppressed_rule_ids.push("R05");
    expect(() => decodeDiagnosisDocument(duplicateSuppression, "/diagnosis", runId)).toThrow(/duplicate rule ids/);

    const missingSuppression = structuredClone(diagnosisBody());
    missingSuppression.findings.suppressed_rule_ids = [];
    expect(() => decodeDiagnosisDocument(missingSuppression, "/diagnosis", runId)).toThrow(/exactly match suppressed findings/);

    const falseSuppression = structuredClone(diagnosisBody());
    falseSuppression.findings.suppressed_rule_ids = ["R05", "R06"];
    expect(() => decodeDiagnosisDocument(falseSuppression, "/diagnosis", runId)).toThrow(/exactly match suppressed findings/);

    const measuredWithoutFinding = structuredClone(diagnosisBody());
    measuredWithoutFinding.narrative.registers.measured_effects[0]!.rule_id = "R02";
    expect(() => decodeDiagnosisDocument(measuredWithoutFinding, "/diagnosis", runId)).toThrow(/diagnosis finding/);

    const rankedHypothesis = structuredClone(diagnosisBody());
    const hypothesisRegister = rankedHypothesis.narrative.registers.plausible_but_unproven as unknown as Array<Record<string, unknown>>;
    hypothesisRegister.push({
      text: "Sampling might matter.", note: "Not measured.", rank: 1,
      evidence: [{ kind: "finding", rule_id: "R01" }],
    });
    (rankedHypothesis.narrative.summary.counts as Record<string, number>).plausible_but_unproven = 1;
    expect(() => decodeDiagnosisDocument(rankedHypothesis, "/diagnosis", runId)).toThrow(/not part of the contract/);

    expect(() => decodeDiagnosisDocument(diagnosisBody(), "/diagnosis", "run_other")).toThrow(/requested run/);

    const incompleteRegistry = structuredClone(registryBody());
    incompleteRegistry.actions.pop();
    expect(() => decodeCorrectiveRegistry(incompleteRegistry, "/registry", runId)).toThrow(/six corrective actions/);

    const malformedPriorHash = structuredClone(registryBody());
    malformedPriorHash.actions[0]!.scope_eligibility[0]!.prior_hash = "not-a-sha";
    expect(() => decodeCorrectiveRegistry(malformedPriorHash, "/registry", runId)).toThrow(/invalid format/);

    const missingAvailableTarget = structuredClone(registryBody());
    delete (missingAvailableTarget.actions[0]!.scope_eligibility[0] as { target?: string }).target;
    expect(() => decodeCorrectiveRegistry(missingAvailableTarget, "/registry", runId)).toThrow(/target and prior_hash/);

    const mismatchedRegistryTarget = structuredClone(registryBody());
    mismatchedRegistryTarget.actions[0]!.scope_eligibility[0]!.target = "run_other";
    expect(() => decodeCorrectiveRegistry(mismatchedRegistryTarget, "/registry", runId)).toThrow(/target must match run_id/);

    const samplingPreview = structuredClone(previewBody());
    samplingPreview.execution.requested_backend = "sampling_policy";
    expect(() => decodeCorrectivePreview(samplingPreview, "/preview", { parentRunId: runId })).toThrow(ContractError);
    expect(() => decodeCorrectivePreview(previewBody(), "/preview", { previewId: `fix_preview_${"f".repeat(20)}` }))
      .toThrow(/does not match the request/);
    expect(() => decodeCorrectivePreview(previewBody(), "/preview", { actionId: "more-concrete" }))
      .toThrow(/does not match the request/);
    const mismatchedPreviewTarget = structuredClone(previewBody());
    mismatchedPreviewTarget.scope_eligibility[0]!.target = "run_other";
    expect(() => decodeCorrectivePreview(mismatchedPreviewTarget, "/preview", { parentRunId: runId }))
      .toThrow(/must match parent_run_id/);

    const unknownChild = structuredClone(resultBody());
    unknownChild.children.corrected.status = "partial";
    expect(() => decodeCorrectiveResult(unknownChild, "/result", { resultId })).toThrow(ContractError);
    expect(() => decodeCorrectiveResult(resultBody(), "/result", { resultId: `fix_result_${"e".repeat(20)}` }))
      .toThrow(/does not match the request/);
    expect(() => decodeCorrectiveResult(resultBody(), "/result", { previewId: `fix_preview_${"e".repeat(20)}` }))
      .toThrow(/does not match the request/);
    expect(() => decodeCorrectiveResult(resultBody(), "/result", { parentRunId: "run_other" }))
      .toThrow(/does not match the request/);
    const mismatchedResultTarget = structuredClone(resultBody());
    mismatchedResultTarget.scope_eligibility[0]!.target = "run_other";
    expect(() => decodeCorrectiveResult(mismatchedResultTarget, "/result", { resultId }))
      .toThrow(/must match parent_run_id/);
  });

  it("retains typed partial child failure and cancelled outcome states without flattening them", () => {
    const body = structuredClone(resultBody());
    body.children.corrected = {
      status: "error",
      run_id: "run_corrected_failed",
      error: { code: "worker_failed", message: "The corrected arm failed." },
    } as unknown as typeof body.children.corrected;
    (body as unknown as { outcome: { status: string; note?: string } }).outcome = {
      status: "cancelled", note: "Cancellation won the confirmation race.",
    };
    body.execution.executed_backend = null as unknown as string;
    body.execution.qualification = null as unknown as string;
    body.execution.qualification_id = null as unknown as string;
    body.execution.identity = null as unknown as Record<string, never>;

    const decoded = decodeCorrectiveResult(body, "/result", { resultId, previewId, parentRunId: runId });
    expect(decoded.children.baseline).toEqual({ status: "success", runId: "run_baseline" });
    expect(decoded.children.corrected).toEqual({
      status: "error",
      runId: "run_corrected_failed",
      error: { code: "worker_failed", message: "The corrected arm failed." },
    });
    expect(decoded.outcome).toEqual({ status: "cancelled", note: "Cancellation won the confirmation race." });
  });

  it("decodes the canonical coherence reason receipt and rejects the non-canonical reasons array", () => {
    const body = structuredClone(resultBody()) as ReturnType<typeof resultBody> & {
      coherence?: { degenerate: boolean; reason?: string; reasons?: string[] };
    };
    body.coherence = { degenerate: true, reason: "repeat-3gram" };
    expect(decodeCorrectiveResult(body, "/result", { resultId }).coherence).toEqual({
      degenerate: true,
      reason: "repeat-3gram",
    });

    body.coherence = { degenerate: true, reasons: ["repeat-3gram"] };
    expect(() => decodeCorrectiveResult(body, "/result", { resultId })).toThrow(/not part of the contract/);
  });

  it("validates idempotency keys without rewriting a logical request identity", () => {
    expect(preserveCorrectiveIdempotencyKey("confirm:stable-key")).toBe("confirm:stable-key");
    expect(preserveCorrectiveIdempotencyKey("confirm:stable-key")).toBe("confirm:stable-key");
    expect(() => preserveCorrectiveIdempotencyKey("short")).toThrow(ContractError);
  });

  it("uses GETs for initial reads and exact prompt-policy/idempotency bodies for explicit mutations", async () => {
    const responses = [
      new Response(JSON.stringify(diagnosisBody()), { status: 200 }),
      new Response(JSON.stringify(registryBody()), { status: 200 }),
      new Response(JSON.stringify(previewBody()), { status: 201 }),
      new Response(JSON.stringify(resultBody()), { status: 200 }),
      new Response(JSON.stringify(resultBody()), { status: 200 }),
      new Response(JSON.stringify({ status: "undone", undo_id: "transaction_1", scope: "once", target: runId }), { status: 200 }),
    ];
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(responses.shift()!));
    vi.stubGlobal("fetch", fetch);

    await diagnosisActionsApi.diagnosis(runId);
    await diagnosisActionsApi.registry(runId);
    await diagnosisActionsApi.preview(runId, "less-verbose");
    await diagnosisActionsApi.confirm(previewId, { idempotencyKey: "confirm:stable-key" });
    await diagnosisActionsApi.keep(resultId, { expectedPriorHash: priorHash, idempotencyKey: "keep:stable-key" });
    await diagnosisActionsApi.undo("transaction_1");

    expect(fetch.mock.calls[0]).toEqual([`/runs/${runId}/diagnosis-findings`, expect.objectContaining({ method: "GET" })]);
    expect(fetch.mock.calls[1]).toEqual([`/runs/${runId}/corrective-actions`, expect.objectContaining({ method: "GET" })]);
    expect(JSON.parse(fetch.mock.calls[2]![1].body)).toEqual({
      action_id: "less-verbose", requested_backend: "prompt_policy",
    });
    expect(JSON.parse(fetch.mock.calls[3]![1].body)).toEqual({ idempotency_key: "confirm:stable-key" });
    expect(JSON.parse(fetch.mock.calls[4]![1].body)).toEqual({
      scope: "once", expected_prior_hash: priorHash, idempotency_key: "keep:stable-key",
    });
    expect(fetch.mock.calls[5]![0]).toBe("/corrective-actions/transaction_1/undo");
  });
});
