import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import {
  CORRECTIVE_ACTION_IDS,
  diagnosisActionsApi,
  type CorrectivePreview,
  type CorrectiveRegistry,
  type CorrectiveResult,
  type DiagnosisDocument,
} from "../../data/diagnosisActions";
import { InspectSurface } from "./Inspect";

const runId = "run_diagnosis";

function diagnosisFixture(id = runId, summaryPrefix = "Recorded"): DiagnosisDocument {
  return {
    findings: {
      schemaVersion: "clozn.diagnosis-findings.v1",
      generatedAt: "2026-10-10T00:00:00Z",
      runId: id,
      redacted: false,
      ruleRegistry: [
        { ruleId: "R01", ruleName: "input_omitted" },
        { ruleId: "R02", ruleName: "budget_pressure" },
        { ruleId: "R03", ruleName: "conflicting_instructions" },
        { ruleId: "R04", ruleName: "duplicate_instructions" },
        { ruleId: "R05", ruleName: "format_drift" },
      ],
      suppressedRuleIds: ["R05"],
      findings: [
        {
          ruleId: "R01", ruleName: "input_omitted", status: "finding", severity: "medium",
          confidence: "exact", summary: `${summaryPrefix} omitted input.`, evidence: [], limitations: [],
          suggestedActions: [{ kind: "resend_context", description: "Resend the omitted source." }],
        },
        {
          ruleId: "R02", ruleName: "budget_pressure", status: "not_observed",
          summary: "No budget pressure was observed.", evidence: [], limitations: [],
        },
        {
          ruleId: "R03", ruleName: "conflicting_instructions", status: "unavailable",
          summary: "Conflict evidence was unavailable.", evidence: [], limitations: [],
        },
        {
          ruleId: "R04", ruleName: "duplicate_instructions", status: "pending",
          summary: "Duplicate detection was not measured.", evidence: [], limitations: [],
        },
        {
          ruleId: "R05", ruleName: "format_drift", status: "suppressed",
          summary: "Format drift was excluded.", evidence: [], limitations: [],
        },
      ],
      statusCounts: { finding: 1, not_observed: 1, unavailable: 1, pending: 1, suppressed: 1 },
    },
    narrative: {
      schemaVersion: "clozn.diagnosis-narrative.v1",
      generatedAt: "2026-10-10T00:00:00Z",
      runId: id,
      comparisonAvailable: true,
      findingsSchemaVersion: "clozn.diagnosis-findings.v1",
      headline: `${summaryPrefix} diagnosis headline.`,
      measuredEffects: [{
        rank: 1, basis: "rule_finding", ruleId: "R01", severity: "medium", confidence: "exact",
        text: "The omitted input changed the answer.", evidence: [{ kind: "finding", ruleId: "R01" }],
      }],
      observedChanges: [{
        dimension: "answer", kind: "changed", text: "The answer changed.",
        evidence: [{ kind: "diff_field", dimension: "answer" }],
      }],
      plausibleButUnproven: [{
        text: "Formatting may also matter.", note: "No intervention isolated formatting.",
        evidence: [{ kind: "finding", ruleId: "R01" }],
      }],
    },
  };
}

function registryFixture(id = runId): CorrectiveRegistry {
  return {
    schemaVersion: "clozn.action-registry.v1",
    version: "1",
    runId: id,
    runFingerprint: "a".repeat(64),
    actions: CORRECTIVE_ACTION_IDS.map((actionId, index) => ({
      id: actionId,
      label: ["More concise", "More concrete", "Use context", "Ask before guessing", "Preserve formatting", "Stop repeating"][index]!,
      description: `Apply ${actionId} to this matched retry.`,
      conflicts: [],
      scopes: ["once"],
      eligibility: { eligible: true },
      evaluationMetrics: [],
      backends: [{
        type: "prompt_policy",
        recipeVersion: "1",
        parameters: { preset: actionId },
        qualification: "generic",
        qualificationId: "clozn.prompt-policy.generic.v1",
        available: true,
      }],
      scopeEligibility: [{
        scope: "once", available: true, target: id, priorHash: "b".repeat(64),
        note: "Selects this corrected child only for the recorded Run.",
      }],
    })),
  };
}

function previewFixture(id = runId): CorrectivePreview {
  return {
    previewId: `fix_preview_${"a".repeat(20)}`,
    status: "ready",
    createdTs: 1,
    expiresTs: 2,
    parentRunId: id,
    parentRunFingerprint: "c".repeat(64),
    action: { id: "less-verbose", label: "More concise", description: "Answer concisely." },
    execution: {
      requestedBackend: "prompt_policy",
      expectedExecutedBackend: "prompt_policy",
      expectedFallback: false,
      qualification: "generic",
      qualificationId: "clozn.prompt-policy.generic.v1",
    },
    scopeEligibility: [{ scope: "once", available: true, priorHash: "b".repeat(64) }],
    comparisonContract: {
      baseline: "matched baseline replay",
      corrected: "matched replay with the action",
      storedOriginal: "recorded original reply",
    },
  };
}

function resultFixture(kept = false): CorrectiveResult {
  const resultId = `fix_result_${"d".repeat(20)}`;
  return {
    schemaVersion: "clozn.corrective-retry-result.v1",
    resultId,
    createdTs: 3,
    parentRunId: runId,
    previewId: `fix_preview_${"a".repeat(20)}`,
    action: { id: "less-verbose", label: "More concise", description: "Answer concisely." },
    userIntent: { actionId: "less-verbose", ...(kept ? { keptScope: "once" as const } : {}) },
    execution: {
      requestedBackend: "prompt_policy", executedBackend: "prompt_policy", fallback: false,
      qualification: "generic", qualificationId: "clozn.prompt-policy.generic.v1",
      identity: { model: "test-model", decoding: "greedy" },
    },
    children: {
      baseline: { status: "success", runId: "run_baseline" },
      corrected: { status: "success", runId: "run_corrected" },
    },
    comparison: {
      storedOriginalReply: "Stored original reply.", baselineReply: "Matched baseline reply.",
      correctedReply: "Corrected reply.", note: "Matched greedy comparison.", changed: true,
    },
    metrics: { word_delta: -4, formatting_preserved: true },
    scopeEligibility: [{
      scope: "once", available: true, target: runId, priorHash: "b".repeat(64),
      note: "Selects this corrected child only for the recorded Run.",
    }],
    outcome: { status: "succeeded" },
    ...(kept ? {
      transaction: {
        id: "corrective_transaction_1", scope: "once" as const, target: runId, before: null,
        after: { resultId, childRunId: "run_corrected", actionId: "less-verbose" as const, selectedTs: 4 },
        createdTs: 4, undoneTs: null,
      },
    } : {}),
  };
}

function installInitialReads() {
  const diagnosis = vi.spyOn(diagnosisActionsApi, "diagnosis").mockResolvedValue(diagnosisFixture());
  const registry = vi.spyOn(diagnosisActionsApi, "registry").mockResolvedValue(registryFixture());
  return { diagnosis, registry };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Diagnosis and corrective actions Inspect instrument", () => {
  it("loads only diagnosis and the six-action prompt-policy registry and renders every evidence register and finding state", async () => {
    const { diagnosis, registry } = installInitialReads();
    const preview = vi.spyOn(diagnosisActionsApi, "preview");
    const confirm = vi.spyOn(diagnosisActionsApi, "confirm");
    const cancel = vi.spyOn(diagnosisActionsApi, "cancel");
    const keep = vi.spyOn(diagnosisActionsApi, "keep");
    const undo = vi.spyOn(diagnosisActionsApi, "undo");

    render(<InspectSurface route={{ surface: "inspect", runId, view: "actions" }} />);

    expect(await screen.findByText("Recorded omitted input.")).toBeInTheDocument();
    expect(diagnosis).toHaveBeenCalledOnce();
    expect(registry).toHaveBeenCalledOnce();
    expect(screen.getByRole("region", { name: "Measured effects" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Observed changes" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Plausible but unproven" })).toBeInTheDocument();

    for (const status of ["finding", "not_observed", "unavailable", "pending", "suppressed"]) {
      expect(document.querySelector(`[data-finding-status="${status}"]`)).not.toBeNull();
    }
    const finding = document.getElementById("diagnosis-finding-R01")!;
    expect(within(finding).getByText("Resend the omitted source.")).toBeInTheDocument();
    expect(within(finding).queryByRole("button", { name: "Resend the omitted source." })).not.toBeInTheDocument();
    expect(screen.getAllByText("prompt_policy")).toHaveLength(6);
    expect(screen.getByLabelText("Available corrective actions").querySelectorAll("button")).toHaveLength(6);
    expect(preview).not.toHaveBeenCalled();
    expect(confirm).not.toHaveBeenCalled();
    expect(cancel).not.toHaveBeenCalled();
    expect(keep).not.toHaveBeenCalled();
    expect(undo).not.toHaveBeenCalled();
  });

  it("requires preview then confirm, supports cancellation, and exposes native child Inspect and Compare links", async () => {
    installInitialReads();
    const preview = vi.spyOn(diagnosisActionsApi, "preview").mockResolvedValue(previewFixture());
    const cancel = vi.spyOn(diagnosisActionsApi, "cancel").mockResolvedValue({ ...previewFixture(), status: "cancel_requested" });
    let resolveCancelledConfirm!: (value: CorrectiveResult) => void;
    const confirm = vi.spyOn(diagnosisActionsApi, "confirm")
      .mockImplementationOnce(() => new Promise((resolve) => { resolveCancelledConfirm = resolve; }))
      .mockResolvedValueOnce(resultFixture());
    render(<InspectSurface route={{ surface: "inspect", runId, view: "actions" }} />);

    fireEvent.click(await screen.findByRole("button", { name: /More concise/ }));
    expect(await screen.findByLabelText("Corrective retry preview")).toBeInTheDocument();
    expect(preview).toHaveBeenCalledWith(runId, "less-verbose", expect.any(AbortSignal));
    expect(confirm).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /Confirm/ }));
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce());
    expect(screen.getByRole("button", { name: "Cancel preview" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel preview" }));
    await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
    resolveCancelledConfirm(resultFixture());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByLabelText("Corrective retry preview")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Corrective retry result")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /More concise/ }));
    await screen.findByLabelText("Corrective retry preview");
    fireEvent.click(screen.getByRole("button", { name: /Confirm/ }));
    expect(await screen.findByText("Corrected reply.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Inspect parent" })).toHaveAttribute("href", `#/inspect/${runId}`);
    expect(screen.getAllByRole("link", { name: "Inspect child" }).map((link) => link.getAttribute("href"))).toEqual([
      "#/inspect/run_baseline", "#/inspect/run_corrected",
    ]);
    expect(screen.getByRole("link", { name: "Compare matched children" })).toHaveAttribute(
      "href", "#/compare/run_baseline/run_corrected",
    );
    expect(screen.getByLabelText("Execution identity")).toHaveTextContent("modeltest-model");
    expect(screen.getByLabelText("Result metrics")).toHaveTextContent("word_delta-4");
    expect(screen.getByLabelText("Result metrics")).toHaveTextContent("formatting_preservedtrue");
  });

  it("preserves the preview and confirmation identity when in-flight cancellation loses the race", async () => {
    installInitialReads();
    vi.spyOn(diagnosisActionsApi, "preview").mockResolvedValue(previewFixture());
    let resolveFirstConfirm!: (value: CorrectiveResult) => void;
    const confirm = vi.spyOn(diagnosisActionsApi, "confirm")
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirstConfirm = resolve; }))
      .mockResolvedValueOnce(resultFixture());
    vi.spyOn(diagnosisActionsApi, "cancel").mockRejectedValue(
      new Error("Request to cancel failed (409 Conflict)"),
    );
    render(<InspectSurface route={{ surface: "inspect", runId, view: "actions" }} />);

    fireEvent.click(await screen.findByRole("button", { name: /More concise/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Confirm/ }));
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce());
    const firstSignal = confirm.mock.calls[0]?.[2];

    fireEvent.click(screen.getByRole("button", { name: "Cancel preview" }));
    expect(await screen.findByText(/Confirm again to recover the active or completed result/)).toBeInTheDocument();
    expect(firstSignal?.aborted).toBe(true);
    expect(screen.getByLabelText("Corrective retry preview")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Confirm/ })).toBeEnabled();

    resolveFirstConfirm(resultFixture());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByLabelText("Corrective retry result")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Confirm/ }));
    expect(await screen.findByText("Corrected reply.")).toBeInTheDocument();
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(confirm.mock.calls[1]?.[1].idempotencyKey).toBe(confirm.mock.calls[0]?.[1].idempotencyKey);
  });

  it("reuses idempotency keys for retries of the same confirm and keep, then offers a separate undo", async () => {
    installInitialReads();
    vi.spyOn(diagnosisActionsApi, "preview").mockResolvedValue(previewFixture());
    const confirm = vi.spyOn(diagnosisActionsApi, "confirm")
      .mockRejectedValueOnce(new Error("temporary confirm failure"))
      .mockResolvedValueOnce(resultFixture());
    const keep = vi.spyOn(diagnosisActionsApi, "keep")
      .mockRejectedValueOnce(new Error("temporary keep failure"))
      .mockResolvedValueOnce(resultFixture(true));
    const undo = vi.spyOn(diagnosisActionsApi, "undo").mockResolvedValue({
      status: "undone", undoId: "corrective_transaction_1", scope: "once", target: runId,
    });
    render(<InspectSurface route={{ surface: "inspect", runId, view: "actions" }} />);

    fireEvent.click(await screen.findByRole("button", { name: /More concise/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Confirm/ }));
    expect(await screen.findByText(/temporary confirm failure/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Confirm/ }));
    await screen.findByText("Corrected reply.");
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(confirm.mock.calls[0]?.[1].idempotencyKey).toBe(confirm.mock.calls[1]?.[1].idempotencyKey);

    fireEvent.click(screen.getByRole("button", { name: "Keep correction" }));
    expect(await screen.findByText(/temporary keep failure/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Keep correction" }));
    expect(await screen.findByText("Correction kept for this Run")).toBeInTheDocument();
    expect(keep).toHaveBeenCalledTimes(2);
    expect(keep.mock.calls[0]?.[1].idempotencyKey).toBe(keep.mock.calls[1]?.[1].idempotencyKey);

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(await screen.findByText("Correction undone")).toBeInTheDocument();
    expect(screen.getByText("Correction undone; recorded Run restored")).toBeInTheDocument();
    expect(undo).toHaveBeenCalledWith("corrective_transaction_1", expect.any(AbortSignal));
  });

  it("never paints old reads or mutation results after the selected Run changes", async () => {
    let resolveOldDiagnosis!: (value: DiagnosisDocument) => void;
    let resolveOldPreview!: (value: CorrectivePreview) => void;
    const diagnosis = vi.spyOn(diagnosisActionsApi, "diagnosis")
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOldDiagnosis = resolve; }))
      .mockResolvedValueOnce(diagnosisFixture("run_new", "New"));
    vi.spyOn(diagnosisActionsApi, "registry")
      .mockResolvedValueOnce(registryFixture("run_old"))
      .mockResolvedValueOnce(registryFixture("run_new"));
    vi.spyOn(diagnosisActionsApi, "preview")
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOldPreview = resolve; }));
    const view = render(<InspectSurface route={{ surface: "inspect", runId: "run_old", view: "actions" }} />);

    fireEvent.click(await screen.findByRole("button", { name: /More concise/ }));
    view.rerender(<InspectSurface route={{ surface: "inspect", runId: "run_new", view: "actions" }} />);
    expect(await screen.findByText("New omitted input.")).toBeInTheDocument();
    resolveOldDiagnosis(diagnosisFixture("run_old", "Old"));
    resolveOldPreview(previewFixture("run_old"));

    await waitFor(() => expect(diagnosis).toHaveBeenCalledTimes(2));
    expect(screen.queryByText("Old omitted input.")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Corrective retry preview")).not.toBeInTheDocument();
  });
});
