import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, waitFor, within } from "../../test/render";
import { DiagnosisRepair } from "./DiagnosisRepair";

function pathOf(request: PendingFetch) {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname
      : request.input.url;
}

function requestFor(requests: PendingFetch[], suffix: string) {
  const request = requests.find((item) => pathOf(item).endsWith(suffix));
  if (!request) throw new Error(`missing request ending ${suffix}`);
  return request;
}

async function initialRequests(controller: ReturnType<typeof createFetchController>) {
  await waitFor(() => expect(controller.requests.length).toBeGreaterThanOrEqual(2));
  return {
    findings: requestFor(controller.requests, "/diagnosis-findings"),
    registry: requestFor(controller.requests, "/corrective-actions"),
  };
}

const FIVE_STATUS_FINDINGS = [
  {
    rule_id: "R01", rule_name: "input_omitted_or_rejected", status: "finding",
    severity: "medium", confidence: "exact",
    summary: "1 input segment was omitted.",
    evidence: [], limitations: [],
    suggested_actions: [{ kind: "resend_context", description: "resend the omitted segment" }],
  },
  {
    rule_id: "R02", rule_name: "context_budget_pressure", status: "not_observed",
    summary: "the prompt used 10% of context tokens.", evidence: [], limitations: [],
  },
  {
    rule_id: "R08", rule_name: "source_below_measurement_floor", status: "unavailable",
    summary: "no readable influence artifact was recorded.", evidence: [], limitations: [],
  },
  {
    rule_id: "R09", rule_name: "source_little_effect", status: "pending",
    summary: "this run never recorded an influence map.", evidence: [], limitations: [],
  },
  {
    rule_id: "R03", rule_name: "conflicting_instructions", status: "suppressed",
    summary: "R03 was suppressed for this evaluation by caller request.", evidence: [], limitations: [],
  },
  {
    rule_id: "R04", rule_name: "duplicate_instructions", status: "finding",
    severity: "low", confidence: "pattern_match",
    summary: "duplicate instructions were found.",
    evidence: [], limitations: [],
    suggested_actions: [],
  },
];

function diagnosisFindingsBody(runId: string, findings: Record<string, unknown>[] = FIVE_STATUS_FINDINGS) {
  const counts = { finding: 0, not_observed: 0, unavailable: 0, pending: 0, suppressed: 0 };
  for (const finding of findings) counts[finding.status as keyof typeof counts] += 1;
  return {
    findings: {
      schema_version: "clozn.diagnosis-findings.v1",
      generated_at: "2026-01-01T00:00:00Z",
      run_id: runId,
      redacted: false,
      rule_registry: findings.map((f) => ({ rule_id: f.rule_id, rule_name: f.rule_name })),
      suppressed_rule_ids: findings.filter((f) => f.status === "suppressed").map((f) => f.rule_id),
      findings,
      summary: { status_counts: counts },
    },
    narrative: {
      schema_version: "clozn.diagnosis-narrative.v1",
      generated_at: "2026-01-01T00:00:00Z",
      run_id: runId,
      comparison_available: false,
      findings_schema_version: "clozn.diagnosis-findings.v1",
      headline: "no comparison run supplied.",
      registers: { observed_changes: [], measured_effects: [], plausible_but_unproven: [] },
      summary: { counts: { observed_changes: 0, measured_effects: 0, plausible_but_unproven: 0 } },
    },
  };
}

function correctiveRegistryBody(runId: string) {
  return {
    schema_version: "clozn.action-registry.v1",
    version: "1",
    run_id: runId,
    run_fingerprint: "fp-" + runId,
    actions: [
      {
        id: "less-verbose",
        label: "More concise",
        description: "For this reply, answer concisely.",
        conflicts: [],
        scopes: ["once", "session", "profile"],
        eligibility: { eligible: true },
        evaluation_metrics: [],
        backends: [{ type: "prompt_policy", available: true }],
        scope_eligibility: [
          { scope: "once", available: true, prior_hash: "hash-once" },
          { scope: "session", available: false, unavailability_reason: "the run has no exact opaque session association" },
          { scope: "profile", available: false, unavailability_reason: "the run did not capture an active profile" },
        ],
      },
    ],
  };
}

function previewBody() {
  return {
    preview_id: `fix_preview_${"a".repeat(20)}`,
    status: "ready",
    created_ts: 1700000000,
    expires_ts: 1700003600,
    parent_run_id: "run-1",
    parent_run_fingerprint: "fp",
    action: { id: "less-verbose", label: "More concise", description: "For this reply, answer concisely." },
    execution: {
      requested_backend: "prompt_policy",
      expected_executed_backend: "prompt_policy",
      expected_fallback: false,
      qualification: "generic",
      qualification_id: "clozn.prompt-policy.generic.v1",
    },
    scope_eligibility: [
      { scope: "once", available: true, prior_hash: "hash-once" },
      { scope: "session", available: false, unavailability_reason: "no session" },
      { scope: "profile", available: false, unavailability_reason: "no profile" },
    ],
    comparison_contract: {
      baseline: "matched greedy replay under the current runtime policy",
      corrected: "matched greedy replay with the bounded action",
      stored_original: "context only; it may have been sampled",
    },
  };
}

async function setUpToPreview(controller: ReturnType<typeof createFetchController>, user: ReturnType<typeof userEvent.setup>) {
  render(<DiagnosisRepair runId="run-1" />);
  const requests = await initialRequests(controller);
  controller.respondJson(requests.findings, diagnosisFindingsBody("run-1"));
  controller.respondJson(requests.registry, correctiveRegistryBody("run-1"));

  const actionButton = await screen.findByRole("button", { name: /More concise/ });
  await user.click(actionButton);
  const previewRequest = await waitFor(() => requestFor(controller.requests, "/corrective-actions/preview"));
  controller.respondJson(previewRequest, previewBody(), { status: 201 });
  await screen.findByText("WILL INJECT");
  return previewRequest;
}

describe("Diagnosis & repair panel", () => {
  test("findings render with all five statuses distinct, with visible text and never color-only", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    render(<DiagnosisRepair runId="run-1" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.findings, diagnosisFindingsBody("run-1"));
    controller.respondJson(requests.registry, correctiveRegistryBody("run-1"));

    expect(await screen.findAllByText("FINDING")).toHaveLength(2);
    expect(screen.getByText("NOT OBSERVED -- checked, nothing found")).toBeInTheDocument();
    expect(screen.getByText("UNAVAILABLE -- could not be checked")).toBeInTheDocument();
    expect(screen.getByText("PENDING -- never measured")).toBeInTheDocument();
    expect(screen.getByText("SUPPRESSED -- excluded from this evaluation")).toBeInTheDocument();

    // pending never looks like "no problem" -- it renders under its own dashed/labelled status, never
    // folded into "not observed".
    const pendingRow = document.querySelector('[data-finding-status="pending"]');
    expect(pendingRow).not.toBeNull();
    expect(within(pendingRow as HTMLElement).getByText("PENDING -- never measured")).toBeInTheDocument();

    // five distinct statuses also means five distinct CSS hooks.
    const classes = new Set(
      ["finding", "not_observed", "unavailable", "pending", "suppressed"].map((status) =>
        document.querySelector(`[data-finding-status="${status}"]`)?.className),
    );
    expect(classes.size).toBe(5);
  });

  test("suggested actions appear only when the finding names them", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    render(<DiagnosisRepair runId="run-1" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.findings, diagnosisFindingsBody("run-1"));
    controller.respondJson(requests.registry, correctiveRegistryBody("run-1"));

    await screen.findAllByText("FINDING");
    const withAction = document.getElementById("diagnosis-finding-R01") as HTMLElement;
    const withoutAction = document.getElementById("diagnosis-finding-R04") as HTMLElement;
    expect(within(withAction).getByLabelText("Suggested direction for R01")).toBeInTheDocument();
    expect(within(withAction).getByText("resend the omitted segment")).toBeInTheDocument();
    expect(within(withoutAction).queryByLabelText("Suggested direction for R04")).not.toBeInTheDocument();
  });

  test("preview shows before execute -- confirm never fires until an explicit click", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();

    const previewRequest = await setUpToPreview(controller, user);
    expect(previewRequest.init?.method).toBe("POST");

    // The preview is visible and describes the exact change -- but nothing has executed yet.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(controller.requests.some((request) => pathOf(request).includes("/confirm"))).toBe(false);

    const confirmButton = screen.getByRole("button", { name: /CONFIRM/ });
    await user.click(confirmButton);
    const confirmRequest = await waitFor(() => requestFor(controller.requests, "/confirm"));
    expect(confirmRequest.init?.method).toBe("POST");
  });

  test("a failed execute leaves the preview and original run state intact", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();

    await setUpToPreview(controller, user);
    const confirmButton = screen.getByRole("button", { name: /CONFIRM/ });
    await user.click(confirmButton);
    const confirmRequest = await waitFor(() => requestFor(controller.requests, "/confirm"));
    controller.respondJson(
      confirmRequest,
      { error: "run evidence changed after preview; refusing stale confirmation" },
      { status: 409 },
    );

    expect(await screen.findByText(/the original run is unchanged/)).toBeInTheDocument();
    // The preview itself is still shown, unconsumed, alongside its source action button -- a caller can
    // retry the same confirm rather than losing the preview on a failed attempt.
    expect(screen.getAllByText("For this reply, answer concisely.")).toHaveLength(2);
    expect(document.querySelector(".diagnosis-repair-preview")).not.toBeNull();
    expect(screen.queryByText("RESULT")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /CONFIRM/ })).toBeEnabled();
  });

  test("a stale response for a previous run is never shown once the run changes", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    const view = render(<DiagnosisRepair runId="run-old" />);
    await waitFor(() => expect(controller.requests.length).toBeGreaterThanOrEqual(2));
    const oldFindings = requestFor(controller.requests, "/run-old/diagnosis-findings");
    const oldRegistry = requestFor(controller.requests, "/run-old/corrective-actions");

    view.rerender(<DiagnosisRepair runId="run-new" />);
    await waitFor(() => expect(controller.requests.length).toBeGreaterThanOrEqual(4));
    expect(oldFindings.signal?.aborted).toBe(true);
    expect(oldRegistry.signal?.aborted).toBe(true);

    const newFindings = requestFor(controller.requests, "/run-new/diagnosis-findings");
    const newRegistry = requestFor(controller.requests, "/run-new/corrective-actions");
    controller.respondJson(newFindings, diagnosisFindingsBody("run-new", [
      {
        rule_id: "R01", rule_name: "input_omitted_or_rejected", status: "not_observed",
        summary: "nothing was omitted for run-new.", evidence: [], limitations: [],
      },
    ]));
    controller.respondJson(newRegistry, correctiveRegistryBody("run-new"));
    expect(await screen.findByText("nothing was omitted for run-new.")).toBeInTheDocument();

    // The old run's (fully populated, distinctly-worded) response arrives late. It must never overwrite
    // the run the panel now shows.
    controller.respondJson(oldFindings, diagnosisFindingsBody("run-old"));
    controller.respondJson(oldRegistry, correctiveRegistryBody("run-old"));
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(screen.getByText("nothing was omitted for run-new.")).toBeInTheDocument();
    expect(screen.queryByText("1 input segment was omitted.")).not.toBeInTheDocument();
  });
});
