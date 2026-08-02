import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen } from "../../test/render";
import { InvestigationExperiment } from "./InvestigationExperiment";

function pathOf(request: PendingFetch) {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname
      : request.input.url;
}

function bodyFor(request: PendingFetch) {
  return JSON.parse(String(request.init?.body ?? "{}")) as Record<string, unknown>;
}

const SPAN_ID = "span_prompt_1";

function spanAddressesBody() {
  return {
    schema_version: "clozn.text-span-addresses.v1",
    run_id: "run-one",
    privacy: "metadata_only",
    source_artifacts: [],
    addresses: [{
      address_id: SPAN_ID,
      kind: "prompt_source",
      native_ref: { collection: "influence.prompt_sources", client_source_id: "doc-1", source_label: "Reference document" },
      resolution: { state: "exact", canonical: { start: 0, end: 12 } },
    }],
  };
}

function plannedBody() {
  return {
    schema_version: "clozn.investigation-experiment.v1",
    experiment_id: "exp-1",
    run_id: "run-one",
    generated_at: "2026-08-01T00:00:00Z",
    phase: "planned",
    intervention: { kind: "remove_span", span_address_id: SPAN_ID },
    eligibility: { state: "eligible" },
    plan: { arm_order: ["baseline", "no_op_replay", "treatment", "random_equal_effect_control"], resolved: { kind: "remove_span", spans: [{ message_index: 0, start: 0, end: 12, basis_sha256_verified: true, span_address_id: SPAN_ID }] } },
  };
}

function completedBody() {
  return {
    ...plannedBody(),
    phase: "completed",
    arms: {
      baseline: { run_id: "run-one", reply_sha256: "a".repeat(64), matches_baseline: true },
      no_op_replay: { run_id: "child-noop", reply_sha256: "a".repeat(64), matches_baseline: true },
      treatment: { run_id: "child-treatment", reply_sha256: "b".repeat(64), matches_baseline: false },
      random_equal_effect_control: { available: false, reason: "no matched control" },
    },
    analysis: { instrument_sane: true, reasons: ["no random control ran"] },
    observed: { treatment_reply_differs_from_baseline: true, note: "factual diff only" },
    causal_claim: { licensed: false, statement: "uncontrolled: no random equal-effect control ran" },
  };
}

describe("Did this matter?", () => {
  test("plans first and executes a controlled child experiment only after an explicit click", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<InvestigationExperiment runId="run-one" />);

    const spans = await controller.nextRequest();
    expect(pathOf(spans)).toContain("/runs/run-one/span-addresses");
    controller.respondJson(spans, spanAddressesBody());
    await screen.findByLabelText("PASSAGE / SOURCE");
    expect(controller.requests.filter((request) => pathOf(request).includes("investigation-experiment")).length).toBe(0);

    await user.click(screen.getByRole("button", { name: "PLAN EXPERIMENT" }));
    const planRequest = await controller.nextRequest();
    expect(pathOf(planRequest)).toContain("/investigation-experiment/plan");
    expect(bodyFor(planRequest)).toEqual({ intervention: { kind: "remove_span", span_address_id: SPAN_ID } });
    controller.respondJson(planRequest, plannedBody());
    await screen.findByText("4 ARMS");
    expect(controller.requests.filter((request) => pathOf(request).endsWith("/investigation-experiment")).length).toBe(0);

    await user.click(screen.getByRole("button", { name: "RUN CONTROLLED EXPERIMENT" }));
    const startRequest = await controller.nextRequest();
    expect(pathOf(startRequest)).toMatch(/\/investigation-experiment$/);
    expect(startRequest.init?.method).toBe("POST");
    controller.respondJson(startRequest, {
      schema_version: "clozn.influence-map-job.v1", job_id: "job-1", run_id: "run-one", kind: "investigation_experiment",
      state: "queued", progress: { phase: "queued", completed_units: 0, total_units: 1, percent: 0 }, cancellable: true,
    });
    const pollRequest = await controller.nextRequest();
    expect(pathOf(pollRequest)).toContain("/investigation-experiment/jobs/job-1");
    controller.respondJson(pollRequest, {
      schema_version: "clozn.influence-map-job.v1", job_id: "job-1", run_id: "run-one", kind: "investigation_experiment",
      state: "completed", progress: { phase: "done", completed_units: 1, total_units: 1, percent: 100 }, cancellable: false,
      result: completedBody(),
    });

    expect(await screen.findByText("NOT LICENSED")).toBeInTheDocument();
    expect(screen.getByText(/uncontrolled: no random equal-effect control ran/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "ld-treatment" })).toHaveAttribute("href", "#/runs/child-treatment");
  });
});
