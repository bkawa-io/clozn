import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, waitFor, within } from "../../test/render";
import { WhatMattered } from "./WhatMattered";

function investigation(runId: string, overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.run-investigation.v1",
    run_id: runId,
    sections: {
      received_context: {
        state: "delivered_not_measured",
        privacy: "metadata_only",
        delivered: [],
        assembled: [],
        omitted: [
          {
            segment_id: "seg-tool",
            source_type: "message",
            source_label: "tool",
            original_order: 2,
            reason: "tool_result_pruned",
          },
        ],
        limits: {},
      },
      text_span_addresses: {
        state: "supported",
        privacy: "metadata_only",
        href: `/runs/${runId}/span-addresses`,
        address_count: 4,
      },
      prompt_source_influence: {
        state: "measured_effect",
        privacy: "metadata_only",
        prompt_sources: [
          {
            id: "p.m000",
            segment_id: "seg-system",
            source_label: "system",
            role: "system",
            selected: true,
            start: 0,
            end: 50,
            text_sha256: "a".repeat(64),
            text_bytes: 50,
          },
          {
            id: "p.m001",
            segment_id: "seg-user",
            source_label: "user",
            role: "user",
            selected: false,
            start: 0,
            end: 80,
            text_sha256: "b".repeat(64),
            text_bytes: 80,
          },
        ],
        prompt_spans: [
          {
            id: "p.m000.c000",
            parent_id: "p.m000",
            level: "coarse",
            segment_id: "seg-system",
            source_label: "system",
            role: "system",
            start: 0,
            end: 50,
            text_sha256: "a".repeat(64),
            text_bytes: 50,
          },
        ],
        answer_spans: [
          {
            id: "a.t0000",
            level: "token",
            token_index: 0,
            token_id: 5,
            start: 0,
            end: 3,
            text_sha256: "c".repeat(64),
            text_bytes: 3,
          },
          {
            id: "a.t0001",
            level: "token",
            token_index: 1,
            token_id: 9,
            start: 3,
            end: 6,
            text_sha256: "d".repeat(64),
            text_bytes: 3,
          },
        ],
        links: [
          {
            context_span_id: "p.m000.c000",
            answer_span_id: "a.t0000",
            context_index: 0,
            answer_index: 0,
            delta_nats: 0.9,
            abs_delta_nats: 0.9,
            effect: "supports",
            clears_floor: true,
            evidence_state: "causally_supported",
          },
          {
            context_span_id: "p.m000.c000",
            answer_span_id: "a.t0001",
            context_index: 0,
            answer_index: 1,
            delta_nats: -0.01,
            abs_delta_nats: 0.01,
            effect: "suppresses",
            clears_floor: false,
            evidence_state: "observed",
          },
        ],
        thresholds: {
          cell_abs_delta_nats: 0.05,
          source_clear_rule: "absolute signed cell delta meets or exceeds cell_abs_delta_nats",
          calibration: "fixed_default_not_model_calibrated",
        },
      },
    },
    actions: [
      {
        id: "measure_prompt_source_influence",
        label: "Measure what mattered",
        kind: "measurement",
        method: "POST",
        href: `/runs/${runId}/influence-map/jobs`,
        availability: "ready",
      },
    ],
    unavailable_measurements: [],
    ...overrides,
  };
}

function unmeasuredInvestigation(runId: string, availability = "ready", reason?: string) {
  const base = investigation(runId);
  return {
    ...base,
    sections: {
      ...base.sections,
      prompt_source_influence: {
        state: "delivered_not_measured",
        privacy: "metadata_only",
        reason: "context delivery is recorded, but no prompt/source influence experiment has run",
        prompt_sources: [],
        prompt_spans: [],
        answer_spans: [],
        links: [],
        thresholds: {},
      },
    },
    actions: [
      {
        id: "measure_prompt_source_influence",
        label: "Measure what mattered",
        kind: "measurement",
        method: "POST",
        href: `/runs/${runId}/influence-map/jobs`,
        availability,
        ...(reason ? { reason } : {}),
      },
    ],
  };
}

function spanDocument(runId: string) {
  return {
    schema_version: "clozn.text-span-addresses.v1",
    run_id: runId,
    privacy: "metadata_only",
    offset_contract: {
      unit: "unicode_code_points",
      interval: "half_open",
      hash_algorithm: "sha256",
      canonicalization: "exact_string_utf8_v1",
    },
    source_artifacts: [],
    addresses: [],
    lineage: { parent_run_id: null, mappings: [] },
  };
}

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
    investigation: requestFor(controller.requests, "/investigation"),
    spans: requestFor(controller.requests, "/span-addresses"),
  };
}

describe("What mattered panel", () => {
  test("renders all four cell states distinctly, with visible text and legend, cross-linked to spans", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    render(<WhatMattered runId="run-one" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.investigation, investigation("run-one"));
    controller.respondJson(requests.spans, spanDocument("run-one"));

    expect(await screen.findByRole("heading", { name: "What mattered?" })).toBeInTheDocument();

    // The legend spells out all four states as literal text -- never color-only.
    const legend = screen.getByRole("list", { name: "Cell state legend" });
    expect(within(legend).getByText("CLEARED FLOOR -- controlled, measured effect")).toBeInTheDocument();
    expect(within(legend).getByText("BELOW FLOOR -- measured, no effect cleared")).toBeInTheDocument();
    expect(within(legend).getByText("OMITTED -- never reached the model")).toBeInTheDocument();
    expect(within(legend).getByText("NOT MEASURED -- reached the model, not scored")).toBeInTheDocument();

    // Every one of the four states is actually rendered as a cell, each with its own visible glyph --
    // a screenshot of the grid alone (no hover, no color perception) must distinguish all four.
    const grid = document.querySelector(".what-mattered-grid");
    expect(grid).not.toBeNull();
    const measuredCell = grid!.querySelector('[data-cell-state="measured_effect"]');
    const belowFloorCell = grid!.querySelector('[data-cell-state="below_measurement_floor"]');
    const omittedCell = grid!.querySelector('[data-cell-state="omitted"]');
    const notMeasuredCell = grid!.querySelector('[data-cell-state="not_measured"]');
    expect(measuredCell).not.toBeNull();
    expect(belowFloorCell).not.toBeNull();
    expect(omittedCell).not.toBeNull();
    expect(notMeasuredCell).not.toBeNull();
    expect(measuredCell!.textContent).toBe("F+");
    expect(belowFloorCell!.textContent).toBe("B-");
    expect(omittedCell!.textContent).toBe("OM");
    expect(notMeasuredCell!.textContent).toBe("NM");
    // Four distinct rendered states also means four distinct CSS hooks -- style is never the ONLY
    // distinguishing signal, but it must still exist and differ per state.
    const classes = new Set(
      [measuredCell, belowFloorCell, omittedCell, notMeasuredCell].map((cell) => cell!.className),
    );
    expect(classes.size).toBe(4);

    // The "not measured" row is the bound-selection-excluded source, distinct from "omitted".
    expect(screen.getByText("user")).toBeInTheDocument();
    expect(screen.getByText("reached the model, excluded by this run's bounded span selection"))
      .toBeInTheDocument();
    expect(screen.getByText("tool")).toBeInTheDocument();
    expect(screen.getByText("tool_result_pruned")).toBeInTheDocument();
  });

  test("viewing the panel triggers no measurement -- only the two read GETs fire until a click", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    render(<WhatMattered runId="run-idle" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.investigation, unmeasuredInvestigation("run-idle"));
    controller.respondJson(requests.spans, spanDocument("run-idle"));

    expect(await screen.findByText("NOT MEASURED")).toBeInTheDocument();
    const button = await screen.findByRole("button", { name: "MEASURE WHAT MATTERED" });
    expect(button).toBeEnabled();

    // Give any accidental effect a turn to fire before asserting its absence.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(controller.requests).toHaveLength(2);
    expect(controller.requests.some((request) => pathOf(request).endsWith("/influence-map/jobs"))).toBe(false);
  });

  test("a disabled measurement action shows its reason and still fires nothing on its own", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    render(<WhatMattered runId="run-no-worker" />);
    const requests = await initialRequests(controller);
    controller.respondJson(
      requests.investigation,
      unmeasuredInvestigation("run-no-worker", "unavailable", "the active worker does not expose token scoring"),
    );
    controller.respondJson(requests.spans, spanDocument("run-no-worker"));

    const button = await screen.findByRole("button", { name: "MEASURE WHAT MATTERED" });
    expect(button).toBeDisabled();
    expect(screen.getByText("the active worker does not expose token scoring")).toBeInTheDocument();
    expect(controller.requests).toHaveLength(2);
  });

  test("clicking measure starts a job and a completed job is reflected only through a fresh fetch", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();

    render(<WhatMattered runId="run-measure" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.investigation, unmeasuredInvestigation("run-measure"));
    controller.respondJson(requests.spans, spanDocument("run-measure"));

    const button = await screen.findByRole("button", { name: "MEASURE WHAT MATTERED" });
    await user.click(button);

    const startRequest = await waitFor(() => requestFor(controller.requests, "/influence-map/jobs"));
    expect(startRequest.init?.method).toBe("POST");
    controller.respondJson(startRequest, {
      schema_version: "clozn.influence-map-job.v1",
      job_id: "job-1",
      run_id: "run-measure",
      state: "completed",
      progress: { phase: "score", completed_units: 1, total_units: 1, percent: 100 },
      cancel_requested: false,
      cancellable: false,
      cached: false,
    });

    // A completed job must never be patched into the view directly -- it must trigger a brand-new,
    // canonical fetch of the same two documents (now reflecting the measurement) before the grid appears.
    await waitFor(() => expect(controller.requests.filter((r) => pathOf(r).endsWith("/investigation")).length)
      .toBe(2));
    const refreshed = controller.requests.filter((r) => pathOf(r).endsWith("/investigation"))[1];
    const refreshedSpans = controller.requests.filter((r) => pathOf(r).endsWith("/span-addresses"))[1];
    controller.respondJson(refreshed, investigation("run-measure"));
    controller.respondJson(refreshedSpans, spanDocument("run-measure"));

    expect(await screen.findByText("F+")).toBeInTheDocument();
  });

  test("a stale response for a previous run identity is never shown once the run changes", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const view = render(<WhatMattered runId="run-old" />);
    await waitFor(() => expect(controller.requests).toHaveLength(2));
    const oldInvestigation = requestFor(controller.requests, "/run-old/investigation");
    const oldSpans = requestFor(controller.requests, "/run-old/span-addresses");

    view.rerender(<WhatMattered runId="run-new" />);
    await waitFor(() => expect(controller.requests).toHaveLength(4));
    expect(oldInvestigation.signal?.aborted).toBe(true);
    expect(oldSpans.signal?.aborted).toBe(true);
    const newInvestigation = requestFor(controller.requests, "/run-new/investigation");
    const newSpans = requestFor(controller.requests, "/run-new/span-addresses");

    controller.respondJson(newInvestigation, unmeasuredInvestigation("run-new"));
    controller.respondJson(newSpans, spanDocument("run-new"));
    expect(await screen.findByText("NOT MEASURED")).toBeInTheDocument();

    // The old run's (fully measured, distinctly-labelled) response arrives late. It must never overwrite
    // the run the user is now looking at -- neither its grid nor its "system"/"user"/"tool" row labels.
    controller.respondJson(oldInvestigation, investigation("run-old"));
    controller.respondJson(oldSpans, spanDocument("run-old"));
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(screen.queryByText("F+")).not.toBeInTheDocument();
    expect(screen.getByText("NOT MEASURED")).toBeInTheDocument();
  });
});
