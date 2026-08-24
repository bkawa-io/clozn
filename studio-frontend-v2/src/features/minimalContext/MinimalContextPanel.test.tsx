import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { MinimalContextPanel } from "./MinimalContextPanel";

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

const resultId = "mcres_0123456789abcdef01234567";
const summary = {
  schema_version: "clozn.minimal-context-search-result.v2",
  result_id: resultId,
  run_id: "run-minimal",
  search_id: "search_0123456789abcdef01234567",
  status: "completed",
  search_status: "completed",
  certificate: "EXACT_MINIMUM",
  stopping_reason: "exact_minimum_proven",
  best: {
    retained_source_ids: ["source-a", "source-c"],
    removed_source_ids: ["source-b"],
    rendered_prompt_token_cost: 60,
    experiment_id: "exp-1",
    arm_id: "arm-1",
    observation_id: "obs-1",
    observation_status: "exact_preserved",
  },
  reduction: {
    objective: "rendered_prompt_tokens.v1",
    original_prompt_token_cost: 100,
    retained_prompt_token_cost: 60,
    removed_source_count: 1,
    retained_source_count: 2,
    fraction: 0.4,
    percent: 40,
  },
  base_execution_fingerprint: "fingerprint-1",
  current_binding: { status: "current", reason: null },
};

const detail = {
  ...summary,
  control_observation_id: "obs-control",
  universe: { source_ids: ["source-a", "source-b", "source-c"], source_count: 3 },
  objective: { kind: "rendered_prompt_tokens", version: "rendered_prompt_tokens.v1" },
  trials: [],
  trajectory: [],
  original: {},
  experiment_accounting: { new_counterfactual_executions: 21, reused_observations: 6 },
  source_inspection: [
    { source_id: "source-a", segment_id: "segment-a", message_index: 0, label: "Policy", provenance_kind: "caller", parent_source_id: null, unicode_range: [0, 21], byte_range: [0, 21], granularity: "whole_segment", text: "Refunds are available.", disposition: "retained" },
    { source_id: "source-b", segment_id: "segment-b", message_index: 1, label: "Exception", provenance_kind: "caller", parent_source_id: null, unicode_range: [0, 25], byte_range: [0, 25], granularity: "whole_segment", text: "Only annual plans qualify.", disposition: "removed" },
    { source_id: "source-c", segment_id: "segment-c", message_index: 2, label: "Window", provenance_kind: "caller", parent_source_id: null, unicode_range: [0, 22], byte_range: [0, 22], granularity: "whole_segment", text: "The window is fourteen days.", disposition: "retained" },
  ],
  proof: {},
  policy: {},
  budget: { max_new_executions: 32, used_new_executions: 21, reused_observation_count: 6, exhausted: false, blocked_by_budget: false },
  inclusion_check: { attempted: true, complete: true, tested_child_count: 2, total_child_count: 2, all_children_failed: true },
};

function job(state: "completed" | "running") {
  return {
    schema_version: "clozn.influence-map-job.v1",
    job_id: "infjob_minimal",
    run_id: "run-minimal",
    kind: "minimal_context",
    state,
    progress: { phase: state === "completed" ? "done" : "searching", completed_units: state === "completed" ? 1 : 0, total_units: 1, percent: state === "completed" ? 100 : 10 },
    cancel_requested: false,
    cancellable: state === "running",
    cached: false,
  };
}

describe("MinimalContextPanel", () => {
  test("shows not-run state, starts the canonical job, then reads the durable result detail", async () => {
    const user = userEvent.setup();
    let hasResult = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/minimal-context/results")) return response({ results: hasResult ? [summary] : [] });
      if (url.endsWith("/minimal-context/jobs") && init?.method === "POST") {
        hasResult = true;
        return response(job("completed"), 202);
      }
      if (url.endsWith(`/minimal-context/results/${resultId}`)) return response(detail);
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<MinimalContextPanel runId="run-minimal" />);
    expect(await screen.findByText("Not run yet")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Reduce context" }));

    expect(await screen.findByText("Exact minimum")).toBeInTheDocument();
    expect(await screen.findByText("2 of 3 context sources retained · 40% fewer rendered prompt tokens")).toBeInTheDocument();
    expect(screen.getByText("Search checked 21 new counterfactuals")).toBeInTheDocument();
    expect(screen.getByText("Reused 6 prior observations")).toBeInTheDocument();
    expect(await screen.findByText("Refunds are available.")).toBeInTheDocument();
    expect(screen.getByText("Only annual plans qualify.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/runs/run-minimal/minimal-context/jobs", expect.objectContaining({ method: "POST", body: "{}" }));
    expect(fetchMock).toHaveBeenCalledWith(`/runs/run-minimal/minimal-context/results/${resultId}`, expect.any(Object));
  });

  test("keeps stale historical results visible and labels them without hiding their detail", async () => {
    const staleSummary = { ...summary, current_binding: { status: "stale", reason: "the current source universe changed" } };
    const staleDetail = { ...detail, current_binding: { status: "stale", reason: "the current source universe changed" } };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/minimal-context/results")) return response({ results: [staleSummary] });
      if (url.endsWith(`/minimal-context/results/${resultId}`)) return response(staleDetail);
      return response({});
    }));

    render(<MinimalContextPanel runId="run-minimal" />);

    expect(await screen.findByText("Historical result — Run/context binding has changed")).toBeInTheDocument();
    expect(await screen.findByText("Refunds are available.")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("the current source universe changed")).toBeInTheDocument());
  });
});
