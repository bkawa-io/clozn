import { describe, expect, test, vi } from "vitest";
import { render, screen, waitFor } from "../../test/render";
import { MinimalContextStudio } from "./MinimalContextStudio";

function response(body: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => body };
}

const detail = {
  id: "run-minimal",
  messages: [
    { role: "system", content: "Source A" },
    { role: "system", content: "Source B" },
    { role: "system", content: "Source C" },
    { role: "user", content: "Current question" },
  ],
  context_units: {
    protected_message_indices: [3],
    units: [
      { source_id: "src_a", message_index: 0, role: "system", unicode_range: [0, 8] as [number, number], derivation: "caller_explicit", source_label: "Source A" },
      { source_id: "src_b", message_index: 1, role: "system", unicode_range: [0, 8] as [number, number], derivation: "caller_explicit", source_label: "Source B" },
      { source_id: "src_c", message_index: 2, role: "system", unicode_range: [0, 8] as [number, number], derivation: "caller_explicit", source_label: "Source C" },
    ],
  },
};

const baseResult = {
  schema_version: "clozn.minimal-context-search-result.v1",
  search_id: "search_aaaaaaaaaaaaaaaaaaaaaaaa",
  status: "completed",
  search_status: "ok",
  reason: null,
  reason_code: null,
  base_execution_fingerprint: "exec_aaaaaaaaaaaaaaaaaaaaaaaa",
  universe: { universe_id: "mcu_aaaaaaaaaaaaaaaaaaaaaaaa", source_ids: ["src_a", "src_b", "src_c"], source_count: 3 },
  objective: { kind: "rendered_prompt_tokens", version: "rendered_prompt_tokens.v1" },
  control_observation_id: "obs_control",
  trials: [],
  trajectory: [],
  best: {
    retained_source_ids: ["src_a", "src_c"],
    removed_source_ids: ["src_b"],
    rendered_prompt_token_cost: 42,
    experiment_id: "exp_aaaaaaaaaaaaaaaaaaaaaaaa",
    arm_id: "arm_0",
    observation_id: "obs_aaaaaaaaaaaaaaaaaaaaaaaa",
    observation_status: "exact_preserved",
  },
  certificate: "INCLUSION_MINIMUM" as const,
  policy: { kind: "adaptive_bounded_deletion", version: "adaptive_bounded_deletion.v1", attempt_inclusion_check: true },
  budget: { max_new_executions: 32, used_new_executions: 12, reused_observation_count: 8, exhausted: false },
  inclusion_check: { attempted: true, complete: false, tested_child_count: 4, total_child_count: 5, all_children_failed: false },
};

function job(state: "queued" | "completed", result?: unknown) {
  return {
    schema_version: "clozn.influence-map-job.v1",
    job_id: "infjob_minimal",
    run_id: "run-minimal",
    kind: "minimal_context",
    state,
    progress: { phase: state === "completed" ? "done" : "searching", completed_units: 1, total_units: 1, percent: state === "completed" ? 100 : 10 },
    cancel_requested: false,
    cancellable: state !== "completed",
    cached: false,
    ...(result === undefined ? {} : { result }),
  };
}

function installFetch(result: unknown | null, branchResponse: unknown = { state: "completed", child_run_id: "child-reduced" }) {
  const requests: Array<{ url: string; init?: RequestInit }> = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    requests.push({ url, init });
    if (url.endsWith("/runs/run-minimal")) return response(detail);
    if (url.endsWith("/minimal-context/jobs") && init?.method === "POST") return response(job("queued"));
    if (url.endsWith("/minimal-context/jobs/infjob_minimal")) return response(job("completed", result));
    if (url.endsWith("/minimal-context/branch") && init?.method === "POST") return response(branchResponse, true, 201);
    return response({});
  }));
  return requests;
}

describe("MinimalContextStudio", () => {
  test("completed job retains the new result and does not fetch result history", async () => {
    const requests = installFetch(baseResult);
    render(<MinimalContextStudio runId="run-minimal" />);
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" });
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).then((button) => button.click());

    expect(await screen.findByText("INCLUSION-MINIMAL")).toBeInTheDocument();
    expect(screen.getByText("The recorded answer was preserved, and deleting any one remaining source was directly tested and caused divergence.")).toBeInTheDocument();
    expect(screen.getByText("Rendered prompt tokens: 42")).toBeInTheDocument();
    expect(requests.some((request) => request.url.endsWith("/minimal-context"))).toBe(false);
  });

  test("renders retained and omitted context from the new universe and best fields", async () => {
    installFetch(baseResult);
    render(<MinimalContextStudio runId="run-minimal" />);
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).then((button) => button.click());
    expect(await screen.findByText("Source A")).toBeInTheDocument();
    expect(screen.getByText("Source B")).toBeInTheDocument();
    expect(screen.getByText("Source C")).toBeInTheDocument();
    expect(screen.getByText("12 new executions")).toBeInTheDocument();
    expect(screen.getByText("8 reused observations")).toBeInTheDocument();
    expect(screen.getByText("4 / 5 children directly tested; 1 remain unknown.")).toBeInTheDocument();
  });

  test("renders BEST VERIFIED language", async () => {
    installFetch({ ...baseResult, certificate: "BEST_VERIFIED", best: { ...baseResult.best, removed_source_ids: [] } });
    render(<MinimalContextStudio runId="run-minimal" />);
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).then((button) => button.click());
    expect(await screen.findByText("BEST VERIFIED")).toBeInTheDocument();
    expect(screen.getByText("Lowest-cost preserving candidate observed within the search budget. A smaller preserving candidate may exist.")).toBeInTheDocument();
  });

  test("renders unavailable results without reading a missing best candidate", async () => {
    const unavailable = {
      ...baseResult,
      status: "unavailable",
      search_status: "unavailable",
      reason: "Exact recorded-answer control could not be reproduced.",
      reason_code: "exact_control_unavailable",
      best: null,
      certificate: null,
    };
    installFetch(unavailable);
    render(<MinimalContextStudio runId="run-minimal" />);
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).then((button) => button.click());
    expect(await screen.findByText("MINIMAL CONTEXT UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("code: exact_control_unavailable")).toBeInTheDocument();
  });

  test("materialization sends only generic winner references", async () => {
    const requests = installFetch(baseResult);
    render(<MinimalContextStudio runId="run-minimal" />);
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).then((button) => button.click());
    await screen.findByRole("button", { name: "BRANCH WITH THIS CONTEXT" });
    await screen.getByRole("button", { name: "BRANCH WITH THIS CONTEXT" }).click();
    await waitFor(() => expect(screen.getByText(/Child branch created/)).toBeInTheDocument());
    const branchRequest = requests.find((request) => request.url.endsWith("/minimal-context/branch"));
    expect(JSON.parse(String(branchRequest?.init?.body))).toEqual({
      experiment_id: "exp_aaaaaaaaaaaaaaaaaaaaaaaa",
      arm_id: "arm_0",
      observation_id: "obs_aaaaaaaaaaaaaaaaaaaaaaaa",
    });
  });

  test("does not offer a branch for a full-context winner", async () => {
    installFetch({ ...baseResult, best: { ...baseResult.best, removed_source_ids: [] } });
    render(<MinimalContextStudio runId="run-minimal" />);
    await screen.findByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).then((button) => button.click());
    await screen.findByText("BEST VERIFIED");
    expect(screen.queryByRole("button", { name: "BRANCH WITH THIS CONTEXT" })).not.toBeInTheDocument();
  });
});
