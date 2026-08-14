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
    { role: "user", content: "Current question" },
  ],
  context_units: {
    protected_message_indices: [1],
    units: [{ source_id: "src_a", message_index: 0, role: "system", unicode_range: [0, 8] as [number, number], derivation: "caller_explicit", source_label: "Source A" }],
  },
};

const exactResult = {
  schema_version: "clozn.minimal-context-result.v1",
  run_id: "run-minimal",
  result_id: "mc_aaaaaaaaaaaaaaaaaaaaaaaa",
  status: "found",
  source_universe: { source_ids: ["src_a"], source_count: 1, search_universe_id: "mcu_aaaaaaaaaaaaaaaaaaaaaaaa" },
  preservation: { kind: "exact_recorded_output" as const },
  candidate: { retained_source_ids: ["src_a"], removed_source_ids: ["src_b"], retained_source_count: 1, within_tolerance: true },
  certificate: { kind: "exact_minimum" as const, candidate_retained_source_count: 1, global_minimality: "proven" as const, inclusion_minimality: "proven" as const },
  coverage: { lower_cardinalities: [{ retained_source_count: 0, candidate_count: 1, tested_count: 1, preserving_count: 0, complete: true }], smaller_candidate_count: 1, smaller_tested_count: 1, smaller_remaining_count: 0 },
};

function installFetch(result: unknown | null) {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/runs/run-minimal") && !url.includes("minimal-context")) return response(detail);
    if (url.endsWith("/runs/run-minimal/minimal-context")) return response({ results: result ? [{ result_id: (result as { result_id: string }).result_id, preservation_kind: (result as { preservation: { kind: string } }).preservation.kind, source_count: 1, retained_source_count: 1, certificate_kind: "exact_minimum", status: "found", universe_id: "mcu_aaaaaaaaaaaaaaaaaaaaaaaa" }] : [] });
    if (url.includes("/minimal-context/mc_")) return response(result);
    if (init?.method === "POST") return response({ error: "exact mode unavailable" }, false, 409);
    return response({});
  }));
}

describe("MinimalContextStudio", () => {
  test("shows the exact proof contract, coverage, protected request, and source detail", async () => {
    installFetch(exactResult);
    render(<MinimalContextStudio runId="run-minimal" />);
    expect(await screen.findByText("EXACT MINIMUM")).toBeInTheDocument();
    expect(screen.getByText("recorded answer reproduced token-for-token")).toBeInTheDocument();
    expect(screen.getByText("Unmeasured candidates are not counted as failed.")).toBeInTheDocument();
    expect(screen.getByText("PROTECTED CURRENT REQUEST")).toBeInTheDocument();
    expect(screen.getByText("Source A")).toBeInTheDocument();
  });

  test("keeps exact unavailability explicit instead of falling back silently", async () => {
    installFetch(null);
    render(<MinimalContextStudio runId="run-minimal" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" })).toBeInTheDocument());
    await screen.getByRole("button", { name: "RUN EXACT MINIMAL CONTEXT" }).click();
    expect((await screen.findByRole("alert")).textContent).toContain("Exact mode was not silently replaced.");
  });
});
