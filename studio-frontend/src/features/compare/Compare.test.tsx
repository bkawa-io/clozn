import userEvent from "@testing-library/user-event";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "../../data/client";
import { ContractError } from "../../data/contracts";
import { decodeRunComparison } from "../../data/runComparison";
import { CompareSurface } from "./Compare";

function comparisonDocument() {
  return {
    schema_version: "clozn.run-diff.v1",
    ok: true,
    run_a: "run_a",
    run_b: "run_b",
    comparison_selection: {
      mode: "explicit", reference_run_id: "run_a", candidate_run_id: "run_b", reason: "explicit pair",
    },
    generated_at: "2026-10-09T12:00:00+00:00",
    privacy_limited: false,
    ranking: { order: ["context", "output"], note: "presentation only" },
    summary_axes: {
      model: { status: "unchanged" }, adapter: { status: "unavailable", note: "Not recorded." },
      template: { status: "unchanged" }, context: { status: "changed" },
      sampling: { status: "unchanged" }, engine: { status: "unchanged" },
      tool_parse: { status: "unavailable", note: "Not recorded." }, output: { status: "changed" },
    },
    differences: [{
      dimension: "context.delivered.messages.count", kind: "changed", value_a: 2, value_b: 1, rank: 30, evidence: [],
    }],
    findings: [{
      classification: "context_omission", status: "observed", summary: "One message was omitted.",
      dimensions: ["context.delivered.messages.count"],
    }],
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.location.hash = "";
});

describe("Compare surface", () => {
  it("keeps the index model-free until a complete distinct pair is submitted", async () => {
    const load = vi.spyOn(studioApi, "runComparison");
    const user = userEvent.setup();
    render(<CompareSurface route={{ surface: "compare" }} />);

    expect(load).not.toHaveBeenCalled();
    expect(screen.getByText(/Enter two immutable Run IDs/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Experiment matrix" })).toHaveAttribute("href", "#/compare/matrix");
    const submit = screen.getByRole("button", { name: "Compare runs" });
    expect(submit).toBeDisabled();

    await user.type(screen.getByLabelText("Reference run A"), "run_a");
    await user.type(screen.getByLabelText("Candidate run B"), "run_b");
    expect(submit).toBeEnabled();
    await user.click(submit);
    expect(window.location.hash).toBe("#/compare/run_a/run_b");
    expect(load).not.toHaveBeenCalled();
  });

  it("renders the canonical read-only delta and links both runs back to Inspect", async () => {
    const comparison = decodeRunComparison(comparisonDocument());
    const load = vi.spyOn(studioApi, "runComparison").mockResolvedValue(comparison);
    render(<CompareSurface route={{ surface: "compare", runA: "run_a", runB: "run_b" }} />);

    expect(await screen.findByText("context omission")).toBeInTheDocument();
    expect(load).toHaveBeenCalledWith("run_a", "run_b", expect.any(AbortSignal));
    expect(screen.getByRole("link", { name: "Inspect A" })).toHaveAttribute("href", "#/inspect/run_a");
    expect(screen.getByRole("link", { name: "Inspect B" })).toHaveAttribute("href", "#/inspect/run_b");
    expect(screen.getByRole("link", { name: "Swap" })).toHaveAttribute("href", "#/compare/run_b/run_a");
    expect(within(screen.getByRole("table")).getByText("context.delivered.messages.count")).toBeInTheDocument();
    expect(screen.getByText(/do not establish causation/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /controlled|replay|change test/i })).not.toBeInTheDocument();
  });

  it("distinguishes a malformed comparison contract from an unavailable request", async () => {
    const load = vi.spyOn(studioApi, "runComparison");
    load.mockRejectedValueOnce(new ContractError("/runs/compare", "$.findings is malformed"));
    const view = render(<CompareSurface route={{ surface: "compare", runA: "run_a", runB: "run_b" }} />);
    expect(await screen.findByText("Malformed comparison contract.")).toBeInTheDocument();

    load.mockRejectedValueOnce(new Error("gateway offline"));
    view.rerender(<CompareSurface route={{ surface: "compare", runA: "run_a", runB: "run_c" }} />);
    expect(await screen.findByText("Comparison unavailable.")).toBeInTheDocument();
  });
});
