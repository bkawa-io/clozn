import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { CompareSurface } from "./CompareSurface";
import type { ComparisonSpecimen } from "./model";

const specimen: ComparisonSpecimen = {
  a: { id: "run-a", label: "Original", outputState: "available", output: "The customer is eligible for a refund.", model: "model-a" },
  b: { id: "run-b", label: "Fork", outputState: "available", output: "The customer is ineligible for a refund.", model: "model-b" },
  relationship: { kind: "related", intervention: { label: "User turn 4", detail: "Policy instruction edited" }, changedConditions: ["Context"] },
  differences: [{ id: "refund", label: "Eligibility conclusion", kind: "changed", a: { start: 16, end: 24 }, b: { start: 16, end: 26 }, alignment: "ambiguous", isFirstOutputDivergence: true }],
  evidenceByDifferenceId: { refund: { state: "not_measured", reason: "Trace capture was not enabled." } },
};

describe("CompareSurface", () => {
  it("keeps structural difference and unavailable local evidence visibly separate", () => {
    render(<CompareSurface specimen={specimen} />);
    expect(screen.getByText("First recorded output divergence")).toBeVisible();
    expect(screen.getByText(/correspondence is ambiguous/i)).toBeVisible();
    expect(screen.getByText("Trace capture was not enabled.")).toBeVisible();
    expect(screen.getByText(/not semantic equivalence or causal evidence/i)).toBeVisible();
  });

  it("does not invent lineage for arbitrary pairs and preserves unavailable output", () => {
    render(<CompareSurface specimen={{ ...specimen, relationship: { kind: "arbitrary" }, a: { ...specimen.a, outputState: "unavailable", output: undefined } }} />);
    expect(screen.getByText("Not inferred")).toBeVisible();
    expect(screen.getByText(/does not establish ancestry/i)).toBeVisible();
    expect(screen.getByText(/Readable output was not retained/i)).toBeVisible();
  });

  it("passes the selected recorded coordinates into Inspect and Test This", async () => {
    const user = userEvent.setup();
    const onInspect = vi.fn();
    const onTestThis = vi.fn();
    render(<CompareSurface specimen={specimen} onInspect={onInspect} onTestThis={onTestThis} />);
    await user.click(screen.getByRole("button", { name: "Inspect evidence" }));
    await user.click(screen.getByRole("button", { name: "Test this" }));
    const selection = { runAId: "run-a", runBId: "run-b", differenceId: "refund", a: { start: 16, end: 24 }, b: { start: 16, end: 26 } };
    expect(onInspect).toHaveBeenCalledWith(selection);
    expect(onTestThis).toHaveBeenCalledWith(selection);
  });

  it("reports an explicitly selected region so a host can preserve it in route state", async () => {
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();
    render(<CompareSurface specimen={{ ...specimen, differences: [
      ...specimen.differences,
      { id: "later", label: "Settlement timing", kind: "region", alignment: "unavailable" },
    ] }} onSelectionChange={onSelectionChange} />);
    await user.click(screen.getByRole("button", { name: "Settlement timing" }));
    expect(onSelectionChange).toHaveBeenCalledWith({ runAId: "run-a", runBId: "run-b", differenceId: "later", a: undefined, b: undefined });
  });
});
