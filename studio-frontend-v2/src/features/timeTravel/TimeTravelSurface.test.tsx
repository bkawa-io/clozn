import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { TimeTravelSurface } from "./TimeTravelSurface";
import type { TimeTravelRun } from "./model";

const run: TimeTravelRun = {
  id: "run_parent",
  response: "The recorded parent response remains unchanged.",
  responseTokens: [{ position: 0, text: "The" }, { position: 1, text: "recorded" }, { position: 2, text: "parent" }],
  loci: [{ id: "answer-locus", label: "Recorded answer locus", boundaryPosition: 1 }],
  fidelityByBoundary: {
    0: { reconstructedReplay: { state: "unavailable", reason: "No replay artifact." }, exactFork: { state: "unavailable", reason: "No checkpoint." }, historicalExactProof: { state: "none" } },
    1: { reconstructedReplay: { state: "available", unavoidableDifferences: ["kv_state_not_restored", "sampler_state_reinitialized"] }, exactFork: { state: "requires_live_plan" }, historicalExactProof: { state: "verified", verifiedExecutionCount: 1 } },
    2: { reconstructedReplay: { state: "available" }, exactFork: { state: "ready_to_execute" }, historicalExactProof: { state: "none" } },
  },
};

describe("TimeTravelSurface", () => {
  test("keeps original recording visible, deep-links a locus, and only requests a live exact plan", async () => {
    const user = userEvent.setup();
    const onCheckExactFork = vi.fn();
    render(<TimeTravelSurface run={run} initialSelection={{ locusId: "answer-locus" }} onCheckExactFork={onCheckExactFork} />);

    expect(screen.getByText("The recorded parent response remains unchanged.")).toBeInTheDocument();
    expect(screen.getByText("Recorded answer locus")).toBeInTheDocument();
    expect(screen.getByText("REQUIRES LIVE PLAN")).toBeInTheDocument();
    expect(screen.getByText(/historical proof records a past completed control match/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /stage exact fork/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Check exact fork" }));
    expect(onCheckExactFork).toHaveBeenCalledWith({ runId: "run_parent", position: 1, locusId: "answer-locus" });
  });

  test("stages a proposal before it crosses the explicit branch action boundary", async () => {
    const user = userEvent.setup();
    const onBranch = vi.fn();
    render(<TimeTravelSurface run={run} initialSelection={{ position: 2 }} interventions={[{ id: "force-is", label: "Force \"is\"", summary: "Recorded candidate token." }]} onBranch={onBranch} />);

    expect(onBranch).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Stage exact fork" }));
    expect(screen.getByText("STAGED — NOT EXECUTED")).toBeInTheDocument();
    expect(onBranch).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Branch here" }));
    expect(onBranch).toHaveBeenCalledWith({ runId: "run_parent", position: 2, mode: "exact", intervention: { id: "force-is", label: "Force \"is\"", summary: "Recorded candidate token." } });
  });

  test("never fabricates a child when the unchanged control diverges", () => {
    render(<TimeTravelSurface run={{ ...run, execution: { state: "control_diverged", detail: "Decoded suffix differed.", interventionRan: false } }} initialSelection={{ position: 1 }} />);
    expect(screen.getByText(/the intervention was not run/i)).toBeInTheDocument();
    expect(screen.queryByText(/child run results/i)).not.toBeInTheDocument();
  });

  test("hands completed children to Compare without asserting a meaningful difference", async () => {
    const user = userEvent.setup();
    const onOpenCompare = vi.fn();
    render(<TimeTravelSurface run={{ ...run, children: [{ runId: "run_child", intervention: { id: "force-is", label: "Force \"is\"" }, exactness: "verified_exact" }] }} initialSelection={{ position: 1 }} onOpenCompare={onOpenCompare} />);
    await user.click(screen.getByRole("button", { name: "Open in Compare" }));
    expect(onOpenCompare).toHaveBeenCalledWith("run_child", "run_parent");
    expect(screen.getByText(/does not establish that a difference is meaningful/i)).toBeInTheDocument();
  });
});
