import { describe, expect, test } from "vitest";
import { boundaryForLocus, canStageFork, fidelityAt } from "./model";
import type { TimeTravelRun } from "./model";

const run: TimeTravelRun = {
  id: "run_parent",
  response: "Recorded answer.",
  loci: [{ id: "locus-1", label: "A close call", boundaryPosition: 4 }],
  fidelityByBoundary: {
    4: {
      reconstructedReplay: { state: "available" },
      exactFork: { state: "requires_live_plan" },
      historicalExactProof: { state: "verified", verifiedExecutionCount: 1 },
    },
  },
};

describe("time travel model", () => {
  test("preserves a locus-to-recorded-boundary coordinate", () => {
    expect(boundaryForLocus(run, "locus-1")).toBe(4);
    expect(boundaryForLocus(run, "missing")).toBeUndefined();
    expect(fidelityAt(run, 4)?.historicalExactProof.state).toBe("verified");
  });

  test("does not consider a live plan executable exactness", () => {
    const fidelity = fidelityAt(run, 4);
    expect(canStageFork(fidelity, "exact")).toBe(false);
    expect(canStageFork(fidelity, "reconstructed")).toBe(true);
  });
});
