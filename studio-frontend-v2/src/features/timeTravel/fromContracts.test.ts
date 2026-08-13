import { expect, it } from "vitest";
import { projectTimeTravelRun } from "./fromContracts";

it("projects position zero as a live-unchecked prompt boundary and keeps historical proof separate", () => {
  const result = projectTimeTravelRun(
    { id: "run-a", response: "Hello", responseTokens: ["He", "llo"] },
    {
      runId: "run-a",
      recordedTokenCount: 2,
      reconstructedReplay: { state: "available", unavoidableDifferences: ["kv_state_not_restored"] },
      exactRewind: { state: "requires_live_plan", staticPrerequisites: {}, liveRequirements: ["unchanged_control"] },
      historicalProof: { state: "available", verifiedBoundaries: [{ position: 0, state: "historically_verified_exact", verifiedExecutionCount: 1, latestExecutionId: "fork_exec_123", regimes: ["prompt_boundary_reprefill"] }] },
      liveExecution: { state: "not_checked", reason: "read_only_projection", authority: "execution_fork_plan" },
    },
  );
  expect(result.responseTokens?.[0]).toEqual({ position: 0, text: "He" });
  expect(result.fidelityByBoundary?.[0]?.exactFork.state).toBe("requires_live_plan");
  expect(result.fidelityByBoundary?.[0]?.historicalExactProof.state).toBe("verified");
});
