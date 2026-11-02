import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractError } from "./contracts";
import {
  decodeContextInteractionsJob,
  type ContextInteractionsJobState,
} from "./contextInteractions";
import { contextInvestigationApi } from "./contextInvestigationClient";

const runId = "run interactions/α";

function resultDocument(status: "running" | "completed" = "completed") {
  return {
    schema_version: "clozn.context-interactions-result.v1",
    run_id: runId,
    status,
    stop_reason: status === "completed" ? "execution_budget" : null,
    seed: 9,
    claim: {
      kind: "exploratory_sampled_marginals",
      redundancy_proven: false,
    },
    source_ids: ["source-a", "source-b"],
    budget: {
      max_new_executions: 16,
      used_new_executions: status === "completed" ? 16 : 4,
      reused_observation_count: 2,
      completed_pairs: status === "completed" ? 7 : 2,
    },
    source_estimates: [{
      source_id: "source-a",
      sample_count: 2,
      unique_pair_count: 2,
      mean_marginal_nats: 1.5,
      sample_stddev_nats: .4,
      leave_one_out_nats: .2,
      mean_minus_leave_one_out_nats: 1.3,
    }, {
      source_id: "source-b",
      sample_count: 0,
      unique_pair_count: 0,
      mean_marginal_nats: null,
      sample_stddev_nats: null,
      leave_one_out_nats: null,
      mean_minus_leave_one_out_nats: null,
    }],
    elapsed_seconds: status === "completed" ? 12.5 : null,
    best: null,
  };
}

function jobDocument(state: ContextInteractionsJobState = "completed") {
  return {
    schema_version: "clozn.analysis-job.v1",
    job_id: "infjob_interactions",
    run_id: runId,
    kind: "context_investigation_interactions",
    state,
    progress: {
      phase: state === "completed" ? "done" : "sampling_context_interactions",
      completed_units: state === "completed" ? 16 : 4,
      total_units: 16,
      percent: state === "completed" ? 100 : 25,
    },
    cancel_requested: false,
    cancellable: state !== "completed",
    cached: false,
    ...(state === "completed" || state === "running"
      ? { result: resultDocument(state === "completed" ? "completed" : "running") }
      : {}),
  };
}

afterEach(() => vi.restoreAllMocks());

describe("Context Interactions boundary", () => {
  it("decodes a progressive result before the job is terminal", () => {
    const job = decodeContextInteractionsJob(jobDocument("running"), "/interactions", runId);

    expect(job.result?.status).toBe("running");
    expect(job.result?.sourceEstimates[0]).toEqual(expect.objectContaining({
      sourceId: "source-a",
      sampleCount: 2,
      meanMarginalNats: 1.5,
      meanMinusLeaveOneOutNats: 1.3,
    }));
    expect(job.result?.sourceEstimates[1].meanMarginalNats).toBeNull();
  });

  it("rejects claims of proven redundancy and incomplete source coverage", () => {
    const overclaim = jobDocument();
    if (!overclaim.result) throw new Error("fixture result missing");
    overclaim.result.claim.redundancy_proven = true;
    expect(() => decodeContextInteractionsJob(overclaim, "/interactions", runId)).toThrow(ContractError);

    const missing = jobDocument();
    if (!missing.result) throw new Error("fixture result missing");
    missing.result.source_estimates.pop();
    expect(() => decodeContextInteractionsJob(missing, "/interactions", runId)).toThrow(ContractError);
  });

  it("starts a bounded deterministic sample on the Run-scoped endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(
      JSON.stringify(jobDocument("queued")),
      { status: 202, headers: { "Content-Type": "application/json" } },
    ));

    const job = await contextInvestigationApi.startInteractions(runId, {
      maxNewExecutions: 24,
      maxPairs: 48,
      seed: 12,
    });

    expect(job.state).toBe("queued");
    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run%20interactions%2F%CE%B1/context-investigation/interactions/jobs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          max_new_executions: 24,
          max_pairs: 48,
          seed: 12,
        }),
      }),
    );
  });
});
