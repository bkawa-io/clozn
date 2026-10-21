import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractError } from "./contracts";
import { contextInvestigationApi } from "./contextInvestigationClient";
import { decodeLeanContextJob } from "./leanContext";

const runId = "run lean/α";

function resultDocument() {
  return {
    schema_version: "clozn.lean-context-result.v1",
    run_id: runId,
    search_id: "lean_0123456789abcdef01234567",
    status: "completed",
    claim: { kind: "best_found_within_budget", minimum_proven: false },
    tolerance: {
      profile: "balanced.v1",
      mean_positive_support_loss_nats_per_token: .05,
      max_token_support_loss_nats: .7,
    },
    universe: { source_ids: ["source-a", "source-b"] },
    original: {
      retained_source_ids: ["source-a", "source-b"],
      removed_source_ids: [],
      rendered_prompt_token_cost: 100,
      tolerance_status: "within_tolerance",
      loss: {
        answer_token_count: 10,
        positive_loss_token_count: 0,
        mean_positive_support_loss_nats_per_token: 0,
        max_token_support_loss_nats: 0,
      },
      evidence: { observation_id: "obs-original" },
    },
    best: {
      retained_source_ids: ["source-a"],
      removed_source_ids: ["source-b"],
      rendered_prompt_token_cost: 62,
      tolerance_status: "within_tolerance",
      loss: {
        answer_token_count: 10,
        positive_loss_token_count: 2,
        mean_positive_support_loss_nats_per_token: .02,
        max_token_support_loss_nats: .2,
      },
      evidence: { observation_id: "obs-best" },
    },
    reduction: {
      removed_prompt_tokens: 38,
      retained_prompt_token_fraction: .62,
      removed_source_count: 1,
      retained_source_count: 1,
    },
    trials: [],
    budget: {
      max_new_executions: 16,
      used_new_executions: 4,
      reused_observation_count: 2,
      exhausted: false,
      blocked_by_budget: false,
    },
    stopping_reason: "search_policy_complete",
  };
}

function jobDocument(state: "queued" | "completed" = "completed") {
  return {
    schema_version: "clozn.analysis-job.v1",
    job_id: "infjob_lean",
    run_id: runId,
    kind: "context_investigation_lean_context",
    state,
    progress: {
      phase: state === "completed" ? "done" : "queued",
      completed_units: state === "completed" ? 1 : 0,
      total_units: 1,
      percent: state === "completed" ? 100 : 0,
    },
    cancel_requested: false,
    cancellable: state !== "completed",
    cached: false,
    ...(state === "completed" ? { result: resultDocument() } : {}),
  };
}

afterEach(() => vi.restoreAllMocks());

describe("Lean Context boundary", () => {
  it("decodes the bounded claim, tolerance, reduction, and reused work", () => {
    const job = decodeLeanContextJob(jobDocument(), "/lean", runId);

    expect(job.result?.claim).toEqual({ kind: "best_found_within_budget", minimumProven: false });
    expect(job.result?.reduction).toEqual({
      removedPromptTokens: 38,
      retainedPromptTokenFraction: .62,
      removedSourceCount: 1,
      retainedSourceCount: 1,
    });
    expect(job.result?.budget.reusedObservationCount).toBe(2);
    expect(job.result?.best?.loss?.meanPositiveSupportLossNatsPerToken).toBe(.02);
  });

  it("rejects a backend result that overclaims a proven minimum", () => {
    const document = jobDocument();
    if (!document.result) throw new Error("fixture result missing");
    document.result.claim.minimum_proven = true;

    expect(() => decodeLeanContextJob(document, "/lean", runId)).toThrow(ContractError);
  });

  it("starts the fixed-profile search with an empty body on the Run-scoped endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(
      JSON.stringify(jobDocument("queued")),
      { status: 202, headers: { "Content-Type": "application/json" } },
    ));

    const job = await contextInvestigationApi.startLeanContext(runId);

    expect(job.state).toBe("queued");
    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run%20lean%2F%CE%B1/context-investigation/lean-context/jobs",
      expect.objectContaining({ method: "POST", body: "{}" }),
    );
  });
});
