import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractError } from "./contracts";
import {
  decodeMinimalContextJob,
  decodeMinimalContextMaterializationResult,
  decodeMinimalContextResult,
  decodeMinimalContextResultList,
  minimalContextApi,
  minimalContextMaterializationBlocker,
  MinimalContextHttpError,
  type MinimalContextResult,
} from "./minimalContext";

type WireDocument = Record<string, any>;

const runId = "run_fixture";
const resultId = "mcres_0123456789abcdef01234567";
const sourceA = "src_aaaaaaaaaaaaaaaaaaaaaaaa";
const sourceB = "src_bbbbbbbbbbbbbbbbbbbbbbbb";
const sourceC = "src_cccccccccccccccccccccccc";
const jobId = "infjob_fixture";

function candidate(retained: string[], removed: string[], cost: number, suffix: string): WireDocument {
  return {
    retained_source_ids: retained,
    removed_source_ids: removed,
    rendered_prompt_token_cost: cost,
    experiment_id: `exp_${suffix}`,
    arm_id: `arm_${suffix}`,
    observation_id: `obs_${suffix}`,
    observation_status: "exact_preserved",
  };
}

const trajectory = [{
  counterfactual_probe_count: 1,
  retained_ids: [sourceA],
  cost: 7,
  retained_unit_count: 1,
  stage: "inclusion_check",
}];

function resultDocument(includeBinding = true): WireDocument {
  return {
    schema_version: "clozn.minimal-context-search-result.v2",
    result_id: resultId,
    run_id: runId,
    search_id: "search_fixture",
    status: "completed",
    search_status: "ok",
    reason: null,
    reason_code: null,
    base_execution_fingerprint: "execution_fixture",
    universe: {
      schema_version: "clozn.context-search-universe.v1",
      run_id: runId,
      universe_id: "mcu_0123456789abcdef01234567",
      basis_context_units_digest: "a".repeat(64),
      policy: {
        kind: "bounded_structural_partition",
        version: "bounded_structural_partition.v1",
        max_units: 50,
        merge: "smallest_adjacent_auto_siblings_leftmost.v1",
        explicit_over_cap: "typed_bound_exceeded",
      },
      source_ids: [sourceA, sourceB],
      source_count: 2,
      coverage: {
        protected_message_indices: [2],
        removable_message_indices: [0, 1],
        removable_content_covered: true,
        partition_verified: true,
        complete_catalog_source_count: 2,
        explicit_source_count: 0,
        selected_source_count: 2,
      },
      status: "planned",
    },
    objective: { kind: "rendered_prompt_tokens", version: "rendered_prompt_tokens.v1" },
    control_observation_id: "obs_control",
    trials: [{
      ordinal: 0,
      stage: "inclusion_check",
      retained_ids: [sourceA],
      cost: 7,
      classification: "preserves",
      disposition: "executed",
      experiment_id: "exp_winner",
      arm_id: "arm_winner",
      observation_id: "obs_winner",
      observation_status: "exact_preserved",
      evidence: {
        disposition: "executed",
        experiment_id: "exp_winner",
        arm_id: "arm_winner",
        observation_id: "obs_winner",
        observation_status: "exact_preserved",
      },
      batch_id: 1,
      parent_retained_ids: [sourceA, sourceB],
    }],
    trajectory,
    original: candidate([sourceA, sourceB], [], 10, "control"),
    best: candidate([sourceA], [sourceB], 7, "winner"),
    reduction: {
      objective: "rendered_prompt_tokens.v1",
      original_prompt_token_cost: 10,
      retained_prompt_token_cost: 7,
      removed_source_count: 1,
      retained_source_count: 1,
      fraction: 0.3,
      percent: 30,
    },
    certificate: "BEST_VERIFIED",
    stopping_reason: "search_policy_complete",
    experiment_accounting: {
      control: { disposition: "executed", observation_id: "obs_control" },
      candidate_trials: 1,
      new_counterfactual_executions: 1,
      reused_observations: 0,
      direct_candidates_observed: 1,
      preserved_count: 1,
      diverged_count: 0,
      unknown_count: 0,
    },
    source_inspection: [
      {
        source_id: sourceA,
        segment_id: "seg_aaaaaaaaaaaaaaaa",
        message_index: 0,
        label: "Policy",
        provenance_kind: "system",
        parent_source_id: null,
        unicode_range: [0, 6],
        byte_range: [0, 6],
        granularity: "exact_span",
        text: "Policy",
        disposition: "retained",
      },
      {
        source_id: sourceB,
        segment_id: "seg_bbbbbbbbbbbbbbbb",
        message_index: 1,
        label: "Example",
        provenance_kind: "retrieval",
        parent_source_id: null,
        unicode_range: [0, 7],
        byte_range: [0, 7],
        granularity: "whole_segment",
        text: "Example",
        disposition: "removed",
      },
    ],
    proof: {
      certificate: "BEST_VERIFIED",
      trajectory,
      inclusion: { attempted: true, complete: true, children: [] },
      exact_minimum: null,
    },
    policy: {
      kind: "adaptive_bounded_deletion",
      version: "adaptive_bounded_deletion.v1",
      attempt_inclusion_check: true,
    },
    budget: {
      max_new_executions: 4,
      used_new_executions: 1,
      reused_observation_count: 0,
      exhausted: false,
      blocked_by_budget: false,
    },
    inclusion_check: {
      attempted: true,
      complete: true,
      tested_child_count: 1,
      total_child_count: 1,
      all_children_failed: true,
    },
    ...(includeBinding ? { current_binding: { status: "current", reason: null } } : {}),
  };
}

function summaryDocument(document = resultDocument()): WireDocument {
  return {
    schema_version: document.schema_version,
    result_id: document.result_id,
    run_id: document.run_id,
    search_id: document.search_id,
    status: document.status,
    search_status: document.search_status,
    certificate: document.certificate,
    stopping_reason: document.stopping_reason,
    best: document.best,
    reduction: document.reduction,
    base_execution_fingerprint: document.base_execution_fingerprint,
    current_binding: document.current_binding,
  };
}

function jobDocument(state = "completed", includeResult = true): WireDocument {
  return {
    schema_version: "clozn.analysis-job.v1",
    job_id: jobId,
    run_id: runId,
    kind: "minimal_context",
    state,
    progress: {
      phase: state === "completed" ? "done" : state,
      completed_units: state === "completed" ? 1 : 0,
      total_units: 1,
      percent: state === "completed" ? 100 : 0,
      ...(state === "running" ? { best_retained_source_count: 1, certificate_candidate_kind: "BEST_VERIFIED" } : {}),
    },
    cancel_requested: state === "cancelling" || state === "cancelled",
    cancellable: !["completed", "failed", "cancelled"].includes(state),
    cached: false,
    created_at: "2026-10-10T00:00:00+00:00",
    updated_at: "2026-10-10T00:00:01+00:00",
    ...(includeResult ? { result: resultDocument(false) } : {}),
  };
}

function materializationDocument(state: "completed" | "failed" = "completed"): WireDocument {
  const shared = {
    schema_version: "clozn.experiment-materialization.v1",
    state,
    parent_run_id: runId,
    experiment_id: "exp_winner",
    arm_id: "arm_winner",
    intervention: { kind: "delete_source", source_ids: [sourceB] },
  };
  return state === "failed" ? { ...shared, reason: "generation_failed" } : {
    ...shared,
    child_run_id: "run_child",
    comparison: {},
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 201 || status === 202 || status === 200 ? "OK" : "Conflict",
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Minimal Context v2 result contracts", () => {
  it("decodes durable proof, source inspection, exact winner evidence, and current binding", () => {
    const result = decodeMinimalContextResult(resultDocument(), "/result", runId, true);

    expect(result).toMatchObject({
      schemaVersion: "clozn.minimal-context-search-result.v2",
      resultId,
      runId,
      certificate: "BEST_VERIFIED",
      best: {
        retainedSourceIds: [sourceA],
        removedSourceIds: [sourceB],
        observationStatus: "exact_preserved",
      },
      reduction: { originalPromptTokenCost: 10, retainedPromptTokenCost: 7, percent: 30 },
      currentBinding: { status: "current", reason: null },
    });
    expect(result.sourceInspection.map((source) => source.disposition)).toEqual(["retained", "removed"]);
    expect(minimalContextMaterializationBlocker(result)).toBeNull();
  });

  it("decodes result summaries without treating them as full proof documents", () => {
    const list = decodeMinimalContextResultList({ results: [summaryDocument()] }, "/results", runId);

    expect(list.results[0]).toMatchObject({ resultId, runId, certificate: "BEST_VERIFIED" });
    expect(list.results[0]).not.toHaveProperty("sourceInspection");
  });

  it("fails closed on identity drift, unexpected top-level fields, partition drift, and false current bindings", () => {
    expect(() => decodeMinimalContextResult({ ...resultDocument(), run_id: "run_other" }, "/result", runId))
      .toThrow(/requested Run/);
    expect(() => decodeMinimalContextResult({ ...resultDocument(), surprise: true }, "/result", runId))
      .toThrow(/not part of the contract/);

    const badPartition = resultDocument();
    badPartition.best.removed_source_ids = [];
    expect(() => decodeMinimalContextResult(badPartition, "/result", runId)).toThrow(/partition/);

    const badBinding = resultDocument();
    badBinding.current_binding = { status: "current", reason: "fingerprint changed" };
    expect(() => decodeMinimalContextResult(badBinding, "/result", runId, true)).toThrow(/must be null/);
  });

  it("requires durable binding and exact winner evidence before materialization", () => {
    const jobResult = decodeMinimalContextResult(resultDocument(false), "/job-result", runId);
    expect(minimalContextMaterializationBlocker(jobResult)).toMatch(/Reload/);

    const stale = decodeMinimalContextResult({
      ...resultDocument(), current_binding: { status: "stale", reason: "the fingerprint changed" },
    }, "/result", runId, true);
    expect(minimalContextMaterializationBlocker(stale)).toBe("the fingerprint changed");

    const weak = resultDocument();
    weak.best.observation_status = "diverged";
    const decodedWeak = decodeMinimalContextResult(weak, "/result", runId, true);
    expect(minimalContextMaterializationBlocker(decodedWeak)).toMatch(/exact-preservation/);
  });
});

describe("Minimal Context job contract", () => {
  it("requires the minimal_context kind, requested identities, and a terminal result", () => {
    const job = decodeMinimalContextJob(jobDocument(), "/job", runId, jobId);
    expect(job).toMatchObject({ state: "completed", kind: "minimal_context", result: { resultId } });

    expect(() => decodeMinimalContextJob({ ...jobDocument(), kind: "influence_map" }, "/job", runId, jobId))
      .toThrow(/minimal_context/);
    expect(() => decodeMinimalContextJob(jobDocument("completed", false), "/job", runId, jobId))
      .toThrow(/result/);
    expect(() => decodeMinimalContextJob({ ...jobDocument(), job_id: "other" }, "/job", runId, jobId))
      .toThrow(/requested job/);
  });

  it("keeps worker failure separate from a product-level unavailable result", () => {
    const failed = jobDocument("failed", false);
    failed.error = { code: "minimal_context_job_failed", message: "worker disconnected" };
    expect(decodeMinimalContextJob(failed, "/job", runId).error?.code).toBe("minimal_context_job_failed");

    delete failed.error;
    expect(() => decodeMinimalContextJob(failed, "/job", runId)).toThrow(/error/);
  });
});

describe("Minimal Context canonical client", () => {
  it("reads the durable result list and exact selected result through encoded GET routes", async () => {
    const encodedRun = "run / α";
    const listBody = { results: [summaryDocument({ ...resultDocument(), run_id: encodedRun })] };
    listBody.results[0].run_id = encodedRun;
    const detail = resultDocument();
    detail.run_id = encodedRun;
    detail.universe.run_id = encodedRun;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json(listBody))
      .mockResolvedValueOnce(json(detail));
    vi.stubGlobal("fetch", fetchMock);

    await minimalContextApi.results(encodedRun);
    await minimalContextApi.result(encodedRun, resultId);

    expect(fetchMock.mock.calls[0][0]).toBe("/runs/run%20%2F%20%CE%B1/minimal-context/results");
    expect(fetchMock.mock.calls[0][1]).toEqual(expect.objectContaining({ method: "GET" }));
    expect(fetchMock.mock.calls[1][0]).toBe(`/runs/run%20%2F%20%CE%B1/minimal-context/results/${resultId}`);
    expect(fetchMock.mock.calls[1][1]).toEqual(expect.objectContaining({ method: "GET" }));
  });

  it("starts the fixed exact bounded-search policy with an empty body", async () => {
    const queued = jobDocument("queued", false);
    const fetchMock = vi.fn().mockResolvedValue(json(queued, 202));
    vi.stubGlobal("fetch", fetchMock);

    await minimalContextApi.start(runId);

    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run_fixture/minimal-context/jobs",
      expect.objectContaining({
        method: "POST",
        body: "{}",
      }),
    );
  });

  it("polls and cancels the same encoded job, with an empty cancellation body", async () => {
    const encodedJob = "job / α";
    const running = jobDocument("running", false);
    running.job_id = encodedJob;
    const cancelling = jobDocument("cancelling", false);
    cancelling.job_id = encodedJob;
    cancelling.cancel_accepted = true;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json(running))
      .mockResolvedValueOnce(json(cancelling));
    vi.stubGlobal("fetch", fetchMock);

    await minimalContextApi.job(runId, encodedJob);
    const cancelled = await minimalContextApi.cancel(runId, encodedJob);

    expect(fetchMock.mock.calls[0][0]).toBe("/runs/run_fixture/minimal-context/jobs/job%20%2F%20%CE%B1");
    expect(fetchMock.mock.calls[1][0]).toBe("/runs/run_fixture/minimal-context/jobs/job%20%2F%20%CE%B1/cancel");
    expect(fetchMock.mock.calls[1][1]).toEqual(expect.objectContaining({ method: "POST", body: "{}" }));
    expect(cancelled.cancelAccepted).toBe(true);
  });

  it("materializes only the decoded result-bound winner and sends no override references", async () => {
    const result = decodeMinimalContextResult(resultDocument(), "/result", runId, true);
    const fetchMock = vi.fn().mockResolvedValue(json(materializationDocument(), 201));
    vi.stubGlobal("fetch", fetchMock);

    const materialized = await minimalContextApi.materialize(runId, result);

    expect(materialized).toEqual(expect.objectContaining({
      state: "completed",
      parentRunId: runId,
      childRunId: "run_child",
      removedSourceIds: [sourceB],
    }));
    expect(fetchMock).toHaveBeenCalledWith(
      `/runs/run_fixture/minimal-context/results/${resultId}/materialize`,
      expect.objectContaining({ method: "POST", body: "{}" }),
    );
  });

  it("decodes the actual exact-replay failure shape returned with 409", async () => {
    const result = decodeMinimalContextResult(resultDocument(), "/result", runId, true);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(materializationDocument("failed"), 409)));

    await expect(minimalContextApi.materialize(runId, result)).resolves.toEqual(expect.objectContaining({
      state: "failed",
      reason: "generation_failed",
      removedSourceIds: [sourceB],
    }));
  });

  it("rejects stale materialization locally and preserves typed backend refusals", async () => {
    const stale = decodeMinimalContextResult({
      ...resultDocument(), current_binding: { status: "stale", reason: "fingerprint changed" },
    }, "/result", runId, true);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await expect(minimalContextApi.materialize(runId, stale)).rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).not.toHaveBeenCalled();

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      error: "result is stale",
      code: "minimal_context_result_stale",
      current_binding: { status: "stale", reason: "fingerprint changed" },
    }, 409)));
    const current = decodeMinimalContextResult(resultDocument(), "/result", runId, true);
    const error = await minimalContextApi.materialize(runId, current).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(MinimalContextHttpError);
    expect(error).toMatchObject({ code: "minimal_context_result_stale", status: 409 });
  });

  it("rejects blank identities before dispatch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(minimalContextApi.start(" ")).rejects.toBeInstanceOf(ContractError);
    await expect(minimalContextApi.job(runId, " ")).rejects.toBeInstanceOf(ContractError);
    await expect(minimalContextApi.results(" ")).rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("Minimal Context exact-replay materialization contract", () => {
  it("treats deletion membership, not display order, as materialization identity", () => {
    const decoded = decodeMinimalContextResult(resultDocument(), "/result", runId, true);
    const result = {
      ...decoded,
      best: { ...decoded.best!, removedSourceIds: [sourceB, sourceC] },
    };
    const materialization = {
      ...materializationDocument(),
      intervention: { kind: "delete_source", source_ids: [sourceC, sourceB] },
    };

    expect(decodeMinimalContextMaterializationResult(
      materialization, "/materialize", runId, result,
    ).removedSourceIds).toEqual([sourceC, sourceB]);
  });

  it("rejects a response that changes winner evidence, deletion set, or child lineage", () => {
    const result = decodeMinimalContextResult(resultDocument(), "/result", runId, true);
    expect(() => decodeMinimalContextMaterializationResult(
      { ...materializationDocument(), arm_id: "arm_other" }, "/materialize", runId, result,
    )).toThrow(/persisted result winner/);
    expect(() => decodeMinimalContextMaterializationResult(
      { ...materializationDocument(), intervention: { kind: "delete_source", source_ids: [sourceA] } },
      "/materialize", runId, result,
    )).toThrow(/persisted result winner/);
    expect(() => decodeMinimalContextMaterializationResult(
      { ...materializationDocument(), child_run_id: runId },
      "/materialize", runId, result,
    )).toThrow(/child, not the source Run/);
  });
});
