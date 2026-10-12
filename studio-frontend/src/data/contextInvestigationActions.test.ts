import { afterEach, describe, expect, it, vi } from "vitest";
import removeResultFixture from "../../../tests/fixtures/schemas/clozn.context-investigation-remove-test-result.v1/valid__unavailable.json";
import counterfactualResultFixture from "../../../tests/fixtures/schemas/clozn.context-investigation-counterfactual-result.v1/valid__completed.json";
import { ContractError } from "./contracts";
import {
  decodeContextCounterfactualJob,
  decodeCounterfactualMaterializationResult,
  decodeRemoveSourceTestJob,
} from "./contextInvestigationActions";
import {
  ContextInvestigationHttpError,
  contextInvestigationApi,
} from "./contextInvestigationClient";

type WireDocument = Record<string, any>;

const runId = "run_fixture";
const sourceId = "src_fixture";
const jobId = "infjob_fixture";

function actionJob(
  kind: "context_investigation_remove_test" | "context_investigation_counterfactual",
  result?: WireDocument,
  state = "completed",
): WireDocument {
  return {
    schema_version: "clozn.influence-map-job.v1",
    job_id: jobId,
    run_id: runId,
    kind,
    state,
    progress: {
      phase: state === "completed" ? "done" : state === "failed" ? "counterfactual_generation" : state,
      completed_units: state === "completed" ? 1 : 0,
      total_units: 1,
      percent: state === "completed" ? 100 : 0,
    },
    cancel_requested: state === "cancelling" || state === "cancelled",
    cancellable: !["completed", "failed", "cancelled"].includes(state),
    cached: false,
    created_at: "2026-10-08T00:00:00+00:00",
    updated_at: "2026-10-08T00:00:01+00:00",
    ...(result === undefined ? {} : { result }),
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 200 || status === 201 || status === 202 ? "OK" : "Conflict",
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Context Investigation source-action contracts", () => {
  it("decodes the real remove-source result fixture without inventing preservation", () => {
    const job = decodeRemoveSourceTestJob(
      actionJob("context_investigation_remove_test", structuredClone(removeResultFixture)),
      "/remove-test/job",
      runId,
      sourceId,
      jobId,
    );

    expect(job.result).toEqual(expect.objectContaining({
      status: "unavailable",
      observationId: null,
      reasonCode: "replay_unavailable",
    }));
  });

  it("decodes the real generated-observation result fixture and preserves its materialization identity", () => {
    const job = decodeContextCounterfactualJob(
      actionJob("context_investigation_counterfactual", structuredClone(counterfactualResultFixture)),
      "/counterfactual/job",
      runId,
      sourceId,
      jobId,
    );

    expect(job.result).toEqual(expect.objectContaining({
      status: "completed",
      experimentId: "exp_fixture",
      armId: "arm_fixture",
      observationId: "obs_fixture",
      generatedText: "Recorded counterfactual continuation.",
    }));
  });

  it("fails closed on cross-run, cross-source, wrong-kind, and overstated evidence", () => {
    const wrongRun = actionJob("context_investigation_remove_test", structuredClone(removeResultFixture));
    wrongRun.run_id = "run_other";
    expect(() => decodeRemoveSourceTestJob(wrongRun, "/job", runId, sourceId)).toThrow(/does not match/);

    const wrongSource = actionJob("context_investigation_remove_test", {
      ...structuredClone(removeResultFixture), source_id: "src_other",
    });
    expect(() => decodeRemoveSourceTestJob(wrongSource, "/job", runId, sourceId)).toThrow(/does not match/);

    expect(() => decodeRemoveSourceTestJob(
      actionJob("context_investigation_counterfactual", structuredClone(removeResultFixture)),
      "/job",
      runId,
      sourceId,
    )).toThrow(/context_investigation_remove_test/);

    const fabricated = structuredClone(removeResultFixture) as WireDocument;
    fabricated.status = "exact_preserved";
    expect(() => decodeRemoveSourceTestJob(
      actionJob("context_investigation_remove_test", fabricated), "/job", runId, sourceId,
    )).toThrow(/observation_id/);
  });

  it("keeps a worker failure distinct from an evidence-level unavailable result", () => {
    const failed = actionJob("context_investigation_counterfactual", undefined, "failed");
    failed.error = {
      code: "context_investigation_counterfactual_job_failed",
      message: "RuntimeError: worker disconnected",
    };
    const job = decodeContextCounterfactualJob(failed, "/job", runId, sourceId, jobId);
    expect(job.state).toBe("failed");
    expect(job.error?.code).toBe("context_investigation_counterfactual_job_failed");
    expect(job.result).toBeUndefined();

    delete failed.error;
    expect(() => decodeContextCounterfactualJob(failed, "/job", runId, sourceId, jobId)).toThrow(/error/);
  });
});

describe("Context Investigation source-action client", () => {
  it("starts the remove-source preservation test with only one canonical source ID", async () => {
    const queued = actionJob("context_investigation_remove_test", undefined, "queued");
    const fetchMock = vi.fn().mockResolvedValue(json(queued, 202));
    vi.stubGlobal("fetch", fetchMock);

    const job = await contextInvestigationApi.startRemoveSourceTest(runId, sourceId);

    expect(job.state).toBe("queued");
    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run_fixture/context-investigation/remove-test/jobs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ source_id: sourceId }),
      }),
    );
  });

  it("polls and cancels the same encoded remove-source job resource", async () => {
    const encodedJobId = "job / α";
    const running = actionJob("context_investigation_remove_test", undefined, "running");
    running.job_id = encodedJobId;
    const cancelling = actionJob("context_investigation_remove_test", undefined, "cancelling");
    cancelling.job_id = encodedJobId;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json(running))
      .mockResolvedValueOnce(json(cancelling));
    vi.stubGlobal("fetch", fetchMock);

    await contextInvestigationApi.removeSourceTestJob(runId, encodedJobId, sourceId);
    await contextInvestigationApi.cancelRemoveSourceTest(runId, encodedJobId, sourceId);

    expect(fetchMock.mock.calls[0][0]).toBe(
      "/runs/run_fixture/context-investigation/remove-test/jobs/job%20%2F%20%CE%B1",
    );
    expect(fetchMock.mock.calls[0][1]).toEqual(expect.objectContaining({ method: "GET" }));
    expect(fetchMock.mock.calls[1][0]).toBe(
      "/runs/run_fixture/context-investigation/remove-test/jobs/job%20%2F%20%CE%B1/cancel",
    );
    expect(fetchMock.mock.calls[1][1]).toEqual(expect.objectContaining({ method: "POST", body: "{}" }));
  });

  it("starts generation, then polls and cancels the same encoded job resource", async () => {
    const queued = actionJob("context_investigation_counterfactual", undefined, "queued");
    const running = actionJob("context_investigation_counterfactual", undefined, "running");
    const cancelling = actionJob("context_investigation_counterfactual", undefined, "cancelling");
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json(queued, 202))
      .mockResolvedValueOnce(json(running))
      .mockResolvedValueOnce(json(cancelling));
    vi.stubGlobal("fetch", fetchMock);

    await contextInvestigationApi.startContextCounterfactual(runId, sourceId);
    await contextInvestigationApi.contextCounterfactualJob(runId, jobId, sourceId);
    await contextInvestigationApi.cancelContextCounterfactual(runId, jobId, sourceId);

    expect(fetchMock.mock.calls[0][0]).toBe("/runs/run_fixture/context-investigation/counterfactuals/jobs");
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({ source_id: sourceId });
    expect(fetchMock.mock.calls[1][0]).toBe(`/runs/run_fixture/context-investigation/counterfactuals/jobs/${jobId}`);
    expect(fetchMock.mock.calls[1][1]).toEqual(expect.objectContaining({ method: "GET" }));
    expect(fetchMock.mock.calls[2][0]).toBe(`/runs/run_fixture/context-investigation/counterfactuals/jobs/${jobId}/cancel`);
    expect(fetchMock.mock.calls[2][1]).toEqual(expect.objectContaining({ method: "POST", body: "{}" }));
  });

  it("surfaces typed capability refusals instead of hiding their backend reason code", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      error: "counterfactual generation is unavailable",
      code: "context_investigation_generation_capability_unavailable",
    }, 503)));

    const error = await contextInvestigationApi.startContextCounterfactual(runId, sourceId)
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ContextInvestigationHttpError);
    expect(error).toEqual(expect.objectContaining({
      status: 503,
      code: "context_investigation_generation_capability_unavailable",
      serverMessage: "counterfactual generation is unavailable",
    }));
  });

  it("rejects blank identities before dispatch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(contextInvestigationApi.startRemoveSourceTest(runId, " ")).rejects.toBeInstanceOf(ContractError);
    await expect(contextInvestigationApi.contextCounterfactualJob(runId, " ", sourceId)).rejects.toBeInstanceOf(ContractError);
    await expect(contextInvestigationApi.materializeContextCounterfactual(runId, {
      experimentId: "exp_fixture", armId: " ", observationId: "obs_fixture",
    })).rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("Context Investigation counterfactual materialization contract", () => {
  const request = {
    experimentId: "exp_fixture",
    armId: "arm_fixture",
    observationId: "obs_fixture",
  } as const;

  function completedMaterialization(): WireDocument {
    return {
      schema_version: "clozn.experiment-materialization.v1",
      state: "completed",
      parent_run_id: runId,
      child_run_id: "run_child",
      experiment_id: request.experimentId,
      arm_id: request.armId,
      observation_id: request.observationId,
      realized_fidelity: "direct_context_generation",
      trace_state: "available",
      comparison: {},
      compare_path: `#/compare/${runId}/run_child`,
    };
  }

  it("POSTs only persisted evidence identities and returns explicit parent/child lineage", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json(completedMaterialization(), 201));
    vi.stubGlobal("fetch", fetchMock);

    const result = await contextInvestigationApi.materializeContextCounterfactual(runId, request);

    expect(result).toEqual(expect.objectContaining({
      state: "completed",
      parentRunId: runId,
      childRunId: "run_child",
      comparePath: `#/compare/${runId}/run_child`,
    }));
    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run_fixture/context-investigation/counterfactuals/materialize",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          experiment_id: request.experimentId,
          arm_id: request.armId,
          observation_id: request.observationId,
        }),
      }),
    );
  });

  it("rejects stale bindings and a response that rewrites the source Run", () => {
    expect(() => decodeCounterfactualMaterializationResult(
      { ...completedMaterialization(), observation_id: "obs_other" },
      "/materialize",
      runId,
      request,
    )).toThrow(/does not match/);

    expect(() => decodeCounterfactualMaterializationResult(
      {
        ...completedMaterialization(),
        child_run_id: runId,
        compare_path: `#/compare/${runId}/${runId}`,
      },
      "/materialize",
      runId,
      request,
    )).toThrow(/child, not the source Run/);
  });
});
