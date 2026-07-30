import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { ObservatoryData, RuntimeState } from "../../data/types";
import { deferred } from "../../test/fetch";
import {
  cancelWorkbenchJob,
  loadTokenWorkbench,
  loadWorkbenchJob,
  postCausalTraceAction,
  postForkAction,
  postMechanisticDiffAction,
  postSourceMeasureAction,
  type WorkbenchDocument,
  type WorkbenchJob,
} from "../../data/tokenWorkbench";
import { useTokenWorkbench } from "./useTokenWorkbench";

vi.mock("../../data/tokenWorkbench", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../data/tokenWorkbench")>();
  return {
    ...actual,
    loadTokenWorkbench: vi.fn(),
    postForkAction: vi.fn(),
    postCausalTraceAction: vi.fn(),
    postSourceMeasureAction: vi.fn(),
    postMechanisticDiffAction: vi.fn(),
    loadWorkbenchJob: vi.fn(),
    cancelWorkbenchJob: vi.fn(),
  };
});

const loadDoc = vi.mocked(loadTokenWorkbench);
const forkAction = vi.mocked(postForkAction);
const causalAction = vi.mocked(postCausalTraceAction);
const sourceAction = vi.mocked(postSourceMeasureAction);
const diffAction = vi.mocked(postMechanisticDiffAction);
const jobStatus = vi.mocked(loadWorkbenchJob);
const jobCancel = vi.mocked(cancelWorkbenchJob);

function run(id: string, tokenPrefix = "t"): ObservatoryData {
  return {
    id,
    label: id,
    model: "model-x",
    quant: "Q5_K_M",
    createdAt: "12:00:00",
    duration: "1.0 s",
    mode: "run",
    prompt: `${id} prompt`,
    response: `${tokenPrefix}0${tokenPrefix}1${tokenPrefix}2`,
    tokens: [0, 1, 2].map((index) => ({
      text: `${tokenPrefix}${index}`,
      entropy: 0.2,
      confidence: 0.8 - index / 10,
    })),
    candidates: [{ token: `${tokenPrefix}0`, score: 0.8, delta: 0 }],
    sources: [],
    configuration: { activeDials: {}, memoryCards: [], adapters: [], changes: [] },
  };
}

const runtime: RuntimeState = {
  status: "connected",
  runs: [],
  engine: { model: "model-x", layerCount: 4, jlens: false, sae: false },
};

/** A fully valid WorkbenchDocument fixture -- every capability defaults to unavailable so a test only
 * has to override the ONE capability it is exercising, never accidentally leaves the other three
 * ambiguous. */
function doc(index: number, overrides: Partial<WorkbenchDocument> = {}): WorkbenchDocument {
  return {
    schemaVersion: "clozn.token-workbench.v1",
    runId: "run-a",
    index,
    run: { id: "run-a" },
    token: {
      index,
      piece: `t${index}`,
      tokenId: 100 + index,
      alternatives: [
        { piece: `t${index}`, tokenId: 100 + index, prob: 0.8 },
        { piece: "alt", tokenId: 999, prob: 0.1 },
      ],
    },
    context: { state: "unavailable", raw: {} },
    comparison: { state: "unavailable", raw: {} },
    readouts: { state: "unavailable", raw: {} },
    capabilities: {
      exactFork: { available: false, snapshotState: "not_attempted", reason: "no worker" },
      sourceMeasurement: { available: false, status: "unavailable", reason: "no worker" },
      causalTrace: { available: false, status: "unavailable", reason: "no worker" },
      mechanisticDiff: { available: false, reason: "no reference run selected" },
    },
    ...overrides,
  };
}

function job(overrides: Partial<WorkbenchJob> = {}): WorkbenchJob {
  return {
    schemaVersion: "clozn.influence-map-job.v1",
    jobId: "job-1",
    runId: "run-a",
    kind: "causal_trace",
    state: "running",
    progress: { phase: "starting", completedUnits: 0, totalUnits: 1, percent: 0 },
    cancelRequested: false,
    cancellable: true,
    cached: false,
    ...overrides,
  };
}

beforeEach(() => {
  loadDoc.mockReset();
  forkAction.mockReset();
  causalAction.mockReset();
  sourceAction.mockReset();
  diffAction.mockReset();
  jobStatus.mockReset();
  jobCancel.mockReset();
  loadDoc.mockImplementation(async (_runId, index) => doc(index));
});

describe("selecting a token issues only the GET", () => {
  test("no action POST fires from a token/run selection alone", async () => {
    const data = run("run-a");
    const { result } = renderHook(() => useTokenWorkbench({
      data, runtime, onSelectRun: vi.fn(),
    }));

    await waitFor(() => expect(loadDoc).toHaveBeenCalledTimes(1));
    expect(loadDoc).toHaveBeenCalledWith("run-a", expect.any(Number), undefined, expect.any(AbortSignal));

    // The fixture's weakest-confidence token is already selected by default (index 2) -- pick a
    // genuinely different one so this assertion actually exercises a selection change.
    const nextToken = result.current.selection.token === 0 ? 1 : 0;
    act(() => result.current.setSelectedToken(nextToken));
    await waitFor(() => expect(loadDoc).toHaveBeenCalledTimes(2));
    expect(loadDoc).toHaveBeenLastCalledWith("run-a", nextToken, undefined, expect.any(AbortSignal));

    act(() => result.current.setSelectedLayer(3));
    act(() => result.current.setView("layers"));
    // Neither layer nor view selection is part of the workbench GET's identity -- selecting them must
    // never trigger a second fetch.
    expect(loadDoc).toHaveBeenCalledTimes(2);

    expect(forkAction).not.toHaveBeenCalled();
    expect(causalAction).not.toHaveBeenCalled();
    expect(sourceAction).not.toHaveBeenCalled();
    expect(diffAction).not.toHaveBeenCalled();
  });
});

describe("a stale workbench response can never overwrite a newer selection", () => {
  test("token 1 resolving after token 2 does not win", async () => {
    const data = run("run-a");
    const first = deferred<WorkbenchDocument>();
    const second = deferred<WorkbenchDocument>();
    loadDoc.mockImplementation(async (_runId, index) => {
      if (index === 1) return first.promise;
      if (index === 2) return second.promise;
      return doc(index);
    });

    const { result } = renderHook(() => useTokenWorkbench({ data, runtime, onSelectRun: vi.fn() }));
    await waitFor(() => expect(loadDoc).toHaveBeenCalledTimes(1));

    act(() => result.current.setSelectedToken(1));
    await waitFor(() => expect(loadDoc).toHaveBeenCalledTimes(2));
    act(() => result.current.setSelectedToken(2));
    await waitFor(() => expect(loadDoc).toHaveBeenCalledTimes(3));

    // The LATER request (token 2) resolves first; the STALE token-1 response arrives after.
    await act(async () => second.resolve(doc(2)));
    await waitFor(() => expect(result.current.doc?.index).toBe(2));
    await act(async () => first.resolve(doc(1)));
    // Still 2 -- the stale response for the superseded selection never landed.
    expect(result.current.doc?.index).toBe(2);
  });
});

describe("action outcomes", () => {
  test("exact_fork reaching cached navigates via onSelectRun", async () => {
    const data = run("run-a");
    loadDoc.mockResolvedValue(doc(0, {
      capabilities: {
        exactFork: { available: true, snapshotState: "not_attempted" },
        sourceMeasurement: { available: false, status: "unavailable", reason: "x" },
        causalTrace: { available: false, status: "unavailable", reason: "x" },
        mechanisticDiff: { available: false, reason: "x" },
      },
    }));
    forkAction.mockResolvedValue({
      outcome: "cached",
      artifact: {
        outcome: { kind: "reconstructed_replay", reasons: [], exactness: {}, unavoidableDifferences: [], retokenized: true },
        child: { id: "child-1", parentId: "run-a", note: "reused" },
      },
    });
    const onSelectRun = vi.fn();
    const { result } = renderHook(() => useTokenWorkbench({ data, runtime, onSelectRun }));
    await waitFor(() => expect(result.current.actions.exact_fork.phase).toBe("idle"));

    act(() => result.current.runAction("exact_fork"));
    await waitFor(() => expect(result.current.actions.exact_fork.phase).toBe("cached"));
    expect(onSelectRun).toHaveBeenCalledWith("child-1");
  });

  test("causal_trace reaching a completed job polls to completion", async () => {
    const data = run("run-a");
    loadDoc.mockResolvedValue(doc(0, {
      capabilities: {
        exactFork: { available: false, snapshotState: "x", reason: "x" },
        sourceMeasurement: { available: false, status: "unavailable", reason: "x" },
        causalTrace: { available: true, status: "ready" },
        mechanisticDiff: { available: false, reason: "x" },
      },
    }));
    causalAction.mockResolvedValue({ outcome: "job", job: job({ state: "running" }) });
    jobStatus.mockResolvedValue(job({
      state: "completed",
      progress: { phase: "done", completedUnits: 1, totalUnits: 1, percent: 100 },
      result: {
        schema_version: "clozn.token-workbench-action.v1",
        action: "causal_trace",
        outcome: "ok",
        result: { ok: true, nodes: [], all_candidates: [] },
      },
    }));
    const { result } = renderHook(() => useTokenWorkbench({ data, runtime, onSelectRun: vi.fn() }));
    await waitFor(() => expect(result.current.actions.causal_trace.phase).toBe("idle"));

    act(() => result.current.runAction("causal_trace"));
    await waitFor(() => expect(result.current.actions.causal_trace.phase).toBe("running"));
    await waitFor(() => expect(result.current.actions.causal_trace.phase).toBe("completed"), { timeout: 3000 });
    expect(jobStatus).toHaveBeenCalled();
  });

  test("mechanistic_diff reaching unavailable carries the typed reason and pair-compatibility report", async () => {
    const data = run("run-a");
    loadDoc.mockResolvedValue(doc(0, {
      referenceRunId: "run-b",
      capabilities: {
        exactFork: { available: false, snapshotState: "x", reason: "x" },
        sourceMeasurement: { available: false, status: "unavailable", reason: "x" },
        causalTrace: { available: false, status: "unavailable", reason: "x" },
        mechanisticDiff: { available: true, reason: "eligible" },
      },
    }));
    diffAction.mockResolvedValue({
      outcome: "unavailable",
      reason: { code: "cross_model_execution_not_wired", message: "not yet wired" },
      pairCompatibility: { verdict: { operations: { per_token_comparison: { permitted: true } } } },
    });
    const { result } = renderHook(() => useTokenWorkbench({
      data, runtime, onSelectRun: vi.fn(), initialState: { reference: "run-b" },
    }));
    await waitFor(() => expect(result.current.actions.mechanistic_diff.phase).toBe("idle"));

    act(() => result.current.runAction("mechanistic_diff"));
    await waitFor(() => expect(result.current.actions.mechanistic_diff.phase).toBe("unavailable"));
    expect(result.current.actions.mechanistic_diff.reason).toBe("not yet wired");
    expect(result.current.actions.mechanistic_diff.pairCompatibility).toBeDefined();
  });

  test("a capability the workbench reports unavailable never becomes runnable", async () => {
    const data = run("run-a");
    loadDoc.mockResolvedValue(doc(0));
    const { result } = renderHook(() => useTokenWorkbench({ data, runtime, onSelectRun: vi.fn() }));
    await waitFor(() => expect(result.current.workbench.status).toBe("loaded"));

    expect(result.current.actions.source_measurement.phase).toBe("unavailable");
    expect(result.current.actions.source_measurement.reason).toBe("no worker");
    act(() => result.current.runAction("source_measurement"));
    // No capability => no request ever leaves this hook for that action.
    expect(sourceAction).not.toHaveBeenCalled();
  });

  test("cancelling a running job stops reporting without claiming the server call stopped", async () => {
    const data = run("run-a");
    loadDoc.mockResolvedValue(doc(0, {
      capabilities: {
        exactFork: { available: false, snapshotState: "x", reason: "x" },
        sourceMeasurement: { available: false, status: "unavailable", reason: "x" },
        causalTrace: { available: true, status: "ready" },
        mechanisticDiff: { available: false, reason: "x" },
      },
    }));
    causalAction.mockResolvedValue({ outcome: "job", job: job({ state: "running" }) });
    jobCancel.mockResolvedValue(job({ state: "cancelled", cancelRequested: true, cancellable: false }));
    jobStatus.mockResolvedValue(job({ state: "cancelled", cancelRequested: true, cancellable: false }));
    const { result } = renderHook(() => useTokenWorkbench({ data, runtime, onSelectRun: vi.fn() }));
    await waitFor(() => expect(result.current.actions.causal_trace.phase).toBe("idle"));

    act(() => result.current.runAction("causal_trace"));
    await waitFor(() => expect(result.current.actions.causal_trace.phase).toBe("running"));
    act(() => result.current.cancelAction("causal_trace"));
    await waitFor(() => expect(jobCancel).toHaveBeenCalled());
    await waitFor(() => expect(result.current.actions.causal_trace.phase).toBe("cancelled"), { timeout: 3000 });
    // Constraint: never imply the underlying compute stopped -- only that this Studio stopped watching.
    expect(result.current.actions.causal_trace.reason).toMatch(/stopped reporting/);
    expect(result.current.actions.causal_trace.reason).toMatch(/keeps running to completion/);
  });
});

describe("selection reported to the URL", () => {
  test("onStateChange reflects every field the selection reducer owns", async () => {
    const data = run("run-a");
    const onStateChange = vi.fn();
    const { result } = renderHook(() => useTokenWorkbench({
      data, runtime, onSelectRun: vi.fn(), onStateChange,
    }));
    await waitFor(() => expect(loadDoc).toHaveBeenCalled());

    act(() => result.current.setView("layers"));
    act(() => result.current.setSelectedToken(2));
    act(() => result.current.setSelectedLayer(3));
    act(() => result.current.setVariantReferenceId("run-b"));

    await waitFor(() => expect(onStateChange).toHaveBeenLastCalledWith({
      view: "layers",
      token: 2,
      reference: "run-b",
      layer: 3,
    }));
  });
});
