import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { InspectRoute } from "../../app/router";
import { studioApi } from "../../data/client";
import {
  minimalContextApi,
  type MinimalContextDurableResult,
  type MinimalContextJob,
  type MinimalContextResult,
  type MinimalContextResultList,
  type MinimalContextResultSummary,
} from "../../data/minimalContext";
import { InspectSurface } from "./Inspect";
import { MinimalContextSurface } from "./MinimalContext";

const runId = "run_minimal";
const resultId = "mcres_111111111111111111111111";

function route(selectedRunId = runId): InspectRoute {
  return { surface: "inspect", runId: selectedRunId, view: "minimal-context" };
}

function summary(overrides: Partial<MinimalContextResultSummary> = {}): MinimalContextResultSummary {
  return {
    resultId,
    runId,
    searchId: "search_minimal",
    status: "completed",
    searchStatus: "ok",
    certificate: "EXACT_MINIMUM",
    stoppingReason: "exact_minimum_proven",
    best: {
      retainedSourceIds: ["source-keep"],
      removedSourceIds: ["source-drop"],
      renderedPromptTokenCost: 12,
      experimentId: "exp_minimal",
      armId: "arm_minimal",
      observationId: "obs_minimal",
      observationStatus: "exact_preserved",
    },
    reduction: {
      objective: "rendered_prompt_tokens.v1",
      originalPromptTokenCost: 31,
      retainedPromptTokenCost: 12,
      removedSourceCount: 1,
      retainedSourceCount: 1,
      fraction: 19 / 31,
      percent: (19 / 31) * 100,
    },
    baseExecutionFingerprint: "fingerprint-minimal",
    ...overrides,
    schemaVersion: "clozn.minimal-context-search-result.v2",
    currentBinding: overrides.currentBinding ?? { status: "current", reason: null },
  };
}

function result(overrides: Partial<MinimalContextDurableResult> = {}): MinimalContextDurableResult {
  return {
    ...summary(),
    reason: null,
    reasonCode: null,
    controlObservationId: "obs_control",
    universe: {
      schemaVersion: "clozn.context-search-universe.v1",
      runId,
      universeId: "mcu_111111111111111111111111",
      basisContextUnitsDigest: "a".repeat(64),
      policy: {
        kind: "flat_markdown_partition",
        version: "flat_markdown_partition.v1",
        maxUnits: 50,
      },
      sourceIds: ["source-keep", "source-drop"],
      sourceCount: 2,
      coverage: {
        removableMessageIndices: [0, 1],
        removableContentCovered: true,
        partitionVerified: true,
        completeCatalogSourceCount: 2,
        selectedSourceCount: 2,
      },
      status: "planned",
    },
    objective: { kind: "rendered_prompt_tokens", version: "rendered_prompt_tokens.v1" },
    trials: [],
    trajectory: [],
    original: {
      retainedSourceIds: ["source-keep", "source-drop"],
      removedSourceIds: [],
      renderedPromptTokenCost: 31,
      experimentId: "exp_control",
      armId: "arm_control",
      observationId: "obs_control",
      observationStatus: "exact_preserved",
    },
    experimentAccounting: {
      control: { disposition: "executed", observationId: "obs_control" },
      candidateTrials: 4,
      newCounterfactualExecutions: 4,
      reusedObservations: 2,
      directCandidatesObserved: 4,
      preservedCount: 2,
      divergedCount: 2,
      unknownCount: 0,
    },
    sourceInspection: [
      {
        sourceId: "source-keep",
        segmentId: "segment-keep",
        messageIndex: 0,
        label: "System policy",
        provenanceKind: "message",
        parentSourceId: null,
        unicodeRange: [0, 18],
        byteRange: [0, 18],
        granularity: "whole_segment",
        text: "Keep this instruction",
        disposition: "retained",
      },
      {
        sourceId: "source-drop",
        segmentId: "segment-drop",
        messageIndex: 1,
        label: "Retrieved note",
        provenanceKind: "retrieval",
        parentSourceId: null,
        unicodeRange: [0, 12],
        byteRange: [0, 12],
        granularity: "exact_span",
        text: null,
        disposition: "removed",
      },
    ],
    proof: {
      certificate: "EXACT_MINIMUM",
      trajectory: [],
      inclusion: { attempted: true, complete: true, children: [] },
      exactMinimum: {},
    },
    policy: {
      kind: "adaptive_bounded_deletion",
      version: "adaptive_bounded_deletion.v1",
      attemptInclusionCheck: true,
    },
    budget: {
      maxNewExecutions: 32,
      usedNewExecutions: 4,
      reusedObservationCount: 2,
      exhausted: false,
      blockedByBudget: false,
    },
    inclusionCheck: {
      attempted: true,
      complete: true,
      testedChildCount: 1,
      totalChildCount: 1,
      allChildrenFailed: true,
    },
    ...overrides,
    schemaVersion: "clozn.minimal-context-search-result.v2",
    currentBinding: overrides.currentBinding ?? { status: "current", reason: null },
  };
}

function resultList(...results: MinimalContextResultSummary[]): MinimalContextResultList {
  return { results };
}

function job(overrides: Partial<MinimalContextJob> = {}): MinimalContextJob {
  return {
    schemaVersion: "clozn.analysis-job.v1",
    jobId: "infjob_minimal",
    runId,
    kind: "minimal_context",
    state: "running",
    progress: {
      phase: "searching",
      completedUnits: 2,
      totalUnits: 8,
      percent: 25,
      bestRetainedSourceCount: 1,
      certificateCandidateKind: "BEST_VERIFIED",
    },
    cancelRequested: false,
    cancellable: true,
    cached: false,
    createdAt: "2026-10-10T00:00:00Z",
    updatedAt: "2026-10-10T00:00:01Z",
    ...overrides,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("Minimal Context Inspect instrument", () => {
  it("reads durable list/detail on mount and requires a second explicit materialization click", async () => {
    const listRead = vi.spyOn(minimalContextApi, "results").mockResolvedValue(resultList(summary()));
    const detailRead = vi.spyOn(minimalContextApi, "result").mockResolvedValue(result());
    const start = vi.spyOn(minimalContextApi, "start");
    const cancel = vi.spyOn(minimalContextApi, "cancel");
    const materialize = vi.spyOn(minimalContextApi, "materialize").mockResolvedValue({
      schemaVersion: "clozn.experiment-materialization.v1",
      state: "completed",
      parentRunId: runId,
      childRunId: "run_child",
      experimentId: "exp_minimal",
      armId: "arm_minimal",
      removedSourceIds: ["source-drop"],
    });
    const compare = vi.spyOn(studioApi, "runComparison").mockResolvedValue({
      schemaVersion: "clozn.reduction-comparison.v1",
      parentRunId: runId,
      childRunId: "run_child",
      context: {
        parentSourceIds: ["source-keep", "source-drop"],
        retainedSourceIds: ["source-keep"],
        removedSourceIds: ["source-drop"],
      },
      renderedPromptTokens: { parent: 31, child: 12, saved: 19, improved: true },
      preservation: { answerExactMatch: true, finishReasonMatch: true, exact: true },
      generationIdentity: { parent: {}, child: {}, match: true },
    });

    render(<InspectSurface route={route()} />);

    expect(await screen.findByRole("heading", { name: "Minimal context" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Exact minimum" })).toBeInTheDocument();
    expect(screen.getByText(/every candidate in the finite search space/i)).toBeInTheDocument();
    expect(screen.getByText("4 / 32")).toBeInTheDocument();
    expect(screen.getByText(/directly tested 1 of 1/i)).toBeInTheDocument();
    expect(screen.getByText("System policy")).toBeInTheDocument();
    expect(screen.getByText("Keep this instruction")).toBeInTheDocument();
    expect(screen.queryByText("Source text was not supplied by this stored result.")).not.toBeInTheDocument();

    expect(listRead).toHaveBeenCalledTimes(1);
    expect(listRead.mock.calls[0]?.[0]).toBe(runId);
    expect(listRead.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    expect(detailRead).toHaveBeenCalledTimes(1);
    expect(detailRead.mock.calls[0]?.[0]).toBe(runId);
    expect(detailRead.mock.calls[0]?.[1]).toBe(resultId);
    expect(detailRead.mock.calls[0]?.[2]).toBeInstanceOf(AbortSignal);
    expect(start).not.toHaveBeenCalled();
    expect(cancel).not.toHaveBeenCalled();
    expect(materialize).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Materialize…" }));
    expect(materialize).not.toHaveBeenCalled();
    expect(screen.getByRole("group", { name: /confirm minimal context child run/i })).toBeInTheDocument();
    expect(screen.getByText(/fresh child replay with model work/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Create child Run" }));
    await waitFor(() => expect(materialize).toHaveBeenCalledTimes(1));
    expect(materialize.mock.calls[0]?.[0]).toBe(runId);
    expect(materialize.mock.calls[0]?.[1].resultId).toBe(resultId);
    expect(materialize.mock.calls[0]?.[2]).toBeInstanceOf(AbortSignal);
    expect(await screen.findByText("Child Run created")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Inspect child" })).toHaveAttribute("href", "#/inspect/run_child");
    expect(await screen.findByRole("heading", { name: "Original vs reduced child" })).toBeInTheDocument();
    expect(screen.getByText("19")).toBeInTheDocument();
    expect(compare).toHaveBeenCalledWith(runId, "run_child", expect.any(AbortSignal));
  });

  it("shows history honestly and blocks materialization for stale and incomplete results", async () => {
    const oldId = "mcres_222222222222222222222222";
    const staleSummary = summary({
      resultId: oldId,
      certificate: "BEST_VERIFIED",
      stoppingReason: "budget_exhausted",
      currentBinding: { status: "stale", reason: "the current source universe changed" },
    });
    vi.spyOn(minimalContextApi, "results").mockResolvedValue(resultList(staleSummary));
    vi.spyOn(minimalContextApi, "result").mockResolvedValue(result({
      ...staleSummary,
      resultId: oldId,
      status: "completed",
      certificate: "BEST_VERIFIED",
      stoppingReason: "budget_exhausted",
      currentBinding: staleSummary.currentBinding,
      budget: {
        maxNewExecutions: 32,
        usedNewExecutions: 32,
        reusedObservationCount: 0,
        exhausted: true,
        blockedByBudget: true,
      },
      inclusionCheck: {
        attempted: true,
        complete: false,
        testedChildCount: 0,
        totalChildCount: 1,
        allChildrenFailed: false,
      },
    }));
    const materialize = vi.spyOn(minimalContextApi, "materialize");

    render(<MinimalContextSurface route={route()} />);

    expect(await screen.findByRole("heading", { name: "Best verified" })).toBeInTheDocument();
    expect(screen.getByText(/a smaller preserving candidate may exist/i)).toBeInTheDocument();
    expect(screen.getByText("Historical result")).toBeInTheDocument();
    expect(screen.getByText("the current source universe changed")).toBeInTheDocument();
    expect(screen.getByText(/1 remain unknown/i)).toBeInTheDocument();
    expect(screen.getByText(/not materializable:.*current source universe changed/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Materialize…" })).not.toBeInTheDocument();
    expect(materialize).not.toHaveBeenCalled();
  });

  it("starts only on request, polls, then rereads the durable list and detail", async () => {
    const completed = result();
    const listRead = vi.spyOn(minimalContextApi, "results")
      .mockResolvedValueOnce(resultList())
      .mockResolvedValueOnce(resultList(summary()));
    vi.spyOn(minimalContextApi, "start").mockResolvedValue(job());
    vi.spyOn(minimalContextApi, "job").mockResolvedValue(job({
      state: "completed",
      cancellable: false,
      progress: { phase: "done", completedUnits: 1, totalUnits: 1, percent: 100 },
      result: completed,
    }));
    const detailRead = vi.spyOn(minimalContextApi, "result").mockResolvedValue(completed);

    render(<MinimalContextSurface route={route()} />);
    expect(await screen.findByRole("heading", { name: /no verified reduction/i })).toBeInTheDocument();
    expect(listRead).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Run exact search" }));

    await waitFor(() => expect(minimalContextApi.start).toHaveBeenCalledWith(
      runId,
      expect.any(AbortSignal),
    ));
    await waitFor(() => expect(minimalContextApi.job).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(listRead).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(detailRead).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("heading", { name: "Exact minimum" })).toBeInTheDocument();
  });

  it("cancels explicitly and never cancels merely because the route changes", async () => {
    vi.spyOn(minimalContextApi, "results").mockResolvedValue(resultList());
    vi.spyOn(minimalContextApi, "start").mockResolvedValue(job());
    vi.spyOn(minimalContextApi, "job").mockReturnValue(new Promise(() => {}));
    const cancel = vi.spyOn(minimalContextApi, "cancel").mockResolvedValue(job({
      state: "cancelled",
      cancellable: false,
      cancelRequested: true,
      progress: { phase: "cancelled", completedUnits: 2, totalUnits: 8, percent: 25 },
    }));

    const view = render(<MinimalContextSurface route={route()} />);
    await screen.findByRole("heading", { name: /no verified reduction/i });
    fireEvent.click(screen.getByRole("button", { name: "Run exact search" }));
    expect(await screen.findByRole("button", { name: "Cancel search" })).toBeInTheDocument();

    view.rerender(<MinimalContextSurface route={route("run_other")} />);
    expect(cancel).not.toHaveBeenCalled();

    await screen.findByRole("heading", { name: /no verified reduction/i });
    fireEvent.click(screen.getByRole("button", { name: "Run exact search" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel search" }));
    await waitFor(() => expect(cancel).toHaveBeenCalledTimes(1));
    expect(cancel.mock.calls[0]?.[0]).toBe("run_other");
  });

  it("ignores late list reads from a prior Run", async () => {
    const first = deferred<MinimalContextResultList>();
    const listRead = vi.spyOn(minimalContextApi, "results")
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce(resultList({ ...summary(), runId: "run_new" }));
    vi.spyOn(minimalContextApi, "result").mockResolvedValue({ ...result(), runId: "run_new" });

    const view = render(<MinimalContextSurface route={route("run_old")} />);
    const oldSignal = listRead.mock.calls[0]?.[1];
    view.rerender(<MinimalContextSurface route={route("run_new")} />);
    expect(oldSignal?.aborted).toBe(true);

    await screen.findByRole("heading", { name: "Exact minimum" });
    await act(async () => first.resolve(resultList()));
    expect(screen.getByRole("combobox", { name: "Stored result" })).toHaveValue(resultId);
    expect(screen.getByText("run_new")).toBeInTheDocument();
    expect(screen.queryByText("run_old")).not.toBeInTheDocument();
  });

  it("reveals only the source text supplied by the durable result", async () => {
    vi.spyOn(minimalContextApi, "results").mockResolvedValue(resultList(summary()));
    vi.spyOn(minimalContextApi, "result").mockResolvedValue(result());

    render(<MinimalContextSurface route={route()} />);
    const sourceRegion = await screen.findByRole("heading", { name: "Retained and removed sources" });
    const sourceSection = sourceRegion.closest("section");
    expect(sourceSection).not.toBeNull();
    fireEvent.click(within(sourceSection!).getByRole("button", { name: /retrieved note.*removed/i }));
    await waitFor(() => expect(
      within(sourceSection!).getByText("Source text was not supplied by this stored result."),
    ).toBeInTheDocument());
    expect(within(sourceSection!).queryByText(/recorded source text unavailable/i)).not.toBeInTheDocument();
  });
});
