import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ContractError } from "../../data/contracts";
import type {
  ContextInvestigationQuery,
  ContextInvestigationReader,
} from "../../data/contextInvestigation";
import { contextInvestigationApi } from "../../data/contextInvestigationClient";
import type { ContextEffectsJob } from "../../data/contextInvestigationJobs";
import type { LeanContextJob } from "../../data/leanContext";
import { InspectSurface } from "./Inspect";

const runId = "run_alpha";
const answerRange = { answerStart: 3, answerEnd: 4 } as const;

function readerFixture(): ContextInvestigationReader {
  const contextText = "A😀B context signal.";
  return {
    schemaVersion: "clozn.context-investigation-reader.v1",
    runId,
    status: "completed",
    measurement: {
      status: "completed",
      experimentId: "experiment_reader",
      evaluator: "score_recorded_continuation",
      intervention: "delete_source",
      sourceUniverse: ["source-a", "source b"],
      sourceUniverseBasis: "context_receipt",
      completedSourceCount: 2,
      unavailableSourceCount: 0,
      failedSourceCount: 0,
      notMeasuredSourceCount: 0,
      baselineObservationId: "observation_baseline",
    },
    context: {
      coordinateBasis: "context_display_unicode.v1",
      separator: "\n\n",
      blocks: [{
        messageIndex: 0,
        role: "user",
        text: contextText,
        segmentId: "segment_0",
        displayUnicodeRange: [0, [...contextText].length],
        sourceIds: ["source-a", "source b"],
      }],
    },
    answer: {
      status: "available",
      text: "Hi 🌕!",
      sha256: "a".repeat(64),
      tokens: [
        { index: 0, tokenId: 10, piece: "Hi ", unicodeRange: [0, 3], logprob: -0.2 },
        { index: 1, tokenId: 11, piece: "🌕", unicodeRange: [3, 4], logprob: -1 },
        { index: 2, tokenId: 12, piece: "!", unicodeRange: [4, 5], logprob: -0.1 },
      ],
      fidelity: "baseline_token_score_observation",
      observationId: "observation_baseline",
    },
    sources: [
      {
        sourceId: "source-a",
        segmentId: "segment_0",
        messageIndex: 0,
        label: null,
        provenanceKind: "message",
        parentSourceId: null,
        unicodeRange: [0, 3],
        byteRange: [0, 6],
        displayUnicodeRange: [0, 3],
        granularity: "exact_span",
        measurement: {
          status: "measured",
          experimentId: "experiment_reader",
          armId: "arm_a",
          observationId: "observation_a",
          intervention: { kind: "delete_source", target: { kind: "context_selection", sourceIds: ["source-a"] } },
        },
        summary: { maxAbsTokenDeltaNats: .42, totalContinuationDeltaNats: .42, aboveFloorLocusCount: 1 },
        loci: ["locus_a"],
      },
      {
        sourceId: "source b",
        segmentId: "segment_0",
        messageIndex: 0,
        label: null,
        provenanceKind: "message",
        parentSourceId: null,
        unicodeRange: [4, 11],
        byteRange: [7, 14],
        displayUnicodeRange: [4, 11],
        granularity: "exact_span",
        measurement: {
          status: "measured",
          experimentId: "experiment_reader",
          armId: "arm_b",
          observationId: "observation_b",
          intervention: { kind: "delete_source", target: { kind: "context_selection", sourceIds: ["source b"] } },
        },
        summary: { maxAbsTokenDeltaNats: .21, totalContinuationDeltaNats: -.21, aboveFloorLocusCount: 1 },
        loci: ["locus_b"],
      },
    ],
    loci: [
      {
        locusId: "locus_a",
        sourceId: "source-a",
        direction: "support",
        classification: "measured_support",
        answerTokenRange: [1, 2],
        answerUnicodeRange: [3, 4],
        answerText: "🌕",
        deltaNats: .42,
        absoluteDeltaNats: .42,
        baselineSelectedLogp: -1,
        intervenedSelectedLogp: -1.42,
        experimentId: "experiment_reader",
        armId: "arm_a",
        baselineObservationId: "observation_baseline",
        interventionObservationId: "observation_a",
        perTokenDeltaNats: [.42],
      },
      {
        locusId: "locus_b",
        sourceId: "source b",
        direction: "suppression",
        classification: "measured_suppression",
        answerTokenRange: [1, 2],
        answerUnicodeRange: [3, 4],
        answerText: "🌕",
        deltaNats: -.21,
        absoluteDeltaNats: .21,
        baselineSelectedLogp: -1,
        intervenedSelectedLogp: -.79,
        experimentId: "experiment_reader",
        armId: "arm_b",
        baselineObservationId: "observation_baseline",
        interventionObservationId: "observation_b",
        perTokenDeltaNats: [-.21],
      },
    ],
    presentation: { measurementFloorNats: .1 },
    capabilities: {
      contextEffects: "measurable",
      measurement: "completed",
      answerSelectionProjection: "available",
    },
  };
}

function queryFixture(directions: readonly ["support" | "below_floor", "suppression" | "below_floor"] = ["support", "suppression"]): ContextInvestigationQuery {
  const deltas = directions[0] === "below_floor" ? [.03, -.02] : [.42, -.21];
  return {
    schemaVersion: "clozn.context-investigation-query.v1",
    status: "completed",
    selection: {
      unicodeRange: [3, 4],
      tokenRange: [1, 2],
      text: "🌕",
      tokenIds: [11],
      tokenPieces: ["🌕"],
    },
    effects: [
      {
        sourceId: "source-a",
        deltaNats: deltas[0],
        direction: directions[0],
        classification: directions[0] === "support" ? "measured_support" : "below_floor",
        measurementStatus: "measured",
        experimentId: "experiment_reader",
        armId: "arm_a",
        baselineObservationId: "observation_baseline",
        interventionObservationId: "observation_a",
      },
      {
        sourceId: "source b",
        deltaNats: deltas[1],
        direction: directions[1],
        classification: directions[1] === "suppression" ? "measured_suppression" : "below_floor",
        measurementStatus: "measured",
        experimentId: "experiment_reader",
        armId: "arm_b",
        baselineObservationId: "observation_baseline",
        interventionObservationId: "observation_b",
      },
    ],
    measurementFloorNats: .1,
  };
}

function notMeasuredReaderFixture(): ContextInvestigationReader {
  const reader = readerFixture();
  reader.status = "not_measured";
  reader.measurement.status = "not_measured";
  reader.measurement.completedSourceCount = 0;
  reader.measurement.notMeasuredSourceCount = 2;
  reader.measurement.baselineObservationId = null;
  reader.sources = reader.sources.map((source) => ({
    ...source,
    measurement: { status: "not_measured", experimentId: null, armId: null, observationId: null },
    summary: { maxAbsTokenDeltaNats: null, totalContinuationDeltaNats: null, aboveFloorLocusCount: 0 },
    loci: [],
  }));
  reader.loci = [];
  reader.capabilities.measurement = "not_measured";
  return reader;
}

function unmeasurableReaderFixture(): ContextInvestigationReader {
  const reader = notMeasuredReaderFixture();
  reader.status = "unavailable";
  reader.reason = "No canonical Context Receipt source is exactly removable.";
  reader.reasonCode = "context_investigation_no_exactly_removable_source";
  reader.measurement.status = "unavailable";
  reader.measurement.experimentId = null;
  reader.measurement.sourceUniverse = [];
  reader.measurement.notMeasuredSourceCount = 0;
  reader.sources = reader.sources.map((source) => ({
    ...source,
    measurement: { status: "unavailable", experimentId: null, armId: null, observationId: null },
  }));
  reader.capabilities.contextEffects = "unavailable";
  reader.capabilities.measurement = "unavailable";
  return reader;
}

function partialReaderFixture(): ContextInvestigationReader {
  const reader = readerFixture();
  const failed = {
    ...reader.sources[1],
    measurement: {
      status: "failed" as const,
      experimentId: "experiment_reader",
      armId: "arm_b",
      observationId: null,
      intervention: { kind: "delete_source" as const, target: { kind: "context_selection" as const, sourceIds: ["source b"] } },
    },
    summary: { maxAbsTokenDeltaNats: null, totalContinuationDeltaNats: null, aboveFloorLocusCount: 0 },
    loci: [],
  };
  const unavailable = {
    ...reader.sources[1],
    sourceId: "source-c",
    unicodeRange: [12, 18] as const,
    byteRange: [15, 21] as const,
    displayUnicodeRange: [12, 18] as const,
    measurement: {
      status: "unavailable" as const,
      experimentId: "experiment_reader",
      armId: "arm_c",
      observationId: null,
      intervention: { kind: "delete_source" as const, target: { kind: "context_selection" as const, sourceIds: ["source-c"] } },
    },
    summary: { maxAbsTokenDeltaNats: null, totalContinuationDeltaNats: null, aboveFloorLocusCount: 0 },
    loci: [],
  };
  reader.status = "partial";
  reader.measurement = {
    ...reader.measurement,
    status: "partial",
    sourceUniverse: ["source-a", "source b", "source-c"],
    completedSourceCount: 1,
    unavailableSourceCount: 1,
    failedSourceCount: 1,
    notMeasuredSourceCount: 0,
  };
  reader.context.blocks = [{
    ...reader.context.blocks[0],
    sourceIds: ["source-a", "source b", "source-c"],
  }];
  reader.sources = [reader.sources[0], failed, unavailable];
  reader.loci = [reader.loci[0]];
  reader.capabilities.measurement = "partial";
  return reader;
}

function partialQueryFixture(): ContextInvestigationQuery {
  const query = queryFixture();
  query.status = "partial";
  query.effects = [
    query.effects[0],
    {
      ...query.effects[1],
      deltaNats: null,
      direction: "failed",
      classification: null,
      measurementStatus: "failed",
      interventionObservationId: null,
    },
    {
      sourceId: "source-c",
      deltaNats: null,
      direction: "unavailable",
      classification: null,
      measurementStatus: "unavailable",
      experimentId: "experiment_reader",
      armId: "arm_c",
      baselineObservationId: "observation_baseline",
      interventionObservationId: null,
    },
  ];
  return query;
}

function nonMeasurementQueryFixture(
  status: "not_measured" | "stale" | "unavailable",
  reason?: string,
): ContextInvestigationQuery {
  return {
    schemaVersion: "clozn.context-investigation-query.v1",
    status,
    ...(reason ? { reason, reasonCode: `context_investigation_${status}` } : {}),
    selection: null,
    effects: [],
    measurementFloorNats: .1,
  };
}

function effectsJobFixture(
  state: ContextEffectsJob["state"],
): ContextEffectsJob {
  const terminal = state === "completed" || state === "failed" || state === "cancelled";
  return {
    schemaVersion: "clozn.analysis-job.v1",
    jobId: "infjob_ui",
    runId,
    kind: "context_investigation_effects",
    state,
    progress: {
      phase: terminal ? (state === "completed" ? "done" : state) : "scoring_recorded_continuation",
      completedUnits: terminal ? 1 : 0,
      totalUnits: 1,
      percent: terminal ? 100 : 0,
    },
    cancelRequested: state === "cancelling" || state === "cancelled",
    cancellable: !terminal,
    cached: false,
    ...(state === "completed" ? {
      result: {
        schemaVersion: "clozn.context-investigation-effects-job-result.v1",
        experimentId: "experiment_reader",
        sourceCount: 2,
        completedSourceCount: 1,
        unavailableSourceCount: 1,
        failedSourceCount: 0,
        notMeasuredSourceCount: 0,
        state: "completed",
      },
    } : {}),
  };
}

function leanContextJobFixture(): LeanContextJob {
  return {
    schemaVersion: "clozn.analysis-job.v1",
    jobId: "infjob_lean_ui",
    runId,
    kind: "context_investigation_lean_context",
    state: "completed",
    progress: { phase: "done", completedUnits: 1, totalUnits: 1, percent: 100 },
    cancelRequested: false,
    cancellable: false,
    cached: false,
    result: {
      schemaVersion: "clozn.lean-context-result.v1",
      runId,
      searchId: "lean_ui",
      status: "completed",
      claim: { kind: "best_found_within_budget", minimumProven: false },
      tolerance: {
        profile: "balanced.v1",
        meanPositiveSupportLossNatsPerToken: .05,
        maxTokenSupportLossNats: .7,
      },
      sourceIds: ["source-a", "source b"],
      original: {
        retainedSourceIds: ["source-a", "source b"],
        removedSourceIds: [],
        renderedPromptTokenCost: 120,
        toleranceStatus: "within_tolerance",
        loss: {
          answerTokenCount: 3,
          positiveLossTokenCount: 0,
          meanPositiveSupportLossNatsPerToken: 0,
          maxTokenSupportLossNats: 0,
        },
        observationId: "obs-original",
      },
      best: {
        retainedSourceIds: ["source-a"],
        removedSourceIds: ["source b"],
        renderedPromptTokenCost: 72,
        toleranceStatus: "within_tolerance",
        loss: {
          answerTokenCount: 3,
          positiveLossTokenCount: 1,
          meanPositiveSupportLossNatsPerToken: .02,
          maxTokenSupportLossNats: .12,
        },
        observationId: "obs-best",
      },
      reduction: {
        removedPromptTokens: 48,
        retainedPromptTokenFraction: .6,
        removedSourceCount: 1,
        retainedSourceCount: 1,
      },
      trials: [],
      budget: {
        maxNewExecutions: 16,
        usedNewExecutions: 4,
        reusedObservationCount: 2,
        exhausted: false,
        blockedByBudget: false,
      },
      stoppingReason: "search_policy_complete",
    },
  };
}

function overlappingUnicodeReaderFixture(): ContextInvestigationReader {
  const reader = readerFixture();
  const text = "A😀e\u0301BC";
  reader.context.blocks = [{
    ...reader.context.blocks[0],
    text,
    displayUnicodeRange: [0, 6],
  }];
  reader.answer = {
    status: "available",
    text,
    sha256: "b".repeat(64),
    tokens: ["A", "😀", "e", "\u0301", "B", "C"].map((piece, index) => ({
      index,
      tokenId: 20 + index,
      piece,
      unicodeRange: [index, index + 1] as const,
      logprob: -1,
    })),
    fidelity: "baseline_token_score_observation",
    observationId: "observation_baseline",
  };
  reader.sources = [
    {
      ...reader.sources[0],
      unicodeRange: [1, 4],
      byteRange: [1, 8],
      displayUnicodeRange: [1, 4],
      summary: { maxAbsTokenDeltaNats: .6, totalContinuationDeltaNats: .6, aboveFloorLocusCount: 1 },
      loci: ["locus_a"],
    },
    {
      ...reader.sources[1],
      unicodeRange: [2, 5],
      byteRange: [5, 9],
      displayUnicodeRange: [2, 5],
      summary: { maxAbsTokenDeltaNats: .6, totalContinuationDeltaNats: -.6, aboveFloorLocusCount: 1 },
      loci: ["locus_b"],
    },
  ];
  reader.loci = [
    {
      ...reader.loci[0],
      answerTokenRange: [1, 4],
      answerUnicodeRange: [1, 4],
      answerText: "😀e\u0301",
      deltaNats: .6,
      absoluteDeltaNats: .6,
      baselineSelectedLogp: -3,
      intervenedSelectedLogp: -3.6,
      perTokenDeltaNats: [.2, .2, .2],
    },
    {
      ...reader.loci[1],
      answerTokenRange: [2, 5],
      answerUnicodeRange: [2, 5],
      answerText: "e\u0301B",
      deltaNats: -.6,
      absoluteDeltaNats: .6,
      baselineSelectedLogp: -3,
      intervenedSelectedLogp: -2.4,
      perTokenDeltaNats: [-.2, -.2, -.2],
    },
  ];
  return reader;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.location.hash = "";
});

describe("Inspect surface", () => {
  it("stays quiet at rest, wakes measured registrations on focus, and locks Unicode coordinates from the keyboard", async () => {
    const query = vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const locus = await screen.findByRole("button", { name: "🌕; 2 measured relationships" });
    expect(locus).toHaveAttribute("tabindex", "0");
    expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0);

    fireEvent.focus(locus);
    await waitFor(() => expect(view.container.querySelectorAll(".map-node")).toHaveLength(2));
    expect(Array.from(view.container.querySelectorAll("[data-source-ids]")).some(
      (element) => element.getAttribute("data-source-ids") === '["source b"]',
    )).toBe(true);

    fireEvent.keyDown(locus, { key: "Enter" });
    await waitFor(() => expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=3&answerEnd=4"));
    expect(query).not.toHaveBeenCalled();
  });

  it("queries only a locked range, gives the exact relationship a visible focus, and exposes provenance", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const query = vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("2 above-floor relationships")).toBeInTheDocument();
    expect(query).toHaveBeenCalledWith(runId, { answerStart: 3, answerEnd: 4, floorNats: .1 }, expect.any(AbortSignal));
    expect(screen.queryByText("+0.42")).not.toBeInTheDocument();

    view.rerender(<InspectSurface route={{ surface: "inspect", runId, ...answerRange, sourceId: "source-a" }} />);
    expect(await screen.findByText("+0.42")).toBeInTheDocument();
    expect(screen.getByText("Support")).toBeInTheDocument();
    expect(screen.getByText("Delete source · recorded continuation · floor 0.10")).toBeInTheDocument();
    expect(view.container.querySelectorAll(".map-node")).toHaveLength(1);
    fireEvent.click(screen.getByText("Evidence"));
    expect(screen.getByText("experiment_reader")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(window.location.hash).toBe("#/inspect/run_alpha");
  });

  it("removes relationship paths when the locked aggregate is explicitly below floor", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture(["below_floor", "below_floor"]));
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("0 above-floor relationships")).toBeInTheDocument();
    expect(screen.getByText("0 support · 0 suppression · 2 below floor")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "🌕; 0 above-floor relationships" })).toBeInTheDocument();
    await waitFor(() => expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0));
  });

  it("keeps not-measured prose readable without manufacturing answer loci", async () => {
    const reader = notMeasuredReaderFixture();
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    const query = vi.spyOn(contextInvestigationApi, "query");
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    expect(await screen.findByText("Not measured")).toBeInTheDocument();
    expect(view.container.querySelector(".answer-prose p")).toHaveTextContent("Hi 🌕!");
    expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0);
    expect(screen.queryByRole("button", { name: /measured relationship/ })).not.toBeInTheDocument();
    expect(query).not.toHaveBeenCalled();
  });

  it("never starts measurement on navigation and refreshes the reader only after explicit completion", async () => {
    const reader = vi.spyOn(contextInvestigationApi, "reader")
      .mockResolvedValueOnce(notMeasuredReaderFixture())
      .mockResolvedValueOnce(partialReaderFixture());
    const start = vi.spyOn(contextInvestigationApi, "startEffects")
      .mockResolvedValue(effectsJobFixture("queued"));
    const poll = vi.spyOn(contextInvestigationApi, "effectsJob")
      .mockResolvedValue(effectsJobFixture("completed"));
    const initialHash = "#/inspect/run_alpha";
    window.location.hash = initialHash;
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const button = await screen.findByRole("button", { name: "Measure context" });
    expect(start).not.toHaveBeenCalled();
    expect(poll).not.toHaveBeenCalled();
    expect(screen.getByText("Each source is removed once and scored against the recorded answer.")).toBeInTheDocument();

    fireEvent.click(button);
    await waitFor(() => expect(start).toHaveBeenCalledWith(runId, {
      sourceIds: ["source-a", "source b"],
    }, expect.any(AbortSignal)));
    await waitFor(() => expect(poll).toHaveBeenCalledWith(runId, "infjob_ui", expect.any(AbortSignal)));
    await waitFor(() => expect(reader).toHaveBeenCalledTimes(2));
    expect(window.location.hash).toBe(initialHash);
  });

  it("cancels an active measurement without changing the Inspect coordinate", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(notMeasuredReaderFixture());
    vi.spyOn(contextInvestigationApi, "query")
      .mockResolvedValue(nonMeasurementQueryFixture("not_measured"));
    vi.spyOn(contextInvestigationApi, "startEffects")
      .mockResolvedValue(effectsJobFixture("running"));
    vi.spyOn(contextInvestigationApi, "effectsJob")
      .mockResolvedValue(effectsJobFixture("running"));
    const cancel = vi.spyOn(contextInvestigationApi, "cancelEffects")
      .mockResolvedValue(effectsJobFixture("cancelled"));
    window.location.hash = "#/inspect/run_alpha?answerStart=3&answerEnd=4";
    render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Measure context" }));
    const cancelButton = await screen.findByRole("button", { name: "Cancel" });
    fireEvent.click(cancelButton);

    await waitFor(() => expect(cancel).toHaveBeenCalledWith(runId, "infjob_ui", expect.any(AbortSignal)));
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();
    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=3&answerEnd=4");
  });

  it("retries status polling without dispatching a second measurement job", async () => {
    const reader = vi.spyOn(contextInvestigationApi, "reader")
      .mockResolvedValueOnce(notMeasuredReaderFixture())
      .mockResolvedValueOnce(partialReaderFixture());
    const start = vi.spyOn(contextInvestigationApi, "startEffects")
      .mockResolvedValue(effectsJobFixture("running"));
    const poll = vi.spyOn(contextInvestigationApi, "effectsJob")
      .mockRejectedValueOnce(new Error("temporary poll failure"))
      .mockResolvedValueOnce(effectsJobFixture("completed"));
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Measure context" }));
    expect(await screen.findByText("temporary poll failure")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry status" }));

    await waitFor(() => expect(poll).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(reader).toHaveBeenCalledTimes(2));
    expect(start).toHaveBeenCalledTimes(1);
  });

  it("keeps a locked not-measured query distinct from a measured zero", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(notMeasuredReaderFixture());
    const query = vi.spyOn(contextInvestigationApi, "query")
      .mockResolvedValue(nonMeasurementQueryFixture("not_measured"));
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    await waitFor(() => expect(view.container.querySelector(".query-status")).toHaveTextContent("Not measured"));
    expect(screen.getByText("0 above-floor relationships")).toBeInTheDocument();
    expect(view.container.querySelectorAll(".map-node")).toHaveLength(0);
    expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0);
    expect(screen.queryByText(/Δ nats/)).not.toBeInTheDocument();
    expect(query).toHaveBeenCalledWith(runId, { ...answerRange, floorNats: .1 }, expect.any(AbortSignal));
  });

  it("keeps readable prose but does not query when the Run has no legal measurement arm", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(unmeasurableReaderFixture());
    const query = vi.spyOn(contextInvestigationApi, "query");
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("No canonical Context Receipt source is exactly removable.")).toBeInTheDocument();
    expect(view.container.querySelector(".answer-prose p")).toHaveTextContent("Hi 🌕!");
    expect(screen.getByText("Evidence unavailable.")).toBeInTheDocument();
    expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0);
    expect(query).not.toHaveBeenCalled();
  });

  it("shows partial source coverage without turning failed or unavailable effects into relationships", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(partialReaderFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(partialQueryFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("1 above-floor relationship")).toBeInTheDocument();
    expect(view.container.querySelector(".reader-status")).toHaveTextContent("Partial");
    await waitFor(() => expect(view.container.querySelector(".query-status")).toHaveTextContent("Partial"));
    expect(screen.getByText("1 support · 0 suppression · 0 below floor · 1 unavailable · 1 failed")).toBeInTheDocument();
    expect(screen.getByRole("img", {
      name: "Context coverage: 1 measured source, 0 not measured sources, 1 unavailable source, 1 failed source. 1 measured answer range.",
    })).toBeInTheDocument();

    const failed = Array.from(view.container.querySelectorAll<HTMLElement>(".context-segment"))
      .find((element) => element.dataset.sourceIds === JSON.stringify(["source b"]));
    const unavailable = Array.from(view.container.querySelectorAll<HTMLElement>(".context-segment"))
      .find((element) => element.dataset.sourceIds === JSON.stringify(["source-c"]));
    expect(failed).toHaveClass("has-failed");
    expect(unavailable).toHaveClass("has-unavailable");
    expect(failed).not.toHaveAttribute("role", "button");
    expect(unavailable).not.toHaveAttribute("role", "button");
    await waitFor(() => expect(view.container.querySelectorAll(".map-node")).toHaveLength(1));
  });

  it.each([
    { status: "stale" as const, label: "Stale", reason: "Persisted evidence no longer binds this Run." },
    { status: "unavailable" as const, label: "Unavailable", reason: "Recorded score vectors are unavailable." },
  ])("renders a typed $status query as an evidence state, not an empty measurement", async ({ status, label, reason }) => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(nonMeasurementQueryFixture(status, reason));
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByRole("alert", { name: "" })).toHaveTextContent(`${label} · ${reason}`);
    expect(screen.getByText("0 above-floor relationships")).toBeInTheDocument();
    await waitFor(() => expect(view.container.querySelectorAll(".map-node")).toHaveLength(0));
    expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0);
  });

  it("renders stale reader prose for orientation while withholding unavailable answer evidence", async () => {
    const reader = readerFixture();
    reader.status = "stale";
    reader.reason = "Persisted evidence no longer reconstructs this Run.";
    reader.reasonCode = "context_investigation_stale";
    reader.measurement.status = "stale";
    reader.measurement.completedSourceCount = 0;
    reader.measurement.notMeasuredSourceCount = 2;
    reader.measurement.baselineObservationId = null;
    reader.answer = { status: "unavailable" };
    reader.sources = [];
    reader.loci = [];
    reader.capabilities.measurement = "stale";
    reader.capabilities.answerSelectionProjection = "unavailable";
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    const query = vi.spyOn(contextInvestigationApi, "query");
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    expect(await screen.findByText("Persisted evidence no longer reconstructs this Run.")).toHaveAttribute("role", "alert");
    expect(screen.getByText("A😀B context signal.")).toBeInTheDocument();
    expect(screen.getByText("Answer unavailable.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /measured relationship/ })).not.toBeInTheDocument();
    expect(query).not.toHaveBeenCalled();
  });

  it("preserves overlapping Unicode source and answer geometry without losing prose", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(overlappingUnicodeReaderFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const overlap = await screen.findByRole("button", { name: "é; 2 measured relationships" });
    expect(view.container.querySelector(".context-block")?.textContent).toBe("A😀éBC");
    expect(view.container.querySelector(".answer-prose p")?.textContent).toBe("A😀éBC");
    const contextOverlap = Array.from(view.container.querySelectorAll<HTMLElement>(".context-segment"))
      .find((element) => element.textContent === "é");
    expect(contextOverlap?.dataset.sourceIds).toBe(JSON.stringify(["source b", "source-a"]));

    fireEvent.focus(overlap);
    await waitFor(() => expect(view.container.querySelectorAll(".map-node")).toHaveLength(2));
  });

  it("distinguishes malformed evidence from an unavailable Inspect read", async () => {
    const reader = vi.spyOn(contextInvestigationApi, "reader");
    reader.mockRejectedValueOnce(new ContractError("/reader", "invalid coordinates"));
    render(<InspectSurface route={{ surface: "inspect", runId }} />);
    expect(await screen.findByText("Malformed evidence contract.")).toBeInTheDocument();

    cleanup();
    reader.mockRejectedValueOnce(new Error("HTTP 409"));
    render(<InspectSurface route={{ surface: "inspect", runId: "run_blocked" }} />);
    expect(await screen.findByText("Inspect unavailable.")).toBeInTheDocument();
  });

  it("never paints a previous Run or query result under a new URL coordinate", async () => {
    let keepReaderPending: ((value: ContextInvestigationReader) => void) | undefined;
    const reader = vi.spyOn(contextInvestigationApi, "reader")
      .mockResolvedValueOnce(readerFixture())
      .mockImplementationOnce(() => new Promise((resolve) => { keepReaderPending = resolve; }));
    const query = vi.spyOn(contextInvestigationApi, "query")
      .mockResolvedValueOnce(queryFixture())
      .mockImplementationOnce(() => new Promise(() => undefined));
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange, sourceId: "source-a" }} />);
    expect(await screen.findByText("+0.42")).toBeInTheDocument();

    view.rerender(<InspectSurface route={{ surface: "inspect", runId, answerStart: 0, answerEnd: 3 }} />);
    expect(screen.getByText("Loading evidence.")).toBeInTheDocument();
    expect(screen.queryByText("+0.42")).not.toBeInTheDocument();

    view.rerender(<InspectSurface route={{ surface: "inspect", runId: "run_beta" }} />);
    expect(screen.getByText("Loading Inspect.")).toBeInTheDocument();
    expect(screen.queryByText("Hi 🌕!")).not.toBeInTheDocument();
    expect(reader).toHaveBeenLastCalledWith("run_beta", expect.any(AbortSignal));
    expect(query).toHaveBeenCalledTimes(2);
    expect(keepReaderPending).toBeTypeOf("function");
  });

  it("uses one roving answer stop and preserves exact whole-segment granularity", async () => {
    const reader = readerFixture();
    const contextLength = [...reader.context.blocks[0].text].length;
    reader.sources = reader.sources.map((source, index) => index === 0 ? {
      ...source,
      unicodeRange: [0, contextLength],
      displayUnicodeRange: [0, contextLength],
      granularity: "whole_segment",
    } : source);
    reader.loci = reader.loci.map((locus, index) => index === 1 ? {
      ...locus,
      answerTokenRange: [2, 3],
      answerUnicodeRange: [4, 5],
      answerText: "!",
    } : locus);
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const moon = await screen.findByRole("button", { name: "🌕; 1 measured relationship" });
    const punctuation = screen.getByRole("button", { name: "!; 1 measured relationship" });
    expect(moon).toHaveAttribute("tabindex", "0");
    expect(punctuation).toHaveAttribute("tabindex", "-1");
    fireEvent.keyDown(moon, { key: "ArrowRight" });
    expect(document.activeElement).toBe(punctuation);
    await waitFor(() => expect(punctuation).toHaveAttribute("tabindex", "0"));

    const block = document.querySelector(".context-block");
    expect(block).toHaveClass("has-whole-measurement");
    expect(block).toHaveAttribute("data-source-ids", '["source-a"]');
  });

  it("turns an expanded context pane into a heatmap and restores the linked layout on collapse", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    await screen.findByText("Measured");
    fireEvent.click(screen.getAllByRole("button", { name: "Expand heatmap" })[0]);

    expect(view.container.querySelector(".inspect-reader")).toHaveClass("is-expanded-context");
    const heated = view.container.querySelector<HTMLElement>('.context-segment[data-source-ids="[\\"source-a\\"]"]');
    expect(heated).toHaveAttribute("data-heat", "true");
    expect(heated?.style.getPropertyValue("--heat-alpha")).not.toBe("");
    fireEvent.mouseEnter(heated!);
    expect(await screen.findByText("Linked text")).toBeInTheDocument();
    expect(screen.getByText("+0.42 Δ")).toBeInTheDocument();

    fireEvent.click(heated!);
    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=3&answerEnd=4&sourceId=source-a");
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    expect(view.container.querySelector(".inspect-reader")).not.toHaveClass("is-expanded");
    expect(view.container.querySelector('[data-heat="true"]')).not.toBeInTheDocument();
  });

  it("ranks context and answer influence and opens a ranking in the linked view", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Efficiency" }));
    expect(screen.getByRole("heading", { name: "Influence overview" })).toBeInTheDocument();
    expect(screen.getByText("This measures influence, not correctness.", { exact: false })).toBeInTheDocument();
    const sourceText = screen.getByText("A😀B", { selector: ".ranking-copy strong" });
    fireEvent.click(sourceText.closest("button")!);

    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=3&answerEnd=4&sourceId=source-a");
    expect(screen.getByRole("button", { name: "Linked view" })).toHaveAttribute("aria-current", "page");
  });

  it("runs Lean Context and labels the result as best-found rather than minimal", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const start = vi.spyOn(contextInvestigationApi, "startLeanContext")
      .mockResolvedValue(leanContextJobFixture());
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Lean Context" }));
    fireEvent.click(screen.getByRole("button", { name: "Find lean context" }));

    await waitFor(() => expect(start).toHaveBeenCalledWith(runId, expect.any(AbortSignal)));
    const promptTokens = await screen.findByText("Prompt tokens");
    expect(promptTokens.closest("article")).toHaveTextContent("120 → 72");
    expect(screen.getByText("60%")).toBeInTheDocument();
    expect(screen.getByText("Best found within this search budget. It is not a proof of the mathematically smallest context.")).toBeInTheDocument();
    expect(screen.getByText("context", { selector: ".lean-removed li" })).toBeInTheDocument();
  });

});
