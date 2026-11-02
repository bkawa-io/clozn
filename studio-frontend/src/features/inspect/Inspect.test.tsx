import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ContractError } from "../../data/contracts";
import type {
  ContextInvestigationQuery,
  ContextInvestigationReader,
} from "../../data/contextInvestigation";
import { contextInvestigationApi } from "../../data/contextInvestigationClient";
import type { ContextEffectsJob } from "../../data/contextInvestigationJobs";
import type { ContextInteractionsJob } from "../../data/contextInteractions";
import type { LeanContextJob } from "../../data/leanContext";
import { InspectSurface, LeanContextPanel } from "./Inspect";

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
        executedScoreCount: 1,
        reusedScoreCount: 1,
        elapsedSeconds: 2.5,
        scoringMode: "fast",
        evidenceQuality: "approximate",
        executionStrategy: "native_prefix_trie",
        numericalRegime: "native_prefix_trie_target_logprob_v1",
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
      generationComparison: {
        status: "completed",
        decodeMode: "greedy",
        maxNewTokens: 40,
        fullContext: { text: "Hi 🌕!", matchesRecorded: true },
        leanContext: { text: "Hi moon!", matchesRecorded: false },
        exactMatch: false,
        normalizedMatch: false,
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
      elapsedSeconds: 5,
      stoppingReason: "search_policy_complete",
    },
  };
}

function interactionsJobFixture(
  state: "queued" | "running" | "completed",
): ContextInteractionsJob {
  const result = state === "queued" ? undefined : {
    schemaVersion: "clozn.context-interactions-result.v1" as const,
    runId,
    status: state === "completed" ? "completed" as const : "running" as const,
    stopReason: state === "completed" ? "execution_budget" : null,
    seed: 0,
    claim: { kind: "exploratory_sampled_marginals" as const, redundancyProven: false as const },
    sourceIds: ["source-a", "source b"],
    budget: {
      maxNewExecutions: 16,
      usedNewExecutions: state === "completed" ? 16 : 4,
      reusedObservationCount: 2,
      completedPairs: state === "completed" ? 6 : 1,
    },
    sourceEstimates: [{
      sourceId: "source-a",
      sampleCount: 1,
      uniquePairCount: 1,
      meanMarginalNats: 1.5,
      sampleStddevNats: null,
      leaveOneOutNats: .42,
      meanMinusLeaveOneOutNats: 1.08,
    }, {
      sourceId: "source b",
      sampleCount: 0,
      uniquePairCount: 0,
      meanMarginalNats: null,
      sampleStddevNats: null,
      leaveOneOutNats: -.21,
      meanMinusLeaveOneOutNats: null,
    }],
    elapsedSeconds: state === "completed" ? 8 : null,
    best: state === "completed" ? {
      retainedSourceIds: ["source-a"],
      renderedPromptTokens: 72,
      loss: null,
    } : null,
  };
  return {
    schemaVersion: "clozn.analysis-job.v1",
    jobId: "infjob_interactions_ui",
    runId,
    kind: "context_investigation_interactions",
    state,
    progress: {
      phase: state === "completed" ? "done" : "sampling_context_interactions",
      completedUnits: state === "queued" ? 0 : state === "completed" ? 16 : 4,
      totalUnits: 16,
      percent: state === "queued" ? 0 : state === "completed" ? 100 : 25,
    },
    cancelRequested: false,
    cancellable: state !== "completed",
    cached: false,
    result,
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

    const locus = await screen.findByRole("button", { name: "Hi 🌕!; 2 measured relationships" });
    expect(locus).toHaveAttribute("tabindex", "0");
    expect(view.container.querySelectorAll(".relationship-path")).toHaveLength(0);

    fireEvent.focus(locus);
    await waitFor(() => expect(view.container.querySelectorAll(".map-node")).toHaveLength(2));
    expect(Array.from(view.container.querySelectorAll("[data-source-ids]")).some(
      (element) => element.getAttribute("data-source-ids") === '["source b"]',
    )).toBe(true);

    fireEvent.keyDown(locus, { key: "Enter" });
    await waitFor(() => expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=0&answerEnd=5"));
    expect(query).not.toHaveBeenCalled();
  });

  it("queries only a locked range, gives the exact relationship a visible focus, and exposes provenance", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const query = vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("2 above-floor relationships")).toBeInTheDocument();
    expect(query).toHaveBeenCalledWith(runId, {
      answerStart: 3,
      answerEnd: 4,
      floorNats: .1,
      sourceIds: ["source-a", "source b"],
      scoringMode: "fast",
    }, expect.any(AbortSignal));
    expect(screen.queryByText("+0.42")).not.toBeInTheDocument();

    view.rerender(<InspectSurface route={{ surface: "inspect", runId, ...answerRange, sourceId: "source-a" }} />);
    expect(await screen.findByText("+0.42")).toBeInTheDocument();
    expect(screen.getAllByText("Support").some((element) => element.classList.contains("evidence-state"))).toBe(true);
    expect(screen.getByText("Delete source · recorded continuation · floor 0.10")).toBeInTheDocument();
    expect(view.container.querySelectorAll(".map-node")).toHaveLength(1);
    fireEvent.click(screen.getByText("Evidence"));
    expect(screen.getByText("experiment_reader")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(window.location.hash).toBe("#/inspect/run_alpha");
  });

  it("draws at most three supporting and three suppressing sources for a pinned answer", async () => {
    const reader = readerFixture();
    const extras = [
      { id: "source-c", direction: "support" as const, delta: .3 },
      { id: "source-d", direction: "suppression" as const, delta: -.19 },
      { id: "source-e", direction: "support" as const, delta: .2 },
      { id: "source-f", direction: "suppression" as const, delta: -.18 },
      { id: "source-g", direction: "support" as const, delta: .1 },
      { id: "source-h", direction: "suppression" as const, delta: -.09 },
    ];
    reader.sources = [...reader.sources, ...extras.map(({ id, delta }, index) => ({
      ...reader.sources[index % 2],
      sourceId: id,
      measurement: { ...reader.sources[index % 2].measurement, armId: `arm_${id}`, observationId: `observation_${id}` },
      summary: { maxAbsTokenDeltaNats: Math.abs(delta), totalContinuationDeltaNats: delta, aboveFloorLocusCount: 0 },
      loci: [],
    }))];
    reader.measurement.sourceUniverse = reader.sources.map((source) => source.sourceId);
    reader.measurement.completedSourceCount = reader.sources.length;
    const queryResult = queryFixture();
    queryResult.effects = [
      ...queryResult.effects,
      ...extras.map(({ id, direction, delta }) => ({
        ...queryResult.effects[direction === "support" ? 0 : 1],
        sourceId: id,
        deltaNats: delta,
        armId: `arm_${id}`,
        interventionObservationId: `observation_${id}`,
      })),
    ];
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryResult);
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("8 above-floor relationships")).toBeInTheDocument();
    await waitFor(() => expect(view.container.querySelectorAll(".map-node")).toHaveLength(6));
    expect(view.container.querySelector(".context-pane")).toHaveClass("has-linked-selection");
    expect(view.container.querySelector('.context-prose .is-related[data-link-direction="support"]')).toBeInTheDocument();
    expect(view.container.querySelector('.context-prose .is-related[data-link-direction="suppression"]')).toBeInTheDocument();
  });

  it("removes relationship paths when the locked aggregate is explicitly below floor", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture(["below_floor", "below_floor"]));
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    expect(await screen.findByText("0 above-floor relationships")).toBeInTheDocument();
    expect(screen.getByText("0 support · 0 suppression · 2 below floor")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hi 🌕!; 0 above-floor relationships" })).toBeInTheDocument();
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
    expect(screen.getByText("Up to 2 source scores, plus a reusable baseline.")).toBeInTheDocument();

    fireEvent.click(button);
    await waitFor(() => expect(start).toHaveBeenCalledWith(runId, {
      granularity: "adaptive",
      maxArms: 2,
      scoringMode: "fast",
    }, expect.any(AbortSignal)));
    await waitFor(() => expect(poll).toHaveBeenCalledWith(runId, "infjob_ui", expect.any(AbortSignal)));
    await waitFor(() => expect(reader).toHaveBeenCalledTimes(2));
    expect(window.location.hash).toBe(initialHash);
  });

  it("shows the arm estimate and keeps granularity configuration in the route", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(notMeasuredReaderFixture());
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    expect(await screen.findByText("2 context spans")).toBeInTheDocument();
    expect(screen.getByText("More arms preserve smaller Markdown sections. Fast chooses native batching for prompt-heavy work and exact prefix-cached scoring when answer decoding would dominate.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Granularity"), { target: { value: "message" } });

    expect(window.location.hash).toBe(
      "#/inspect/run_alpha?granularity=message&armBudget=1",
    );
  });

  it("runs a custom experiment for only the selected recorded spans", async () => {
    const reader = vi.spyOn(contextInvestigationApi, "reader")
      .mockResolvedValue(notMeasuredReaderFixture());
    const start = vi.spyOn(contextInvestigationApi, "startEffects")
      .mockResolvedValue(effectsJobFixture("queued"));
    vi.spyOn(contextInvestigationApi, "effectsJob")
      .mockImplementation(() => new Promise(() => undefined));
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.change(await screen.findByLabelText("Granularity"), {
      target: { value: "custom" },
    });
    expect(screen.getByText("No sources selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Measure context" })).toBeDisabled();

    const choices = screen.getAllByRole("checkbox");
    expect(choices).toHaveLength(2);
    fireEvent.click(choices[0]);
    await waitFor(() => expect(reader).toHaveBeenLastCalledWith(
      runId,
      expect.any(AbortSignal),
      { sourceIds: ["source-a"], scoringMode: "fast" },
    ));
    expect(screen.getByText("1 context span")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Measure context" }));
    await waitFor(() => expect(start).toHaveBeenCalledWith(
      runId,
      { sourceIds: ["source-a"], scoringMode: "fast" },
      expect.any(AbortSignal),
    ));
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
    expect(screen.getByText("Working…")).toBeInTheDocument();
    expect(screen.getByText("Links and heatmaps update as each scoring batch finishes. Foreground chat takes priority between batches.")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).not.toHaveAttribute("value");
    fireEvent.click(cancelButton);

    await waitFor(() => expect(cancel).toHaveBeenCalledWith(runId, "infjob_ui", expect.any(AbortSignal)));
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();
    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=3&answerEnd=4");
  });

  it("updates the reader while a measurement job is still running", async () => {
    let resolvePartial!: (reader: ContextInvestigationReader) => void;
    const partialReader = new Promise<ContextInvestigationReader>((resolve) => {
      resolvePartial = resolve;
    });
    const reader = vi.spyOn(contextInvestigationApi, "reader")
      .mockResolvedValueOnce(notMeasuredReaderFixture())
      .mockReturnValueOnce(partialReader);
    vi.spyOn(contextInvestigationApi, "startEffects")
      .mockResolvedValue(effectsJobFixture("queued"));
    const running = effectsJobFixture("running");
    running.progress = {
      phase: "scoring_recorded_continuation",
      completedUnits: 1,
      totalUnits: 2,
      percent: 50,
    };
    const poll = vi.spyOn(contextInvestigationApi, "effectsJob")
      .mockResolvedValue(running);
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Measure context" }));
    await waitFor(() => expect(poll).toHaveBeenCalled());
    resolvePartial(partialReaderFixture());

    expect(await screen.findByText("Partial")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(reader).toHaveBeenCalledTimes(2);
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
    expect(query).toHaveBeenCalledWith(runId, {
      ...answerRange,
      floorNats: .1,
      sourceIds: ["source-a", "source b"],
      scoringMode: "fast",
    }, expect.any(AbortSignal));
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
    expect(screen.getByRole("group", {
      name: "Context coverage: 1 measured source, 0 not measured sources, 1 unavailable source, 1 failed source. 1 answer phrase. Answer lens: tension.",
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
    expect(screen.getByText("No recorded response is available for this turn.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /measured relationship/ })).not.toBeInTheDocument();
    expect(query).not.toHaveBeenCalled();
  });

  it("disables the unified Lean Context page when no recorded answer is available", async () => {
    const reader = readerFixture();
    reader.answer = { status: "unavailable" };
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    render(<InspectSurface route={{ surface: "inspect", runId, view: "lean-context" }} />);

    expect(await screen.findByText("Lean Context cannot run for this turn.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Find lean context" })).toBeDisabled();
  });

  it("preserves overlapping Unicode source and answer geometry without losing prose", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(overlappingUnicodeReaderFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const overlap = await screen.findByRole("button", { name: "A😀éBC; 2 measured relationships" });
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
    expect(reader).toHaveBeenLastCalledWith("run_beta", expect.any(AbortSignal), { scoringMode: "fast" });
    expect(query).toHaveBeenCalledTimes(2);
    expect(keepReaderPending).toBeTypeOf("function");
  });

  it("keeps readable phrase spans in the answer heatmap and preserves whole-segment context granularity", async () => {
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

    const phrase = await screen.findByRole("button", { name: "Hi 🌕!; 2 measured relationships" });
    expect(phrase).toHaveAttribute("tabindex", "0");
    expect(screen.queryByRole("button", { name: "🌕; 1 measured relationship" })).not.toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Expand heatmap" })[1]);
    const expandedPhrase = screen.getByRole("button", { name: "Hi 🌕!; 2 measured relationships" });
    expect(expandedPhrase).toHaveAttribute("tabindex", "0");
    expect(screen.queryByRole("button", { name: "🌕; 1 measured relationship" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "!; 1 measured relationship" })).not.toBeInTheDocument();

    const block = document.querySelector(".context-block");
    expect(block).toHaveClass("has-whole-measurement");
    expect(block).toHaveAttribute("data-source-ids", '["source-a"]');
  });

  it("keeps the context heatmap interactive before, during, and after expansion", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    await screen.findByRole("button", { name: "Tension" });
    const heated = view.container.querySelector<HTMLElement>('.context-segment[data-source-ids="[\\"source-a\\"]"]');
    const cooler = view.container.querySelector<HTMLElement>('.context-segment[data-source-ids="[\\"source b\\"]"]');
    expect(heated).toHaveAttribute("data-heat", "true");
    expect(heated?.style.getPropertyValue("--heat-band")).toBe("5");
    expect(cooler).toHaveAttribute("data-heat", "true");
    expect(cooler?.style.getPropertyValue("--heat-band")).toBe("2");
    expect(heated).toHaveAttribute("role", "button");
    fireEvent.mouseEnter(heated!);
    expect(await screen.findByText("Corresponding text")).toBeInTheDocument();
    expect(screen.getByText("+0.42 Δ")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Expand heatmap" })[0]);
    expect(view.container.querySelector(".inspect-reader")).toHaveClass("is-expanded-context");
    expect(screen.getByRole("dialog", { name: "Expanded context heatmap" })).toContainElement(view.container.querySelector(".heatmap-peek"));
    fireEvent.click(heated!);
    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=0&answerEnd=5&sourceId=source-a");
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    expect(view.container.querySelector(".inspect-reader")).not.toHaveClass("is-expanded");
    expect(heated).toHaveAttribute("data-heat", "true");
    fireEvent.click(screen.getAllByRole("button", { name: "Expand heatmap" })[1]);
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Expanded answer heatmap" }), { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(document.body.style.overflow).toBe("");
  });

  it("encodes support and suppression separately in the always-visible answer heatmap", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    await screen.findByRole("button", { name: "Tension" });
    const phrase = screen.getByRole("button", { name: "Hi 🌕!; 2 measured relationships" });
    expect(screen.getByRole("button", { name: "Tension" })).toHaveAttribute("aria-pressed", "true");
    expect(phrase).toHaveAttribute("data-lens", "tension");
    expect(phrase).toHaveAttribute("data-direction", "mixed");
    expect(phrase.style.getPropertyValue("--heat-support-fill")).not.toBe("");
    expect(phrase.style.getPropertyValue("--heat-suppression-fill")).not.toBe("");
    expect(view.container.querySelector('.overview-answer .overview-range[data-lens="tension"]')).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Suppression" }));
    expect(phrase).toHaveAttribute("data-lens", "suppression");
    expect(phrase).toHaveAttribute("data-direction", "suppression");
    expect(screen.getByText(/negative pull, not model uncertainty/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Overall" }));
    expect(phrase).toHaveAttribute("data-lens", "influence");
    expect(phrase).toHaveAttribute("data-direction", "mixed");
    expect(Number.parseFloat(phrase.style.getPropertyValue("--heat-support-share"))).toBeCloseTo(66.7, 1);

    fireEvent.mouseEnter(phrase);
    const peek = await waitFor(() => {
      const element = view.container.querySelector<HTMLElement>(".heatmap-peek");
      expect(element).toHaveTextContent("Corresponding text");
      return element!;
    });
    expect(peek).toHaveTextContent("+0.42 support");
    expect(peek).toHaveTextContent("−0.21 suppression");
    expect(peek).toHaveTextContent("2 linked sources");
  });

  it("bubbles the strongest supporting and suppressing context to the top after selecting an answer span", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    const panel = await screen.findByRole("complementary", { name: "Context competing around this answer span" });
    expect(within(panel).getByRole("heading", { name: "Top positive influence" })).toBeInTheDocument();
    expect(within(panel).getByRole("heading", { name: "Top negative influence" })).toBeInTheDocument();
    expect(within(panel).getByText("+0.42 Δ")).toBeInTheDocument();
    expect(within(panel).getByText("−0.21 Δ")).toBeInTheDocument();

    const summaryLink = within(panel).getByRole("button", { name: /Focus support span/ });
    expect(summaryLink).not.toHaveAttribute("data-source-ids");
    expect(document.querySelector('.context-prose [data-source-ids="[\\"source-a\\"]"]')).toBeInTheDocument();

    fireEvent.click(summaryLink);
    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=3&answerEnd=4&sourceId=source-a");

    fireEvent.click(screen.getByRole("button", { name: "Suppression" }));
    const suppressionPanel = screen.getByRole("complementary", { name: "Context suppressing this answer span" });
    expect(within(suppressionPanel).getAllByRole("heading", { level: 4 })[0])
      .toHaveTextContent("Top negative influence");
  });

  it("bubbles the answer spans most affected by a selected context span", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange, sourceId: "source-a" }} />);

    const panel = await screen.findByRole("complementary", { name: "Most influenced answer spans" });
    expect(within(panel).getByText("Hi 🌕!")).toBeInTheDocument();
    expect(within(panel).getByText("+0.42 Δ")).toBeInTheDocument();
    expect(within(panel).getByText("No above-floor suppression was recorded.")).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: /Focus support span/ })).not.toHaveAttribute("data-answer-start");
    expect(document.querySelector('.answer-prose [data-answer-start="0"]')).toBeInTheDocument();
  });

  it("keeps up to three supporting and three suppressing answer passages active when context is pinned", async () => {
    const reader = readerFixture();
    const first = "First linked answer sentence carries the science evidence. ";
    const second = "Second linked answer sentence carries the risk mitigation.";
    const third = " Third linked answer sentence is weaker supporting evidence.";
    const fourth = " Fourth linked answer sentence is still useful supporting evidence.";
    const fifth = " Fifth linked answer sentence is deliberately omitted.";
    if (reader.answer.status !== "available") throw new Error("fixture answer must be available");
    reader.answer = {
      ...reader.answer,
      text: first + second + third + fourth + fifth,
      tokens: [
        { index: 0, tokenId: 20, piece: first, unicodeRange: [0, [...first].length], logprob: -.2 },
        { index: 1, tokenId: 21, piece: second, unicodeRange: [[...first].length, [...first, ...second].length], logprob: -.3 },
        { index: 2, tokenId: 22, piece: third, unicodeRange: [[...first, ...second].length, [...first, ...second, ...third].length], logprob: -.4 },
        { index: 3, tokenId: 23, piece: fourth, unicodeRange: [[...first, ...second, ...third].length, [...first, ...second, ...third, ...fourth].length], logprob: -.5 },
        { index: 4, tokenId: 24, piece: fifth, unicodeRange: [[...first, ...second, ...third, ...fourth].length, [...first, ...second, ...third, ...fourth, ...fifth].length], logprob: -.6 },
      ],
    };
    reader.sources = [{
      ...reader.sources[0],
      summary: { maxAbsTokenDeltaNats: .42, totalContinuationDeltaNats: .57, aboveFloorLocusCount: 5 },
      loci: ["locus_a", "locus_a_second", "locus_a_third", "locus_a_fourth", "locus_a_fifth"],
    }];
    reader.loci = [{
      ...reader.loci[0],
      answerTokenRange: [0, 1],
      answerUnicodeRange: [0, [...first].length],
      answerText: first,
      perTokenDeltaNats: [.42],
    }, {
      ...reader.loci[0],
      locusId: "locus_a_second",
      answerTokenRange: [1, 2],
      answerUnicodeRange: [[...first].length, [...first, ...second].length],
      answerText: second,
      direction: "suppression",
      deltaNats: -.3,
      absoluteDeltaNats: .3,
      baselineSelectedLogp: -.3,
      intervenedSelectedLogp: 0,
      perTokenDeltaNats: [-.3],
    }, {
      ...reader.loci[0],
      locusId: "locus_a_third",
      answerTokenRange: [2, 3],
      answerUnicodeRange: [[...first, ...second].length, [...first, ...second, ...third].length],
      answerText: third,
      deltaNats: .2,
      absoluteDeltaNats: .2,
      baselineSelectedLogp: -.4,
      intervenedSelectedLogp: -.6,
      perTokenDeltaNats: [.2],
    }, {
      ...reader.loci[0],
      locusId: "locus_a_fourth",
      answerTokenRange: [3, 4],
      answerUnicodeRange: [[...first, ...second, ...third].length, [...first, ...second, ...third, ...fourth].length],
      answerText: fourth,
      deltaNats: .15,
      absoluteDeltaNats: .15,
      baselineSelectedLogp: -.5,
      intervenedSelectedLogp: -.65,
      perTokenDeltaNats: [.15],
    }, {
      ...reader.loci[0],
      locusId: "locus_a_fifth",
      answerTokenRange: [4, 5],
      answerUnicodeRange: [[...first, ...second, ...third, ...fourth].length, [...first, ...second, ...third, ...fourth, ...fifth].length],
      answerText: fifth,
      deltaNats: .1,
      absoluteDeltaNats: .1,
      baselineSelectedLogp: -.6,
      intervenedSelectedLogp: -.7,
      perTokenDeltaNats: [.1],
    }];
    reader.measurement.sourceUniverse = ["source-a"];
    reader.measurement.completedSourceCount = 1;
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    const queryResult = queryFixture();
    queryResult.effects = queryResult.effects.slice(0, 1);
    const query = vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryResult);
    const firstEnd = [...first].length;
    const view = render(<InspectSurface route={{
      surface: "inspect",
      runId,
      answerStart: 0,
      answerEnd: firstEnd,
      sourceId: "source-a",
    }} />);

    const panel = await screen.findByRole("complementary", { name: "Most influenced answer spans" });
    expect(within(panel).getAllByRole("button")).toHaveLength(2);
    expect(view.container.querySelectorAll(".answer-segment.is-source-linked")).toHaveLength(4);
    expect(view.container.querySelectorAll(".overview-answer .overview-range.is-selected")).toHaveLength(4);
    expect(view.container.querySelectorAll(".answer-gate")).toHaveLength(4);
    expect(view.container.querySelector(".answer-pane")).toHaveClass("has-linked-selection");
    expect(view.container.querySelector('.answer-segment.is-source-linked[data-link-direction="support"]')).toBeInTheDocument();
    expect(view.container.querySelector('.answer-segment.is-source-linked[data-link-direction="suppression"], .answer-segment.is-source-linked[data-link-direction="mixed"]')).toBeInTheDocument();
    expect(query).toHaveBeenCalledTimes(1);
  });

  it("scales answer heat by relative influence per token across the full visual ramp", async () => {
    const reader = readerFixture();
    const phrases = [
      "First answer sentence has low influence across these words. ",
      "Second answer sentence has medium influence across these words. ",
      "Third answer sentence has strong influence across these words.",
    ];
    const ranges = phrases.map((phrase, index) => {
      const start = phrases.slice(0, index).reduce((length, value) => length + [...value].length, 0);
      return [start, start + [...phrase].length] as const;
    });
    if (reader.answer.status !== "available") throw new Error("fixture answer must be available");
    reader.answer = {
      ...reader.answer,
      text: phrases.join(""),
      tokens: ranges.map(([start, end], index) => ({
        index,
        tokenId: 100 + index,
        piece: phrases[index],
        unicodeRange: [start, end],
        logprob: -1,
      })),
    };
    reader.loci = ranges.map(([start, end], index) => ({
      ...reader.loci[0],
      locusId: `density-${index}`,
      answerTokenRange: [index, index + 1],
      answerUnicodeRange: [start, end],
      answerText: phrases[index],
      deltaNats: [.1, 1, 5][index],
      absoluteDeltaNats: [.1, 1, 5][index],
      perTokenDeltaNats: [[.1], [1], [5]][index],
    }));
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    await screen.findByRole("button", { name: "Tension" });
    expect(view.container.querySelectorAll('.answer-segment[data-heat="true"]')).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Overall" }));
    const heated = [...view.container.querySelectorAll<HTMLElement>('.answer-segment[data-heat="true"]')];
    expect(heated).toHaveLength(3);
    expect(heated.map((element) => element.style.getPropertyValue("--heat-band")))
      .toEqual(["0", "1", "5"]);
    expect(heated.map((element) => element.style.getPropertyValue("--heat-edge-width")))
      .toEqual(["0px", "0px", "2px"]);
    expect(screen.getByText("0–5.000 nats/token")).toBeInTheDocument();
    expect(heated.every((element) => element.dataset.direction === "support")).toBe(true);
    const overview = [...view.container.querySelectorAll<HTMLElement>('.overview-answer button[data-heat="true"]')];
    expect(overview.map((element) => element.style.getPropertyValue("--heat-band")))
      .toEqual(heated.map((element) => element.style.getPropertyValue("--heat-band")));
    expect(heated.filter(element => element.dataset.peak)).toEqual([heated[2]]);
    fireEvent.click(screen.getByRole("button", { name: /Highest overall effect.*Third answer sentence/ }));
    expect(window.location.hash).toBe(`#/inspect/${runId}?answerStart=${ranges[2][0]}&answerEnd=${ranges[2][1]}`);
  });

  it("keeps hover a temporary preview without opening the pinned drawer", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const answer = await screen.findByRole("button", { name: "Hi 🌕!; 2 measured relationships" });
    fireEvent.mouseEnter(answer);

    expect(view.container.querySelector(".evidence-map")).toHaveClass("is-preview");
    expect(screen.getByRole("group", { name: "Influence relationships, preview" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Clear pinned/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("complementary", { name: "Context competing around this answer span" })).not.toBeInTheDocument();
  });

  it("pins a whole measured context passage from the passage surface", async () => {
    const reader = readerFixture();
    const contextLength = [...reader.context.blocks[0].text].length;
    reader.context.blocks[0].sourceIds = ["source-a"];
    reader.sources = [{
      ...reader.sources[0],
      unicodeRange: [0, contextLength],
      displayUnicodeRange: [0, contextLength],
      granularity: "whole_segment",
    }];
    reader.loci = [reader.loci[0]];
    reader.measurement.sourceUniverse = ["source-a"];
    reader.measurement.completedSourceCount = 1;
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    const passage = await screen.findByRole("button", { name: /Pin context passage: “A😀B context signal/ });
    expect(passage).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(passage);
    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=0&answerEnd=5&sourceId=source-a");

    view.rerender(<InspectSurface route={{ surface: "inspect", runId, answerStart: 0, answerEnd: 5, sourceId: "source-a" }} />);
    expect(await screen.findByRole("button", { name: "Clear pinned selection" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Pin context passage: “A😀B context signal/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("complementary", { name: "Most influenced answer spans" })).toBeInTheDocument();
  });

  it("keeps a pinned answer, drawer, and relationship track stable across unrelated hovers", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId, ...answerRange }} />);

    await screen.findByText("2 above-floor relationships");
    expect(screen.getByRole("button", { name: "Clear pinned selection" })).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "Context competing around this answer span" })).toBeInTheDocument();
    expect(view.container.querySelector(".evidence-map")).toHaveClass("is-locked");
    expect(screen.getByRole("group", { name: "Influence relationships, pinned" })).toBeInTheDocument();
    expect(view.container.querySelector<HTMLElement>(".heatmap-peek > div:first-of-type p")).toHaveTextContent("🌕");

    const context = view.container.querySelector<HTMLElement>('.context-segment[data-source-ids="[\\"source b\\"]"]');
    fireEvent.mouseEnter(context!);
    expect(view.container.querySelector<HTMLElement>(".heatmap-peek > div:first-of-type p")).toHaveTextContent("🌕");
    expect(screen.getByRole("complementary", { name: "Context competing around this answer span" })).toBeInTheDocument();
    expect(view.container.querySelector(".evidence-map")).toHaveClass("is-locked");

    fireEvent.mouseLeave(context!);
    expect(view.container.querySelector<HTMLElement>(".heatmap-peek > div:first-of-type p")).toHaveTextContent("🌕");

    fireEvent.click(screen.getByRole("button", { name: "Hi 🌕!; 2 above-floor relationships" }));
    expect(window.location.hash).toBe("#/inspect/run_alpha");
    view.rerender(<InspectSurface route={{ surface: "inspect", runId }} />);
    await waitFor(() => expect(screen.queryByRole("button", { name: "Clear pinned selection" })).not.toBeInTheDocument());
    expect(screen.queryByRole("complementary", { name: "Context competing around this answer span" })).not.toBeInTheDocument();
  });

  it("keeps opposing token effects visible when a source has zero net effect", async () => {
    const reader = readerFixture();
    reader.loci = [{
      ...reader.loci[0], answerTokenRange: [0, 2], answerUnicodeRange: [0, 4],
      perTokenDeltaNats: [2, -2], deltaNats: 0, absoluteDeltaNats: 4,
    }];
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    vi.spyOn(contextInvestigationApi, "query").mockResolvedValue(queryFixture(["below_floor", "below_floor"]));
    const view = render(<InspectSurface route={{ surface: "inspect", runId, answerStart: 0, answerEnd: 5 }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Suppression" }));
    const panel = screen.getByRole("complementary", { name: "Context suppressing this answer span" });
    expect(within(panel).getByText("+2.00 Δ")).toBeInTheDocument();
    expect(within(panel).getByText("−2.00 Δ")).toBeInTheDocument();
    expect(within(panel).getAllByRole("heading", { level: 4 })[0]).toHaveTextContent("Top negative influence");
    fireEvent.click(within(panel).getByRole("button", { name: /Focus suppression span/ }));
    expect(window.location.hash).toBe(`#/inspect/${runId}?answerStart=0&answerEnd=5&sourceId=source-a`);
    view.rerender(<InspectSurface route={{ surface: "inspect", runId, answerStart: 0, answerEnd: 5, sourceId: "source-a" }} />);
    expect(await screen.findByText("Net effect over selection")).toBeInTheDocument();
  });

  it("does not exaggerate a uniformly tiny answer effect to the highest intensity", async () => {
    const reader = readerFixture();
    reader.loci = [{ ...reader.loci[0], perTokenDeltaNats: [.001], deltaNats: .001, absoluteDeltaNats: .001 }];
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(reader);
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);
    fireEvent.click(await screen.findByRole("button", { name: "Overall" }));
    const phrase = view.container.querySelector<HTMLElement>('.answer-segment[data-heat="true"]')!;
    expect(phrase.style.getPropertyValue("--heat-band")).toBe("0");
    expect(screen.getByText("0–0.100 nats/token")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Highest overall effect/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Explore overall: Hi/ }));
    expect(window.location.hash).toBe(`#/inspect/${runId}?answerStart=0&answerEnd=5`);
  });

  it("summarizes context and answer influence and opens a result in the linked view", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Highlights" }));
    expect(screen.getByText("Strongest")).toBeInTheDocument();
    expect(screen.getByText("Weakest")).toBeInTheDocument();
    expect(screen.getByText("Most affected")).toBeInTheDocument();
    expect(screen.getByText("Run details")).toBeInTheDocument();
    const sourceText = screen.getByText("A😀B", { selector: ".ranking-copy strong" });
    fireEvent.click(sourceText.closest("button")!);

    expect(window.location.hash).toBe("#/inspect/run_alpha?answerStart=0&answerEnd=5&sourceId=source-a");
    expect(screen.getByRole("button", { name: "Attribution map" })).toHaveAttribute("aria-current", "page");
  });

  it("runs Lean Context and labels the result as best-found rather than minimal", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const start = vi.spyOn(contextInvestigationApi, "startLeanContext")
      .mockResolvedValue(leanContextJobFixture());
    const view = render(<InspectSurface route={{ surface: "inspect", runId, view: "lean-context" }} />);

    const searchDepth = await screen.findByLabelText("Search depth");
    expect(searchDepth).toHaveValue("16");
    expect(screen.getByRole("heading", { name: "Original context" })).toBeInTheDocument();
    expect(screen.getByText("A😀B context signal.")).toBeInTheDocument();
    expect(screen.queryByText("Measured")).not.toBeInTheDocument();
    fireEvent.change(searchDepth, { target: { value: "32" } });
    fireEvent.click(screen.getByRole("button", { name: "Find lean context" }));

    await waitFor(() => expect(start).toHaveBeenCalledWith(
      runId,
      { maxNewExecutions: 32 },
      expect.any(AbortSignal),
    ));
    const promptTokens = await screen.findByText("Prompt tokens");
    expect(promptTokens.closest("div")).toHaveTextContent("120 → 72");
    expect(screen.getByText("40%")).toBeInTheDocument();
    expect(screen.getByText("Best candidate found within this search budget—not proof of the mathematically smallest context.", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Original context" })).toBeInTheDocument();
    expect(screen.getByText("context", { selector: ".lean-source.is-removed" })).toBeInTheDocument();
    expect(screen.getByText("A😀B", { selector: ".lean-source.is-retained" })).toBeInTheDocument();
    expect(view.container.querySelectorAll(".lean-answer-card blockquote")).toHaveLength(2);
    expect(screen.getAllByText("Answer changed").length).toBeGreaterThan(0);
    expect(screen.getByText("🌕", { selector: ".lean-answer-change.is-before" })).toBeInTheDocument();
    expect(screen.getByText("moon", { selector: ".lean-answer-change.is-after" })).toBeInTheDocument();
    expect(screen.getByText("This candidate passed the recorded-answer score tolerance, but removing the marked context changed a fresh generation. Treat it as a failed behavioral check, not a safe reduction.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Kept only" }));
    expect(screen.getByRole("heading", { name: "Context to keep" })).toBeInTheDocument();
    expect(screen.queryByText("context", { selector: ".lean-source.is-removed" })).not.toBeInTheDocument();
    expect(screen.getByText("A😀B", { selector: ".lean-source.is-retained" })).toBeInTheDocument();
  });

  it("renders a saved exact-match Lean Context result as one shared before-and-after answer", () => {
    const completed = leanContextJobFixture();
    if (!completed.result) throw new Error("fixture must include a result");
    const exactJob: LeanContextJob = {
      ...completed,
      result: {
        ...completed.result,
        generationComparison: {
          ...completed.result.generationComparison,
          fullContext: { text: "Hi 🌕!", matchesRecorded: true },
          leanContext: { text: "Hi 🌕!", matchesRecorded: true },
          exactMatch: true,
          normalizedMatch: true,
        },
      },
    };
    const view = render(<LeanContextPanel
      reader={readerFixture()}
      route={{ surface: "inspect", runId, view: "lean-context" }}
      initialJob={exactJob}
    />);

    expect(screen.getByText("Exact match")).toBeInTheDocument();
    expect(screen.getByLabelText("Before and after answers are an exact match")).toHaveTextContent("Before Full context");
    expect(screen.getByLabelText("Before and after answers are an exact match")).toHaveTextContent("After Lean context");
    expect(view.container.querySelectorAll(".lean-answer-exact blockquote")).toHaveLength(1);
    expect(view.container.querySelector(".lean-answer-card")).not.toBeInTheDocument();
  });

  it("leads with the outcome and omits an empty removed-context section when no reduction passes", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const completed = leanContextJobFixture();
    if (!completed.result?.original) throw new Error("fixture must include an original candidate");
    vi.spyOn(contextInvestigationApi, "startLeanContext").mockResolvedValue({
      ...completed,
      result: {
        ...completed.result,
        best: completed.result.original,
        reduction: {
          removedPromptTokens: 0,
          retainedPromptTokenFraction: 1,
          removedSourceCount: 0,
          retainedSourceCount: 2,
        },
      },
    });
    const view = render(<InspectSurface route={{ surface: "inspect", runId, view: "lean-context" }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Find lean context" }));

    expect(await screen.findByText("Nothing was removed")).toBeInTheDocument();
    expect(screen.getByText("Context removed").closest("div")).toHaveTextContent("0%");
    expect(view.container.querySelector(".lean-source.is-removed")).not.toBeInTheDocument();
    expect(view.container.querySelectorAll(".lean-source.is-retained")).toHaveLength(2);
  });

  it("fills the interaction heatmap from a non-terminal sampled result", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const start = vi.spyOn(contextInvestigationApi, "startInteractions")
      .mockResolvedValue(interactionsJobFixture("queued"));
    vi.spyOn(contextInvestigationApi, "interactionsJob")
      .mockResolvedValue(interactionsJobFixture("running"));
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Highlights" }));
    fireEvent.click(screen.getByText("Test whether influence changes across context combinations"));
    expect(screen.getByText("A large shift suggests interaction with other context; it does not by itself prove redundancy.", { exact: false })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run stability test" }));

    await waitFor(() => expect(start).toHaveBeenCalledWith(runId, {
      maxNewExecutions: 16,
      maxPairs: 32,
      seed: 0,
    }, expect.any(AbortSignal)));
    expect(await screen.findByText("+1.50 Δ")).toBeInTheDocument();
    expect(screen.getByText("1 / 2 spans sampled")).toBeInTheDocument();
    expect(screen.getByText("Waiting for a completed pair")).toBeInTheDocument();
    expect(screen.getByText("+1.08 vs leave-one-out")).toBeInTheDocument();
    expect(view.container.querySelector(".interaction-row.direction-support .interaction-bar")).toBeInTheDocument();
    expect(view.container.querySelector(".interaction-marker")).toBeInTheDocument();
  });

  it("summarizes unsampled interaction spans after a completed budget instead of leaving placeholders", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    vi.spyOn(contextInvestigationApi, "startInteractions")
      .mockResolvedValue(interactionsJobFixture("completed"));
    const view = render(<InspectSurface route={{ surface: "inspect", runId }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Highlights" }));
    fireEvent.click(screen.getByText("Test whether influence changes across context combinations"));
    fireEvent.click(screen.getByRole("button", { name: "Run stability test" }));

    expect(await screen.findByText("1 span not sampled within this budget.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for a completed pair")).not.toBeInTheDocument();
    expect(view.container.querySelectorAll(".interaction-row")).toHaveLength(1);
  });

  it("shows honest indeterminate progress while a Lean Context model batch is running", async () => {
    vi.spyOn(contextInvestigationApi, "reader").mockResolvedValue(readerFixture());
    const { result: _completedResult, ...baseJob } = leanContextJobFixture();
    const runningJob: LeanContextJob = {
      ...baseJob,
      state: "running",
      progress: {
        phase: "searching_lean_context",
        completedUnits: 0,
        totalUnits: 16,
        percent: 0,
      },
      cancellable: true,
    };
    vi.spyOn(contextInvestigationApi, "startLeanContext").mockResolvedValue(runningJob);
    vi.spyOn(contextInvestigationApi, "leanContextJob").mockResolvedValue(runningJob);
    render(<InspectSurface route={{ surface: "inspect", runId, view: "lean-context" }} />);

    fireEvent.click(await screen.findByRole("button", { name: "Find lean context" }));

    expect(await screen.findByText("Working…")).toBeInTheDocument();
    expect(screen.getByText("Existing measurements are reused first. New model work reports after the current search batch finishes.")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).not.toHaveAttribute("value");
  });

});
