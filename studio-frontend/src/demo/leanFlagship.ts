import type { ContextInvestigationReader } from "../data/contextInvestigation";
import type { LeanContextJob } from "../data/leanContext";

export const leanDemoRunId = "run_lean_libraries_demo";

const messages = [
  {
    role: "system",
    text: "You are an editorial research assistant. Use only the supplied reporting notes, preserve corrections over earlier drafts, and avoid claims the evidence cannot support.",
  },
  {
    role: "user",
    text: "I am writing an essay arguing that libraries should replace emergency cooling centers during heat waves. Open dramatically with the 2,400 people the libraries saved, then explain why medical tents are obsolete.",
  },
  {
    role: "assistant",
    text: "Libraries are becoming the emergency rooms of a warming city. During Redwood's heat wave they saved 2,400 people after 5 p.m., proving that familiar neighborhood branches can replace temporary medical tents.",
  },
  {
    role: "user",
    text: "Reporting notes and correction: the emergency lasted 18 days. The library system recorded 2,400 cooling visits after 5 p.m.; that is a visit count, not 2,400 unique people, rescues, or measured health outcomes. Medical tents remained the frontline response for clinical care. Libraries extended hours at selected branches when staff and security were added, offering shade, water, charging, and trusted information. The useful argument is that ordinary access can become emergency capacity—not that libraries replace medical response.",
  },
  {
    role: "user",
    text: "Write one compact paragraph for the essay. Distinguish visits from health outcomes, compare libraries with tents without claiming one replaces the other, and close on the idea that everyday civic access can become emergency capacity.",
  },
] as const;

const answer = "During Redwood's 18-day heat emergency, libraries recorded 2,400 cooling visits after 5 p.m.—evidence of demand, not a count of patients or proof of better health outcomes. Medical tents remained the city's frontline response for people needing clinical care, while selected library branches supplied a different kind of resilience: familiar, distributed rooms that could extend their hours when staffing and security were added. Libraries should not replace emergency sites; they can widen the places where residents find shade, water, power, and trusted information before a crisis becomes medical. The lesson is more precise than a heroic rescue story: everyday civic access can become emergency capacity.";

const codePoints = (value: string) => [...value].length;
const bytes = (value: string) => new TextEncoder().encode(value).length;
const sourceIds = messages.map((_, index) => `lean-message-${index}`);

let displayOffset = 0;
const blocks = messages.map((message, index) => {
  const length = codePoints(message.text);
  const start = displayOffset;
  displayOffset += length + (index === messages.length - 1 ? 0 : 2);
  return {
    messageIndex: index,
    role: message.role,
    text: message.text,
    segmentId: `lean-segment-${index}`,
    displayUnicodeRange: [start, start + length] as const,
    sourceIds: [sourceIds[index]],
  };
});

export const leanDemoReader: ContextInvestigationReader = {
  schemaVersion: "clozn.context-investigation-reader.v1",
  runId: leanDemoRunId,
  status: "completed",
  measurement: {
    status: "completed",
    experimentId: "experiment_lean_libraries_demo",
    evaluator: "score_recorded_continuation",
    intervention: "delete_source",
    sourceUniverse: sourceIds,
    sourceUniverseBasis: "message_scoped_units",
    completedSourceCount: sourceIds.length,
    unavailableSourceCount: 0,
    failedSourceCount: 0,
    notMeasuredSourceCount: 0,
    baselineObservationId: "observation_lean_libraries_baseline",
  },
  context: {
    coordinateBasis: "context_display_unicode.v1",
    separator: "\n\n",
    blocks,
  },
  answer: {
    status: "available",
    text: answer,
    sha256: "d".repeat(64),
    tokens: [{
      index: 0,
      tokenId: 1,
      piece: answer,
      unicodeRange: [0, codePoints(answer)],
      logprob: -.18,
    }],
    fidelity: "recorded_run_trace",
    observationId: "observation_lean_libraries_baseline",
  },
  sources: messages.map((message, index) => ({
    sourceId: sourceIds[index],
    segmentId: `lean-segment-${index}`,
    messageIndex: index,
    label: null,
    provenanceKind: "message",
    parentSourceId: null,
    unicodeRange: [0, codePoints(message.text)],
    byteRange: [0, bytes(message.text)],
    displayUnicodeRange: blocks[index].displayUnicodeRange,
    granularity: "whole_segment",
    measurement: {
      status: "measured",
      experimentId: "experiment_lean_libraries_demo",
      armId: `arm_lean_message_${index}`,
      observationId: `observation_lean_message_${index}`,
      intervention: {
        kind: "delete_source",
        target: { kind: "context_selection", sourceIds: [sourceIds[index]] },
      },
    },
    summary: {
      maxAbsTokenDeltaNats: 0,
      totalContinuationDeltaNats: 0,
      aboveFloorLocusCount: 0,
    },
    loci: [],
  })),
  loci: [],
  presentation: { measurementFloorNats: .1 },
  capabilities: {
    contextEffects: "measurable",
    measurement: "completed",
    answerSelectionProjection: "available",
  },
};

const retainedSourceIds = [sourceIds[0], sourceIds[3], sourceIds[4]];
const removedSourceIds = [sourceIds[1], sourceIds[2]];

export const leanDemoJob: LeanContextJob = {
  schemaVersion: "clozn.analysis-job.v1",
  jobId: "job_lean_libraries_demo",
  runId: leanDemoRunId,
  kind: "context_investigation_lean_context",
  state: "completed",
  progress: { phase: "done", completedUnits: 16, totalUnits: 16, percent: 100 },
  cancelRequested: false,
  cancellable: false,
  cached: true,
  result: {
    schemaVersion: "clozn.lean-context-result.v1",
    runId: leanDemoRunId,
    searchId: "lean_libraries_demo",
    status: "completed",
    claim: { kind: "best_found_within_budget", minimumProven: false },
    tolerance: {
      profile: "balanced.v1",
      meanPositiveSupportLossNatsPerToken: .05,
      maxTokenSupportLossNats: .7,
    },
    sourceIds,
    original: {
      retainedSourceIds: sourceIds,
      removedSourceIds: [],
      renderedPromptTokenCost: 318,
      toleranceStatus: "within_tolerance",
      loss: {
        answerTokenCount: 109,
        positiveLossTokenCount: 0,
        meanPositiveSupportLossNatsPerToken: 0,
        maxTokenSupportLossNats: 0,
      },
      observationId: "observation_lean_libraries_original",
    },
    best: {
      retainedSourceIds,
      removedSourceIds,
      renderedPromptTokenCost: 176,
      toleranceStatus: "within_tolerance",
      loss: {
        answerTokenCount: 109,
        positiveLossTokenCount: 8,
        meanPositiveSupportLossNatsPerToken: .006,
        maxTokenSupportLossNats: .083,
      },
      observationId: "observation_lean_libraries_best",
    },
    generationComparison: {
      status: "completed",
      decodeMode: "greedy",
      maxNewTokens: 180,
      fullContext: { text: answer, matchesRecorded: true },
      leanContext: { text: answer, matchesRecorded: true },
      exactMatch: true,
      normalizedMatch: true,
    },
    reduction: {
      removedPromptTokens: 142,
      retainedPromptTokenFraction: 176 / 318,
      removedSourceCount: removedSourceIds.length,
      retainedSourceCount: retainedSourceIds.length,
    },
    trials: [],
    budget: {
      maxNewExecutions: 16,
      usedNewExecutions: 12,
      reusedObservationCount: 5,
      exhausted: false,
      blockedByBudget: false,
    },
    elapsedSeconds: 8.4,
    stoppingReason: "search_policy_complete",
  },
};
