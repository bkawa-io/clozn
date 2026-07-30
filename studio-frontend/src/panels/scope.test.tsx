import userEvent from "@testing-library/user-event";
import { act } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { loadRunInspection, loadRuntimeState } from "../data/api";
import type { ObservatoryData, RunSummary, RuntimeState } from "../data/types";
import type { WorkbenchActionResult, WorkbenchDocument } from "../data/tokenWorkbench";
import { deferred } from "../test/fetch";
import { render, screen, waitFor, within } from "../test/render";
import { ScopePanel } from "./scope";

/** The bottom "Output tokens" tape (Observatory.tsx's own token-panel listbox) is the one place a
 * committed token's text is unambiguous -- the same reading also appears, deliberately, in several
 * other spots (the inspector summary, the action tray's fork row), so a bare `getByText` finds
 * duplicates. Scoping to this listbox matches the pattern Observatory.test.tsx already uses. */
function outputToken(name: string) {
  return within(screen.getByRole("listbox", { name: "Output tokens" })).getByRole("option", { name });
}

vi.mock("../data/api", () => ({
  createFork: vi.fn(),
  loadRunInspection: vi.fn(),
  loadRuntimeState: vi.fn(),
}));

vi.mock("../features/observatory/layerApi", () => ({
  loadLayerEvidence: vi.fn(async () => ({
    residual: {
      available: false,
      tokens: [],
      norms: [],
      layerMean: [],
      nLayer: 0,
      nTokens: 0,
      textChars: 0,
      truncated: false,
    },
    jlens: {
      available: false,
      layers: [],
      availableLayers: [],
      textChars: 0,
      truncated: false,
    },
  })),
}));

vi.mock("../data/tokenWorkbench", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../data/tokenWorkbench")>();
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

function reading(id: string, tokenPrefix: string): ObservatoryData {
  return {
    id,
    label: id,
    model: "model-x",
    quant: "Q5_K_M",
    createdAt: "12:00:00",
    duration: "1.0 s",
    mode: "run",
    prompt: `${id} prompt`,
    response: `${tokenPrefix}0${tokenPrefix}1`,
    tokens: [0, 1].map((index) => ({
      text: `${tokenPrefix}${index}`,
      entropy: 0.2,
      confidence: 0.8,
    })),
    candidates: [
      { token: `${tokenPrefix}0`, score: 0.8, delta: 0 },
      { token: "alt", score: 0.1, delta: -0.7, tokenId: 99 },
    ],
    sources: [],
    configuration: { activeDials: {}, memoryCards: [], adapters: [], changes: [] },
  };
}

function summary(data: ObservatoryData): RunSummary {
  return {
    id: data.id,
    label: data.label,
    prompt: data.prompt ?? "",
    response: data.response ?? "",
    createdAt: data.createdAt,
    source: "test",
    client: "test",
    model: data.model,
    substrate: "engine",
    duration: data.duration,
    flags: [],
    warningCount: 0,
    activeDialCount: 0,
    memoryCardCount: 0,
  };
}

const runA = reading("run-a", "a");
const runB = reading("run-b", "b");
const childOfA = reading("child-of-a", "d");
const runtime: RuntimeState = {
  status: "connected",
  runs: [summary(runA), summary(runB), summary(childOfA)],
  engine: { model: "model-x", layerCount: 4, jlens: false, sae: false },
};

const loadInspection = vi.mocked(loadRunInspection);
const loadRuntime = vi.mocked(loadRuntimeState);

function workbenchDoc(runId: string, index: number, overrides: Partial<WorkbenchDocument> = {}): WorkbenchDocument {
  return {
    schemaVersion: "clozn.token-workbench.v1",
    runId,
    index,
    run: { id: runId },
    token: {
      index,
      piece: `${runId}-${index}`,
      alternatives: [
        { piece: `${runId}-${index}`, prob: 0.9 },
        { piece: "alt", tokenId: 99, prob: 0.1 },
      ],
    },
    context: { state: "unavailable", raw: {} },
    comparison: { state: "unavailable", raw: {} },
    readouts: { state: "unavailable", raw: {} },
    capabilities: {
      exactFork: { available: true, snapshotState: "not_attempted" },
      sourceMeasurement: { available: false, status: "unavailable", reason: "no worker" },
      causalTrace: { available: false, status: "unavailable", reason: "no worker" },
      mechanisticDiff: { available: false, reason: "no reference run selected" },
    },
    ...overrides,
  };
}

beforeEach(async () => {
  location.hash = "#/scope";
  loadInspection.mockReset();
  loadRuntime.mockReset();
  loadInspection.mockImplementation(async (runId) => {
    if (runId === runA.id) return runA;
    if (runId === runB.id) return runB;
    if (runId === childOfA.id) return childOfA;
    throw new Error(`unknown test run ${runId}`);
  });
  loadRuntime.mockResolvedValue(runtime);
  const wb = await import("../data/tokenWorkbench");
  vi.mocked(wb.loadTokenWorkbench).mockReset();
  vi.mocked(wb.loadTokenWorkbench).mockImplementation(async (runId, index) => workbenchDoc(runId, index));
  vi.mocked(wb.postForkAction).mockReset();
  vi.mocked(wb.postCausalTraceAction).mockReset();
  vi.mocked(wb.postSourceMeasureAction).mockReset();
  vi.mocked(wb.postMechanisticDiffAction).mockReset();
  vi.mocked(wb.loadWorkbenchJob).mockReset();
  vi.mocked(wb.cancelWorkbenchJob).mockReset();
});

describe("ScopePanel fork race safety", () => {
  test("a slow fork response for run A never paints over a run selected meanwhile", async () => {
    const wb = await import("../data/tokenWorkbench");
    const forkResponse = deferred<WorkbenchActionResult<import("../data/api").ForkArtifact>>();
    vi.mocked(wb.postForkAction).mockReturnValue(forkResponse.promise);
    const user = userEvent.setup();

    render(<ScopePanel runtime={runtime} inspectorOpen params={{ runId: runA.id }} />);

    // Run A is loaded first (the mounting useEffect calls selectRun(runA.id)).
    await waitFor(() => expect(outputToken("a0")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole("article", { name: "FORK" })).toBeInTheDocument());
    const forkRow = screen.getByRole("article", { name: "FORK" });
    await waitFor(() => expect(within(forkRow).getByRole("button", { name: "RUN" })).toBeEnabled());

    await user.click(within(forkRow).getByRole("button", { name: "RUN" }));
    expect(vi.mocked(wb.postForkAction)).toHaveBeenCalledWith(runA.id, expect.any(Number), "alt", 99);
    await waitFor(() => expect(within(forkRow).getByRole("button", { name: "STARTING" })).toBeInTheDocument());

    // While that fork request for A is still in flight, the user picks a different run: B.
    await user.selectOptions(screen.getByRole("combobox", { name: "RUN" }), runB.id);
    await waitFor(() => expect(outputToken("b0")).toBeInTheDocument());
    expect(screen.queryByRole("option", { name: "a0" })).not.toBeInTheDocument();

    // NOW the slow fork response for A arrives.
    await act(async () => {
      forkResponse.resolve({
        outcome: "cached",
        artifact: {
          outcome: {
            kind: "exact_execution_fork",
            reasons: [{ code: "exact_preconditions_met", message: "ok" }],
            exactness: { regime: "prompt_boundary_reprefill", source: "reprefill", proofStatus: "confirmed" },
          },
          child: { id: childOfA.id, parentId: runA.id },
        },
      });
    });

    // The stale response must never load its child, never touch the displayed run, and never rewrite
    // the URL to the forked child -- run B stays exactly as selected.
    expect(outputToken("b0")).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "a0" })).not.toBeInTheDocument();
    expect(screen.queryByText(new RegExp(`CHILD ${childOfA.id}`))).not.toBeInTheDocument();
    expect(loadInspection).not.toHaveBeenCalledWith(childOfA.id);
    expect(location.hash).not.toContain(childOfA.id);
    expect(location.hash).toContain(runB.id);
  });
});
