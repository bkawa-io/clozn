import userEvent from "@testing-library/user-event";
import { act } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { createFork, loadRunInspection, loadRuntimeState } from "../data/api";
import type { ForkResult } from "../data/api";
import type { ObservatoryData, RunSummary, RuntimeState } from "../data/types";
import { deferred } from "../test/fetch";
import { render, screen, waitFor, within } from "../test/render";
import { ScopePanel } from "./scope";

/** The bottom "Output tokens" tape (Observatory.tsx's own token-panel listbox) is the one place a
 * committed token's text is unambiguous -- the same reading also appears, deliberately, in several
 * other spots (the inspector summary, the fork-control's "from -> to" line), so a bare `getByText`
 * finds duplicates. Scoping to this listbox matches the pattern Observatory.test.tsx already uses. */
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
  loadCausalTrace: vi.fn(),
}));

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
const runtime: RuntimeState = {
  status: "connected",
  runs: [summary(runA), summary(runB)],
  engine: { model: "model-x", layerCount: 4, jlens: false, sae: false },
};

const loadInspection = vi.mocked(loadRunInspection);
const createForkMock = vi.mocked(createFork);
const loadRuntime = vi.mocked(loadRuntimeState);

beforeEach(() => {
  location.hash = "#/scope";
  loadInspection.mockReset();
  createForkMock.mockReset();
  loadRuntime.mockReset();
  loadInspection.mockImplementation(async (runId) => {
    if (runId === runA.id) return runA;
    if (runId === runB.id) return runB;
    throw new Error(`unknown test run ${runId}`);
  });
  loadRuntime.mockResolvedValue(runtime);
});

describe("ScopePanel fork race safety", () => {
  test("a slow fork response for run A never paints over a run selected meanwhile", async () => {
    const forkResponse = deferred<ForkResult>();
    createForkMock.mockReturnValue(forkResponse.promise);
    const user = userEvent.setup();

    render(<ScopePanel runtime={runtime} inspectorOpen params={{ runId: runA.id }} />);

    // Run A is loaded first (the mounting useEffect calls selectRun(runA.id)).
    await waitFor(() => expect(outputToken("a0")).toBeInTheDocument());

    // The fork button's default forced token is the recorded alternative "alt" (tokenId 99) --
    // firing the fork request without needing to click a candidate first.
    await user.click(screen.getByRole("button", { name: "FORK RUN" }));
    expect(createForkMock).toHaveBeenCalledWith(runA.id, expect.any(Number), "alt", 99);
    await waitFor(() => expect(screen.getByRole("button", { name: "FORKING" })).toBeInTheDocument());

    // While that fork request for A is still in flight, the user picks a different run: B.
    await user.selectOptions(screen.getByRole("combobox", { name: "RUN" }), runB.id);
    await waitFor(() => expect(outputToken("b0")).toBeInTheDocument());
    expect(screen.queryByRole("option", { name: "a0" })).not.toBeInTheDocument();

    // NOW the slow fork response for A arrives.
    await act(async () => {
      forkResponse.resolve({
        outcome: {
          kind: "exact_execution_fork",
          reasons: [{ code: "exact_preconditions_met", message: "ok" }],
          exactness: { regime: "prompt_boundary_reprefill", source: "reprefill", proofStatus: "confirmed" },
        },
        child: { id: "child-of-a", parentId: runA.id },
      });
    });

    // The stale response must never load its child, never touch the displayed run, and never rewrite
    // the URL to the forked child -- run B stays exactly as selected.
    expect(outputToken("b0")).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "a0" })).not.toBeInTheDocument();
    expect(screen.queryByText(/CHILD child-of-a/)).not.toBeInTheDocument();
    expect(loadInspection).not.toHaveBeenCalledWith("child-of-a");
    expect(location.hash).not.toContain("child-of-a");
    expect(location.hash).toContain(runB.id);
  });
});
