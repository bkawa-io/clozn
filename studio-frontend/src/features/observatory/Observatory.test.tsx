import userEvent from "@testing-library/user-event";
import { act } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { loadRunInspection, loadRuntimeState } from "../../data/api";
import type {
  ObservatoryData,
  RunSummary,
  RuntimeState,
} from "../../data/types";
import type { WorkbenchDocument } from "../../data/tokenWorkbench";
import { deferred } from "../../test/fetch";
import { render, screen, waitFor, within } from "../../test/render";
import { ScopePanel } from "../../panels/scope";
import { Observatory } from "./Observatory";
import {
  parseScopeUrl,
  scopeRouteParams,
  serializeScopeUrl,
  type ScopeSelectionState,
} from "./urlState";

vi.mock("../../data/api", () => ({
  loadRunInspection: vi.fn(),
  loadRuntimeState: vi.fn(),
}));

vi.mock("./layerApi", () => ({
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

vi.mock("../../data/tokenWorkbench", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../data/tokenWorkbench")>();
  return {
    ...actual,
    loadTokenWorkbench: vi.fn(),
    postForceTokenAction: vi.fn(),
    postCausalTraceAction: vi.fn(),
    postSourceMeasureAction: vi.fn(),
    postMechanisticDiffAction: vi.fn(),
    loadWorkbenchJob: vi.fn(),
    cancelWorkbenchJob: vi.fn(),
  };
});

function reading(
  id: string,
  label: string,
  tokenPrefix: string,
  model = "model-current",
): ObservatoryData {
  return {
    id,
    label,
    model,
    quant: "Q5_K_M",
    createdAt: "12:00:00",
    duration: "1.0 s",
    mode: "run",
    prompt: "shared prompt",
    response: `${tokenPrefix}0${tokenPrefix}1${tokenPrefix}2`,
    tokens: [0, 1, 2].map((index) => ({
      text: `${tokenPrefix}${index}`,
      entropy: 0.1 + index / 10,
      confidence: 0.9 - index / 10,
    })),
    candidates: [
      { token: `${tokenPrefix}0`, score: 0.8, delta: 0 },
      { token: "alternate", score: 0.15, delta: -0.65, tokenId: 4242 },
    ],
    sources: [],
    configuration: {
      adapters: [],
      changes: [],
    },
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
  };
}

const current = reading("run-current", "Current run", "c");
const referenceA = reading("run-reference-a", "Reference A", "a", "model-a");
const referenceB = reading("run-reference-b", "Reference B", "b", "model-b");
const child = reading("run-child", "Forked child", "d");
const runtime: RuntimeState = {
  status: "connected",
  runs: [summary(current), summary(referenceA), summary(referenceB), summary(child)],
  engine: {
    model: "model-current",
    layerCount: 6,
    jlens: false,
    sae: false,
  },
};
const loadInspection = vi.mocked(loadRunInspection);
const loadRuntime = vi.mocked(loadRuntimeState);

async function importWorkbenchMocks() {
  return await import("../../data/tokenWorkbench");
}

/** A fully valid workbench document -- every capability defaults to unavailable so a test only has to
 * override the one it cares about. */
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
        { piece: `${runId}-${index}`, prob: 0.8 },
        { piece: "alternate", tokenId: 4242, prob: 0.15 },
      ],
    },
    context: { state: "unavailable", raw: {} },
    comparison: { state: "unavailable", raw: {} },
    readouts: { state: "unavailable", raw: {} },
    capabilities: {
      exactFork: { available: false, snapshotState: "worker_unreachable", reason: "no worker reachable" },
      sourceMeasurement: { available: false, status: "unavailable", reason: "no worker" },
      causalTrace: { available: false, status: "unavailable", reason: "no worker" },
      mechanisticDiff: { available: false, reason: "no reference run selected" },
    },
    ...overrides,
  };
}

function observatory(
  overrides: Partial<React.ComponentProps<typeof Observatory>> = {},
) {
  return (
    <Observatory
      data={current}
      runtime={runtime}
      inspectorOpen
      runStatus="idle"
      onSelectRun={() => {}}
      {...overrides}
    />
  );
}

beforeEach(async () => {
  location.hash = "#/scope";
  loadInspection.mockReset();
  loadInspection.mockImplementation(async (runId) => {
    if (runId === current.id) return current;
    if (runId === referenceA.id) return referenceA;
    if (runId === referenceB.id) return referenceB;
    if (runId === child.id) return child;
    throw new Error(`unknown test run ${runId}`);
  });
  loadRuntime.mockReset();
  loadRuntime.mockResolvedValue(runtime);
  const wb = await importWorkbenchMocks();
  vi.mocked(wb.loadTokenWorkbench).mockReset();
  vi.mocked(wb.loadTokenWorkbench).mockImplementation(async (runId, index) => workbenchDoc(runId, index));
  vi.mocked(wb.postForceTokenAction).mockReset();
  vi.mocked(wb.postCausalTraceAction).mockReset();
  vi.mocked(wb.postSourceMeasureAction).mockReset();
  vi.mocked(wb.postMechanisticDiffAction).mockReset();
  vi.mocked(wb.loadWorkbenchJob).mockReset();
  vi.mocked(wb.cancelWorkbenchJob).mockReset();
});

describe("Scope URL integration", () => {
  test("restores view, token, reference, and layer after the run loads", async () => {
    const hash = "#/runs/run-current/scope?layer=4&reference=run-reference-a&token=2&view=layers";
    const route = parseScopeUrl(hash)!;
    location.hash = hash;
    const user = userEvent.setup();

    render(
      <ScopePanel
        runtime={runtime}
        inspectorOpen
        params={scopeRouteParams(route)}
      />,
    );

    expect(await screen.findByRole("heading", { name: "Layer evidence" })).toBeInTheDocument();
    const tape = screen.getByRole("listbox", { name: "Output tokens" });
    expect(within(tape).getAllByRole("option")[2]).toHaveAttribute("aria-selected", "true");
    const inspector = screen.getByRole("heading", { name: "Layer inspector" }).closest("aside");
    expect(inspector).not.toBeNull();
    expect(within(inspector!).getAllByText("L4").length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(location.hash).toBe(
        "#/runs/run-current/scope?view=layers&token=2&reference=run-reference-a&layer=4",
      );
    });

    await user.click(screen.getByRole("button", { name: "VARIANTS" }));
    expect(await screen.findByRole("combobox", { name: "REFERENCE RUN" }))
      .toHaveValue(referenceA.id);
    await waitFor(() => {
      expect(parseScopeUrl(location.hash)?.state).toMatchObject({
        view: "variants",
        token: 2,
        reference: referenceA.id,
        layer: 4,
      });
    });
  });

  test("clamps token and layer after run bounds are known", async () => {
    // Simulate the backend's OWN auto-selected comparison reference (clozn.runs.token_workbench's
    // `_comparison_section`, composed via useTokenWorkbench's one-time auto-pick) -- this Studio no
    // longer runs a parallel client-side "same prompt" heuristic for the default reference.
    const wb = await importWorkbenchMocks();
    vi.mocked(wb.loadTokenWorkbench).mockImplementation(async (runId, index) => workbenchDoc(runId, index, {
      comparison: { state: "supported", raw: { selection: { mode: "previous_compatible", reference_run_id: referenceA.id } } },
    }));
    const changed = vi.fn();
    render(observatory({
      initialState: { view: "layers", token: 999, layer: 999 },
      onStateChange: changed,
    }));

    await waitFor(() => {
      expect(changed).toHaveBeenLastCalledWith({
        view: "layers",
        token: 2,
        reference: referenceA.id,
        layer: 5,
      });
    });
    const tape = screen.getByRole("listbox", { name: "Output tokens" });
    expect(within(tape).getAllByRole("option")[2]).toHaveAttribute("aria-selected", "true");
  });

  test("keyboard token selection updates the canonical URL with replaceState", async () => {
    const route = parseScopeUrl("#/runs/run-current/scope?token=0")!;
    location.hash = "#/runs/run-current/scope?token=0";
    const replace = vi.spyOn(history, "replaceState");
    const user = userEvent.setup();

    render(
      <ScopePanel
        runtime={runtime}
        inspectorOpen
        params={scopeRouteParams(route)}
      />,
    );

    expect(await screen.findByText("COMPLETED RUN")).toBeInTheDocument();
    const tape = screen.getByRole("listbox", { name: "Output tokens" });
    const first = within(tape).getAllByRole("option")[0];
    first.focus();
    await user.keyboard("{ArrowRight}");

    await waitFor(() => {
      expect(parseScopeUrl(location.hash)?.state.token).toBe(1);
    });
    expect(replace).toHaveBeenCalled();
    expect(within(tape).getAllByRole("option")[1]).toHaveAttribute("aria-selected", "true");
  });
});

describe("Observatory async and action boundaries", () => {
  test("an older reference response cannot overwrite a newer route selection", async () => {
    const first = deferred<ObservatoryData>();
    const second = deferred<ObservatoryData>();
    loadInspection.mockImplementation((runId) => {
      if (runId === referenceA.id) return first.promise;
      if (runId === referenceB.id) return second.promise;
      return Promise.resolve(current);
    });

    const view = render(observatory({
      initialState: { view: "variants", token: 0, reference: referenceA.id, layer: 0 },
    }));
    await waitFor(() => expect(loadInspection).toHaveBeenCalledWith(
      referenceA.id,
      expect.any(AbortSignal),
    ));
    const firstSignal = loadInspection.mock.calls.find(
      ([runId]) => runId === referenceA.id,
    )?.[1];

    view.rerender(observatory({
      initialState: { view: "variants", token: 0, reference: referenceB.id, layer: 0 },
    }));
    await waitFor(() => expect(loadInspection).toHaveBeenCalledWith(
      referenceB.id,
      expect.any(AbortSignal),
    ));
    expect(firstSignal?.aborted).toBe(true);

    await act(async () => second.resolve(referenceB));
    expect(await screen.findByText("b0")).toBeInTheDocument();
    await act(async () => first.resolve(referenceA));
    expect(screen.queryByText("a0")).not.toBeInTheDocument();
    expect(screen.getByText("b0")).toBeInTheDocument();
  });

  test("selecting a token never runs an action -- only the FORK button does", async () => {
    const wb = await importWorkbenchMocks();
    const user = userEvent.setup();
    const onStateChange = vi.fn((state: ScopeSelectionState) => {
      history.replaceState(null, "", serializeScopeUrl(current.id, state));
    });
    render(observatory({ initialState: { view: "trace", token: 0, layer: 0 }, onStateChange }));

    await waitFor(() => expect(vi.mocked(wb.loadTokenWorkbench)).toHaveBeenCalledTimes(1));
    const tape = screen.getByRole("listbox", { name: "Output tokens" });
    await user.click(within(tape).getAllByRole("option")[2]);
    await waitFor(() => expect(vi.mocked(wb.loadTokenWorkbench)).toHaveBeenCalledTimes(2));
    expect(vi.mocked(wb.postForceTokenAction)).not.toHaveBeenCalled();
    expect(vi.mocked(wb.postCausalTraceAction)).not.toHaveBeenCalled();
    expect(vi.mocked(wb.postSourceMeasureAction)).not.toHaveBeenCalled();
    await waitFor(() => expect(parseScopeUrl(location.hash)?.state.token).toBe(2));
  });

  test("a completed force-token result stays as evidence until explicit materialization", async () => {
    const wb = await importWorkbenchMocks();
    vi.mocked(wb.loadTokenWorkbench).mockImplementation(async (runId, index) => {
      if (runId === current.id) {
        return workbenchDoc(runId, index, {
          capabilities: {
            exactFork: { available: true, snapshotState: "not_attempted" },
            sourceMeasurement: { available: false, status: "unavailable", reason: "no worker" },
            causalTrace: { available: false, status: "unavailable", reason: "no worker" },
            mechanisticDiff: { available: false, reason: "no reference run selected" },
          },
        });
      }
      return workbenchDoc(runId, index);
    });
    vi.mocked(wb.postForceTokenAction).mockResolvedValue({
      outcome: "cached",
      artifact: {
        schema_version: "clozn.time-travel-result.v1",
        run_id: current.id,
        status: "completed",
        fidelity: "RECONSTRUCTED",
        experiment_id: "exp-1",
        arm_id: "arm-1",
        observation_id: "obs-1",
        continuation: { generated_suffix_text: " alternative" },
      },
    });
    const onSelectRun = vi.fn();
    const user = userEvent.setup();
    const view = render(observatory({ onSelectRun }));

    await waitFor(() => expect(screen.getByRole("article", { name: "FORCE TOKEN" })).toBeInTheDocument());
    const forkRow = screen.getByRole("article", { name: "FORCE TOKEN" });
    await waitFor(() => expect(within(forkRow).getByRole("button", { name: "RUN" })).toBeEnabled());
    await user.click(within(forkRow).getByRole("button", { name: "RUN" }));

    expect(onSelectRun).not.toHaveBeenCalled();
    expect(await screen.findByText("GENERATED OBSERVATION obs-1 · RECONSTRUCTED")).toBeInTheDocument();
    expect(screen.getByText(/Materialization is explicit/)).toBeInTheDocument();
  });

  test("an unavailable action shows its typed reason as visible text, never color alone", async () => {
    render(observatory());
    const traceRow = await screen.findByRole("article", { name: "CAUSAL TRACE" });
    expect(within(traceRow).getByText("no worker")).toBeInTheDocument();
    expect(within(traceRow).getByRole("button", { name: "UNAVAILABLE" })).toBeDisabled();
  });
});
