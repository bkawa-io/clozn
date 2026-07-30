import userEvent from "@testing-library/user-event";
import { act } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { loadRunInspection } from "../../data/api";
import type {
  ObservatoryData,
  RunSummary,
  RuntimeState,
} from "../../data/types";
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
  createFork: vi.fn(),
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
  loadCausalTrace: vi.fn(),
}));

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
      activeDials: {},
      memoryCards: [],
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
    activeDialCount: 0,
    memoryCardCount: 0,
  };
}

const current = reading("run-current", "Current run", "c");
const referenceA = reading("run-reference-a", "Reference A", "a", "model-a");
const referenceB = reading("run-reference-b", "Reference B", "b", "model-b");
const runtime: RuntimeState = {
  status: "connected",
  runs: [summary(current), summary(referenceA), summary(referenceB)],
  engine: {
    model: "model-current",
    layerCount: 6,
    jlens: false,
    sae: false,
  },
};
const idleFork = { status: "idle" } as const;
const loadInspection = vi.mocked(loadRunInspection);

function observatory(
  overrides: Partial<React.ComponentProps<typeof Observatory>> = {},
) {
  return (
    <Observatory
      data={current}
      runtime={runtime}
      inspectorOpen
      runStatus="idle"
      forkState={idleFork}
      onSelectRun={() => {}}
      onFork={() => {}}
      {...overrides}
    />
  );
}

beforeEach(() => {
  location.hash = "#/scope";
  loadInspection.mockReset();
  loadInspection.mockImplementation(async (runId) => {
    if (runId === current.id) return current;
    if (runId === referenceA.id) return referenceA;
    if (runId === referenceB.id) return referenceB;
    throw new Error(`unknown test run ${runId}`);
  });
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

  test("selection only changes state; the explicit fork button alone executes", async () => {
    const onFork = vi.fn();
    const onStateChange = vi.fn((state: ScopeSelectionState) => {
      history.replaceState(null, "", serializeScopeUrl(current.id, state));
    });
    const user = userEvent.setup();

    render(observatory({
      initialState: { view: "trace", token: 0, layer: 0 },
      onFork,
      onStateChange,
    }));

    const tape = screen.getByRole("listbox", { name: "Output tokens" });
    await user.click(within(tape).getAllByRole("option")[2]);
    expect(onFork).not.toHaveBeenCalled();
    expect(parseScopeUrl(location.hash)?.state.token).toBe(2);

    await user.click(screen.getByRole("button", { name: /alternate/i }));
    expect(onFork).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "FORK RUN" }));
    expect(onFork).toHaveBeenCalledTimes(1);
    // The candidate's recorded numeric token id rides along so the gateway can attempt the exact
    // execution-fork path directly instead of falling back to a piece-text match -- see
    // ObservatoryProps.onFork's doc comment and docs/EXECUTION_FORK_CONTRACT.md.
    expect(onFork).toHaveBeenCalledWith(2, "alternate", 4242);
  });
});

describe("Fork outcome panel", () => {
  test("exact_execution_fork reads as the strong result and shows its exactness facts", () => {
    render(observatory({
      forkState: {
        status: "success",
        parentId: current.id,
        childId: "child-1",
        note: "exact execution fork: the worker restored its exact recorded KV state and applied the "
          + "forced token there directly on its token id -- no text splice, nothing to retokenize",
        outcome: {
          kind: "exact_execution_fork",
          reasons: [{
            code: "exact_preconditions_met",
            message: "an exact checkpoint was captured and its intervention completed",
          }],
          exactness: {
            regime: "generated_token_live_kv",
            source: "live_kv",
            proofStatus: "confirmed",
            truncateTo: 42,
          },
          unchangedControl: {
            required: true,
            status: "matched",
            result: {
              status: "matched",
              exactMatch: true,
              note: "parent suffix token ids and text matched exactly",
            },
          },
          intervention: {
            type: "force_token",
            tokenId: 4242,
            tokenPiece: "alternate",
            restoreMode: "live_kv_truncated",
          },
          executionId: "fork_exec_abc123",
        },
      },
    }));

    expect(screen.getByText("EXACT EXECUTION FORK")).toBeInTheDocument();
    expect(screen.getByText(/no text splice, nothing to retokenize/i, {
      selector: ".fork-outcome-summary",
    })).toBeInTheDocument();
    expect(screen.getByText("GENERATED TOKEN LIVE KV")).toBeInTheDocument();
    expect(screen.getByText("LIVE KV TRUNCATED")).toBeInTheDocument();
    expect(screen.getByText('FORCE TOKEN → "alternate" (id 4242)')).toBeInTheDocument();
    expect(screen.getByText("MATCHED · EXACT MATCH")).toBeInTheDocument();
    expect(screen.getByText("CONFIRMED")).toBeInTheDocument();
    expect(screen.getByText("CHILD child-1")).toBeInTheDocument();
  });

  test("reconstructed_replay reads as visibly weaker and names the retokenization risk", () => {
    render(observatory({
      forkState: {
        status: "success",
        parentId: current.id,
        childId: "child-2",
        note: "greedy continuation (sample=false): a deterministic what-if",
        outcome: {
          kind: "reconstructed_replay",
          reasons: [{
            code: "checkpoint_not_supplied",
            message: "no exact checkpoint was supplied; the eligible path explicitly reconstructs text",
          }],
          exactness: {
            regime: "reconstructed_text",
            source: "text_retokenization",
            proofStatus: "not_applicable",
          },
          unavoidableDifferences: [
            "kv_state_not_restored",
            "sampler_state_reinitialized",
            "prompt_prefix_retokenized",
            "batch_shape_not_preserved",
          ],
          retokenized: true,
        },
      },
    }));

    expect(screen.getByText("RECONSTRUCTED REPLAY")).toBeInTheDocument();
    expect(screen.getByText("RETOKENIZED")).toBeInTheDocument();
    expect(screen.getByText(/BPE token boundaries can shift/i)).toBeInTheDocument();
    expect(screen.getByText(/NOT guaranteed to run on the exact recorded token ids/)).toBeInTheDocument();
    expect(screen.getByText("KV STATE NOT RESTORED")).toBeInTheDocument();
    expect(screen.getByText("SAMPLER STATE REINITIALIZED")).toBeInTheDocument();
    // Never styled as though it were the strong outcome: no exactness metric list, no exact badge text.
    expect(screen.queryByText("EXACT EXECUTION FORK")).not.toBeInTheDocument();
  });

  test("unavailable shows the gateway's typed reason instead of a generic failure", () => {
    render(observatory({
      forkState: {
        status: "unavailable",
        parentId: current.id,
        outcome: {
          kind: "unavailable",
          reasons: [{
            code: "checkpoint_expired",
            message: "the referenced checkpoint has expired or been evicted",
          }],
        },
      },
    }));

    expect(screen.getByText("FORK UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("CHECKPOINT EXPIRED")).toBeInTheDocument();
    expect(screen.getByText("the referenced checkpoint has expired or been evicted")).toBeInTheDocument();
    // No child was created: nothing to compare, no generic "fork failed" copy.
    expect(screen.queryByText(/^CHILD /)).not.toBeInTheDocument();
    expect(screen.queryByText(/fork failed/i)).not.toBeInTheDocument();
  });
});
