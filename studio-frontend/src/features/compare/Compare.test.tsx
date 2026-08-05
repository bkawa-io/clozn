import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { loadRunInspection } from "../../data/api";
import type { ObservatoryData, RunSummary, RuntimeState } from "../../data/types";
import { render, screen, waitFor, within } from "../../test/render";
import { Compare } from "./Compare";

vi.mock("../../data/api", () => ({ loadRunInspection: vi.fn() }));

function reading(id: string, label: string, tokenPrefix: string): ObservatoryData {
  return {
    id,
    label,
    model: "model-a",
    quant: "Q5_K_M",
    createdAt: "2026-08-04T12:00:00Z",
    duration: "1.0 s",
    mode: "run",
    tokens: [0, 1].map((index) => ({
      text: `${tokenPrefix}${index}`,
      entropy: 0.1 + index / 10,
      confidence: 0.9 - index / 10,
    })),
    candidates: [],
    sources: [],
    configuration: { activeDials: {}, memoryCards: [], adapters: [], changes: [] },
  };
}

function summary(data: ObservatoryData): RunSummary {
  return {
    id: data.id,
    label: data.label,
    prompt: "shared prompt",
    response: "shared response",
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

const a = reading("run-a", "Reference run", "a");
const b = reading("run-b", "Candidate run", "b");
const runtime: RuntimeState = {
  status: "connected",
  runs: [summary(a), summary(b)],
};

function diffBody() {
  return {
    schema_version: "clozn.run-diff.v1",
    run_a: a.id,
    run_b: b.id,
    summary_axes: {
      model: { status: "unchanged" },
      adapter: { status: "unchanged" },
      template: { status: "unchanged" },
      context: { status: "unavailable", note: "The older run has no retained context receipt." },
      sampling: { status: "changed" },
      engine: { status: "unchanged" },
      tool_parse: { status: "unchanged" },
      output: { status: "changed" },
    },
    differences: [
      {
        dimension: "context.delivered.messages",
        kind: "unavailable",
        rank: 0,
        evidence: [],
        note: "The older run has no retained context receipt.",
      },
      {
        dimension: "generation.temperature",
        kind: "changed",
        rank: 1,
        evidence: [],
        value_a: 0,
        value_b: 0.7,
      },
      {
        dimension: "output.tool_parse",
        kind: "diff_failed",
        rank: 2,
        evidence: [],
        note: "The recorded tool output could not be decoded for this dimension.",
      },
    ],
    findings: [{
      classification: "sampling_changed",
      status: "observed",
      summary: "Sampling settings changed.",
      dimensions: ["generation.temperature"],
    }],
  };
}

function plannedBody() {
  return {
    schema_version: "clozn.run-change-test.v1",
    run_a: a.id,
    run_b: b.id,
    status: "planned",
    dry_run: true,
    match_criterion: { kind: "exact_output", note: "Exact output only." },
    budget: { max_runs: 4, max_seconds: 120, runs_used: 0, duration_ms: 0, remaining_runs: 4, remaining_seconds: 120 },
    tests: [{
      kind: "context",
      status: "not_run",
      ran: false,
      runs_used: 0,
      duration_ms: 0,
      budget: { max_runs: 4, max_seconds: 120, runs_used: 0, duration_ms: 0, remaining_runs: 4, remaining_seconds: 120 },
      evidence: [],
      arms: [],
      stop_reason: "planned",
      reason: "planned only; no model run was started",
    }],
    summary: { classification: "context", causally_supported: [], entangled: false },
  };
}

function completedBody() {
  return {
    ...plannedBody(),
    status: "completed",
    dry_run: false,
    budget: { max_runs: 4, max_seconds: 120, runs_used: 2, duration_ms: 250, remaining_runs: 2, remaining_seconds: 119.75 },
    tests: [{
      ...plannedBody().tests[0],
      status: "causally_supported",
      ran: true,
      runs_used: 2,
      stop_reason: undefined,
      reason: "control reproduced the candidate and the context treatment recovered the baseline",
      evidence: [{ run_id: "child-control", arm: "control" }, { run_id: "child-treatment", arm: "treatment" }],
    }],
    summary: { classification: "context", causally_supported: ["context"], entangled: false },
  };
}

const inspect = vi.mocked(loadRunInspection);

beforeEach(() => {
  inspect.mockImplementation(async (id) => id === a.id ? a : b);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Compare", () => {
  test("uses the ranked paired delta as the primary statement and preserves an unavailable dimension", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(diffBody()), { status: 200 })));

    render(<Compare runtime={runtime} initialA={a.id} initialB={b.id} inspectorOpen />);

    const delta = await screen.findByTestId("paired-delta");
    const axes = within(delta).getByTestId("paired-delta-summary-axes");
    expect(within(axes).getAllByRole("listitem")).toHaveLength(8);

    const unavailableRow = delta.querySelector('[data-delta-kind="unavailable"]') as HTMLElement;
    expect(unavailableRow).toBeTruthy();
    expect(within(unavailableRow).getByText("The older run has no retained context receipt.")).toBeInTheDocument();
    expect(within(unavailableRow).getByRole("img", {
      name: "Unavailable -- The older run has no retained context receipt.",
    })).toBeInTheDocument();
    const failedRow = delta.querySelector('[data-delta-kind="diff_failed"]') as HTMLElement;
    expect(within(failedRow).getByRole("img", {
      name: "Diff failed -- The recorded tool output could not be decoded for this dimension.",
    })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show token alignment" })).toHaveAttribute("aria-expanded", "false");
  });

  test("does not plan or execute a controlled test until its explicit preview and run clicks", async () => {
    const requests: Array<{ method: string; body?: string }> = [];
    const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const body = typeof init?.body === "string" ? init.body : undefined;
      requests.push({ method, body });
      if (method === "GET") return new Response(JSON.stringify(diffBody()), { status: 200 });
      if (body && JSON.parse(body).plan === true) return new Response(JSON.stringify(plannedBody()), { status: 200 });
      return new Response(JSON.stringify(completedBody()), { status: 200 });
    });
    vi.stubGlobal("fetch", fetch);
    const user = userEvent.setup();

    render(<Compare runtime={runtime} initialA={a.id} initialB={b.id} inspectorOpen />);

    await screen.findByRole("button", { name: "Preview change test" });
    expect(requests).toEqual([{ method: "GET", body: undefined }]);

    await user.click(screen.getByRole("button", { name: "Preview change test" }));
    await screen.findByRole("button", { name: "Run controlled test" });
    expect(requests).toHaveLength(2);
    expect(JSON.parse(requests[1].body ?? "{}")).toMatchObject({ a: a.id, b: b.id, plan: true });

    await user.click(screen.getByRole("button", { name: "Run controlled test" }));
    await waitFor(() => expect(requests).toHaveLength(3));
    expect(JSON.parse(requests[2].body ?? "{}")).toMatchObject({ a: a.id, b: b.id, tests: ["context"] });
    expect(screen.getByText("Causally supported: context.")).toBeInTheDocument();
  });
});
