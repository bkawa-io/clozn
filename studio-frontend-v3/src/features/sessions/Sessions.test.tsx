import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SessionsSurface } from "./Sessions";

const sessionId = "session_0123456789abcdef01234567";
const session = {
  schema_version: "clozn.session.v1",
  id: sessionId,
  created_ts: 10,
  created_at: "2026-08-25T00:00:00",
  privacy: { visibility: "visible" },
  materialized_from: "explicit",
  title: "Recorded exchange",
  run_count: 24,
  turn_count: 20,
  last_activity_ts: 20,
  preview: { run_id: "run_a", prompt_summary: "First prompt", response_summary: "First response" },
};

function traceDocument(turnCount = 2) {
  const totals = { turn_count: turnCount, duration_ms_total: 15, prompt_tokens_total: 4, generated_tokens_total: 3 };
  const diagnostics = { findings: [], status_counts: { finding: 0, not_observed: 1, unavailable: 0, pending: 0, suppressed: 0 } };
  const turns = Array.from({ length: turnCount }, (_, index) => {
    const runId = index === 0 ? "run_a" : index === 1 ? "run_b" : `run_${index + 1}`;
    return {
      run_id: runId,
      recorded_ts: 10 + index,
      created_at: `2026-08-25T00:00:${String(index + 1).padStart(2, "0")}`,
      source: "cli",
      client: "tester",
      model: "qwen",
      prompt_summary: index === 0 ? "First prompt" : index === 1 ? "Second prompt" : `Prompt ${index + 1}`,
      response_summary: index === 0 ? "First response" : index === 1 ? "Second response" : `Response ${index + 1}`,
      redacted: false,
      timing: { duration_ms: 7 },
      cumulative: { ...totals, turn_count: index + 1, duration_ms_total: index + 1 },
      diagnostic_highlights: diagnostics,
      ...(index === 1 ? { error: "recorded failure", turn_comparison: { available: false, reason: "comparison unavailable" } } : {}),
    };
  });
  return {
    schema_version: "clozn.session-trace.v1",
    generated_at: "2026-08-25T00:00:00Z",
    session_id: sessionId,
    session: { id: sessionId, title: "Recorded exchange", privacy: { visibility: "visible" } },
    page: { cursor: null, next_cursor: null, limit: 100, count: turnCount },
    turns,
    branches: turnCount > 1 ? [{ parent_run_id: "run_b", children: [{ id: "run_b_branch", source: "branch" }] }] : [],
    totals_through_this_page: totals,
    diagnostic_rule_registry: [{ rule_id: "R01", rule_name: "rule" }],
    first_went_wrong_candidates: [],
  };
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

function stubSessionApi(trace = traceDocument()) {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === `/sessions/${sessionId}`) return Promise.resolve(json(session));
    if (url.startsWith(`/sessions/${sessionId}/trace`)) return Promise.resolve(json(trace));
    return Promise.resolve(json({ runs: [] }));
  }));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.location.hash = "";
});

describe("Sessions surface", () => {
  it("renders a real session preview and keeps standalone runs separate", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/sessions") return Promise.resolve(json({ sessions: [session] }));
      return Promise.resolve(json({ runs: [{ id: "standalone", session_key: null, prompt_summary: "A standalone prompt" }] }));
    }));
    render(<SessionsSurface route={{ surface: "sessions" }} />);
    expect(await screen.findByText("First prompt")).toBeInTheDocument();
    expect(screen.getByText("A standalone prompt")).toBeInTheDocument();
    expect(screen.getByText("20 turns")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Conversations" })).not.toBeInTheDocument();
    expect(screen.queryByText("Runs without a session identity.")).not.toBeInTheDocument();
  });

  it("selects a turn from the keyboard, keeps focus with it, and writes its run identity to the URL", async () => {
    stubSessionApi();
    render(<SessionsSurface route={{ surface: "session", sessionId }} />);
    const secondTurn = await screen.findByRole("button", { name: /Turn 2/ });
    expect(document.querySelector(".scrubber-preview")).toBeNull();
    await userEvent.click(secondTurn);
    await waitFor(() => expect(window.location.hash).toContain("run=run_b"));
    await waitFor(() => expect(document.activeElement).toBe(secondTurn));
    await userEvent.keyboard("{ArrowUp}");
    await waitFor(() => expect(window.location.hash).toContain("run=run_a"));
    expect(screen.getByRole("link", { name: "Select Run run_b" })).toHaveAttribute("href", expect.stringContaining("run=run_b"));
    expect(screen.queryByText("Open run")).not.toBeInTheDocument();
    expect(screen.queryByText("Diagnostics")).not.toBeInTheDocument();
    expect(screen.queryByText("Previous turn")).not.toBeInTheDocument();
    expect(screen.queryByText("Cumulative time")).not.toBeInTheDocument();
    expect(screen.getByText("Failure recorded")).toBeInTheDocument();
    expect(screen.getByText("1 branch")).toBeInTheDocument();
  });

  it("previews only the nearest hovered turn and dismisses the preview at rest", async () => {
    stubSessionApi();
    render(<SessionsSurface route={{ surface: "session", sessionId }} />);
    const scrubber = await screen.findByLabelText("Conversation turns");
    vi.spyOn(scrubber, "getBoundingClientRect").mockReturnValue({ top: 0, bottom: 100, height: 100, left: 0, right: 30, width: 30, x: 0, y: 0, toJSON: () => ({}) } as DOMRect);
    fireEvent.pointerMove(scrubber, { clientY: 90 });
    expect(document.querySelector(".scrubber-preview")).toHaveTextContent("Second prompt");
    expect(document.querySelector(".scrubber-preview")).toHaveTextContent("Second response");
    fireEvent.pointerDown(scrubber, { clientY: 90 });
    await waitFor(() => expect(window.location.hash).toContain("run=run_b"));
    fireEvent.pointerLeave(scrubber);
    expect(document.querySelector(".scrubber-preview")).toBeNull();
  });

  it("keeps unavailable and malformed session responses as distinct states", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      if (String(input) === "/sessions") return Promise.reject(new Error("offline"));
      return Promise.resolve(json({ runs: [] }));
    }));
    render(<SessionsSurface route={{ surface: "sessions" }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("sessions unavailable.");
    cleanup();

    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      if (String(input) === "/sessions") return Promise.resolve(json({ sessions: [{ invalid: true }] }));
      return Promise.resolve(json({ runs: [] }));
    }));
    render(<SessionsSurface route={{ surface: "sessions" }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Malformed sessions contract.");
  });

  it("keeps a one-turn scrubber centered and renders long traces as ordinary turns", async () => {
    stubSessionApi(traceDocument(1));
    const { unmount } = render(<SessionsSurface route={{ surface: "session", sessionId }} />);
    const onlyTurn = await screen.findByRole("button", { name: "Turn 1" });
    expect(onlyTurn).toHaveStyle({ "--turn-position": "50%" });
    unmount();

    stubSessionApi(traceDocument(80));
    render(<SessionsSurface route={{ surface: "session", sessionId }} />);
    expect(await screen.findAllByRole("button", { name: /^Turn / })).toHaveLength(80);
    expect(screen.getByText("Prompt 80")).toBeInTheDocument();
    expect(screen.queryByText("run_b_branch")).not.toBeInTheDocument();
  });
});
