import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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
  run_count: 2,
  last_activity_ts: 20,
  preview: { run_id: "run_a", prompt_summary: "First prompt", response_summary: "First response" },
};

function traceDocument() {
  const totals = { turn_count: 2, duration_ms_total: 15, prompt_tokens_total: 4, generated_tokens_total: 3 };
  const diagnostics = { findings: [], status_counts: { finding: 0, not_observed: 1, unavailable: 0, pending: 0, suppressed: 0 } };
  return {
    schema_version: "clozn.session-trace.v1",
    generated_at: "2026-08-25T00:00:00Z",
    session_id: sessionId,
    session: { id: sessionId, title: "Recorded exchange", privacy: { visibility: "visible" } },
    page: { cursor: null, next_cursor: null, limit: 100, count: 2 },
    turns: [
      { run_id: "run_a", recorded_ts: 10, created_at: "2026-08-25T00:00:01", source: "cli", client: "tester", model: "qwen", prompt_summary: "First prompt", response_summary: "First response", redacted: false, timing: { duration_ms: 7 }, cumulative: { ...totals, turn_count: 1, duration_ms_total: 7 }, diagnostic_highlights: diagnostics },
      { run_id: "run_b", recorded_ts: 20, created_at: "2026-08-25T00:00:02", source: "cli", client: "tester", model: "qwen", prompt_summary: "Second prompt", response_summary: "Second response", redacted: false, error: "recorded failure", cumulative: totals, diagnostic_highlights: diagnostics, turn_comparison: { available: false, reason: "comparison unavailable" } },
    ],
    branches: [{ parent_run_id: "run_b", children: [{ id: "run_b_branch", source: "branch" }] }],
    totals_through_this_page: totals,
    diagnostic_rule_registry: [{ rule_id: "R01", rule_name: "rule" }],
    first_went_wrong_candidates: [],
  };
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

afterEach(() => {
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
    expect(screen.queryByText("Runs without a session identity.")).toBeInTheDocument();
  });

  it("selects a turn from the keyboard and writes its run identity to the URL", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `/sessions/${sessionId}`) return Promise.resolve(json(session));
      if (url.startsWith(`/sessions/${sessionId}/trace`)) return Promise.resolve(json(traceDocument()));
      return Promise.resolve(json({ runs: [] }));
    }));
    render(<SessionsSurface route={{ surface: "session", sessionId }} />);
    const secondTurn = await screen.findByRole("button", { name: /Turn 2/ });
    await userEvent.click(secondTurn);
    await waitFor(() => expect(window.location.hash).toContain("run=run_b"));
    await userEvent.keyboard("{ArrowUp}");
    await waitFor(() => expect(window.location.hash).toContain("run=run_a"));
    expect(screen.getByText("Failure recorded")).toBeInTheDocument();
    expect(screen.getByText("1 branch")).toBeInTheDocument();
  });
});
