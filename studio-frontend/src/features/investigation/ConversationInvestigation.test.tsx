import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, waitFor, within } from "../../test/render";
import { ConversationInvestigation } from "./ConversationInvestigation";

/**
 * Covers F3's own honesty requirements (see this file's sibling doc comment in
 * ConversationInvestigation.tsx):
 *   - pagination never pretends to be complete -- a session with more pages says so plainly;
 *   - "first suspicious turn" renders ONLY from a real backend candidate, and an absent kind states
 *     that plainly rather than leaving blank space;
 *   - D1's five-value status vocabulary is never collapsed, even when every count is zero;
 *   - branches never flatten into the linear turn list;
 *   - a session that genuinely does not exist (404) is reported as exactly that, not a generic failure.
 */

function pathOf(request: PendingFetch): string {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname + request.input.search
      : request.input.url;
}

function turnFixture(runId: string, overrides: Record<string, unknown> = {}) {
  return {
    run_id: runId,
    recorded_ts: 1700000000,
    created_at: "2023-11-14T00:00:00",
    source: "chat",
    client: "web",
    model: "test-model",
    prompt_summary: `prompt for ${runId}`,
    response_summary: `response for ${runId}`,
    redacted: false,
    cumulative: { turn_count: 1, duration_ms_total: 100, prompt_tokens_total: 10, generated_tokens_total: 5 },
    diagnostic_highlights: {
      findings: [],
      status_counts: { finding: 0, not_observed: 10, unavailable: 0, pending: 2, suppressed: 0 },
    },
    ...overrides,
  };
}

function pageFixture(sessionId: string, turns: unknown[], overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.session-trace.v1",
    generated_at: "2023-11-14T00:00:00Z",
    session_id: sessionId,
    session: { id: sessionId, privacy: {} },
    page: { cursor: null, next_cursor: null, limit: 50, count: turns.length },
    turns,
    branches: [],
    totals_through_this_page: {
      turn_count: turns.length, duration_ms_total: 100, prompt_tokens_total: 10, generated_tokens_total: 5,
    },
    diagnostic_rule_registry: [],
    first_went_wrong_candidates: [],
    ...overrides,
  };
}

beforeEach(() => {
  location.hash = "";
});

describe("Conversation investigation view", () => {
  test("renders the first page and states plainly that a further page exists", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    expect(pathOf(request)).toBe("/sessions/session_abc/trace?limit=50");
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1")], {
      page: { cursor: null, next_cursor: "CURSOR1", limit: 50, count: 1 },
    }));

    expect(await screen.findByText("prompt for run-1")).toBeInTheDocument();
    // Exact match on the mode chip -- a regex here would also hit the banner paragraph below, which
    // repeats the phrase in a longer sentence.
    expect(screen.getByText("PARTIAL TRACE")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /LOAD NEXT 50 TURNS/ })).toBeInTheDocument();
  });

  test("loading more appends turns rather than replacing them, and fetches with the real cursor", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<ConversationInvestigation sessionId="session_abc" />);

    const first = await controller.nextRequest();
    controller.respondJson(first, pageFixture("session_abc", [turnFixture("run-1")], {
      page: { cursor: null, next_cursor: "CURSOR1", limit: 50, count: 1 },
    }));
    await screen.findByText("prompt for run-1");

    await user.click(screen.getByRole("button", { name: /LOAD NEXT 50 TURNS/ }));
    const second = await controller.nextRequest();
    expect(pathOf(second)).toBe("/sessions/session_abc/trace?cursor=CURSOR1&limit=50");
    controller.respondJson(second, pageFixture("session_abc", [turnFixture("run-2")], {
      page: { cursor: "CURSOR1", next_cursor: null, limit: 50, count: 1 },
    }));

    expect(await screen.findByText("prompt for run-2")).toBeInTheDocument();
    expect(screen.getByText("prompt for run-1")).toBeInTheDocument();
    expect(screen.queryByText(/PARTIAL TRACE/)).not.toBeInTheDocument();
    expect(screen.getByText(/this is the full session/)).toBeInTheDocument();
  });

  test("branches never flatten into the linear turn list", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1")], {
      branches: [{ parent_run_id: "run-1", children: [{ id: "branch-child-1", source: "fork" }] }],
    }));
    await screen.findByText("prompt for run-1");

    const timeline = screen.getByRole("list", { name: /Session turns/ });
    expect(within(timeline).getAllByRole("listitem")).toHaveLength(1);

    const branchSection = screen.getByRole("heading", { name: "Branches" }).closest("section")!;
    expect(within(branchSection).getByRole("link", { name: "OPEN" })).toHaveAttribute(
      "href", "#/runs/branch-child-1",
    );
    expect(within(branchSection).getByText("1 BRANCH")).toBeInTheDocument();
  });

  test("first suspicious turn only ever shows a real backend candidate -- absent kinds say so plainly", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1"), turnFixture("run-2")], {
      first_went_wrong_candidates: [
        {
          kind: "first_finding", run_id: "run-1", recorded_ts: 1700000000,
          summary: "diagnostic rule(s) reported a finding on this turn: R03", rule_ids: ["R03"],
        },
      ],
    }));
    await screen.findByText("prompt for run-1");

    expect(screen.getByText(/diagnostic rule\(s\) reported a finding on this turn: R03/)).toBeInTheDocument();
    // The other two candidate kinds are honestly absent -- never blank, never invented.
    expect(screen.getByText(
      "No identity or generation-setting drift was found between consecutive turns in this session.",
    )).toBeInTheDocument();
    expect(screen.getByText(
      "No turn in this session recorded an error or a cancelled/failed termination.",
    )).toBeInTheDocument();

    const turnEl = document.getElementById("investigation-turn-run-1")!;
    const scrollSpy = vi.spyOn(turnEl, "scrollIntoView").mockImplementation(() => {});
    // Scoped to the candidate card's own button -- the timeline footer also offers a "jump to first
    // flagged turn" shortcut with overlapping wording (see the acceptance criterion this exercises:
    // navigating from the final answer back to the first divergent turn), so an unscoped name match
    // would be ambiguous between the two, real, honest affordances.
    await user.click(screen.getByRole("button", { name: "JUMP TO run-1" }));
    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
    expect(within(turnEl).getByText("Open this turn in Debug")).toBeInTheDocument();
  });

  test("when no rule ever fired in this session, the candidate card says so -- never a fabricated arrow", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1")]));
    await screen.findByText("prompt for run-1");

    expect(screen.getByText("No turn in this session triggered a diagnostic rule finding.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /JUMP TO/ })).not.toBeInTheDocument();
  });

  test("D1's five-value status vocabulary is never collapsed, even at zero", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1", {
      diagnostic_highlights: {
        findings: [],
        status_counts: { finding: 0, not_observed: 12, unavailable: 0, pending: 2, suppressed: 0 },
      },
    })]));
    await screen.findByText("prompt for run-1");

    await user.click(screen.getByRole("button", { expanded: false }));
    expect(await screen.findByText("0 FINDING")).toBeInTheDocument();
    expect(screen.getByText("12 NOT OBSERVED")).toBeInTheDocument();
    expect(screen.getByText("0 UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("2 PENDING")).toBeInTheDocument();
    expect(screen.getByText("0 SUPPRESSED")).toBeInTheDocument();
  });

  test("the session's first turn states there is no earlier turn, rather than a fabricated comparison", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1")]));
    await screen.findByText("prompt for run-1");

    await user.click(screen.getByRole("button", { expanded: false }));
    expect(await screen.findByText(
      "This is the first recorded turn of the session -- there is no earlier turn to compare against.",
    )).toBeInTheDocument();
  });

  test("a session that genuinely does not exist is reported as exactly that, not a generic failure", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<ConversationInvestigation sessionId="ghost-session" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, { error: "session not found" }, { status: 404 });

    expect(await screen.findByText(/was not found/)).toBeInTheDocument();
  });

  test("a redacted turn is flagged, never rendered as if it were ordinary", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1", { redacted: true })]));
    await screen.findByText("prompt for run-1");

    expect(screen.getByText("REDACTED")).toBeInTheDocument();
  });

  test("staging two turns reuses the existing A/B compare surface rather than a second implementation", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<ConversationInvestigation sessionId="session_abc" />);

    const request = await controller.nextRequest();
    controller.respondJson(request, pageFixture("session_abc", [turnFixture("run-1"), turnFixture("run-2")]));
    await screen.findByText("prompt for run-1");

    const toggles = screen.getAllByRole("button", { expanded: false });
    await user.click(toggles[0]);
    await user.click(toggles[1]);

    const turn1 = document.getElementById("investigation-turn-run-1")!;
    const turn2 = document.getElementById("investigation-turn-run-2")!;
    await user.click(within(turn1).getByRole("button", { name: "STAGE A" }));
    await user.click(within(turn2).getByRole("button", { name: "STAGE B" }));

    expect(screen.getByRole("link", { name: "COMPARE" })).toHaveAttribute("href", "#/compare/run-1/run-2");
  });
});
