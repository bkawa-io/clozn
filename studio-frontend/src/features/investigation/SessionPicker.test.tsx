import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen } from "../../test/render";
import { SessionPicker } from "./SessionPicker";

/**
 * F3's entry point. Covers: the real `GET /sessions` list renders with real per-session facts (never a
 * placeholder), an empty install says so honestly, a failed request is reported rather than swallowed,
 * and the "open by id" router never fabricates a lookup -- it only ever navigates.
 */

function pathOf(request: PendingFetch): string {
  return typeof request.input === "string" ? request.input : request.input.toString();
}

function sessionFixture(id: string, overrides: Record<string, unknown> = {}) {
  return {
    id, created_at: "2023-11-14T00:00:00", created_ts: 1700000000,
    privacy: { visibility: "visible" }, materialized_from: "explicit",
    ...overrides,
  };
}

beforeEach(() => {
  location.hash = "";
});

describe("Session picker", () => {
  test("lists real sessions with their own facts, linking into the investigation view", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<SessionPicker />);

    const request = await controller.nextRequest();
    expect(pathOf(request)).toBe("/sessions?limit=100");
    controller.respondJson(request, {
      sessions: [
        sessionFixture("session_abc", { title: "Debugging the retry loop", run_count: 4, last_activity_ts: 1700000500 }),
      ],
    });

    expect(await screen.findByText("Debugging the retry loop")).toBeInTheDocument();
    expect(screen.getByText("4 TURNS")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Debugging the retry loop/ })).toHaveAttribute(
      "href", "#/sessions/session_abc/investigate",
    );
  });

  test("an install with no sessions yet says so honestly, not a blank list", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<SessionPicker />);

    const request = await controller.nextRequest();
    controller.respondJson(request, { sessions: [] });

    expect(await screen.findByText("No sessions recorded yet.")).toBeInTheDocument();
  });

  test("a failed list request is reported, never silently swallowed", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<SessionPicker />);

    const request = await controller.nextRequest();
    controller.respondJson(request, { error: "boom" }, { status: 500 });

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });

  test("opening a session by id only ever routes -- it never performs its own lookup", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<SessionPicker />);

    const request = await controller.nextRequest();
    controller.respondJson(request, { sessions: [] });
    await screen.findByText("No sessions recorded yet.");

    const before = controller.requests.length;
    await user.type(screen.getByLabelText("OPEN A SESSION BY ID"), "thread-42");
    await user.click(screen.getByRole("button", { name: "OPEN" }));

    expect(location.hash).toBe("#/sessions/thread-42/investigate");
    // Routing only -- no second fetch fired on submit.
    expect(controller.requests.length).toBe(before);
  });
});
