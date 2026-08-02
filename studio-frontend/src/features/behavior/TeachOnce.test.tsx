import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { render, screen, waitFor } from "../../test/render";
import { TeachOnce } from "./TeachOnce";
import type { Correction } from "../../data/corrections";

const draft: Correction = {
  schema_version: "clozn.correction.v1",
  id: "corr_0123456789abcdef01234567",
  scope: { kind: "session", value: "session-a" },
  type: "style",
  content: "Use short paragraphs.",
  content_hash: "0123456789abcdef",
  enabled: false,
  created_ts: 1,
  created_at: "2026-08-02T00:00:00",
};

describe("Teach Once", () => {
  test("keeps a draft inert until explicit confirmation", async () => {
    const user = userEvent.setup();
    let current = [] as typeof draft[];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/corrections" && method === "GET") {
        return new Response(JSON.stringify({ schema_version: "clozn.correction.v1", corrections: current }), { status: 200 });
      }
      if (url === "/corrections" && method === "POST") {
        current = [draft];
        return new Response(JSON.stringify(draft), { status: 201 });
      }
      if (url.endsWith(`/corrections/${draft.id}/confirm`)) {
        current = [{ ...draft, enabled: true, confirmed_ts: 2 }];
        return new Response(JSON.stringify({ correction: current[0] }), { status: 200 });
      }
      throw new Error(`unexpected request ${method} ${url}`);
    }));

    render(<TeachOnce />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());
    await user.type(screen.getByPlaceholderText("explicit identifier"), "session-a");
    await user.type(screen.getByPlaceholderText("e.g. Answer in short paragraphs."), "Use short paragraphs.");
    await user.click(screen.getByRole("button", { name: "SAVE DRAFT" }));
    await waitFor(() => expect(screen.getByText("DRAFT")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "CONFIRM" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "CONFIRM" }));
    await waitFor(() => expect(screen.getByText("CONFIRMED")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "DISABLE" })).toBeInTheDocument();
  });

  test("requires an explicit run pair before verification can promote a draft", async () => {
    const user = userEvent.setup();
    let current = [draft];
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/corrections" && method === "GET") {
        return new Response(JSON.stringify({ schema_version: "clozn.correction.v1", corrections: current }), { status: 200 });
      }
      if (url.endsWith(`/corrections/${draft.id}/verify`)) {
        current = [{ ...draft, enabled: true, confirmed_ts: 3 }];
        return new Response(JSON.stringify({
          schema_version: "clozn.correction-verification.v1",
          correction_id: draft.id,
          target_run_id: "run_target",
          child_run_id: "run_child",
          match_criterion: "exact_output",
          comparison: { available: true, matched: false },
          verification: "passed",
          promoted: true,
          reason: "child differs",
          correction: current[0],
        }), { status: 200 });
      }
      throw new Error(`unexpected request ${method} ${url}`);
    });

    render(<TeachOnce />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "VERIFY + PROMOTE" })).toBeDisabled();
    await user.type(screen.getByLabelText("TARGET FAILURE RUN ID"), "run_target");
    await user.type(screen.getByLabelText("CHILD RETRY RUN ID"), "run_child");
    expect(screen.getByRole("button", { name: "VERIFY + PROMOTE" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "VERIFY + PROMOTE" }));
    await waitFor(() => expect(screen.getByText("CONFIRMED")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(`/corrections/${draft.id}/verify`, expect.objectContaining({ method: "POST" }));
  });
});
