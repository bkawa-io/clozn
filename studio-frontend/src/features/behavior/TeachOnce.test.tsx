import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { render, screen, waitFor, within } from "../../test/render";
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
  created_at: "2026-08-02T00:00:00Z",
};

const persistedRuns = [
  { id: "run_target", label: "Recorded target · run_target" },
  { id: "run_child", label: "Recorded child · run_child" },
  { id: "run_resolution", label: "Recorded resolution · run_resolution" },
];

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

function correctionList(corrections: Correction[]) {
  return { schema_version: "clozn.correction-list.v1", corrections };
}

describe("Teach Once", () => {
  test("shows the five-level scope containment hierarchy instead of only a flat picker", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      if (String(input) === "/corrections") return json(correctionList([]));
      throw new Error(`unexpected request ${String(input)}`);
    }));

    render(<TeachOnce />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());

    const hierarchy = screen.getByRole("region", { name: "SCOPE CONTAINMENT" });
    expect(within(hierarchy).getByText("SESSION ⊂ CLIENT ⊂ MODEL ⊂ PROJECT ⊂ GLOBAL")).toBeInTheDocument();
    for (const label of ["SESSION", "CLIENT", "MODEL", "PROJECT", "GLOBAL"]) {
      expect(within(hierarchy).getByText(label, { exact: true })).toBeInTheDocument();
    }

    await user.selectOptions(screen.getByLabelText("SCOPE"), "project");
    const project = within(hierarchy).getByText("PROJECT", { exact: true }).closest("li");
    expect(project).toHaveAttribute("aria-current", "step");
  });

  test("keeps a draft inert until explicit confirmation and exposes its durable state facts", async () => {
    const user = userEvent.setup();
    let current: Correction[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/corrections" && method === "GET") return json(correctionList(current));
      if (url === "/corrections" && method === "POST") {
        current = [draft];
        return json(draft, 201);
      }
      if (url.endsWith(`/corrections/${draft.id}/confirm`)) {
        current = [{ ...draft, enabled: true, confirmed_ts: 2 }];
        return json({ correction: current[0] });
      }
      throw new Error(`unexpected request ${method} ${url}`);
    }));

    render(<TeachOnce />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());
    await user.type(screen.getByPlaceholderText("explicit identifier"), "session-a");
    await user.type(screen.getByPlaceholderText("e.g. Answer in short paragraphs."), "Use short paragraphs.");
    await user.click(screen.getByRole("button", { name: "SAVE DRAFT" }));

    const list = screen.getByLabelText("Saved corrections");
    await waitFor(() => expect(within(list).getByText("DRAFT", { exact: true })).toBeInTheDocument());
    const draftCard = list.querySelector('[data-correction-state="draft"]');
    expect(draftCard).not.toBeNull();
    expect(within(draftCard as HTMLElement).getByText("NOT CONFIRMED", { exact: true })).toBeInTheDocument();
    expect(within(draftCard as HTMLElement).getByText("0123456789abcdef", { exact: true })).toBeInTheDocument();
    expect(within(draftCard as HTMLElement).getByText("2026-08-02T00:00:00Z", { exact: true })).toBeInTheDocument();

    await user.click(within(draftCard as HTMLElement).getByRole("button", { name: "CONFIRM" }));
    await waitFor(() => expect(within(list).getByText("ENABLED", { exact: true })).toBeInTheDocument());
    expect(within((list.querySelector('[data-correction-state="enabled"] .behavior-correction-state') as HTMLElement)).getByText("CONFIRMED", { exact: true })).toBeInTheDocument();
  });

  test("compares two already-persisted runs without generating a retry", async () => {
    const user = userEvent.setup();
    let current: Correction[] = [draft];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/corrections" && method === "GET") return json(correctionList(current));
      if (url.endsWith(`/corrections/${draft.id}/verify`)) {
        current = [{ ...draft, enabled: true, confirmed_ts: 3 }];
        return json({
          schema_version: "clozn.correction-verification.v1",
          correction_id: draft.id,
          target_run_id: "run_target",
          child_run_id: "run_child",
          match_criterion: "exact_output",
          comparison: { available: true, matched: false },
          verification: "passed",
          promoted: true,
          reason: "The child did not reproduce the target failure.",
          correction: current[0],
        });
      }
      throw new Error(`unexpected request ${method} ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<TeachOnce runs={persistedRuns} />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());
    expect(screen.getByText(/This comparison does not create a retry/i)).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/retry"))).toBe(false);

    await user.selectOptions(screen.getByLabelText("TARGET FAILURE RUN"), "run_target");
    await user.selectOptions(screen.getByLabelText("CHILD RUN"), "run_child");
    await user.click(screen.getByRole("button", { name: "COMPARE PERSISTED RUNS + PROMOTE IF PASSED" }));

    await waitFor(() => expect(screen.getByText("Verification comparison")).toBeInTheDocument());
    expect(screen.getByText("This comparison reads two already-persisted runs. It did not generate a retry or run the model.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      `/corrections/${draft.id}/verify`,
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/retry"))).toBe(false);
  });

  test("keeps enabled and disabled corrections distinct and exposes an explicit undo", async () => {
    const user = userEvent.setup();
    const enabled = { ...draft, id: "corr_1123456789abcdef01234567", enabled: true, confirmed_ts: 4 };
    const disabled = {
      ...draft,
      id: "corr_2123456789abcdef01234567",
      scope: { kind: "global_local" as const },
      enabled: false,
      confirmed_ts: 5,
      disabled_ts: 6,
    };
    let current: Correction[] = [draft, enabled, disabled];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/corrections" && method === "GET") return json(correctionList(current));
      if (url.endsWith(`/corrections/${enabled.id}/undo`)) {
        current = [draft, { ...enabled, enabled: false }, disabled];
        return json(current[1]);
      }
      throw new Error(`unexpected request ${method} ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<TeachOnce />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());
    const list = screen.getByLabelText("Saved corrections");
    const enabledCard = list.querySelector('[data-correction-state="enabled"]') as HTMLElement;
    const disabledCard = list.querySelector('[data-correction-state="disabled"]') as HTMLElement;
    const enabledState = enabledCard.querySelector(".behavior-correction-state") as HTMLElement;
    const disabledState = disabledCard.querySelector(".behavior-correction-state") as HTMLElement;

    expect(within(enabledCard).getByText("ENABLED", { exact: true })).toBeInTheDocument();
    expect(within(enabledState).getByText("CONFIRMED", { exact: true })).toBeInTheDocument();
    expect(within(disabledState).getByText("DISABLED", { exact: true })).toBeInTheDocument();
    expect(within(disabledState).getByText("CONFIRMED", { exact: true })).toBeInTheDocument();
    expect(within(enabledCard).getByRole("button", { name: "DISABLE" })).toBeInTheDocument();
    expect(within(disabledCard).getByRole("button", { name: "ENABLE" })).toBeInTheDocument();

    await user.click(within(enabledCard).getByRole("button", { name: "UNDO LAST CHANGE" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      `/corrections/${enabled.id}/undo`,
      expect.objectContaining({ method: "POST" }),
    ));
  });

  test("shows actual stored correction resolution only after an explicit read", async () => {
    const user = userEvent.setup();
    const enabled = { ...draft, enabled: true, confirmed_ts: 4 };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url === "/corrections" && method === "GET") return json(correctionList([enabled]));
      if (url === "/runs/run_target" && method === "GET") return json({ id: "run_target" });
      if (url === "/runs/run_resolution" && method === "GET") {
        return json({
          applied_corrections: [{
            correction_id: enabled.id,
            type: "style",
            scope: { kind: "session", value: "session-a" },
            content_hash: enabled.content_hash,
          }],
          correction_conflicts: [],
        });
      }
      throw new Error(`unexpected request ${method} ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<TeachOnce runs={persistedRuns} />);
    await waitFor(() => expect(screen.getByText("READY")).toBeInTheDocument());
    expect(screen.getByRole("img", { name: /Resolution absent/i })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalledWith("/runs/run_target", expect.anything());

    await user.click(screen.getByRole("button", { name: "READ RECORDED RESOLUTION" }));
    await waitFor(() => expect(screen.getByRole("img", { name: /Resolution was not computed for it/i })).toBeInTheDocument());

    await user.selectOptions(screen.getByLabelText("Recorded run for correction resolution"), "run_resolution");
    await user.click(screen.getByRole("button", { name: "READ RECORDED RESOLUTION" }));

    await waitFor(() => expect(screen.getByText("Resolution was recorded: no conflicts were present.")).toBeInTheDocument());
    expect(within(screen.getByLabelText("Applied corrections recorded on this run")).getByText(enabled.content_hash, { exact: true })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/runs/run_resolution");
  });
});
