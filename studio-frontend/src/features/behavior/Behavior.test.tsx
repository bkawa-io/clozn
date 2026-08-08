import { describe, expect, test, vi } from "vitest";
import { render, screen, waitFor } from "../../test/render";
import type { RuntimeState } from "../../data/types";
import { Behavior } from "./Behavior";

const runtime: RuntimeState = {
  status: "connected",
  runs: [],
  engine: { model: "model.gguf", jlens: false, sae: false },
};

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200 });
}

describe("Behavior", () => {
  test("opens the one-shot retry surface by default and has no durable Teach Once / Corrections module", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      switch (String(input)) {
        case "/steer/axes": return json({ axes: [] });
        case "/sampling/mode": return json({ sampling: false });
        default: throw new Error(`unexpected request ${String(input)}`);
      }
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Behavior runtime={runtime} inspectorOpen={false} />);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Fix this answer" })).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /ONE-SHOT RETRIES/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: /CORRECTIONS/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Corrections" })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => String(input).startsWith("/corrections"))).toBe(false);
  });

  test("has no dependency on the retired Profile API: no request, no nav, dials still work", async () => {
    // No /profiles/* handler at all -- Behavior must never ask for it. If it did, this mock throws
    // and the test fails loudly rather than silently degrading.
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      switch (String(input)) {
        case "/steer/axes": return json({
          axes: [{ name: "concise", poles: ["verbose", "concise"], value: 0.4, max: 1.5, calibrated: true }],
        });
        case "/sampling/mode": return json({ sampling: false });
        default: throw new Error(`unexpected request ${String(input)}`);
      }
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Behavior runtime={runtime} inspectorOpen={false} />);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Fix this answer" })).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /PROFILES/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/Active profile/i)).not.toBeInTheDocument();

    await screen.getByRole("button", { name: /TONE DIALS/ }).click();
    await waitFor(() => expect(screen.getByRole("heading", { name: "Tone dials" })).toBeInTheDocument());
    expect(screen.getAllByText("concise").length).toBeGreaterThan(0);

    expect(fetchMock.mock.calls.some(([input]) => String(input).startsWith("/profiles"))).toBe(false);
  });

  test("has no persistent Guard default UI: no /guard/mode request, Sampling still works", async () => {
    // No /guard/mode handler at all -- Behavior must never ask for it. If it did, this mock throws and
    // the test fails loudly rather than silently degrading. The disposition guard is now a per-request
    // `clozn_guard` intervention (see docs/CAPABILITIES.md); there is nothing left to persist here.
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      switch (String(input)) {
        case "/steer/axes": return json({ axes: [] });
        case "/sampling/mode": return json({ sampling: true, sample_temperature: 0.7, sample_top_p: 0.9,
                                             sample_top_k: 40, sample_repeat_penalty: 1.1 });
        default: throw new Error(`unexpected request ${String(input)}`);
      }
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<Behavior runtime={runtime} inspectorOpen={false} />);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Fix this answer" })).toBeInTheDocument());
    await screen.getByRole("button", { name: /RUNTIME DEFAULTS/ }).click();
    await waitFor(() => expect(screen.getByText("DECODING")).toBeInTheDocument());
    expect(screen.queryByText(/GUARD/)).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => String(input).startsWith("/guard/mode"))).toBe(false);
  });
});
