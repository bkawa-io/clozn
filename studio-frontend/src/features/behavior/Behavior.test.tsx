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
        case "/guard/mode": return json({ enabled: false });
        case "/profiles/list": return json({ profiles: [] });
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
});
