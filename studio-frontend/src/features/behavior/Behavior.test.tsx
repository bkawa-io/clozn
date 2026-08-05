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
  test("opens the durable corrections surface by default while retaining one-shot retries as a module", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      switch (String(input)) {
        case "/corrections": return json({ schema_version: "clozn.correction-list.v1", corrections: [] });
        case "/steer/axes": return json({ axes: [] });
        case "/sampling/mode": return json({ sampling: false });
        case "/guard/mode": return json({ enabled: false });
        case "/profiles/list": return json({ profiles: [] });
        default: throw new Error(`unexpected request ${String(input)}`);
      }
    }));

    render(<Behavior runtime={runtime} inspectorOpen={false} />);

    await waitFor(() => expect(screen.getByRole("heading", { name: "Corrections" })).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /CORRECTIONS/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /ONE-SHOT RETRIES/ })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Fix this answer" })).not.toBeInTheDocument();
  });
});
