import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { render, screen } from "../../test/render";
import { RunEventRail } from "./RunEventRail";
import { RunFrame } from "./RunFrame";

describe("RunFrame", () => {
  test("renders recorded identity, environment, evidence, warnings, lineage, and caller-owned actions", () => {
    render(
      <RunFrame
        run={{
          id: "run_0123456789abcdef",
          label: "Anchor answer",
          createdAt: "2026-08-02T16:20:00Z",
          source: "gateway",
          client: "studio",
          model: "clozn-7b",
          substrate: "metal",
          finishReason: "length",
          parentRunId: "run_parent",
          flags: ["truncated"],
          warningCount: 1,
        }}
        runtime={{ status: "connected", engine: { model: "clozn-7b", jlens: true, sae: false } }}
        performance={{ device: "M3 Max", gpuLayers: 32, samplerMode: "top_p" }}
        artifacts={[{ id: "trace", label: "Token trace", kind: "inspection" }]}
        lineage={{ children: [{ id: "run_child", label: "Forked reply" }] }}
        actions={<button type="button">OPEN TRACE</button>}
      >
        <p>Workspace content</p>
      </RunFrame>,
    );

    expect(screen.getByRole("heading", { name: "Anchor answer" })).toBeInTheDocument();
    expect(screen.getAllByText("Truncated")).toHaveLength(2);
    expect(screen.getByText("Length")).toBeInTheDocument();
    expect(screen.getByText("M3 Max")).toBeInTheDocument();
    expect(screen.getByText(/Token trace/)).toBeInTheDocument();
    expect(screen.getByText("Forked reply")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "OPEN TRACE" })).toBeInTheDocument();
    expect(screen.getByText("Workspace content")).toBeInTheDocument();
  });
});

describe("RunEventRail", () => {
  test("selects semantic events while suppressing token-by-token entries", async () => {
    const onSelectEvent = vi.fn();
    const user = userEvent.setup();
    render(
      <RunEventRail
        selectedEventId="generation"
        onSelectEvent={onSelectEvent}
        events={[
          { id: "start", label: "Run started", kind: "run-start", timestamp: "2026-08-02T16:20:00Z" },
          { id: "token-1", label: "Response token 1", kind: "token", granularity: "token" },
          { id: "generation", label: "Generation", kind: "generation", detail: "42 tokens" },
          { id: "token-batch", label: "Response token batch", kind: "token", count: 41 },
        ]}
      />,
    );

    expect(screen.getByRole("button", { name: /generation/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByText("Response token 1")).not.toBeInTheDocument();
    expect(screen.getByText("42 token events omitted from this semantic rail.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /run started/i }));
    expect(onSelectEvent).toHaveBeenCalledWith(expect.objectContaining({ id: "start" }));
  });
});
