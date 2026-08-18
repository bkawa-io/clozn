import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import type { RuntimeState } from "../../data/types";
import { render, screen, within } from "../../test/render";

vi.mock("./Lens", () => ({
  Lens: ({ initialRunId }: { initialRunId?: string }) => <section><h2>Linked reader</h2><p>{initialRunId}</p></section>,
}));

vi.mock("./ReceivedContext", () => ({
  ReceivedContext: () => <section><h2>Delivery evidence</h2></section>,
}));

vi.mock("./ContextReceipt", () => ({
  ContextReceipt: () => <section><h2>Context receipt</h2></section>,
}));

vi.mock("./WhatMattered", () => ({
  WhatMattered: () => <section><h2>Influence evidence</h2></section>,
}));

vi.mock("./DiagnosisRepair", () => ({
  DiagnosisRepair: () => <section><h2>Why evidence</h2></section>,
}));

vi.mock("./ClaimVerification", () => ({
  ClaimVerification: () => <section><h2>Claim evidence</h2></section>,
}));

vi.mock("./SecondOpinion", () => ({
  SecondOpinion: () => <section><h2>Second opinion evidence</h2></section>,
}));

vi.mock("./InvestigationExperiment", () => ({
  InvestigationExperiment: () => <section><h2>Passage experiment</h2></section>,
}));

vi.mock("./TimeMachine", () => ({
  TimeMachine: () => <section><h2>Time machine evidence</h2></section>,
}));

vi.mock("../diagnostics/RunDiagnostics", async () => {
  const { EvidenceMark } = await vi.importActual<typeof import("../../components/EvidenceMark")>("../../components/EvidenceMark");
  return {
    RunTimingInstrument: () => (
      <section>
        <h2>Recorded performance</h2>
        <EvidenceMark
          variant="chip"
          state="unavailable"
          label="Performance artifact unavailable"
          reason="The timing trace request failed for this run."
        />
      </section>
    ),
  };
});

vi.mock("../../panels/scope", () => ({
  ScopePanel: () => <section><h2>Mechanism workbench</h2></section>,
}));

import { RunReader } from "./RunReader";

const runtime: RuntimeState = {
  status: "connected",
  runs: [{
    id: "run-alpha",
    label: "Run alpha",
    prompt: "prompt",
    response: "response",
    createdAt: "2026-08-04T12:00:00Z",
    source: "api",
    client: "studio",
    model: "model-alpha",
    substrate: "local",
    duration: "120 ms",
    finishReason: "stop",
    flags: [],
    warningCount: 0,
  }],
};

describe("RunReader", () => {
  test("renders the complete section navigation and lands on the linked reader", () => {
    render(<RunReader runtime={runtime} initialRunId="run-alpha" />);

    const nav = screen.getByRole("navigation", { name: "Run sections" });
    expect(within(nav).getByRole("button", { name: "Read" })).toHaveAttribute("aria-current", "page");
    for (const label of [
      "What it received", "What was sent", "What mattered", "Why", "Claims", "Second opinion",
      "Without this passage", "Timing", "Time machine", "Mechanism", "The record",
    ]) {
      expect(within(nav).getByRole("button", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByRole("heading", { name: "Linked reader" })).toBeInTheDocument();
  });

  test("uses section navigation and the deterministic question router without generating a reply", async () => {
    const user = userEvent.setup();
    render(<RunReader runtime={runtime} initialRunId="run-alpha" />);

    await user.click(screen.getByRole("button", { name: "Timing" }));
    expect(screen.getByRole("heading", { name: "Recorded performance" })).toBeInTheDocument();
    expect(location.hash).toBe("#/runs/run-alpha?section=timing");

    await user.type(screen.getByRole("textbox", { name: "ROUTE A QUESTION" }), "why did this happen?");
    await user.click(screen.getByRole("button", { name: "ROUTE" }));
    expect(screen.getByRole("heading", { name: "Why evidence" })).toBeInTheDocument();
    expect(location.hash).toBe("#/runs/run-alpha?section=why");
  });

  test("contains an instrument-local absence instead of removing the reader or its other sections", async () => {
    const user = userEvent.setup();
    render(<RunReader runtime={runtime} initialRunId="run-alpha" initialSection="timing" />);

    expect(screen.getByRole("img", {
      name: "Performance artifact unavailable -- The timing trace request failed for this run.",
    })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Run sections" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Read" }));
    expect(screen.getByRole("heading", { name: "Linked reader" })).toBeInTheDocument();
  });
});
