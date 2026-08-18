import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { RunSummary, RuntimeState } from "../../data/types";
import { render, screen, within } from "../../test/render";
import { ConfidenceSparkline, Runs } from "./Runs";

vi.mock("../../data/api", () => ({
  loadRunFamily: vi.fn(),
  loadRunFacts: vi.fn(),
}));

function run(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    id: "run-one",
    label: "First recorded answer",
    prompt: "Explain the retained policy.",
    response: "The retained policy applies to the answer.",
    createdAt: "2026-08-03T10:20:30Z",
    source: "openai-compatible",
    client: "client-a",
    model: "clozn-7b-instruct",
    substrate: "metal",
    duration: "1.3 s",
    durationMs: 1300,
    finishReason: "stop_sequence",
    flags: [],
    warningCount: 0,
    ...overrides,
  };
}

function runtime(runs: RunSummary[]): RuntimeState {
  return { status: "connected", runs };
}

beforeEach(async () => {
  const api = await import("../../data/api");
  vi.mocked(api.loadRunFamily).mockResolvedValue([]);
  vi.mocked(api.loadRunFacts).mockResolvedValue({ tokenCount: 0, traceAvailable: false });
});

describe("Runs", () => {
  test("hides refinement filters behind a toggle and counts the ones that are excluding runs", async () => {
    const user = userEvent.setup();
    const first = run();
    const second = run({
      id: "run-two",
      label: "Second recorded answer",
      model: "clozn-3b-instruct",
      source: "studio",
      finishReason: "length",
      flags: ["truncated"],
      warningCount: 1,
    });

    render(<Runs runtime={runtime([first, second])} inspectorOpen={false} />);

    // Search stays in the header; everything else starts collapsed. Seven controls across the top
    // made the first thing you saw a configuration panel rather than your runs.
    expect(screen.queryByLabelText("MODEL filter")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^FILTER/ }));
    expect(screen.getByLabelText("ADAPTER filter")).toBeDisabled();
    expect(screen.getByLabelText("HAS INFLUENCE filter")).toBeDisabled();
    expect(screen.getByText(/Adapter identity is not indexed; no run is assumed to be base/i)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("MODEL filter"), "clozn-3b-instruct");
    expect(screen.queryByTestId("run-ledger-row-run-one")).not.toBeInTheDocument();
    expect(screen.getByTestId("run-ledger-row-run-two")).toBeInTheDocument();

    // A collapsed panel must never hide a filter that is silently excluding runs, so the toggle
    // carries the count of active ones.
    expect(screen.getByRole("button", { name: /FILTER · 1/ })).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("ENTRY POINT filter"), "studio");
    await user.selectOptions(screen.getByLabelText("FINISH REASON filter"), "length");
    expect(screen.getByTestId("run-ledger-row-run-two")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Flagged filter"), "unflagged");
    expect(screen.getByText("NO MATCHING RUNS")).toBeInTheDocument();
  });

  test("puts model and entry point on the card without a provenance column", () => {
    const recorded = run({ source: "openai-compatible", model: "clozn-7b-instruct" });
    render(<Runs runtime={runtime([recorded])} inspectorOpen={false} />);

    // The provenance chip group is gone. It spent a 206px column on three chips, one of which
    // ("ADAPTER: NOT RECORDED") was a literal constant on every row of every page. The facts that
    // vary survive on the card's own metadata line.
    const row = screen.getByTestId("run-ledger-row-run-one");
    expect(within(row).queryByRole("group", { name: /^Provenance for/ })).not.toBeInTheDocument();
    expect(row).toHaveTextContent("clozn-7b-instruct");
    expect(row).toHaveTextContent("openai-compatible");
    expect(row).not.toHaveTextContent("NOT RECORDED");
  });

  test("renders a confidence signal only when the index actually measured one", () => {
    const measured = run({ tokenCount: 100, lowConfidenceCount: 30, confidenceMean: 0.6, confidenceMin: 0.02 });
    const { unmount } = render(<Runs runtime={runtime([measured])} inspectorOpen={false} />);
    // A SHARE, not a count: "30 SHAKY" means something different in a 40-token answer than a 400.
    expect(screen.getByText("30% SHAKY")).toBeInTheDocument();
    unmount();

    // No trace recorded -> the run carries no confidence keys at all, and the card says NOTHING
    // rather than "0%" or "NOT RECORDED". Absence is not a measurement of zero.
    render(<Runs runtime={runtime([run()])} inspectorOpen={false} />);
    expect(screen.queryByText(/SHAKY/)).not.toBeInTheDocument();
    expect(screen.queryByText(/NOT RECORDED/)).not.toBeInTheDocument();
  });

  test("does not stage runs for comparison -- Compare picks its own pair", async () => {
    const user = userEvent.setup();
    const first = run();
    const second = run({ id: "run-two", label: "Second recorded answer" });
    render(<Runs runtime={runtime([first, second])} inspectorOpen={false} />);

    // Runs used to carry THREE routes to the same act: per-row A/B buttons, a comparison tray, and a
    // selection dock repeating "A STAGE"/"B STAGE". All three fed a page that never needed them --
    // Compare holds its own idA/idB and defaults to the two most recent runs on mount
    // (features/compare/Compare.tsx). They cost this surface a column, a floating bar that covered a
    // run row, and a whole panel, to hand another page a choice it already makes.
    expect(screen.queryByRole("button", { name: /Stage .* as run A/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Stage selected run as/ })).not.toBeInTheDocument();
    expect(screen.queryByText("SELECTION DOCK")).not.toBeInTheDocument();

    // Selecting still works; it just drives the inspector rather than a staging slot.
    const secondRow = screen.getByTestId("run-ledger-row-run-two");
    await user.click(within(secondRow).getByRole("button", { name: "Select run Second recorded answer" }));
    expect(secondRow).toBeInTheDocument();
  });

  test("draws a confidence sparkline only from supplied token confidences", () => {
    render(<ConfidenceSparkline values={[0.1, 0.7, 0.4]} />);
    expect(screen.getByRole("img", { name: "Confidence sparkline from 3 recorded token confidences" })).toHaveAttribute("data-confidence-sparkline", "recorded");
  });
});
