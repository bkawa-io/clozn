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
    activeDialCount: 0,
    memoryCardCount: 0,
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
  test("filters only recorded provenance and leaves index-only evidence as explained absence", async () => {
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

    expect(screen.getByLabelText("ADAPTER filter")).toBeDisabled();
    expect(screen.getByLabelText("HAS INFLUENCE filter")).toBeDisabled();
    expect(screen.getByText(/Adapter identity is not indexed; no run is assumed to be base/i)).toBeInTheDocument();
    expect(screen.getByText(/Influence presence is not indexed; the ledger does not claim a missing map is no influence/i)).toBeInTheDocument();

    const firstRow = screen.getByTestId("run-ledger-row-run-one");
    const evidence = within(firstRow).getByRole("group", { name: "Evidence index for First recorded answer" });
    expect(within(evidence).getAllByRole("img")).toHaveLength(4);
    expect(within(evidence).getByRole("img", { name: /Context receipt -- The loaded run index does not include/i })).toBeInTheDocument();
    expect(within(evidence).getByRole("img", { name: /Performance -- A listed duration is not a loaded performance trace/i })).toBeInTheDocument();
    expect(within(firstRow).getByRole("img", { name: /Confidence not recorded -- Token confidence is not included in the loaded run index/i })).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("MODEL filter"), "clozn-3b-instruct");
    expect(screen.queryByTestId("run-ledger-row-run-one")).not.toBeInTheDocument();
    expect(screen.getByTestId("run-ledger-row-run-two")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("ENTRY POINT filter"), "studio");
    await user.selectOptions(screen.getByLabelText("FINISH REASON filter"), "length");
    expect(screen.getByTestId("run-ledger-row-run-two")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Flagged filter"), "unflagged");
    expect(screen.getByText("NO MATCHING RUNS")).toBeInTheDocument();
  });

  test("shows model, absent adapter, and entry point as separate provenance facts", () => {
    const recorded = run({ source: "openai-compatible", model: "clozn-7b-instruct" });
    render(<Runs runtime={runtime([recorded])} inspectorOpen={false} />);

    const row = screen.getByTestId("run-ledger-row-run-one");
    const provenance = within(row).getByRole("group", { name: "Provenance for First recorded answer" });
    expect(provenance).toHaveTextContent("MODEL");
    expect(provenance).toHaveTextContent("clozn-7b-instruct");
    expect(provenance).toHaveTextContent("ADAPTER");
    expect(provenance).toHaveTextContent("NOT RECORDED");
    expect(provenance).toHaveTextContent("ENTRY");
    expect(provenance).toHaveTextContent("openai-compatible");
    expect(provenance).not.toHaveTextContent("BASE");
  });

  test("keeps the selection dock inside Runs and routes only after an explicit staged pair", async () => {
    const user = userEvent.setup();
    const first = run();
    const second = run({ id: "run-two", label: "Second recorded answer" });
    render(<Runs runtime={runtime([first, second])} inspectorOpen={false} />);

    const firstRow = screen.getByTestId("run-ledger-row-run-one");
    await user.click(within(firstRow).getByRole("button", { name: "Select run First recorded answer" }));
    await user.click(screen.getByRole("button", { name: "Stage selected run as A" }));

    const secondRow = screen.getByTestId("run-ledger-row-run-two");
    await user.click(within(secondRow).getByRole("button", { name: "Select run Second recorded answer" }));
    expect(screen.getByRole("link", { name: "Open Second recorded answer in Run" })).toHaveAttribute("href", "#/runs/run-two");

    await user.click(screen.getByRole("button", { name: "Stage selected run as B" }));
    expect(screen.getByRole("link", { name: "Compare staged runs" })).toHaveAttribute("href", "#/compare/run-one/run-two");
  });

  test("draws a confidence sparkline only from supplied token confidences", () => {
    render(<ConfidenceSparkline values={[0.1, 0.7, 0.4]} />);
    expect(screen.getByRole("img", { name: "Confidence sparkline from 3 recorded token confidences" })).toHaveAttribute("data-confidence-sparkline", "recorded");
  });
});
