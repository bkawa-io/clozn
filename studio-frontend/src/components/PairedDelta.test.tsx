import { describe, expect, test } from "vitest";
import { render, screen, within } from "../test/render";
import {
  PairedDelta,
  type PairedDeltaRow,
  type PairedDeltaSummaryAxis,
} from "./PairedDelta";

const UNAVAILABLE_REASON = "The candidate run does not retain the template receipt.";

describe("PairedDelta", () => {
  test("sorts comparison rows by presentation rank without mutating the caller's order", () => {
    const rows: PairedDeltaRow[] = [
      { id: "output", dimension: "Output", kind: "changed", rank: 40, valueA: "brief", valueB: "long" },
      { id: "model", dimension: "Model", kind: "changed", rank: 5, valueA: "A", valueB: "B" },
      { id: "sampling", dimension: "Sampling", kind: "changed", rank: 20, valueA: "0.2", valueB: "0.7" },
    ];
    const { container } = render(<PairedDelta rows={rows} />);

    const renderedRows = Array.from(container.querySelectorAll<HTMLElement>("[data-delta-row]"));
    expect(renderedRows.map((row) => row.dataset.deltaRow)).toEqual(["model", "sampling", "output"]);
    expect(rows.map((row) => row.id)).toEqual(["output", "model", "sampling"]);
  });

  test("renders unavailable as an EvidenceMark absence with its reason, never a flat delta", () => {
    const { container } = render(
      <PairedDelta
        rows={[{ id: "template", dimension: "Template", kind: "unavailable", rank: 10, reason: UNAVAILABLE_REASON }]}
      />,
    );

    const row = container.querySelector<HTMLElement>('[data-delta-row="template"]') as HTMLElement;
    expect(within(row).getByRole("img", { name: `Unavailable -- ${UNAVAILABLE_REASON}` })).toBeInTheDocument();
    expect(within(row).getByText(UNAVAILABLE_REASON)).toBeInTheDocument();
    expect(row.querySelector('[data-delta-visual="dumbbell"]')).toBeNull();
    expect(row.querySelector('[data-delta-visual="coincident"]')).toBeNull();
    expect(row.querySelector(".paired-delta-values")).toBeNull();
  });

  test("renders synthetic unchanged as a quiet coincident mark, visibly separate from unavailable", () => {
    const { container } = render(
      <PairedDelta
        rows={[
          { id: "model", dimension: "Model", kind: "unchanged", rank: 1, valueA: "same", valueB: "same" },
          { id: "engine", dimension: "Engine", kind: "unavailable", rank: 2, reason: UNAVAILABLE_REASON },
        ]}
      />,
    );

    const unchanged = container.querySelector<HTMLElement>('[data-delta-row="model"]') as HTMLElement;
    const unavailable = container.querySelector<HTMLElement>('[data-delta-row="engine"]') as HTMLElement;
    expect(within(unchanged).getByText("Unchanged")).toBeInTheDocument();
    expect(unchanged.querySelector('[data-delta-visual="coincident"]')).not.toBeNull();
    expect(unchanged.querySelector(".evidence-mark")).toBeNull();
    expect(unavailable.querySelector('[data-delta-visual="coincident"]')).toBeNull();
    expect(within(unavailable).getByRole("img", { name: `Unavailable -- ${UNAVAILABLE_REASON}` })).toBeInTheDocument();
  });

  test("renders every supplied summary axis inside the chip strip", () => {
    const axes: PairedDeltaSummaryAxis[] = [
      { id: "model", label: "Model", status: "changed" },
      { id: "adapter", label: "Adapter", status: "unchanged" },
      { id: "context", label: "Context", status: "unavailable", note: "No receipt was recorded." },
    ];
    render(<PairedDelta rows={[]} summaryAxes={axes} />);

    const chipStrip = screen.getByTestId("paired-delta-summary-axes");
    const strip = within(chipStrip);
    for (const axis of axes) {
      const chip = chipStrip.querySelector<HTMLElement>(`[data-summary-axis="${axis.id}"]`) as HTMLElement;
      expect(chip).not.toBeNull();
      expect(strip.getByText(axis.label)).toBeInTheDocument();
    }
  });
});
