import { describe, expect, test } from "vitest";
import { render, screen, within } from "../test/render";
import {
  InfluenceMatrix,
  type InfluenceMatrixProps,
} from "./InfluenceMatrix";

const CONTEXT_SPANS = [
  { id: "context-a", text: "System instructions for this response" },
  { id: "context-b", text: "A source excerpt that was considered" },
];

const ANSWER_SPANS = [
  { id: "answer-a", text: "The answer begins here" },
  { id: "answer-b", text: "The answer continues here" },
];

function renderMatrix(overrides: Partial<InfluenceMatrixProps> = {}) {
  const props: InfluenceMatrixProps = {
    contextSpans: CONTEXT_SPANS,
    answerSpans: ANSWER_SPANS,
    measuredLinks: [],
    floorNats: 0.02,
    ...overrides,
  };
  return render(<InfluenceMatrix {...props} />);
}

function cellFor(container: HTMLElement, contextSpanId: string, answerSpanId: string): HTMLElement {
  const cells = Array.from(container.querySelectorAll<HTMLElement>(".influence-matrix-cell"));
  const cell = cells.find((candidate) => (
    candidate.title.includes(`context_span_id: ${contextSpanId}`)
    && candidate.title.includes(`answer_span_id: ${answerSpanId}`)
  ));
  if (!cell) throw new Error(`Missing ${contextSpanId} × ${answerSpanId} cell.`);
  return cell;
}

describe("InfluenceMatrix", () => {
  test("positive and negative equal-magnitude deltas receive distinct signed treatments", () => {
    const { container } = renderMatrix({
      measuredLinks: [
        {
          contextSpanId: "context-a",
          answerSpanId: "answer-a",
          deltaNats: 0.5,
          evidenceState: "causally_supported",
          clearsFloor: true,
        },
        {
          contextSpanId: "context-a",
          answerSpanId: "answer-b",
          deltaNats: -0.5,
          evidenceState: "causally_supported",
          clearsFloor: true,
        },
      ],
    });

    const supports = cellFor(container, "context-a", "answer-a");
    const suppresses = cellFor(container, "context-a", "answer-b");

    expect(supports).toHaveClass("is-supports");
    expect(suppresses).toHaveClass("is-suppresses");
    expect(supports).toHaveAttribute("data-polarity", "supports");
    expect(suppresses).toHaveAttribute("data-polarity", "suppresses");
    expect(supports).toHaveAttribute("data-evidence-state", "causally_supported");
    expect(supports.title).toContain("evidence_state: causally_supported");
    expect(supports.style.getPropertyValue("--influence-matrix-fill"))
      .toBe(suppresses.style.getPropertyValue("--influence-matrix-fill"));
  });

  test("below-floor cells are hatched-state cells, not not-measured evidence marks", () => {
    const reason = "The intervention budget excluded this span.";
    const { container } = renderMatrix({
      measuredLinks: [
        {
          contextSpanId: "context-a",
          answerSpanId: "answer-a",
          deltaNats: 0.01,
          evidenceState: "observed",
          clearsFloor: false,
        },
      ],
      unavailableCells: [
        {
          contextSpanId: "context-b",
          answerSpanId: "answer-a",
          evidenceState: "not_measured",
          reason,
        },
      ],
    });

    const belowFloor = cellFor(container, "context-a", "answer-a");
    const notMeasured = cellFor(container, "context-b", "answer-a");

    expect(belowFloor).toHaveClass("is-below-floor");
    expect(belowFloor).toHaveAttribute("data-evidence-state", "observed");
    expect(belowFloor.title).toContain("evidence_state: observed");
    expect(belowFloor.querySelector(".influence-matrix-below-floor-glyph")).not.toBeNull();
    expect(belowFloor.querySelector(".evidence-mark")).toBeNull();
    expect(notMeasured).toHaveClass("is-not-measured");
    expect(within(notMeasured).getByRole("img", { name: `Not measured -- ${reason}` })).toBeInTheDocument();
  });

  test("an outlier is symmetrically clamped at the labelled percentile", () => {
    const answerSpans = Array.from({ length: 20 }, (_, index) => ({
      id: `answer-${index + 1}`,
      text: `Answer span ${index + 1}`,
    }));
    const measuredLinks = answerSpans.map((answerSpan, index) => ({
      contextSpanId: "context-a",
      answerSpanId: answerSpan.id,
      deltaNats: index === answerSpans.length - 1 ? 100 : 1,
      evidenceState: "causally_supported" as const,
      clearsFloor: true as const,
    }));
    const { container } = renderMatrix({
      contextSpans: [CONTEXT_SPANS[0]],
      answerSpans,
      measuredLinks,
    });

    const clamp = screen.getByText(/SYMMETRIC CLAMP AT 95TH PERCENTILE/);
    expect(clamp).toHaveAttribute("data-clamp-applied", "true");
    expect(clamp).toHaveTextContent("±5.9500 NATS");
    expect(cellFor(container, "context-a", "answer-20").style.getPropertyValue("--influence-matrix-fill"))
      .toBe("90%");
  });

  test("the legend names both signed poles, the neutral midpoint, and the floor", () => {
    renderMatrix({ floorNats: 0.125 });

    const legendElement = screen.getByRole("list", { name: "Influence matrix legend" });
    const legend = within(legendElement);
    expect(legend.getByText("SUPPORTS (+)")).toBeInTheDocument();
    expect(legend.getByText("NEUTRAL (0)")).toBeInTheDocument();
    expect(legend.getByText("SUPPRESSES (−)")).toBeInTheDocument();
    expect(legendElement.querySelector(".influence-matrix-legend-floor")).toHaveTextContent("0.1250 NATS");
  });
});
