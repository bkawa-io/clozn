import { describe, expect, test } from "vitest";
import { render, screen } from "../test/render";
import { EvidenceMark } from "./EvidenceMark";

const NOT_MEASURED_REASON = "No source map has been computed for this run yet.";
const UNAVAILABLE_REASON = "Scoring is unavailable on this build.";

describe("EvidenceMark", () => {
  test("each of the four states renders and is identifiable by its accessible name", () => {
    render(
      <>
        <EvidenceMark state="measured" />
        <EvidenceMark state="below_floor" />
        <EvidenceMark state="not_measured" reason={NOT_MEASURED_REASON} />
        <EvidenceMark state="unavailable" reason={UNAVAILABLE_REASON} />
      </>,
    );

    expect(screen.getByRole("img", { name: "Measured" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Below floor" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: `Not measured -- ${NOT_MEASURED_REASON}` })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: `Unavailable -- ${UNAVAILABLE_REASON}` })).toBeInTheDocument();
  });

  test("reason text is present for both absence states, as visible text in chip variant", () => {
    render(
      <>
        <EvidenceMark state="not_measured" reason={NOT_MEASURED_REASON} variant="chip" />
        <EvidenceMark state="unavailable" reason={UNAVAILABLE_REASON} variant="chip" />
      </>,
    );

    // Rule 3: an absence is never rendered as an empty glyph -- the reason shows up as real text a
    // reader can select and copy, not just something implied by colour.
    expect(screen.getByText(NOT_MEASURED_REASON)).toBeInTheDocument();
    expect(screen.getByText(UNAVAILABLE_REASON)).toBeInTheDocument();
  });

  test("dot variant carries the reason through title, since it has no visible text of its own", () => {
    render(<EvidenceMark state="unavailable" reason={UNAVAILABLE_REASON} />);
    expect(screen.getByRole("img", { name: `Unavailable -- ${UNAVAILABLE_REASON}` }))
      .toHaveAttribute("title", UNAVAILABLE_REASON);
  });

  test("the four states produce four different class names -- form, not colour, is the real signal", () => {
    const { container } = render(
      <>
        <EvidenceMark state="measured" />
        <EvidenceMark state="below_floor" />
        <EvidenceMark state="not_measured" reason={NOT_MEASURED_REASON} />
        <EvidenceMark state="unavailable" reason={UNAVAILABLE_REASON} />
      </>,
    );
    const marks = Array.from(container.querySelectorAll(".evidence-mark"));
    expect(marks).toHaveLength(4);
    const classNames = new Set(marks.map((mark) => mark.className));
    expect(classNames.size).toBe(4);
  });

  test("a caller-supplied label overrides the default without dropping the reason from the accessible name", () => {
    render(<EvidenceMark state="not_measured" reason="Outside the span budget." label="Not scored" />);
    expect(screen.getByRole("img", { name: "Not scored -- Outside the span budget." })).toBeInTheDocument();
  });

  test("chip variant renders a distinct decorative glyph in addition to the visible label", () => {
    const { container } = render(<EvidenceMark state="measured" variant="chip" />);
    expect(container.querySelector(".evidence-mark-glyph[aria-hidden='true']")).not.toBeNull();
    expect(screen.getByText("Measured")).toBeInTheDocument();
  });
});
