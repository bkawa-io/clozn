import { describe, expect, test } from "vitest";
import { render, screen, within } from "../test/render";
import { CompositionBar, type CompositionSegment } from "./CompositionBar";

const OMITTED_REASON = "Dropped by the context-budget policy before assembly.";

describe("CompositionBar", () => {
  test("renders each segment proportional to its own recorded value, not to a rescaled share", () => {
    const segments: CompositionSegment[] = [
      { id: "present", label: "Delivered", value: 700, kind: "present" },
      { id: "reduced", label: "Redacted", value: 200, kind: "reduced" },
      { id: "absent", label: "History", value: 100, kind: "absent", reason: OMITTED_REASON },
    ];
    const { container } = render(<CompositionBar segments={segments} />);

    const present = container.querySelector(".composition-bar-segment.is-present") as HTMLElement;
    const reduced = container.querySelector(".composition-bar-segment.is-reduced") as HTMLElement;
    const absent = container.querySelector(".composition-bar-segment.is-absent") as HTMLElement;

    // flex-grow carries the raw value directly, so the ratio between segments' grow factors is exactly
    // the ratio between their recorded values -- this is what "proportional" means for a flex-based bar.
    expect(present.style.flexGrow).toBe("700");
    expect(reduced.style.flexGrow).toBe("200");
    expect(absent.style.flexGrow).toBe("100");
  });

  test("an omitted segment renders as its own reasoned bar segment, never as a shorter bar", () => {
    const segments: CompositionSegment[] = [
      { id: "present", label: "Delivered", value: 80, kind: "present" },
      { id: "absent", label: "History", value: 20, kind: "absent", reason: OMITTED_REASON },
    ];
    const { container } = render(<CompositionBar segments={segments} />);

    // The reason is real, selectable text -- not a colour or a tooltip a reader has to hover to find.
    expect(screen.getByText(OMITTED_REASON)).toBeInTheDocument();

    // The absent segment is a real rectangle in the track, sized by its own recorded value...
    const absent = container.querySelector(".composition-bar-track .composition-bar-segment.is-absent") as HTMLElement;
    expect(absent).not.toBeNull();
    expect(absent.style.flexGrow).toBe("20");

    // ...and the present segment is not inflated to compensate for the absence: it still reflects its
    // own recorded 80, not 100% of the bar.
    const present = container.querySelector(".composition-bar-track .composition-bar-segment.is-present") as HTMLElement;
    expect(present.style.flexGrow).toBe("80");
  });

  test("segments that fall short of the stated total produce a labelled unaccounted remainder, not a rescale to 100%", () => {
    const segments: CompositionSegment[] = [
      { id: "present", label: "Delivered", value: 700, kind: "present" },
    ];
    const { container } = render(<CompositionBar segments={segments} total={1000} unit="tokens" />);

    const unaccounted = container.querySelector(".composition-bar-segment.is-unaccounted") as HTMLElement;
    expect(unaccounted).not.toBeNull();
    expect(unaccounted.style.flexGrow).toBe("300");

    // The present segment keeps its own recorded value -- it is never stretched to fill the bar just
    // because it happens to be the only caller-supplied segment.
    const present = container.querySelector(".composition-bar-segment.is-present") as HTMLElement;
    expect(present.style.flexGrow).toBe("700");

    // Scoped to the key: a wide segment also carries its own inline label, so an unscoped query
    // matches twice. That duplication is intended -- the key is the guaranteed home for every
    // label, the inline one is opportunistic -- so the assertion names which one it means.
    const key = within(container.querySelector(".composition-bar-key") as HTMLElement);
    expect(key.getByText("Unaccounted")).toBeInTheDocument();
    expect(screen.getByText(/300 tokens more than the recorded segments/)).toBeInTheDocument();
  });

  test("segments that already sum to the total produce no unaccounted row", () => {
    const segments: CompositionSegment[] = [
      { id: "present", label: "Delivered", value: 60, kind: "present" },
      { id: "absent", label: "History", value: 40, kind: "absent", reason: OMITTED_REASON },
    ];
    const { container } = render(<CompositionBar segments={segments} total={100} />);
    expect(container.querySelector(".composition-bar-segment.is-unaccounted")).toBeNull();
    expect(screen.queryByText("Unaccounted")).not.toBeInTheDocument();
  });

  test("the key lists every segment, including the synthesized unaccounted remainder", () => {
    const segments: CompositionSegment[] = [
      { id: "present", label: "Delivered", value: 700, kind: "present" },
      { id: "reduced", label: "Redacted", value: 90, kind: "reduced" },
      { id: "absent", label: "History", value: 90, kind: "absent", reason: OMITTED_REASON },
    ];
    const { container } = render(<CompositionBar segments={segments} total={1000} unit="tokens" />);

    const key = container.querySelector(".composition-bar-key");
    expect(key).not.toBeNull();
    const rows = key!.querySelectorAll(".composition-bar-key-row");
    // 3 caller segments + 1 synthesized unaccounted row (1000 - 880 = 120).
    expect(rows).toHaveLength(4);

    const keyScope = within(key as HTMLElement);
    for (const label of ["Delivered", "Redacted", "History", "Unaccounted"]) {
      expect(keyScope.getByText(label)).toBeInTheDocument();
    }
  });

  test("a single full-total segment renders without a key -- nothing needs disambiguating", () => {
    const segments: CompositionSegment[] = [
      { id: "present", label: "Delivered", value: 100, kind: "present" },
    ];
    const { container } = render(<CompositionBar segments={segments} />);
    expect(container.querySelector(".composition-bar-key")).toBeNull();
  });

  test("an empty segment list with no stated total renders an explicit empty state, not a blank bar", () => {
    render(<CompositionBar segments={[]} />);
    expect(screen.getByText("NO SEGMENTS RECORDED")).toBeInTheDocument();
  });
});
