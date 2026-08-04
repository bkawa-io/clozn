import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { render, screen, within } from "../test/render";
import { TypedActionOffer, type TypedActionOfferAction } from "./TypedActionOffer";

const ABSENCE_REASON = "No source-influence measurement has been recorded for this run.";
const COST = "Runs one bounded source-influence measurement against the recorded output.";
const PRECONDITIONS = [
  "The recorded prompt and output must still be available.",
  "A measurement worker must be attached to this run.",
];
const BLOCKER_REASON = "A measurement worker is not attached to this run.";

function AvailableOffer({ onAction }: { onAction: () => void }) {
  return (
    <TypedActionOffer
      title="Measure source influence"
      absence={{ state: "not_measured", reason: ABSENCE_REASON }}
      cost={COST}
      preconditions={PRECONDITIONS}
      action={{ availability: "available", label: "Run measurement", onAction }}
    />
  );
}

describe("TypedActionOffer", () => {
  test("renders the absence reason, cost, and every precondition as visible decision context", () => {
    render(<AvailableOffer onAction={vi.fn()} />);

    expect(screen.getByText(ABSENCE_REASON)).toBeInTheDocument();
    expect(screen.getByText(COST)).toBeInTheDocument();
    const preconditions = within(screen.getByRole("list", { name: "Preconditions" }));
    for (const precondition of PRECONDITIONS) {
      expect(preconditions.getByText(precondition)).toBeInTheDocument();
    }
  });

  test("does not invoke an available action while rendering, then invokes it exactly once on an intentional click", async () => {
    const onAction = vi.fn();
    const user = userEvent.setup();
    render(<AvailableOffer onAction={onAction} />);

    expect(onAction).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Run measurement" }));
    expect(onAction).toHaveBeenCalledTimes(1);
  });

  test("keeps a blocked offer's named blocker visible and ignores a callback injected by an untyped caller", async () => {
    const callbackRead = vi.fn();
    const malformedCallback = vi.fn();
    const user = userEvent.setup();
    // JavaScript callers can bypass the public union. The getter is a tripwire: a blocked branch must
    // not even read its injected callback, much less invoke it when the disabled button is clicked.
    const malformedBlockedAction = {
      availability: "blocked" as const,
      label: "Run measurement",
      blockerReason: BLOCKER_REASON,
      get onAction() {
        callbackRead();
        return malformedCallback;
      },
    } as unknown as TypedActionOfferAction;
    render(
      <TypedActionOffer
        title="Measure source influence"
        absence={{ state: "unavailable", reason: ABSENCE_REASON }}
        cost={COST}
        preconditions={PRECONDITIONS}
        action={malformedBlockedAction}
      />,
    );

    const button = screen.getByRole("button", { name: "Run measurement" });
    expect(button).toBeDisabled();
    expect(screen.getByRole("note")).toHaveTextContent(`Blocked by: ${BLOCKER_REASON}`);
    expect(button).toHaveAccessibleDescription(`Blocked by: ${BLOCKER_REASON}`);
    expect(callbackRead).not.toHaveBeenCalled();
    expect(malformedCallback).not.toHaveBeenCalled();
    await user.click(button);
    expect(callbackRead).not.toHaveBeenCalled();
    expect(malformedCallback).not.toHaveBeenCalled();
  });

  test("renders exactly one actionable control", () => {
    render(<AvailableOffer onAction={vi.fn()} />);

    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.queryAllByRole("link")).toHaveLength(0);
  });

  test("composes EvidenceMark with its required absence reason", () => {
    render(<AvailableOffer onAction={vi.fn()} />);

    expect(screen.getByRole("img", { name: `Not measured -- ${ABSENCE_REASON}` }))
      .toHaveAttribute("title", ABSENCE_REASON);
  });
});
