import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { LinkedEvidenceReader } from "./LinkedEvidenceReader";
import type { LinkedReaderSpecimen } from "./model";

const specimen: LinkedReaderSpecimen = {
  runId: "run-1",
  answer: "A refund is available.",
  answerLoci: [{ id: "answer-refund", start: 2, end: 8 }],
  context: [{ id: "billing", label: "Billing policy", state: "available", text: "A renewal can be refunded within fourteen days." }],
};

describe("Linked evidence reader", () => {
  test("hover reveals recorded relationships and click locks the same answer locus", async () => {
    const user = userEvent.setup();
    const loadSelection = vi.fn(async () => ({
      state: "available" as const,
      method: "forced score intervention",
      related: [{
        id: "source-refund",
        documentId: "billing",
        start: 17,
        end: 25,
        effect: "supports" as const,
        deltaNats: -1.42,
        evidenceState: "causally_supported" as const,
      }],
    }));
    render(<LinkedEvidenceReader specimen={specimen} loadSelection={loadSelection} />);

    const answer = screen.getByRole("button", { name: /refund, measured answer locus/i });
    await user.hover(answer);
    await waitFor(() => expect(loadSelection).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/-1.4200/)).toBeInTheDocument();
    const context = await screen.findByRole("button", { name: /refunded, linked context span/i });
    expect(context).toHaveClass("is-supports");

    await user.unhover(answer);
    expect(context).toBeInTheDocument();
    await user.hover(context);
    expect(answer).toHaveClass("is-active");
    await user.click(context);
    await user.unhover(context);
    expect(context).toHaveAttribute("aria-pressed", "true");
    expect(answer).toHaveAttribute("aria-pressed", "true");

    await user.click(answer);
    expect(answer).toHaveAttribute("aria-pressed", "true");
    await user.unhover(answer);
    expect(screen.getByText(/selected relationship/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Test this" }));
    expect(window.location.hash).toBe("#/time-travel/run-1?mode=token&answer=answer-refund&answerStart=2&answerEnd=8&source=source-refund&sourceStart=17&sourceEnd=25");
  });

  test("not measured remains a typed absence instead of an empty relationship", async () => {
    const user = userEvent.setup();
    render(
      <LinkedEvidenceReader
        specimen={specimen}
        loadSelection={async () => ({ state: "not_measured", reason: "No influence map was recorded.", related: [] })}
      />,
    );
    await user.click(screen.getByRole("button", { name: /refund, measured answer locus/i }));
    expect(await screen.findByText("No influence map was recorded.")).toBeInTheDocument();
    expect(screen.queryByText(/no measured context link/i)).not.toBeInTheDocument();
  });

  test("one context span can reveal and lock every linked answer phrase", async () => {
    const user = userEvent.setup();
    const multiSpecimen: LinkedReaderSpecimen = {
      runId: "run-many",
      answer: "Refunds stop.",
      answerLoci: [{ id: "refunds", start: 0, end: 7 }, { id: "stop", start: 8, end: 12 }],
      context: [{ id: "billing", label: "Billing policy", state: "available", text: "Refund window policy." }],
    };
    render(<LinkedEvidenceReader specimen={multiSpecimen} loadSelection={async () => ({
      state: "available",
      related: [{ id: "policy-source", documentId: "billing", start: 0, end: 13, effect: "supports", deltaNats: 0.5, evidenceState: "causally_supported" }],
    })} />);

    const source = await screen.findByRole("button", { name: /refund window, linked context span, 2 related answer phrases/i });
    const refunds = screen.getByRole("button", { name: /refunds, measured answer locus/i });
    const stop = screen.getByRole("button", { name: /stop, measured answer locus/i });
    await user.hover(source);
    expect(refunds).toHaveClass("is-active");
    expect(stop).toHaveClass("is-active");
    await user.click(source);
    await user.unhover(source);
    expect(refunds).toHaveAttribute("aria-pressed", "true");
    expect(stop).toHaveAttribute("aria-pressed", "true");
  });
});
