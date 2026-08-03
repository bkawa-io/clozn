import { fireEvent } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { render, screen } from "../../test/render";
import { EvidenceDeck } from "./EvidenceDeck";

const laneTokens = [
  { text: "One", confidence: 0.8, entropy: 0.2, band: "strong" as const },
  { text: " two", confidence: 0.3, entropy: 0.8, band: "shaky" as const },
];

describe("EvidenceDeck", () => {
  test("uses controlled tabs and renders the selected supplied section", () => {
    const onSelectedSectionChange = vi.fn();
    const { rerender } = render(
      <EvidenceDeck
        selectedSection="events"
        onSelectedSectionChange={onSelectedSectionChange}
        sections={{
          events: { content: <p>Generation phase completed.</p> },
          performance: { content: <p>42 milliseconds.</p> },
        }}
      />,
    );

    expect(screen.getByRole("tab", { name: "Events" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Generation phase completed.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Performance" }));
    expect(onSelectedSectionChange).toHaveBeenCalledWith("performance");

    rerender(
      <EvidenceDeck
        selectedSection="performance"
        onSelectedSectionChange={onSelectedSectionChange}
        sections={{ performance: { content: <p>42 milliseconds.</p> } }}
      />,
    );
    expect(screen.getByText("42 milliseconds.")).toBeInTheDocument();
  });

  test("collapses and resizes through controlled callbacks without hiding tab state", () => {
    const onCollapsedChange = vi.fn();
    const onHeightChange = vi.fn();
    render(
      <EvidenceDeck
        height={320}
        onHeightChange={onHeightChange}
        onCollapsedChange={onCollapsedChange}
        sections={{ evidence: { content: <p>Recorded evidence.</p> } }}
      />,
    );

    const handle = screen.getByRole("separator", { name: "Resize evidence deck" });
    expect(handle).toHaveAttribute("aria-valuenow", "320");
    fireEvent.keyDown(handle, { key: "ArrowUp" });
    expect(onHeightChange).toHaveBeenCalledWith(352);

    fireEvent.click(screen.getByRole("button", { name: "Collapse deck" }));
    expect(onCollapsedChange).toHaveBeenCalledWith(true);
    expect(screen.queryByText("Recorded evidence.")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Evidence" })).toHaveAttribute("aria-selected", "true");
  });

  test("renders an honest unavailable state when no panel artifact was supplied", () => {
    render(
      <EvidenceDeck
        selectedSection="lineage"
        sections={{ lineage: { availability: { state: "not_captured", detail: "Lineage receipt was not retained." } } }}
      />,
    );

    expect(screen.getByText("NOT CAPTURED")).toBeInTheDocument();
    expect(screen.getByText("Lineage receipt was not retained.")).toBeInTheDocument();
  });

  test("embeds EvidenceLanes and forwards its token/range selection unchanged", () => {
    const onSelectionChange = vi.fn();
    render(
      <EvidenceDeck
        evidenceLanes={{
          tokens: laneTokens,
          selectedToken: 0,
          selectedRange: null,
          onSelectionChange,
          semanticEvents: [],
          finish: { reason: "stop" },
        }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Confidence token 2, 30 percent/i }));
    expect(onSelectionChange).toHaveBeenCalledWith({ tokenIndex: 1, range: null });
  });
});
