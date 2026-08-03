import { fireEvent } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { render, screen } from "../../test/render";
import { EvidenceLanes, type SemanticEvidenceEvent } from "./EvidenceLanes";

const tokens = [
  {
    text: "First",
    confidence: 0.82,
    entropy: 0.18,
    band: "strong" as const,
    sources: [{
      sourceId: "source-a",
      label: "Reference note",
      effect: "supports" as const,
      deltaNats: 0.44,
      evidenceState: "causally_supported" as const,
    }],
  },
  {
    text: " second",
    confidence: 0.24,
    entropy: 0.91,
    band: "shaky" as const,
    observedSources: [{
      sourceId: "source-b",
      label: "Background",
      effect: "neutral" as const,
      deltaNats: 0.02,
      evidenceState: "observed" as const,
    }],
  },
  { text: " third", confidence: 0.61, entropy: 0.42, band: "okay" as const },
];

const claimEvent: SemanticEvidenceEvent = {
  id: "claim-1",
  label: "Claim boundary",
  startToken: 0,
  endToken: 1,
  kind: "claim",
  detail: "First claim",
};

describe("EvidenceLanes", () => {
  test("shows synchronized evidence states, including below-floor and truncation evidence", () => {
    render(
      <EvidenceLanes
        tokens={tokens}
        selectedToken={1}
        selectedRange={{ start: 0, end: 1 }}
        sourceAvailability={{ available: true }}
        semanticEvents={[claimEvent]}
        finish={{ reason: "length", truncated: true, tokenIndex: 2, detail: "Reached the configured output cap." }}
      />,
    );

    expect(screen.getByText("Evidence lanes")).toBeInTheDocument();
    expect(screen.getByText("TOKENS 1–2")).toBeInTheDocument();
    expect(screen.getByText("24% · 0.240")).toBeInTheDocument();
    expect(screen.getByText("0.910 BITS · SHAKY")).toBeInTheDocument();
    expect(screen.getByText("1 BELOW FLOOR")).toBeInTheDocument();
    expect(screen.getByText("1 IN TOKENS 1–2")).toBeInTheDocument();
    expect(screen.getByText("TRUNCATED · LENGTH")).toBeInTheDocument();
    expect(screen.getByText("Reached the configured output cap.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Source influence token 2, measured below the evidence floor/i }))
      .toHaveClass("is-observed");
    expect(screen.getByRole("button", { name: /Truncated marker at token 3, length/i }))
      .toHaveClass("is-truncated");
  });

  test("publishes one shared token or interval selection from every lane", () => {
    const onSelectionChange = vi.fn();
    const onSelectToken = vi.fn();
    const onSelectRange = vi.fn();
    const onSelectSemanticEvent = vi.fn();

    render(
      <EvidenceLanes
        tokens={tokens}
        selectedToken={1}
        selectedRange={null}
        sourceAvailability={{ available: true }}
        semanticEvents={[claimEvent]}
        finish={{ reason: "stop" }}
        onSelectionChange={onSelectionChange}
        onSelectToken={onSelectToken}
        onSelectRange={onSelectRange}
        onSelectSemanticEvent={onSelectSemanticEvent}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Confidence token 3, 61 percent/i }));
    expect(onSelectionChange).toHaveBeenLastCalledWith({ tokenIndex: 2, range: null });
    expect(onSelectToken).toHaveBeenLastCalledWith(2);
    expect(onSelectRange).toHaveBeenLastCalledWith(null);

    fireEvent.click(
      screen.getByRole("button", { name: /Source influence token 1, 1 cleared link/i }),
      { shiftKey: true },
    );
    expect(onSelectionChange).toHaveBeenLastCalledWith({ tokenIndex: 0, range: { start: 0, end: 1 } });
    expect(onSelectRange).toHaveBeenLastCalledWith({ start: 0, end: 1 });

    fireEvent.click(screen.getByRole("button", { name: /claim.*Claim boundary.*#1–2/i }));
    expect(onSelectionChange).toHaveBeenLastCalledWith({ tokenIndex: 0, range: { start: 0, end: 1 } });
    expect(onSelectSemanticEvent).toHaveBeenLastCalledWith(expect.objectContaining({ id: "claim-1" }));
  });

  test("names unavailable evidence rather than rendering it as an empty lane", () => {
    render(
      <EvidenceLanes
        tokens={[{ text: "No readings", entropy: 0.5 }]}
        selectedToken={0}
        confidenceAvailability={{ available: false, reason: "The provider did not retain probabilities." }}
        sourceAvailability={{ available: false, reason: "Influence scoring is unavailable on this run." }}
        semanticEventsAvailability={{ available: false, reason: "No semantic event recorder ran." }}
        finish={null}
      />,
    );

    expect(screen.getByText(/CONFIDENCE UNAVAILABLE · The provider did not retain probabilities/i)).toBeInTheDocument();
    expect(screen.getByText(/SOURCE EVIDENCE UNAVAILABLE · Influence scoring is unavailable on this run/i)).toBeInTheDocument();
    expect(screen.getByText(/SEMANTIC EVENTS UNAVAILABLE · No semantic event recorder ran/i)).toBeInTheDocument();
    expect(screen.getByText(/FINISH MARKER UNAVAILABLE · The response finish marker was not recorded/i)).toBeInTheDocument();
  });
});
