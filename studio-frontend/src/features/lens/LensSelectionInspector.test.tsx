import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import type { TokenReading } from "../../data/types";
import { render, screen } from "../../test/render";
import { LensSelectionInspector } from "./LensSelectionInspector";

const tracedTokens: TokenReading[] = [{
  text: "answer",
  entropy: 0.337,
  confidence: 0.9184,
  band: "okay",
  sources: [{
    sourceId: "source-a",
    label: "System instruction",
    effect: "supports",
    deltaNats: 0.291,
    evidenceState: "causally_supported",
  }],
  observedSources: [{
    sourceId: "source-b",
    label: "Retrieved note",
    effect: "neutral",
    deltaNats: 0.004,
    evidenceState: "observed",
  }],
  alternatives: [{ token: "reply", score: 0.81, delta: -0.04 }],
}];

describe("LensSelectionInspector", () => {
  test("keeps cleared and below-floor source links distinct for a selected token", async () => {
    const onSelectSource = vi.fn();
    const user = userEvent.setup();
    render(
      <LensSelectionInspector
        selection={{ kind: "token", index: 0 }}
        tokens={tracedTokens}
        onSelectSource={onSelectSource}
        actions={<button type="button">Open forensic view</button>}
        events={<p>Decode phase recorded</p>}
        influenceMethod={{ mode: "leave_one_out", claimLimit: "a single causal attribution", caveat: "controlled score delta" }}
        influenceThresholds={{ cellAbsDeltaNats: 0.01 }}
        contextCoverage={{ totalSources: 2, measuredSources: 2, omittedSources: 0, measuredSpans: 2, complete: true }}
      />,
    );

    expect(screen.getByRole("heading", { name: "answer" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Cleared links" })).toHaveTextContent("CLEARED FLOOR");
    expect(screen.getByRole("region", { name: "Observed below floor" })).toHaveTextContent("BELOW FLOOR");
    expect(screen.getByText("Measured evidence")).toBeInTheDocument();
    expect(screen.getByText("Decode phase recorded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open forensic view" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /System instruction/i }));
    expect(onSelectSource).toHaveBeenCalledWith("source-a");
  });

  test("shows source measurement states without upgrading an observed absence to a cleared link", () => {
    const source = {
      id: "source-a",
      label: "System instruction",
      text: "Use concise answers.",
      role: "system",
      measured: false,
      clearEffect: false,
    };
    const { rerender } = render(
      <LensSelectionInspector
        selection={{ kind: "source", sourceId: source.id }}
        sources={[source]}
        tokens={[]}
      />,
    );

    expect(screen.getByText("Not measured")).toBeInTheDocument();
    expect(screen.queryByText("Cleared links")).not.toBeInTheDocument();

    rerender(
      <LensSelectionInspector
        selection={{ kind: "source", sourceId: source.id }}
        sources={[{ ...source, measured: true, clearEffect: false }]}
        tokens={[]}
      />,
    );
    expect(screen.getByText("Measured · no cleared link")).toBeInTheDocument();
  });

  test("distinguishes missing token capture from an unavailable source artifact", () => {
    const { rerender } = render(
      <LensSelectionInspector
        selection={{ kind: "token", index: 3 }}
        tokens={[]}
        tokenTrace={{ state: "not_captured", detail: "Token traces were disabled for this run." }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Token trace unavailable" })).toBeInTheDocument();
    expect(screen.getByText("Not captured")).toBeInTheDocument();
    expect(screen.getByText("Token traces were disabled for this run.")).toBeInTheDocument();

    rerender(
      <LensSelectionInspector
        selection={{ kind: "span", start: 0, end: 0 }}
        tokens={[{ ...tracedTokens[0], sources: [], observedSources: [] }]}
        sourceEvidence={{ state: "unavailable", detail: "Scoring requires a compatible worker." }}
      />,
    );
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.getByText("Scoring requires a compatible worker.")).toBeInTheDocument();
  });
});
