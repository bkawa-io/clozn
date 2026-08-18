import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import type { ObservatoryData } from "../../data/types";
import { render, screen } from "../../test/render";
import { LensContextCanvas, type LensContextDelivery } from "./LensContextCanvas";

function inspection(overrides: Partial<ObservatoryData> = {}): ObservatoryData {
  return {
    id: "run-context",
    label: "Context canvas fixture",
    model: "test-model",
    quant: "Q4",
    createdAt: "now",
    duration: "1 s",
    mode: "run",
    prompt: "Summarize the policy from the retrieved note.",
    response: "Use the retained policy.",
    tokens: [{
      text: "Use",
      entropy: .2,
      sources: [{ sourceId: "policy", label: "System policy", effect: "supports", deltaNats: .2, evidenceState: "causally_supported" }],
    }],
    candidates: [],
    sources: [{ id: "policy", text: "Follow the confirmed policy.", role: "system", kind: "policy", label: "System policy", measured: true, clearEffect: true }],
    contextSources: [
      { id: "policy", text: "Follow the confirmed policy.", role: "system", kind: "policy", label: "System policy", measured: true, clearEffect: true },
      { id: "question", text: "Summarize the policy from the retrieved note.", role: "user", kind: "message", label: "User message", measured: false },
      { id: "retrieved-note", text: "The retained policy applies only to confirmed records.", role: "user", kind: "retrieval", label: "Knowledge base note", measured: true, clearEffect: false },
    ],
    contextCoverage: { totalSources: 3, measuredSources: 2, omittedSources: 1, measuredSpans: 2, complete: false },
    influenceMethod: { mode: "matched_control", claimLimit: "source relevance or necessity", caveat: "Measured links are bounded to the selected context spans." },
    influenceThresholds: { cellAbsDeltaNats: .05 },
    configuration: { adapters: [], changes: [] },
    ...overrides,
  };
}

const recordedDelivery: LensContextDelivery = {
  status: "partial",
  detail: "One retrieved note was omitted by the context budget.",
  requested: [{ sourceLabel: "System message", deliveredTokens: 12 }],
  delivered: [{ sourceLabel: "System message", deliveredTokens: 12, included: true }],
  assembled: [{ sourceLabel: "Rendered system message", deliveredBytes: 320, included: true }],
  rendered: {
    text: "<system>Follow the confirmed policy.</system>",
    tokens: 42,
    bytes: 320,
    templateFingerprint: "chatml-v3",
    contentAvailable: true,
  },
  limits: { promptTokens: 42, contextWindowTokens: 8192, requestedMaxTokens: 256, generatedTokens: 18 },
};

describe("LensContextCanvas input stage", () => {
  test("keeps Conversation readable and routes controlled source selection without upgrading observed evidence", async () => {
    const user = userEvent.setup();
    const onSelectedSourceChange = vi.fn();
    const data = inspection();
    const view = render(<LensContextCanvas data={data} selectedSourceId={null} onSelectedSourceChange={onSelectedSourceChange} />);

    expect(screen.getByRole("tab", { name: "Conversation" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getAllByText("Summarize the policy from the retrieved note.")).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "Messages and instructions" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Retrieved context" })).toBeInTheDocument();
    expect(screen.getByText("1 OUTPUT TOKEN CLEARED")).toBeInTheDocument();
    expect(screen.getByText("MEASURED · NO EFFECT CLEARED")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /knowledge base note/i }));
    expect(onSelectedSourceChange).toHaveBeenCalledWith("retrieved-note");

    view.rerender(<LensContextCanvas data={data} selectedSourceId="retrieved-note" onSelectedSourceChange={onSelectedSourceChange} />);
    const selected = screen.getByRole("button", { name: /knowledge base note/i });
    expect(selected).toHaveAttribute("aria-pressed", "true");
    await user.click(selected);
    expect(onSelectedSourceChange).toHaveBeenLastCalledWith(null);
  });

  test("makes Delivery and Rendered primary representations, and lets dedicated receipt content own Delivery", async () => {
    const user = userEvent.setup();
    const data = inspection();
    const onRepresentationChange = vi.fn();
    const view = render(
      <LensContextCanvas data={data} delivery={recordedDelivery} onRepresentationChange={onRepresentationChange} />,
    );

    await user.click(screen.getByRole("tab", { name: "Delivery" }));
    expect(onRepresentationChange).toHaveBeenCalledWith("delivery");
    expect(screen.getByRole("tab", { name: "Delivery" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("PARTIALLY DELIVERED")).toBeInTheDocument();
    expect(screen.getByText("One retrieved note was omitted by the context budget.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Requested" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Delivered" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Assembled" })).toBeInTheDocument();
    expect(screen.getByText("8,192")).toBeInTheDocument();
    expect(screen.getByText("chatml-v3")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Rendered" }));
    expect(screen.getByText("<system>Follow the confirmed policy.</system>")).toBeInTheDocument();
    expect(screen.getByText("Exact model-facing prompt")).toBeInTheDocument();

    view.rerender(
      <LensContextCanvas
        data={data}
        representation="delivery"
        delivery={recordedDelivery}
        deliveryContent={<p>Dedicated receipt component owns this tab.</p>}
      />,
    );
    expect(screen.getByText("Dedicated receipt component owns this tab.")).toBeInTheDocument();
    expect(screen.queryByText("PARTIALLY DELIVERED")).not.toBeInTheDocument();

    view.rerender(
      <LensContextCanvas
        data={data}
        representation="rendered"
        delivery={recordedDelivery}
        renderedContent={<p>The authorized rendered receipt owns this tab.</p>}
      />,
    );
    expect(screen.getByText("The authorized rendered receipt owns this tab.")).toBeInTheDocument();
    expect(screen.queryByText("Exact model-facing prompt")).not.toBeInTheDocument();
  });

  test("fails closed for absent delivery and source measurement, while preserving delegated measurement actions", async () => {
    const user = userEvent.setup();
    const onStopWaitingForSources = vi.fn();
    const onMeasureSources = vi.fn();
    const data = inspection({
      influenceMethod: undefined,
      contextSources: [{ id: "unmeasured", text: "A retrieved passage with a stale link.", role: "user", kind: "retrieval", label: "Unmeasured passage" }],
      tokens: [{ text: "Use", entropy: .2, sources: [{ sourceId: "unmeasured", label: "Unmeasured passage", effect: "supports", deltaNats: .4, evidenceState: "causally_supported" }] }],
    });
    const view = render(
      <LensContextCanvas
        data={data}
        sourceMeasurementStatus="measuring"
        sourceMeasurementJob={{ schemaVersion: "clozn.influence-map-job.v1", jobId: "job-1", runId: data.id, state: "running", progress: { phase: "scoring", completedUnits: 2, totalUnits: 5, percent: 40 }, cancelRequested: false, cancellable: true, cached: false }}
        onStopWaitingForSources={onStopWaitingForSources}
      />,
    );

    expect(screen.getByText("NOT MEASURED")).toBeInTheDocument();
    expect(screen.queryByText("1 OUTPUT TOKEN CLEARED")).not.toBeInTheDocument();
    expect(screen.getByText("MEASURING CONTEXT SUPPORT · SCORING · 2/5")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "STOP WAITING" }));
    expect(onStopWaitingForSources).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("tab", { name: "Delivery" }));
    expect(screen.getByText("DELIVERY UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("Requested input was not recorded in this receipt.")).toBeInTheDocument();

    view.rerender(
      <LensContextCanvas
        data={data}
        representation="conversation"
        sourceMeasurementStatus="error"
        sourceAbsence={{ kind: "typed", code: "scoring_unavailable", status: "unavailable", message: "No scorer." }}
        onMeasureSources={onMeasureSources}
      />,
    );
    expect(screen.getByText("SCORING UNAVAILABLE ON THIS BUILD")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "MEASURE SOURCES" }));
    expect(onMeasureSources).toHaveBeenCalledOnce();
  });
});
