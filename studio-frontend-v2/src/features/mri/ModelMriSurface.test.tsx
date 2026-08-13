import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelMriSurface } from "./ModelMriSurface";
import type { MriSpecimen } from "./model";

afterEach(cleanup);

const specimen: MriSpecimen = {
  runId: "run_24", sequenceId: "answer:0", tokens: [{ index: 0, text: "Refund" }, { index: 1, text: " policy" }], layers: [{ index: 2, label: "block" }, { index: 3 }],
  channels: [
    { id: "routing", label: "Routing mass", kind: "attention-routing", family: "attention", capability: "available", artifactMode: "recorded", method: "post-softmax capture" },
    { id: "write", label: "Write effect", kind: "attention-write-effect", family: "causal", capability: "artifact-unavailable", reason: "No qualified intervention artifact was retained." },
  ],
  observationsByChannelId: { routing: [
    { locus: { runId: "run_24", sequenceId: "answer:0", tokenIndex: 0, layerIndex: 2 }, evidence: { kind: "measured", finding: "supported", detail: "Routing row retained." }, findings: ["Head 4 routed to T1"], sourceTokens: [{ tokenIndex: 1, label: "policy" }] },
    { locus: { runId: "run_24", sequenceId: "answer:0", tokenIndex: 1, layerIndex: 3 }, evidence: { kind: "measured", finding: "unsupported", detail: "Measured routing did not support the selected claim." } },
  ] },
};

describe("ModelMriSurface", () => {
  it("renders a token × layer evidence atlas without numeric activity readings", () => {
    render(<ModelMriSurface specimen={specimen} />);
    expect(screen.getByRole("heading", { name: "Recorded slice coverage" })).toBeVisible();
    expect(screen.getByLabelText("Token 0, layer 2: Measured")).toBeVisible();
    expect(screen.getByLabelText("Token 1, layer 2: Not captured")).toBeVisible();
    expect(screen.queryByText(/activation|0\.\d+|%/i)).not.toBeInTheDocument();
  });

  it("reports selected loci through a stable run, sequence, token, and layer coordinate", async () => {
    const user = userEvent.setup();
    const onSelectionChange = vi.fn();
    render(<ModelMriSurface specimen={specimen} onSelectionChange={onSelectionChange} />);
    await user.click(screen.getByLabelText("Token 1, layer 3: Measured, unsupported"));
    expect(onSelectionChange).toHaveBeenLastCalledWith({ runId: "run_24", sequenceId: "answer:0", tokenIndex: 1, layerIndex: 3 }, expect.objectContaining({ evidence: { kind: "measured", finding: "unsupported", detail: "Measured routing did not support the selected claim." } }));
    expect(screen.getByText("Measured, unsupported", { selector: ".mri-inspector h3" })).toBeVisible();
  });

  it("keeps routing and write-effect qualification separate", async () => {
    const user = userEvent.setup();
    render(<ModelMriSurface specimen={specimen} />);
    await user.selectOptions(screen.getByLabelText("Instrument"), "write");
    expect(screen.getAllByText("Artifact unavailable").length).toBeGreaterThan(0);
    expect(screen.getByText("No qualified intervention artifact was retained.")).toBeVisible();
    expect(screen.getByText(/Write effect records the result of ablating/)).toBeVisible();
  });
});
