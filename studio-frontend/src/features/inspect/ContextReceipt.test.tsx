import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { ContractError } from "../../data/contracts";
import { studioApi } from "../../data/client";
import type { ContextReceiptView } from "../../data/contextReceipt";
import { contextInvestigationApi } from "../../data/contextInvestigationClient";
import { InspectSurface } from "./Inspect";

const runId = "run_receipt";

function receiptView(privacy: "full" | "metadata_only" = "full"): ContextReceiptView {
  return {
    shape: "recorded",
    runId,
    receipt: {
      kind: "v1",
      schemaVersion: "clozn.context-receipt.v1",
      runId,
      privacy,
      delivered: [
        {
          segmentId: "seg_1a2b3c4d5e6f7081",
          sourceType: "message",
          sourceLabel: "system",
          originalOrder: 0,
          deliveredBytes: 42,
          included: true,
          contentHash: "1a2b3c4d5e6f7081",
          redactionState: "hash_only",
        },
        {
          segmentId: "seg_2b3c4d5e6f708192",
          sourceType: "message",
          sourceLabel: "user",
          originalOrder: 1,
          included: false,
          redactionState: "hash_only",
        },
      ],
      assembled: [],
      rendered: { sha256: "a".repeat(64), tokens: 61, estimated: false },
      contextWindowTokens: 4096,
      reservedOutputTokens: 12,
      transformations: [],
      omissions: [{ segmentId: "seg_2b3c4d5e6f708192", reason: "history_policy" }],
      termination: { reason: "max_tokens", generatedTokens: 12 },
    },
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Context Receipt Inspect instrument", () => {
  it("reads canonical receipt facts without requesting duplicated prompt content", async () => {
    const read = vi.spyOn(studioApi, "contextReceipt").mockResolvedValue(receiptView());
    const contextReader = vi.spyOn(contextInvestigationApi, "reader");

    render(<InspectSurface route={{ surface: "inspect", runId, view: "receipt" }} />);

    expect(await screen.findByRole("heading", { name: "Context receipt" })).toBeInTheDocument();
    expect(screen.getByText("system")).toBeInTheDocument();
    expect(screen.getByText("Included")).toBeInTheDocument();
    expect(screen.getByText("Omitted")).toBeInTheDocument();
    expect(screen.getByText("61 exact")).toBeInTheDocument();
    expect(screen.getByText("4,096")).toBeInTheDocument();
    expect(screen.getAllByText("12")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: "Reveal retained prompt" })).not.toBeInTheDocument();
    expect(read).toHaveBeenCalledTimes(1);
    expect(contextReader).not.toHaveBeenCalled();
  });

  it("reports the stored privacy tier without treating the receipt as a content store", async () => {
    const read = vi.spyOn(studioApi, "contextReceipt").mockResolvedValue(receiptView("metadata_only"));
    render(<InspectSurface route={{ surface: "inspect", runId, view: "receipt" }} />);

    expect(await screen.findByText("metadata only")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reveal retained prompt" })).not.toBeInTheDocument();
    expect(read).toHaveBeenCalledTimes(1);
  });

  it("keeps absent and unsupported artifacts distinct", async () => {
    const read = vi.spyOn(studioApi, "contextReceipt").mockResolvedValueOnce({
      shape: "absent",
      runId,
      note: "no context receipt was recorded for this run",
    });
    const view = render(<InspectSurface route={{ surface: "inspect", runId, view: "receipt" }} />);
    expect(await screen.findByText("No Context Receipt was recorded for this Run.")).toBeInTheDocument();

    read.mockResolvedValueOnce({
      shape: "recorded",
      runId: "run_future",
      receipt: { kind: "unsupported", schemaVersion: "clozn.context-receipt.v2", runId: "run_future", rawKeys: ["schema_version", "run_id"] },
    });
    view.rerender(<InspectSurface route={{ surface: "inspect", runId: "run_future", view: "receipt" }} />);
    expect(await screen.findByText("Recorded receipt uses an unsupported schema.")).toBeInTheDocument();

  });

  it("distinguishes malformed evidence from transport unavailability", async () => {
    const read = vi.spyOn(studioApi, "contextReceipt").mockRejectedValueOnce(
      new ContractError("/runs/run_receipt/context-receipt", "$.shape must be recognized"),
    );
    const view = render(<InspectSurface route={{ surface: "inspect", runId, view: "receipt" }} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Malformed Context Receipt contract.");

    read.mockRejectedValueOnce(new Error("gateway offline"));
    view.rerender(<InspectSurface route={{ surface: "inspect", runId: "run_offline", view: "receipt" }} />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Context Receipt unavailable."));
    expect(screen.getByRole("alert")).toHaveTextContent("gateway offline");
  });
});
