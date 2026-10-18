import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "../../data/client";
import { RunDetailsDisclosure } from "./ContextReceipt";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("Run details disclosure", () => {
  it("folds runtime and Context Receipt evidence into one disclosure", async () => {
    vi.spyOn(studioApi, "run").mockResolvedValue({
      id: "run_a", model: "local-model", substrate: "engine",
      identity: { engine_build: "build-a", model_sha256: "a".repeat(64) },
    });
    vi.spyOn(studioApi, "contextReceipt").mockResolvedValue({
      shape: "recorded", runId: "run_a", receipt: {
        kind: "v1", schemaVersion: "clozn.context-receipt.v1", runId: "run_a",
        delivered: [], rendered: { tokens: 42 }, termination: { reason: "eos" },
      },
    });
    render(<RunDetailsDisclosure runId="run_a" />);
    fireEvent.click(screen.getByText("Context, provenance, and runtime details"));
    await waitFor(() => expect(screen.getByText("local-model")).toBeInTheDocument());
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(studioApi.run).toHaveBeenCalledWith("run_a", expect.any(AbortSignal));
  });
});
