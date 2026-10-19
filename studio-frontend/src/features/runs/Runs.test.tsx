import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "../../data/client";
import { RunsSurface } from "./Runs";

afterEach(() => { cleanup(); vi.restoreAllMocks(); window.location.hash = ""; });

describe("Runs surface", () => {
  it("loads the run list and opens a run workspace", async () => {
    vi.spyOn(studioApi, "runs").mockResolvedValue([{
      id: "run_a", model: "local", promptSummary: "What changed?", responseSummary: "The date changed.", flags: [],
    }]);
    render(<RunsSurface />);
    expect(await screen.findByText("What changed?")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /What changed/ })).toHaveAttribute("href", "#/inspect/run_a");
  });

  it("loads the working example and creates a durable run", async () => {
    vi.spyOn(studioApi, "runs").mockResolvedValue([]);
    const create = vi.spyOn(studioApi, "createRun").mockResolvedValue({ runId: "run_new", answer: "ORCHID-731" });
    render(<RunsSurface />);
    fireEvent.click(screen.getByRole("button", { name: "Load example" }));
    expect(screen.getByDisplayValue(/what is the launch code/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({
      context: expect.stringContaining("The launch code is ORCHID-731."),
      question: expect.stringContaining("what is the launch code"),
    }), expect.any(AbortSignal)));
    await waitFor(() => expect(window.location.hash).toBe("#/inspect/run_new"));
  });
});
