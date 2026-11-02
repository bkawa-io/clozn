import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "../../data/client";
import type { RunSummary } from "../../data/contracts";
import { RunsSurface } from "./Runs";

afterEach(() => { cleanup(); vi.restoreAllMocks(); window.location.hash = ""; });

const runs: RunSummary[] = [
  {
    id: "run_second", model: "local", promptSummary: "Refine the library thesis",
    responseSummary: "Use a stronger bridge.", conversationId: "run_first",
    conversationTurn: 2, conversationSize: 2, analysisStatus: "partial",
    analysisSourceCount: 8, analysisCompletedSourceCount: 3, analysisScoringMode: "exact", flags: ["low-confidence"],
  },
  {
    id: "run_first", model: "local", promptSummary: "Draft a climate essay",
    responseSummary: "Start with the evidence.", conversationId: "run_first",
    conversationTurn: 1, conversationSize: 2, analysisStatus: "completed",
    analysisSourceCount: 6, analysisCompletedSourceCount: 6, flags: [],
  },
  {
    id: "run_other", model: "local", promptSummary: "Plan a dinner",
    responseSummary: "Make soup.", conversationId: "run_other",
    conversationTurn: 1, conversationSize: 1, analysisStatus: "not_measured",
    analysisSourceCount: 2, analysisCompletedSourceCount: 0, flags: [],
  },
];

describe("Runs surface", () => {
  it("shows only conversation summaries with opening and latest previews", async () => {
    vi.spyOn(studioApi, "runs").mockResolvedValue(runs);
    render(<RunsSurface route={{ surface: "runs" }} />);

    const conversation = await screen.findByRole("link", { name: /Draft a climate essay/ });
    expect(conversation).toHaveAttribute("href", "#/conversations/run_first");
    expect(screen.queryByText("First message")).not.toBeInTheDocument();
    expect(screen.queryByText("Latest reply")).not.toBeInTheDocument();
    expect(screen.getByLabelText("First message: Draft a climate essay")).toBeInTheDocument();
    expect(screen.getByLabelText("Latest reply: Use a stronger bridge.")).toBeInTheDocument();
    expect(conversation).toHaveTextContent("Draft a climate essay");
    expect(conversation).toHaveTextContent("Use a stronger bridge.");
    expect(screen.queryByRole("link", { name: /Analyze response/ })).not.toBeInTheDocument();
  });

  it("stays focused on recorded conversations instead of exposing a second chat interface", async () => {
    vi.spyOn(studioApi, "runs").mockResolvedValue([]);
    render(<RunsSurface route={{ surface: "runs" }} />);
    expect(await screen.findByText(/Point an OpenAI-compatible chat client at Clozn/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load example" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create run" })).not.toBeInTheDocument();
  });

  it("searches whole conversations while retaining the opening and latest previews", async () => {
    vi.spyOn(studioApi, "runs").mockResolvedValue(runs);
    render(<RunsSurface route={{ surface: "runs" }} />);

    fireEvent.change(await screen.findByLabelText("Search conversations"), {
      target: { value: "stronger bridge" },
    });
    expect(screen.getByRole("link", { name: /Draft a climate essay/ })).toHaveTextContent("Use a stronger bridge.");
    expect(screen.queryByText("Plan a dinner")).not.toBeInTheDocument();
    expect(screen.getByText("1 conversations")).toBeInTheDocument();
  });

  it("opens a full chat transcript at the latest turn with per-message analysis links", async () => {
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockReturnValue(2000);
    vi.spyOn(studioApi, "runs").mockResolvedValue(runs);
    const details = vi.spyOn(studioApi, "run").mockImplementation(async (id) => ({
      id,
      messages: [{
        role: "user",
        content: id === "run_first" ? "Full first user message" : "Full second user message",
      }],
      response: id === "run_first" ? "Full first assistant response" : "Full second assistant response",
      identity: {},
    }));
    render(<RunsSurface route={{ surface: "runs", conversationId: "run_first" }} />);

    expect(await screen.findByText("Full first user message")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Turn 1, user message" })).toHaveTextContent("Full first user message");
    expect(screen.getByRole("region", { name: "Turn 2, assistant response" })).toHaveTextContent("Full second assistant response");
    expect(screen.getByRole("navigation", { name: "Conversation turns" })).toBeInTheDocument();
    const secondTurn = screen.getByRole("button", { name: /Jump to turn 2: Refine the library thesis/ });
    expect(secondTurn).toHaveAttribute("aria-current", "true");
    expect(secondTurn).toHaveTextContent("Refine the library thesis");
    fireEvent.click(secondTurn);
    expect(secondTurn).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("link", { name: "Examine turn 2 from the request" })).toHaveAttribute("href", "#/inspect/run_second?conversationId=run_first&scoringMode=exact");
    expect(screen.getByRole("link", { name: "Examine turn 2 from the response" })).toHaveAttribute("href", "#/inspect/run_second?conversationId=run_first&scoringMode=exact");
    expect(screen.queryByText("low confidence")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete record" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "← All conversations" })).toHaveAttribute("href", "#/runs");
    await waitFor(() => expect(document.querySelector<HTMLElement>(".conversation-transcript")?.scrollTop).toBe(2000));
    // At the bottom, a tall preceding turn must not steal the active marker.
    // jsdom's equal zero-sized rectangles would otherwise select the first turn.
    fireEvent.scroll(document.querySelector<HTMLElement>(".conversation-transcript")!);
    expect(secondTurn).toHaveAttribute("aria-current", "true");
    await waitFor(() => expect(details).toHaveBeenCalledTimes(2));
  });
});
