import { describe, expect, test } from "vitest";
import { render, screen, within } from "../test/render";
import {
  StateBoard,
  type StateBoardCard,
} from "./StateBoard";

const UNAVAILABLE_REASON = "The engine process did not complete its startup handshake.";

function renderBoard(cards: readonly StateBoardCard[]) {
  return render(<StateBoard cards={cards} title="Runtime conditions" />);
}

describe("StateBoard", () => {
  test("renders worker and engine conditions with both visible labels and distinct structural forms", () => {
    const { container } = renderBoard([
      {
        id: "workers",
        kind: "resident_workers",
        capacity: { used: 2, limit: 3 },
        workers: [
          { id: "ready", label: "Qwen worker", status: "resident", note: "Ready for routed work." },
          { id: "busy", label: "Llama worker", status: "busy", note: "Serving one request." },
          { id: "starting", label: "Mistral worker", status: "starting", note: "Loading its model identity." },
        ],
      },
      {
        id: "engine",
        kind: "engine_health",
        health: { status: "degraded", note: "The engine is reachable, but one optional artifact is unavailable." },
      },
    ]);

    const workers = container.querySelector<HTMLElement>('[data-state-board-card="workers"]') as HTMLElement;
    const engine = container.querySelector<HTMLElement>('[data-state-board-card="engine"]') as HTMLElement;
    const workerScope = within(workers);

    for (const [id, label, status] of [
      ["ready", "Qwen worker", "Resident"],
      ["busy", "Llama worker", "Busy"],
      ["starting", "Mistral worker", "Starting"],
    ]) {
      const worker = workers.querySelector<HTMLElement>(`[data-worker-id="${id}"]`) as HTMLElement;
      expect(within(worker).getByText(label)).toBeInTheDocument();
      expect(within(worker).getByText(status)).toBeInTheDocument();
    }
    expect(workerScope.getByText("Resident")).toBeInTheDocument();
    expect(engine.querySelector(".state-board-condition-form.is-degraded")).not.toBeNull();
    expect(within(engine).getByText("Degraded")).toBeInTheDocument();

    const forms = Array.from(workers.querySelectorAll(".state-board-worker-form"));
    expect(new Set(forms.map((form) => form.className)).size).toBe(3);
  });

  test("uses a bounded filled-slots hard-cap meter with explicit N of M, never a progressbar", () => {
    const { container } = renderBoard([{
      id: "workers",
      kind: "resident_workers",
      capacity: { used: 3, limit: 4, label: "resident workers" },
      workers: [],
    }]);

    const workers = container.querySelector<HTMLElement>('[data-state-board-card="workers"]') as HTMLElement;
    const meter = within(workers).getByRole("img", { name: "resident workers: 3 of 4; hard cap." });
    expect(within(workers).getByText("3 of 4")).toBeInTheDocument();
    expect(meter.querySelectorAll(".state-board-capacity-slot.is-filled")).toHaveLength(3);
    expect(within(workers).queryByRole("progressbar")).not.toBeInTheDocument();
  });

  test("keeps passed, failed, partial, blocked, and not-run qualification states distinct", () => {
    const { container } = renderBoard([{
      id: "qualification",
      kind: "qualification",
      steps: [
        { id: "identity", label: "Exact identity", status: "passed", note: "Artifact hash matched." },
        { id: "smoke", label: "Live smoke", status: "failed", note: "The response did not meet the gate." },
        { id: "white-box", label: "White-box evidence", status: "partial", note: "Core probes passed; lab evidence remains pending." },
        { id: "lab", label: "External lab", status: "blocked", reason: "The required lab adapter is not installed." },
        { id: "replay", label: "Replay battery", status: "not_run", reason: "The model was not selected for this battery." },
      ],
    }]);

    const qualification = container.querySelector<HTMLElement>('[data-state-board-card="qualification"]') as HTMLElement;
    for (const [id, status, className] of [
      ["identity", "Passed", "is-passed"],
      ["smoke", "Failed", "is-failed"],
      ["white-box", "Partial", "is-partial"],
      ["lab", "Blocked", "is-blocked"],
      ["replay", "Not run", "is-not-run"],
    ]) {
      const step = qualification.querySelector<HTMLElement>(`[data-qualification-step="${id}"]`) as HTMLElement;
      expect(within(step).getByText(status)).toBeInTheDocument();
      expect(step).toHaveClass(className);
    }
  });

  test("renders unavailable engine health through EvidenceMark with its required reason", () => {
    renderBoard([{
      id: "engine",
      kind: "engine_health",
      health: { status: "unavailable", reason: UNAVAILABLE_REASON },
    }]);

    expect(screen.getByRole("img", { name: `Engine unavailable -- ${UNAVAILABLE_REASON}` })).toBeInTheDocument();
    expect(screen.getByText(UNAVAILABLE_REASON)).toBeInTheDocument();
  });

  test("renders every capability flag with its actual availability and citeable note", () => {
    const { container } = renderBoard([{
      id: "capabilities",
      kind: "capabilities",
      flags: [
        { id: "context", label: "Context receipt", available: true, note: "Recorded receipts can be opened from a run." },
        {
          id: "jlens",
          label: "J-lens",
          available: false,
          note: "This surface should leave the control disabled.",
          reason: "No qualified readout artifact matches the active model.",
        },
      ],
    }]);

    const capabilities = container.querySelector<HTMLElement>('[data-state-board-card="capabilities"]') as HTMLElement;
    const context = capabilities.querySelector<HTMLElement>('[data-capability-id="context"]') as HTMLElement;
    const jlens = capabilities.querySelector<HTMLElement>('[data-capability-id="jlens"]') as HTMLElement;

    expect(within(context).getByText("Available")).toBeInTheDocument();
    expect(within(context).getByText("Recorded receipts can be opened from a run.")).toBeInTheDocument();
    expect(within(jlens).getByRole("img", { name: "Unavailable -- No qualified readout artifact matches the active model." })).toBeInTheDocument();
    expect(within(jlens).getByText("This surface should leave the control disabled.")).toBeInTheDocument();
    expect(capabilities.querySelectorAll("[data-capability-id]")).toHaveLength(2);
  });
});
