import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { render, screen, waitFor, within } from "../../test/render";
import { AskAnotherQuestion } from "./AskAnotherQuestion";

/**
 * Covers C4's own acceptance criteria (notes/EPIC_ROADMAP_A-F_Q_M.md):
 *   - every offered question maps to a real capability, or renders disabled with a specific reason --
 *     never silently hidden;
 *   - free-text routing never fabricates an explanation;
 *   - investigation history is distinct from (and outlives) a single mount, scoped per run;
 *   - every action reaches a real surface, never re-implements one.
 *
 * This panel fires zero requests by design (see AskAnotherQuestion.tsx's own doc comment) -- every test
 * below stubs `fetch` to THROW if called at all, so an accidental request is a hard failure, not a
 * silently-ignored one.
 */

// The five real destination headings this panel scrolls to, as siblings -- the same DOM shape Lens.tsx
// actually renders (every lens.evidence slot panel mounted together on one page; see SlotHost.tsx).
function Destinations() {
  return (
    <>
      <h3 id="diagnosis-repair-title">Why, and what to try</h3>
      <h3 id="diagnosis-repair-retries-title">Corrective retries</h3>
      <h3 id="what-mattered-title">What mattered?</h3>
      <h3 id="received-context-title">What did the model receive?</h3>
      <h3 id="claim-verify-title">Are the claims supported?</h3>
      <h3 id="second-opinion-title">Would another model disagree?</h3>
      <h3 id="investigation-experiment-title">Did this matter?</h3>
    </>
  );
}

function renderPanel(runId: string) {
  return render(
    <>
      <AskAnotherQuestion runId={runId} />
      <Destinations />
    </>,
  );
}

function throwingFetch() {
  return vi.fn(() => {
    throw new Error("AskAnotherQuestion must never fetch -- it only scrolls and records history");
  });
}

function startsWith(prefix: string) {
  return (accessibleName: string) => accessibleName.startsWith(prefix);
}

beforeEach(() => {
  window.localStorage.clear();
});

describe("Ask another question", () => {
  test("offers all seven questions, each with a visible capability state -- none silently hidden", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    renderPanel("run-one");

    const grid = screen.getByRole("list", { name: "Investigation questions" });
    const labels = [
      "Why?",
      "What mattered?",
      "What did the model receive?",
      "Which claims are supported?",
      "Retry with a correction",
      "Would another model disagree?",
      "What happens without this passage?",
    ];
    for (const label of labels) {
      expect(within(grid).getByRole("button", { name: startsWith(label) })).toBeInTheDocument();
    }

    // All seven have a real destination now; E4's destination itself may still report that this run's
    // gateway has no resident comparison model.
    expect(screen.getByText("7 / 7 RUNNABLE")).toBeInTheDocument();
  });

  test("the second-opinion question opens its real destination without making a request", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    const button = screen.getByRole("button", { name: startsWith("Would another model disagree?") });
    expect(button).toBeEnabled();
    const target = document.getElementById("second-opinion-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});
    await user.click(button);
    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
    expect(screen.getByRole("heading", { name: "Would another model disagree?" })).toBeInTheDocument();
  });

  test("the passage question opens the explicit C3 experiment surface", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    renderPanel("run-one");

    const button = screen.getByRole("button", { name: startsWith("What happens without this passage?") });
    expect(button).toBeEnabled();
    expect(within(button).getByText("OPENS REAL EVIDENCE")).toBeInTheDocument();
    const target = document.getElementById("investigation-experiment-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});
    await userEvent.setup().click(button);
    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
  });

  test("clicking an available question scrolls to its real target and records exactly one history entry", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    const target = document.getElementById("diagnosis-repair-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});

    await user.click(screen.getByRole("button", { name: startsWith("Why?") }));

    const historyRegion = screen.getByRole("region", { name: "Past investigations for this run" });
    expect(await within(historyRegion).findByText("Why?")).toBeInTheDocument();
    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
    // The entry-count badge in the section header reflects exactly one recorded investigation.
    expect(historyRegion.querySelector(".section-title span")).toHaveTextContent("1");
  });

  test("free text that matches a runnable question routes to it and echoes the typed text into history, verbatim", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    const target = document.getElementById("what-mattered-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});

    await user.type(screen.getByLabelText("ROUTE A QUESTION"), "did this actually matter to the answer");
    await user.click(screen.getByRole("button", { name: "ROUTE" }));

    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
    const historyRegion = screen.getByRole("region", { name: "Past investigations for this run" });
    expect(await within(historyRegion).findByText('"did this actually matter to the answer"')).toBeInTheDocument();
    expect(within(historyRegion).getByText("What mattered?")).toBeInTheDocument();
  });

  test("pressing Enter in the router submits the matched question", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    const target = document.getElementById("diagnosis-repair-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});
    await user.type(screen.getByLabelText("ROUTE A QUESTION"), "why did this happen{Enter}");

    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
    const historyRegion = screen.getByRole("region", { name: "Past investigations for this run" });
    expect(await within(historyRegion).findByText('"why did this happen"')).toBeInTheDocument();
    expect(within(historyRegion).getByText("Why?")).toBeInTheDocument();
  });

  test("free text that matches the second-opinion question routes to its real destination", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    const target = document.getElementById("second-opinion-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});
    await user.type(screen.getByLabelText("ROUTE A QUESTION"), "would a different model disagree with this");
    await user.click(screen.getByRole("button", { name: "ROUTE" }));

    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
    expect(screen.getByRole("heading", { name: "Would another model disagree?" })).toBeInTheDocument();
  });

  test("free text that matches nothing states plainly that it cannot run -- never guesses, never records", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    await user.type(screen.getByLabelText("ROUTE A QUESTION"), "asdf qwerty zzz");
    await user.click(screen.getByRole("button", { name: "ROUTE" }));

    const notice = await screen.findByRole("status");
    expect(notice).toHaveTextContent('"asdf qwerty zzz"');
    expect(notice).toHaveTextContent("doesn't match a question clozn can run yet");
    expect(screen.getByText("No investigations recorded yet for this run.")).toBeInTheDocument();
  });

  test("investigation history is scoped per run and survives a remount -- never leaks across runs", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    const first = renderPanel("run-a");

    const target = document.getElementById("received-context-title")!;
    vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});
    await user.click(screen.getByRole("button", { name: startsWith("What did the model receive?") }));
    const firstHistory = screen.getByRole("region", { name: "Past investigations for this run" });
    expect(await within(firstHistory).findByText("What did the model receive?")).toBeInTheDocument();
    first.unmount();

    // A different run starts with no history of its own.
    const second = render(<><AskAnotherQuestion runId="run-b" /><Destinations /></>);
    expect(screen.getByText("No investigations recorded yet for this run.")).toBeInTheDocument();
    second.unmount();

    // Returning to the original run still shows the investigation recorded against it earlier.
    renderPanel("run-a");
    const thirdHistory = screen.getByRole("region", { name: "Past investigations for this run" });
    expect(await within(thirdHistory).findByText("What did the model receive?")).toBeInTheDocument();
  });

  test("revisiting a past investigation from history scrolls to the same real target again", async () => {
    vi.stubGlobal("fetch", throwingFetch());
    const user = userEvent.setup();
    renderPanel("run-one");

    const target = document.getElementById("claim-verify-title")!;
    const scrollSpy = vi.spyOn(target, "scrollIntoView").mockImplementation(() => {});
    await user.click(screen.getByRole("button", { name: startsWith("Which claims are supported?") }));

    const historyRegion = screen.getByRole("region", { name: "Past investigations for this run" });
    await within(historyRegion).findByText("Which claims are supported?");
    scrollSpy.mockClear();

    const historyEntry = within(historyRegion).getByRole("button", { name: /Which claims are supported\?/ });
    await user.click(historyEntry);
    await waitFor(() => expect(scrollSpy).toHaveBeenCalled());
  });

  test("the free-text field is captioned as a router, never as a place to get an answer", () => {
    vi.stubGlobal("fetch", throwingFetch());
    renderPanel("run-one");
    expect(screen.getByPlaceholderText(/never answers directly/)).toBeInTheDocument();
  });
});
