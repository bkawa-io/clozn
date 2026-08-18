import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import type { RunSummary } from "../../data/types";
import { render, screen } from "../../test/render";
import {
  LensRunNavigator,
  lensRunNavigatorLabel,
  lensRunNavigatorStatus,
} from "./LensRunNavigator";

function run(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    id: "run_0123456789abcdef",
    label: "Recorded answer",
    prompt: "Explain the result using the retained policy.",
    response: "The retained policy applies.",
    createdAt: "2026-08-02T18:50:00Z",
    source: "studio",
    client: "studio",
    model: "clozn-7b-instruct",
    substrate: "metal",
    duration: "1.3 s",
    flags: [],
    warningCount: 0,
    ...overrides,
  };
}

describe("LensRunNavigator", () => {
  test("derives compact prompt labels and displays each recorded run state with run metadata", () => {
    const complete = run({ id: "run-complete", finishReason: "stop_sequence" });
    const truncated = run({ id: "run-truncated", prompt: "A longer run", finishReason: "length", duration: "8.2 s" });
    const failed = run({ id: "run-failed", prompt: "A failed run", flags: ["error"], model: "clozn-3b" });
    const recorded = run({ id: "run-recorded", prompt: "   whitespace\n    normalized prompt   ", duration: "" });

    render(
      <LensRunNavigator
        runs={[complete, truncated, failed, recorded]}
        selectedRunId={complete.id}
        onSelectRun={vi.fn()}
      />,
    );

    expect(screen.getByText("COMPLETE")).toBeInTheDocument();
    expect(screen.getByText("TRUNCATED")).toBeInTheDocument();
    expect(screen.getByText("ERROR")).toBeInTheDocument();
    expect(screen.getByText("RECORDED")).toBeInTheDocument();
    expect(screen.getAllByText("clozn-7b-instruct")).toHaveLength(3);
    expect(screen.getByText("8.2 s")).toBeInTheDocument();
    expect(screen.getByText("FINISH STOP SEQUENCE")).toBeInTheDocument();
    expect(screen.getByText("DURATION UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("whitespace normalized prompt")).toBeInTheDocument();
    expect(lensRunNavigatorStatus(failed)).toBe("error");
  });

  test("uses controlled selection and calls back with the selected run id", async () => {
    const user = userEvent.setup();
    const first = run({ id: "run-first", prompt: "First prompt" });
    const second = run({ id: "run-second", prompt: "Second prompt" });
    const onSelectRun = vi.fn();
    const view = render(
      <LensRunNavigator runs={[first, second]} selectedRunId={first.id} onSelectRun={onSelectRun} />,
    );

    expect(screen.getByRole("button", { name: /first prompt/i })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: /second prompt/i })).toHaveAttribute("aria-pressed", "false");
    await user.click(screen.getByRole("button", { name: /second prompt/i }));
    expect(onSelectRun).toHaveBeenCalledWith(second.id);

    view.rerender(<LensRunNavigator runs={[first, second]} selectedRunId={second.id} onSelectRun={onSelectRun} />);
    expect(screen.getByRole("button", { name: /second prompt/i })).toHaveAttribute("aria-current", "page");
  });

  test("renders a useful empty state and has a prompt-derived fallback label", () => {
    render(<LensRunNavigator runs={[]} onSelectRun={vi.fn()} />);
    expect(screen.getByText("No recorded runs are available.")).toBeInTheDocument();
    expect(lensRunNavigatorLabel(run({ prompt: "", label: "Fallback run label" }))).toBe("Fallback run label");
    expect(lensRunNavigatorLabel(run({ prompt: "", label: "" }))).toBe("Untitled prompt");
  });
});
