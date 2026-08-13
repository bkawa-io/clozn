import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TurnTimeTravelSurface } from "./TurnTimeTravelSurface";

describe("TurnTimeTravelSurface", () => {
  it("keeps the conversation strand primary and token execution one level below", async () => {
    const user = userEvent.setup();
    const onOpenTokenExecution = vi.fn();
    render(<TurnTimeTravelSurface
      run={{ id: "run-parent", model: "model-a", messages: [{ role: "user", content: "Question" }], response: "Recorded answer" }}
      family={[]}
      linkedSelection={{ differenceId: "difference-1", compareRunId: "run-reference" }}
      onOpenTokenExecution={onOpenTokenExecution}
    />);

    expect(screen.getByRole("heading", { name: "Conversation strand" })).toBeVisible();
    expect(screen.getAllByText("Recorded answer").length).toBeGreaterThan(0);
    expect(screen.getByText("difference · difference-1")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Inspect token execution" }));
    expect(onOpenTokenExecution).toHaveBeenCalledOnce();
  });
});
