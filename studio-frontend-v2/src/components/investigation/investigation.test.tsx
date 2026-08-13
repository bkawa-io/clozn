import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { InvestigationLocus } from "../../core/investigation";
import { EvidenceState } from "./EvidenceState";
import { ProvenanceCaption } from "./ProvenanceCaption";
import { assignCollisionLanes, RegistrationRail } from "./RegistrationRail";
import { TestThisLauncher } from "./TestThisLauncher";

afterEach(cleanup);

const first: InvestigationLocus = { kind: "answer-token", runId: "run-1", answerId: "answer-1", tokenIndex: 3 };
const second: InvestigationLocus = { kind: "answer-token", runId: "run-1", answerId: "answer-1", tokenIndex: 4 };

describe("EvidenceState", () => {
  it("presents unavailable and exactness as distinct information", () => {
    render(<EvidenceState state={{ measurement: { kind: "unavailable" }, exactness: { kind: "reconstructed" } }} />);
    expect(screen.getByLabelText("Unavailable; Reconstructed")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeVisible();
    expect(screen.getByText("(Reconstructed)")).toBeVisible();
  });
});

describe("ProvenanceCaption", () => {
  it("renders only supplied provenance details", () => {
    render(<ProvenanceCaption method="causal patch" artifactMode="replayed" exactness={{ kind: "historical" }} />);
    expect(screen.getByText(/Method: causal patch/)).toBeVisible();
    expect(screen.getByText(/Replayed/)).toBeVisible();
    expect(screen.getByText(/Historically verified/)).toBeVisible();
  });
});

describe("RegistrationRail", () => {
  const marks = [
    { locus: first, label: "Token three", position: 0.5 },
    { locus: second, label: "Token four", position: 0.5, related: true },
  ];

  it("allocates collision lanes and supports roving keyboard navigation", async () => {
    expect(assignCollisionLanes(marks).map(({ lane }) => lane)).toEqual([0, 1]);
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(<RegistrationRail marks={marks} onSelect={onSelect} selected={first} />);
    const firstButton = screen.getByRole("option", { name: "Token three" });
    expect(firstButton).toHaveAttribute("aria-selected", "true");
    await user.click(firstButton);
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("option", { name: "Token four" })).toHaveFocus();
    expect(onSelect).toHaveBeenLastCalledWith(expect.objectContaining({ label: "Token four" }));
  });
});

describe("TestThisLauncher", () => {
  it("states its consequence and forwards the precise locus", async () => {
    const onLaunch = vi.fn();
    const user = userEvent.setup();
    render(<TestThisLauncher locus={first} onLaunch={onLaunch} />);
    expect(screen.getByText("May run model work or create a child run.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Test this" }));
    expect(onLaunch).toHaveBeenCalledWith(first);
  });

  it("reports a launch error without claiming that a test started", async () => {
    const user = userEvent.setup();
    render(<TestThisLauncher locus={first} onLaunch={() => Promise.reject(new Error("nope"))} />);
    await user.click(screen.getByRole("button", { name: "Test this" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not start this test");
  });
});
