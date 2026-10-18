import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

vi.mock("../features/inspect/Inspect", () => ({ InspectSurface: () => <div>Run workspace</div> }));
vi.mock("../features/runs/Runs", () => ({ RunsSurface: () => <div>Runs surface</div> }));

afterEach(() => {
  cleanup();
  window.location.hash = "";
});

describe("v1 Studio shell", () => {
  it("shows only the run list at the neutral route", () => {
    window.location.hash = "#/runs";
    render(<App />);
    expect(screen.getByText("Runs surface")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Runs" })).toHaveAttribute("aria-current", "page");
    expect(screen.queryByText("Sessions")).not.toBeInTheDocument();
    expect(screen.queryByText("Runtime")).not.toBeInTheDocument();
    expect(screen.queryByText("Compare")).not.toBeInTheDocument();
  });

  it("offers exactly the two run workspace modes", () => {
    window.location.hash = "#/inspect/run%20alpha?view=minimal-context";
    render(<App />);
    const navigation = screen.getByRole("navigation", { name: "Run workspace modes" });
    expect(navigation).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Context Effects" })).toHaveAttribute(
      "href", "#/inspect/run%20alpha?view=context-effects",
    );
    expect(screen.getByRole("link", { name: "Minimal Context" })).toHaveAttribute("aria-current", "page");
    expect(screen.queryByText("Received context")).not.toBeInTheDocument();
    expect(screen.queryByText("Context receipt")).not.toBeInTheDocument();
  });
});
