import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

vi.mock("../features/inspect/Inspect", () => ({
  InspectSurface: () => <div>Inspect surface</div>,
}));
vi.mock("../features/sessions/Sessions", () => ({
  SessionsSurface: () => <div>Sessions surface</div>,
}));
vi.mock("../features/runtime/Runtime", () => ({
  RuntimeSurface: () => <div>Runtime surface</div>,
}));
vi.mock("../features/compare/Compare", () => ({
  CompareSurface: () => <div>Compare surface</div>,
}));

afterEach(() => {
  cleanup();
  window.location.hash = "";
});

describe("App Inspect navigation", () => {
  it("links every shipped Inspect surface for the selected Run", () => {
    window.location.hash = "#/inspect/run%20alpha?view=receipt&sessionId=session%20beta";
    render(<App />);

    const navigation = screen.getByRole("navigation", { name: "Inspect views" });
    expect(navigation).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Context effects" })).toHaveAttribute(
      "href", "#/inspect/run%20alpha?view=context-effects&sessionId=session%20beta",
    );
    expect(screen.getByRole("link", { name: "Received context" })).toHaveAttribute(
      "href", "#/inspect/run%20alpha?view=context&sessionId=session%20beta",
    );
    expect(screen.getByRole("link", { name: "Context receipt" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Minimal context" })).toHaveAttribute(
      "href", "#/inspect/run%20alpha?view=minimal-context&sessionId=session%20beta",
    );
    expect(screen.getByRole("link", { name: "Actions" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Claims" })).toBeInTheDocument();
  });

  it("treats the bare Inspect route as Context effects", () => {
    window.location.hash = "#/inspect/run_alpha";
    render(<App />);

    expect(screen.getByRole("link", { name: "Context effects" })).toHaveAttribute("aria-current", "page");
  });
});
