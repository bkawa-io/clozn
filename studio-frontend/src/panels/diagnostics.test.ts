import { describe, expect, test } from "vitest";
import panel from "./diagnostics";

describe("diagnostics panel routing", () => {
  test("claims canonical run and section routes", () => {
    expect(panel.match("#/diagnostics")).toEqual({ view: "overview" });
    expect(panel.match("#/runs/run_alpha/diagnostics")).toEqual({ runId: "run_alpha", view: "overview" });
    expect(panel.match("#/runs/run%2Falpha/diagnostics/source%20influence")).toEqual({
      runId: "run/alpha",
      view: "source influence",
    });
  });

  test("maps retired Scope and Investigation URLs into read-only diagnostics", () => {
    expect(panel.match("#/runs/run_alpha/scope?token=4&view=layers")).toEqual({
      runId: "run_alpha",
      view: "generation",
    });
    expect(panel.match("#/sessions/session_alpha/investigate")).toEqual({
      sessionId: "session_alpha",
      view: "overview",
    });
    expect(panel.match("#/scope")).toEqual({ view: "overview" });
    expect(panel.match("#/investigation")).toEqual({ view: "overview" });
  });

  test("does not claim Lens or the runs index", () => {
    expect(panel.match("#/runs/run_alpha/lens")).toBeNull();
    expect(panel.match("#/runs")).toBeNull();
  });
});
