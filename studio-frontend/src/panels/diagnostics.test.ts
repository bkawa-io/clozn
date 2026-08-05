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

  test("maps retired Scope URLs into read-only diagnostics", () => {
    expect(panel.match("#/runs/run_alpha/scope?token=4&view=layers")).toEqual({
      runId: "run_alpha",
      view: "generation",
    });
    expect(panel.match("#/scope")).toEqual({ view: "overview" });
  });

  test("leaves the session routes to F3's investigation panel", () => {
    // These were written as retirements before F3 existed. panels/investigation.tsx now ships a real
    // ConversationInvestigation and SessionPicker, and this panel was shadowing both -- registry.ts
    // breaks an `order` tie with id.localeCompare, so "diagnostics" beat "investigation" and a live
    // surface was unreachable by string comparison.
    //
    // Worse, claiming the deep link handed this panel a sessionId with no runId, so RunDiagnostics
    // scanned only the loaded page of runs and fell back to a default one on a miss: you asked for a
    // session and silently got an unrelated run (STUDIO_UI_AUDIT 3.6, 5).
    expect(panel.match("#/sessions/session_alpha/investigate")).toBeNull();
    expect(panel.match("#/investigation")).toBeNull();
  });

  test("does not claim Lens or the runs index", () => {
    expect(panel.match("#/runs/run_alpha/lens")).toBeNull();
    expect(panel.match("#/runs")).toBeNull();
  });
});
