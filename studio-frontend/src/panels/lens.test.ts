import { describe, expect, test } from "vitest";
import panel, { matchRunReaderRoute } from "./lens";

describe("run reader panel routing", () => {
  test("uses #/runs/<id> as the canonical reader and keeps the prior Lens path compatible", () => {
    expect(panel.navLabel).toBe("Run");
    expect(panel.match("#/lens")).toEqual({ section: "read" });
    expect(panel.match("#/runs/run_alpha/lens")).toEqual({ runId: "run_alpha", section: "read" });
    expect(panel.match("#/runs/run%2Falpha/lens/")).toEqual({ runId: "run/alpha", section: "read" });
    expect(panel.match("#/runs/run_alpha")).toEqual({ runId: "run_alpha", section: "read" });
    expect(panel.match("#/runs/run_alpha?section=timing")).toEqual({ runId: "run_alpha", section: "timing" });
  });

  test("maps old run-specific Diagnostics and Scope links into their S2 instruments", () => {
    expect(matchRunReaderRoute("#/runs/run_alpha/diagnostics/runtime")).toEqual({
      runId: "run_alpha",
      section: "timing",
    });
    expect(matchRunReaderRoute("#/runs/run_alpha/diagnostics/raw")).toEqual({
      runId: "run_alpha",
      section: "record",
    });
    expect(matchRunReaderRoute("#/runs/run_alpha/scope?token=4&view=layers")).toEqual(expect.objectContaining({
      runId: "run_alpha",
      section: "mechanism",
      token: "4",
      tokenIndex: "4",
      view: "layers",
    }));
  });

  test("does not claim the runs index or session investigation links", () => {
    expect(panel.match("#/runs")).toBeNull();
    expect(panel.match("#/sessions/session_alpha/investigate")).toBeNull();
  });
});
