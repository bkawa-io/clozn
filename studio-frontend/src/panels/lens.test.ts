import { describe, expect, test } from "vitest";
import panel from "./lens";

describe("lens panel routing", () => {
  test("uses an explicit canonical reader route and preserves bare run links", () => {
    expect(panel.match("#/lens")).toEqual({});
    expect(panel.match("#/runs/run_alpha/lens")).toEqual({ runId: "run_alpha" });
    expect(panel.match("#/runs/run%2Falpha/lens/")).toEqual({ runId: "run/alpha" });
    expect(panel.match("#/runs/run_alpha")).toEqual({ runId: "run_alpha" });
  });

  test("does not claim diagnostics or old Scope links", () => {
    expect(panel.match("#/runs/run_alpha/diagnostics")).toBeNull();
    expect(panel.match("#/runs/run_alpha/scope")).toBeNull();
  });
});
