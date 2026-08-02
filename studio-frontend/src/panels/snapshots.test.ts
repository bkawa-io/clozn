import { describe, expect, test } from "vitest";
import panel from "./snapshots";

describe("snapshots panel routing", () => {
  test("claims the list route and a run-scoped deep link", () => {
    expect(panel.match("#/snapshots")).toEqual({});
    expect(panel.match("#/snapshots/")).toEqual({});
    expect(panel.match("#/snapshots/run_alpha")).toEqual({ runId: "run_alpha" });
    expect(panel.match("#/runs/run_alpha")).toBeNull();
  });
});
