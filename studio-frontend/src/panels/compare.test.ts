import { describe, expect, test } from "vitest";
import comparePanel from "./compare";
import experimentsPanel from "./experiments";

describe("compare panel routing", () => {
  test("keeps a run pair route while exposing the canonical matrix mode", () => {
    expect(comparePanel.match("#/compare/run%2Fa/run-b")).toEqual({ runA: "run/a", runB: "run-b" });
    expect(comparePanel.match("#/compare/matrix/experiment-7?suite=target&status=fail")).toEqual({
      mode: "matrix",
      matrixBase: "canonical",
      id: "experiment-7",
      q: "suite=target&status=fail",
    });
  });

  test("claims legacy experiment URLs as Compare matrix mode and hides the former standalone nav item", () => {
    expect(comparePanel.match("#/experiments/experiment-7?variant=candidate")).toEqual({
      mode: "matrix",
      matrixBase: "legacy",
      id: "experiment-7",
      q: "variant=candidate",
    });
    expect(experimentsPanel.hiddenFromNav).toBe(true);
    expect(experimentsPanel.routeName({})).toBe("COMPARE MATRIX");
  });
});
