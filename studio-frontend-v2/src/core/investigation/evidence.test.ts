import { describe, expect, it } from "vitest";
import { evidenceExactnessLabel, evidenceStateLabel } from "./evidence";

describe("evidence states", () => {
  it("does not collapse below-floor into unsupported", () => {
    expect(evidenceStateLabel({ measurement: { kind: "below-floor", floor: 0.1 } })).toBe("Observed below measurement floor");
    expect(evidenceStateLabel({ measurement: { kind: "measured", finding: "unsupported" } })).toBe("Measured, unsupported");
  });

  it("labels historical verification separately from exact proof", () => {
    expect(evidenceExactnessLabel({ kind: "exact" })).toBe("Exact");
    expect(evidenceExactnessLabel({ kind: "historical" })).toBe("Historically verified");
  });
});
