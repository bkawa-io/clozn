import { describe, expect, it } from "vitest";
import { capabilityLabel, defaultMriLocus, evidenceLabel, locusKey, observationAt } from "./model";

describe("Model MRI model", () => {
  it("keys stable coordinates without readable labels", () => {
    expect(locusKey({ runId: "run / 1", sequenceId: "answer:0", tokenIndex: 4, layerIndex: 12 })).toBe("run%20%2F%201:answer%3A0:4:12");
  });

  it("does not turn absent observations into zeros or findings", () => {
    const locus = { runId: "run_1", sequenceId: "answer", tokenIndex: 1, layerIndex: 2 };
    expect(observationAt([], locus)).toBeUndefined();
    expect(evidenceLabel(undefined)).toBe("Not captured");
  });

  it("keeps unsupported evidence and unavailable capability distinct", () => {
    expect(evidenceLabel({ kind: "measured", finding: "unsupported" })).toBe("Measured, unsupported");
    expect(capabilityLabel("artifact-unavailable")).toBe("Artifact unavailable");
  });

  it("uses the first real token and layer as a selection fallback", () => {
    expect(defaultMriLocus({ runId: "run_1", sequenceId: "answer", tokens: [{ index: 8, text: "Hello" }], layers: [{ index: 4 }], channels: [] })).toEqual({ runId: "run_1", sequenceId: "answer", tokenIndex: 8, layerIndex: 4 });
  });
});
