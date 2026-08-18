import type { ObservatoryData } from "../../data/types";

export interface VariantRelation {
  referenceLabel: string;
  currentLabel: string;
  kind: "adapter" | "fork" | "model" | "run";
  evidence: string;
}

function sameStrings(a: string[], b: string[]) {
  return [...a].sort().join("\u0000") === [...b].sort().join("\u0000");
}

export function describeVariant(reference: ObservatoryData, current: ObservatoryData): VariantRelation {
  const referenceAdapters = reference.configuration.adapters;
  const currentAdapters = current.configuration.adapters;
  if (!sameStrings(referenceAdapters, currentAdapters)) {
    return {
      referenceLabel: referenceAdapters.length ? "REFERENCE ADAPTER" : "BASE RUN",
      currentLabel: currentAdapters.length ? "ADAPTER VARIANT" : "CURRENT RUN",
      kind: "adapter",
      evidence: "RECORDED ADAPTER IDENTITY",
    };
  }
  if (current.parentRunId === reference.id || reference.parentRunId === current.id) {
    return {
      referenceLabel: current.parentRunId === reference.id ? "PARENT RUN" : "CHILD RUN",
      currentLabel: current.parentRunId === reference.id ? "FORK" : "PARENT RUN",
      kind: "fork",
      evidence: "RECORDED RUN LINEAGE",
    };
  }
  if (reference.model !== current.model) {
    return {
      referenceLabel: "REFERENCE MODEL",
      currentLabel: "MODEL VARIANT",
      kind: "model",
      evidence: "RECORDED MODEL IDENTITY",
    };
  }
  return {
    referenceLabel: "REFERENCE RUN",
    currentLabel: "CURRENT RUN",
    kind: "run",
    evidence: "STRUCTURAL TOKEN ALIGNMENT",
  };
}
