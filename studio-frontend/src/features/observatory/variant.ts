import type { ObservatoryData } from "../../data/types";

export interface DialDifference {
  name: string;
  reference: number;
  current: number;
}

export interface VariantRelation {
  referenceLabel: string;
  currentLabel: string;
  kind: "adapter" | "steering" | "memory" | "fork" | "model" | "run";
  evidence: string;
}

export function dialDifferences(reference: ObservatoryData, current: ObservatoryData): DialDifference[] {
  const names = new Set([
    ...Object.keys(reference.configuration.activeDials),
    ...Object.keys(current.configuration.activeDials),
  ]);
  return [...names].map((name) => ({
    name,
    reference: reference.configuration.activeDials[name] ?? 0,
    current: current.configuration.activeDials[name] ?? 0,
  })).filter((item) => Math.abs(item.current - item.reference) > .0001)
    .sort((a, b) => Math.abs(b.current - b.reference) - Math.abs(a.current - a.reference));
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
  if (dialDifferences(reference, current).length) {
    return {
      referenceLabel: Object.keys(reference.configuration.activeDials).length ? "REFERENCE STEERING" : "BASE RUN",
      currentLabel: Object.keys(current.configuration.activeDials).length ? "STEERING VARIANT" : "CURRENT RUN",
      kind: "steering",
      evidence: "RECORDED DIAL DIFFERENCE",
    };
  }
  if (
    !sameStrings(reference.configuration.memoryCards, current.configuration.memoryCards)
    || reference.configuration.memoryStrength !== current.configuration.memoryStrength
  ) {
    return {
      referenceLabel: "REFERENCE MEMORY",
      currentLabel: "MEMORY VARIANT",
      kind: "memory",
      evidence: "RECORDED MEMORY DIFFERENCE",
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
