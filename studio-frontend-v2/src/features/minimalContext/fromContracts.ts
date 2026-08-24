import type { MinimalContextCertificate, MinimalContextResult } from "../../data/contracts";
import type { MinimalContextProjection } from "./model";

export function certificateLabel(certificate: MinimalContextCertificate | null): string {
  if (certificate === "EXACT_MINIMUM") return "Exact minimum";
  if (certificate === "INCLUSION_MINIMUM") return "Inclusion-minimal";
  if (certificate === "BEST_VERIFIED") return "Best verified";
  return "No certificate";
}
export function stoppingReasonLabel(reason: MinimalContextResult["stoppingReason"]): string {
  if (reason === "exact_minimum_proven") return "Exact minimum proven";
  if (reason === "inclusion_minimum_proven") return "Inclusion minimum proven";
  if (reason === "budget_exhausted") return "Budget exhausted";
  if (reason === "control_unavailable") return "Control unavailable";
  if (reason === "search_unavailable") return "Search unavailable";
  if (reason === "cancelled") return "Cancelled";
  return "Search policy complete";
}

export function projectMinimalContext(result: MinimalContextResult): MinimalContextProjection {
  const reduction = result.reduction;
  const accounting = result.experimentAccounting;
  const newCounterfactualExecutions = typeof accounting.new_counterfactual_executions === "number"
    ? accounting.new_counterfactual_executions
    : undefined;
  const reusedObservations = typeof accounting.reused_observations === "number"
    ? accounting.reused_observations
    : undefined;
  return {
    certificate: result.certificate,
    certificateLabel: certificateLabel(result.certificate),
    sourceCount: result.universe.sourceCount ?? result.universe.sourceIds.length,
    retainedSourceCount: reduction.retainedSourceCount,
    removedSourceCount: reduction.removedSourceCount,
    originalPromptTokenCost: reduction.originalPromptTokenCost,
    retainedPromptTokenCost: reduction.retainedPromptTokenCost,
    reductionPercent: reduction.percent,
    newCounterfactualExecutions,
    reusedObservations,
    retained: result.sourceInspection.filter((source) => source.disposition === "retained"),
    removed: result.sourceInspection.filter((source) => source.disposition === "removed"),
  };
}
