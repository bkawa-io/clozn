import type {
  MinimalContextCertificate,
  MinimalContextResult,
  MinimalContextResultSummary,
  MinimalContextSourceInspection,
} from "../../data/contracts";

export interface MinimalContextProjection {
  certificate: MinimalContextCertificate | null;
  certificateLabel: string;
  sourceCount?: number;
  retainedSourceCount?: number;
  removedSourceCount?: number;
  originalPromptTokenCost?: number;
  retainedPromptTokenCost?: number;
  reductionPercent?: number;
  newCounterfactualExecutions?: number;
  reusedObservations?: number;
  retained: readonly MinimalContextSourceInspection[];
  removed: readonly MinimalContextSourceInspection[];
}

export type MinimalContextRecord = MinimalContextResult | MinimalContextResultSummary;
