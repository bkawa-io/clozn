/**
 * Evidence is deliberately modelled on independent axes. A missing
 * measurement is not a zero effect, and a reconstructed artifact is not an
 * exact one.
 */
export type EvidenceMeasurement =
  | { readonly kind: "measured"; readonly finding: "supported" | "unsupported"; readonly value?: number }
  | { readonly kind: "below-floor"; readonly floor: number; readonly observed?: number }
  | { readonly kind: "not-measured"; readonly reason?: string }
  | { readonly kind: "unavailable"; readonly reason?: string }
  | { readonly kind: "omitted"; readonly reason?: string }
  | { readonly kind: "failed"; readonly reason?: string };

export type EvidenceArtifactMode = "recorded" | "replayed" | "recomputed";

export type EvidenceExactness =
  | { readonly kind: "exact"; readonly proof?: string }
  | { readonly kind: "reconstructed"; readonly qualification?: string }
  | { readonly kind: "historical"; readonly verifiedAt?: string; readonly qualification?: string }
  | { readonly kind: "unverified" };

export interface EvidenceState {
  readonly measurement: EvidenceMeasurement;
  readonly artifactMode?: EvidenceArtifactMode;
  readonly exactness?: EvidenceExactness;
}

export function evidenceStateLabel(state: EvidenceState): string {
  const measurement = state.measurement;
  if (measurement.kind === "measured") {
    return measurement.finding === "supported" ? "Measured support" : "Measured, unsupported";
  }
  return {
    "below-floor": "Observed below measurement floor",
    "not-measured": "Not measured",
    unavailable: "Unavailable",
    omitted: "Omitted",
    failed: "Measurement failed",
  }[measurement.kind];
}

export function evidenceExactnessLabel(exactness: EvidenceExactness | undefined): string | undefined {
  if (!exactness) return undefined;
  return {
    exact: "Exact",
    reconstructed: "Reconstructed",
    historical: "Historically verified",
    unverified: "Exactness unverified",
  }[exactness.kind];
}
