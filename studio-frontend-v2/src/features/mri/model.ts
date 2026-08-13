/**
 * Model MRI is deliberately a coordinate viewer, not an activation heat map.
 * A cell represents one recorded claim about a stable token × layer locus.
 */
export type MriSurfacePhase = "loading" | "ready" | "unavailable" | "error" | "stale";

export type MriChannelFamily = "residual" | "readout" | "feature" | "attention" | "causal" | "decision";

export type MriChannelKind =
  | "residual-magnitude"
  | "j-lens"
  | "sae-features"
  | "attention-routing"
  | "attention-read-effect"
  | "attention-write-effect"
  | "causal-intervention"
  | "decision-distribution";

export type MriCapabilityState = "available" | "runtime-unsupported" | "artifact-unavailable" | "not-qualified" | "not-reported";
export type MriArtifactMode = "recorded" | "replayed" | "recomputed";

export interface MriLocus {
  readonly runId: string;
  readonly sequenceId: string;
  readonly tokenIndex: number;
  readonly layerIndex: number;
}

export interface MriToken {
  readonly index: number;
  /** The readable token piece as retained by the run. */
  readonly text: string;
  readonly label?: string;
}

export interface MriLayer {
  readonly index: number;
  readonly label?: string;
}

export interface MriChannel {
  readonly id: string;
  readonly label: string;
  readonly kind: MriChannelKind;
  readonly family: MriChannelFamily;
  readonly capability: MriCapabilityState;
  readonly reason?: string;
  /** Required when a channel is available; a current analysis is not silently historical. */
  readonly artifactMode?: MriArtifactMode;
  readonly method?: string;
  readonly artifactIdentity?: string;
}

export type MriEvidence =
  | { readonly kind: "measured"; readonly finding: "supported" | "unsupported"; readonly detail?: string }
  | { readonly kind: "not-measured"; readonly reason?: string }
  | { readonly kind: "unavailable"; readonly reason?: string }
  | { readonly kind: "failed"; readonly reason?: string }
  | { readonly kind: "omitted"; readonly reason?: string };

/** A source-position reference is only meaningful for the attention-routing instrument. */
export interface MriSourceReference {
  readonly tokenIndex: number;
  readonly label?: string;
}

export interface MriObservation {
  readonly locus: MriLocus;
  readonly evidence: MriEvidence;
  /** Human-readable labels emitted by the selected instrument, e.g. SAE feature annotations. */
  readonly findings?: readonly string[];
  readonly sourceTokens?: readonly MriSourceReference[];
}

export interface MriSpecimen {
  readonly runId: string;
  readonly sequenceId: string;
  readonly tokens: readonly MriToken[];
  readonly layers: readonly MriLayer[];
  readonly channels: readonly MriChannel[];
  /** Observations are grouped by channel so data from unlike instruments cannot be fused. */
  readonly observationsByChannelId?: Readonly<Record<string, readonly MriObservation[] | undefined>>;
  readonly recordedAt?: string;
}

export function locusKey(locus: MriLocus): string {
  return `${encodeURIComponent(locus.runId)}:${encodeURIComponent(locus.sequenceId)}:${locus.tokenIndex}:${locus.layerIndex}`;
}

export function sameLocus(left: MriLocus | undefined, right: MriLocus | undefined): boolean {
  return left !== undefined && right !== undefined && locusKey(left) === locusKey(right);
}

export function evidenceLabel(evidence: MriEvidence | undefined): string {
  if (!evidence) return "Not captured";
  if (evidence.kind === "measured") return evidence.finding === "supported" ? "Measured" : "Measured, unsupported";
  return {
    "not-measured": "Not measured",
    unavailable: "Unavailable",
    failed: "Measurement failed",
    omitted: "Omitted",
  }[evidence.kind];
}

export function evidenceTone(evidence: MriEvidence | undefined): "measured" | "unsupported" | "neutral" | "failed" {
  if (!evidence) return "neutral";
  if (evidence.kind === "measured") return evidence.finding === "supported" ? "measured" : "unsupported";
  return evidence.kind === "failed" ? "failed" : "neutral";
}

export function capabilityLabel(capability: MriCapabilityState): string {
  return {
    available: "Available",
    "runtime-unsupported": "Runtime unsupported",
    "artifact-unavailable": "Artifact unavailable",
    "not-qualified": "Not qualified",
    "not-reported": "Not reported",
  }[capability];
}

export function channelGuide(kind: MriChannelKind): string {
  return {
    "residual-magnitude": "Residual magnitude is a recorded quantity, not a causal importance claim.",
    "j-lens": "J-lens reports a model-scoped readout candidate at this locus; it is not the recorded forward pass unless marked recorded.",
    "sae-features": "SAE rows name activated stored features. Feature presence is not a causal claim.",
    "attention-routing": "Routing mass describes where attention routed. It does not establish causal importance.",
    "attention-read-effect": "Read effect records the result of disrupting routing. It is distinct from routing mass and write effect.",
    "attention-write-effect": "Write effect records the result of ablating a head output. It is distinct from routing mass and read effect.",
    "causal-intervention": "Only this instrument may support a causal intervention claim at a measured target.",
    "decision-distribution": "Decision distribution is a per-token output instrument; it is not an internal activation map.",
  }[kind];
}

export function observationsFor(channelId: string | undefined, specimen?: MriSpecimen): readonly MriObservation[] {
  return channelId && specimen?.observationsByChannelId?.[channelId] ? specimen.observationsByChannelId[channelId]! : [];
}

export function observationAt(observations: readonly MriObservation[], locus: MriLocus): MriObservation | undefined {
  return observations.find((observation) => sameLocus(observation.locus, locus));
}

export function defaultMriLocus(specimen?: MriSpecimen): MriLocus | undefined {
  const token = specimen?.tokens[0];
  const layer = specimen?.layers[0];
  return specimen && token && layer ? { runId: specimen.runId, sequenceId: specimen.sequenceId, tokenIndex: token.index, layerIndex: layer.index } : undefined;
}
