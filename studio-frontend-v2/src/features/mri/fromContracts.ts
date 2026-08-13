import type { RunRecord } from "../../data/contracts";
import type { MriSpecimen } from "./model";

/**
 * Builds an honest baseline specimen from retained run tokens. The current v2 read boundary does not
 * expose a typed internal-instrument artifact, so every channel is explicitly not reported and there
 * are no synthetic observations.
 */
export function projectRecordedMriSpecimen(run: RunRecord): MriSpecimen {
  return {
    runId: run.id,
    sequenceId: "recorded-response",
    tokens: (run.responseTokens ?? []).map((text, index) => ({ index, text })),
    layers: [],
    recordedAt: run.createdAt ?? undefined,
    channels: [
      { id: "j-lens", label: "J-lens", kind: "j-lens", family: "readout", capability: "not-reported", reason: "No typed J-lens artifact was loaded for this run." },
      { id: "sae", label: "SAE features", kind: "sae-features", family: "feature", capability: "not-reported", reason: "No typed SAE artifact was loaded for this run." },
      { id: "routing", label: "Attention routing", kind: "attention-routing", family: "attention", capability: "not-reported", reason: "Attention routing was not reported by the current read contract." },
      { id: "read-effect", label: "Attention read effect", kind: "attention-read-effect", family: "causal", capability: "not-reported", reason: "No causal read-effect measurement was loaded." },
      { id: "write-effect", label: "Attention write effect", kind: "attention-write-effect", family: "causal", capability: "not-reported", reason: "No causal write-effect measurement was loaded." },
    ],
    observationsByChannelId: {},
  };
}
