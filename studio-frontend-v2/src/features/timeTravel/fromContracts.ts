import type { RewindFidelity, RunRecord } from "../../data/contracts";
import type { BoundaryFidelity, TimeTravelRun } from "./model";

function detail(values: readonly string[] | undefined): string | undefined {
  return values?.length ? values.map((value) => value.replaceAll("_", " ")).join(" · ") : undefined;
}

/** Project only recorded run tokens and read-only rewind fidelity; no live exactness is inferred. */
export function projectTimeTravelRun(run: RunRecord, fidelity: RewindFidelity): TimeTravelRun {
  const historical = new Map(fidelity.historicalProof.verifiedBoundaries.map((boundary) => [boundary.position, boundary]));
  const tokenCount = Math.min(run.responseTokens?.length ?? 0, fidelity.recordedTokenCount ?? Number.MAX_SAFE_INTEGER);
  const base = (position: number): BoundaryFidelity => {
    const proof = historical.get(position);
    return {
      reconstructedReplay: fidelity.reconstructedReplay.state === "available" ? {
        state: "available",
        unavoidableDifferences: fidelity.reconstructedReplay.unavoidableDifferences,
      } : {
        state: "unavailable",
        reason: detail(fidelity.reconstructedReplay.reasons),
      },
      exactFork: fidelity.exactRewind.state === "requires_live_plan" ? {
        state: "requires_live_plan",
        requirements: fidelity.exactRewind.liveRequirements,
      } : {
        state: "unavailable",
        reason: detail(fidelity.exactRewind.reasons),
      },
      historicalExactProof: proof ? {
        state: "verified",
        verifiedExecutionCount: proof.verifiedExecutionCount,
        detail: `${proof.latestExecutionId} · ${proof.regimes.join(" · ")}`,
      } : { state: "none" },
    };
  };
  return {
    id: run.id,
    response: run.response ?? "",
    responseTokens: run.responseTokens?.slice(0, tokenCount).map((text, position) => ({ position, text })),
    model: run.model ?? undefined,
    parentRunId: run.parentRunId ?? undefined,
    sessionKey: run.sessionKey ?? undefined,
    fidelityByBoundary: Object.fromEntries(Array.from({ length: tokenCount }, (_, position) => [position, base(position)])),
  };
}
