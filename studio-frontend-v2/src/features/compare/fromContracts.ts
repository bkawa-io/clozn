import type { RunComparison, RunRecord } from "../../data/contracts";
import type { ComparedExecution, ComparisonSpecimen, LocalEvidence, PairRelationship, StructuralDifference, TextRange } from "./model";

export interface RecordedComparisonStructure {
  relationship: PairRelationship;
  differences: readonly StructuralDifference[];
  evidenceByDifferenceId?: ComparisonSpecimen["evidenceByDifferenceId"];
}

function label(run: RunRecord, fallback: string): string {
  return run.promptSummary?.trim() || run.id || fallback;
}

function output(run: RunRecord, fallback: string): ComparedExecution {
  // RunRecord intentionally does not assign a meaning to null response text. Keep it unavailable
  // instead of guessing that privacy redaction, an empty response, or a failed capture occurred.
  const state = typeof run.response === "string" ? "available" : "unavailable";
  return {
    id: run.id,
    label: label(run, fallback),
    outputState: state,
    output: typeof run.response === "string" ? run.response : undefined,
    model: run.model ?? undefined,
    recordedAt: run.createdAt ?? (run.createdTs == null ? undefined : new Date(run.createdTs * 1_000).toISOString()),
  };
}

/**
 * Maps only fields already present on a run. Structural comparison data remains an explicit input:
 * the browser must never manufacture alignment or first-divergence evidence from raw strings.
 */
export function projectComparisonSpecimen(
  runA: RunRecord,
  runB: RunRecord,
  structure: RecordedComparisonStructure,
): ComparisonSpecimen {
  return {
    a: output(runA, "Run A"),
    b: output(runB, "Run B"),
    relationship: structure.relationship,
    differences: structure.differences,
    evidenceByDifferenceId: structure.evidenceByDifferenceId,
  };
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function browserRange(text: string | null | undefined, value: unknown): TextRange | undefined {
  if (typeof text !== "string") return undefined;
  const row = record(value);
  if (row?.state !== "exact" || !Number.isInteger(row.start) || !Number.isInteger(row.end)) return undefined;
  const points = Array.from(text);
  const start = row.start as number;
  const end = row.end as number;
  if (start < 0 || end < start || end > points.length) return undefined;
  return { start: points.slice(0, start).join("").length, end: points.slice(0, end).join("").length };
}

function divergenceRanges(comparison: RunComparison, runA: RunRecord, runB: RunRecord): { a?: TextRange; b?: TextRange } {
  const output = comparison.differences.find((difference) => difference.dimension === "output.text");
  const tokenDiff = record(output?.evidence[0]);
  const view = record(tokenDiff?.first_divergence_view);
  const locations = record(view?.recorded_answer_location);
  return {
    a: browserRange(runA.response, locations?.a),
    b: browserRange(runB.response, locations?.b),
  };
}

/**
 * Adapt the server's recorded structural comparison without re-diffing text in the browser.
 * Only its exact first-divergence locations become prose highlights; every other dimension remains a
 * selectable structural record with no invented text coordinate.
 */
export function recordedStructureFromComparison(
  comparison: RunComparison,
  runA: RunRecord,
  runB: RunRecord,
): RecordedComparisonStructure {
  const related = runB.parentRunId === runA.id || runA.parentRunId === runB.id;
  const changedConditions = Object.entries(comparison.axes).filter(([, axis]) => axis.status === "changed").map(([axis]) => axis);
  const relationship: PairRelationship = related ? {
    kind: "related",
    detail: runB.parentRunId === runA.id ? `${runB.id} records ${runA.id} as its parent.` : `${runA.id} records ${runB.id} as its parent.`,
    changedConditions,
  } : {
    kind: "arbitrary",
    detail: comparison.selection?.reason,
    changedConditions,
  };
  const ranges = divergenceRanges(comparison, runA, runB);
  // Keep metadata conditions above the prose reader. The spine is exclusively an output navigator.
  const outputDifferences = comparison.differences.map((difference, index) => ({ difference, index }))
    .filter(({ difference }) => difference.dimension.startsWith("output."))
    .sort((left, right) => Number(right.difference.dimension === "output.text") - Number(left.difference.dimension === "output.text") || left.difference.rank - right.difference.rank);
  const differences: StructuralDifference[] = outputDifferences.map(({ difference, index }) => ({
    id: `recorded-difference-${index}`,
    label: difference.dimension.replaceAll(".", " / "),
    kind: difference.kind === "added" ? "inserted" : difference.kind === "removed" ? "deleted" : "changed",
    ...(difference.dimension === "output.text" ? ranges : {}),
    alignment: difference.dimension === "output.text" && (ranges.a || ranges.b) ? "recorded" : "unavailable",
    alignmentDetail: difference.note,
    isFirstOutputDivergence: difference.dimension === "output.text" && Boolean(ranges.a || ranges.b),
  }));
  const evidenceByDifferenceId: Record<string, LocalEvidence> = {};
  outputDifferences.forEach(({ difference, index }) => {
    const relevant = comparison.findings.filter((finding) => finding.dimensions.includes(difference.dimension));
    evidenceByDifferenceId[`recorded-difference-${index}`] = difference.kind === "unavailable" || difference.kind === "diff_failed" ? {
      state: "unavailable",
      reason: difference.note,
    } : {
      state: "available",
      method: "clozn.run-diff.v1 · recorded structural comparison",
      observations: relevant.length ? relevant.map((finding) => ({
        label: finding.status.replaceAll("_", " "),
        value: finding.summary,
        provenance: finding.classification,
      })) : [{ label: "Recorded change", value: difference.kind, provenance: difference.dimension }],
    };
  });
  return { relationship, differences, evidenceByDifferenceId };
}
