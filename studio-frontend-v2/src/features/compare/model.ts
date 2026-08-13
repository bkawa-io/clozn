/**
 * Compare deliberately accepts recorded structure rather than deriving a string diff in the client.
 * The coordinates are UTF-16 offsets, matching browser text selection and DOM string indexing.
 */
export interface TextRange {
  start: number;
  end: number;
}

export type OutputState = "available" | "unavailable" | "redacted";

export interface ComparedExecution {
  id: string;
  label: string;
  outputState: OutputState;
  output?: string;
  model?: string;
  recordedAt?: string;
  conditions?: readonly { label: string; value: string }[];
}

export type PairRelationship =
  | { kind: "arbitrary"; detail?: string; changedConditions?: readonly string[] }
  | {
    kind: "related";
    detail?: string;
    /** A recorded fork/change coordinate. This is intentionally distinct from output divergence. */
    intervention?: { label: string; detail: string };
    changedConditions?: readonly string[];
  };

/** Alignment describes only recorded navigation correspondence, never meaning or mechanism. */
export type AlignmentState = "recorded" | "ambiguous" | "unavailable";

export type StructuralDifferenceKind = "changed" | "inserted" | "deleted" | "region";

export interface StructuralDifference {
  id: string;
  label: string;
  kind: StructuralDifferenceKind;
  /** A region can be absent on one side for an insertion/deletion. */
  a?: TextRange;
  b?: TextRange;
  alignment: AlignmentState;
  alignmentDetail?: string;
  /** This is a recorded output coordinate, not a causal coordinate. */
  isFirstOutputDivergence?: boolean;
}

export type LocalEvidenceState = "available" | "not_measured" | "unavailable" | "error";

export interface LocalEvidence {
  state: LocalEvidenceState;
  reason?: string;
  /** Recorded observations, labelled by their source. Their presence does not establish causality. */
  observations?: readonly { label: string; value: string; provenance?: string }[];
  method?: string;
}

export interface ComparisonSpecimen {
  a: ComparedExecution;
  b: ComparedExecution;
  relationship: PairRelationship;
  /** Only regions supplied by a recorded comparison artifact appear in this navigation spine. */
  differences: readonly StructuralDifference[];
  /** Evidence is keyed to a recorded difference-region id. Omit it when no local evidence was requested. */
  evidenceByDifferenceId?: Readonly<Record<string, LocalEvidence>>;
}

export interface CompareSelection {
  runAId: string;
  runBId: string;
  differenceId: string;
  a?: TextRange;
  b?: TextRange;
}

export function validRange(text: string | undefined, range: TextRange | undefined): TextRange | undefined {
  if (!text || !range || !Number.isInteger(range.start) || !Number.isInteger(range.end)) return undefined;
  if (range.start < 0 || range.end <= range.start || range.end > text.length) return undefined;
  return range;
}

export function usableDifferences(specimen: ComparisonSpecimen): StructuralDifference[] {
  return specimen.differences.filter((difference) => difference.id.trim().length > 0);
}

export interface DifferenceTextPart {
  text: string;
  differenceId?: string;
}

/**
 * Partition output using one side's recorded regions. Overlapping legacy coordinates are skipped in
 * stable order rather than duplicating text or implying precedence that the artifact did not record.
 */
export function differenceTextParts(text: string | undefined, differences: readonly StructuralDifference[], side: "a" | "b"): DifferenceTextPart[] {
  if (!text) return [];
  const coordinates = differences
    .map((difference) => ({ id: difference.id, range: validRange(text, difference[side]) }))
    .filter((entry): entry is { id: string; range: TextRange } => entry.range !== undefined)
    .sort((left, right) => left.range.start - right.range.start || left.range.end - right.range.end || left.id.localeCompare(right.id));
  const parts: DifferenceTextPart[] = [];
  let cursor = 0;
  for (const coordinate of coordinates) {
    if (coordinate.range.start < cursor) continue;
    if (coordinate.range.start > cursor) parts.push({ text: text.slice(cursor, coordinate.range.start) });
    parts.push({ text: text.slice(coordinate.range.start, coordinate.range.end), differenceId: coordinate.id });
    cursor = coordinate.range.end;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) });
  return parts;
}

export function selectedDifference(specimen: ComparisonSpecimen, id: string | undefined): StructuralDifference | undefined {
  return id ? usableDifferences(specimen).find((difference) => difference.id === id) : undefined;
}
