export type InfluenceEffect = "supports" | "suppresses" | "neutral";
export type InfluenceEvidenceState = "causally_supported" | "observed";

export interface TextLocus {
  id: string;
  start: number;
  end: number;
  label?: string;
  /** Atomic backend answer-span ids represented by this readable selection phrase. */
  memberIds?: readonly string[];
}

export interface ContextDocument {
  id: string;
  label: string;
  text?: string;
  state: "available" | "omitted" | "unavailable";
  detail?: string;
}

export interface RelatedContextLocus extends TextLocus {
  documentId: string;
  answerLocusId?: string;
  effect: InfluenceEffect;
  deltaNats: number;
  evidenceState: InfluenceEvidenceState;
}

export interface InfluenceSelection {
  state: "loading" | "available" | "not_measured" | "unavailable" | "error";
  reason?: string;
  method?: string;
  floorNats?: number;
  related: RelatedContextLocus[];
}

export interface LinkedReaderSpecimen {
  runId: string;
  answer: string;
  answerLoci: TextLocus[];
  context: ContextDocument[];
}

export interface DecisionLocus {
  id: string;
  position: number;
  start?: number;
  end?: number;
  emittedToken?: string;
  emittedProbability?: number;
  rivalTokenId?: number;
  rivalProbability?: number;
  margin?: number;
  meaningful: boolean;
}

export function validTextLoci(text: string, loci: readonly TextLocus[]): TextLocus[] {
  return loci
    .filter((locus) => Number.isInteger(locus.start) && Number.isInteger(locus.end)
      && locus.start >= 0 && locus.end > locus.start && locus.end <= text.length)
    .sort((left, right) => left.start - right.start || left.end - right.end || left.id.localeCompare(right.id));
}

export interface TextFragment {
  text: string;
  locus?: TextLocus;
}

/** Partition readable prose without dropping punctuation or inventing overlap precedence. */
export function textFragments(text: string, loci: readonly TextLocus[]): TextFragment[] {
  const usable = validTextLoci(text, loci);
  const fragments: TextFragment[] = [];
  let cursor = 0;
  for (const locus of usable) {
    // Overlaps can occur in legacy artifacts. The reader keeps the earlier stable coordinate instead
    // of duplicating prose or silently splicing two incompatible selections over the same characters.
    if (locus.start < cursor) continue;
    if (locus.start > cursor) fragments.push({ text: text.slice(cursor, locus.start) });
    fragments.push({ text: text.slice(locus.start, locus.end), locus });
    cursor = locus.end;
  }
  if (cursor < text.length) fragments.push({ text: text.slice(cursor) });
  return fragments;
}
