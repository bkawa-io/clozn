import type {
  InfluenceEvidenceState,
  TokenReading,
  TokenSourceReading,
} from "../../data/types";

export interface ResponseClaim {
  start: number;
  end: number;
  text: string;
  tokenCount: number;
  shakyCount: number;
  linkedCount: number;
  meanConfidence?: number;
}

export interface RangeSummary {
  start: number;
  end: number;
  text: string;
  tokenCount: number;
  shakyCount: number;
  linkedCount: number;
  meanConfidence?: number;
}

export interface SourceAggregate {
  sourceId: string;
  label: string;
  effect: TokenSourceReading["effect"];
  deltaNats: number;
  tokenCount: number;
  clearTokenCount: number;
  observedTokenCount: number;
  evidenceStates: InfluenceEvidenceState[];
}

export interface InfluenceSplit {
  supports: number;
  suppresses: number;
  neutral: number;
}

function breakCount(text: string) {
  return text.match(/\r\n|\r|\n/g)?.length ?? 0;
}

function lastRecordedIndex(tokens: TokenReading[]) {
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    if (tokens[index].text) return index;
  }
  return -1;
}

function isClaimBoundary(tokens: TokenReading[], index: number, lastIndex: number) {
  const text = tokens[index]?.text ?? "";
  if (breakCount(text) >= 2) return true;

  const trimmed = text.trimEnd();
  if (/[!?]["')\]]*$/.test(trimmed)) return true;
  if (!/\.[."')\]]*$/.test(trimmed)) return false;
  if (index === lastIndex) return true;

  const nextText = tokens.slice(index + 1, lastIndex + 1)
    .map((token) => token.text)
    .find((candidate) => candidate.trim().length);
  return nextText ? /^\s*["'([{]*[A-Z]/.test(nextText) : true;
}

function meanConfidence(tokens: TokenReading[]) {
  const values = tokens
    .map((token) => token.confidence)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!values.length) return undefined;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function summarizeRange(tokens: TokenReading[], start: number, end: number): RangeSummary {
  const safeStart = Math.max(0, Math.min(start, tokens.length - 1));
  const safeEnd = Math.max(safeStart, Math.min(end, tokens.length - 1));
  const selected = tokens.slice(safeStart, safeEnd + 1);
  return {
    start: safeStart,
    end: safeEnd,
    text: selected.map((token) => token.text).join("").trim(),
    tokenCount: selected.length,
    shakyCount: selected.filter((token) => token.band === "shaky").length,
    linkedCount: selected.filter((token) => token.sources?.length).length,
    meanConfidence: meanConfidence(selected),
  };
}

export function buildResponseClaims(tokens: TokenReading[]): ResponseClaim[] {
  const lastIndex = lastRecordedIndex(tokens);
  if (lastIndex < 0) return [];

  const claims: ResponseClaim[] = [];
  let start = 0;
  for (let index = 0; index <= lastIndex; index += 1) {
    if (!isClaimBoundary(tokens, index, lastIndex)) continue;
    const summary = summarizeRange(tokens, start, index);
    if (summary.text) claims.push(summary);
    start = index + 1;
  }
  if (start <= lastIndex) {
    const summary = summarizeRange(tokens, start, lastIndex);
    if (summary.text) claims.push(summary);
  }
  return claims;
}

export function aggregateSources(tokens: TokenReading[], start: number, end: number): SourceAggregate[] {
  const aggregates = new Map<string, {
    sourceId: string;
    label: string;
    deltaNats: number;
    tokenIndexes: Set<number>;
    clearTokenIndexes: Set<number>;
    observedTokenIndexes: Set<number>;
  }>();

  for (let index = Math.max(0, start); index <= Math.min(end, tokens.length - 1); index += 1) {
    if (!tokens[index].text) continue;
    const readings = [
      ...(tokens[index].sources ?? []).map((source) => ({ source, clear: true })),
      ...(tokens[index].observedSources ?? []).map((source) => ({ source, clear: false })),
    ];
    for (const { source, clear } of readings) {
      const aggregate = aggregates.get(source.sourceId) ?? {
        sourceId: source.sourceId,
        label: source.label,
        deltaNats: 0,
        tokenIndexes: new Set<number>(),
        clearTokenIndexes: new Set<number>(),
        observedTokenIndexes: new Set<number>(),
      };
      aggregate.deltaNats += source.deltaNats;
      aggregate.tokenIndexes.add(index);
      (clear ? aggregate.clearTokenIndexes : aggregate.observedTokenIndexes).add(index);
      aggregates.set(source.sourceId, aggregate);
    }
  }

  return [...aggregates.values()]
    .map((source): SourceAggregate => ({
      sourceId: source.sourceId,
      label: source.label,
      deltaNats: source.deltaNats,
      tokenCount: source.tokenIndexes.size,
      clearTokenCount: source.clearTokenIndexes.size,
      observedTokenCount: source.observedTokenIndexes.size,
      evidenceStates: [
        ...(source.clearTokenIndexes.size ? ["causally_supported" as const] : []),
        ...(source.observedTokenIndexes.size ? ["observed" as const] : []),
      ],
      effect: source.deltaNats > 0 ? "supports" : source.deltaNats < 0 ? "suppresses" : "neutral",
    }))
    .sort((a, b) => Math.abs(b.deltaNats) - Math.abs(a.deltaNats));
}

export function weakestTokenInRange(tokens: TokenReading[], start: number, end: number) {
  let weakest = Math.max(0, start);
  for (let index = weakest + 1; index <= Math.min(end, tokens.length - 1); index += 1) {
    const confidence = tokens[index].confidence;
    const weakestConfidence = tokens[weakest].confidence;
    if (
      typeof confidence === "number"
      && (typeof weakestConfidence !== "number" || confidence < weakestConfidence)
    ) {
      weakest = index;
    }
  }
  return weakest;
}

export function influenceSplit(tokens: TokenReading[]): InfluenceSplit {
  return tokens.reduce(
    (counts, token) => {
      if (!token.text) return counts;
      const dominant = [...(token.sources ?? [])].sort(
        (a, b) => Math.abs(b.deltaNats) - Math.abs(a.deltaNats),
      )[0];
      if (dominant) counts[dominant.effect] += 1;
      return counts;
    },
    { supports: 0, suppresses: 0, neutral: 0 },
  );
}
