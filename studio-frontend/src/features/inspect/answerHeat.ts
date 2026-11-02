import type { ContextInvestigationLocus, UnicodeRange } from "../../data/contextInvestigation";
import type { CodePointRange } from "./unicodeRanges";

export type AnswerLens = "influence" | "tension" | "suppression";

export function answerDisplayScale(scores: readonly number[]) {
  const sorted = [...scores].sort((a, b) => a - b);
  const peak = sorted.at(-1) ?? 0;
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[middle] : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
  const floor = 0.1; // Display floor in nats/token. Avoid amplifying tiny effects.
  return {
    ceiling: Math.max(floor, peak), peak, median,
    // Editorial cue only. Heat and reported scores retain all measured values.
    standout: sorted.length > 1 && peak >= floor && peak >= 1.5 * median,
    peakCount: sorted.filter(score => score === peak).length,
  };
}
export interface AnswerHeat {
  support: number;
  suppression: number;
  total: number;
  intensity: number;
  tension: number;
  tokenCount: number;
  direction: "support" | "suppression" | "mixed";
}

// Reader loci contain only above-floor token deltas. Keep both signs: a source
// can support some words and suppress others. Deduplicate by source/token so
// grouping the same evidence differently cannot change the visualization.
export function signedContributions(
  loci: readonly ContextInvestigationLocus[],
  tokens: readonly { unicodeRange: UnicodeRange }[],
  range: CodePointRange,
): Map<string, Map<number, number>> {
  const sources = new Map<string, Map<number, number>>();
  for (const locus of loci) {
    const values = sources.get(locus.sourceId) ?? new Map<number, number>();
    locus.perTokenDeltaNats.forEach((delta, offset) => {
      const index = locus.answerTokenRange[0] + offset;
      const token = tokens[index];
      if (token && token.unicodeRange[0] < range.end && token.unicodeRange[1] > range.start) values.set(index, delta);
    });
    if (values.size) sources.set(locus.sourceId, values);
  }
  return sources;
}

export function answerHeatForRange(
  loci: readonly ContextInvestigationLocus[],
  tokens: readonly { unicodeRange: UnicodeRange }[],
  range: CodePointRange,
): AnswerHeat {
  const perToken = new Map<number, { positive: number; negative: number }>();
  for (const values of signedContributions(loci, tokens, range).values()) {
    for (const [index, delta] of values) {
      const sides = perToken.get(index) ?? { positive: 0, negative: 0 };
      if (delta > 0) sides.positive += delta;
      else sides.negative -= delta;
      perToken.set(index, sides);
    }
  }
  let support = 0, suppression = 0, tension = 0;
  for (const { positive, negative } of perToken.values()) {
    support += positive;
    suppression += negative;
    const minority = Math.min(positive, negative);
    const total = positive + negative;
    // Opposing effects must occur on the same answer token. Effects on separate
    // words alone are not evidence of competing context for the same choice.
    tension += total ? 2 * minority * minority / total : 0;
  }
  const tokenCount = tokens.filter((token) => token.unicodeRange[0] < range.end && token.unicodeRange[1] > range.start).length;
  const total = support + suppression;
  const balance = total ? Math.min(support, suppression) / Math.max(support, suppression) : 0;
  return {
    support, suppression, total, tokenCount, tension,
    intensity: tokenCount ? total / tokenCount : 0,
    direction: balance >= .35 ? "mixed" : support >= suppression ? "support" : "suppression",
  };
}

export function answerHeatForLens(heat: AnswerHeat, lens: AnswerLens): AnswerHeat {
  const total = lens === "tension" ? heat.tension : lens === "suppression" ? heat.suppression : heat.total;
  return {
    ...heat, total, intensity: heat.tokenCount ? total / heat.tokenCount : 0,
    direction: lens === "tension"
      ? "mixed"
      : lens === "suppression"
        ? "suppression"
        : heat.support > 0 && heat.suppression > 0 ? "mixed" : heat.direction,
  };
}
