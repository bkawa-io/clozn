import { describe, expect, it } from "vitest";
import type { ContextInvestigationLocus } from "../../data/contextInvestigation";
import { answerDisplayScale, answerHeatForLens, answerHeatForRange, signedContributions } from "./answerHeat";

const tokens = [0, 1, 2].map((index) => ({ unicodeRange: [index, index + 1] as const }));
const range = { start: 0, end: 3 };
function locus(sourceId: string, start: number, deltas: number[]): ContextInvestigationLocus {
  const deltaNats = deltas.reduce((sum, value) => sum + value, 0);
  return {
    locusId: `${sourceId}:${start}`, sourceId, answerTokenRange: [start, start + deltas.length],
    answerUnicodeRange: [start, start + deltas.length], answerText: "x".repeat(deltas.length),
    direction: deltaNats < 0 ? "suppression" : "support",
    classification: deltaNats < 0 ? "measured_suppression" : "measured_support",
    deltaNats, absoluteDeltaNats: Math.abs(deltaNats), baselineSelectedLogp: -1, intervenedSelectedLogp: -1 - deltaNats,
    experimentId: "experiment", armId: sourceId, baselineObservationId: "baseline",
    interventionObservationId: sourceId, perTokenDeltaNats: deltas,
  };
}

describe("answer heat evidence", () => {
  it("preserves a real peak and avoids inventing one for a flat or tiny signal", () => {
    expect(answerDisplayScale([.043, .083, .305, .112, .114, .233, .111, .154, .119, .007]))
      .toMatchObject({ ceiling: .305, median: .113, standout: true, peakCount: 1 });
    expect(answerDisplayScale([.2, .21, .2]).standout).toBe(false);
    expect(answerDisplayScale([0, .002, .009])).toMatchObject({ ceiling: .1, standout: false });
    expect(answerDisplayScale([0, 0, 0, .3, .3])).toMatchObject({ median: 0, standout: true, peakCount: 2 });
  });
  it("does not call effects on different answer tokens tension", () => {
    const heat = answerHeatForRange([locus("a", 0, [4]), locus("b", 1, [-4])], tokens, range);
    expect(heat.support).toBe(4);
    expect(heat.suppression).toBe(4);
    expect(heat.tension).toBe(0);
    expect(answerHeatForLens(heat, "tension").intensity).toBe(0);
  });

  it("prioritizes balanced opposing effects over a strong one-sided effect", () => {
    const balanced = answerHeatForRange([locus("a", 0, [12]), locus("b", 0, [-10])], tokens, range);
    const skewed = answerHeatForRange([locus("a", 0, [30]), locus("b", 0, [-1])], tokens, range);
    expect(balanced.tension).toBeGreaterThan(skewed.tension);
    expect(balanced.direction).toBe("mixed");
    expect(skewed.direction).toBe("support");
    expect(answerHeatForLens(skewed, "influence").direction).toBe("mixed");
  });

  it("preserves suppression dominance even with a positive contribution", () => {
    const heat = answerHeatForRange([locus("a", 0, [.2]), locus("b", 0, [-4])], tokens, range);
    expect(heat.direction).toBe("suppression");
    expect(answerHeatForLens(heat, "suppression").intensity).toBeCloseTo(4 / 3);
  });

  it("normalizes by all tokens in the passage, including unaffected tokens", () => {
    const heat = answerHeatForRange([locus("a", 0, [3])], tokens, range);
    expect(heat.intensity).toBe(1);
    expect(heat.tokenCount).toBe(3);
    expect(answerHeatForRange([], tokens, range).intensity).toBe(0);
  });

  it("keeps both signs from one source without netting them away", () => {
    const loci = [locus("a", 0, [2, -2])];
    const heat = answerHeatForRange(loci, tokens, range);
    expect(heat.total).toBe(4);
    expect(heat.support).toBe(2);
    expect(heat.suppression).toBe(2);
    expect([...signedContributions(loci, tokens, range).get("a")!.values()]).toEqual([2, -2]);
  });

  it("is invariant to splitting or duplicating the same source/token evidence", () => {
    const whole = locus("a", 0, [2, 3]);
    const split = [locus("a", 0, [2]), locus("a", 1, [3])];
    expect(answerHeatForRange([whole], tokens, range)).toEqual(answerHeatForRange(split, tokens, range));
    expect(answerHeatForRange([whole, ...split], tokens, range)).toEqual(answerHeatForRange(split, tokens, range));
    expect(answerHeatForRange([whole], tokens, { start: 1, end: 2 }).support).toBe(3);
  });
});
