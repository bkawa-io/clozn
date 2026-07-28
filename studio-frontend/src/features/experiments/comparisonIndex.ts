import type { VariantComparison } from "./types";

/**
 * A fast lookup from (suite, case, seed, variant) to "this exact cell was a regression/gain relative to
 * baseline" -- built once from `summary.comparisons`, which the server already computed and verified
 * (see api.ts's header comment). The matrix's "regressed"/"gained" quick filters read this instead of
 * re-deriving pass/fail deltas from cells themselves.
 */
export function comparisonKey(suite: string, caseName: string, seed: number, variant: string): string {
  return `${suite}|${caseName}|${seed}|${variant}`;
}

export interface ComparisonIndex {
  regressed: Set<string>;
  gained: Set<string>;
}

export function buildComparisonIndex(comparisons: VariantComparison[]): ComparisonIndex {
  const regressed = new Set<string>();
  const gained = new Set<string>();
  for (const comparison of comparisons) {
    const variant = comparison.variant;
    for (const label of comparison.targetRegressions) regressed.add(comparisonKey("target", label.case, label.seed, variant));
    for (const label of comparison.guardRegressions) regressed.add(comparisonKey("guard", label.case, label.seed, variant));
    for (const label of comparison.targetGains) gained.add(comparisonKey("target", label.case, label.seed, variant));
    for (const label of comparison.guardFixes) gained.add(comparisonKey("guard", label.case, label.seed, variant));
  }
  return { regressed, gained };
}
