import { comparisonKey, type ComparisonIndex } from "./comparisonIndex";
import { dominantStatus } from "./format";
import type { CellStatus, ThinCell } from "./types";

export interface MatrixCellAggregate {
  suite: string;
  case: string;
  variant: string;
  counts: Partial<Record<CellStatus, number>>;
  dominant: CellStatus;
  /** Replicate seeds actually present at this coordinate, under the current seed filter. */
  seeds: number[];
  /** One entry per replicate seed, in seed order -- drives the per-replicate status dots in the badge. */
  perSeed: { seed: number; status: CellStatus }[];
  regressed: boolean;
  gained: boolean;
}

export interface MatrixRow {
  suite: string;
  case: string;
  byVariant: Map<string, MatrixCellAggregate>;
  /** OR across every non-baseline variant column -- the row-level reading of the "regressed"/"gained"
   * quick filter when no single variant column is selected. */
  anyRegressed: boolean;
  anyGained: boolean;
}

/**
 * Groups thin cells into (suite, case) rows x variant columns, aggregating replicate seeds into one
 * badge per cell. `rowsShape` comes from the manifest (every declared suite/case pair), not from the
 * cells actually present -- `validate_result` guarantees the case x variant x seed matrix is complete
 * for anything the server successfully served, but building rows from the declared shape rather than
 * "whatever cells happened to arrive" keeps this function honest about a genuinely partial fetch too
 * (e.g. a cells response scoped to one suite) instead of silently dropping rows.
 */
export function buildMatrixRows(params: {
  cells: ThinCell[];
  rowsShape: { suite: string; case: string }[];
  variants: string[];
  seedFilter: number | null;
  index: ComparisonIndex;
}): MatrixRow[] {
  const { cells, rowsShape, variants, seedFilter, index } = params;

  const bucket = new Map<string, ThinCell[]>();
  for (const cell of cells) {
    if (seedFilter != null && cell.seed !== seedFilter) continue;
    const key = `${cell.suite}|${cell.case}|${cell.variant}`;
    const existing = bucket.get(key);
    if (existing) existing.push(cell);
    else bucket.set(key, [cell]);
  }

  return rowsShape.map(({ suite, case: caseName }) => {
    const byVariant = new Map<string, MatrixCellAggregate>();
    let anyRegressed = false;
    let anyGained = false;
    for (const variant of variants) {
      const matched = bucket.get(`${suite}|${caseName}|${variant}`);
      if (!matched || !matched.length) continue;
      const counts: Partial<Record<CellStatus, number>> = {};
      const seeds: number[] = [];
      const perSeed: { seed: number; status: CellStatus }[] = [];
      let regressed = false;
      let gained = false;
      for (const cell of matched) {
        const status = (cell.status as CellStatus) ?? "unscored";
        counts[status] = (counts[status] ?? 0) + 1;
        seeds.push(cell.seed);
        perSeed.push({ seed: cell.seed, status });
        if (index.regressed.has(comparisonKey(suite, caseName, cell.seed, variant))) regressed = true;
        if (index.gained.has(comparisonKey(suite, caseName, cell.seed, variant))) gained = true;
      }
      perSeed.sort((a, b) => a.seed - b.seed);
      if (regressed) anyRegressed = true;
      if (gained) anyGained = true;
      byVariant.set(variant, {
        suite,
        case: caseName,
        variant,
        counts,
        dominant: dominantStatus(counts as Record<string, number>),
        seeds: seeds.sort((a, b) => a - b),
        perSeed,
        regressed,
        gained,
      });
    }
    return { suite, case: caseName, byVariant, anyRegressed, anyGained };
  });
}
