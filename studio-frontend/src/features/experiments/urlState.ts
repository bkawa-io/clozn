/**
 * Matrix filters and the open-cell selection, serialized to the query string of the
 * `#/experiments/<id>?...` hash so a filtered view or an open drill-down cell is a shareable/reloadable
 * link, per the roadmap plan's "filters serialized to URL" requirement.
 *
 * `src/panels/experiments.tsx`'s `match()` only pulls the raw query string out of the hash (mirroring
 * `scope.tsx`'s `?token=N` pattern) -- parsing it into typed filter/selection state happens here, once,
 * rather than duplicating regex groups per field in the panel's `match()`.
 */

export type StatusFilter = "" | "pass" | "fail" | "error" | "unscored" | "regressed" | "gained";

export interface MatrixFilters {
  suite: "" | "target" | "guard";
  status: StatusFilter;
  /** "" means every variant column is shown. A specific name narrows to baseline + that one variant. */
  variant: string;
  /** null means aggregate across every replicate seed; a number narrows to exactly that seed. */
  seed: number | null;
  q: string;
}

export const DEFAULT_FILTERS: MatrixFilters = { suite: "", status: "", variant: "", seed: null, q: "" };

export interface CellSelection {
  suite: string;
  case: string;
  variant: string;
  seed: number;
}

function isSuite(value: string): value is MatrixFilters["suite"] {
  return value === "" || value === "target" || value === "guard";
}

function isStatusFilter(value: string): value is StatusFilter {
  return ["", "pass", "fail", "error", "unscored", "regressed", "gained"].includes(value);
}

export function parseUrlState(rawQuery: string | undefined): {
  filters: MatrixFilters;
  selection: CellSelection | null;
} {
  const params = new URLSearchParams(rawQuery ?? "");
  const suite = params.get("suite") ?? "";
  const status = params.get("status") ?? "";
  const seedRaw = params.get("seed");
  const seed = seedRaw != null && seedRaw !== "" && Number.isFinite(Number(seedRaw)) ? Number(seedRaw) : null;

  const filters: MatrixFilters = {
    suite: isSuite(suite) ? suite : "",
    status: isStatusFilter(status) ? status : "",
    variant: params.get("variant") ?? "",
    seed,
    q: params.get("q") ?? "",
  };

  let selection: CellSelection | null = null;
  const cellRaw = params.get("cell");
  if (cellRaw) {
    const parts = cellRaw.split("::").map((part) => decodeURIComponent(part));
    const [suiteName, caseName, variantName, seedStr] = parts;
    const cellSeed = Number(seedStr);
    if (suiteName && caseName && variantName && parts.length === 4 && Number.isFinite(cellSeed)) {
      selection = { suite: suiteName, case: caseName, variant: variantName, seed: cellSeed };
    }
  }
  return { filters, selection };
}

export function serializeUrlState(filters: MatrixFilters, selection: CellSelection | null): string {
  const params = new URLSearchParams();
  if (filters.suite) params.set("suite", filters.suite);
  if (filters.status) params.set("status", filters.status);
  if (filters.variant) params.set("variant", filters.variant);
  if (filters.seed != null) params.set("seed", String(filters.seed));
  if (filters.q) params.set("q", filters.q);
  if (selection) {
    params.set(
      "cell",
      [selection.suite, selection.case, selection.variant, String(selection.seed)]
        .map((part) => encodeURIComponent(part))
        .join("::"),
    );
  }
  return params.toString();
}
