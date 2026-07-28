import { useMemo } from "react";
import { buildComparisonIndex } from "./comparisonIndex";
import { buildMatrixRows, type MatrixCellAggregate, type MatrixRow } from "./buildMatrix";
import { statusLabel } from "./format";
import type { CellSelection, MatrixFilters, StatusFilter } from "./urlState";
import type { ExperimentDetail } from "./types";
import { VirtualList } from "./VirtualList";

const ROW_HEIGHT = 40;

interface MatrixProps {
  detail: ExperimentDetail;
  filters: MatrixFilters;
  onFiltersChange: (next: MatrixFilters) => void;
  selection: CellSelection | null;
  onSelectCell: (cell: CellSelection) => void;
}

function rowMatchesStatus(row: MatrixRow, status: StatusFilter, variantsToCheck: string[]): boolean {
  if (!status) return true;
  if (status === "regressed") return variantsToCheck.some((v) => row.byVariant.get(v)?.regressed);
  if (status === "gained") return variantsToCheck.some((v) => row.byVariant.get(v)?.gained);
  return variantsToCheck.some((v) => (row.byVariant.get(v)?.counts[status] ?? 0) > 0);
}

function CellBadge({
  aggregate,
  isSelected,
  onSelect,
}: {
  aggregate: MatrixCellAggregate | undefined;
  isSelected: boolean;
  onSelect: (seed: number) => void;
}) {
  if (!aggregate) {
    return <span className="experiments-cell is-missing" title="No cell recorded at this coordinate">—</span>;
  }
  const label = aggregate.perSeed.length > 1
    ? `${aggregate.counts[aggregate.dominant] ?? 0}/${aggregate.perSeed.length} ${statusLabel(aggregate.dominant)}`
    : statusLabel(aggregate.dominant);
  return (
    <button
      type="button"
      className={`experiments-cell is-${aggregate.dominant} ${isSelected ? "is-selected" : ""} ${aggregate.regressed ? "is-regressed" : ""} ${aggregate.gained ? "is-gained" : ""}`}
      aria-pressed={isSelected}
      title={`${aggregate.case} · ${aggregate.variant} · ${aggregate.perSeed.map((p) => `seed ${p.seed}: ${p.status}`).join(", ")}`}
      onClick={() => onSelect(aggregate.perSeed[0]?.seed ?? 0)}
    >
      <span className="experiments-cell-label">{label}</span>
      {aggregate.perSeed.length > 1 && (
        <span className="experiments-cell-dots" aria-hidden="true">
          {aggregate.perSeed.map((p) => <i className={`is-${p.status}`} key={p.seed} />)}
        </span>
      )}
      {aggregate.regressed && <span className="experiments-cell-flag" aria-label="regressed vs baseline">▼</span>}
      {aggregate.gained && <span className="experiments-cell-flag" aria-label="gained vs baseline">▲</span>}
    </button>
  );
}

export function Matrix({ detail, filters, onFiltersChange, selection, onSelectCell }: MatrixProps) {
  const baseline = detail.summary.baselineVariant ?? detail.manifest.baselineVariant;
  const allVariants = useMemo(
    () => detail.manifest.variants.map((v) => v.name).filter(Boolean),
    [detail.manifest.variants],
  );
  const columns = filters.variant && allVariants.includes(filters.variant)
    ? [...new Set([baseline, filters.variant])]
    : allVariants;

  const index = useMemo(() => buildComparisonIndex(detail.summary.comparisons), [detail.summary.comparisons]);

  const rowsShape = useMemo(() => {
    const shape: { suite: string; case: string }[] = [];
    for (const suite of ["target", "guard"] as const) {
      for (const c of detail.manifest.suites[suite]?.cases ?? []) shape.push({ suite, case: c.name });
    }
    return shape;
  }, [detail.manifest.suites]);

  const allRows = useMemo(
    () => buildMatrixRows({ cells: detail.cells, rowsShape, variants: allVariants, seedFilter: filters.seed, index }),
    [detail.cells, rowsShape, allVariants, filters.seed, index],
  );

  const variantsToCheck = filters.variant ? [filters.variant] : allVariants.filter((v) => v !== baseline);
  const query = filters.q.trim().toLowerCase();
  const filteredRows = allRows.filter((row) => {
    if (filters.suite && row.suite !== filters.suite) return false;
    if (query && !row.case.toLowerCase().includes(query)) return false;
    return rowMatchesStatus(row, filters.status, variantsToCheck);
  });

  const targetRows = filteredRows.filter((r) => r.suite === "target");
  const guardRows = filteredRows.filter((r) => r.suite === "guard");
  const showTarget = filters.suite !== "guard";
  const showGuard = filters.suite !== "target";

  function set<K extends keyof MatrixFilters>(key: K, value: MatrixFilters[K]) {
    onFiltersChange({ ...filters, [key]: value });
  }

  function renderRow(row: MatrixRow) {
    return (
      <div
        className="experiments-row"
        style={{ gridTemplateColumns: `minmax(160px, 260px) repeat(${columns.length}, minmax(96px, 1fr))` }}
      >
        <span className="experiments-row-label" title={row.case}>{row.case}</span>
        {columns.map((variant) => {
          const aggregate = row.byVariant.get(variant);
          const isSelected = !!selection && selection.suite === row.suite && selection.case === row.case
            && selection.variant === variant;
          return (
            <CellBadge
              aggregate={aggregate}
              isSelected={isSelected}
              onSelect={(seed) => onSelectCell({ suite: row.suite, case: row.case, variant, seed })}
              key={variant}
            />
          );
        })}
      </div>
    );
  }

  return (
    <section className="instrument experiments-matrix-instrument" aria-labelledby="experiments-matrix-title">
      <header className="instrument-head experiments-matrix-head">
        <div>
          <span className="eyebrow">CASE × VARIANT</span>
          <h2 id="experiments-matrix-title">Matrix</h2>
        </div>
        <div className="experiments-filters">
          <label>
            <span>SUITE</span>
            <select value={filters.suite} onChange={(e) => set("suite", e.target.value as MatrixFilters["suite"])}>
              <option value="">All</option>
              <option value="target">Target</option>
              <option value="guard">Guard</option>
            </select>
          </label>
          <label>
            <span>STATUS</span>
            <select value={filters.status} onChange={(e) => set("status", e.target.value as StatusFilter)}>
              <option value="">All</option>
              <option value="pass">Pass</option>
              <option value="fail">Fail</option>
              <option value="error">Error</option>
              <option value="unscored">Unscored</option>
              <option value="regressed">Regressed vs baseline</option>
              <option value="gained">Gained vs baseline</option>
            </select>
          </label>
          <label>
            <span>VARIANT</span>
            <select value={filters.variant} onChange={(e) => set("variant", e.target.value)}>
              <option value="">All variants</option>
              {allVariants.map((v) => <option value={v} key={v}>{v}</option>)}
            </select>
          </label>
          <label>
            <span>SEED</span>
            <select
              value={filters.seed == null ? "" : String(filters.seed)}
              onChange={(e) => set("seed", e.target.value === "" ? null : Number(e.target.value))}
            >
              <option value="">Aggregate</option>
              {detail.seeds.map((seed) => <option value={String(seed)} key={seed}>{seed}</option>)}
            </select>
          </label>
          <label className="experiments-search">
            <span>CASE</span>
            <input
              type="search"
              value={filters.q}
              onChange={(e) => set("q", e.target.value)}
              placeholder="Filter by case name"
            />
          </label>
        </div>
      </header>

      <div
        className="experiments-matrix-columns"
        aria-hidden="true"
        style={{ gridTemplateColumns: `minmax(160px, 260px) repeat(${columns.length}, minmax(96px, 1fr))` }}
      >
        <span />
        {columns.map((variant) => (
          <span key={variant} className={variant === baseline ? "is-baseline" : ""}>
            {variant}{variant === baseline ? " (baseline)" : ""}
          </span>
        ))}
      </div>

      <div className="experiments-matrix-body">
        {showTarget && (
          <>
            <div className="experiments-suite-label">TARGET · {targetRows.length} cases</div>
            <VirtualList
              items={targetRows}
              rowHeight={ROW_HEIGHT}
              renderRow={renderRow}
              keyFor={(row) => `target-${row.case}`}
              ariaLabel="Target suite cases"
              emptyLabel="NO MATCHING TARGET CASES"
              className="experiments-suite-list"
            />
          </>
        )}
        {showGuard && (
          <>
            <div className="experiments-suite-label">GUARD · {guardRows.length} cases</div>
            <VirtualList
              items={guardRows}
              rowHeight={ROW_HEIGHT}
              renderRow={renderRow}
              keyFor={(row) => `guard-${row.case}`}
              ariaLabel="Guard suite cases"
              emptyLabel="NO MATCHING GUARD CASES"
              className="experiments-suite-list"
            />
          </>
        )}
      </div>
    </section>
  );
}
