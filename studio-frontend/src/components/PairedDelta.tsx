import { EvidenceMark } from "./EvidenceMark";
import "./PairedDelta.css";

/**
 * C4 -- a ranked, paired readout for a comparison that has already been interpreted by its caller.
 * This intentionally owns presentation-shaped values rather than importing a run-diff response type:
 * adapters at the boundary decide how a persisted artifact becomes a label/value pair, while this
 * component keeps the visual invariants usable for any A/B comparison.
 *
 * Run-diff artifacts omit no-ops, but a caller can have independent evidence that a dimension matched.
 * `unchanged` is therefore a local synthetic row kind, not an assertion that the network artifact ever
 * emitted a no-op. It remains a measured, quiet paired state and must never be reused for an absence.
 */

export type PairedDeltaValue = string | number | boolean | null;

export type PairedDeltaComparableKind = "changed" | "added" | "removed" | "unchanged";

export type PairedDeltaAbsenceKind = "unavailable" | "diff_failed";

export type PairedDeltaRowKind = PairedDeltaComparableKind | PairedDeltaAbsenceKind;

interface PairedDeltaRowBase {
  /** Stable key for one dimension within this comparison. */
  id: string;
  /** Human-readable dimension name; callers can keep machine paths in `id` if useful. */
  dimension: string;
  /** Lower ranks render first. This is ordering, not an evidence-strength score. */
  rank: number;
  /** Optional context that applies to a comparable row without replacing either endpoint value. */
  note?: string;
}

export type PairedDeltaChangedRow = PairedDeltaRowBase & {
  kind: "changed" | "unchanged";
  valueA: PairedDeltaValue;
  valueB: PairedDeltaValue;
};

export type PairedDeltaAddedRow = PairedDeltaRowBase & {
  kind: "added";
  valueA?: never;
  valueB: PairedDeltaValue;
};

export type PairedDeltaRemovedRow = PairedDeltaRowBase & {
  kind: "removed";
  valueA: PairedDeltaValue;
  valueB?: never;
};

/**
 * Absence cannot carry endpoint values. Keeping `reason` required while making value fields impossible
 * prevents a caller from accidentally presenting a failed comparison as a flat or zero-valued delta.
 */
export type PairedDeltaAbsenceRow = PairedDeltaRowBase & {
  kind: PairedDeltaAbsenceKind;
  reason: string;
  valueA?: never;
  valueB?: never;
};

export type PairedDeltaComparableRow =
  | PairedDeltaChangedRow
  | PairedDeltaAddedRow
  | PairedDeltaRemovedRow;

export type PairedDeltaRow = PairedDeltaComparableRow | PairedDeltaAbsenceRow;

export type PairedDeltaAxisStatus = "changed" | "unchanged" | "unavailable";

export interface PairedDeltaSummaryAxis {
  id: string;
  label: string;
  status: PairedDeltaAxisStatus;
  note?: string;
}

export type PairedDeltaFindingStatus =
  | "observed"
  | "eliminated"
  | "reproduced"
  | "correlated"
  | "causally_supported";

export interface PairedDeltaFinding {
  id: string;
  label: string;
  summary: string;
  status: PairedDeltaFindingStatus;
}

export interface PairedDeltaProps {
  rows: readonly PairedDeltaRow[];
  summaryAxes?: readonly PairedDeltaSummaryAxis[];
  findings?: readonly PairedDeltaFinding[];
  /** Descriptive run names; the A/B codes remain visible in the mandatory legend. */
  aLabel?: string;
  bLabel?: string;
  title?: string;
  className?: string;
}

const FINDING_STATUS_ORDER: readonly PairedDeltaFindingStatus[] = [
  "observed",
  "eliminated",
  "reproduced",
  "correlated",
  "causally_supported",
];

function readableLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function rowKindLabel(kind: PairedDeltaRowKind): string {
  switch (kind) {
    case "changed": return "Changed";
    case "added": return "Added";
    case "removed": return "Removed";
    case "unchanged": return "Unchanged";
    case "unavailable": return "Unavailable";
    case "diff_failed": return "Diff failed";
    default: {
      const exhaustive: never = kind;
      return exhaustive;
    }
  }
}

function formatValue(value: PairedDeltaValue | undefined): string {
  if (value === undefined) return "Not present";
  if (value === null) return "null";
  return String(value);
}

function isAbsenceRow(row: PairedDeltaRow): row is PairedDeltaAbsenceRow {
  return row.kind === "unavailable" || row.kind === "diff_failed";
}

function valuesOf(row: PairedDeltaComparableRow): { valueA: PairedDeltaValue | undefined; valueB: PairedDeltaValue | undefined } {
  switch (row.kind) {
    case "changed":
    case "unchanged":
      return { valueA: row.valueA, valueB: row.valueB };
    case "added":
      return { valueA: undefined, valueB: row.valueB };
    case "removed":
      return { valueA: row.valueA, valueB: undefined };
    default: {
      const exhaustive: never = row;
      return exhaustive;
    }
  }
}

function sortedByRank(rows: readonly PairedDeltaRow[]): PairedDeltaRow[] {
  // A stable tie-break preserves the caller's intentional order for equal presentation ranks.
  return rows
    .map((row, index) => ({ row, index }))
    .sort((first, second) => first.row.rank - second.row.rank || first.index - second.index)
    .map(({ row }) => row);
}

function PairValues({
  aLabel,
  bLabel,
  valueA,
  valueB,
}: {
  aLabel: string;
  bLabel: string;
  valueA: PairedDeltaValue | undefined;
  valueB: PairedDeltaValue | undefined;
}) {
  return (
    <span className="paired-delta-values" aria-hidden="true">
      <span data-delta-value="a"><b>A · {aLabel}</b>{formatValue(valueA)}</span>
      <span data-delta-value="b"><b>B · {bLabel}</b>{formatValue(valueB)}</span>
    </span>
  );
}

function DumbbellMark({
  row,
  aLabel,
  bLabel,
}: {
  row: PairedDeltaComparableRow;
  aLabel: string;
  bLabel: string;
}) {
  const { valueA, valueB } = valuesOf(row);
  const description = `${row.dimension}. ${rowKindLabel(row.kind)}. A (${aLabel}): ${formatValue(valueA)}. B (${bLabel}): ${formatValue(valueB)}.`;

  if (row.kind === "unchanged") {
    return (
      <div
        className="paired-delta-dumbbell is-unchanged"
        data-delta-visual="coincident"
        role="img"
        aria-label={description}
        title={description}
      >
        <span className="paired-delta-unchanged-stack" aria-hidden="true">
          <span className="paired-delta-dot is-b" />
          <span className="paired-delta-dot is-a" />
        </span>
        <PairValues aLabel={aLabel} bLabel={bLabel} valueA={valueA} valueB={valueB} />
      </div>
    );
  }

  return (
    <div
      className={`paired-delta-dumbbell is-${row.kind}`}
      data-delta-visual="dumbbell"
      role="img"
      aria-label={description}
      title={description}
    >
      <span className="paired-delta-connector" aria-hidden="true" />
      <span className={`paired-delta-dot is-a${valueA === undefined ? " is-missing" : ""}`} aria-hidden="true" />
      <span className={`paired-delta-dot is-b${valueB === undefined ? " is-missing" : ""}`} aria-hidden="true" />
      <PairValues aLabel={aLabel} bLabel={bLabel} valueA={valueA} valueB={valueB} />
    </div>
  );
}

function DeltaRow({ row, aLabel, bLabel }: { row: PairedDeltaRow; aLabel: string; bLabel: string }) {
  const absence = isAbsenceRow(row);
  const reason = absence && row.reason.trim()
    ? row.reason
    : "No comparison reason was recorded.";

  return (
    <li
      className={`paired-delta-row is-${row.kind}`}
      data-delta-row={row.id}
      data-delta-kind={row.kind}
      data-rank={row.rank}
      tabIndex={0}
    >
      <div className="paired-delta-row-head">
        <span className="paired-delta-rank">Rank {row.rank}</span>
        <strong>{row.dimension}</strong>
        {!absence && <span className="paired-delta-kind">{rowKindLabel(row.kind)}</span>}
      </div>
      <div className="paired-delta-row-body">
        {absence ? (
          <EvidenceMark
            variant="chip"
            state="unavailable"
            label={rowKindLabel(row.kind)}
            reason={reason}
          />
        ) : (
          <DumbbellMark row={row} aLabel={aLabel} bLabel={bLabel} />
        )}
      </div>
      {row.note && <p className="paired-delta-note">{row.note}</p>}
    </li>
  );
}

function SummaryAxes({ axes }: { axes: readonly PairedDeltaSummaryAxis[] }) {
  if (axes.length === 0) return null;

  return (
    <ul className="paired-delta-summary-axes" data-testid="paired-delta-summary-axes" aria-label="Comparison summary axes">
      {axes.map((axis) => {
        const status = readableLabel(axis.status);
        const description = axis.note ? `${axis.label}: ${status}. ${axis.note}` : `${axis.label}: ${status}.`;
        return (
          <li
            className={`paired-delta-axis-chip is-${axis.status}`}
            data-summary-axis={axis.id}
            key={axis.id}
            title={description}
          >
            <span>{axis.label}</span>
            <b>{status}</b>
          </li>
        );
      })}
    </ul>
  );
}

function FindingLadder({ finding }: { finding: PairedDeltaFinding }) {
  const currentPosition = FINDING_STATUS_ORDER.indexOf(finding.status) + 1;

  return (
    <article className="paired-delta-finding" data-finding-status={finding.status} data-finding-id={finding.id}>
      <header>
        <span className="paired-delta-finding-step">Step {currentPosition} of {FINDING_STATUS_ORDER.length}</span>
        <strong>{finding.label}</strong>
      </header>
      <p>{finding.summary}</p>
      {/* The ladder's fixed order, labels, and current-step marker carry evidence state. Five hues would
          make this an arbitrary palette rather than an ordinal reading aid. */}
      <ol className="paired-delta-status-ladder" aria-label={`Evidence ladder for ${finding.label}`}>
        {FINDING_STATUS_ORDER.map((status, index) => {
          const current = status === finding.status;
          return (
            <li
              className={current ? "is-current" : undefined}
              data-ladder-status={status}
              aria-current={current ? "step" : undefined}
              key={status}
            >
              <span>{index + 1}</span>
              {readableLabel(status)}
            </li>
          );
        })}
      </ol>
    </article>
  );
}

export function PairedDelta({
  rows,
  summaryAxes = [],
  findings = [],
  aLabel = "Run A",
  bLabel = "Run B",
  title = "Paired delta",
  className,
}: PairedDeltaProps) {
  const sortedRows = sortedByRank(rows);

  return (
    <section className={["paired-delta", className].filter(Boolean).join(" ")} data-testid="paired-delta" aria-label={title}>
      <header className="paired-delta-head">
        <h2>{title}</h2>
        <div className="paired-delta-legend" role="group" aria-label="A and B legend">
          <span className="paired-delta-legend-title">A / B</span>
          <span className="paired-delta-legend-entry">
            <span className="paired-delta-dot is-a" aria-hidden="true" />
            <b>A</b>
            <span>{aLabel}</span>
          </span>
          <span className="paired-delta-legend-entry">
            <span className="paired-delta-dot is-b" aria-hidden="true" />
            <b>B</b>
            <span>{bLabel}</span>
          </span>
        </div>
      </header>

      <SummaryAxes axes={summaryAxes} />

      <ol className="paired-delta-rows" data-testid="paired-delta-rows" aria-label="Ranked comparison rows">
        {sortedRows.map((row) => <DeltaRow key={row.id} row={row} aLabel={aLabel} bLabel={bLabel} />)}
      </ol>

      {findings.length > 0 && (
        <section className="paired-delta-findings" aria-label="Comparison findings">
          <h3>Findings</h3>
          {findings.map((finding) => <FindingLadder finding={finding} key={finding.id} />)}
        </section>
      )}
    </section>
  );
}
