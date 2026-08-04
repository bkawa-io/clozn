import type { CSSProperties } from "react";
import { EvidenceMark } from "./EvidenceMark";
import "./InfluenceMatrix.css";

/**
 * C6 -- a context-span × answer-span intervention matrix. The API deliberately describes rendered
 * evidence rather than a receipt or transport shape: adapters belong at call sites, so this component
 * never inherits a network schema's vocabulary or its lifecycle assumptions.
 */

export interface InfluenceMatrixContextSpan {
  id: string;
  text: string;
}

export interface InfluenceMatrixAnswerSpan {
  id: string;
  text: string;
}

interface InfluenceMatrixCellReference {
  contextSpanId: string;
  answerSpanId: string;
}

export type InfluenceMatrixMeasuredEvidenceState = "causally_supported" | "observed";

/** A measured intervention either cleared the floor or did not; neither case is an absence. */
export type InfluenceMatrixMeasuredLink =
  | (InfluenceMatrixCellReference & {
      deltaNats: number;
      evidenceState: "causally_supported";
      clearsFloor: true;
      reason?: never;
    })
  | (InfluenceMatrixCellReference & {
      deltaNats: number;
      evidenceState: "observed";
      clearsFloor: false;
      reason?: never;
    });

/**
 * The two no-measurement states require a reason at the type boundary. A caller cannot make a cold
 * square look like a zero-valued result by omitting its explanation.
 */
export type InfluenceMatrixUnavailableCell =
  | (InfluenceMatrixCellReference & {
      evidenceState: "not_measured";
      reason: string;
      deltaNats?: never;
    })
  | (InfluenceMatrixCellReference & {
      evidenceState: "omitted";
      reason: string;
      deltaNats?: never;
    });

export interface InfluenceMatrixProps {
  contextSpans: readonly InfluenceMatrixContextSpan[];
  answerSpans: readonly InfluenceMatrixAnswerSpan[];
  measuredLinks: readonly InfluenceMatrixMeasuredLink[];
  unavailableCells?: readonly InfluenceMatrixUnavailableCell[];
  /** The per-cell absolute delta that distinguishes a cleared-floor result from an observed one. */
  floorNats: number;
  /** Symmetric absolute-value percentile used to cap the diverging scale. Defaults to the 95th. */
  clampPercentile?: number;
  title?: string;
  className?: string;
}

export interface InfluenceMatrixClamp {
  ceilingNats: number;
  percentile: number;
  applied: boolean;
}

type InfluenceMatrixCell = InfluenceMatrixMeasuredLink | InfluenceMatrixUnavailableCell;

type CellPolarity = "supports" | "suppresses" | "neutral";

type InfluenceMatrixRenderState = "measured_effect" | "below_floor" | "not_measured" | "omitted";

export const DEFAULT_INFLUENCE_MATRIX_CLAMP_PERCENTILE = 95;

const UNSPECIFIED_CELL_REASON = "No evidence state was supplied for this cell.";

function cellKey(contextSpanId: string, answerSpanId: string): string {
  return `${contextSpanId}\u0000${answerSpanId}`;
}

function boundedPercentile(percentile: number | undefined): number {
  if (percentile == null || !Number.isFinite(percentile)) return DEFAULT_INFLUENCE_MATRIX_CLAMP_PERCENTILE;
  return Math.min(100, Math.max(0, percentile));
}

/**
 * The ceiling is a symmetric absolute-value quantile: one extreme can never set a one-sided colour
 * range or force ordinary positive and negative effects into the same near-neutral lightness.
 */
export function getInfluenceMatrixClamp(
  measuredLinks: readonly InfluenceMatrixMeasuredLink[],
  requestedPercentile: number | undefined = DEFAULT_INFLUENCE_MATRIX_CLAMP_PERCENTILE,
): InfluenceMatrixClamp {
  const percentile = boundedPercentile(requestedPercentile);
  const magnitudes = measuredLinks
    .map((link) => Math.abs(link.deltaNats))
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  const maximum = magnitudes.at(-1) ?? 0;

  if (magnitudes.length === 0) return { ceilingNats: 0, percentile, applied: false };

  const index = (percentile / 100) * (magnitudes.length - 1);
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  const fraction = index - lowerIndex;
  const ceilingNats = magnitudes[lowerIndex] + ((magnitudes[upperIndex] - magnitudes[lowerIndex]) * fraction);

  return {
    ceilingNats,
    percentile,
    applied: ceilingNats < maximum,
  };
}

function formatNats(value: number): string {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(4)}`;
}

function formatMagnitude(value: number): string {
  return Math.abs(value).toFixed(4);
}

function ordinal(percentile: number): string {
  if (!Number.isInteger(percentile)) return `${percentile}TH`;
  const remainder = percentile % 100;
  if (remainder >= 11 && remainder <= 13) return `${percentile}TH`;
  switch (percentile % 10) {
    case 1: return `${percentile}ST`;
    case 2: return `${percentile}ND`;
    case 3: return `${percentile}RD`;
    default: return `${percentile}TH`;
  }
}

function polarityFor(deltaNats: number): CellPolarity {
  if (deltaNats > 0) return "supports";
  if (deltaNats < 0) return "suppresses";
  return "neutral";
}

function isMeasuredCell(cell: InfluenceMatrixCell): cell is InfluenceMatrixMeasuredLink {
  return cell.evidenceState === "causally_supported" || cell.evidenceState === "observed";
}

/** Render states explain the matrix's visual grammar; evidence states remain the recorded contract. */
function renderStateFor(cell: InfluenceMatrixCell): InfluenceMatrixRenderState {
  if (isMeasuredCell(cell)) return cell.clearsFloor ? "measured_effect" : "below_floor";
  switch (cell.evidenceState) {
    case "not_measured": return "not_measured";
    case "omitted": return "omitted";
    default: {
      const exhaustive: never = cell;
      return exhaustive;
    }
  }
}

function cellClassName(cell: InfluenceMatrixCell): string {
  const renderState = renderStateFor(cell);
  const stateClassName = `is-${renderState.replaceAll("_", "-")}`;
  return isMeasuredCell(cell) && cell.clearsFloor
    ? `${stateClassName} is-${polarityFor(cell.deltaNats)}`
    : stateClassName;
}

function cellTooltip(cell: InfluenceMatrixCell): string {
  const delta = isMeasuredCell(cell) ? formatNats(cell.deltaNats) : "not measured";
  const reason = isMeasuredCell(cell) ? "" : `; reason: ${cell.reason}`;
  return [
    `delta_nats: ${delta}`,
    `evidence_state: ${cell.evidenceState}`,
    `context_span_id: ${cell.contextSpanId}`,
    `answer_span_id: ${cell.answerSpanId}${reason}`,
  ].join("; ");
}

function measuredCellStyle(cell: InfluenceMatrixMeasuredLink, clamp: InfluenceMatrixClamp): CSSProperties {
  const magnitude = Math.min(Math.abs(cell.deltaNats), clamp.ceilingNats);
  const ratio = clamp.ceilingNats > 0 ? magnitude / clamp.ceilingNats : 0;
  // Only lightness changes with magnitude. Hue names the sign; it never stands in for strength.
  return { "--influence-matrix-fill": `${18 + (ratio * 72)}%` } as CSSProperties;
}

function fallbackCell(contextSpanId: string, answerSpanId: string): InfluenceMatrixUnavailableCell {
  return {
    contextSpanId,
    answerSpanId,
    evidenceState: "not_measured",
    reason: UNSPECIFIED_CELL_REASON,
  };
}

function CellContent({ cell }: { cell: InfluenceMatrixCell }) {
  if (isMeasuredCell(cell)) {
    if (!cell.clearsFloor) {
      return <span className="influence-matrix-below-floor-glyph" aria-hidden="true">≈</span>;
    }
    const polarity = polarityFor(cell.deltaNats);
    return <span className="influence-matrix-polarity" aria-hidden="true">{
      polarity === "supports" ? "+" : polarity === "suppresses" ? "−" : "0"
    }</span>;
  }

  switch (cell.evidenceState) {
    case "not_measured":
      return <EvidenceMark state="not_measured" label="Not measured" reason={cell.reason} className="influence-matrix-absence-mark" />;
    case "omitted":
      return <EvidenceMark state="unavailable" label="Omitted" reason={cell.reason} className="influence-matrix-absence-mark" />;
    default: {
      const exhaustive: never = cell;
      return exhaustive;
    }
  }
}

function InfluenceMatrixLegend({ floorNats, clamp }: Pick<InfluenceMatrixProps, "floorNats"> & {
  clamp: InfluenceMatrixClamp;
}) {
  return (
    <ul className="influence-matrix-legend" aria-label="Influence matrix legend">
      <li>
        <span className="influence-matrix-legend-swatch is-supports" aria-hidden="true" />
        <span>SUPPORTS (+)</span>
      </li>
      <li>
        <span className="influence-matrix-legend-swatch is-neutral" aria-hidden="true" />
        <span>NEUTRAL (0)</span>
      </li>
      <li>
        <span className="influence-matrix-legend-swatch is-suppresses" aria-hidden="true" />
        <span>SUPPRESSES (−)</span>
      </li>
      <li>
        <span className="influence-matrix-legend-swatch is-below-floor" aria-hidden="true" />
        <span>BELOW FLOOR</span>
      </li>
      <li>
        <EvidenceMark state="not_measured" reason="No intervention was recorded for this cell." />
        <span>NOT MEASURED</span>
      </li>
      <li>
        <EvidenceMark state="unavailable" label="Omitted" reason="The context span did not reach measurement." />
        <span>OMITTED</span>
      </li>
      <li className="influence-matrix-legend-floor">FLOOR · |Δ| ≥ {formatMagnitude(floorNats)} NATS</li>
      <li
        className="influence-matrix-legend-clamp"
        data-clamp-applied={String(clamp.applied)}
      >
        SYMMETRIC CLAMP AT {ordinal(clamp.percentile)} PERCENTILE · ±{formatMagnitude(clamp.ceilingNats)} NATS
      </li>
    </ul>
  );
}

export function InfluenceMatrix({
  contextSpans,
  answerSpans,
  measuredLinks,
  unavailableCells = [],
  floorNats,
  clampPercentile,
  title = "Influence matrix",
  className,
}: InfluenceMatrixProps) {
  const cellIndex = new Map<string, InfluenceMatrixCell>();
  for (const cell of unavailableCells) cellIndex.set(cellKey(cell.contextSpanId, cell.answerSpanId), cell);
  for (const link of measuredLinks) cellIndex.set(cellKey(link.contextSpanId, link.answerSpanId), link);
  const clamp = getInfluenceMatrixClamp(measuredLinks, clampPercentile);

  return (
    <section className={["influence-matrix", className].filter(Boolean).join(" ")} aria-label={title}>
      <header className="influence-matrix-header">
        <div>
          <span className="influence-matrix-eyebrow">CONTEXT × ANSWER</span>
          <h3>{title}</h3>
        </div>
        <span className="influence-matrix-clamp-note" data-clamp-applied={String(clamp.applied)}>
          {clamp.applied ? "OUTLIERS CLAMPED" : "SYMMETRIC SCALE"} · {ordinal(clamp.percentile)} PERCENTILE
        </span>
      </header>

      <InfluenceMatrixLegend floorNats={floorNats} clamp={clamp} />

      {contextSpans.length === 0 || answerSpans.length === 0 ? (
        <div className="influence-matrix-empty">NO SPAN INTERSECTIONS RECORDED</div>
      ) : (
        <div className="influence-matrix-scroll">
          <table className="influence-matrix-table">
            <thead>
              <tr>
                <th scope="col" className="influence-matrix-corner">CONTEXT / ANSWER</th>
                {answerSpans.map((answerSpan) => (
                  <th scope="col" className="influence-matrix-column-header" key={answerSpan.id}>
                    <span title={answerSpan.text}>{answerSpan.text}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {contextSpans.map((contextSpan) => (
                <tr key={contextSpan.id}>
                  <th scope="row" className="influence-matrix-row-header">
                    <span title={contextSpan.text}>{contextSpan.text}</span>
                  </th>
                  {answerSpans.map((answerSpan) => {
                    const cell = cellIndex.get(cellKey(contextSpan.id, answerSpan.id))
                      ?? fallbackCell(contextSpan.id, answerSpan.id);
                    const style = isMeasuredCell(cell) ? measuredCellStyle(cell, clamp) : undefined;
                    return (
                      <td
                        className={`influence-matrix-cell ${cellClassName(cell)}`}
                        data-evidence-state={cell.evidenceState}
                        data-render-state={renderStateFor(cell)}
                        data-polarity={isMeasuredCell(cell) && cell.clearsFloor
                          ? polarityFor(cell.deltaNats)
                          : undefined}
                        key={answerSpan.id}
                        style={style}
                        title={cellTooltip(cell)}
                        aria-label={cellTooltip(cell)}
                      >
                        <CellContent cell={cell} />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
