import { EvidenceMark } from "./EvidenceMark";
import "./CompositionBar.css";

/**
 * C5 -- the composition bar. One stacked bar answering a single proportion question: how much of
 * what was assembled actually reached the model. Two known consumers: the context receipt (delivered
 * vs. omitted request segments against the rendered token count) and cumulative snapshot-pin storage
 * (`kv_bytes` / `envelope_bytes` against a byte budget). Neither vocabulary is hardcoded here --
 * `present` / `reduced` / `absent` is deliberately generic. For a context receipt, one reasonable
 * mapping is: a delivered segment with `redaction_state: "full"` is `present`; one delivered but
 * redacted or hashed-only is `reduced` (it reached the model, just not faithfully); a segment with
 * `included: false` is `absent`, carrying its `reason`. That mapping lives in the caller, not here.
 *
 * The rule this component exists to enforce (see the roadmap spec for C5): omitted content is a
 * labelled segment carrying its reason, never a shorter bar. A composition bar that just renders
 * "delivered" a little narrower than it could have been says "there was less content" -- indistinguishable
 * from the assembler simply having less to work with. A hatched, reasoned segment says "content was
 * dropped, and here is why", which the research behind this product treats as a distinct, higher-
 * confidence signal than either a confident full bar or a silently shorter one. Zero-filling (or its
 * visual cousin, shrinking the bar without a label) is the lowest-confidence treatment measured; this
 * component structurally cannot produce it, because "absent" is not a smaller number, it is its own
 * segment with mandatory text.
 */

export type CompositionSegmentKind = "present" | "reduced" | "absent";

interface CompositionSegmentBase {
  /** Stable key, also used as the React list key. Keep it unique within one bar's `segments`. */
  id: string;
  label: string;
  /**
   * Exact recorded count in whatever unit the caller passes via `unit` (tokens, bytes, ...). Never
   * pre-rounded -- this is the one place display formatting happens, and rule 4 is that segment
   * values are exact recorded counts, not approximations.
   */
  value: number;
}

/**
 * Discriminated on `kind`, mirroring EvidenceMark's own reason-only-on-absence pattern (see
 * EvidenceMark.tsx): `reason` is required on `"absent"` and structurally impossible to pass on
 * `"present"` / `"reduced"` -- TypeScript rejects it outright on those two members. That is what
 * actually enforces "omitted content always carries its reason" instead of leaving it to every call
 * site to remember to pass one.
 */
export type CompositionSegment =
  | (CompositionSegmentBase & { kind: "present"; reason?: never })
  | (CompositionSegmentBase & { kind: "reduced"; reason?: never })
  | (CompositionSegmentBase & { kind: "absent"; reason: string });

export interface CompositionBarProps {
  /** What was assembled, broken into labelled shares. Left-to-right render order matches array order. */
  segments: CompositionSegment[];
  /**
   * The stated total this composition is measured against -- e.g. a token budget, or the rendered
   * prompt's own token count. Defaults to the sum of `segments` when omitted, in which case the bar
   * always accounts for exactly 100% because there is nothing independent to compare against.
   * Passing an explicit `total` is what lets the unaccounted-remainder rule (below) actually fire.
   */
  total?: number;
  /** Unit shown after each value, e.g. "tokens" or "bytes". Omit for a unitless count. */
  unit?: string;
  /** Heading rendered above the bar. Optional -- callers embedding this inside a card that already
   * has its own heading can omit it and supply their own via `aria-labelledby` upstream. */
  title?: string;
  className?: string;
}

type ResolvedKind = CompositionSegmentKind | "unaccounted";

interface ResolvedRow {
  id: string;
  label: string;
  value: number;
  kind: ResolvedKind;
  reason?: string;
  percent: number;
}

// Below this share of the bar, a segment's own rectangle is too narrow to hold readable text without
// clipping mid-word. Below the threshold the label moves to the key instead; the key is unconditionally
// rendered whenever there is more than one row, so nothing is ever lost -- only relocated.
const INLINE_LABEL_MIN_PERCENT = 12;

function percentOf(value: number, denominator: number): number {
  return denominator > 0 ? (value / denominator) * 100 : 0;
}

function formatCount(value: number, unit: string | undefined): string {
  // toLocaleString adds grouping separators only -- it does not round. Rule 4: exact recorded counts,
  // shown, not rounded away.
  return `${value.toLocaleString()}${unit ? ` ${unit}` : ""}`;
}

/**
 * Turns caller segments into renderable rows, synthesizing an explicit "unaccounted" row whenever the
 * segments do not sum to the stated total. This is the load-bearing function for rule 3: the
 * denominator used for every percentage is `max(statedTotal, sum)`, never just one or the other, so a
 * shortfall widens the bar with a labelled gap instead of quietly stretching the real segments to fill
 * 100%, and an overshoot (segments summing to more than the stated total -- a data-quality signal in
 * its own right) still renders every segment at its true share instead of being clamped invisible.
 */
function resolveRows(
  segments: CompositionSegment[],
  total: number | undefined,
  unit: string | undefined,
): { rows: ResolvedRow[]; denominator: number } {
  const sum = segments.reduce((running, segment) => running + segment.value, 0);
  const statedTotal = total ?? sum;
  const remainder = statedTotal - sum;
  const denominator = Math.max(statedTotal, sum, 1);

  const rows: ResolvedRow[] = segments.map((segment) => ({
    id: segment.id,
    label: segment.label,
    value: segment.value,
    kind: segment.kind,
    reason: segment.kind === "absent" ? segment.reason : undefined,
    percent: percentOf(segment.value, denominator),
  }));

  if (remainder > 0) {
    rows.push({
      id: "__unaccounted",
      label: "Unaccounted",
      value: remainder,
      kind: "unaccounted",
      reason: `The stated total is ${formatCount(remainder, unit)} more than the recorded segments add up to -- not covered by any recorded segment.`,
      percent: percentOf(remainder, denominator),
    });
  } else if (remainder < 0) {
    const overshoot = -remainder;
    rows.push({
      id: "__unaccounted",
      label: "Exceeds stated total",
      value: overshoot,
      kind: "unaccounted",
      reason: `Recorded segments sum to ${formatCount(overshoot, unit)} more than the stated total.`,
      percent: percentOf(overshoot, denominator),
    });
  }

  return { rows, denominator };
}

function CompositionKeyEntry({ row, unit }: { row: ResolvedRow; unit: string | undefined }) {
  // "absent" and the synthetic "unaccounted" row are both, at bottom, the same kind of fact: no
  // content is here, and here is why. EvidenceMark is the one shared primitive for exactly that
  // question elsewhere in the component system (see EvidenceMark.tsx) -- reusing its "unavailable"
  // chip rather than drawing a second swatch-and-caption treatment is what "do not invent a second
  // treatment for absence" means in practice.
  if (row.kind === "absent" || row.kind === "unaccounted") {
    return (
      <li className={`composition-bar-key-row is-${row.kind}`}>
        <EvidenceMark variant="chip" state="unavailable" reason={row.reason ?? "Reason not recorded."} label={row.label} />
        <span className="composition-bar-key-value">{formatCount(row.value, unit)} · {Math.round(row.percent)}%</span>
      </li>
    );
  }
  return (
    <li className={`composition-bar-key-row is-${row.kind}`}>
      <span className="composition-bar-key-entry">
        <span className={`composition-bar-swatch is-${row.kind}`} aria-hidden="true" />
        <strong>{row.label}</strong>
      </span>
      <span className="composition-bar-key-value">{formatCount(row.value, unit)} · {Math.round(row.percent)}%</span>
    </li>
  );
}

export function CompositionBar({ segments, total, unit, title, className }: CompositionBarProps) {
  const { rows } = resolveRows(segments, total, unit);
  const isEmpty = rows.length === 0;
  // The key is mandatory whenever there is more than one row to disambiguate -- a single full-width
  // segment is already self-explanatory, but two or more never rely on inline labels alone (mark spec).
  const showKey = rows.length >= 2;
  const summary = rows
    .map((row) => `${row.label} ${formatCount(row.value, unit)} (${Math.round(row.percent)}%)${row.reason ? ` -- ${row.reason}` : ""}`)
    .join("; ");

  return (
    <div className={["composition-bar", className].filter(Boolean).join(" ")}>
      {title && <div className="composition-bar-title">{title}</div>}
      {isEmpty ? (
        <div className="composition-bar-empty">NO SEGMENTS RECORDED</div>
      ) : (
        <div className="composition-bar-track" role="img" aria-label={summary}>
          {rows.map((row) => (
            <div
              key={row.id}
              className={`composition-bar-segment is-${row.kind}`}
              // flex-grow carries the raw value, not a pre-computed percent -- flexbox distributes the
              // track's width (minus the fixed 2px inter-segment gaps) proportionally on its own, so
              // there is no separate percent-to-pixel rounding step that could drift from `percent`
              // below. flex-basis: 0 means the grow ratio is the only thing that decides width.
              style={{ flexGrow: Math.max(row.value, 0), flexBasis: 0 }}
              title={`${row.label} · ${formatCount(row.value, unit)} · ${Math.round(row.percent)}%${row.reason ? ` · ${row.reason}` : ""}`}
            >
              {row.percent >= INLINE_LABEL_MIN_PERCENT && (
                <span className="composition-bar-segment-label">
                  <strong>{row.label}</strong>
                  <span>{formatCount(row.value, unit)}</span>
                </span>
              )}
            </div>
          ))}
        </div>
      )}
      {showKey && (
        <ul className="composition-bar-key" aria-label="Composition segments">
          {rows.map((row) => <CompositionKeyEntry row={row} unit={unit} key={row.id} />)}
        </ul>
      )}
    </div>
  );
}
