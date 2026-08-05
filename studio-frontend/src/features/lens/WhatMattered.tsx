import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import {
  InfluenceMatrix,
  type InfluenceMatrixMeasuredLink,
  type InfluenceMatrixUnavailableCell,
} from "../../components/InfluenceMatrix";
import { TypedActionOffer, type TypedActionOfferAbsence } from "../../components/TypedActionOffer";
import {
  cancelRunInfluenceMapJob,
  loadRunInfluenceMapJob,
  startRunInfluenceMapJob,
} from "../../data/api";
import {
  loadRunInvestigation,
  loadSpanAddresses,
  type InfluenceLink,
  type InfluenceSpanRef,
  type InvestigationAction,
  type InvestigationState,
  type PromptSourceInfluenceSection,
  type ReceivedSegment,
  type RunInvestigationReceipt,
  type SpanAddress,
  type SpanAddressDocument,
} from "../../data/received-context";
import type { InfluenceMapJob } from "../../data/types";

/**
 * "What mattered?" -- the cross-linked prompt-span x answer-span heatmap.
 *
 * This is NOT a restatement of ReceivedContext ("what did the model receive?"). That surface answers
 * delivery; this one answers attribution -- which prompt spans measurably moved which answer spans, using
 * `sections.prompt_source_influence` from the SAME `/runs/<id>/investigation` document ReceivedContext
 * already reads, cross-linked to durable span addresses from `/runs/<id>/span-addresses`.
 *
 * THE FOUR CELL STATES (never collapsed into one "cold" value -- see CellState below):
 *   - measured_effect:        a link cleared the measurement floor. A real, controlled intervention
 *                              effect (`evidence_state: causally_supported`).
 *   - below_measurement_floor: the link was measured but did not clear the floor. An honest observed
 *                              non-effect, never "irrelevant" (`evidence_state: observed`).
 *   - omitted:                the prompt content never reached the model at all (context_receipt's own
 *                              `omitted` segments). No measurement is even possible.
 *   - not_measured:           the prompt content DID reach the model (it is in the assembled context)
 *                              but this influence run's bounded span budget never scored it
 *                              (`prompt_sources[].selected === false`), OR no influence measurement has
 *                              been run for this whole answer yet.
 *
 * Rendering this component fires exactly two GETs (investigation, span-addresses). It never starts a
 * measurement on its own -- `startMeasurement` below only runs from an explicit button click.
 */

interface WhatMatteredProps {
  runId: string;
}

type Resource<T> =
  | { status: "idle" | "loading" }
  | { status: "failed" }
  | { status: "ready"; value: T };

type MeasureStatus = "idle" | "measuring" | "error";

export type CellState = "measured_effect" | "below_measurement_floor" | "omitted" | "not_measured";

type GridRow =
  | { kind: "measured"; key: string; label: string; detail?: string; spanId: string }
  | { kind: "not_measured"; key: string; label: string; detail?: string; spanId: string }
  | {
      kind: "omitted";
      key: string;
      label: string;
      detail?: string;
      segmentId?: string;
      clientSourceId?: string;
    };

interface GridColumn {
  key: string;
  label: string;
  spanId: string;
  tokenIndex?: number;
}

interface Grid {
  rows: GridRow[];
  columns: GridColumn[];
  linkIndex: Map<string, InfluenceLink>;
}

interface MatrixInput {
  contextSpans: Array<{ id: string; text: string }>;
  answerSpans: Array<{ id: string; text: string }>;
  measuredLinks: InfluenceMatrixMeasuredLink[];
  unavailableCells: InfluenceMatrixUnavailableCell[];
  floorNats: number;
}

export const CELL_STATE_LABEL: Record<CellState, string> = {
  measured_effect: "CLEARED FLOOR",
  below_measurement_floor: "BELOW FLOOR",
  omitted: "OMITTED",
  not_measured: "NOT MEASURED",
};

function rowLabel(span: InfluenceSpanRef): string {
  return (
    span.sourceLabel
    ?? span.name
    ?? span.role
    ?? span.sourceKind
    ?? span.kind
    ?? `SPAN ${span.id.slice(0, 10)}`
  );
}

export function buildRows(
  section: PromptSourceInfluenceSection,
  omitted: ReceivedSegment[],
): GridRow[] {
  const measured: GridRow[] = section.promptSpans.map((span) => ({
    kind: "measured",
    key: `measured:${span.id}`,
    label: rowLabel(span),
    detail: span.level ? span.level.toUpperCase() : undefined,
    spanId: span.id,
  }));
  const notMeasured: GridRow[] = section.promptSources
    .filter((source) => source.selected === false)
    .map((source) => ({
      kind: "not_measured",
      key: `not-measured:${source.id}`,
      label: rowLabel(source),
      detail: "reached the model, excluded by this run's bounded span selection",
      spanId: source.id,
    }));
  const omittedRows: GridRow[] = omitted.map((segment, index) => ({
    kind: "omitted",
    key: `omitted:${segment.segmentId ?? segment.clientSourceId ?? index}`,
    label: segment.sourceLabel ?? segment.sourceType ?? "UNLABELED INPUT",
    detail: segment.reason ?? "never reached the model",
    segmentId: segment.segmentId,
    clientSourceId: segment.clientSourceId,
  }));
  return [...measured, ...notMeasured, ...omittedRows];
}

export function buildColumns(section: PromptSourceInfluenceSection): GridColumn[] {
  return section.answerSpans.map((span, index) => ({
    key: `answer:${span.id}`,
    label: span.tokenIndex != null ? `T${span.tokenIndex + 1}` : `#${index + 1}`,
    spanId: span.id,
    tokenIndex: span.tokenIndex,
  }));
}

function linkKey(contextSpanId: string, answerSpanId: string): string {
  return `${contextSpanId}|${answerSpanId}`;
}

export function buildLinkIndex(links: InfluenceLink[]): Map<string, InfluenceLink> {
  const index = new Map<string, InfluenceLink>();
  for (const link of links) index.set(linkKey(link.contextSpanId, link.answerSpanId), link);
  return index;
}

/** The one function this whole feature exists to get right: never collapse the four states into a single
 * heat value. `not_measured` is the fallback for a `measured` row with no link for this column -- the
 * matrix `context_answer_influence` persists is complete for what it measured, so this should not happen
 * in practice, but a missing value stays missing rather than being guessed at. */
export function cellFor(
  row: GridRow,
  col: GridColumn,
  linkIndex: Map<string, InfluenceLink>,
): { state: CellState; link?: InfluenceLink } {
  switch (row.kind) {
    case "omitted":
      return { state: "omitted" };
    case "not_measured":
      return { state: "not_measured" };
    case "measured": {
      const link = linkIndex.get(linkKey(row.spanId, col.spanId));
      if (!link) return { state: "not_measured" };
      return {
        state: link.evidenceState === "causally_supported" ? "measured_effect" : "below_measurement_floor",
        link,
      };
    }
    default: {
      const exhaustive: never = row;
      return exhaustive;
    }
  }
}

export function cellGlyph(state: CellState, link?: InfluenceLink): string {
  switch (state) {
    case "measured_effect":
      return link && link.effect !== "neutral" ? (link.effect === "supports" ? "F+" : "F-") : "F";
    case "below_measurement_floor":
      return link && link.effect !== "neutral" ? (link.effect === "supports" ? "B+" : "B-") : "B0";
    case "omitted":
      return "OM";
    case "not_measured":
      return "NM";
    default: {
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}

/**
 * C6 is the overview; the older table below remains the addressable ledger. The adapter preserves all
 * four existing grid states and refuses to coerce a malformed recorded link into a different evidence
 * state just to make a colourful square.
 */
function matrixInput(section: PromptSourceInfluenceSection, grid: Grid): MatrixInput {
  const measuredLinks: InfluenceMatrixMeasuredLink[] = [];
  const unavailableByCell = new Map<string, InfluenceMatrixUnavailableCell>();
  const key = (contextSpanId: string, answerSpanId: string) => `${contextSpanId}\u0000${answerSpanId}`;

  for (const link of section.links) {
    if (link.clearsFloor && link.evidenceState === "causally_supported") {
      measuredLinks.push({
        contextSpanId: link.contextSpanId,
        answerSpanId: link.answerSpanId,
        deltaNats: link.deltaNats,
        evidenceState: "causally_supported",
        clearsFloor: true,
      });
    } else if (!link.clearsFloor && link.evidenceState === "observed") {
      measuredLinks.push({
        contextSpanId: link.contextSpanId,
        answerSpanId: link.answerSpanId,
        deltaNats: link.deltaNats,
        evidenceState: "observed",
        clearsFloor: false,
      });
    } else {
      unavailableByCell.set(key(link.contextSpanId, link.answerSpanId), {
        contextSpanId: link.contextSpanId,
        answerSpanId: link.answerSpanId,
        evidenceState: "not_measured",
        reason: "The recorded link has an inconsistent floor/evidence-state pair and cannot be rendered as a measured effect.",
      });
    }
  }

  const measuredKeys = new Set(measuredLinks.map((link) => key(link.contextSpanId, link.answerSpanId)));
  for (const row of grid.rows) {
    for (const column of grid.columns) {
      const contextSpanId = matrixRowId(row);
      const cellKey = key(contextSpanId, column.spanId);
      if (measuredKeys.has(cellKey) || unavailableByCell.has(cellKey)) continue;
      const { state } = cellFor(row, column, grid.linkIndex);
      if (state === "omitted") {
        unavailableByCell.set(cellKey, {
          contextSpanId,
          answerSpanId: column.spanId,
          evidenceState: "omitted",
          reason: row.detail ?? "This context passage never reached the model.",
        });
      } else if (state === "not_measured") {
        unavailableByCell.set(cellKey, {
          contextSpanId,
          answerSpanId: column.spanId,
          evidenceState: "not_measured",
          reason: row.detail ?? section.reason ?? "This context/answer pair was not scored.",
        });
      }
    }
  }

  return {
    contextSpans: grid.rows.map((row) => ({ id: matrixRowId(row), text: row.detail ? `${row.label} — ${row.detail}` : row.label })),
    answerSpans: grid.columns.map((column) => ({ id: column.spanId, text: column.label })),
    measuredLinks,
    unavailableCells: [...unavailableByCell.values()],
    floorNats: section.thresholds.cellAbsDeltaNats ?? 0.05,
  };
}

/** Omitted receipt segments have no influence span id because they never reached the model. C6 still
 * needs a stable row identity to render their explicit impossible-to-measure cells. */
function matrixRowId(row: GridRow): string {
  switch (row.kind) {
    case "measured":
    case "not_measured":
      return row.spanId;
    case "omitted":
      return `omitted:${row.segmentId ?? row.clientSourceId ?? row.key}`;
    default: {
      const exhaustive: never = row;
      return exhaustive;
    }
  }
}

function actionAbsence(section: PromptSourceInfluenceSection): TypedActionOfferAbsence {
  const reason = section.reason ?? "No prompt-to-answer influence measurement is retained for this run.";
  return section.state === "delivered_not_measured" || section.state === "omitted"
    ? { state: "not_measured", label: "Influence not measured", reason }
    : { state: "unavailable", label: "Influence unavailable", reason };
}

function cellClassName(state: CellState): string {
  switch (state) {
    case "measured_effect": return "is-cell-measured";
    case "below_measurement_floor": return "is-cell-below-floor";
    case "omitted": return "is-cell-omitted";
    case "not_measured": return "is-cell-not-measured";
    default: {
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}

function stateBannerLabel(state: InvestigationState): string {
  switch (state) {
    case "measured_effect": return "MEASURED EFFECT";
    case "below_measurement_floor": return "BELOW MEASUREMENT FLOOR";
    case "delivered_not_measured": return "NOT MEASURED";
    case "unavailable": return "UNAVAILABLE";
    case "failed": return "MEASUREMENT FAILED";
    case "inconclusive": return "INCONCLUSIVE";
    case "omitted": return "OMITTED";
    case "supported": return "SUPPORTED";
    default: {
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}

/** `undefined` when `value` is absent -- callers look this up through `addressLookup`, which treats an
 * `undefined` key as "no match" rather than risking a collision on some placeholder string. */
function addressKey(collection: string, part: "id" | "seg" | "src", value: string | undefined): string | undefined {
  return value ? `${collection}::${part}:${value}` : undefined;
}

function addressLookup(
  index: Map<string, SpanAddress>,
  key: string | undefined,
): SpanAddress | undefined {
  return key == null ? undefined : index.get(key);
}

export function buildAddressIndex(addresses: SpanAddress[]): Map<string, SpanAddress> {
  const index = new Map<string, SpanAddress>();
  for (const address of addresses) {
    const { collection, id, segmentId, clientSourceId } = address.nativeRef;
    if (!collection) continue;
    for (const key of [
      addressKey(collection, "id", id),
      addressKey(collection, "seg", segmentId),
      addressKey(collection, "src", clientSourceId),
    ]) {
      if (key && !index.has(key)) index.set(key, address);
    }
  }
  return index;
}

function rowAddress(row: GridRow, index: Map<string, SpanAddress>): SpanAddress | undefined {
  switch (row.kind) {
    case "measured":
      return addressLookup(index, addressKey("influence.prompt_spans", "id", row.spanId));
    case "not_measured":
      return addressLookup(index, addressKey("influence.prompt_sources", "id", row.spanId));
    case "omitted":
      return (
        addressLookup(index, addressKey("context_receipt.delivered", "seg", row.segmentId))
        ?? addressLookup(index, addressKey("context_receipt.delivered", "src", row.clientSourceId))
      );
    default: {
      const exhaustive: never = row;
      return exhaustive;
    }
  }
}

function columnAddress(col: GridColumn, index: Map<string, SpanAddress>): SpanAddress | undefined {
  return addressLookup(index, addressKey("influence.answer_spans", "id", col.spanId));
}

function spanHref(runId: string, address: SpanAddress): string {
  return `/runs/${encodeURIComponent(runId)}/span-addresses#${address.addressId}`;
}

function statusLabel(value: string) {
  return value.replaceAll("_", " ").toUpperCase();
}

function waitMs(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      window.clearTimeout(timeout);
      reject(new DOMException("Aborted", "AbortError"));
    }
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

const MEASURE_ACTION_ID = "measure_prompt_source_influence";

function Legend() {
  return (
    <ul className="what-mattered-legend" aria-label="Cell state legend">
      <li className="is-cell-measured">
        <b>{cellGlyph("measured_effect")}</b>
        <span>{CELL_STATE_LABEL.measured_effect} -- controlled, measured effect</span>
      </li>
      <li className="is-cell-below-floor">
        <b>{cellGlyph("below_measurement_floor")}</b>
        <span>{CELL_STATE_LABEL.below_measurement_floor} -- measured, no effect cleared</span>
      </li>
      <li className="is-cell-omitted">
        <b>{cellGlyph("omitted")}</b>
        <span>{CELL_STATE_LABEL.omitted} -- never reached the model</span>
      </li>
      <li className="is-cell-not-measured">
        <b>{cellGlyph("not_measured")}</b>
        <span>{CELL_STATE_LABEL.not_measured} -- reached the model, not scored</span>
      </li>
    </ul>
  );
}

export function WhatMattered({ runId }: WhatMatteredProps) {
  const [investigation, setInvestigation] = useState<Resource<RunInvestigationReceipt>>({
    status: "idle",
  });
  const [spans, setSpans] = useState<Resource<SpanAddressDocument>>({ status: "idle" });
  const [refreshToken, setRefreshToken] = useState(0);
  const [measureStatus, setMeasureStatus] = useState<MeasureStatus>("idle");
  const [measureJob, setMeasureJob] = useState<InfluenceMapJob | null>(null);
  const [measureError, setMeasureError] = useState<string | null>(null);
  const measureController = useRef<AbortController | null>(null);
  const measureJobId = useRef<string | null>(null);

  // THE fetch this view is allowed to make on its own: two GETs, keyed to run identity + refresh
  // generation. A stale response for a run/generation this component has moved on from is dropped, never
  // rendered -- the same guarantee ReceivedContext's own investigation fetch makes (see its own effect).
  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setInvestigation({ status: "loading" });
    setSpans({ status: "loading" });
    void loadRunInvestigation(runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setInvestigation({ status: "ready", value });
    }).catch(() => {
      if (!controller.signal.aborted) setInvestigation({ status: "failed" });
    });
    void loadSpanAddresses(runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setSpans({ status: "ready", value });
    }).catch(() => {
      if (!controller.signal.aborted) setSpans({ status: "failed" });
    });
    return () => controller.abort();
  }, [runId, refreshToken]);

  // A measurement started against a previous run must never be reported against this one, or leave a
  // "measuring" banner stuck on a run that never asked for it.
  useEffect(() => {
    measureController.current?.abort();
    measureController.current = null;
    measureJobId.current = null;
    setMeasureStatus("idle");
    setMeasureJob(null);
    setMeasureError(null);
  }, [runId]);

  async function startMeasurement() {
    if (measureStatus === "measuring" || !runId) return;
    const controller = new AbortController();
    measureController.current = controller;
    measureJobId.current = null;
    setMeasureStatus("measuring");
    setMeasureError(null);
    setMeasureJob(null);
    let job: InfluenceMapJob;
    try {
      job = await startRunInfluenceMapJob(runId, controller.signal);
      measureJobId.current = job.jobId;
      setMeasureJob(job);
      while (job.state !== "completed" && job.state !== "failed" && job.state !== "cancelled") {
        await waitMs(300, controller.signal);
        job = await loadRunInfluenceMapJob(runId, job.jobId, controller.signal);
        setMeasureJob(job);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        setMeasureStatus("idle");
        return;
      }
      setMeasureError(error instanceof Error ? error.message : "measurement request failed");
      setMeasureStatus("error");
      return;
    } finally {
      if (measureController.current === controller) measureController.current = null;
    }
    if (job.state === "cancelled") {
      setMeasureStatus("idle");
      return;
    }
    if (job.state === "failed") {
      setMeasureError(job.error?.message ?? "the measurement job failed");
      setMeasureStatus("error");
      return;
    }
    // Never patch the job artifact into local state -- re-fetch the canonical investigation + span
    // documents through the same identity-keyed effect above, so the grid always reflects exactly what
    // is persisted, never a locally reconstructed guess.
    setMeasureStatus("idle");
    setMeasureJob(null);
    setRefreshToken((generation) => generation + 1);
  }

  async function stopMeasurement() {
    if (measureStatus !== "measuring") return;
    const controller = measureController.current;
    const jobId = measureJobId.current;
    if (runId && jobId) {
      try {
        await cancelRunInfluenceMapJob(runId, jobId);
      } catch {
        // Best effort -- this view simply stops waiting locally either way.
      }
    }
    controller?.abort();
    measureController.current = null;
    setMeasureStatus("idle");
  }

  const influenceSection = investigation.status === "ready"
    ? investigation.value.promptSourceInfluence
    : undefined;
  const omittedSegments = investigation.status === "ready"
    ? investigation.value.receivedContext.omitted
    : [];
  const measureAction: InvestigationAction | undefined = investigation.status === "ready"
    ? investigation.value.actions.find((action) => action.id === MEASURE_ACTION_ID)
    : undefined;
  const isMeasured = influenceSection?.state === "measured_effect"
    || influenceSection?.state === "below_measurement_floor";

  const grid: Grid | null = useMemo(() => {
    if (!influenceSection || !isMeasured) return null;
    const columns = buildColumns(influenceSection);
    if (!columns.length) return null;
    return {
      rows: buildRows(influenceSection, omittedSegments),
      columns,
      linkIndex: buildLinkIndex(influenceSection.links),
    };
  }, [influenceSection, isMeasured, omittedSegments]);

  const matrix = useMemo(
    () => grid && influenceSection ? matrixInput(influenceSection, grid) : null,
    [grid, influenceSection],
  );

  const addressIndex = useMemo(
    () => spans.status === "ready" ? buildAddressIndex(spans.value.addresses) : new Map<string, SpanAddress>(),
    [spans],
  );

  const maxAbsDeltaNats = useMemo(() => {
    let max = 0;
    for (const link of influenceSection?.links ?? []) if (link.absDeltaNats > max) max = link.absDeltaNats;
    return max;
  }, [influenceSection]);

  return (
    <section className="what-mattered" aria-labelledby="what-mattered-title">
      <header className="what-mattered-head">
        <div>
          <span className="eyebrow">CROSS-LINKED</span>
          <h3 id="what-mattered-title">What mattered?</h3>
        </div>
        {influenceSection && (
          <span className={`what-mattered-overall is-${influenceSection.state}`}>
            {statusLabel(influenceSection.state)}
          </span>
        )}
      </header>

      <p className="what-mattered-boundary">
        Cross-links measured prompt spans to answer spans. A cleared-floor cell is a controlled, measured
        effect -- not proof the model read or relied on that exact text. A cold cell can mean three
        different things: never reached the model, reached it but was never scored, or was scored and
        cleared nothing -- each is labelled below, never merged into one color.
      </p>

      {investigation.status === "idle" || investigation.status === "loading" ? (
        <div className="what-mattered-empty">LOADING CROSS-LINK EVIDENCE</div>
      ) : investigation.status === "failed" || !influenceSection ? (
        <div className="what-mattered-notice is-failed" role="alert">
          <EvidenceMark
            variant="chip"
            state="unavailable"
            label="INVESTIGATION REQUEST FAILED"
            reason="The cross-linked influence view could not be loaded."
          />
        </div>
      ) : (
        <>
          {spans.status === "failed" && (
            <div className="what-mattered-notice is-failed" role="alert">
              <EvidenceMark
                variant="chip"
                state="unavailable"
                label="STABLE SPAN REQUEST FAILED"
                reason="Cross-linked evidence remains visible, but stable span links are unavailable."
              />
            </div>
          )}

          <Legend />

          {grid ? (
            <>
              {matrix && (
                <InfluenceMatrix
                  title="Measured influence overview"
                  contextSpans={matrix.contextSpans}
                  answerSpans={matrix.answerSpans}
                  measuredLinks={matrix.measuredLinks}
                  unavailableCells={matrix.unavailableCells}
                  floorNats={matrix.floorNats}
                />
              )}
              <p className="what-mattered-thresholds">
                {influenceSection.thresholds.cellAbsDeltaNats != null
                  ? `FLOOR ${influenceSection.thresholds.cellAbsDeltaNats.toFixed(4)} NATS`
                  : "FLOOR UNKNOWN"}
                {" · "}
                {grid.rows.length} ROWS {"×"} {grid.columns.length} ANSWER SPANS
              </p>
              <div className="what-mattered-grid-wrap">
                <table className="what-mattered-grid" aria-labelledby="what-mattered-title">
                  <thead>
                    <tr>
                      <th scope="col" className="what-mattered-corner">SOURCE / ANSWER</th>
                      {grid.columns.map((col) => {
                        const address = columnAddress(col, addressIndex);
                        return (
                          <th scope="col" key={col.key} className="what-mattered-col-head">
                            {col.tokenIndex != null ? (
                              <a href={`#/runs/${encodeURIComponent(runId)}?section=what-mattered`}>
                                {col.label}
                              </a>
                            ) : address ? (
                              <a href={spanHref(runId, address)}>{col.label}</a>
                            ) : (
                              <span>{col.label}</span>
                            )}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {grid.rows.map((row) => {
                      const address = rowAddress(row, addressIndex);
                      return (
                        <tr key={row.key}>
                          <th scope="row" className={`what-mattered-row-head is-${row.kind}`}>
                            {address ? (
                              <a href={spanHref(runId, address)}>{row.label}</a>
                            ) : (
                              <span>{row.label}</span>
                            )}
                            {row.detail && <small>{row.detail}</small>}
                          </th>
                          {grid.columns.map((col) => {
                            const { state, link } = cellFor(row, col, grid.linkIndex);
                            const ratio = link && maxAbsDeltaNats > 0
                              ? Math.min(1, link.absDeltaNats / maxAbsDeltaNats)
                              : 0;
                            const style = link
                              ? ({ "--cell-intensity": String(ratio) } as CSSProperties)
                              : undefined;
                            const detail = link
                              ? `${CELL_STATE_LABEL[state]} · ${link.deltaNats >= 0 ? "+" : ""}${
                                  link.deltaNats.toFixed(4)
                                } nats`
                              : CELL_STATE_LABEL[state];
                            return (
                              <td
                                key={col.key}
                                className={`what-mattered-cell ${cellClassName(state)}`}
                                data-cell-state={state}
                                style={style}
                                title={detail}
                              >
                                {cellGlyph(state, link)}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className={`what-mattered-banner is-${influenceSection.state}`}>
              <strong>{stateBannerLabel(influenceSection.state)}</strong>
              {measureStatus === "measuring" ? (
                <>
                  <p className="what-mattered-measuring">
                    {measureJob
                      ? `MEASURING · ${measureJob.progress.phase.toUpperCase()} · ${
                          measureJob.progress.completedUnits
                        }/${measureJob.progress.totalUnits}`
                      : "STARTING MEASUREMENT"}
                  </p>
                  <button type="button" onClick={() => void stopMeasurement()}>STOP WAITING</button>
                </>
              ) : (
                <TypedActionOffer
                  title="Measure what mattered"
                  absence={actionAbsence(influenceSection)}
                  cost="Runs a bounded prompt × answer intervention job for this recorded run."
                  preconditions={[
                    "The recorded run and its retained context must be available.",
                    "A measurement-capable worker must be available for this run.",
                  ]}
                  action={measureAction?.availability === "ready"
                    ? {
                      availability: "available",
                      label: "MEASURE WHAT MATTERED",
                      onAction: () => void startMeasurement(),
                    }
                    : {
                      availability: "blocked",
                      label: "MEASURE WHAT MATTERED",
                      blockerReason: measureAction?.reason ?? "No measurement action was reported for this run.",
                    }}
                />
              )}
              {measureStatus === "error" && measureError && (
                <p className="what-mattered-measure-error" role="alert">{measureError}</p>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
