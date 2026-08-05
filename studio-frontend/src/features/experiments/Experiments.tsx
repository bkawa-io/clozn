import { useEffect, useState } from "react";
import { useTopbar } from "../../panels/topbar";
import { listExperiments, loadExperimentDetail } from "./api";
import { CellDrawer } from "./CellDrawer";
import { formatPassRate, formatTimestamp, shortId } from "./format";
import { CiPreviewPanel, HistoryPanel } from "./HistoryAndCi";
import { Matrix } from "./Matrix";
import { DEFAULT_FILTERS, parseUrlState, serializeUrlState } from "./urlState";
import type { CellSelection, MatrixFilters } from "./urlState";
import type { ExperimentDetail, ExperimentList } from "./types";
import "../../styles/experiments.css";

export type ExperimentsPresentation = "standalone" | "compare";

export interface ExperimentsProps {
  id?: string;
  /** The raw query string this route's hash carried, e.g. `"suite=target&status=fail"` -- parsed by
   * `urlState.ts`, not by the panel's own `match()` (see that file's header comment). */
  rawQuery?: string;
  /** `#/experiments` remains valid, while Compare's canonical matrix route keeps filters on its own URL. */
  routeBase?: string;
  /** Compare composes this existing instrument rather than maintaining a parallel experiment surface. */
  presentation?: ExperimentsPresentation;
}

function pathFor(routeBase: string, id?: string) {
  const base = routeBase.replace(/\/$/, "");
  return id ? `${base}/${encodeURIComponent(id)}` : base;
}

function pushUrl(routeBase: string, id: string | undefined, query: string) {
  const path = pathFor(routeBase, id);
  history.replaceState(null, "", query ? `${path}?${query}` : path);
}

function ExperimentsList({ routeBase, presentation }: { routeBase: string; presentation: ExperimentsPresentation }) {
  const [list, setList] = useState<ExperimentList | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    void listExperiments({}, controller.signal)
      .then((next) => {
        if (controller.signal.aborted) return;
        setList(next);
        setStatus("ready");
      })
      .catch(() => {
        if (!controller.signal.aborted) setStatus("error");
      });
    return () => controller.abort();
  }, []);

  useTopbar(
    () => ({
      stats: <span className="top-stat"><b>EXPERIMENTS</b>{list?.total ?? 0}</span>,
      modeChip: presentation === "compare" ? "MATRIX" : undefined,
    }),
    [list?.total, presentation],
  );

  return (
    <section className="instrument experiments-list-instrument" aria-labelledby="experiments-list-title">
      <header className="instrument-head">
        <div>
          <span className="eyebrow">{presentation === "compare" ? "RECORDED A / B OUTCOMES" : "EXPERIMENT RESULTS"}</span>
          <h1 id="experiments-list-title">{presentation === "compare" ? "Experiment matrices" : "Experiments"}</h1>
        </div>
      </header>

      {status === "loading" && <div className="experiments-state">LOADING EXPERIMENTS</div>}
      {status === "error" && <div className="experiments-state is-error">EXPERIMENT LIST UNAVAILABLE</div>}
      {status === "ready" && list && (
        <>
          {list.broken.length > 0 && (
            <div className="experiments-broken" role="alert">
              <strong>{list.broken.length} RESULT FILE{list.broken.length === 1 ? "" : "S"} FAILED TO LOAD</strong>
              <ul>{list.broken.map((b) => <li key={b.path}>{b.path}: {b.error}</li>)}</ul>
            </div>
          )}
          <div className="experiments-table" role="list" aria-label="Experiment results">
            <div className="experiments-table-head" aria-hidden="true">
              <span>NAME</span><span>CREATED</span><span>BASELINE</span><span>VARIANTS</span><span>CELLS</span>
            </div>
            {list.experiments.map((entry) => (
              <a
                className="experiments-table-row"
                role="listitem"
                href={pathFor(routeBase, entry.experimentId)}
                key={entry.experimentId}
              >
                <span className="experiments-name">
                  <strong>{entry.name}</strong>
                  <small>{shortId(entry.experimentId)}</small>
                  {entry.suiteFingerprint && (
                    <small>{entry.suiteFingerprint.algorithm}:{shortId(entry.suiteFingerprint.sha256)}</small>
                  )}
                </span>
                <span>{formatTimestamp(entry.createdAt)}</span>
                <span>{entry.baselineVariant ?? "—"}</span>
                <span>{entry.variants.join(", ") || "—"}</span>
                <span>{entry.cellCount}</span>
              </a>
            ))}
            {list.experiments.length === 0 && <div className="experiments-empty">NO EXPERIMENT RESULTS RECORDED</div>}
          </div>
        </>
      )}
    </section>
  );
}

function SummaryStrip({ detail }: { detail: ExperimentDetail }) {
  const baseline = detail.summary.baselineVariant ?? detail.manifest.baselineVariant;
  const errorCount = detail.cells.filter((c) => c.status === "error").length;
  return (
    <section className="instrument experiments-summary-instrument" aria-labelledby="experiments-summary-title">
      <header className="instrument-head compact">
        <div>
          <span className="eyebrow">{detail.experimentId}</span>
          <h2 id="experiments-summary-title">{detail.name}</h2>
        </div>
        <div className="experiments-summary-meta">
          <span><b>CREATED</b>{formatTimestamp(detail.createdAt)}</span>
          <span><b>BASELINE</b>{baseline || "—"}</span>
          <span><b>SEEDS</b>{detail.seeds.length}</span>
          <span><b>CELLS</b>{detail.cells.length}</span>
          {detail.suiteFingerprint && (
            <span><b>FINGERPRINT</b>{shortId(detail.suiteFingerprint.sha256)}</span>
          )}
          {errorCount > 0 && <span className="is-error"><b>ERRORS</b>{errorCount}</span>}
        </div>
      </header>
      <div className="experiments-summary-variants">
        {detail.manifest.variants.map((variant) => {
          const aggregate = detail.summary.aggregates[variant.name];
          const comparison = detail.summary.comparisons.find((c) => c.variant === variant.name);
          const isBaseline = variant.name === baseline;
          return (
            <article className={`experiments-variant-card ${isBaseline ? "is-baseline" : ""}`} key={variant.name}>
              <header>
                <strong>{variant.name}</strong>
                <span>{variant.kind}{isBaseline ? " · baseline" : ""}</span>
              </header>
              <div className="experiments-variant-rates">
                <span><b>TARGET</b>{formatPassRate(aggregate?.target?.passRate ?? null)}</span>
                <span><b>GUARD</b>{formatPassRate(aggregate?.guard?.passRate ?? null)}</span>
              </div>
              {comparison && !isBaseline && (
                <div className="experiments-variant-comparison">
                  <span className="is-gain">+{comparison.targetGains.length} gains</span>
                  <span className="is-regression">-{comparison.targetRegressions.length} regressions</span>
                  <span className="is-guard-regression">{comparison.guardRegressions.length} guard regressions</span>
                  <span className="is-guard-fix">{comparison.guardFixes.length} guard fixes</span>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ExperimentWorkspace({
  id,
  rawQuery,
  routeBase,
  presentation,
}: {
  id: string;
  rawQuery: string | undefined;
  routeBase: string;
  presentation: ExperimentsPresentation;
}) {
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  // Lazy initializers run exactly once per mount -- the key={id} at the call site below remounts this
  // component whenever the experiment id changes, so a fresh mount always re-parses that navigation's
  // own query string rather than carrying stale filter/selection state from a previously viewed result.
  const [filters, setFilters] = useState<MatrixFilters>(() => parseUrlState(rawQuery).filters ?? DEFAULT_FILTERS);
  const [selection, setSelection] = useState<CellSelection | null>(() => parseUrlState(rawQuery).selection);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setDetail(null);
    void loadExperimentDetail(id, controller.signal)
      .then((next) => {
        if (controller.signal.aborted) return;
        setDetail(next);
        setStatus("ready");
      })
      .catch(() => {
        if (!controller.signal.aborted) setStatus("error");
      });
    return () => controller.abort();
  }, [id]);

  useEffect(() => {
    pushUrl(routeBase, id, serializeUrlState(filters, selection));
  }, [id, routeBase, filters, selection]);

  useTopbar(
    () => ({
      stats: <span className="top-stat"><b>EXPERIMENT</b>{detail?.name ?? id}</span>,
      modeChip: status === "loading" ? "LOADING" : status === "error" ? "ERROR" : presentation === "compare" ? "MATRIX" : "READY",
    }),
    [id, detail?.name, presentation, status],
  );

  if (status === "loading") return <div className="experiments-state">LOADING EXPERIMENT</div>;
  if (status === "error" || !detail) {
    return (
      <div className="experiments-state is-error">
        <span>EXPERIMENT UNAVAILABLE</span>
        <a href={routeBase}>BACK TO LIST</a>
      </div>
    );
  }

  return (
    <>
      <HistoryPanel experimentId={id} />
      <SummaryStrip detail={detail} />
      <Matrix
        detail={detail}
        filters={filters}
        onFiltersChange={setFilters}
        selection={selection}
        onSelectCell={setSelection}
      />
      <CiPreviewPanel detail={detail} />
      {selection && (
        <CellDrawer
          experimentId={id}
          detail={detail}
          selection={selection}
          onClose={() => setSelection(null)}
          onSelectSeed={(seed) => setSelection((current) => (current ? { ...current, seed } : current))}
        />
      )}
    </>
  );
}

export function Experiments({
  id,
  rawQuery,
  routeBase = "#/experiments",
  presentation = "standalone",
}: ExperimentsProps) {
  return id
    ? <ExperimentWorkspace id={id} rawQuery={rawQuery} routeBase={routeBase} presentation={presentation} key={`${routeBase}:${id}`} />
    : <ExperimentsList routeBase={routeBase} presentation={presentation} />;
}
