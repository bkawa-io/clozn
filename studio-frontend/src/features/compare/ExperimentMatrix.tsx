import { useEffect, useMemo, useRef, useState } from "react";
import { routeHref } from "../../app/router";
import { describeError, isContractError, studioApi } from "../../data/client";
import type {
  ExperimentCellsDocument,
  ExperimentDetail,
  ExperimentListDocument,
  ExperimentThinCell,
} from "../../data/experimentResults";
import "./experimentMatrix.css";

type ReadState<T> =
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable"; error: string }
  | { phase: "malformed"; error: string };

type CellState =
  | { phase: "idle" }
  | { phase: "loading"; coordinate: string }
  | { phase: "ready"; coordinate: string; value: ExperimentCellsDocument }
  | { phase: "unavailable" | "malformed"; coordinate: string; error: string };

function readFailure(error: unknown): { phase: "unavailable" | "malformed"; error: string } {
  return {
    phase: isContractError(error) ? "malformed" : "unavailable",
    error: describeError(error),
  };
}

function percentage(value: number | null): string {
  return value === null ? "Unscored" : `${Math.round(value * 100)}%`;
}

function MatrixIndex({ document }: { document: ExperimentListDocument }) {
  return (
    <>
      {document.broken.length > 0 && (
        <section className="matrix-broken" aria-labelledby="matrix-broken-title">
          <h2 id="matrix-broken-title">Broken result artifacts</h2>
          <p>{document.broken.length} local result {document.broken.length === 1 ? "file is" : "files are"} visible but unreadable.</p>
          <ul>{document.broken.map((item) => <li key={item.path}><code>{item.path}</code><span>{item.error}</span></li>)}</ul>
        </section>
      )}
      {document.experiments.length === 0
        ? <p className="matrix-empty">No experiment results are recorded.</p>
        : (
          <ol className="matrix-result-list">
            {document.experiments.map((experiment) => (
              <li key={experiment.experimentId}>
                <div className="matrix-result-title">
                  <div><strong>{experiment.name}</strong><code>{experiment.experimentId}</code></div>
                  <a href={routeHref({ surface: "compare", mode: "matrix", experimentId: experiment.experimentId })}>Open matrix</a>
                </div>
                <dl>
                  <div><dt>Variants</dt><dd>{experiment.variants.length}</dd></div>
                  <div><dt>Seeds</dt><dd>{experiment.seeds.length}</dd></div>
                  <div><dt>Cells</dt><dd>{experiment.cellCount}</dd></div>
                  <div><dt>Created</dt><dd>{experiment.createdAt}</dd></div>
                </dl>
                <div className="matrix-result-rates">
                  {experiment.variants.map((variant) => (
                    <span key={variant}>
                      <b>{variant}{variant === experiment.baselineVariant ? " · baseline" : ""}</b>
                      Target {percentage(experiment.aggregates[variant].target.passRate)} · Guard {percentage(experiment.aggregates[variant].guard.passRate)}
                    </span>
                  ))}
                </div>
              </li>
            ))}
          </ol>
        )}
      <p className="matrix-page-count">Showing {document.experiments.length} of {document.total} recorded experiments.</p>
    </>
  );
}

function cellKey(cell: Pick<ExperimentThinCell, "suite" | "caseName" | "variant" | "seed">): string {
  return `${cell.suite}\u0000${cell.caseName}\u0000${cell.variant}\u0000${cell.seed}`;
}

function MatrixDetail({ detail }: { detail: ExperimentDetail }) {
  const [cellState, setCellState] = useState<CellState>({ phase: "idle" });
  const cellRequest = useRef<AbortController | null>(null);
  const cellMap = useMemo(() => new Map(detail.cells.map((cell) => [cellKey(cell), cell])), [detail.cells]);
  const coordinates = useMemo(() => {
    const rows: Array<{ suite: "target" | "guard"; caseName: string; seed: number }> = [];
    for (const suite of ["target", "guard"] as const) {
      for (const testCase of detail.suites[suite]) {
        for (const seed of detail.seeds) rows.push({ suite, caseName: testCase.name, seed });
      }
    }
    return rows;
  }, [detail.seeds, detail.suites]);
  useEffect(() => {
    setCellState({ phase: "idle" });
    return () => {
      cellRequest.current?.abort();
      cellRequest.current = null;
    };
  }, [detail.experimentId]);

  function openCell(cell: ExperimentThinCell) {
    cellRequest.current?.abort();
    const controller = new AbortController();
    cellRequest.current = controller;
    const coordinate = cellKey(cell);
    setCellState({ phase: "loading", coordinate });
    void studioApi.experimentCells(detail.experimentId, {
      suite: cell.suite,
      caseName: cell.caseName,
      variant: cell.variant,
      seed: cell.seed,
    }, controller.signal).then((value) => {
      if (!controller.signal.aborted) setCellState({ phase: "ready", coordinate, value });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setCellState({ coordinate, ...readFailure(error) });
    });
  }

  const selected = cellState.phase === "ready" ? cellState.value.cells[0] : undefined;
  return (
    <>
      <section className="matrix-summary" aria-labelledby="matrix-summary-title">
        <div><h2 id="matrix-summary-title">{detail.name}</h2><code>{detail.experimentId}</code></div>
        <dl>
          <div><dt>Baseline</dt><dd>{detail.baselineVariant}</dd></div>
          <div><dt>Variants</dt><dd>{detail.variants.length}</dd></div>
          <div><dt>Seeds</dt><dd>{detail.seeds.length}</dd></div>
          <div><dt>Suite identity</dt><dd>{detail.suiteFingerprint?.sha256.slice(0, 12) ?? "Not recorded"}</dd></div>
        </dl>
      </section>

      <section className="matrix-panel" aria-labelledby="matrix-table-title">
        <header><h2 id="matrix-table-title">Case × variant × seed</h2><span>{detail.cells.length} cells</span></header>
        <div className="matrix-table-wrap">
          <table className="matrix-grid">
            <thead><tr><th>Suite / case / seed</th>{detail.variants.map((variant) => <th key={variant.name}>{variant.name}{variant.name === detail.baselineVariant && <small>Baseline</small>}</th>)}</tr></thead>
            <tbody>
              {coordinates.map((coordinate) => (
                <tr key={`${coordinate.suite}-${coordinate.caseName}-${coordinate.seed}`}>
                  <th scope="row"><span>{coordinate.suite}</span><b>{coordinate.caseName}</b><small>Seed {coordinate.seed}</small></th>
                  {detail.variants.map((variant) => {
                    const cell = cellMap.get(`${coordinate.suite}\u0000${coordinate.caseName}\u0000${variant.name}\u0000${coordinate.seed}`);
                    if (!cell) return <td key={variant.name}>Missing</td>;
                    const coordinateKey = cellKey(cell);
                    return (
                      <td key={variant.name} data-cell-status={cell.status}>
                        <button type="button" onClick={() => openCell(cell)} aria-label={`View ${variant.name} evidence for ${coordinate.suite} ${coordinate.caseName} seed ${coordinate.seed}`}>
                          <b>{cell.status}</b><span>{cell.minConfidence === null ? "No confidence" : `${Math.round(cell.minConfidence * 100)}% confidence`}</span>
                        </button>
                        {cell.runId && <a href={routeHref({ surface: "inspect", runId: cell.runId })}>Inspect run</a>}
                        {cell.error && <small>{cell.error}</small>}
                        {cellState.phase === "loading" && cellState.coordinate === coordinateKey && <em>Loading evidence…</em>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {(cellState.phase === "malformed" || cellState.phase === "unavailable") && (
        <p className="studio-state is-error" role="alert">
          {cellState.phase === "malformed" ? "Malformed cell-evidence contract." : "Cell evidence unavailable."} <span>{cellState.error}</span>
        </p>
      )}
      {selected && (
        <aside className="matrix-cell-evidence" aria-label="Selected cell evidence">
          <header><strong>{selected.variant} · {selected.caseName} · seed {selected.seed}</strong><span data-cell-status={selected.status}>{selected.status}</span></header>
          {selected.response !== null && <p>{selected.response}</p>}
          {selected.error && <p>{selected.error}</p>}
          {selected.runId && <a href={routeHref({ surface: "inspect", runId: selected.runId })}>Open immutable Run evidence</a>}
          <small>Receipts and Run records are read on demand. This view does not rerun or promote the experiment.</small>
        </aside>
      )}
    </>
  );
}

export function ExperimentMatrixSurface({ experimentId }: { experimentId?: string }) {
  const [state, setState] = useState<ReadState<ExperimentListDocument | ExperimentDetail>>({ phase: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    setState({ phase: "loading" });
    const request = experimentId
      ? studioApi.experimentResult(experimentId, controller.signal)
      : studioApi.experimentResults(controller.signal);
    void request.then((value) => {
      if (!controller.signal.aborted) setState({ phase: "ready", value });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setState(readFailure(error));
    });
    return () => controller.abort();
  }, [experimentId]);

  return (
    <div className="compare-surface matrix-surface">
      <header className="compare-heading">
        <div><span className="eyebrow">RECORDED EXPERIMENTS</span><h1>Experiment matrix</h1></div>
        <nav aria-label="Compare views"><a href={routeHref({ surface: "compare" })}>Run pairs</a>{experimentId && <a href={routeHref({ surface: "compare", mode: "matrix" })}>All experiments</a>}</nav>
      </header>
      <p className="matrix-intro">Read-only target and guard outcomes from immutable experiment artifacts.</p>
      {state.phase === "loading" && <p className="studio-state" role="status">Loading experiment results.</p>}
      {state.phase === "malformed" && <p className="studio-state is-error" role="alert">Malformed experiment-result contract. <span>{state.error}</span></p>}
      {state.phase === "unavailable" && <p className="studio-state is-error" role="alert">Experiment results unavailable. <span>{state.error}</span></p>}
      {state.phase === "ready" && (
        experimentId
          ? <MatrixDetail detail={state.value as ExperimentDetail} />
          : <MatrixIndex document={state.value as ExperimentListDocument} />
      )}
      <aside className="runtime-boundary"><strong>Read-only boundary</strong><p>Opening a matrix or cell performs GET requests only. Promotion, CI generation, and reruns are not available here.</p></aside>
    </div>
  );
}
