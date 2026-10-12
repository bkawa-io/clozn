import { useEffect, useState, type FormEvent } from "react";
import { routeHref, type CompareRoute } from "../../app/router";
import { describeError, isContractError, studioApi } from "../../data/client";
import type { ComparisonDifference, RunComparison } from "../../data/runComparison";
import { ExperimentMatrixSurface } from "./ExperimentMatrix";
import "./compare.css";

type LoadState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "ready"; value: RunComparison }
  | { phase: "unavailable"; error: string }
  | { phase: "malformed"; error: string };

type RunCompareRoute = Exclude<CompareRoute, { mode: "matrix" }>;

const AXIS_LABELS: Record<RunComparison["axes"][number]["id"], string> = {
  model: "Model",
  adapter: "Adapter",
  template: "Template",
  context: "Context",
  sampling: "Sampling",
  engine: "Engine",
  tool_parse: "Tool parse",
  output: "Output",
};

function displayValue(value: unknown, recorded: boolean): string {
  if (!recorded) return "Not recorded";
  if (value === null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return "Recorded value could not be displayed";
  }
}

function DifferenceValue({ difference, side }: { difference: ComparisonDifference; side: "a" | "b" }) {
  const key = side === "a" ? "valueA" : "valueB";
  return <code>{displayValue(difference[key], key in difference)}</code>;
}

function PairForm({ initialA = "", initialB = "" }: { initialA?: string; initialB?: string }) {
  const [runA, setRunA] = useState(initialA);
  const [runB, setRunB] = useState(initialB);
  const valid = Boolean(runA.trim() && runB.trim() && runA !== runB);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (valid) window.location.hash = routeHref({ surface: "compare", runA, runB });
  }

  return (
    <form className="compare-pair-form" onSubmit={submit} aria-label="Choose recorded runs">
      <label>
        <span>Reference run A</span>
        <input value={runA} onChange={(event) => setRunA(event.target.value)} placeholder="run_…" autoComplete="off" />
      </label>
      <span className="compare-arrow" aria-hidden="true">→</span>
      <label>
        <span>Candidate run B</span>
        <input value={runB} onChange={(event) => setRunB(event.target.value)} placeholder="run_…" autoComplete="off" />
      </label>
      <button type="submit" disabled={!valid}>Compare runs</button>
      {runA && runB && runA === runB && <small role="alert">Choose two distinct recorded runs.</small>}
    </form>
  );
}

function ComparisonResult({ comparison }: { comparison: RunComparison }) {
  return (
    <>
      {comparison.privacyLimited && (
        <p className="compare-privacy" role="status">
          Privacy-limited comparison: at least one dimension was compared from retained metadata rather than full content.
        </p>
      )}

      <section className="compare-panel" aria-labelledby="compare-summary-title">
        <header><h2 id="compare-summary-title">Change summary</h2><span>{comparison.differences.length} differences</span></header>
        <ul className="compare-axes" aria-label="Comparison axes">
          {comparison.axes.map((axis) => (
            <li key={axis.id} data-axis-status={axis.status}>
              <b>{AXIS_LABELS[axis.id]}</b>
              <span>{axis.status}</span>
              {axis.note && <small>{axis.note}</small>}
            </li>
          ))}
        </ul>
      </section>

      <section className="compare-panel" aria-labelledby="compare-findings-title">
        <header><h2 id="compare-findings-title">Recorded findings</h2><span>{comparison.findings.length}</span></header>
        {comparison.findings.length === 0
          ? <p className="compare-empty">No classified findings were recorded for this pair.</p>
          : (
            <ol className="compare-findings">
              {comparison.findings.map((finding, index) => (
                <li key={`${finding.classification}-${index}`}>
                  <div><b>{finding.classification.replaceAll("_", " ")}</b><span data-finding-status={finding.status}>{finding.status.replaceAll("_", " ")}</span></div>
                  <p>{finding.summary}</p>
                  {finding.dimensions.length > 0 && <small>Evidence: {finding.dimensions.join(", ")}</small>}
                </li>
              ))}
            </ol>
          )}
        <p className="compare-evidence-note">These are structural observations. They do not establish causation without a separate controlled test.</p>
      </section>

      <section className="compare-panel" aria-labelledby="compare-differences-title">
        <header><h2 id="compare-differences-title">Ranked differences</h2><span>Presentation order</span></header>
        {comparison.differences.length === 0
          ? <p className="compare-empty">No recorded dimensions differed.</p>
          : (
            <div className="compare-table-wrap">
              <table className="compare-table">
                <thead><tr><th>Dimension</th><th>State</th><th>Run A</th><th>Run B</th></tr></thead>
                <tbody>
                  {comparison.differences.map((difference, index) => (
                    <tr key={`${difference.dimension}-${index}`} data-difference-kind={difference.kind}>
                      <th scope="row"><code>{difference.dimension}</code>{difference.note && <small>{difference.note}</small>}</th>
                      <td>{difference.kind.replaceAll("_", " ")}</td>
                      <td><DifferenceValue difference={difference} side="a" /></td>
                      <td><DifferenceValue difference={difference} side="b" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </section>
    </>
  );
}

function RunCompareSurface({ route }: { route: RunCompareRoute }) {
  const [state, setState] = useState<LoadState>({ phase: "idle" });
  const hasPair = route.runA !== undefined && route.runB !== undefined;

  useEffect(() => {
    if (!hasPair) {
      setState({ phase: "idle" });
      return;
    }
    const controller = new AbortController();
    setState({ phase: "loading" });
    void studioApi.runComparison(route.runA, route.runB, controller.signal).then((value) => {
      if (!controller.signal.aborted) setState({ phase: "ready", value });
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setState({ phase: isContractError(error) ? "malformed" : "unavailable", error: describeError(error) });
    });
    return () => controller.abort();
  }, [hasPair, route.runA, route.runB]);

  return (
    <div className="compare-surface">
      <header className="compare-heading">
        <div><span className="eyebrow">RECORDED RUN DELTA</span><h1>Compare</h1></div>
        <nav aria-label="Compare views">
          <a href={routeHref({ surface: "compare", mode: "matrix" })}>Experiment matrix</a>
          {hasPair && (
            <>
            <a href={routeHref({ surface: "inspect", runId: route.runA })}>Inspect A</a>
            <a href={routeHref({ surface: "inspect", runId: route.runB })}>Inspect B</a>
            <a href={routeHref({ surface: "compare", runA: route.runB, runB: route.runA })}>Swap</a>
            </>
          )}
        </nav>
      </header>

      <PairForm key={`${route.runA ?? ""}:${route.runB ?? ""}`} initialA={route.runA} initialB={route.runB} />

      {!hasPair && <p className="compare-intro">Enter two immutable Run IDs to inspect their recorded identity, context, settings, and output differences.</p>}
      {state.phase === "loading" && <p className="studio-state" role="status">Loading comparison.</p>}
      {state.phase === "malformed" && <p className="studio-state is-error" role="alert">Malformed comparison contract. <span>{state.error}</span></p>}
      {state.phase === "unavailable" && <p className="studio-state is-error" role="alert">Comparison unavailable. <span>{state.error}</span></p>}
      {state.phase === "ready" && <ComparisonResult comparison={state.value} />}
    </div>
  );
}

export function CompareSurface({ route }: { route: CompareRoute }) {
  return "mode" in route && route.mode === "matrix"
    ? <ExperimentMatrixSurface experimentId={route.experimentId} />
    : <RunCompareSurface route={route} />;
}
