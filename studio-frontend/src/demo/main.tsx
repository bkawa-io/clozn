import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/spline-sans";
import "@fontsource-variable/spline-sans-mono";
import "../styles/tokens.css";
import "../styles/base.css";
import "../styles/shell.css";
import { useHashRoute } from "../app/useHashRoute";
import { decodeContextInvestigationReader, decodeContextInvestigationQuery } from "../data/contextInvestigation";
import { InspectReader, LeanContextPanel } from "../features/inspect/Inspect";
import evidence from "./flagship.json";
import { leanDemoJob, leanDemoReader, leanDemoRunId } from "./leanFlagship";
import "./demo.css";

const runId = evidence.reader.run_id;
const queries = new Map(Object.entries(evidence.queries));
const repo = "https://github.com/bkawa-io/clozn";
const sourceCount = evidence.reader.measurement.source_universe.length;
const replay = new URLSearchParams(window.location.search).get("replay") === "1";

type JsonObject = Record<string, unknown>;

function object(value: unknown, name: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as JsonObject;
}

function recordedReaderAt(completedSourceCount: number): unknown {
  const document = structuredClone(evidence.reader) as unknown;
  if (completedSourceCount >= sourceCount) return document;
  const reader = object(document, "reader");
  const measurement = object(reader.measurement, "reader.measurement");
  const capabilities = object(reader.capabilities, "reader.capabilities");
  const universe = measurement.source_universe;
  if (!Array.isArray(universe) || universe.some((value) => typeof value !== "string")) {
    throw new TypeError("reader.measurement.source_universe must contain strings");
  }
  const measured = new Set(universe.slice(0, completedSourceCount));
  const status = completedSourceCount ? "partial" : "not_measured";
  reader.status = status;
  measurement.status = status;
  measurement.completed_source_count = completedSourceCount;
  measurement.not_measured_source_count = universe.length - completedSourceCount;
  if (!completedSourceCount) {
    measurement.experiment_id = null;
    measurement.baseline_observation_id = null;
  }
  capabilities.measurement = status;

  const loci = Array.isArray(reader.loci) ? reader.loci : [];
  reader.loci = loci.filter((value) => {
    const locus = object(value, "reader.loci[]");
    return typeof locus.source_id === "string" && measured.has(locus.source_id);
  });
  const visibleLocusIds = new Set((reader.loci as unknown[]).map((value) => {
    const locus = object(value, "reader.loci[]");
    return locus.locus_id;
  }));
  const sources = Array.isArray(reader.sources) ? reader.sources : [];
  reader.sources = sources.map((value) => {
    const source = object(value, "reader.sources[]");
    const sourceId = source.source_id;
    if (typeof sourceId !== "string" || !universe.includes(sourceId)) return source;
    if (measured.has(sourceId)) {
      const sourceLoci = Array.isArray(source.loci) ? source.loci : [];
      source.loci = sourceLoci.filter((locusId) => visibleLocusIds.has(locusId));
      return source;
    }
    source.measurement = {
      status: "not_measured",
      experiment_id: null,
      arm_id: null,
      observation_id: null,
    };
    source.summary = {
      max_abs_token_delta_nats: null,
      total_continuation_delta_nats: null,
      above_floor_locus_count: 0,
    };
    source.loci = [];
    return source;
  });
  return reader;
}

async function savedQuery(_id: string, request: { answerStart: number; answerEnd: number }) {
  const saved = queries.get(`${request.answerStart}:${request.answerEnd}`);
  if (!saved) throw new Error("Choose a highlighted passage from the recorded answer.");
  return decodeContextInvestigationQuery(saved, "Recorded passage");
}

function Demo() {
  const [completedSourceCount, setCompletedSourceCount] = useState(replay ? 0 : sourceCount);
  useEffect(() => {
    if (!replay) return undefined;
    const delay = completedSourceCount === 0
      ? 900
      : completedSourceCount === sourceCount ? 2200 : 650;
    const timer = window.setTimeout(() => {
      setCompletedSourceCount((current) => current === sourceCount ? 0 : current + 1);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [completedSourceCount]);
  const reader = useMemo(() => decodeContextInvestigationReader(
    recordedReaderAt(completedSourceCount), "Recorded example", runId,
  ), [completedSourceCount]);
  const location = useHashRoute();
  const showLeanContext = window.location.hash.startsWith("#/lean-context");
  const route = location.surface === "inspect" && location.runId === runId
    ? location : { surface: "inspect" as const, runId };
  return <div className="studio-shell demo-shell">
    <header className="studio-topbar">
      <a className="studio-brand" href="./"><i aria-hidden="true"/><strong>CLOZN</strong></a>
      <nav aria-label="Project">
        {showLeanContext ? <a href="./">Context Effects</a> : <span aria-current="page">Context Effects</span>}
        {showLeanContext ? <span aria-current="page">Lean Context</span> : <a href="#/lean-context">Lean Context</a>}
        <a href={`${repo}#install-from-source`}>Run locally ↗</a>
        <a href={repo}>GitHub ↗</a>
      </nav>
    </header>
    <main className="studio-workspace"><div className={`inspect-surface${showLeanContext ? " lean-context-surface demo-lean-surface" : ""}`}>
      <header className="inspect-heading demo-heading">
        <div>
          <h1>{showLeanContext ? "Lean Context" : "Context Effects"}</h1>
          <code>{showLeanContext
            ? "Libraries as heat-resilience infrastructure · 5 messages"
            : `Zephyr Basin landing brief · ${sourceCount} context spans`}</code>
        </div>
        {replay && !showLeanContext && <span>Recorded replay</span>}
      </header>
      <section className={`inspect-instrument${showLeanContext ? " lean-context-instrument" : ""}`} aria-label={showLeanContext ? "Recorded lean context" : "Recorded context effects"}>
        {showLeanContext
          ? <LeanContextPanel
              reader={leanDemoReader}
              route={{ surface: "inspect", runId: leanDemoRunId, view: "lean-context" }}
              initialJob={leanDemoJob}
              allowSearch={false}
            />
          : <InspectReader reader={reader} route={route} querySelection={savedQuery} initialAnswerLens="influence"/>}
      </section>
    </div></main>
  </div>;
}

document.documentElement.dataset.theme = "clozn-dark";
createRoot(document.getElementById("root")!).render(<Demo/>);
