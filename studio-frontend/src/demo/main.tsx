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
const heroQuestions = [
  "How much of my prompt engineering does absolutely nothing?",
  "Is my token-burning RAG system useful—or just expensive?",
  "Which sentence actually made the model say that?",
  "What can I delete without changing the answer?",
  "Where is my context helping—and where is it fighting me?",
];
const heroProgress = [0, 1, 2, 3, 4, 5];

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

function QuestionBand() {
  return <aside className="demo-question-band" aria-label="Questions Clozn answers">
    <span>Questions Clozn answers</span>
    <div><div className="demo-question-track">
      <span>{heroQuestions.map((question) => <strong key={question}>{question}<i aria-hidden="true"/></strong>)}</span>
      <span aria-hidden="true">
        {heroQuestions.map((question) => <strong key={`repeat-${question}`}>{question}<i/></strong>)}
      </span>
    </div></div>
  </aside>;
}

function HeroAttribution() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setStep(heroProgress.length - 1);
      return undefined;
    }
    const timer = window.setTimeout(() => {
      setStep((current) => current === heroProgress.length - 1 ? 0 : current + 1);
    }, step === 0 ? 600 : step === heroProgress.length - 1 ? 2600 : 850);
    return () => window.clearTimeout(timer);
  }, [step]);
  const revealed = (threshold: number) => step >= threshold;
  return <figure className={`hero-attribution${step === 0 ? " is-resetting" : ""}`} id="hero-map">
    <figcaption>
      <span><i aria-hidden="true"/>Recorded measurement</span>
      <strong>{heroProgress[step]} / 5 context spans</strong>
    </figcaption>
    <div className="hero-measurement-strip" aria-hidden="true">
      {Array.from({ length: 5 }, (_, index) => {
        const measured = index < heroProgress[step];
        const direction = index === 4 ? " suppression" : " support";
        return <i className={measured ? `is-measured${direction}` : ""} key={index}/>;
      })}
    </div>
    <div className="hero-map-grid">
      <section className="hero-document hero-context" aria-labelledby="hero-context-title">
        <header><span id="hero-context-title">Context</span><small>{heroProgress[step]} measured passages</small></header>
        <div>
          <p><b>Mission objective</b><br/>
            The rover must reach exposed <mark className={`support${revealed(1) ? " is-visible" : ""}`}>hydrated-sulfate deposits within 2.0 km</mark> of landing.
          </p>
          <p><b>Zephyr Basin</b><br/>
            Outcrops begin <mark className={`support peak${revealed(2) ? " is-visible" : ""}`}>1.8 km from the ellipse</mark>. The traverse is <mark className={`support${revealed(3) ? " is-visible" : ""}`}>mostly 4–8 degrees</mark>. Winter sunlight averages <mark className={`support${revealed(4) ? " is-visible" : ""}`}>8.4 hours</mark> and orbiter visibility is 94%.
          </p>
          <p><b>Cinder Plain</b><br/>
            Orbital spectroscopy found <mark className={`suppression${revealed(5) ? " is-visible" : ""}`}>no hydrated-sulfate exposure within 6 km</mark>.
          </p>
        </div>
      </section>
      <div className="hero-threads" aria-hidden="true">
        <svg viewBox="0 0 100 420" preserveAspectRatio="none">
          <path className={`support thread-1${revealed(1) ? " is-visible" : ""}`} pathLength="1" d="M0 73 C46 73 52 122 100 122"/>
          <path className={`support thread-2${revealed(2) ? " is-visible" : ""}`} pathLength="1" d="M0 188 C42 188 58 183 100 183"/>
          <path className={`support thread-3${revealed(3) ? " is-visible" : ""}`} pathLength="1" d="M0 232 C45 232 54 236 100 236"/>
          <path className={`support thread-4${revealed(4) ? " is-visible" : ""}`} pathLength="1" d="M0 276 C40 276 60 294 100 294"/>
          <path className={`suppression thread-5${revealed(5) ? " is-visible" : ""}`} pathLength="1" d="M0 365 C45 365 56 346 100 346"/>
          <circle className={`support${revealed(2) ? " is-visible" : ""}`} cx="4" cy="188" r="3"/>
          <circle className={`support${revealed(2) ? " is-visible" : ""}`} cx="96" cy="183" r="3"/>
          <circle className={`suppression${revealed(5) ? " is-visible" : ""}`} cx="4" cy="365" r="3"/>
          <circle className={`suppression${revealed(5) ? " is-visible" : ""}`} cx="96" cy="346" r="3"/>
        </svg>
      </div>
      <section className="hero-document hero-answer" aria-labelledby="hero-answer-title">
        <header><span id="hero-answer-title">Answer</span><small>support + suppression</small></header>
        <div>
          <p>The recommended landing site is <mark className={`support peak${revealed(2) ? " is-visible" : ""}`}>Zephyr Basin</mark>. It meets the science requirement with hydrated-sulfate outcrops <mark className={`support${revealed(1) ? " is-visible" : ""}`}>well within the 2.0 km limit</mark>.</p>
          <p>Mobility remains safe: the traverse stays <mark className={`support${revealed(3) ? " is-visible" : ""}`}>between 4 and 8 degrees</mark>. Power and communications thresholds are exceeded with <mark className={`support${revealed(4) ? " is-visible" : ""}`}>8.4 hours of winter sunlight and 94% visibility</mark>.</p>
          <p><mark className={`suppression${revealed(5) ? " is-visible" : ""}`}>Cinder Plain is rejected because no hydrated-sulfate exposure exists within 6 km</mark>, making it scientifically unviable.</p>
        </div>
      </section>
    </div>
    <footer>
      <span><i className="support"/>Supports the answer</span>
      <span><i className="suppression"/>Suppresses the answer</span>
      <strong>{step === heroProgress.length - 1 ? "Measurement complete" : "Scoring recorded answer…"}</strong>
    </footer>
  </figure>;
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
  useEffect(() => {
    if (showLeanContext) document.querySelector<HTMLElement>(".demo-page")?.scrollTo(0, 0);
  }, [showLeanContext]);
  const route = location.surface === "inspect" && location.runId === runId
    ? location : { surface: "inspect" as const, runId };
  return <div className="studio-shell demo-shell">
    <header className="studio-topbar">
      <a className="studio-brand" href="./"><i aria-hidden="true"/><strong>CLOZN</strong></a>
      <nav aria-label="Project">
        <a href={showLeanContext ? "./#demo" : "#demo"}>Context Effects</a>
        {showLeanContext ? <span aria-current="page">Lean Context</span> : <a href="#/lean-context">Lean Context</a>}
        <a href={`${repo}#install-from-source`}>Run locally ↗</a>
        <a href={repo}>GitHub ↗</a>
      </nav>
    </header>
    <main className={`studio-workspace demo-page${showLeanContext ? " is-tool-view" : ""}`}>
      {!showLeanContext && <>
        <QuestionBand/>
        <section className="demo-hero" aria-labelledby="demo-hero-title">
          <div className="demo-hero-copy">
            <div>
              <span>Context observability for local models</span>
              <h1 id="demo-hero-title">See what your context actually did.</h1>
            </div>
            <div>
              <p>Measure which prompt and retrieved passages support or suppress each part of an answer. Then find the smallest context that preserves it.</p>
              <nav aria-label="Demo actions">
                <a className="demo-primary-action" href="#demo">Explore the recorded demo ↓</a>
                <a href={repo}>View the source ↗</a>
              </nav>
            </div>
          </div>
          <HeroAttribution/>
        </section>
      </>}
      <section className={`demo-explorer${showLeanContext ? " is-tool-view" : ""}`} id="demo">
        <div className={`inspect-surface${showLeanContext ? " lean-context-surface demo-lean-surface" : ""}`}>
          <header className="inspect-heading demo-heading">
            <div>
              <h2>{showLeanContext ? "Lean Context" : "Explore Context Effects"}</h2>
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
        </div>
      </section>
    </main>
  </div>;
}

document.documentElement.dataset.theme = "clozn-dark";
createRoot(document.getElementById("root")!).render(<Demo/>);
