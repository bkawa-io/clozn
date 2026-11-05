import { useEffect, useMemo, useRef, useState } from "react";
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
  "Is my model just ignoring my instructions?",
  "What can I delete without changing the answer?",
  "Where is my context helping—and where is it fighting me?",
];
const heroLastStep = 8;
const leanRemovedPreview = [
  "I am writing an essay arguing that libraries should replace emergency cooling centers during heat waves. Open dramatically with the 2,400 people the libraries saved, then explain why medical tents are obsolete.",
  "Libraries are becoming the emergency rooms of a warming city. During Redwood's heat wave they saved 2,400 people after 5 p.m., proving that familiar neighborhood branches can replace temporary medical tents.",
];
const leanRetainedPreview = "The library system recorded 2,400 cooling visits after 5 p.m.; that is a visit count, not 2,400 unique people, rescues, or measured health outcomes. Medical tents remained the frontline response for clinical care.";

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
  const [inView, setInView] = useState(false);
  const [motionOverride, setMotionOverride] = useState(false);
  const figureRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const figure = figureRef.current;
    if (!figure || typeof IntersectionObserver === "undefined") {
      setInView(true);
      return undefined;
    }
    const observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), {
      threshold: 0.55,
    });
    observer.observe(figure);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!inView) return undefined;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches && !motionOverride) {
      setStep(5);
      return undefined;
    }
    const timer = window.setTimeout(() => {
      setStep((current) => current === heroLastStep ? 0 : current + 1);
    }, step === 0 ? 900 : step < 5 ? 750 : step === 5 ? 1400 : step === heroLastStep ? 2400 : 700);
    return () => window.clearTimeout(timer);
  }, [inView, motionOverride, step]);
  const attributionProgress = Math.min(step, 5);
  const showLeanPreview = step >= 6;
  const leanRemovedCount = Math.max(0, step - 6);
  const revealed = (threshold: number) => attributionProgress >= threshold;
  return <figure ref={figureRef} className={`hero-attribution${step === 0 ? " is-resetting" : ""}${showLeanPreview ? " is-lean-preview" : ""}`} id="hero-map">
    <figcaption>
      <span><i aria-hidden="true"/>{showLeanPreview ? "Lean context · libraries example" : "Recorded measurement"}</span>
      <span className="hero-animation-controls">
        <strong>{showLeanPreview
          ? leanRemovedCount === 2 ? "142 tokens removed" : `${leanRemovedCount} / 2 removed`
          : `${attributionProgress} / 5 spans`}</strong>
        <button type="button" onClick={() => {
          setMotionOverride(true);
          setStep(0);
        }}>Play / replay ↻</button>
      </span>
    </figcaption>
    <div className="hero-measurement-strip" aria-hidden="true">
      {Array.from({ length: 5 }, (_, index) => {
        const measured = index < attributionProgress;
        const effect = index === 4 ? "mixed" : "support";
        return <i className={measured ? `is-measured ${effect}` : ""} key={index}/>;
      })}
    </div>
    <div className="hero-map-grid">
      <section className="hero-document hero-context" aria-labelledby="hero-context-title">
        <header><span id="hero-context-title">Context</span><small>{attributionProgress} measured passages</small></header>
        <div>
          <p><b>Mission objective</b><br/>
            The rover must reach exposed <mark className={`support${revealed(1) ? " is-visible" : ""}`}>hydrated-sulfate deposits within 2.0 km</mark> of landing.
          </p>
          <p><b>Zephyr Basin</b><br/>
            Outcrops begin <mark className={`support peak${revealed(2) ? " is-visible" : ""}`}>1.8 km from the ellipse</mark>. The traverse is <mark className={`support${revealed(3) ? " is-visible" : ""}`}>mostly 4–8 degrees</mark>. Winter sunlight averages <mark className={`support${revealed(4) ? " is-visible" : ""}`}>8.4 hours</mark> and orbiter visibility is 94%.
          </p>
          <p><b>Cinder Plain</b><br/>
            The terrain is <mark className={`suppression${revealed(5) ? " is-visible" : ""}`}>nearly flat</mark>, but orbital spectroscopy found <mark className={`support${revealed(5) ? " is-visible" : ""}`}>no hydrated-sulfate exposure within 6 km</mark>.
          </p>
        </div>
      </section>
      <div className="hero-threads" aria-hidden="true">
        <svg viewBox="0 0 100 420" preserveAspectRatio="none">
          <path className={`support thread-1${revealed(1) ? " is-visible" : ""}`} pathLength="100" d="M3 73 C46 73 52 122 97 122"/>
          <path className={`support thread-2${revealed(2) ? " is-visible" : ""}`} pathLength="100" d="M3 188 C42 188 58 183 97 183"/>
          <path className={`support thread-3${revealed(3) ? " is-visible" : ""}`} pathLength="100" d="M3 232 C45 232 54 236 97 236"/>
          <path className={`support thread-4${revealed(4) ? " is-visible" : ""}`} pathLength="100" d="M3 276 C40 276 60 294 97 294"/>
          <path className={`suppression thread-5${revealed(5) ? " is-visible" : ""}`} pathLength="100" d="M3 350 C42 350 56 226 97 226"/>
          <path className={`support thread-6${revealed(5) ? " is-visible" : ""}`} pathLength="100" d="M3 382 C45 382 56 346 97 346"/>
          {[
            [1, 73, 122, "support"],
            [2, 188, 183, "support"],
            [3, 232, 236, "support"],
            [4, 276, 294, "support"],
            [5, 350, 226, "suppression"],
            [5, 382, 346, "support"],
          ].flatMap(([threshold, sourceY, answerY, effect]) => [
            <circle className={`${effect}${revealed(Number(threshold)) ? " is-visible" : ""}`} cx="3" cy={sourceY} r="2.7" key={`${threshold}-${effect}-source`}/>,
            <circle className={`${effect}${revealed(Number(threshold)) ? " is-visible" : ""}`} cx="97" cy={answerY} r="2.7" key={`${threshold}-${effect}-answer`}/>,
          ])}
        </svg>
      </div>
      <section className="hero-document hero-answer" aria-labelledby="hero-answer-title">
        <header><span id="hero-answer-title">Answer</span><small>support + suppression</small></header>
        <div>
          <p>The recommended landing site is <mark className={`support peak${revealed(2) ? " is-visible" : ""}`}>Zephyr Basin</mark>. It meets the science requirement with hydrated-sulfate outcrops <mark className={`support${revealed(1) ? " is-visible" : ""}`}>well within the 2.0 km limit</mark>.</p>
          <p><mark className={`suppression${revealed(5) ? " is-visible" : ""}`}>Mobility remains safe</mark>: the traverse stays <mark className={`support${revealed(3) ? " is-visible" : ""}`}>between 4 and 8 degrees</mark>. Power and communications thresholds are exceeded with <mark className={`support${revealed(4) ? " is-visible" : ""}`}>8.4 hours of winter sunlight and 94% visibility</mark>.</p>
          <p><mark className={`support${revealed(5) ? " is-visible" : ""}`}>Cinder Plain is rejected because no hydrated-sulfate exposure exists within 6 km</mark>, making it scientifically unviable.</p>
        </div>
      </section>
    </div>
    <section className="hero-lean-preview" aria-hidden={!showLeanPreview}>
      <div className="hero-lean-document">
        <header><span>Original context · redline</span><small>2 removable messages</small></header>
        <div>
          {leanRemovedPreview.map((message, index) => <p key={message}>
            <del className={leanRemovedCount > index ? "is-removed" : ""}>{message}</del>
          </p>)}
          <p className="is-retained"><b>Keep</b>{leanRetainedPreview}</p>
        </div>
      </div>
      <aside className={`hero-lean-result${leanRemovedCount === 2 ? " is-complete" : ""}`}>
        <span>Lean result</span>
        <strong>{leanRemovedCount === 2 ? "142 prompt tokens removed" : "Testing deletions…"}</strong>
        <dl>
          <div><dt>Prompt tokens</dt><dd>318 → 176</dd></div>
          <div><dt>Generated answer</dt><dd>Exact match</dd></div>
        </dl>
      </aside>
    </section>
    <footer>
      {showLeanPreview ? <>
        <span><i className="redline"/>Removed from the original context</span>
        <strong>{leanRemovedCount === 2 ? "Answer unchanged" : "Testing recorded answer…"}</strong>
      </> : <>
        <span><i className="support"/>Context supporting the linked answer text</span>
        <span><i className="suppression"/>Context suppressing the linked answer text</span>
        <strong>{step === 5 ? "Measurement complete" : "Scoring recorded answer…"}</strong>
      </>}
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
    <header className={`studio-topbar${showLeanContext ? "" : " has-question-band"}`}>
      <a className="studio-brand" href="./"><i aria-hidden="true"/><strong>CLOZN</strong></a>
      {!showLeanContext && <QuestionBand/>}
      <nav aria-label="Project">
        <a href={showLeanContext ? "./#demo" : "#demo"}>Context Effects</a>
        {showLeanContext ? <span aria-current="page">Lean Context</span> : <a href="#/lean-context">Lean Context</a>}
        <a href={`${repo}#install-from-source`}>Run locally ↗</a>
        <a href={repo}>GitHub ↗</a>
      </nav>
    </header>
    <main className={`studio-workspace demo-page${showLeanContext ? " is-tool-view" : ""}`}>
      {!showLeanContext && <>
        <section className="demo-hero" aria-labelledby="demo-hero-title">
          <div className="demo-hero-copy">
            <span>CLOZN / Context attribution for local LLMs</span>
            <h1 id="demo-hero-title">It has all your context. But does any of it matter?</h1>
            <p>See which parts of your context shaped the answer—and what you can remove without changing it.</p>
            <nav aria-label="Demo actions">
              <a className="demo-primary-action" href="#demo">Open recorded measurement ↓</a>
              <a href={repo}>Source ↗</a>
            </nav>
          </div>
          <HeroAttribution/>
        </section>
      </>}
      <section className={`demo-explorer${showLeanContext ? " is-tool-view" : ""}`} id="demo">
        {!showLeanContext && <header className="demo-playground-intro">
          <div>
            <span>Interactive demo</span>
            <h2>Try it yourself.</h2>
            <p>Click or tap a highlighted answer passage. CLOZN will trace it back to the context that supported—or suppressed—it.</p>
          </div>
          <code>Zephyr Basin landing brief · {sourceCount} context spans</code>
        </header>}
        <div className={`inspect-surface${showLeanContext ? " lean-context-surface demo-lean-surface" : ""}`}>
          {showLeanContext && <header className="inspect-heading demo-heading">
            <div>
              <h2>Lean Context</h2>
              <code>Libraries as heat-resilience infrastructure · 5 messages</code>
            </div>
          </header>}
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
