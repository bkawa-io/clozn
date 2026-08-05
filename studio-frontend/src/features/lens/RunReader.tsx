import { useEffect, useMemo, useState, type FormEvent } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import type { RunSummary, RuntimeState } from "../../data/types";
import {
  findInvestigationQuestion,
  INVESTIGATION_QUESTIONS,
  matchInvestigationIntent,
  type InvestigationQuestionId,
} from "../../data/askAnotherQuestion";
import { RunTimingInstrument } from "../diagnostics/RunDiagnostics";
import { ScopePanel } from "../../panels/scope";
import type { ScopeSelectionState, ScopeUrlState } from "../observatory/urlState";
import { ClaimVerification } from "./ClaimVerification";
import { ContextReceipt } from "./ContextReceipt";
import { DiagnosisRepair } from "./DiagnosisRepair";
import { InvestigationExperiment } from "./InvestigationExperiment";
import { Lens } from "./Lens";
import { ReceivedContext } from "./ReceivedContext";
import { RunEventRail, type RunEventRailEvent } from "./RunEventRail";
import { SecondOpinion } from "./SecondOpinion";
import { TimeMachine } from "./TimeMachine";
import { WhatMattered } from "./WhatMattered";
import "./RunReader.css";

/** The reader's section ids are route vocabulary, not labels. Keeping them closed means a legacy URL
 * can only select a real instrument rather than silently falling back to an unrelated surface. */
export const RUN_READER_SECTION_IDS = [
  "read",
  "what-received",
  "what-sent",
  "what-mattered",
  "why",
  "claims",
  "second-opinion",
  "without-passage",
  "timing",
  "time-machine",
  "mechanism",
  "record",
] as const;

export type RunReaderSectionId = (typeof RUN_READER_SECTION_IDS)[number];

export interface RunReaderProps {
  runtime: RuntimeState;
  /** Undefined deliberately means "open the most recent recorded run", as the old bare Lens route did. */
  initialRunId?: string;
  /** A route may name a section directly; unknown values fail closed to the reader. */
  initialSection?: string;
  /** The old Scope route's validated selection, retained when it opens the embedded mechanism drill-down. */
  mechanismState?: ScopeUrlState;
}

interface RunReaderSection {
  id: RunReaderSectionId;
  label: string;
  questionId?: InvestigationQuestionId;
}

const QUESTION_SECTION: Partial<Record<InvestigationQuestionId, RunReaderSectionId>> = {
  why: "why",
  what_received: "what-received",
  what_mattered: "what-mattered",
  claims_supported: "claims",
  retry_correction: "why",
  second_opinion: "second-opinion",
  without_passage: "without-passage",
};

const QUESTION_SECTION_LABEL: Partial<Record<InvestigationQuestionId, string>> = {
  why: "Why",
  what_mattered: "What mattered",
  claims_supported: "Claims",
  second_opinion: "Second opinion",
  without_passage: "Without this passage",
};

// The capability registry supplies the entries; this reader-specific order follows a reading flow:
// overview attribution before diagnosis, then validation and explicit counterfactual work.
const QUESTION_NAV_ORDER: InvestigationQuestionId[] = [
  "what_mattered",
  "why",
  "claims_supported",
  "second_opinion",
  "without_passage",
];

/**
 * The question registry remains the authority for investigation capabilities. Its corrective-retry
 * entry resolves to Why because retry is an explicit offer within that evidence section, not a second
 * destination that would split the causal record in two.
 */
export const RUN_READER_SECTIONS: readonly RunReaderSection[] = [
  { id: "read", label: "Read" },
  { id: "what-received", label: "What it received", questionId: "what_received" },
  { id: "what-sent", label: "What was sent" },
  ...INVESTIGATION_QUESTIONS
    .filter((question) => question.id !== "what_received" && question.id !== "retry_correction")
    .sort((left, right) => QUESTION_NAV_ORDER.indexOf(left.id) - QUESTION_NAV_ORDER.indexOf(right.id))
    .map((question): RunReaderSection => ({
      id: QUESTION_SECTION[question.id]!,
      label: QUESTION_SECTION_LABEL[question.id] ?? question.label,
      questionId: question.id,
    })),
  { id: "timing", label: "Timing" },
  { id: "time-machine", label: "Time machine" },
  { id: "mechanism", label: "Mechanism" },
  { id: "record", label: "The record" },
];

function isRunReaderSection(value: string | undefined): value is RunReaderSectionId {
  return value != null && (RUN_READER_SECTION_IDS as readonly string[]).includes(value);
}

/** Maps a route value through the closed section vocabulary rather than treating a URL substring as UI. */
export function runReaderSection(value?: string): RunReaderSectionId {
  return isRunReaderSection(value) ? value : "read";
}

/** The canonical route stays rooted at one run. A query selects a local instrument without turning it
 * into a second top-level destination. */
export function runReaderHref(
  runId: string,
  section: RunReaderSectionId = "read",
  mechanism?: ScopeSelectionState,
): string {
  const query = new URLSearchParams();
  if (section !== "read") query.set("section", section);
  if (section === "mechanism" && mechanism) {
    query.set("view", mechanism.view);
    query.set("token", String(mechanism.token));
    if (mechanism.reference) query.set("reference", mechanism.reference);
    query.set("layer", String(mechanism.layer));
  }
  const suffix = query.size ? `?${query.toString()}` : "";
  return `#/runs/${encodeURIComponent(runId)}${suffix}`;
}

function selectedRun(runtime: RuntimeState, runId: string): RunSummary | undefined {
  return runtime.runs.find((run) => run.id === runId);
}

function recordEvents(run: RunSummary | undefined): RunEventRailEvent[] {
  if (!run) return [];
  return [
    { id: "recorded", label: "Run recorded", kind: "run-start", timestamp: run.createdAt, status: "complete" },
    {
      id: "finish",
      label: "Generation finished",
      kind: "run-finish",
      detail: run.finishReason ?? "Finish reason was not recorded",
      status: run.finishReason === "length" || run.flags.includes("truncated") ? "warning" : "complete",
    },
    ...run.flags.map((flag, index): RunEventRailEvent => ({
      id: `flag-${index}`,
      label: flag.replaceAll("_", " "),
      kind: "warning",
      status: "warning",
    })),
  ];
}

function RunRecord({ runId, runtime }: { runId: string; runtime: RuntimeState }) {
  const run = selectedRun(runtime, runId);
  const parent = run?.parentRunId ? selectedRun(runtime, run.parentRunId) : undefined;
  const children = runtime.runs.filter((candidate) => candidate.parentRunId === runId);

  if (!run) {
    return (
      <section className="run-reader-record" aria-labelledby="run-reader-record-title">
        <header>
          <span className="eyebrow">THE RECORD</span>
          <h2 id="run-reader-record-title">Recorded ledger</h2>
        </header>
        <EvidenceMark
          variant="chip"
          state="unavailable"
          label="Run record unavailable"
          reason={`No recorded run named "${runId}" is present in this runtime snapshot.`}
        />
      </section>
    );
  }

  const rawRecord = {
    id: run.id,
    label: run.label,
    createdAt: run.createdAt,
    source: run.source,
    client: run.client,
    model: run.model,
    substrate: run.substrate,
    duration: run.duration,
    finishReason: run.finishReason ?? null,
    parentRunId: run.parentRunId ?? null,
    sessionKey: run.sessionKey ?? null,
    flags: run.flags,
    warningCount: run.warningCount,
  };

  return (
    <section className="run-reader-record" aria-labelledby="run-reader-record-title">
      <header>
        <div>
          <span className="eyebrow">THE RECORD</span>
          <h2 id="run-reader-record-title">Recorded ledger</h2>
        </div>
        <span>READ ONLY</span>
      </header>
      <p>
        This is the underlying run ledger: provenance and retained artifact references, not an inferred
        explanation of why the answer happened.
      </p>
      <dl>
        <div><dt>Run id</dt><dd><code>{run.id}</code></dd></div>
        <div><dt>Recorded</dt><dd>{run.createdAt}</dd></div>
        <div><dt>Model</dt><dd>{run.model}</dd></div>
        <div><dt>Client</dt><dd>{run.client}</dd></div>
        <div><dt>Entry point</dt><dd>{run.source}</dd></div>
        <div><dt>Finish</dt><dd>{run.finishReason ?? "Not recorded"}</dd></div>
      </dl>

      <section className="run-reader-record-lineage" aria-label="Run lineage">
        <h3>Lineage</h3>
        {parent ? <a href={runReaderHref(parent.id, "record")}>Parent · {parent.label}</a> : <p>No parent was recorded.</p>}
        {children.length > 0 ? (
          <ul>
            {children.map((child) => <li key={child.id}><a href={runReaderHref(child.id, "record")}>{child.label}</a></li>)}
          </ul>
        ) : <p>No child runs were recorded.</p>}
      </section>

      <RunEventRail events={recordEvents(run)} ariaLabel="Recorded run events" />

      <details className="run-reader-raw-record">
        <summary>Show retained run fields</summary>
        <pre>{JSON.stringify(rawRecord, null, 2)}</pre>
      </details>
    </section>
  );
}

function MissingRun({ section, runId }: { section: RunReaderSectionId; runId: string }) {
  const title = RUN_READER_SECTIONS.find((entry) => entry.id === section)?.label ?? "Run section";
  return (
    <section className="run-reader-missing" aria-labelledby="run-reader-missing-title">
      <header>
        <span className="eyebrow">{title}</span>
        <h2 id="run-reader-missing-title">No run selected</h2>
      </header>
      <EvidenceMark
        variant="chip"
        state="unavailable"
        label="Run unavailable"
        reason={runId
          ? `The route names "${runId}", but no recorded run is available to this section.`
          : "No recorded run is available to this section."}
      />
    </section>
  );
}

function SectionBody({
  section,
  runId,
  runtime,
  mechanismState,
  onRunChange,
  onMechanismStateChange,
}: {
  section: RunReaderSectionId;
  runId: string;
  runtime: RuntimeState;
  mechanismState?: ScopeUrlState;
  onRunChange: (nextRunId: string) => void;
  onMechanismStateChange: (state: ScopeSelectionState) => void;
}) {
  if (!runId) return <MissingRun section={section} runId={runId} />;

  switch (section) {
    case "read":
      return <Lens runtime={runtime} initialRunId={runId} />;
    case "what-received":
      return <ReceivedContext runId={runId} />;
    case "what-sent":
      return <ContextReceipt runId={runId} defaultDetailedOpen />;
    case "what-mattered":
      return <WhatMattered runId={runId} />;
    case "why":
      return <DiagnosisRepair runId={runId} />;
    case "claims":
      return <ClaimVerification runId={runId} />;
    case "second-opinion":
      return <SecondOpinion runId={runId} />;
    case "without-passage":
      return <InvestigationExperiment runId={runId} />;
    case "timing":
      return <RunTimingInstrument runId={runId} />;
    case "time-machine":
      return <TimeMachine runId={runId} />;
    case "mechanism":
      return (
        <ScopePanel
          runtime={runtime}
          inspectorOpen
          params={{
            runId,
            ...(mechanismState?.view ? { view: mechanismState.view } : {}),
            ...(mechanismState?.token != null ? { token: String(mechanismState.token) } : {}),
            ...(mechanismState?.reference ? { reference: mechanismState.reference } : {}),
            ...(mechanismState?.layer != null ? { layer: String(mechanismState.layer) } : {}),
          }}
          embedded
          initialState={mechanismState}
          onRunChange={onRunChange}
          onEmbeddedStateChange={onMechanismStateChange}
        />
      );
    case "record":
      return <RunRecord runId={runId} runtime={runtime} />;
    default: {
      const exhaustive: never = section;
      return exhaustive;
    }
  }
}

export function RunReader({ runtime, initialRunId, initialSection, mechanismState }: RunReaderProps) {
  const [runId, setRunId] = useState(() => initialRunId ?? runtime.runs[0]?.id ?? "");
  const [section, setSection] = useState<RunReaderSectionId>(() => runReaderSection(initialSection));
  const [questionText, setQuestionText] = useState("");
  const [routeNotice, setRouteNotice] = useState<string | null>(null);

  useEffect(() => {
    if (initialRunId) setRunId(initialRunId);
    else if (!runId && runtime.runs[0]?.id) setRunId(runtime.runs[0].id);
  }, [initialRunId, runId, runtime.runs]);

  useEffect(() => {
    setSection(runReaderSection(initialSection));
    setRouteNotice(null);
  }, [initialSection]);

  const run = useMemo(() => selectedRun(runtime, runId), [runtime, runId]);
  const sessionLabel = run?.sessionKey ? `Session ${run.sessionKey}` : null;

  function navigate(nextRunId: string, nextSection: RunReaderSectionId, nextMechanism?: ScopeSelectionState) {
    setRunId(nextRunId);
    setSection(nextSection);
    setRouteNotice(null);
    // A section is a view of the same run, so it belongs in the run URL rather than creating a new
    // route family. The assignment deliberately creates a history entry for user-initiated navigation.
    window.location.hash = runReaderHref(nextRunId, nextSection, nextMechanism);
  }

  function selectSection(nextSection: RunReaderSectionId) {
    if (!runId) {
      setSection(nextSection);
      return;
    }
    navigate(runId, nextSection);
  }

  function routeQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const questionId = matchInvestigationIntent(questionText);
    if (!questionId) {
      setRouteNotice("That text does not match a recorded-evidence question. Choose a section instead.");
      return;
    }
    const question = findInvestigationQuestion(questionId);
    if (question.capability === "unavailable") {
      setRouteNotice(`${question.label} is unavailable: ${question.reason}`);
      return;
    }
    const target = QUESTION_SECTION[question.id];
    if (!target) {
      setRouteNotice(`${question.label} has no run-reader section yet.`);
      return;
    }
    setQuestionText("");
    selectSection(target);
  }

  return (
    <section className="run-reader" aria-label="Run reader">
      <header className="run-reader-header">
        <div className="run-reader-title">
          <span className="eyebrow">RUN</span>
          <h1>{run?.label ?? (runId ? "Recorded run" : "Run reader")}</h1>
          <p>{runId ? <code>{runId}</code> : "Pick a recorded run to inspect its evidence."}</p>
        </div>
        {run && (
          <div className="run-reader-identity" aria-label="Run identity">
            <span><b>MODEL</b>{run.model}</span>
            <span><b>CLIENT</b>{run.client}</span>
            <span><b>ENTRY POINT</b>{run.source}</span>
            {sessionLabel && <a href={`#/sessions/${encodeURIComponent(run.sessionKey!)}/investigate`}>{sessionLabel}</a>}
          </div>
        )}
        {runtime.runs.length > 1 && (
          <label className="run-reader-picker">
            <span>RUN</span>
            <select value={runId} onChange={(event) => navigate(event.target.value, section)}>
              {runtime.runs.map((entry) => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
            </select>
          </label>
        )}
      </header>

      <div className="run-reader-layout">
        <aside className="run-reader-sidebar">
          <nav className="run-reader-nav" aria-label="Run sections">
            {RUN_READER_SECTIONS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                className={section === entry.id ? "is-active" : ""}
                aria-current={section === entry.id ? "page" : undefined}
                onClick={() => selectSection(entry.id)}
              >
                <span>{entry.label}</span>
                {entry.questionId && <small aria-hidden="true">evidence</small>}
              </button>
            ))}
          </nav>

          <form className="run-reader-route-question" aria-label="Route a recorded-evidence question" onSubmit={routeQuestion}>
            <label htmlFor="run-reader-question">
              <span>ROUTE A QUESTION</span>
              <input
                id="run-reader-question"
                value={questionText}
                onChange={(event) => { setQuestionText(event.target.value); setRouteNotice(null); }}
                placeholder="Find recorded evidence; never generates a reply"
              />
            </label>
            <button type="submit" disabled={!questionText.trim()}>ROUTE</button>
          </form>
          {routeNotice && <p className="run-reader-route-notice" role="status">{routeNotice}</p>}
        </aside>

        <main className="run-reader-main" aria-live="polite">
          <SectionBody
            section={section}
            runId={runId}
            runtime={runtime}
            mechanismState={mechanismState}
            onRunChange={(nextRunId) => navigate(nextRunId, "mechanism")}
            onMechanismStateChange={(state) => navigate(runId, "mechanism", state)}
          />
        </main>
      </div>
    </section>
  );
}
