import { useEffect, useMemo, useRef, useState } from "react";
import {
  filterJournal,
  groupRunsByDay,
  journalTime,
  runState,
  runTitle,
  shortId,
  type JournalRun,
} from "./presenters";
import "./runs.css";

interface RunsJournalProps {
  runs: JournalRun[];
  phase: "loading" | "ready" | "error";
  error?: string;
}

export function RunsJournal({ runs, phase, error }: RunsJournalProps) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string>();
  const selectionRails = useRef<HTMLElement>(null);
  const filtered = useMemo(() => filterJournal(runs, query), [runs, query]);
  const groups = useMemo(() => groupRunsByDay(filtered), [filtered]);
  const selected = runs.find((run) => run.id === selectedId);

  useEffect(() => {
    if (!selected || !window.matchMedia?.("(max-width: 900px)").matches) return;
    requestAnimationFrame(() => selectionRails.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  }, [selected]);

  return (
    <div className={`runs-layout${selected ? " has-selection" : ""}`}>
      <section className="runs-journal instrument" aria-labelledby="runs-title">
        <header className="surface-heading">
          <div>
            <span className="eyebrow">EXECUTION JOURNAL</span>
            <h1 id="runs-title">Runs</h1>
            <p>Newest-first record of what happened. Session and lineage appear separately after selection.</p>
          </div>
          <label className="journal-search">
            <span className="sr-only">Search runs</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search prompt, response, model…" />
          </label>
        </header>

        {phase === "loading" && <p className="surface-state" role="status">Reading the local execution journal…</p>}
        {phase === "error" && <p className="surface-state is-error" role="alert">Journal unavailable · {error}</p>}
        {phase === "ready" && groups.length === 0 && <p className="surface-state">No matching runs.</p>}

        <div className="journal-days">
          {groups.map((group) => (
            <section className="journal-day" key={group.day} aria-labelledby={`day-${group.day}`}>
              <header><h2 id={`day-${group.day}`}>{group.day}</h2><span>{group.runs.length}</span></header>
              <ol>
                {group.runs.map((run) => {
                  const state = runState(run);
                  return (
                    <li key={run.id}>
                      <button type="button" className={selectedId === run.id ? "is-selected" : undefined} onClick={() => setSelectedId(run.id)} aria-pressed={selectedId === run.id}>
                        <time>{journalTime(run.createdAt)}</time>
                        <span className={`run-state is-${state}`}>{state}</span>
                        <span className="run-copy">
                          <strong>{runTitle(run)}</strong>
                          {run.response && <span>{run.response}</span>}
                          <small>{[run.model, run.source, shortId(run.id)].filter(Boolean).join(" · ")}</small>
                        </span>
                        {(run.warningCount > 0 || run.flags.length > 0) && <span className="run-signal">{run.warningCount + run.flags.length}</span>}
                      </button>
                    </li>
                  );
                })}
              </ol>
            </section>
          ))}
        </div>
      </section>

      {selected && (
        <aside ref={selectionRails} className="run-context-rails instrument" aria-labelledby="selection-title">
          <header>
            <span className="eyebrow">SELECTED EXECUTION</span>
            <h2 id="selection-title">{shortId(selected.id)}</h2>
            <button type="button" onClick={() => setSelectedId(undefined)} aria-label="Close run selection">×</button>
          </header>
          <div className="selected-run-actions">
            <a className="primary-action" href={`#/runs/${encodeURIComponent(selected.id)}`}>Inspect run</a>
            <a href={`#/time-travel/${encodeURIComponent(selected.id)}`}>Time Travel</a>
            <a href={`#/mri/${encodeURIComponent(selected.id)}`}>Model MRI</a>
          </div>
          <section className="context-rail is-session">
            <header><span>SESSION</span><b>Conversational continuity</b></header>
            {selected.sessionKey
              ? <p><i aria-hidden="true" /> Session <code>{selected.sessionKey}</code></p>
              : <p className="is-absent">No session identity was recorded for this run.</p>}
          </section>
          <section className="context-rail is-lineage">
            <header><span>LINEAGE</span><b>Derivation</b></header>
            {selected.parentRunId
              ? <p><i aria-hidden="true" /> Derived from <code>{shortId(selected.parentRunId)}</code></p>
              : <p><i aria-hidden="true" /> Original run</p>}
          </section>
          <dl className="selected-run-facts">
            <div><dt>Model</dt><dd>{selected.model || "Not recorded"}</dd></div>
            <div><dt>Entry</dt><dd>{selected.source || "Not recorded"}</dd></div>
            <div><dt>Finish</dt><dd>{selected.finishReason || "Not recorded"}</dd></div>
          </dl>
        </aside>
      )}
    </div>
  );
}
