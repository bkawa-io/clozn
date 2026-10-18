import { useEffect, useState, type FormEvent } from "react";
import { routeHref } from "../../app/router";
import { describeError, isContractError, studioApi, type RunDraft } from "../../data/client";
import type { RunSummary } from "../../data/contracts";
import "./runs.css";

type RunsState =
  | { phase: "loading" }
  | { phase: "ready"; runs: readonly RunSummary[] }
  | { phase: "unavailable" | "malformed"; message: string };

const EXAMPLE: RunDraft = {
  instructions: "Answer from the supplied facts. Return only the requested value.",
  contextBlocks: [
    "The launch code is ORCHID-731.",
    "The launch site is North Annex.",
    "The launch window begins at dawn.",
    "The authorized badge color is amber.",
  ],
  question: "According to the supplied facts, what is the launch code? Answer with the code only.",
};

function formatTime(run: RunSummary): string {
  const date = run.createdTs == null ? new Date(run.createdAt ?? "") : new Date(run.createdTs * 1000);
  if (Number.isNaN(date.getTime())) return "Time not recorded";
  return new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  }).format(date);
}

function shortId(value: string): string {
  return value.length > 22 ? `${value.slice(0, 10)}…${value.slice(-8)}` : value;
}

function Composer({ onCreated }: { onCreated: (runId: string) => void }) {
  const [instructions, setInstructions] = useState("");
  const [contextBlocks, setContextBlocks] = useState([""]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const validBlocks = contextBlocks.filter((block) => block.trim());

  function loadExample() {
    setInstructions(EXAMPLE.instructions);
    setContextBlocks([...EXAMPLE.contextBlocks]);
    setQuestion(EXAMPLE.question);
    setError(undefined);
  }

  function updateBlock(index: number, value: string) {
    setContextBlocks((current) => current.map((block, currentIndex) => currentIndex === index ? value : block));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || !validBlocks.length || busy) return;
    const controller = new AbortController();
    setBusy(true);
    setError(undefined);
    try {
      const created = await studioApi.createRun({ instructions, contextBlocks, question }, controller.signal);
      onCreated(created.runId);
    } catch (requestError: unknown) {
      setError(describeError(requestError));
      setBusy(false);
    }
  }

  return (
    <section className="run-composer" aria-labelledby="new-run-title">
      <header>
        <div><span className="eyebrow">NEW RUN</span><h2 id="new-run-title">Ask with context</h2></div>
        <button type="button" className="secondary-action" disabled={busy} onClick={loadExample}>Load example</button>
      </header>
      <form onSubmit={(event) => { void submit(event); }}>
        <label className="composer-instructions">
          <span>Instructions <small>optional · protected</small></span>
          <textarea value={instructions} disabled={busy} onChange={(event) => setInstructions(event.target.value)} rows={2} placeholder="How should the model answer?" />
        </label>
        <fieldset>
          <legend>Context blocks <small>measured and reducible</small></legend>
          <div className="context-block-editor">
            {contextBlocks.map((block, index) => (
              <label key={index}>
                <span>Context {index + 1}</span>
                <textarea value={block} disabled={busy} onChange={(event) => updateBlock(index, event.target.value)} rows={3} placeholder="Paste one source block" />
                {contextBlocks.length > 1 && (
                  <button type="button" disabled={busy} onClick={() => setContextBlocks((current) => current.filter((_item, currentIndex) => currentIndex !== index))}>Remove</button>
                )}
              </label>
            ))}
          </div>
          <button type="button" className="secondary-action" disabled={busy || contextBlocks.length >= 16} onClick={() => setContextBlocks((current) => [...current, ""])}>
            Add context block
          </button>
        </fieldset>
        <label className="composer-question">
          <span>Question <small>protected</small></span>
          <textarea value={question} disabled={busy} onChange={(event) => setQuestion(event.target.value)} rows={3} placeholder="What should the model answer?" required />
        </label>
        <div className="composer-submit">
          <p>{validBlocks.length} of 16 context blocks · answer limit 128 tokens</p>
          <button type="submit" disabled={busy || !question.trim() || !validBlocks.length}>{busy ? "Creating run…" : "Create run"}</button>
        </div>
        {error && <p className="composer-error" role="alert">Run creation failed. <span>{error}</span></p>}
      </form>
    </section>
  );
}

function RunCard({ run, onDelete }: { run: RunSummary; onDelete: (run: RunSummary) => void }) {
  return (
    <article className="run-card">
      <a href={routeHref({ surface: "inspect", runId: run.id })}>
        <div className="run-card-heading">
          <code>{shortId(run.id)}</code>
          <time dateTime={run.createdAt ?? undefined}>{formatTime(run)}</time>
        </div>
        <h3>{run.promptSummary || "Question not recorded"}</h3>
        <p>{run.responseSummary || "No recorded answer"}</p>
        <div className="run-card-meta">
          <span>{run.model || "Model not recorded"}</span>
          {run.parentRunId && <span>Reduced child</span>}
          {run.flags.map((flag) => <span key={flag}>{flag.replaceAll("-", " ")}</span>)}
        </div>
      </a>
      <button type="button" className="run-delete" onClick={() => onDelete(run)} aria-label={`Delete Run ${run.id}`}>Delete</button>
    </article>
  );
}

export function RunsSurface() {
  const [state, setState] = useState<RunsState>({ phase: "loading" });
  const [revision, setRevision] = useState(0);
  const [deleteError, setDeleteError] = useState<string>();

  useEffect(() => {
    const controller = new AbortController();
    setState({ phase: "loading" });
    void studioApi.runs(controller.signal).then((runs) => {
      if (!controller.signal.aborted) setState({ phase: "ready", runs });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setState({ phase: isContractError(error) ? "malformed" : "unavailable", message: describeError(error) });
      }
    });
    return () => controller.abort();
  }, [revision]);

  async function deleteRun(run: RunSummary) {
    if (!window.confirm(`Delete ${run.id}? This cannot be undone.`)) return;
    setDeleteError(undefined);
    try {
      await studioApi.deleteRun(run.id);
      setRevision((value) => value + 1);
    } catch (error: unknown) {
      setDeleteError(describeError(error));
    }
  }

  return (
    <div className="runs-surface">
      <header className="runs-heading">
        <div><span className="eyebrow">LOCAL EVIDENCE</span><h1>Runs</h1></div>
        {state.phase === "ready" && <span>{state.runs.length} recorded</span>}
      </header>
      <Composer onCreated={(runId) => { window.location.hash = routeHref({ surface: "inspect", runId }); }} />
      <section className="run-index" aria-labelledby="recorded-runs-title">
        <header><h2 id="recorded-runs-title">Recorded runs</h2><button type="button" className="secondary-action" onClick={() => setRevision((value) => value + 1)}>Refresh</button></header>
        {state.phase === "loading" && <p className="studio-state" role="status">Loading runs.</p>}
        {(state.phase === "malformed" || state.phase === "unavailable") && <p className="studio-state is-error" role="alert">{state.phase === "malformed" ? "Malformed run list." : "Run list unavailable."} <span>{state.message}</span></p>}
        {state.phase === "ready" && !state.runs.length && <p className="runs-empty">No runs yet. Load the example or compose the first one.</p>}
        {state.phase === "ready" && state.runs.length > 0 && <div className="run-grid">{state.runs.map((run) => <RunCard key={run.id} run={run} onDelete={(item) => { void deleteRun(item); }} />)}</div>}
        {deleteError && <p className="studio-state is-error" role="alert">Run deletion failed. <span>{deleteError}</span></p>}
      </section>
    </div>
  );
}
