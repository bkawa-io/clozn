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
  prompt: [
    "## Launch code\nThe launch code is ORCHID-731.",
    "## Launch site\nThe launch site is North Annex.",
    "## Launch window\nThe launch window begins at dawn.",
    "## Badge\nThe authorized badge color is amber.",
    "## Question\nAccording to the supplied facts, what is the launch code? Answer with the code only.",
  ].join("\n\n"),
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
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  function loadExample() {
    setInstructions(EXAMPLE.instructions);
    setPrompt(EXAMPLE.prompt);
    setError(undefined);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!prompt.trim() || busy) return;
    const controller = new AbortController();
    setBusy(true);
    setError(undefined);
    try {
      const created = await studioApi.createRun({ instructions, prompt }, controller.signal);
      onCreated(created.runId);
    } catch (requestError: unknown) {
      setError(describeError(requestError));
      setBusy(false);
    }
  }

  return (
    <section className="run-composer" aria-labelledby="new-run-title">
      <header>
        <div><span className="eyebrow">NEW RUN</span><h2 id="new-run-title">Send a prompt</h2></div>
        <button type="button" className="secondary-action" disabled={busy} onClick={loadExample}>Load example</button>
      </header>
      <form onSubmit={(event) => { void submit(event); }}>
        <label className="composer-instructions">
          <span>Instructions <small>optional · analyzed</small></span>
          <textarea value={instructions} disabled={busy} onChange={(event) => setInstructions(event.target.value)} rows={2} placeholder="How should the model answer?" />
        </label>
        <label className="composer-prompt">
          <span>Prompt <small>all input text is analyzed</small></span>
          <textarea value={prompt} disabled={busy} onChange={(event) => setPrompt(event.target.value)} rows={13} placeholder="Enter a question, conversation text, or Markdown context" required />
        </label>
        <div className="composer-submit">
          <p>Input is chunked automatically · answer limit 128 tokens</p>
          <button type="submit" disabled={busy || !prompt.trim()}>{busy ? "Creating run…" : "Create run"}</button>
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
        <h3>{run.promptSummary || "Prompt not recorded"}</h3>
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
