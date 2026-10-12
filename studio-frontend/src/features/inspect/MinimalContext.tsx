import { useEffect, useMemo, useRef, useState } from "react";
import { routeHref, type InspectRoute } from "../../app/router";
import { describeError, isContractError } from "../../data/client";
import {
  isTerminalMinimalContextJob,
  minimalContextApi,
  minimalContextMaterializationBlocker,
  type MinimalContextDurableResult,
  type MinimalContextJob,
  type MinimalContextMaterializationResult,
  type MinimalContextResult,
  type MinimalContextResultList,
  type MinimalContextResultSummary,
  type MinimalContextSourceInspection,
} from "../../data/minimalContext";
import "./minimalContext.css";

type Resource<T> =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable"; message: string }
  | { phase: "malformed"; message: string };

type BusyAction = "idle" | "starting" | "cancelling" | "materializing";

const EXACT_REQUEST = {
  maxUnits: 50,
  maxNewCounterfactualObservations: 32,
  attemptInclusionCheck: true,
} as const;

const CERTIFICATE_COPY = {
  EXACT_MINIMUM: {
    label: "Exact minimum",
    explanation: "Every candidate in the finite search space was directly classified. No smaller preserving set exists in this recorded universe.",
  },
  INCLUSION_MINIMUM: {
    label: "Inclusion minimum",
    explanation: "Removing any one remaining source was directly tested and caused divergence. A different smaller combination may still exist.",
  },
  BEST_VERIFIED: {
    label: "Best verified",
    explanation: "This is the lowest-cost preserving candidate observed within the search budget. A smaller preserving candidate may exist.",
  },
} as const;

function shortId(value: string | null | undefined): string {
  if (!value) return "Unavailable";
  return value.length > 24 ? `${value.slice(0, 11)}…${value.slice(-8)}` : value;
}

function sentence(value: string): string {
  const spaced = value.replaceAll("_", " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function resourceFailure<T>(error: unknown): Resource<T> {
  return {
    phase: isContractError(error) ? "malformed" : "unavailable",
    message: describeError(error),
  };
}

function certificateLabel(certificate: MinimalContextResult["certificate"]): string {
  return certificate ? CERTIFICATE_COPY[certificate].label : "No certificate";
}

function resultLabel(result: MinimalContextResultSummary): string {
  const proof = result.certificate ? certificateLabel(result.certificate) : sentence(result.status);
  const binding = result.currentBinding.status === "current" ? "current" : result.currentBinding.status;
  return `${proof} · ${binding} · ${shortId(result.resultId)}`;
}

function phaseLabel(phase: string): string {
  const labels: Record<string, string> = {
    planning_context: "Planning recorded context",
    checking_exact_eligibility: "Checking exact-search eligibility",
    unchanged_control: "Checking the unchanged control",
    searching: "Searching preserving candidates",
    verifying_candidate: "Verifying the candidate",
    validating: "Validating the evidence",
    persisting: "Persisting the result",
    done: "Complete",
    cancelled: "Cancelled",
  };
  return labels[phase] ?? sentence(phase);
}

function HistoryState({ resource }: { resource: Resource<MinimalContextResultList> }) {
  if (resource.phase === "loading") return <p className="minimal-primary-state" role="status">Reading stored searches…</p>;
  if (resource.phase === "malformed" || resource.phase === "unavailable") {
    return (
      <p className="minimal-primary-state is-error" role="alert">
        {resource.phase === "malformed" ? "Malformed Minimal Context history." : "Minimal Context history unavailable."}
        <span>{resource.message}</span>
      </p>
    );
  }
  return null;
}

function JobPanel({
  job,
  busy,
  error,
  onCancel,
  onRetry,
}: {
  job: MinimalContextJob;
  busy: BusyAction;
  error?: string;
  onCancel: () => void;
  onRetry: () => void;
}) {
  const active = !isTerminalMinimalContextJob(job);
  return (
    <section className={`minimal-job state-${job.state}`} aria-label="Minimal Context search status" aria-live="polite">
      <div>
        <span className="eyebrow">Exact search</span>
        <strong>{job.error?.message ?? phaseLabel(job.progress.phase)}</strong>
        {job.progress.bestRetainedSourceCount !== undefined && (
          <small>Best preserving candidate: {job.progress.bestRetainedSourceCount.toLocaleString()} sources</small>
        )}
      </div>
      <div className="minimal-job-meter">
        <span>{job.progress.completedUnits.toLocaleString()} / {job.progress.totalUnits.toLocaleString()}</span>
        <strong>{Math.round(job.progress.percent)}%</strong>
        <progress max={100} value={job.progress.percent}>{job.progress.percent}%</progress>
      </div>
      {job.progress.certificateCandidateKind && (
        <span className="minimal-candidate-certificate">
          Candidate: {certificateLabel(job.progress.certificateCandidateKind)}
        </span>
      )}
      {active && job.cancellable && (
        <button type="button" disabled={busy !== "idle"} onClick={onCancel}>
          {busy === "cancelling" ? "Cancelling…" : "Cancel search"}
        </button>
      )}
      {error && (
        <p className="minimal-job-error" role="alert">
          <span>{error}</span>
          <button type="button" onClick={onRetry}>{active ? "Retry status" : "Retry stored results"}</button>
        </p>
      )}
    </section>
  );
}

function ProofSummary({ result }: { result: MinimalContextResult }) {
  const certificate = result.certificate ? CERTIFICATE_COPY[result.certificate] : undefined;
  const sourceCount = result.universe.sourceIds.length;
  const retainedCount = result.best?.retainedSourceIds.length;
  const inclusion = result.inclusionCheck;
  const untestedChildren = Math.max(0, inclusion.totalChildCount - inclusion.testedChildCount);
  const reduction = result.reduction;
  return (
    <>
      <section className={`minimal-verdict state-${result.status}`} aria-labelledby="minimal-verdict-title">
        <div className="minimal-collapse-count" aria-label={`${sourceCount} context sources, ${retainedCount ?? "unknown"} retained`}>
          <strong>{sourceCount.toLocaleString()}</strong><span>sources</span>
          <b aria-hidden="true">→</b>
          <strong>{retainedCount?.toLocaleString() ?? "—"}</strong><span>retained</span>
        </div>
        <div>
          <span>{result.status === "completed" ? "Recorded result" : sentence(result.status)}</span>
          <h3 id="minimal-verdict-title">{certificate?.label ?? "No preserving-set certificate"}</h3>
          <p>{certificate?.explanation ?? result.reason ?? "The search did not establish a verified minimum."}</p>
        </div>
      </section>

      <section className="minimal-proof" aria-labelledby="minimal-proof-title">
        <header><span>DIRECT OBSERVATIONS</span><h3 id="minimal-proof-title">Search evidence</h3></header>
        <dl>
          <div><dt>New executions</dt><dd>{result.budget.usedNewExecutions.toLocaleString()} / {result.budget.maxNewExecutions.toLocaleString()}</dd></div>
          <div><dt>Reused observations</dt><dd>{result.budget.reusedObservationCount.toLocaleString()}</dd></div>
          <div><dt>Prompt tokens</dt><dd>{reduction ? `${reduction.originalPromptTokenCost.toLocaleString()} → ${reduction.retainedPromptTokenCost.toLocaleString()}` : "Not established"}</dd></div>
          <div><dt>Token reduction</dt><dd>{reduction ? `${reduction.percent.toLocaleString(undefined, { maximumFractionDigits: 1 })}%` : "Not established"}</dd></div>
          <div><dt>Budget</dt><dd>{result.budget.blockedByBudget ? "Blocked further work" : result.budget.exhausted ? "Exhausted" : "Not exhausted"}</dd></div>
          <div><dt>Stopped because</dt><dd>{sentence(result.stoppingReason)}</dd></div>
        </dl>
        <p>
          {inclusion.attempted
            ? `Inclusion check directly tested ${inclusion.testedChildCount.toLocaleString()} of ${inclusion.totalChildCount.toLocaleString()} one-source deletions${inclusion.complete ? "; the check is complete." : untestedChildren ? `; ${untestedChildren.toLocaleString()} remain unknown.` : "."}`
            : "An inclusion check was not attempted."}
        </p>
        <small>Unknown candidates are not counted as failures.</small>
      </section>
    </>
  );
}

interface DisplaySource {
  sourceId: string;
  disposition: "retained" | "removed";
  inspection?: MinimalContextSourceInspection;
}

function ResultSources({ result }: { result: MinimalContextResult }) {
  const sources = useMemo<readonly DisplaySource[]>(() => {
    const inspections = new Map(result.sourceInspection.map((source) => [source.sourceId, source]));
    const retained = new Set(result.best?.retainedSourceIds ?? []);
    return result.universe.sourceIds.map((sourceId) => {
      const inspection = inspections.get(sourceId);
      return {
        sourceId,
        disposition: inspection?.disposition ?? (retained.has(sourceId) ? "retained" : "removed"),
        ...(inspection ? { inspection } : {}),
      };
    });
  }, [result]);
  const [selectedId, setSelectedId] = useState<string | undefined>(sources[0]?.sourceId);
  useEffect(() => setSelectedId(sources[0]?.sourceId), [result.resultId]);
  const selected = sources.find((source) => source.sourceId === selectedId);
  const retainedCount = sources.filter((source) => source.disposition === "retained").length;
  const removedCount = sources.length - retainedCount;
  return (
    <section className="minimal-sources" aria-labelledby="minimal-sources-title">
      <header>
        <div><span>WINNING CANDIDATE</span><h3 id="minimal-sources-title">Retained and removed sources</h3></div>
        <p>{retainedCount} retained · {removedCount} removed</p>
      </header>
      {sources.length ? (
        <div className="minimal-source-browser">
          <ol>
            {sources.map((source, index) => (
              <li key={source.sourceId}>
                <button
                  type="button"
                  aria-pressed={selectedId === source.sourceId}
                  className={`state-${source.disposition}`}
                  onClick={() => setSelectedId(source.sourceId)}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <strong>{source.inspection?.label ?? `Source ${index + 1}`}</strong>
                  <small>{source.disposition}</small>
                </button>
              </li>
            ))}
          </ol>
          {selected && (
            <article aria-label="Selected source evidence">
              <header>
                <span className={`state-${selected.disposition}`}>{selected.disposition}</span>
                <code title={selected.sourceId}>{shortId(selected.sourceId)}</code>
              </header>
              {selected.inspection?.text !== null && selected.inspection?.text !== undefined
                ? <blockquote>{selected.inspection.text}</blockquote>
                : <p className="minimal-source-text-unavailable">Source text was not supplied by this stored result.</p>}
              <dl>
                <div><dt>Granularity</dt><dd>{selected.inspection ? sentence(selected.inspection.granularity) : "Not recorded"}</dd></div>
                <div><dt>Message</dt><dd>{selected.inspection?.messageIndex === null || selected.inspection?.messageIndex === undefined ? "Not recorded" : selected.inspection.messageIndex + 1}</dd></div>
                <div><dt>Provenance</dt><dd>{selected.inspection?.provenanceKind ? sentence(selected.inspection.provenanceKind) : "Not recorded"}</dd></div>
                <div><dt>Unicode range</dt><dd>{selected.inspection?.unicodeRange ? `${selected.inspection.unicodeRange[0]}–${selected.inspection.unicodeRange[1]}` : "Not recorded"}</dd></div>
              </dl>
            </article>
          )}
        </div>
      ) : <p className="minimal-source-text-unavailable">No source inspection records were stored.</p>}
    </section>
  );
}

function ResultUnavailable({ result }: { result: MinimalContextResult }) {
  return (
    <section className={`minimal-result-state state-${result.status}`} role={result.status === "unavailable" ? "alert" : "status"}>
      <strong>{result.status === "cancelled" ? "Search cancelled" : "Exact search unavailable"}</strong>
      <span>{result.reason ?? (result.status === "cancelled" ? "The search stopped without a durable verified result." : "The recorded answer could not be checked exactly.")}</span>
      {result.reasonCode && <code>{result.reasonCode}</code>}
    </section>
  );
}

function MaterializationPanel({
  result,
  busy,
  confirmation,
  outcome,
  error,
  onAsk,
  onDismiss,
  onConfirm,
}: {
  result: MinimalContextResult;
  busy: BusyAction;
  confirmation: boolean;
  outcome?: MinimalContextMaterializationResult;
  error?: string;
  onAsk: () => void;
  onDismiss: () => void;
  onConfirm: () => void;
}) {
  const blocker = minimalContextMaterializationBlocker(result);
  return (
    <section className="minimal-materialization" aria-labelledby="minimal-materialization-title">
      <div>
        <span>IMMUTABLE CHILD RUN</span>
        <h3 id="minimal-materialization-title">Use this verified context</h3>
        <p>Creating a child Run is a separate explicit action. The recorded parent Run remains unchanged.</p>
      </div>
      {blocker ? (
        <p className="minimal-materialization-blocked" role="status">Not materializable: {blocker}</p>
      ) : outcome?.state === "completed" ? (
        <div className="minimal-materialization-result" role="status">
          <strong>Child Run created</strong>
          <code>{shortId(outcome.childRunId)}</code>
          <a href={routeHref({ surface: "inspect", runId: outcome.childRunId })}>Inspect child</a>
          <a href={routeHref({ surface: "compare", runA: outcome.parentRunId, runB: outcome.childRunId })}>Compare parent and child</a>
        </div>
      ) : confirmation ? (
        <div className="minimal-materialization-confirm" role="group" aria-label="Confirm Minimal Context child Run">
          <strong>Create an immutable child Run?</strong>
          <span>The persisted winner and exact-preservation evidence authorize a fresh child replay with model work.</span>
          <button type="button" disabled={busy !== "idle"} onClick={onConfirm}>
            {busy === "materializing" ? "Creating…" : "Create child Run"}
          </button>
          <button type="button" disabled={busy !== "idle"} onClick={onDismiss}>Keep result only</button>
        </div>
      ) : (
        <button type="button" disabled={busy !== "idle"} onClick={onAsk}>Materialize…</button>
      )}
      {outcome?.state === "failed" && <p className="minimal-action-error" role="alert">{outcome.reason}</p>}
      {error && <p className="minimal-action-error" role="alert">{error}</p>}
    </section>
  );
}

export function MinimalContextSurface({ route }: { route: InspectRoute }) {
  const [history, setHistory] = useState<Resource<MinimalContextResultList>>({ phase: "loading" });
  const [selectedResultId, setSelectedResultId] = useState<string>();
  const [detail, setDetail] = useState<Resource<MinimalContextDurableResult>>({ phase: "idle" });
  const [detailRevision, setDetailRevision] = useState(0);
  const [job, setJob] = useState<MinimalContextJob>();
  const [busy, setBusy] = useState<BusyAction>("idle");
  const [jobError, setJobError] = useState<string>();
  const [pollRevision, setPollRevision] = useState(0);
  const [confirmMaterialization, setConfirmMaterialization] = useState(false);
  const [materialization, setMaterialization] = useState<MinimalContextMaterializationResult>();
  const [materializationError, setMaterializationError] = useState<string>();
  const actionController = useRef<AbortController | undefined>(undefined);
  const refreshedTerminalJob = useRef<string | undefined>(undefined);

  useEffect(() => {
    const controller = new AbortController();
    actionController.current?.abort();
    actionController.current = undefined;
    refreshedTerminalJob.current = undefined;
    setHistory({ phase: "loading" });
    setSelectedResultId(undefined);
    setDetail({ phase: "idle" });
    setJob(undefined);
    setBusy("idle");
    setJobError(undefined);
    setPollRevision(0);
    setConfirmMaterialization(false);
    setMaterialization(undefined);
    setMaterializationError(undefined);
    void minimalContextApi.results(route.runId, controller.signal).then((value) => {
      if (controller.signal.aborted) return;
      setHistory({ phase: "ready", value });
      setSelectedResultId(value.results[0]?.resultId);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setHistory(resourceFailure(error));
    });
    return () => {
      controller.abort();
      actionController.current?.abort();
      actionController.current = undefined;
    };
  }, [route.runId]);

  useEffect(() => {
    if (!selectedResultId) {
      setDetail({ phase: "idle" });
      return undefined;
    }
    const controller = new AbortController();
    setDetail({ phase: "loading" });
    setConfirmMaterialization(false);
    setMaterialization(undefined);
    setMaterializationError(undefined);
    void minimalContextApi.result(route.runId, selectedResultId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setDetail({ phase: "ready", value });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setDetail(resourceFailure(error));
    });
    return () => controller.abort();
  }, [detailRevision, route.runId, selectedResultId]);

  useEffect(() => {
    if (!job || isTerminalMinimalContextJob(job)) return undefined;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const next = await minimalContextApi.job(route.runId, job.jobId, controller.signal);
        if (controller.signal.aborted) return;
        setJob(next);
        setJobError(undefined);
        if (!isTerminalMinimalContextJob(next)) timer = window.setTimeout(() => { void poll(); }, 750);
      } catch (error: unknown) {
        if (!controller.signal.aborted) setJobError(describeError(error));
      }
    };
    void poll();
    return () => {
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.jobId, job?.state, pollRevision, route.runId]);

  useEffect(() => {
    if (!job || !isTerminalMinimalContextJob(job) || refreshedTerminalJob.current === job.jobId) return undefined;
    refreshedTerminalJob.current = job.jobId;
    const controller = new AbortController();
    setDetail({ phase: selectedResultId ? "loading" : "idle" });
    void minimalContextApi.results(route.runId, controller.signal).then((value) => {
      if (controller.signal.aborted) return;
      setHistory({ phase: "ready", value });
      const preferred = job.state === "completed" ? job.result?.resultId : undefined;
      const nextId = preferred && value.results.some((item) => item.resultId === preferred)
        ? preferred
        : selectedResultId && value.results.some((item) => item.resultId === selectedResultId)
          ? selectedResultId
          : value.results[0]?.resultId;
      setSelectedResultId(nextId);
      setDetailRevision((revision) => revision + 1);
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setHistory(resourceFailure(error));
      setJobError(`Stored result refresh failed: ${describeError(error)}`);
    });
    return () => controller.abort();
  }, [job, pollRevision, route.runId, selectedResultId]);

  const beginAction = (kind: BusyAction) => {
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setBusy(kind);
    return controller;
  };

  const start = async () => {
    const controller = beginAction("starting");
    setJobError(undefined);
    setMaterialization(undefined);
    setConfirmMaterialization(false);
    setMaterializationError(undefined);
    try {
      const next = await minimalContextApi.start(route.runId, EXACT_REQUEST, controller.signal);
      if (!controller.signal.aborted) {
        refreshedTerminalJob.current = undefined;
        setJob(next);
      }
    } catch (error: unknown) {
      if (!controller.signal.aborted) setJobError(describeError(error));
    } finally {
      if (!controller.signal.aborted) setBusy("idle");
    }
  };

  const cancel = async () => {
    if (!job?.cancellable) return;
    const controller = beginAction("cancelling");
    setJobError(undefined);
    try {
      const next = await minimalContextApi.cancel(route.runId, job.jobId, controller.signal);
      if (!controller.signal.aborted) setJob(next);
    } catch (error: unknown) {
      if (!controller.signal.aborted) setJobError(describeError(error));
    } finally {
      if (!controller.signal.aborted) setBusy("idle");
    }
  };

  const materialize = async () => {
    if (detail.phase !== "ready" || minimalContextMaterializationBlocker(detail.value)) return;
    const controller = beginAction("materializing");
    setMaterializationError(undefined);
    try {
      const outcome = await minimalContextApi.materialize(route.runId, detail.value, controller.signal);
      if (!controller.signal.aborted) {
        setMaterialization(outcome);
        setConfirmMaterialization(false);
      }
    } catch (error: unknown) {
      if (!controller.signal.aborted) setMaterializationError(describeError(error));
    } finally {
      if (!controller.signal.aborted) setBusy("idle");
    }
  };

  const activeJob = Boolean(job && !isTerminalMinimalContextJob(job));
  const sessionHref = route.sessionId
    ? routeHref({ surface: "session", sessionId: route.sessionId, runId: route.runId })
    : routeHref({ surface: "sessions" });
  const summaries = history.phase === "ready" ? history.value.results : [];
  return (
    <div className="inspect-surface minimal-context-surface">
      <header className="inspect-heading">
        <div>
          <a className="back-link" href={sessionHref}>← {route.sessionId ? "Session" : "Sessions"}</a>
          <h1>Inspect</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {detail.phase === "ready" && (
          <span className={`minimal-overall state-${detail.value.status}`}>{certificateLabel(detail.value.certificate)}</span>
        )}
      </header>

      <section className="inspect-instrument minimal-context-instrument" aria-labelledby="minimal-context-title">
        <header>
          <div><span>EXACT RECORDED OUTPUT</span><h2 id="minimal-context-title">Minimal context</h2></div>
          <code>stored evidence · explicit work</code>
        </header>
        <div className="minimal-boundary">
          <p>Find the smallest directly verified source set that preserves the recorded answer token for token.</p>
          <span>Opening this view reads stored results only.</span>
        </div>

        <HistoryState resource={history} />
        {history.phase === "ready" && (
          <div className="minimal-controls">
            <label>
              <span>Stored result</span>
              <select
                value={selectedResultId ?? ""}
                disabled={!summaries.length || activeJob || busy !== "idle"}
                onChange={(event) => setSelectedResultId(event.target.value || undefined)}
              >
                {!summaries.length && <option value="">No stored results</option>}
                {summaries.map((summary) => <option key={summary.resultId} value={summary.resultId}>{resultLabel(summary)}</option>)}
              </select>
            </label>
            <div>
              <span>Bounded search</span>
              <small>Up to 50 sources · 32 new counterfactual executions · inclusion check</small>
            </div>
            <button type="button" disabled={activeJob || busy !== "idle"} onClick={() => { void start(); }}>
              {busy === "starting" ? "Starting…" : summaries.length ? "Run exact search again" : "Run exact search"}
            </button>
          </div>
        )}

        {job && <JobPanel job={job} busy={busy} error={jobError} onCancel={() => { void cancel(); }} onRetry={() => {
          if (isTerminalMinimalContextJob(job)) refreshedTerminalJob.current = undefined;
          setJobError(undefined);
          setPollRevision((revision) => revision + 1);
        }} />}
        {!job && jobError && <p className="minimal-action-error minimal-top-error" role="alert">{jobError}</p>}

        {detail.phase === "loading" && <p className="minimal-primary-state" role="status">Reading stored result…</p>}
        {(detail.phase === "malformed" || detail.phase === "unavailable") && (
          <p className="minimal-primary-state is-error" role="alert">
            {detail.phase === "malformed" ? "Malformed Minimal Context result." : "Stored Minimal Context result unavailable."}
            <span>{detail.message}</span>
          </p>
        )}
        {history.phase === "ready" && !summaries.length && !job && (
          <section className="minimal-empty" aria-label="No stored Minimal Context result">
            <span>NO STORED SEARCH</span>
            <h3>No verified reduction has been recorded.</h3>
            <p>Starting an exact search performs bounded model work. It does not change this Run.</p>
          </section>
        )}
        {detail.phase === "ready" && (
          <>
            {detail.value.currentBinding.status !== "current" && (
              <p className="minimal-binding-warning" role="alert">
                <strong>{detail.value.currentBinding.status === "stale" ? "Historical result" : "Parent Run unavailable"}</strong>
                <span>{detail.value.currentBinding.reason ?? "This result is not bound to the current parent execution."}</span>
              </p>
            )}
            {detail.value.status !== "completed" && <ResultUnavailable result={detail.value} />}
            <ProofSummary result={detail.value} />
            {detail.value.best && <ResultSources result={detail.value} />}
            <MaterializationPanel
              result={detail.value}
              busy={busy}
              confirmation={confirmMaterialization}
              outcome={materialization}
              error={materializationError}
              onAsk={() => setConfirmMaterialization(true)}
              onDismiss={() => setConfirmMaterialization(false)}
              onConfirm={() => { void materialize(); }}
            />
          </>
        )}
      </section>

      <aside className="minimal-read-boundary">
        <strong>Evidence boundary</strong>
        <p>Stored result reads do not run the model. Search and child-Run creation happen only after their explicit buttons.</p>
      </aside>
    </div>
  );
}
