import { useEffect, useMemo, useState } from "react";
import { studioApi } from "../../data/client";
import type {
  JobSnapshot,
  MinimalContextResult,
  MinimalContextResultSummary,
  MinimalContextSourceInspection,
} from "../../data/contracts";
import { certificateLabel, projectMinimalContext, stoppingReasonLabel } from "./fromContracts";
import "./minimal-context.css";

const TERMINAL_JOB_STATES = new Set<JobSnapshot["state"]>(["completed", "failed", "cancelled"]);

function phaseLabel(phase: string): string {
  const labels: Record<string, string> = {
    planning_context: "Planning context",
    checking_exact_eligibility: "Checking exact eligibility",
    unchanged_control: "Checking unchanged control",
    searching: "Searching",
    verifying_candidate: "Verifying candidate",
    validating: "Validating proof",
    persisting: "Persisting result",
    done: "Complete",
    cancelled: "Cancelled",
  };
  return labels[phase] ?? phase.replaceAll("_", " ");
}

function numberText(value: number | undefined): string | undefined {
  return value === undefined ? undefined : value.toLocaleString();
}

function percentageText(value: number | undefined): string | undefined {
  return value === undefined ? undefined : `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
}

function sourceLabel(source: MinimalContextSourceInspection): string {
  return source.label?.trim() || source.sourceId;
}

function SourceGroup({ title, sources }: { title: string; sources: readonly MinimalContextSourceInspection[] }) {
  return (
    <section className="minimal-context-source-group" aria-labelledby={`minimal-context-${title.toLowerCase()}-title`}>
      <header>
        <h3 id={`minimal-context-${title.toLowerCase()}-title`}>{title}</h3>
        <span>{sources.length.toLocaleString()}</span>
      </header>
      {sources.length ? (
        <ul>
          {sources.map((source) => (
            <li key={source.sourceId}>
              <div className="minimal-context-source-heading">
                <strong>{sourceLabel(source)}</strong>
                <span>{source.granularity.replaceAll("_", " ")}</span>
              </div>
              <p>{source.text ?? "Recorded source text unavailable."}</p>
              <code>{source.sourceId}</code>
            </li>
          ))}
        </ul>
      ) : <p className="minimal-context-muted">No persisted source records in this result.</p>}
    </section>
  );
}

function ResultView({ result }: { result: MinimalContextResult }) {
  const projection = useMemo(() => projectMinimalContext(result), [result]);
  const reduction = [
    projection.retainedSourceCount !== undefined && projection.sourceCount !== undefined
      ? `${numberText(projection.retainedSourceCount)} of ${numberText(projection.sourceCount)} context sources retained`
      : undefined,
    projection.reductionPercent !== undefined ? `${percentageText(projection.reductionPercent)} fewer rendered prompt tokens` : undefined,
  ].filter((value): value is string => value !== undefined).join(" · ");
  const tokenCosts = projection.originalPromptTokenCost !== undefined && projection.retainedPromptTokenCost !== undefined
    ? `${numberText(projection.originalPromptTokenCost)} → ${numberText(projection.retainedPromptTokenCost)} rendered prompt tokens`
    : undefined;
  const hasSourceInspection = result.sourceInspection.length > 0;
  const stale = result.currentBinding.status !== "current";

  return (
    <div className="minimal-context-result">
      {stale && <aside className="minimal-context-stale" role="status"><strong>Historical result — Run/context binding has changed</strong><span>{result.currentBinding.reason ?? "This result is retained for historical reading."}</span></aside>}
      {result.status !== "completed" || !result.best ? (
        <section className="minimal-context-unavailable" role="alert">
          <strong>Minimal Context unavailable</strong>
          <p>{result.reason ?? "Exact recorded-output preservation did not produce a verified candidate."}</p>
          <span>Stopping: {stoppingReasonLabel(result.stoppingReason)}</span>
          {result.reasonCode && <code>{result.reasonCode}</code>}
        </section>
      ) : (
        <>
          <section className="minimal-context-result-summary" aria-label="Minimal Context result summary">
            <div>
              <span className="eyebrow">{projection.certificateLabel}</span>
              <h3>{reduction || "Persisted reduction summary unavailable."}</h3>
              {tokenCosts && <p>{tokenCosts}</p>}
            </div>
            <dl>
              <div><dt>Certificate</dt><dd>{certificateLabel(result.certificate)}</dd></div>
              <div><dt>Stopping</dt><dd>{stoppingReasonLabel(result.stoppingReason)}</dd></div>
            </dl>
          </section>
          <section className="minimal-context-search-facts" aria-label="Minimal Context search evidence">
            {projection.newCounterfactualExecutions !== undefined && <span>Search checked {numberText(projection.newCounterfactualExecutions)} new counterfactuals</span>}
            {projection.reusedObservations !== undefined && <span>Reused {numberText(projection.reusedObservations)} prior observations</span>}
            {result.inclusionCheck.attempted && <span>Inclusion check: {numberText(result.inclusionCheck.testedChildCount)} / {numberText(result.inclusionCheck.totalChildCount)} children tested</span>}
          </section>
          {hasSourceInspection ? (
            <div className="minimal-context-source-groups">
              <SourceGroup title="Retained" sources={projection.retained} />
              <SourceGroup title="Removed" sources={projection.removed} />
            </div>
          ) : <p className="minimal-context-muted">Source inspection is unavailable in this persisted result; no source disposition was inferred.</p>}
        </>
      )}
    </div>
  );
}

function ProgressView({ job, onCancel, cancelling }: { job: JobSnapshot<MinimalContextResult>; onCancel: () => void; cancelling: boolean }) {
  const progress = job.progress;
  return (
    <section className="minimal-context-progress" aria-live="polite">
      <div className="minimal-context-progress-heading">
        <div><span className="eyebrow">MINIMAL CONTEXT JOB</span><strong>{phaseLabel(progress.phase)}</strong></div>
        <span>{Math.round(progress.percent)}%</span>
      </div>
      <progress value={progress.percent} max="100" aria-label={`Minimal Context progress ${Math.round(progress.percent)} percent`} />
      <div className="minimal-context-progress-facts">
        <span>{numberText(progress.completedUnits)} / {numberText(progress.totalUnits)} search units</span>
        {progress.bestRetainedSourceCount !== undefined && <span>Best verified: {numberText(progress.bestRetainedSourceCount)} sources retained</span>}
        {progress.certificateCandidateKind && <span>Candidate: {certificateLabel(progress.certificateCandidateKind)}</span>}
        {job.cancellable && <button type="button" disabled={cancelling} onClick={onCancel}>{cancelling ? "Cancelling…" : "Cancel"}</button>}
      </div>
    </section>
  );
}

function History({ summaries, selectedId, onSelect }: { summaries: readonly MinimalContextResultSummary[]; selectedId?: string; onSelect: (resultId: string) => void }) {
  if (!summaries.length) return null;
  return (
    <section className="minimal-context-history" aria-labelledby="minimal-context-history-title">
      <header><span className="eyebrow">DURABLE RESULTS</span><h3 id="minimal-context-history-title">Result history</h3></header>
      <ul>
        {summaries.map((summary) => {
          const isStale = summary.currentBinding.status !== "current";
          return (
            <li key={summary.resultId}>
              <button type="button" className={selectedId === summary.resultId ? "is-selected" : undefined} aria-pressed={selectedId === summary.resultId} onClick={() => onSelect(summary.resultId)}>
                <span>{summary.certificate ? certificateLabel(summary.certificate) : summary.status.replaceAll("_", " ")}</span>
                <strong>{summary.resultId}</strong>
                <small>{isStale ? "Historical result — Run/context binding has changed" : stoppingReasonLabel(summary.stoppingReason)}</small>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export interface MinimalContextPanelProps {
  runId: string;
}

export function MinimalContextPanel({ runId }: MinimalContextPanelProps) {
  const [summaries, setSummaries] = useState<readonly MinimalContextResultSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [result, setResult] = useState<MinimalContextResult | undefined>();
  const [historyPhase, setHistoryPhase] = useState<"loading" | "ready" | "error">("loading");
  const [detailPhase, setDetailPhase] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [job, setJob] = useState<JobSnapshot<MinimalContextResult> | undefined>();
  const [actionError, setActionError] = useState<string | undefined>();
  const [refreshKey, setRefreshKey] = useState(0);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setHistoryPhase("loading");
    setSummaries([]);
    setSelectedId(undefined);
    setResult(undefined);
    setDetailPhase("idle");
    setJob(undefined);
    setActionError(undefined);
    void studioApi.minimalContextResults(runId, controller.signal)
      .then((next) => {
        if (controller.signal.aborted) return;
        setSummaries(next);
        setSelectedId(next[0]?.resultId);
        setHistoryPhase("ready");
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setHistoryPhase("error");
          setActionError(error instanceof Error ? error.message : "Minimal Context results request failed.");
        }
      });
    return () => controller.abort();
  }, [runId, refreshKey]);

  useEffect(() => {
    if (!selectedId) {
      setResult(undefined);
      setDetailPhase("idle");
      return;
    }
    const controller = new AbortController();
    setDetailPhase("loading");
    void studioApi.minimalContextResult(runId, selectedId, controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) {
          setResult(next);
          setDetailPhase("ready");
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          setResult(undefined);
          setDetailPhase("error");
          setActionError(error instanceof Error ? error.message : "Minimal Context result request failed.");
        }
      });
    return () => controller.abort();
  }, [runId, selectedId]);

  useEffect(() => {
    if (!job || TERMINAL_JOB_STATES.has(job.state)) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void studioApi.minimalContextJob(runId, job.jobId, controller.signal)
        .then((next) => {
          if (!controller.signal.aborted) setJob(next);
        })
        .catch((error) => {
          if (!controller.signal.aborted) setActionError(error instanceof Error ? error.message : "Minimal Context job request failed.");
        });
    }, 500);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [job, runId]);

  useEffect(() => {
    if (!job || job.state !== "completed") return;
    // The job's optional terminal payload is operational state. Re-read the persisted
    // result list/detail so the panel only presents durable evidence.
    setRefreshKey((current) => current + 1);
  }, [job?.jobId, job?.state]);

  const start = async () => {
    setStarting(true);
    setActionError(undefined);
    try {
      setJob(await studioApi.startMinimalContext(runId));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Minimal Context could not be started.");
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!job) return;
    setCancelling(true);
    try {
      setJob(await studioApi.cancelMinimalContext(runId, job.jobId));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Minimal Context cancellation failed.");
    } finally {
      setCancelling(false);
    }
  };

  const activeJob = job && !TERMINAL_JOB_STATES.has(job.state) ? job : undefined;
  const showNotRun = historyPhase === "ready" && summaries.length === 0 && !activeJob;

  return (
    <section className="minimal-context-panel" aria-labelledby="minimal-context-title">
      <header className="minimal-context-panel-heading">
        <div>
          <span className="eyebrow">RUN → ANSWER</span>
          <h2 id="minimal-context-title">Minimal Context</h2>
          <p>Find the smallest recorded context that still reproduces this answer exactly.</p>
        </div>
        <button className="minimal-context-primary-action" type="button" onClick={() => void start()} disabled={starting || Boolean(activeJob)}>
          {starting ? "Starting…" : "Reduce context"}
        </button>
      </header>

      {historyPhase === "loading" && <p className="minimal-context-state" role="status">Reading persisted Minimal Context results…</p>}
      {historyPhase === "error" && <p className="minimal-context-state is-error" role="alert">Minimal Context unavailable · {actionError}</p>}
      {actionError && historyPhase !== "error" && <p className="minimal-context-state is-error" role="alert">{actionError}</p>}
      {activeJob && <ProgressView job={activeJob} cancelling={cancelling} onCancel={() => { if (!cancelling) void cancel(); }} />}
      {job?.state === "cancelled" && <p className="minimal-context-state">Minimal Context search cancelled. No new durable result was published.</p>}
      {job?.state === "failed" && <p className="minimal-context-state is-error" role="alert">Minimal Context search failed · {job.error?.message ?? "The shared job did not complete."}</p>}
      {detailPhase === "loading" && <p className="minimal-context-state" role="status">Reading the persisted result detail…</p>}
      {detailPhase === "error" && <p className="minimal-context-state is-error" role="alert">The selected Minimal Context result could not be read.</p>}
      {showNotRun && <section className="minimal-context-not-run"><strong>Not run yet</strong><p>No Minimal Context result has been recorded for this Run.</p></section>}
      {result && detailPhase === "ready" && <ResultView result={result} />}
      <History summaries={summaries} selectedId={selectedId} onSelect={setSelectedId} />
    </section>
  );
}
