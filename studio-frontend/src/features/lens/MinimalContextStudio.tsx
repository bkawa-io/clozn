import { useEffect, useMemo, useState } from "react";
import {
  branchMinimalContextWinner,
  cancelMinimalContextJob,
  loadMinimalContextRun,
  pollMinimalContextJob,
  startMinimalContextJob,
  type MinimalContextCertificate,
  type MinimalContextJob,
  type MinimalContextResult,
  type MinimalContextRunDetail,
  type MinimalContextSourceUnit,
} from "../../data/minimalContext";

const CERTIFICATE_LABEL: Record<MinimalContextCertificate, string> = {
  INCLUSION_MINIMUM: "INCLUSION-MINIMAL",
  BEST_VERIFIED: "BEST VERIFIED",
};

const CERTIFICATE_EXPLANATION: Record<MinimalContextCertificate, string> = {
  INCLUSION_MINIMUM: "The recorded answer was preserved, and deleting any one remaining source was directly tested and caused divergence.",
  BEST_VERIFIED: "Lowest-cost preserving candidate observed within the search budget. A smaller preserving candidate may exist.",
};

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function phaseLabel(phase: string): string {
  if (phase === "planning_context") return "Planning context";
  if (phase === "checking_exact_eligibility") return "Checking exact eligibility";
  if (phase === "unchanged_control") return "Checking unchanged control";
  if (phase === "searching") return "Searching";
  if (phase === "verifying_candidate") return "Verifying candidate";
  if (phase === "validating") return "Validating proof";
  if (phase === "persisting") return "Persisting result";
  return phase.replaceAll("_", " ");
}

function numberText(value: unknown, fallback = "—") {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : fallback;
}

function contextCatalog(detail: MinimalContextRunDetail | null): Map<string, MinimalContextSourceUnit> {
  const map = new Map<string, MinimalContextSourceUnit>();
  for (const unit of detail?.context_units?.units ?? []) map.set(unit.source_id, unit);
  for (const [rowIndex, row] of (detail?.context_receipt?.delivered ?? []).entries()) {
    if (row.segment_id && detail?.messages) {
      const index = row.sources?.[0]?.message_index ?? rowIndex;
      if (index >= 0 && !map.has(row.segment_id)) {
        const message = detail.messages[index];
        map.set(row.segment_id, {
          source_id: row.segment_id,
          message_index: index,
          role: String(message?.role ?? ""),
          unicode_range: [0, String(message?.content ?? "").length],
          source_kind: "whole_message",
          derivation: "message_root",
          source_label: row.source_label,
          provenance_kind: "message",
        });
      }
    }
    for (const source of row.sources ?? []) if (source.source_id) map.set(source.source_id, source);
  }
  return map;
}

function unitText(detail: MinimalContextRunDetail | null, unit?: MinimalContextSourceUnit): string {
  if (!unit) return "Recorded source text unavailable.";
  const message = detail?.messages?.[unit.message_index];
  const content = typeof message?.content === "string" ? message.content : "";
  return content.slice(unit.unicode_range[0], unit.unicode_range[1]);
}

function sourceKind(unit: MinimalContextSourceUnit): string {
  if (unit.derivation === "caller_explicit") return "EXPLICIT CALLER SOURCE";
  if (unit.derivation === "caller_fallback_root") return "EXPLICIT MESSAGE ROOT";
  if (unit.derivation === "auto_structural" || unit.parent_source_id) return "AUTO-DERIVED STRUCTURE";
  return "MESSAGE ROOT";
}

function Hero({ result }: { result: MinimalContextResult }) {
  const certificate = result.certificate ?? null;
  const sourceCount = result.universe.source_ids.length;
  const retained = result.best?.retained_source_ids.length;
  return (
    <section className="minimal-context-hero" aria-label="Minimal Context result">
      <div className="minimal-context-hero-count">
        <strong>{numberText(sourceCount)}</strong><span>CONTEXT UNITS</span>
        <b aria-hidden="true">→</b>
        <strong>{numberText(retained)}</strong><span>RETAINED</span>
      </div>
      <div className="minimal-context-hero-proof">
        <span>{certificate ? CERTIFICATE_LABEL[certificate] : result.status.replaceAll("_", " ").toUpperCase()}</span>
        {certificate && <p>{CERTIFICATE_EXPLANATION[certificate]}</p>}
      </div>
    </section>
  );
}

function SearchProof({ result }: { result: MinimalContextResult }) {
  const { budget, inclusion_check: inclusion } = result;
  const unknown = Math.max(0, inclusion.total_child_count - inclusion.tested_child_count);
  return (
    <section className="minimal-context-section" aria-labelledby="minimal-context-proof-title">
      <header><span className="eyebrow">DIRECT SEARCH</span><h2 id="minimal-context-proof-title">Observed evidence</h2></header>
      <div className="minimal-context-coverage-summary">
        <strong>{numberText(budget.used_new_executions)} new executions</strong>
        <span>{numberText(budget.reused_observation_count)} reused observations</span>
        <span>{numberText(budget.max_new_executions)} execution budget</span>
      </div>
      {inclusion.attempted ? (
        <p className="minimal-context-muted">
          Inclusion check: {inclusion.tested_child_count.toLocaleString()} / {inclusion.total_child_count.toLocaleString()} children directly tested
          {inclusion.complete ? "; all diverged." : unknown ? `; ${unknown.toLocaleString()} remain unknown.` : "."}
        </p>
      ) : <p className="minimal-context-muted">Inclusion check was not requested.</p>}
      <p className="minimal-context-muted">Rendered prompt tokens: {numberText(result.best?.rendered_prompt_token_cost)}</p>
      <p className="minimal-context-unmeasured">Unknown candidates are not counted as failures.</p>
    </section>
  );
}

function ContextCollapse({ detail, result }: { detail: MinimalContextRunDetail | null; result: MinimalContextResult }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const catalog = useMemo(() => contextCatalog(detail), [detail]);
  const unitDerivation = new Map((detail?.context_units?.units ?? []).map((unit) => [unit.source_id, unit.derivation]));
  const retained = new Set(result.best?.retained_source_ids ?? []);
  const selected = selectedId ? catalog.get(selectedId) : undefined;
  const protectedIndices = new Set(detail?.context_units?.protected_message_indices ?? []);
  const protectedMessages = [...protectedIndices].map((index) => detail?.messages?.[index]).filter(Boolean);
  return (
    <section className="minimal-context-section" aria-labelledby="minimal-context-collapse-title">
      <header><span className="eyebrow">CONTEXT COLLAPSE</span><h2 id="minimal-context-collapse-title">Retained versus omitted</h2></header>
      <div className="minimal-context-unit-list">
        {result.universe.source_ids.map((sourceId, index) => {
          const unit = catalog.get(sourceId) ?? { source_id: sourceId, message_index: 0, role: "", unicode_range: [0, 0] as [number, number] };
          const isRetained = retained.has(sourceId);
          const derivation = unitDerivation.get(sourceId) ?? (unit.parent_source_id ? "auto_structural" : unit.derivation);
          return (
            <button key={sourceId} type="button" className={`minimal-context-unit ${isRetained ? "is-retained" : "is-omitted"} ${selectedId === sourceId ? "is-selected" : ""}`} onClick={() => setSelectedId(selectedId === sourceId ? null : sourceId)} aria-pressed={selectedId === sourceId}>
              <span>{String(index + 1).padStart(2, "0")}</span><b aria-hidden="true">{isRetained ? "█" : "░"}</b><strong>{unit.source_label ?? unit.role ?? "Context"}</strong><small>{derivation === "caller_explicit" ? "EXPLICIT" : "AUTO"}</small>
            </button>
          );
        })}
      </div>
      <div className="minimal-context-protected"><span className="eyebrow">PROTECTED CURRENT REQUEST</span>{protectedMessages.length ? protectedMessages.map((message, index) => <p key={index}><strong>{String(message?.role ?? "message").toUpperCase()}</strong>{message?.content}</p>) : <p>Protected request text was not recorded.</p>}</div>
      {selected && <aside className="minimal-context-source-detail" aria-label="Selected source detail"><header><span>{selected.role.toUpperCase()} · MESSAGE {selected.message_index + 1}</span><code>{selected.source_id}</code></header><blockquote>{unitText(detail, selected)}</blockquote><dl><div><dt>Range</dt><dd>{selected.unicode_range[0]}–{selected.unicode_range[1]} Unicode</dd></div><div><dt>Provenance</dt><dd>{sourceKind({ ...selected, derivation: unitDerivation.get(selected.source_id) ?? selected.derivation })}</dd></div><div><dt>Intervention</dt><dd>{retained.has(selected.source_id) ? "Retained in preserving candidate" : "Omitted by candidate intervention"}</dd></div>{selected.source_label && <div><dt>Caller label</dt><dd>{selected.source_label}</dd></div>}</dl></aside>}
    </section>
  );
}

function Unavailable({ result }: { result: MinimalContextResult }) {
  return <section className="minimal-context-error" role="alert"><strong>MINIMAL CONTEXT UNAVAILABLE</strong><span>{result.reason ?? "Exact recorded-answer control could not be reproduced."}</span>{result.reason_code && <code>code: {result.reason_code}</code>}</section>;
}

function Progress({ job, onCancel }: { job: MinimalContextJob; onCancel: () => void }) {
  return <section className="minimal-context-progress" aria-live="polite"><div><span className="eyebrow">MINIMAL CONTEXT JOB</span><strong>{phaseLabel(job.progress.phase)}</strong>{job.progress.bestRetainedSourceCount != null && <small>best verified: {job.progress.bestRetainedSourceCount} units</small>}</div><div><b>{job.progress.completedUnits.toLocaleString()} / {job.progress.totalUnits.toLocaleString()}</b><span>{Math.round(job.progress.percent)}%</span></div>{job.progress.certificateCandidateKind && <p>{CERTIFICATE_LABEL[job.progress.certificateCandidateKind]}</p>}{job.cancellable && <button type="button" onClick={onCancel}>CANCEL</button>}</section>;
}

export interface MinimalContextStudioProps { runId: string }

export function MinimalContextStudio({ runId }: MinimalContextStudioProps) {
  const [detail, setDetail] = useState<MinimalContextRunDetail | null>(null);
  const [result, setResult] = useState<MinimalContextResult | null>(null);
  const [job, setJob] = useState<MinimalContextJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [branchError, setBranchError] = useState<string | null>(null);
  const [childId, setChildId] = useState<string | null>(null);
  const [branching, setBranching] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true); setDetail(null); setResult(null); setJob(null); setError(null); setBranchError(null); setChildId(null);
    void loadMinimalContextRun(runId).then((nextDetail) => {
      if (active) { setDetail(nextDetail); setLoading(false); }
    });
    return () => { active = false; };
  }, [runId]);

  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled"].includes(job.state)) return;
    const timer = window.setTimeout(() => { void pollMinimalContextJob(runId, job.jobId).then(setJob).catch((caught: Error) => setError(caught.message)); }, 250);
    return () => window.clearTimeout(timer);
  }, [job, runId]);

  useEffect(() => {
    if (!job) return;
    if (job.state === "completed") {
      if (job.result) { setResult(job.result); setError(null); }
      else setError("The completed Minimal Context job did not include a search result.");
    } else if (job.state === "failed" && job.error) setError(job.error.message);
  }, [job]);

  async function start() {
    setError(null); setBranchError(null); setChildId(null); setResult(null);
    try { setJob(await startMinimalContextJob(runId, { preservation: { kind: "exact_recorded_output" }, universe: { max_units: 50 }, max_new_counterfactual_observations: 128, attempt_inclusion_check: true })); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Minimal Context could not start."); }
  }

  async function cancel() {
    if (!job) return;
    try { setJob(await cancelMinimalContextJob(runId, job.jobId)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Cancellation failed."); }
  }

  async function branch() {
    const best = result?.best;
    if (!best || !best.experiment_id || !best.arm_id || !best.observation_id
        || best.observation_status !== "exact_preserved" || !best.removed_source_ids.length) return;
    setBranching(true); setBranchError(null); setChildId(null);
    try {
      const created = record(await branchMinimalContextWinner(runId, {
        experiment_id: best.experiment_id, arm_id: best.arm_id, observation_id: best.observation_id,
      }));
      const createdId = typeof created.child_run_id === "string" ? created.child_run_id : null;
      if (!createdId) throw new Error("The branch response did not include a child run id.");
      setChildId(createdId);
    } catch (caught) { setBranchError(caught instanceof Error ? caught.message : "The reduced-context branch could not be created."); }
    finally { setBranching(false); }
  }

  const best = result?.best;
  const branchEligible = result?.status === "completed"
    && Boolean(best && best.experiment_id && best.arm_id && best.observation_id
      && best.observation_status === "exact_preserved" && best.removed_source_ids.length);

  return <div className="minimal-context-studio">
    <header className="minimal-context-header"><div><span className="eyebrow">MINIMAL CONTEXT</span><h1>Reduce context</h1><p>Find the smallest directly verified context set that preserves the recorded answer.</p></div></header>
    {job && <Progress job={job} onCancel={cancel} />}
    {error && <div className="minimal-context-error" role="alert"><strong>MINIMAL CONTEXT UNAVAILABLE</strong><span>{error}</span></div>}
    {!loading && result?.status === "unavailable" && <Unavailable result={result} />}
    {!loading && result?.status === "completed" && <><Hero result={result} /><div className="minimal-context-meta"><span>{result.universe.universe_id ?? "UNIVERSE UNAVAILABLE"}</span><span>{result.search_id}</span></div><SearchProof result={result} />{result.best && <ContextCollapse detail={detail} result={result} />}<div className="minimal-context-actions">{branchEligible && <button type="button" onClick={() => void branch()} disabled={branching}>{branching ? "BRANCHING…" : "BRANCH WITH THIS CONTEXT"}</button>}{childId && <p role="status">Child branch created — <a href={`#/runs/${encodeURIComponent(childId)}`}>OPEN {childId.slice(-8)}</a></p>}{branchError && <p className="minimal-context-error" role="alert">{branchError}</p>}</div></>}
    {!loading && !result && !job && !error && <section className="minimal-context-empty"><span className="eyebrow">NO VERIFIED RESULT</span><h2>Reduce this recorded context</h2><p>Check whether the recorded answer can be reproduced token-for-token.</p><button type="button" onClick={start}>RUN EXACT MINIMAL CONTEXT</button></section>}
  </div>;
}
