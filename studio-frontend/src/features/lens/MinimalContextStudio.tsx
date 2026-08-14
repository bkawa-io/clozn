import { useEffect, useMemo, useState } from "react";
import {
  cancelMinimalContextJob,
  branchFromMinimalContext,
  listMinimalContextResults,
  loadMinimalContextResult,
  loadMinimalContextRun,
  pollMinimalContextJob,
  startMinimalContextJob,
  type MinimalContextCertificate,
  type MinimalContextCriterion,
  type MinimalContextJob,
  type MinimalContextResult,
  type MinimalContextRunDetail,
  type MinimalContextSourceUnit,
  type MinimalContextSummary,
} from "../../data/minimalContext";
import "./MinimalContextStudio.css";

const CERTIFICATE_LABEL: Record<MinimalContextCertificate, string> = {
  exact_minimum: "EXACT MINIMUM",
  inclusion_minimum: "INCLUSION-MINIMAL",
  best_verified: "BEST VERIFIED",
};

const CERTIFICATE_EXPLANATION: Record<MinimalContextCertificate, string> = {
  exact_minimum: "All smaller-cardinality sets were directly ruled out.",
  inclusion_minimum: "No retained unit can be individually removed.",
  best_verified: "Smaller unmeasured sets may exist.",
};

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function certificateRank(value?: MinimalContextCertificate): number {
  return value === "exact_minimum" ? 0 : value === "inclusion_minimum" ? 1 : 2;
}

function phaseLabel(phase: string): string {
  if (phase === "planning_context") return "Planning context";
  if (phase === "checking_exact_eligibility") return "Checking exact eligibility";
  if (phase === "unchanged_control") return "Checking unchanged control";
  if (phase === "searching") return "Searching";
  if (phase === "verifying_candidate") return "Verifying candidate";
  const cardinality = phase.match(/^certifying_cardinality_(\d+)$/);
  if (cardinality) return `Certifying ${cardinality[1]}-source layer`;
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
          role: String(message.role ?? ""),
          unicode_range: [0, String(message.content ?? "").length],
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

function compatibleSummary(summary: MinimalContextSummary, criterion: MinimalContextCriterion) {
  return summary.preservation_kind === criterion && summary.status === "found";
}

function bestSummary(summaries: MinimalContextSummary[], criterion: MinimalContextCriterion) {
  return summaries
    .filter((summary) => compatibleSummary(summary, criterion))
    .sort((left, right) => (
      certificateRank(left.certificate_kind) - certificateRank(right.certificate_kind)
      || (left.retained_source_count ?? Number.MAX_SAFE_INTEGER) - (right.retained_source_count ?? Number.MAX_SAFE_INTEGER)
      || left.result_id.localeCompare(right.result_id)
    ))[0];
}

function Hero({ result }: { result: MinimalContextResult }) {
  const certificate = result.certificate?.kind;
  const sourceCount = result.source_universe.source_count;
  const retained = result.candidate?.retained_source_count;
  const exact = result.preservation.kind === "exact_recorded_output";
  return (
    <section className="minimal-context-hero" aria-label="Minimal Context result">
      <div className="minimal-context-hero-count">
        <strong>{numberText(sourceCount)}</strong><span>CONTEXT UNITS</span>
        <b aria-hidden="true">→</b>
        <strong>{numberText(retained)}</strong>
      </div>
      <div className="minimal-context-hero-proof">
        <span>{certificate ? CERTIFICATE_LABEL[certificate] : result.status.replaceAll("_", " ").toUpperCase()}</span>
        <p>
          {exact
            ? "recorded answer reproduced token-for-token"
            : `teacher-forced likelihood preserved within ${numberText(result.preservation.tolerance_nats, "the recorded tolerance")} nats`}
        </p>
        {certificate && <small>{CERTIFICATE_EXPLANATION[certificate]}</small>}
      </div>
    </section>
  );
}

function Coverage({ result }: { result: MinimalContextResult }) {
  const rows = result.coverage?.lower_cardinalities ?? [];
  const exact = result.preservation.kind === "exact_recorded_output";
  return (
    <section className="minimal-context-section" aria-labelledby="minimal-context-proof-title">
      <header><span className="eyebrow">PROOF COVERAGE</span><h2 id="minimal-context-proof-title">Minimality frontier</h2></header>
      <p className="minimal-context-muted">
        {exact ? "Retained source count against directly checked smaller candidates." : "Distance-to-baseline checks across the retained-count frontier."}
      </p>
      {exact && rows.length > 0 ? (
        <div className="minimal-context-coverage-table" role="table" aria-label="Proof coverage by retained source count">
          {rows.map((row) => {
            const remaining = Math.max(0, row.candidate_count - row.tested_count);
            return (
              <div className="minimal-context-coverage-row" role="row" key={row.retained_source_count}>
                <strong>{row.retained_source_count} {row.retained_source_count === 1 ? "source" : "sources"}</strong>
                <span>{row.complete ? `${row.tested_count.toLocaleString()} / ${row.candidate_count.toLocaleString()} ruled out` : `${row.tested_count.toLocaleString()} / ${row.candidate_count.toLocaleString()} tested`}</span>
                {!row.complete && <em>{remaining.toLocaleString()} remain unmeasured</em>}
              </div>
            );
          })}
          {result.candidate && <div className="minimal-context-coverage-row is-pass" role="row"><strong>{result.candidate.retained_source_count} sources</strong><span>PASS · preserving candidate</span></div>}
        </div>
      ) : (
        <div className="minimal-context-coverage-summary">
          <strong>{numberText(result.coverage?.smaller_tested_count)} tested</strong>
          <span>{numberText(result.coverage?.smaller_remaining_count)} remain unmeasured</span>
        </div>
      )}
      <p className="minimal-context-unmeasured">Unmeasured candidates are not counted as failed.</p>
    </section>
  );
}

function ContextCollapse({ detail, result, onError }: { detail: MinimalContextRunDetail | null; result: MinimalContextResult; onError: (message: string) => void }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [branching, setBranching] = useState(false);
  const catalog = useMemo(() => contextCatalog(detail), [detail]);
  const unitDerivation = new Map((detail?.context_units?.units ?? []).map((unit) => [unit.source_id, unit.derivation]));
  const retained = new Set(result.candidate?.retained_source_ids ?? []);
  const selected = selectedId ? catalog.get(selectedId) : undefined;
  const protectedIndices = new Set(detail?.context_units?.protected_message_indices ?? []);
  const protectedMessages = [...protectedIndices].map((index) => detail?.messages?.[index]).filter(Boolean);
  async function branch(action: "remove_and_branch" | "add_back_and_branch" | "branch_with_only") {
    if (!selectedId || branching) return;
    setBranching(true);
    try {
      const body = await branchFromMinimalContext(result.run_id, {
        result_id: result.result_id,
        action,
        source_ids: [selectedId],
      });
      const childId = typeof body.child_run_id === "string" ? body.child_run_id : "";
      if (!childId) throw new Error(typeof body.reason === "string" ? body.reason : "The branch did not produce a child Run.");
      window.location.hash = `#/compare/${encodeURIComponent(result.run_id)}/${encodeURIComponent(childId)}`;
    } catch (caught) {
      onError(caught instanceof Error ? caught.message : "The source branch could not be created.");
    } finally {
      setBranching(false);
    }
  }
  return (
    <section className="minimal-context-section" aria-labelledby="minimal-context-collapse-title">
      <header><span className="eyebrow">CONTEXT COLLAPSE</span><h2 id="minimal-context-collapse-title">Retained versus omitted</h2></header>
      <div className="minimal-context-unit-list">
        {result.source_universe.source_ids.map((sourceId, index) => {
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
      {selected && <aside className="minimal-context-source-detail" aria-label="Selected source detail"><header><span>{selected.role.toUpperCase()} · MESSAGE {selected.message_index + 1}</span><code>{selected.source_id}</code></header><blockquote>{unitText(detail, selected)}</blockquote><dl><div><dt>Range</dt><dd>{selected.unicode_range[0]}–{selected.unicode_range[1]} Unicode</dd></div><div><dt>Provenance</dt><dd>{sourceKind({ ...selected, derivation: unitDerivation.get(selected.source_id) ?? selected.derivation })}</dd></div><div><dt>Intervention</dt><dd>{retained.has(selected.source_id) ? "Retained in preserving candidate" : "Omitted by candidate intervention"}</dd></div>{selected.source_label && <div><dt>Caller label</dt><dd>{selected.source_label}</dd></div>}</dl><div className="minimal-context-branch-actions"><button type="button" disabled={branching} onClick={() => void branch(retained.has(selected.source_id) ? "remove_and_branch" : "add_back_and_branch")}>{branching ? "BRANCHING…" : retained.has(selected.source_id) ? "REMOVE + BRANCH" : "ADD BACK + BRANCH"}</button><button type="button" disabled={branching} onClick={() => void branch("branch_with_only")}>BRANCH WITH ONLY THIS SET</button></div></aside>}
    </section>
  );
}

function Progress({ job, onCancel }: { job: MinimalContextJob; onCancel: () => void }) {
  return <section className="minimal-context-progress" aria-live="polite"><div><span className="eyebrow">MINIMAL CONTEXT JOB</span><strong>{phaseLabel(job.progress.phase)}</strong>{job.progress.bestRetainedSourceCount != null && <small>best verified: {job.progress.bestRetainedSourceCount} units</small>}</div><div><b>{job.progress.completedUnits.toLocaleString()} / {job.progress.totalUnits.toLocaleString()}</b><span>{Math.round(job.progress.percent)}%</span></div>{job.progress.certificateCandidateKind && <p>{CERTIFICATE_LABEL[job.progress.certificateCandidateKind]}</p>}{job.cancellable && <button type="button" onClick={onCancel}>CANCEL</button>}</section>;
}

export interface MinimalContextStudioProps { runId: string }

export function MinimalContextStudio({ runId }: MinimalContextStudioProps) {
  const [detail, setDetail] = useState<MinimalContextRunDetail | null>(null);
  const [summaries, setSummaries] = useState<MinimalContextSummary[]>([]);
  const [criterion, setCriterion] = useState<MinimalContextCriterion>("exact_recorded_output");
  const [selectedResultId, setSelectedResultId] = useState("");
  const [result, setResult] = useState<MinimalContextResult | null>(null);
  const [job, setJob] = useState<MinimalContextJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    const [nextDetail, nextSummaries] = await Promise.all([loadMinimalContextRun(runId), listMinimalContextResults(runId)]);
    setDetail(nextDetail); setSummaries(nextSummaries); setLoading(false);
    const compatible = bestSummary(nextSummaries, criterion);
    if (compatible) { setSelectedResultId(compatible.result_id); setResult(await loadMinimalContextResult(runId, compatible.result_id)); }
  }
  useEffect(() => { setLoading(true); void refresh(); }, [runId]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const compatible = bestSummary(summaries, criterion);
    if (!compatible) { setSelectedResultId(""); setResult(null); return; }
    setSelectedResultId(compatible.result_id); void loadMinimalContextResult(runId, compatible.result_id).then(setResult);
  }, [criterion, runId, summaries]);
  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled"].includes(job.state)) return;
    const timer = window.setTimeout(() => { void pollMinimalContextJob(runId, job.jobId).then(setJob).catch((caught: Error) => setError(caught.message)); }, 250);
    return () => window.clearTimeout(timer);
  }, [job, runId]);
  useEffect(() => { if (job?.state === "completed") void refresh(); if (job?.state === "failed" && job.error) setError(job.error.message); }, [job?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setError(null); setResult(null);
    try { setJob(await startMinimalContextJob(runId, { preservation: { kind: criterion }, universe: { max_units: 50 }, search_probe_budget: 128, certification_probe_budget: 2000, search_seed: 0 })); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Minimal Context could not start."); }
  }
  async function cancel() { if (!job) return; try { setJob(await cancelMinimalContextJob(runId, job.jobId)); } catch (caught) { setError(caught instanceof Error ? caught.message : "Cancellation failed."); } }
  const selectedSummary = summaries.find((summary) => summary.result_id === selectedResultId);
  return <div className="minimal-context-studio">
    <header className="minimal-context-header"><div><span className="eyebrow">MINIMAL CONTEXT</span><h1>Reduce context</h1><p>Find the smallest directly verified context set under an explicit preservation criterion.</p></div><div className="minimal-context-criteria" role="group" aria-label="Preservation criterion"><button type="button" className={criterion === "exact_recorded_output" ? "is-active" : ""} onClick={() => setCriterion("exact_recorded_output")}>Exact recorded output</button><button type="button" className={criterion === "teacher_forced_likelihood" ? "is-active" : ""} onClick={() => setCriterion("teacher_forced_likelihood")}>Teacher-forced likelihood</button></div></header>
    {job && <Progress job={job} onCancel={cancel} />}
    {error && <div className="minimal-context-error" role="alert"><strong>MINIMAL CONTEXT UNAVAILABLE</strong><span>{error}</span>{criterion === "exact_recorded_output" && <small>Exact mode was not silently replaced. Choose Teacher-forced likelihood to run a separate criterion.</small>}</div>}
    {!loading && result ? <><Hero result={result} /><div className="minimal-context-meta"><span>{selectedSummary?.universe_id ?? result.source_universe.search_universe_id ?? "UNIVERSE UNAVAILABLE"}</span><span>{result.result_id}</span></div><Coverage result={result} /><ContextCollapse detail={detail} result={result} onError={setError} /></> : !job && <section className="minimal-context-empty"><span className="eyebrow">NO VERIFIED RESULT</span><h2>Reduce this recorded context</h2><p>{criterion === "exact_recorded_output" ? "Check whether the recorded answer can be reproduced token-for-token." : "Check whether the recorded continuation remains within the chosen likelihood tolerance."}</p><button type="button" onClick={start}>RUN {criterion === "exact_recorded_output" ? "EXACT" : "LIKELIHOOD"} MINIMAL CONTEXT</button></section>}
    {summaries.length > 0 && <label className="minimal-context-history"><span>RESULT HISTORY</span><select value={selectedResultId} onChange={(event) => { setSelectedResultId(event.target.value); void loadMinimalContextResult(runId, event.target.value).then(setResult); }}>{summaries.map((summary) => <option key={summary.result_id} value={summary.result_id}>{summary.preservation_kind === "exact_recorded_output" ? "Exact" : "Likelihood"} · {summary.certificate_kind ? CERTIFICATE_LABEL[summary.certificate_kind] : summary.status} · {summary.retained_source_count ?? "—"} retained</option>)}</select></label>}
  </div>;
}
