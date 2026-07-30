import { useEffect, useRef, useState } from "react";
import {
  cancelCorrectivePreview,
  confirmCorrectivePreview,
  keepCorrectiveResult,
  loadCorrectiveActions,
  previewCorrectiveAction,
  undoCorrectiveKeep,
  type CorrectivePreviewReceipt,
  type CorrectiveRegistry,
  type CorrectiveResult,
  type CorrectiveScope,
  type CorrectiveScopeEligibility,
} from "../../data/correctiveFlow";
import {
  loadDiagnosisRepair,
  type RepairDiagnosis,
  type RepairEvidence,
  type RepairFinding,
  type RepairFindingStatus,
  type RepairNarrativeEvidenceRef,
} from "../../data/diagnosisRepair";

/**
 * D5 -- the guided corrective-retry panel. Renders D1/D2's rule findings + plain-language narrative
 * (`data/diagnosisRepair.ts`) alongside D3's real preview -> confirm -> keep corrective-retry mechanics
 * (`data/correctiveFlow.ts`, the SAME client `features/behavior/Behavior.tsx`'s "Fix this answer" module
 * already uses -- moved to `src/data/` rather than duplicated; see that module's own doc comment).
 *
 * THE GAP THIS PANEL DOES NOT PAPER OVER
 * ---------------------------------------
 * Each finding's `suggestedActions[].kind` (resend_context, clarify_output_format, ...) is D1's OWN
 * provisional vocabulary -- `clozn.diagnosis-findings.v1`'s schema says outright that it "anticipates"
 * D3's action-kind vocabulary but that D3 "does not exist as a registered schema yet" and "callers must
 * not assume D3 will keep them verbatim." Checked against `clozn/behavior/registry.py`: D3's six real
 * action ids (less-verbose, more-concrete, use-context, ask-before-guessing, preserve-formatting,
 * stop-repeating) share no members with D1's kind vocabulary. There is no backend bridge from one to the
 * other, so this panel never claims one -- a finding's suggested direction renders as description text
 * only, never as a button that silently picks a corrective-flow action_id on the finding's behalf. The
 * six real, executable retries render in their own "Corrective retries" section below, applicable to the
 * whole run rather than claimed as any one finding's own fix.
 *
 * Rendering this panel fires two GETs only (diagnosis-findings, corrective-actions) -- no action ever
 * runs from a mount or a run-selection change; every POST below traces to an explicit click.
 */

export interface DiagnosisRepairProps {
  runId: string;
}

type FindingsResource =
  | { status: "idle" | "loading" }
  | { status: "failed"; message: string }
  | { status: "ready"; value: RepairDiagnosis };

type RegistryResource =
  | { status: "idle" | "loading" }
  | { status: "failed"; message: string }
  | { status: "ready"; value: CorrectiveRegistry };

/** A NEW closed union for the repair flow's own lifecycle -- deliberately never flattened into
 * `RepairFindingStatus` (D1/D2's evidence vocabulary) or `ActionState["phase"]` (the token-workbench
 * action tray's own, unrelated, state machine). Each value drives materially different button
 * enablement, so every reader of this file gets a `never`-exhaustiveness guard, not a string compare. */
export type RepairFlowPhase =
  | "idle"
  | "previewing"
  | "preview_ready"
  | "confirming"
  | "result_ready"
  | "keeping"
  | "kept";

export function flowPhaseLabel(phase: RepairFlowPhase): string {
  switch (phase) {
    case "idle": return "SELECT A CORRECTIVE RETRY";
    case "previewing": return "PREPARING PREVIEW";
    case "preview_ready": return "PREVIEW READY -- CONFIRM TO RUN A MATCHED RETRY";
    case "confirming": return "RUNNING MATCHED RETRY";
    case "result_ready": return "RETRY COMPLETE";
    case "keeping": return "KEEPING RESULT";
    case "kept": return "KEPT";
    default: {
      const exhaustive: never = phase;
      return exhaustive;
    }
  }
}

/** "once" selects the corrected child as THIS run's own revision -- a branch tied to one run, nothing
 * else ever changes. "session"/"profile" mutate a standing preference that shapes every future run in
 * that scope until undone. The UI must never blur these -- see the module docstring on D5's own
 * "branching retries vs mutable preferences visually distinct" requirement. */
export function keepScopeKind(scope: CorrectiveScope): "branch" | "preference" {
  switch (scope) {
    case "once": return "branch";
    case "session": return "preference";
    case "profile": return "preference";
    default: {
      const exhaustive: never = scope;
      return exhaustive;
    }
  }
}

export function findingStatusMeta(status: RepairFindingStatus): { label: string; className: string } {
  switch (status) {
    case "finding":
      return { label: "FINDING", className: "is-finding" };
    case "not_observed":
      return { label: "NOT OBSERVED -- checked, nothing found", className: "is-not-observed" };
    case "unavailable":
      return { label: "UNAVAILABLE -- could not be checked", className: "is-unavailable" };
    case "pending":
      return { label: "PENDING -- never measured", className: "is-pending" };
    case "suppressed":
      return { label: "SUPPRESSED -- excluded from this evaluation", className: "is-suppressed" };
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "the request failed";
}

function idempotencyKey(prefix: string): string {
  const suffix = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}:${suffix}`;
}

function evidenceLabel(item: RepairEvidence): string {
  return item.kind === "field" ? `${item.path} = ${JSON.stringify(item.value)}` : `span ${item.addressId.slice(-8)}`;
}

function evidenceHref(runId: string, item: RepairEvidence): string | undefined {
  return item.kind === "text_span"
    ? `#/runs/${encodeURIComponent(runId)}/span-addresses#${item.addressId}`
    : undefined;
}

/** `#diagnosis-finding-<ruleId>` is deliberately never an `href` -- this app's `location.hash` IS its
 * router (src/panels/registry.ts's `resolveRoute`); a bare in-page fragment with no leading `/panel`
 * segment matches no panel's `match()` and falls through to `resolveRoute`'s own first-panel fallback,
 * which would navigate away from Lens entirely instead of scrolling. `scrollIntoView` on the finding's
 * own `id` gets the same "jump to it" result without touching the route. */
function scrollToFinding(ruleId: string) {
  document.getElementById(`diagnosis-finding-${ruleId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function NarrativeEvidenceRefTag({ runId, item }: { runId: string; item: RepairNarrativeEvidenceRef }) {
  switch (item.kind) {
    case "finding":
      return (
        <button type="button" className="diagnosis-repair-jump" onClick={() => scrollToFinding(item.ruleId)}>
          {item.ruleId}
        </button>
      );
    case "text_span":
      return (
        <a href={`#/runs/${encodeURIComponent(runId)}/span-addresses#${item.addressId}`}>
          span {item.addressId.slice(-8)}
        </a>
      );
    case "diff_field":
      return <span>{item.dimension}</span>;
    default: {
      const exhaustive: never = item;
      return exhaustive;
    }
  }
}

function KeepScopeButton({
  scope, busy, onKeep,
}: {
  scope: CorrectiveScopeEligibility;
  busy: boolean;
  onKeep: (scope: CorrectiveScope) => void;
}) {
  const branch = keepScopeKind(scope.scope) === "branch";
  return (
    <button
      type="button"
      disabled={!scope.available || busy}
      title={scope.unavailability_reason}
      onClick={() => onKeep(scope.scope)}
    >
      <strong>{branch ? "USE THIS CORRECTION FOR THIS RUN" : scope.scope.toUpperCase()}</strong>
      <span>
        {scope.available
          ? scope.note ?? (branch
            ? "selects the corrected child as this run's own revision"
            : `${(scope.before ?? []).length} → ${(scope.after ?? []).length} active actions`)
          : scope.unavailability_reason}
      </span>
    </button>
  );
}

function FindingRow({ runId, finding }: { runId: string; finding: RepairFinding }) {
  const meta = findingStatusMeta(finding.status);
  return (
    <article
      className={`diagnosis-repair-finding ${meta.className}`}
      data-finding-status={finding.status}
      id={`diagnosis-finding-${finding.ruleId}`}
    >
      <header>
        <div>
          <b>{finding.ruleId}</b>
          <span>{finding.ruleName.replaceAll("_", " ")}</span>
        </div>
        <span className={`diagnosis-repair-status ${meta.className}`}>{meta.label}</span>
      </header>
      {finding.status === "finding" && (
        <p className="diagnosis-repair-finding-severity">
          <b>{finding.severity.toUpperCase()}</b> SEVERITY · <b>{finding.confidence.toUpperCase().replaceAll("_", " ")}</b> CONFIDENCE
        </p>
      )}
      <p className="diagnosis-repair-finding-summary">{finding.summary}</p>
      {finding.limitations.length > 0 && (
        <ul className="diagnosis-repair-finding-limitations">
          {finding.limitations.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      )}
      {finding.evidence.length > 0 && (
        <ul className="diagnosis-repair-finding-evidence" aria-label={`Evidence for ${finding.ruleId}`}>
          {finding.evidence.map((item, index) => {
            const href = evidenceHref(runId, item);
            return <li key={index}>{href ? <a href={href}>{evidenceLabel(item)}</a> : evidenceLabel(item)}</li>;
          })}
        </ul>
      )}
      {finding.status === "finding" && finding.suggestedActions.length > 0 && (
        <div className="diagnosis-repair-suggested" aria-label={`Suggested direction for ${finding.ruleId}`}>
          <p className="diagnosis-repair-suggested-caption">
            SUGGESTED DIRECTION -- not yet an automated retry; see Corrective retries below for what can
            run directly.
          </p>
          <ul className="diagnosis-repair-suggested-actions">
            {finding.suggestedActions.map((action, index) => (
              <li key={index}>
                <b>{action.kind.replaceAll("_", " ").toUpperCase()}</b>
                <span>{action.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}

export function DiagnosisRepair({ runId }: DiagnosisRepairProps) {
  const [findings, setFindings] = useState<FindingsResource>({ status: "idle" });
  const [registry, setRegistry] = useState<RegistryResource>({ status: "idle" });

  const [selectedActionId, setSelectedActionId] = useState("");
  const [preview, setPreview] = useState<CorrectivePreviewReceipt>();
  const [result, setResult] = useState<CorrectiveResult>();
  const [flowPhase, setFlowPhase] = useState<RepairFlowPhase>("idle");
  const [flowError, setFlowError] = useState("");

  // One monotonic id per RUN SELECTION -- the same guard scope.tsx's own selectRun uses. Every async
  // response below (the two GETs on mount/run-change, and every POST an explicit click starts) captures
  // this value and checks it before writing state, so a response for a run the panel has since navigated
  // away from can never land on top of the run now showing.
  const requestIdRef = useRef(0);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    setSelectedActionId("");
    setPreview(undefined);
    setResult(undefined);
    setFlowPhase("idle");
    setFlowError("");
    if (!runId) {
      setFindings({ status: "idle" });
      setRegistry({ status: "idle" });
      return;
    }
    setFindings({ status: "loading" });
    setRegistry({ status: "loading" });
    const controller = new AbortController();
    void loadDiagnosisRepair(runId, { signal: controller.signal }).then((value) => {
      if (requestIdRef.current !== requestId) return;
      setFindings({ status: "ready", value });
    }).catch((error) => {
      if (requestIdRef.current !== requestId) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setFindings({ status: "failed", message: describeError(error) });
    });
    void loadCorrectiveActions(runId, controller.signal).then((value) => {
      if (requestIdRef.current !== requestId) return;
      setRegistry({ status: "ready", value });
      setSelectedActionId((current) => current || value.actions[0]?.id || "");
    }).catch((error) => {
      if (requestIdRef.current !== requestId) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setRegistry({ status: "failed", message: describeError(error) });
    });
    return () => controller.abort();
  }, [runId]);

  async function startPreview(actionId: string) {
    if (!runId) return;
    const generation = requestIdRef.current;
    setSelectedActionId(actionId);
    setPreview(undefined);
    setResult(undefined);
    setFlowError("");
    setFlowPhase("previewing");
    try {
      const next = await previewCorrectiveAction(runId, actionId, "prompt_policy");
      if (requestIdRef.current !== generation) return;
      setPreview(next);
      setFlowPhase("preview_ready");
    } catch (error) {
      if (requestIdRef.current !== generation) return;
      setFlowPhase("idle");
      setFlowError(`${describeError(error)} -- no run was created; the original run is unchanged.`);
    }
  }

  async function cancelActivePreview() {
    if (!preview) return;
    const generation = requestIdRef.current;
    try {
      await cancelCorrectivePreview(preview.preview_id);
    } catch {
      // Best effort -- an unconfirmed preview also simply expires server-side.
    }
    if (requestIdRef.current !== generation) return;
    setPreview(undefined);
    setFlowPhase("idle");
    setFlowError("");
  }

  async function confirmActivePreview() {
    if (!preview) return;
    const generation = requestIdRef.current;
    setFlowPhase("confirming");
    setFlowError("");
    try {
      const next = await confirmCorrectivePreview(preview.preview_id, idempotencyKey("diagnosis-repair-confirm"));
      if (requestIdRef.current !== generation) return;
      setResult(next);
      setFlowPhase("result_ready");
      if (next.outcome.status !== "succeeded") {
        setFlowError(
          `${next.outcome.note ?? `the retry did not complete (${next.outcome.status.replaceAll("_", " ")})`} -- the original run is unchanged.`,
        );
      }
    } catch (error) {
      // The preview is untouched server-side by a failed confirm attempt (corrective_flow only marks it
      // "confirming" mid-call; a thrown request here means that call never returned a result at all) --
      // stay on preview_ready so the same preview can be retried, never silently drop it.
      if (requestIdRef.current !== generation) return;
      setFlowPhase("preview_ready");
      setFlowError(`${describeError(error)} -- the original run is unchanged.`);
    }
  }

  async function keepActiveResult(scope: CorrectiveScope) {
    if (!result) return;
    const eligibility = result.scope_eligibility.find((item) => item.scope === scope);
    if (!eligibility?.available || !eligibility.prior_hash) return;
    const generation = requestIdRef.current;
    setFlowPhase("keeping");
    setFlowError("");
    try {
      const kept = await keepCorrectiveResult(
        result.result_id, scope, eligibility.prior_hash, idempotencyKey("diagnosis-repair-keep"),
      );
      if (requestIdRef.current !== generation) return;
      setResult(kept);
      setFlowPhase("kept");
    } catch (error) {
      if (requestIdRef.current !== generation) return;
      setFlowPhase("result_ready");
      setFlowError(`${describeError(error)} -- nothing was kept; the original run is unchanged.`);
    }
  }

  async function undoActiveKeep() {
    if (!result?.transaction?.id) return;
    const generation = requestIdRef.current;
    try {
      await undoCorrectiveKeep(result.transaction.id);
      if (requestIdRef.current !== generation) return;
      setResult({ ...result, transaction: { ...result.transaction, undone_ts: Date.now() / 1000 } });
    } catch (error) {
      if (requestIdRef.current !== generation) return;
      setFlowError(describeError(error));
    }
  }

  function discardResult() {
    setPreview(undefined);
    setResult(undefined);
    setFlowPhase("idle");
    setFlowError("");
  }

  const doc = findings.status === "ready" ? findings.value : undefined;
  const registryDoc = registry.status === "ready" ? registry.value : undefined;
  const busy = flowPhase === "previewing" || flowPhase === "confirming" || flowPhase === "keeping";

  return (
    <section className="diagnosis-repair" aria-labelledby="diagnosis-repair-title">
      <header className="diagnosis-repair-head">
        <div>
          <span className="eyebrow">DIAGNOSIS &amp; REPAIR</span>
          <h3 id="diagnosis-repair-title">Why, and what to try</h3>
        </div>
        {doc && <span className="diagnosis-repair-headline">{doc.narrative.headline}</span>}
      </header>

      {findings.status === "idle" || findings.status === "loading" ? (
        <div className="diagnosis-repair-empty">LOADING DIAGNOSIS</div>
      ) : findings.status === "failed" ? (
        <div className="diagnosis-repair-notice is-failed" role="alert">
          <strong>DIAGNOSIS REQUEST FAILED</strong>
          <span>{findings.message}</span>
        </div>
      ) : (
        <>
          {doc!.findings.redacted && (
            <p className="diagnosis-repair-redacted">
              This run&apos;s text is redacted -- text-based rules could not run.
            </p>
          )}

          <div className="diagnosis-repair-registers">
            <section className="diagnosis-repair-register is-measured" aria-label="Measured effects">
              <header><span>MEASURED EFFECTS</span><b>RANKED -- EACH NAMES A RULE</b></header>
              {doc!.narrative.measuredEffects.length ? (
                <ol>
                  {doc!.narrative.measuredEffects.map((entry) => (
                    <li key={entry.ruleId}>
                      <span className="diagnosis-repair-rank">#{entry.rank}</span>
                      <span>{entry.text}</span>
                      <button type="button" className="diagnosis-repair-jump" onClick={() => scrollToFinding(entry.ruleId)}>
                        {entry.ruleId}
                      </button>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="diagnosis-repair-register-empty">No ranked measured effects for this run.</p>
              )}
            </section>

            <section className="diagnosis-repair-register is-observed" aria-label="Observed changes">
              <header><span>OBSERVED CHANGES</span><b>NEVER RANKED</b></header>
              {!doc!.narrative.comparisonAvailable ? (
                <p className="diagnosis-repair-register-empty">No comparison run was supplied for this evaluation.</p>
              ) : doc!.narrative.observedChanges.length ? (
                <ul>
                  {doc!.narrative.observedChanges.map((entry, index) => {
                    const ref = entry.evidence[0];
                    return (
                      <li key={index}>
                        {entry.text}
                        {ref && (
                          <>
                            {" ["}
                            <NarrativeEvidenceRefTag runId={runId} item={ref} />
                            {"]"}
                          </>
                        )}
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="diagnosis-repair-register-empty">No structural differences against the comparison run.</p>
              )}
            </section>

            <section className="diagnosis-repair-register is-plausible" aria-label="Plausible but unproven">
              <header><span>PLAUSIBLE -- UNPROVEN</span><b>NEVER RANKED ABOVE MEASURED EFFECTS</b></header>
              {doc!.narrative.plausibleButUnproven.length ? (
                <ul>
                  {doc!.narrative.plausibleButUnproven.map((entry, index) => (
                    <li key={index}>
                      <p>{entry.text}</p>
                      <small>{entry.note}</small>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="diagnosis-repair-register-empty">No unproven hypotheses for this run.</p>
              )}
            </section>
          </div>

          <div className="diagnosis-repair-findings" aria-label="Rule findings">
            {doc!.findings.findings.map((finding) => (
              <FindingRow runId={runId} finding={finding} key={finding.ruleId} />
            ))}
          </div>
        </>
      )}

      <section className="diagnosis-repair-retries" aria-label="Corrective retries">
        <header className="section-title">
          <h3>Corrective retries</h3>
          <span>{registryDoc?.actions.length ?? 0}</span>
        </header>
        <p className="diagnosis-repair-retries-note">
          Bounded, reversible retries this run can try directly -- not claimed by any specific finding
          above; pick whichever reads closest to a finding&apos;s own suggested direction.
        </p>

        {registry.status === "idle" || registry.status === "loading" ? (
          <div className="diagnosis-repair-empty">LOADING CORRECTIVE ACTIONS</div>
        ) : registry.status === "failed" ? (
          <div className="diagnosis-repair-notice is-failed" role="alert">
            <strong>CORRECTIVE ACTIONS UNAVAILABLE</strong>
            <span>{registry.message}</span>
          </div>
        ) : (
          <>
            <div className="diagnosis-repair-action-list" aria-label="Available corrective actions">
              {registryDoc!.actions.map((action) => (
                <button
                  type="button"
                  className={selectedActionId === action.id ? "is-selected" : ""}
                  aria-pressed={selectedActionId === action.id}
                  disabled={busy}
                  onClick={() => void startPreview(action.id)}
                  key={action.id}
                >
                  <strong>{action.label}</strong>
                  <span>{action.description}</span>
                </button>
              ))}
            </div>

            <p className="diagnosis-repair-flow-status" role="status">{flowPhaseLabel(flowPhase)}</p>
            {flowError && <p className="diagnosis-repair-flow-error" role="alert">{flowError}</p>}

            {preview && !result && (
              <section className="diagnosis-repair-preview">
                <header><span>PREVIEW</span><b>{preview.preview_id.slice(-8)}</b></header>
                <dl>
                  <div><dt>WILL INJECT</dt><dd>{preview.action.description}</dd></div>
                  <div><dt>REQUESTED BACKEND</dt><dd>{preview.execution.requested_backend}</dd></div>
                  <div><dt>WILL EXECUTE</dt><dd>{preview.execution.expected_executed_backend}</dd></div>
                  <div><dt>FALLBACK</dt><dd>{preview.execution.expected_fallback ? "YES" : "NO"}</dd></div>
                </dl>
                {preview.comparison_contract && (
                  <dl className="diagnosis-repair-preview-contract">
                    <div><dt>BASELINE ARM</dt><dd>{preview.comparison_contract.baseline}</dd></div>
                    <div><dt>CORRECTED ARM</dt><dd>{preview.comparison_contract.corrected}</dd></div>
                    <div><dt>STORED ORIGINAL</dt><dd>{preview.comparison_contract.stored_original}</dd></div>
                  </dl>
                )}
                {preview.execution.unavailability_reason && <p>{preview.execution.unavailability_reason}</p>}
                <div className="diagnosis-repair-preview-buttons">
                  <button type="button" disabled={flowPhase === "confirming"} onClick={() => void cancelActivePreview()}>
                    CANCEL
                  </button>
                  <button
                    type="button"
                    className="is-primary"
                    disabled={flowPhase === "confirming"}
                    onClick={() => void confirmActivePreview()}
                  >
                    {flowPhase === "confirming" ? "RUNNING…" : "CONFIRM -- RUN MATCHED RETRY"}
                  </button>
                </div>
              </section>
            )}

            {result && (
              <section className="diagnosis-repair-result">
                <header>
                  <div><span>RESULT</span><strong>{result.action.label}</strong></div>
                  <b className={`diagnosis-repair-outcome is-${result.outcome.status}`}>
                    {result.outcome.status.toUpperCase().replaceAll("_", " ")}
                  </b>
                </header>
                <div className="diagnosis-repair-result-outputs">
                  <article>
                    <header><span>STORED ORIGINAL</span></header>
                    <p>{result.comparison.stored_original_reply || "No stored reply."}</p>
                  </article>
                  <article>
                    <header>
                      <span>MATCHED BASELINE</span>
                      <b>{result.children.baseline.run_id?.slice(-8) ?? result.children.baseline.status}</b>
                    </header>
                    <p>{result.comparison.baseline_reply || "No baseline output."}</p>
                  </article>
                  <article className="is-corrected">
                    <header>
                      <span>CORRECTED</span>
                      <b>{result.children.corrected.run_id?.slice(-8) ?? result.children.corrected.status}</b>
                    </header>
                    <p>{result.comparison.corrected_reply || "No corrected output."}</p>
                  </article>
                </div>
                <p className="diagnosis-repair-result-note">{result.comparison.note}</p>

                {result.outcome.status === "succeeded" ? (
                  <>
                    {result.children.corrected.run_id && (
                      <a
                        className="diagnosis-repair-compare-link"
                        href={`#/compare/${encodeURIComponent(result.parent_run_id)}/${encodeURIComponent(result.children.corrected.run_id)}`}
                      >
                        OPEN PAIRED COMPARISON -- PARENT / CORRECTED
                      </a>
                    )}

                    {!result.transaction ? (
                      <div className="diagnosis-repair-keep">
                        <div className="diagnosis-repair-keep-group is-branch">
                          <header><span>PER-RUN -- BRANCHING RETRY</span><b>NEVER CHANGES FUTURE RUNS</b></header>
                          {result.scope_eligibility.filter((s) => keepScopeKind(s.scope) === "branch").map((s) => (
                            <KeepScopeButton scope={s} busy={busy} onKeep={(scope) => void keepActiveResult(scope)} key={s.scope} />
                          ))}
                        </div>
                        <div className="diagnosis-repair-keep-group is-preference">
                          <header><span>PREFERENCE CHANGE</span><b>APPLIES TO FUTURE RUNS UNTIL UNDONE</b></header>
                          {result.scope_eligibility.filter((s) => keepScopeKind(s.scope) === "preference").map((s) => (
                            <KeepScopeButton scope={s} busy={busy} onKeep={(scope) => void keepActiveResult(scope)} key={s.scope} />
                          ))}
                        </div>
                        <button type="button" className="diagnosis-repair-discard" onClick={discardResult}>
                          DISCARD -- KEEP NOTHING
                        </button>
                      </div>
                    ) : (
                      <div className="diagnosis-repair-kept">
                        <span>
                          KEPT -- {keepScopeKind(result.transaction.scope) === "branch" ? "PER-RUN" : "PREFERENCE"}
                          {" · "}{result.transaction.scope.toUpperCase()}
                        </span>
                        <button
                          type="button"
                          disabled={Boolean(result.transaction.undone_ts) || busy}
                          onClick={() => void undoActiveKeep()}
                        >
                          {result.transaction.undone_ts ? "UNDONE" : "UNDO"}
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="diagnosis-repair-result-failed" role="alert">
                    The retry did not succeed -- the original run is unchanged.
                  </p>
                )}
              </section>
            )}
          </>
        )}
      </section>
    </section>
  );
}
