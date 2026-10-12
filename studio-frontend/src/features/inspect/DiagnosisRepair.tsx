import { useEffect, useRef, useState } from "react";
import { routeHref, type InspectRoute } from "../../app/router";
import { describeError, isContractError } from "../../data/client";
import {
  diagnosisActionsApi,
  type CorrectiveActionId,
  type CorrectiveChild,
  type CorrectivePreview,
  type CorrectiveRegistry,
  type CorrectiveResult,
  type DiagnosisDocument,
  type DiagnosisEvidence,
  type DiagnosisFinding,
  type DiagnosisFindingStatus,
  type DiagnosisNarrativeEvidence,
} from "../../data/diagnosisActions";
import "./diagnosisRepair.css";

type Resource<T> =
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable" | "malformed"; error: string };

type FlowPhase =
  | "idle"
  | "previewing"
  | "preview_ready"
  | "confirming"
  | "cancelling"
  | "result_ready"
  | "keeping"
  | "kept"
  | "undone"
  | "undoing";

function shortId(value: string | undefined): string {
  if (!value) return "—";
  return value.length > 19 ? `${value.slice(0, 8)}…${value.slice(-7)}` : value;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function failedResource<T>(error: unknown): Resource<T> {
  return {
    phase: isContractError(error) ? "malformed" : "unavailable",
    error: describeError(error),
  };
}

function idempotencyKey(prefix: string): string {
  const suffix = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}:${suffix}`;
}

function flowLabel(phase: FlowPhase): string {
  switch (phase) {
    case "idle": return "Select a corrective retry";
    case "previewing": return "Preparing preview";
    case "preview_ready": return "Preview ready — confirm to run a matched retry";
    case "confirming": return "Running matched retry";
    case "cancelling": return "Requesting cancellation";
    case "result_ready": return "Retry complete";
    case "keeping": return "Keeping corrected child";
    case "kept": return "Corrected child kept for this Run";
    case "undone": return "Correction undone; recorded Run restored";
    case "undoing": return "Undoing correction";
  }
}

function findingStatus(status: DiagnosisFindingStatus): { label: string; className: string } {
  switch (status) {
    case "finding": return { label: "Finding", className: "is-finding" };
    case "not_observed": return { label: "Not observed — checked, nothing found", className: "is-not-observed" };
    case "unavailable": return { label: "Unavailable — could not be checked", className: "is-unavailable" };
    case "pending": return { label: "Pending — never measured", className: "is-pending" };
    case "suppressed": return { label: "Suppressed — excluded from this evaluation", className: "is-suppressed" };
  }
}

function evidenceText(evidence: DiagnosisEvidence): string {
  return evidence.kind === "field"
    ? `${evidence.path} = ${JSON.stringify(evidence.value)}`
    : `span ${shortId(evidence.addressId)}`;
}

function sortedEvidenceValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortedEvidenceValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortedEvidenceValue(item)]));
  }
  return value;
}

function evidenceValue(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === undefined) return "not recorded";
  return JSON.stringify(sortedEvidenceValue(value));
}

function EvidenceFields({ label, fields }: { label: string; fields: Readonly<Record<string, unknown>> | null }) {
  const entries = fields ? Object.entries(fields).sort(([left], [right]) => left.localeCompare(right)) : [];
  return (
    <dl className="corrective-evidence-fields" aria-label={label}>
      {entries.length > 0 ? entries.map(([key, value]) => (
        <div key={key}><dt>{key}</dt><dd><code>{evidenceValue(value)}</code></dd></div>
      )) : <div><dt>Status</dt><dd><code>Not recorded</code></dd></div>}
    </dl>
  );
}

function scrollToFinding(ruleId: string): void {
  document.getElementById(`diagnosis-finding-${ruleId}`)?.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

function NarrativeEvidence({ runId, evidence }: { runId: string; evidence: DiagnosisNarrativeEvidence }) {
  switch (evidence.kind) {
    case "finding":
      return <button type="button" onClick={() => scrollToFinding(evidence.ruleId)}>{evidence.ruleId}</button>;
    case "diff_field":
      return <span>{evidence.dimension}</span>;
    case "text_span":
      return (
        <a href={routeHref({ surface: "inspect", runId, view: "context" })}>
          span {shortId(evidence.addressId)}
        </a>
      );
  }
}

function FindingCard({ runId, finding }: { runId: string; finding: DiagnosisFinding }) {
  const status = findingStatus(finding.status);
  return (
    <article
      className={`diagnosis-finding ${status.className}`}
      data-finding-status={finding.status}
      id={`diagnosis-finding-${finding.ruleId}`}
    >
      <header>
        <div>
          <code>{finding.ruleId}</code>
          <strong>{finding.ruleName.replaceAll("_", " ")}</strong>
        </div>
        <span className={`diagnosis-status ${status.className}`}>{status.label}</span>
      </header>
      {finding.status === "finding" && (
        <p className="diagnosis-finding-confidence">
          {finding.severity} severity · {finding.confidence.replaceAll("_", " ")} confidence
        </p>
      )}
      <p className="diagnosis-finding-summary">{finding.summary}</p>
      {finding.limitations.length > 0 && (
        <ul className="diagnosis-limitations">
          {finding.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}
        </ul>
      )}
      {finding.evidence.length > 0 && (
        <ul className="diagnosis-evidence" aria-label={`Evidence for ${finding.ruleId}`}>
          {finding.evidence.map((evidence, index) => (
            <li key={index}>
              {evidence.kind === "text_span" ? (
                <a href={routeHref({ surface: "inspect", runId, view: "context" })}>{evidenceText(evidence)}</a>
              ) : evidenceText(evidence)}
            </li>
          ))}
        </ul>
      )}
      {finding.status === "finding" && finding.suggestedActions.length > 0 && (
        <section className="diagnosis-suggestions" aria-label={`Suggested direction for ${finding.ruleId}`}>
          <span>Suggested direction · descriptive only</span>
          <ul>
            {finding.suggestedActions.map((suggestion, index) => (
              <li key={index}>
                <strong>{suggestion.kind.replaceAll("_", " ")}</strong>
                <p>{suggestion.description}</p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}

function ChildState({
  label,
  child,
  reply,
}: {
  label: string;
  child: CorrectiveChild;
  reply: string;
}) {
  const detail = child.status === "error" ? child.error.message : child.status.replaceAll("_", " ");
  return (
    <article className={`corrective-child state-${child.status}`}>
      <header>
        <span>{label}</span>
        <strong>{child.status === "success" ? shortId(child.runId) : detail}</strong>
      </header>
      <p>{reply || "No recorded output."}</p>
      {child.status === "success" && (
        <a href={routeHref({ surface: "inspect", runId: child.runId })}>Inspect child</a>
      )}
    </article>
  );
}

export function DiagnosisRepairSurface({ route }: { route: InspectRoute }) {
  const [diagnosis, setDiagnosis] = useState<Resource<DiagnosisDocument>>({ phase: "loading" });
  const [registry, setRegistry] = useState<Resource<CorrectiveRegistry>>({ phase: "loading" });
  const [selectedActionId, setSelectedActionId] = useState<CorrectiveActionId>();
  const [preview, setPreview] = useState<CorrectivePreview>();
  const [result, setResult] = useState<CorrectiveResult>();
  const [phase, setPhase] = useState<FlowPhase>("idle");
  const [flowError, setFlowError] = useState<string>();
  const generationRef = useRef(0);
  const operationControllers = useRef(new Set<AbortController>());
  const confirmRequest = useRef<{ previewId: string; key: string } | undefined>(undefined);
  const keepRequest = useRef<{ resultId: string; key: string } | undefined>(undefined);
  const confirmController = useRef<AbortController | undefined>(undefined);

  useEffect(() => {
    const generation = ++generationRef.current;
    for (const operation of operationControllers.current) operation.abort();
    operationControllers.current.clear();
    confirmRequest.current = undefined;
    keepRequest.current = undefined;
    confirmController.current = undefined;
    setDiagnosis({ phase: "loading" });
    setRegistry({ phase: "loading" });
    setSelectedActionId(undefined);
    setPreview(undefined);
    setResult(undefined);
    setPhase("idle");
    setFlowError(undefined);

    const controller = new AbortController();
    void diagnosisActionsApi.diagnosis(route.runId, { signal: controller.signal }).then((value) => {
      if (!controller.signal.aborted && generationRef.current === generation) {
        setDiagnosis({ phase: "ready", value });
      }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted && generationRef.current === generation && !isAbort(error)) {
        setDiagnosis(failedResource(error));
      }
    });
    void diagnosisActionsApi.registry(route.runId, controller.signal).then((value) => {
      if (!controller.signal.aborted && generationRef.current === generation) {
        setRegistry({ phase: "ready", value });
      }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted && generationRef.current === generation && !isAbort(error)) {
        setRegistry(failedResource(error));
      }
    });
    return () => {
      controller.abort();
      if (generationRef.current === generation) generationRef.current += 1;
      for (const operation of operationControllers.current) operation.abort();
      operationControllers.current.clear();
      confirmController.current = undefined;
    };
  }, [route.runId]);

  const beginOperation = () => {
    const controller = new AbortController();
    operationControllers.current.add(controller);
    return { controller, generation: generationRef.current };
  };

  const finishOperation = (controller: AbortController) => {
    operationControllers.current.delete(controller);
  };

  const isCurrent = (controller: AbortController, generation: number) =>
    !controller.signal.aborted && generationRef.current === generation;

  async function startPreview(actionId: CorrectiveActionId) {
    const { controller, generation } = beginOperation();
    setSelectedActionId(actionId);
    setPreview(undefined);
    setResult(undefined);
    setFlowError(undefined);
    setPhase("previewing");
    confirmRequest.current = undefined;
    keepRequest.current = undefined;
    try {
      const next = await diagnosisActionsApi.preview(route.runId, actionId, controller.signal);
      if (!isCurrent(controller, generation)) return;
      setPreview(next);
      setPhase("preview_ready");
    } catch (error: unknown) {
      if (!isCurrent(controller, generation) || isAbort(error)) return;
      setPhase("idle");
      setFlowError(`${describeError(error)} The recorded Run is unchanged.`);
    } finally {
      finishOperation(controller);
    }
  }

  async function cancelPreview() {
    if (!preview) return;
    confirmController.current?.abort();
    confirmController.current = undefined;
    const { controller, generation } = beginOperation();
    setFlowError(undefined);
    setPhase("cancelling");
    try {
      await diagnosisActionsApi.cancel(preview.previewId, controller.signal);
      if (!isCurrent(controller, generation)) return;
      setPreview(undefined);
      setSelectedActionId(undefined);
      setPhase("idle");
      confirmRequest.current = undefined;
    } catch (error: unknown) {
      if (!isCurrent(controller, generation) || isAbort(error)) return;
      setPhase("preview_ready");
      setFlowError(
        `${describeError(error)} Cancellation could not be confirmed. Confirm again to recover the active or completed result.`,
      );
    } finally {
      finishOperation(controller);
    }
  }

  async function confirmPreview() {
    if (!preview) return;
    const { controller, generation } = beginOperation();
    confirmController.current = controller;
    const logicalRequest = confirmRequest.current?.previewId === preview.previewId
      ? confirmRequest.current
      : { previewId: preview.previewId, key: idempotencyKey("diagnosis-confirm") };
    confirmRequest.current = logicalRequest;
    setFlowError(undefined);
    setPhase("confirming");
    try {
      const next = await diagnosisActionsApi.confirm(
        preview.previewId,
        { idempotencyKey: logicalRequest.key },
        controller.signal,
      );
      if (!isCurrent(controller, generation)) return;
      setResult(next);
      setPhase("result_ready");
      keepRequest.current = undefined;
      if (next.outcome.status !== "succeeded") {
        setFlowError(`${next.outcome.note ?? "The matched retry did not succeed."} The recorded Run is unchanged.`);
      }
    } catch (error: unknown) {
      if (!isCurrent(controller, generation) || isAbort(error)) return;
      setPhase("preview_ready");
      setFlowError(`${describeError(error)} The preview is still available and the recorded Run is unchanged.`);
    } finally {
      if (confirmController.current === controller) confirmController.current = undefined;
      finishOperation(controller);
    }
  }

  async function keepResult() {
    if (!result) return;
    const eligibility = result.scopeEligibility[0];
    if (!eligibility.available || !eligibility.priorHash) return;
    const { controller, generation } = beginOperation();
    const logicalRequest = keepRequest.current?.resultId === result.resultId
      ? keepRequest.current
      : { resultId: result.resultId, key: idempotencyKey("diagnosis-keep") };
    keepRequest.current = logicalRequest;
    setFlowError(undefined);
    setPhase("keeping");
    try {
      const next = await diagnosisActionsApi.keep(result.resultId, {
        expectedPriorHash: eligibility.priorHash,
        idempotencyKey: logicalRequest.key,
      }, controller.signal);
      if (!isCurrent(controller, generation)) return;
      setResult(next);
      setPhase("kept");
    } catch (error: unknown) {
      if (!isCurrent(controller, generation) || isAbort(error)) return;
      setPhase("result_ready");
      setFlowError(`${describeError(error)} Nothing was kept; the recorded Run is unchanged.`);
    } finally {
      finishOperation(controller);
    }
  }

  async function undoKeep() {
    if (!result?.transaction || result.transaction.undoneTs !== null) return;
    const { controller, generation } = beginOperation();
    setFlowError(undefined);
    setPhase("undoing");
    try {
      await diagnosisActionsApi.undo(result.transaction.id, controller.signal);
      if (!isCurrent(controller, generation)) return;
      setResult({
        ...result,
        transaction: { ...result.transaction, undoneTs: Date.now() / 1000 },
      });
      setPhase("undone");
    } catch (error: unknown) {
      if (!isCurrent(controller, generation) || isAbort(error)) return;
      setPhase("kept");
      setFlowError(describeError(error));
    } finally {
      finishOperation(controller);
    }
  }

  function discardResult() {
    setPreview(undefined);
    setResult(undefined);
    setSelectedActionId(undefined);
    setFlowError(undefined);
    setPhase("idle");
    confirmRequest.current = undefined;
    keepRequest.current = undefined;
  }

  const sessionHref = route.sessionId
    ? routeHref({ surface: "session", sessionId: route.sessionId, runId: route.runId })
    : routeHref({ surface: "sessions" });
  const diagnosisDocument = diagnosis.phase === "ready" ? diagnosis.value : undefined;
  const registryDocument = registry.phase === "ready" ? registry.value : undefined;
  const busy = phase === "previewing" || phase === "confirming" || phase === "cancelling"
    || phase === "keeping" || phase === "undoing";
  const baseline = result?.children.baseline;
  const corrected = result?.children.corrected;
  const pairReady = baseline?.status === "success" && corrected?.status === "success";
  const keepEligibility = result?.scopeEligibility[0];

  return (
    <div className="inspect-surface diagnosis-repair-surface">
      <header className="inspect-heading">
        <div>
          <a className="back-link" href={sessionHref}>← {route.sessionId ? "Session" : "Sessions"}</a>
          <h1>Inspect</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {diagnosisDocument && <span className="diagnosis-headline">{diagnosisDocument.narrative.headline}</span>}
      </header>

      <section className="inspect-instrument diagnosis-instrument" aria-labelledby="diagnosis-title">
        <header className="diagnosis-section-heading">
          <div><span className="eyebrow">Diagnosis</span><h2 id="diagnosis-title">Why, and what to try</h2></div>
          {diagnosisDocument && <span>{diagnosisDocument.findings.findings.length} rules</span>}
        </header>

        {diagnosis.phase === "loading" ? (
          <p className="diagnosis-load-state" role="status">Loading diagnosis.</p>
        ) : diagnosis.phase !== "ready" ? (
          <div className={`diagnosis-request-state is-${diagnosis.phase}`} role="alert">
            <strong>{diagnosis.phase === "malformed" ? "Diagnosis response malformed" : "Diagnosis unavailable"}</strong>
            <span>{diagnosis.error}</span>
          </div>
        ) : (
          <>
            {diagnosis.value.findings.redacted && (
              <p className="diagnosis-redacted">Run text is redacted. Text-based rules could not run.</p>
            )}
            <div className="diagnosis-registers">
              <section className="diagnosis-register is-measured" aria-label="Measured effects">
                <header><strong>Measured effects</strong><span>Ranked · each names a rule</span></header>
                {diagnosis.value.narrative.measuredEffects.length > 0 ? (
                  <ol>
                    {diagnosis.value.narrative.measuredEffects.map((entry) => (
                      <li key={`${entry.rank}:${entry.ruleId}`}>
                        <span>#{entry.rank}</span><p>{entry.text}</p>
                        <button type="button" onClick={() => scrollToFinding(entry.ruleId)}>{entry.ruleId}</button>
                      </li>
                    ))}
                  </ol>
                ) : <p>No ranked measured effects for this Run.</p>}
              </section>
              <section className="diagnosis-register is-observed" aria-label="Observed changes">
                <header><strong>Observed changes</strong><span>Never ranked</span></header>
                {!diagnosis.value.narrative.comparisonAvailable ? (
                  <p>No comparison Run was supplied for this evaluation.</p>
                ) : diagnosis.value.narrative.observedChanges.length > 0 ? (
                  <ul>
                    {diagnosis.value.narrative.observedChanges.map((entry, index) => (
                      <li key={index}>
                        <p>{entry.text}</p>
                        {entry.evidence[0] && <NarrativeEvidence runId={route.runId} evidence={entry.evidence[0]} />}
                      </li>
                    ))}
                  </ul>
                ) : <p>No structural differences against the comparison Run.</p>}
              </section>
              <section className="diagnosis-register is-plausible" aria-label="Plausible but unproven">
                <header><strong>Plausible · unproven</strong><span>Never ranked above measured effects</span></header>
                {diagnosis.value.narrative.plausibleButUnproven.length > 0 ? (
                  <ul>
                    {diagnosis.value.narrative.plausibleButUnproven.map((entry, index) => (
                      <li key={index}><p>{entry.text}</p><small>{entry.note}</small></li>
                    ))}
                  </ul>
                ) : <p>No unproven hypotheses for this Run.</p>}
              </section>
            </div>
            <div className="diagnosis-findings" aria-label="Rule findings">
              {diagnosis.value.findings.findings.map((finding) => (
                <FindingCard runId={route.runId} finding={finding} key={finding.ruleId} />
              ))}
            </div>
          </>
        )}

        <section className="corrective-retries" aria-labelledby="corrective-retries-title">
          <header className="diagnosis-section-heading">
            <div>
              <span className="eyebrow">Bounded · reversible · prompt_policy</span>
              <h2 id="corrective-retries-title">Corrective retries</h2>
            </div>
            <span>{registryDocument?.actions.length ?? 0} actions</span>
          </header>
          <p className="corrective-retries-note">
            These registry actions apply to the whole Run. Diagnosis suggestions above are descriptive and do not select one automatically.
          </p>

          {registry.phase === "loading" ? (
            <p className="diagnosis-load-state" role="status">Loading corrective actions.</p>
          ) : registry.phase !== "ready" ? (
            <div className={`diagnosis-request-state is-${registry.phase}`} role="alert">
              <strong>{registry.phase === "malformed" ? "Corrective action response malformed" : "Corrective actions unavailable"}</strong>
              <span>{registry.error}</span>
            </div>
          ) : (
            <>
              <div className="corrective-action-list" aria-label="Available corrective actions">
                {registry.value.actions.map((action) => (
                  <button
                    type="button"
                    className={selectedActionId === action.id ? "is-selected" : undefined}
                    aria-pressed={selectedActionId === action.id}
                    disabled={busy || !action.eligibility.eligible}
                    title={action.eligibility.eligible ? undefined : action.eligibility.reason}
                    onClick={() => { void startPreview(action.id); }}
                    key={action.id}
                  >
                    <span><strong>{action.label}</strong><code>prompt_policy</code></span>
                    <small>{action.description}</small>
                    <b>{action.eligibility.eligible ? "Preview" : "Unavailable"}</b>
                  </button>
                ))}
              </div>
              <p className={`corrective-flow-state state-${phase}`} role="status">{flowLabel(phase)}</p>
              {flowError && <p className="corrective-flow-error" role="alert">{flowError}</p>}

              {preview && !result && (
                <section className="corrective-preview" aria-label="Corrective retry preview">
                  <header><div><span className="eyebrow">Preview</span><strong>{preview.action.label}</strong></div><code>{shortId(preview.previewId)}</code></header>
                  <dl>
                    <div><dt>Will inject</dt><dd>{preview.action.description}</dd></div>
                    <div><dt>Requested backend</dt><dd>{preview.execution.requestedBackend}</dd></div>
                    <div><dt>Will execute</dt><dd>{preview.execution.expectedExecutedBackend}</dd></div>
                    <div><dt>Fallback</dt><dd>{preview.execution.expectedFallback ? "Yes" : "No"}</dd></div>
                  </dl>
                  <dl className="corrective-preview-contract">
                    <div><dt>Matched baseline</dt><dd>{preview.comparisonContract.baseline}</dd></div>
                    <div><dt>Corrected arm</dt><dd>{preview.comparisonContract.corrected}</dd></div>
                    <div><dt>Stored original</dt><dd>{preview.comparisonContract.storedOriginal}</dd></div>
                  </dl>
                  <div className="corrective-preview-actions">
                    <button type="button" disabled={phase === "cancelling"} onClick={() => { void cancelPreview(); }}>
                      {phase === "cancelling" ? "Cancelling…" : "Cancel preview"}
                    </button>
                    <button type="button" disabled={phase === "confirming" || phase === "cancelling"} onClick={() => { void confirmPreview(); }}>
                      {phase === "confirming" ? "Running…" : "Confirm · run matched retry"}
                    </button>
                  </div>
                </section>
              )}

              {result && (
                <section className="corrective-result" aria-label="Corrective retry result">
                  <header>
                    <div><span className="eyebrow">Result</span><strong>{result.action.label}</strong></div>
                    <span className={`corrective-outcome state-${result.outcome.status}`}>
                      {result.outcome.status.replaceAll("_", " ")}
                    </span>
                  </header>
                  <div className="corrective-original">
                    <span>Stored original</span>
                    <p>{result.comparison.storedOriginalReply || "No stored reply."}</p>
                  </div>
                  <div className="corrective-children">
                    <ChildState label="Matched baseline" child={result.children.baseline} reply={result.comparison.baselineReply} />
                    <ChildState label="Corrected" child={result.children.corrected} reply={result.comparison.correctedReply} />
                  </div>
                  <div className="corrective-recorded-evidence">
                    <section><span className="eyebrow">Execution identity</span><EvidenceFields label="Execution identity" fields={result.execution.identity} /></section>
                    <section><span className="eyebrow">Result metrics</span><EvidenceFields label="Result metrics" fields={result.metrics} /></section>
                  </div>
                  <p className="corrective-result-note">{result.comparison.note}</p>
                  <nav className="corrective-result-links" aria-label="Corrective result links">
                    <a href={routeHref({ surface: "inspect", runId: result.parentRunId })}>Inspect parent</a>
                    {pairReady && (
                      <a href={routeHref({ surface: "compare", runA: baseline.runId, runB: corrected.runId })}>
                        Compare matched children
                      </a>
                    )}
                  </nav>

                  {result.outcome.status === "succeeded" && !result.transaction && (
                    <div className="corrective-keep">
                      <div>
                        <strong>Keep corrected child for this Run</strong>
                        <span>{keepEligibility?.available
                          ? keepEligibility.note ?? "Selects the corrected child as this Run’s revision."
                          : keepEligibility?.unavailabilityReason ?? "Keeping is unavailable."}</span>
                      </div>
                      <button
                        type="button"
                        disabled={busy || !keepEligibility?.available || !keepEligibility.priorHash}
                        onClick={() => { void keepResult(); }}
                      >
                        {phase === "keeping" ? "Keeping…" : "Keep correction"}
                      </button>
                      <button type="button" disabled={busy} onClick={discardResult}>Keep nothing</button>
                    </div>
                  )}
                  {result.transaction && (
                    <div className="corrective-kept" role="status">
                      <div>
                        <strong>{result.transaction.undoneTs === null ? "Correction kept for this Run" : "Correction undone"}</strong>
                        <code>{shortId(result.transaction.id)}</code>
                      </div>
                      <button
                        type="button"
                        disabled={busy || result.transaction.undoneTs !== null}
                        onClick={() => { void undoKeep(); }}
                      >
                        {result.transaction.undoneTs !== null ? "Undone" : phase === "undoing" ? "Undoing…" : "Undo"}
                      </button>
                    </div>
                  )}
                  {result.outcome.status !== "succeeded" && (
                    <p className="corrective-result-failed" role="alert">The retry did not succeed. The recorded Run is unchanged.</p>
                  )}
                </section>
              )}
            </>
          )}
        </section>
      </section>
    </div>
  );
}
