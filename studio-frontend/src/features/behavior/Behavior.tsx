import { useEffect, useState } from "react";
import type { RuntimeState } from "../../data/types";
import {
  cancelCorrectivePreview,
  confirmCorrectivePreview,
  keepCorrectiveResult,
  loadBehaviorWorkspace,
  loadCorrectiveActions,
  measureCorrectiveSourceUse,
  previewCorrectiveAction,
  saveSampling,
  undoCorrectiveKeep,
  type CorrectiveAction,
  type CorrectiveBackend,
  type CorrectivePreviewReceipt,
  type CorrectiveRegistry,
  type CorrectiveResult,
  type CorrectiveScope,
  type SamplingSettings,
  type SourceUseComparison,
} from "./api";

interface BehaviorProps {
  runtime: RuntimeState;
  inspectorOpen: boolean;
}

type BehaviorView = "fixes" | "runtime";
type LoadStatus = "loading" | "ready" | "error";
type OperationStatus = "idle" | "draft" | "pending" | "applied" | "failed" | "reverted";

interface OperationState {
  status: OperationStatus;
  action: string;
  detail?: string;
}

const modules: Array<{ id: BehaviorView; label: string }> = [
  { id: "fixes", label: "ONE-SHOT RETRIES" },
  { id: "runtime", label: "RUNTIME DEFAULTS" },
];

function basename(value?: string) {
  return value?.split(/[\\/]/).pop() || "—";
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error || "Operation failed");
}

function idempotencyKey(prefix: string) {
  const suffix = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}:${suffix}`;
}

function metricText(value: unknown) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "YES" : "NO";
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function Behavior({ runtime, inspectorOpen }: BehaviorProps) {
  const [view, setView] = useState<BehaviorView>("fixes");
  const [loadStatus, setLoadStatus] = useState<LoadStatus>("loading");
  const [sampling, setSampling] = useState<SamplingSettings>();
  const [samplingDraft, setSamplingDraft] = useState<SamplingSettings>();
  const [errors, setErrors] = useState<Awaited<ReturnType<typeof loadBehaviorWorkspace>>["errors"]>({});
  const [operation, setOperation] = useState<OperationState>({
    status: "idle",
    action: "NO PENDING CHANGE",
  });
  const [fixRegistry, setFixRegistry] = useState<CorrectiveRegistry>();
  const [fixLoadError, setFixLoadError] = useState("");
  const [selectedRunId, setSelectedRunId] = useState(
    runtime.runs.find((run) => !run.parentRunId)?.id ?? runtime.runs[0]?.id ?? "",
  );
  const [selectedFixId, setSelectedFixId] = useState("");
  const [fixBackend, setFixBackend] = useState<CorrectiveBackend>("prompt_policy");
  const [fixPreview, setFixPreview] = useState<CorrectivePreviewReceipt>();
  const [fixResult, setFixResult] = useState<CorrectiveResult>();
  const [fixConfirming, setFixConfirming] = useState(false);
  const [sourceUse, setSourceUse] = useState<SourceUseComparison>();

  function installWorkspace(next: Awaited<ReturnType<typeof loadBehaviorWorkspace>>) {
    setSampling(next.sampling);
    setSamplingDraft(next.sampling);
    setErrors(next.errors);
  }

  useEffect(() => {
    const controller = new AbortController();
    setLoadStatus("loading");
    void loadBehaviorWorkspace(controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      installWorkspace(next);
      setLoadStatus(next.sampling ? "ready" : "error");
    }).catch(() => {
      if (!controller.signal.aborted) setLoadStatus("error");
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (selectedRunId && runtime.runs.some((run) => run.id === selectedRunId)) return;
    setSelectedRunId(
      runtime.runs.find((run) => !run.parentRunId)?.id ?? runtime.runs[0]?.id ?? "",
    );
  }, [runtime.runs, selectedRunId]);

  useEffect(() => {
    if (view !== "fixes" || !selectedRunId) return;
    const controller = new AbortController();
    setFixLoadError("");
    void loadCorrectiveActions(selectedRunId, controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      setFixRegistry(next);
      setSelectedFixId((current) =>
        next.actions.some((action) => action.id === current)
          ? current
          : next.actions[0]?.id ?? "");
    }).catch((error) => {
      if (!controller.signal.aborted) setFixLoadError(errorMessage(error));
    });
    return () => controller.abort();
  }, [view, selectedRunId]);

  const samplingDirty = Boolean(
    sampling
    && samplingDraft
    && (
      sampling.sampling !== samplingDraft.sampling
      || sampling.sample_temperature !== samplingDraft.sample_temperature
      || sampling.sample_top_p !== samplingDraft.sample_top_p
      || sampling.sample_top_k !== samplingDraft.sample_top_k
      || sampling.sample_repeat_penalty !== samplingDraft.sample_repeat_penalty
    ),
  );
  const pendingCount = Number(samplingDirty);
  const selectedFix: CorrectiveAction | undefined = fixRegistry?.actions.find(
    (action) => action.id === selectedFixId,
  );
  const selectedRun = runtime.runs.find((run) => run.id === selectedRunId);

  function updateSamplingDraft(
    next: SamplingSettings,
    label: string,
    value: string,
  ) {
    setSamplingDraft(next);
    setOperation({
      status: "draft",
      action: `DRAFT · ${label}`,
      detail: value,
    });
  }

  function revertSamplingDraft() {
    setSamplingDraft(sampling);
    setOperation({
      status: "reverted",
      action: "DECODING DRAFT REVERTED",
    });
  }

  async function commitSampling() {
    if (!samplingDraft) return;
    setOperation({ status: "pending", action: "APPLYING RUNTIME DEFAULTS" });
    try {
      const next = await saveSampling(samplingDraft);
      setSampling(next);
      setSamplingDraft(next);
      setOperation({ status: "applied", action: "RUNTIME DEFAULTS APPLIED" });
    } catch (error) {
      setOperation({ status: "failed", action: "RUNTIME APPLY FAILED", detail: errorMessage(error) });
    }
  }

  function chooseFix(actionId: string) {
    setSelectedFixId(actionId);
    setFixPreview(undefined);
    setFixResult(undefined);
    setSourceUse(undefined);
    setOperation({ status: "draft", action: "ANSWER FIX SELECTED", detail: actionId });
  }

  async function createFixPreview() {
    if (!selectedRunId || !selectedFix) return;
    setOperation({
      status: "pending",
      action: "PREPARING ANSWER FIX",
      detail: selectedFix.label,
    });
    try {
      const preview = await previewCorrectiveAction(
        selectedRunId, selectedFix.id, fixBackend,
      );
      setFixPreview(preview);
      setFixResult(undefined);
      setSourceUse(undefined);
      setOperation({
        status: "draft",
        action: "PREVIEW READY · CONFIRM REQUIRED",
        detail: `${preview.execution.requested_backend} → ${preview.execution.expected_executed_backend}`,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "ANSWER FIX PREVIEW FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function confirmFixPreview() {
    if (!fixPreview) return;
    setFixConfirming(true);
    setOperation({
      status: "pending",
      action: "RUNNING ONE-SHOT MATCHED RETRY",
      detail: fixPreview.action.label,
    });
    try {
      const result = await confirmCorrectivePreview(
        fixPreview.preview_id,
        idempotencyKey("confirm"),
      );
      setFixResult(result);
      setOperation({
        status: result.outcome.status === "succeeded" ? "applied" : "failed",
        action: result.outcome.status === "succeeded"
          ? "MATCHED RETRY COMPLETE"
          : `MATCHED RETRY ${result.outcome.status.toUpperCase()}`,
        detail: `${result.execution.requested_backend} → ${result.execution.executed_backend ?? "not executed"}`,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "MATCHED RETRY FAILED",
        detail: errorMessage(error),
      });
    } finally {
      setFixConfirming(false);
    }
  }

  async function cancelFix() {
    if (!fixPreview) return;
    try {
      await cancelCorrectivePreview(fixPreview.preview_id);
      setOperation({
        status: "reverted",
        action: fixConfirming ? "CANCELLATION REQUESTED" : "PREVIEW CANCELLED",
        detail: "No future request is affected.",
      });
      if (!fixConfirming) setFixPreview(undefined);
    } catch (error) {
      setOperation({
        status: "failed",
        action: "CANCEL FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function keepFix(scope: CorrectiveScope) {
    if (!fixResult) return;
    const eligibility = fixResult.scope_eligibility.find((item) => item.scope === scope);
    if (!eligibility?.prior_hash) return;
    setOperation({
      status: "pending",
      action: `KEEPING FIX · ${scope.toUpperCase()}`,
      detail: fixResult.action.label,
    });
    try {
      const kept = await keepCorrectiveResult(
        fixResult.result_id,
        scope,
        eligibility.prior_hash,
        idempotencyKey("keep"),
      );
      setFixResult(kept);
      setOperation({
        status: "applied",
        action: `FIX KEPT · ${scope.toUpperCase()}`,
        detail: kept.transaction?.id,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "KEEP REFUSED",
        detail: errorMessage(error),
      });
    }
  }

  async function undoFix() {
    if (!fixResult?.transaction?.id) return;
    setOperation({ status: "pending", action: "UNDOING ANSWER FIX" });
    try {
      await undoCorrectiveKeep(fixResult.transaction.id);
      setFixResult({
        ...fixResult,
        transaction: { ...fixResult.transaction, undone_ts: Date.now() / 1000 },
      });
      setOperation({
        status: "reverted",
        action: "ANSWER FIX UNDONE",
        detail: fixResult.transaction.scope,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "UNDO REFUSED",
        detail: errorMessage(error),
      });
    }
  }

  async function measureSourceUse() {
    if (!fixResult) return;
    setOperation({
      status: "pending",
      action: "MEASURING SOURCE USE · EXPENSIVE",
      detail: "Scoring both matched child runs.",
    });
    try {
      const measured = await measureCorrectiveSourceUse(fixResult);
      setSourceUse(measured);
      setOperation({
        status: "applied",
        action: "SOURCE-USE COMPARISON READY",
        detail: `${measured.delta_observed_source_dependence_ratio >= 0 ? "+" : ""}${measured.delta_observed_source_dependence_ratio.toFixed(3)}`,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "SOURCE-USE MEASUREMENT FAILED",
        detail: errorMessage(error),
      });
    }
  }

  return (
    <>
      <aside className="instrument behavior-stack" aria-labelledby="behavior-stack-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">INTERVENTION SCOPE</span>
            <h2 id="behavior-stack-title">Behavior</h2>
          </div>
          <strong>{pendingCount} DRAFT</strong>
        </header>
        <nav className="behavior-modules" aria-label="Behavior modules">
          {modules.map((module) => (
            <button
              type="button"
              className={view === module.id ? "is-active" : ""}
              aria-pressed={view === module.id}
              onClick={() => setView(module.id)}
              key={module.id}
            >
              <span>{module.label}</span>
            </button>
          ))}
        </nav>
        <section className="behavior-stack-state">
          <header><span>MODEL</span></header>
          <dl>
            <div><dt>Model</dt><dd>{basename(runtime.engine?.model)}</dd></div>
          </dl>
        </section>
      </aside>

      <section className="instrument behavior-console" aria-labelledby="behavior-console-title">
        {loadStatus === "loading" ? (
          <div className="behavior-load-state">READING BEHAVIOR STATE</div>
        ) : view === "fixes" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">REVERSIBLE · REGISTRY DRIVEN</span>
                <h1 id="behavior-console-title">Fix this answer</h1>
              </div>
              <div className="behavior-head-stats">
                <span><b>ACTIONS</b>{fixRegistry?.actions.length ?? 0}</span>
                <span><b>RUN</b>{selectedRunId ? selectedRunId.slice(-6) : "—"}</span>
                <span><b>STATE</b>{fixResult?.outcome.status ?? fixPreview?.status ?? "select"}</span>
              </div>
            </header>
            <div className="behavior-fix-stage">
              <section className="behavior-fix-setup">
                <label className="behavior-fix-run">
                  <span>RECORDED RUN</span>
                  <select value={selectedRunId} onChange={(event) => {
                    setSelectedRunId(event.target.value);
                    setFixPreview(undefined);
                    setFixResult(undefined);
                    setSourceUse(undefined);
                  }}>
                    {runtime.runs.map((run) => (
                      <option value={run.id} key={run.id}>{run.label}</option>
                    ))}
                  </select>
                </label>
                {fixLoadError ? (
                  <div className="behavior-unavailable">{fixLoadError}</div>
                ) : (
                  <div className="behavior-fix-actions" aria-label="Corrective action registry">
                    {fixRegistry?.actions.map((action) => (
                      <button
                        type="button"
                        className={selectedFixId === action.id ? "is-selected" : ""}
                        aria-pressed={selectedFixId === action.id}
                        onClick={() => chooseFix(action.id)}
                        key={action.id}
                      >
                        <strong>{action.label}</strong>
                        <span>{action.description}</span>
                      </button>
                    ))}
                  </div>
                )}
                {selectedFix && (
                  <div className="behavior-fix-backends">
                    <span>EXECUTION BACKEND</span>
                    <button
                      type="button"
                      className={fixBackend === "prompt_policy" ? "is-selected" : ""}
                      onClick={() => {
                        setFixBackend("prompt_policy");
                        setFixPreview(undefined);
                      }}
                    >
                      PROMPT POLICY · GENERIC
                    </button>
                    <button
                      type="button"
                      className={fixBackend === "control_vector" ? "is-selected" : ""}
                      onClick={() => {
                        setFixBackend("control_vector");
                        setFixPreview(undefined);
                      }}
                    >
                      CONTROL VECTOR · {
                        selectedFix.backends.some((backend) =>
                          backend.type === "control_vector" && backend.available === true)
                          ? "QUALIFIED"
                          : "FALLBACK EXPECTED"
                      }
                    </button>
                  </div>
                )}
                <button
                  type="button"
                  className="is-primary behavior-fix-preview-button"
                  disabled={!selectedRun || !selectedFix || operation.status === "pending"}
                  onClick={() => void createFixPreview()}
                >
                  PREVIEW ACTION
                </button>
              </section>

              {fixPreview && !fixResult && (
                <section className="behavior-fix-confirm">
                  <header>
                    <span>PREVIEW RECEIPT</span>
                    <b>{fixPreview.preview_id.slice(-8)}</b>
                  </header>
                  <dl>
                    <div><dt>ACTION</dt><dd>{fixPreview.action.label}</dd></div>
                    <div><dt>REQUESTED</dt><dd>{fixPreview.execution.requested_backend}</dd></div>
                    <div><dt>WILL EXECUTE</dt><dd>{fixPreview.execution.expected_executed_backend}</dd></div>
                    <div><dt>FALLBACK</dt><dd>{fixPreview.execution.expected_fallback ? "YES" : "NO"}</dd></div>
                  </dl>
                  {fixPreview.execution.unavailability_reason && (
                    <p>{fixPreview.execution.unavailability_reason}</p>
                  )}
                  <p>
                    Confirm runs one matched greedy baseline and one corrected child. It never
                    changes behavior for a future, unrelated request.
                  </p>
                  <div>
                    <button type="button" onClick={() => void cancelFix()}>
                      {fixConfirming ? "REQUEST CANCEL" : "CANCEL"}
                    </button>
                    <button
                      type="button"
                      className="is-primary"
                      disabled={fixConfirming}
                      onClick={() => void confirmFixPreview()}
                    >
                      {fixConfirming ? "RUNNING…" : "CONFIRM ONE-SHOT RETRY"}
                    </button>
                  </div>
                </section>
              )}

              {fixResult && (
                <section className="behavior-fix-result">
                  <header>
                    <div>
                      <span>STRUCTURED OUTCOME</span>
                      <strong>{fixResult.action.label}</strong>
                    </div>
                    <b className={`is-${fixResult.outcome.status}`}>
                      {fixResult.outcome.status.toUpperCase()}
                    </b>
                  </header>
                  <div className="behavior-fix-execution">
                    <span>REQUESTED <b>{fixResult.execution.requested_backend}</b></span>
                    <span>EXECUTED <b>{fixResult.execution.executed_backend ?? "none"}</b></span>
                    <span>FALLBACK <b>{fixResult.execution.fallback ? "YES" : "NO"}</b></span>
                    <span>QUALIFICATION <b>{fixResult.execution.qualification_id ?? "unavailable"}</b></span>
                  </div>
                  <div className="behavior-fix-outputs">
                    <article>
                      <header><span>STORED ORIGINAL</span><b>CONTEXT ONLY</b></header>
                      <p>{fixResult.comparison.stored_original_reply || "No stored reply."}</p>
                    </article>
                    <article>
                      <header>
                        <span>MATCHED GREEDY BASELINE</span>
                        <b>{fixResult.children.baseline.run_id?.slice(-8) ?? fixResult.children.baseline.status}</b>
                      </header>
                      <p>{fixResult.comparison.baseline_reply || "No baseline output."}</p>
                      {fixResult.children.baseline.error && (
                        <small>{fixResult.children.baseline.error.code} · {fixResult.children.baseline.error.message}</small>
                      )}
                    </article>
                    <article className="is-corrected">
                      <header>
                        <span>CORRECTED</span>
                        <b>{fixResult.children.corrected.run_id?.slice(-8) ?? fixResult.children.corrected.status}</b>
                      </header>
                      <p>{fixResult.comparison.corrected_reply || "No corrected output."}</p>
                      {fixResult.children.corrected.error && (
                        <small>{fixResult.children.corrected.error.code} · {fixResult.children.corrected.error.message}</small>
                      )}
                    </article>
                  </div>
                  <p className="behavior-fix-comparison-note">{fixResult.comparison.note}</p>
                  <div className="behavior-fix-metrics">
                    {Object.entries(fixResult.metrics).map(([name, value]) => (
                      <span key={name}><b>{name.replaceAll("_", " ")}</b>{metricText(value)}</span>
                    ))}
                    {!Object.keys(fixResult.metrics).length && <span><b>METRICS</b>UNAVAILABLE</span>}
                  </div>
                  {fixResult.outcome.status === "succeeded" && !fixResult.transaction && (
                    <div className="behavior-fix-scopes">
                      <header><span>KEEP CORRECTED RESULT</span><b>SEPARATE MUTATION</b></header>
                      {fixResult.scope_eligibility.map((scope) => (
                        <button
                          type="button"
                          disabled={!scope.available || operation.status === "pending"}
                          title={scope.unavailability_reason}
                          onClick={() => void keepFix(scope.scope)}
                          key={scope.scope}
                        >
                          <strong>{scope.scope.toUpperCase()}</strong>
                          <span>{scope.available
                            ? scope.note ?? `${(scope.before ?? []).length} → ${(scope.after ?? []).length} active actions`
                            : scope.unavailability_reason}</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {fixResult.transaction && (
                    <div className="behavior-fix-undo">
                      <span>KEPT FOR {fixResult.transaction.scope.toUpperCase()} · {fixResult.transaction.id}</span>
                      <button
                        type="button"
                        disabled={Boolean(fixResult.transaction.undone_ts) || operation.status === "pending"}
                        onClick={() => void undoFix()}
                      >
                        {fixResult.transaction.undone_ts ? "UNDONE" : "UNDO"}
                      </button>
                    </div>
                  )}
                  {fixResult.outcome.status === "succeeded" && (
                    <div className="behavior-fix-source-use">
                      <button
                        type="button"
                        disabled={operation.status === "pending"}
                        onClick={() => void measureSourceUse()}
                      >
                        MEASURE SOURCE USE · EXPENSIVE
                      </button>
                      {sourceUse && (
                        <div>
                          <strong>
                            OBSERVED SOURCE DEPENDENCE Δ {
                              sourceUse.delta_observed_source_dependence_ratio >= 0 ? "+" : ""
                            }{sourceUse.delta_observed_source_dependence_ratio.toFixed(3)}
                          </strong>
                          <span>{sourceUse.caveat}</span>
                        </div>
                      )}
                    </div>
                  )}
                </section>
              )}
            </div>
          </>
        ) : (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">SERVER DEFAULTS</span>
                <h1 id="behavior-console-title">Runtime</h1>
              </div>
            </header>
            <div className="behavior-runtime-stage">
              <section className="behavior-runtime-block">
                <header><span>DECODING</span><b>{samplingDirty ? "DRAFT" : "CURRENT"}</b></header>
                {samplingDraft ? (
                  <div className="behavior-runtime-fields">
                    <label className="behavior-toggle">
                      <input
                        type="checkbox"
                        checked={samplingDraft.sampling}
                        onChange={(event) => updateSamplingDraft(
                          { ...samplingDraft, sampling: event.target.checked },
                          "SAMPLING",
                          event.target.checked ? "ON" : "OFF",
                        )}
                      />
                      <span>SAMPLING</span>
                    </label>
                    {([
                      ["sample_temperature", "TEMPERATURE", 0, 2, .05],
                      ["sample_top_p", "TOP P", 0, 1, .01],
                      ["sample_top_k", "TOP K", 0, 200, 1],
                      ["sample_repeat_penalty", "REPEAT PENALTY", .5, 2, .01],
                    ] as const).map(([key, label, min, max, step]) => (
                      <label key={key}>
                        <span>{label}</span>
                        <input
                          type="number"
                          min={min}
                          max={max}
                          step={step}
                          value={samplingDraft[key]}
                          onChange={(event) => updateSamplingDraft(
                            {
                              ...samplingDraft,
                              [key]: Number(event.target.value),
                            },
                            label,
                            event.target.value,
                          )}
                        />
                      </label>
                    ))}
                    <div className="behavior-runtime-actions">
                      <button type="button" disabled={!samplingDirty || operation.status === "pending"} onClick={revertSamplingDraft}>REVERT</button>
                      <button type="button" className="is-primary" disabled={!samplingDirty || operation.status === "pending"} onClick={() => void commitSampling()}>APPLY DECODING</button>
                    </div>
                  </div>
                ) : <div className="behavior-unavailable">{errors.sampling || "SAMPLING ROUTE UNAVAILABLE"}</div>}
              </section>
            </div>
          </>
        )}
      </section>

      {inspectorOpen && (
        <aside className="instrument behavior-inspector" aria-labelledby="behavior-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">CHANGE INSPECTOR</span>
              <h2 id="behavior-inspector-title">Consequence</h2>
            </div>
            <span className={`behavior-operation-chip is-${operation.status}`}>{operation.status.toUpperCase()}</span>
          </header>
          <section className="behavior-operation">
            <span>LAST OPERATION</span>
            <strong>{operation.action}</strong>
            {operation.detail && <p>{operation.detail}</p>}
          </section>

          <section className="behavior-scope-facts">
            <header><span>CURRENT SCOPE</span><b>{view.toUpperCase()}</b></header>
            <dl>
              <div><dt>MODEL</dt><dd>{basename(runtime.engine?.model)}</dd></div>
              <div><dt>PENDING DRAFTS</dt><dd>{pendingCount}</dd></div>
            </dl>
          </section>
        </aside>
      )}
    </>
  );
}
