import { useEffect, useMemo, useState, type CSSProperties } from "react";
import type { RuntimeState } from "../../data/types";
import {
  applyAxis,
  applyConcept,
  cancelCorrectivePreview,
  confirmCorrectivePreview,
  keepCorrectiveResult,
  loadBehaviorWorkspace,
  loadCorrectiveActions,
  measureCorrectiveSourceUse,
  previewAxis,
  previewConcept,
  previewCorrectiveAction,
  saveSampling,
  undoCorrectiveKeep,
  type AxisPreview,
  type BehaviorAxis,
  type ConceptPreview,
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

type BehaviorView = "fixes" | "dials" | "concepts" | "runtime";
type LoadStatus = "loading" | "ready" | "error";
type OperationStatus = "idle" | "draft" | "pending" | "applied" | "failed" | "reverted";

interface OperationState {
  status: OperationStatus;
  action: string;
  detail?: string;
}

const modules: Array<{ id: BehaviorView; label: string }> = [
  { id: "fixes", label: "ONE-SHOT RETRIES" },
  { id: "dials", label: "TONE DIALS" },
  { id: "concepts", label: "CONCEPT STEERING" },
  { id: "runtime", label: "RUNTIME DEFAULTS" },
];

const PREVIEW_PROMPT = "Tell me about your day.";

function changed(current: number, draft: number | undefined) {
  return draft != null && Math.abs(current - draft) > 0.0001;
}

function formatValue(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

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
  const [axes, setAxes] = useState<BehaviorAxis[]>([]);
  const [drafts, setDrafts] = useState<Record<string, number>>({});
  const [selectedAxisName, setSelectedAxisName] = useState("");
  const [sampling, setSampling] = useState<SamplingSettings>();
  const [samplingDraft, setSamplingDraft] = useState<SamplingSettings>();
  const [errors, setErrors] = useState<Awaited<ReturnType<typeof loadBehaviorWorkspace>>["errors"]>({});
  const [operation, setOperation] = useState<OperationState>({
    status: "idle",
    action: "NO PENDING CHANGE",
  });
  const [previewPrompt, setPreviewPrompt] = useState(PREVIEW_PROMPT);
  const [axisPreview, setAxisPreview] = useState<AxisPreview>();
  const [conceptPreview, setConceptPreview] = useState<ConceptPreview>();
  const [concept, setConcept] = useState("");
  const [conceptStrength, setConceptStrength] = useState(1);
  const [activeConcepts, setActiveConcepts] = useState<Record<string, number>>({});
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
    setAxes(next.axes);
    setDrafts(Object.fromEntries(next.axes.map((axis) => [axis.name, axis.value])));
    setSelectedAxisName((current) =>
      next.axes.some((axis) => axis.name === current) ? current : next.axes[0]?.name ?? "");
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
      setLoadStatus(next.axes.length || next.sampling ? "ready" : "error");
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

  const selectedAxis = axes.find((axis) => axis.name === selectedAxisName);
  const dirtyAxes = axes.filter((axis) => changed(axis.value, drafts[axis.name]));
  const activeAxes = axes.filter((axis) => Math.abs(axis.value) > 0.0001);
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
  const pendingCount = dirtyAxes.length + Number(samplingDirty);
  const selectedDraft = selectedAxis ? drafts[selectedAxis.name] ?? selectedAxis.value : 0;
  const sortedActiveAxes = useMemo(
    () => [...activeAxes].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [activeAxes],
  );
  const selectedFix: CorrectiveAction | undefined = fixRegistry?.actions.find(
    (action) => action.id === selectedFixId,
  );
  const selectedRun = runtime.runs.find((run) => run.id === selectedRunId);

  function updateDraft(name: string, value: number) {
    setSelectedAxisName(name);
    setDrafts((current) => ({ ...current, [name]: value }));
    const axis = axes.find((item) => item.name === name);
    setOperation({
      status: axis && changed(axis.value, value) ? "draft" : "idle",
      action: axis && changed(axis.value, value) ? `DRAFT · ${name.toUpperCase()}` : "NO PENDING CHANGE",
      detail: axis ? `${formatValue(axis.value)} → ${formatValue(value)}` : undefined,
    });
  }

  function revertAxisDrafts() {
    setDrafts(Object.fromEntries(axes.map((axis) => [axis.name, axis.value])));
    setOperation({
      status: "reverted",
      action: "DRAFTS REVERTED",
      detail: `${dirtyAxes.length} ${dirtyAxes.length === 1 ? "AXIS" : "AXES"}`,
    });
  }

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

  async function commitAxis(axis: BehaviorAxis) {
    const value = drafts[axis.name] ?? axis.value;
    setOperation({
      status: "pending",
      action: `APPLYING · ${axis.name.toUpperCase()}`,
      detail: formatValue(value),
    });
    try {
      const result = await applyAxis(axis.name, value);
      setAxes((current) => current.map((item) =>
        item.name === axis.name ? { ...item, value } : item));
      setDrafts((current) => ({ ...current, [axis.name]: value }));
      setOperation({
        status: "applied",
        action: `APPLIED · ${axis.name.toUpperCase()}`,
        detail: result.warning || formatValue(value),
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: `FAILED · ${axis.name.toUpperCase()}`,
        detail: errorMessage(error),
      });
    }
  }

  async function commitAllAxes() {
    if (!dirtyAxes.length) return;
    setOperation({
      status: "pending",
      action: "APPLYING DIAL STACK",
      detail: `${dirtyAxes.length} ${dirtyAxes.length === 1 ? "AXIS" : "AXES"}`,
    });
    const applied: Record<string, number> = {};
    try {
      for (const axis of dirtyAxes) {
        const value = drafts[axis.name] ?? axis.value;
        await applyAxis(axis.name, value);
        applied[axis.name] = value;
      }
      setAxes((current) => current.map((axis) =>
        applied[axis.name] == null ? axis : { ...axis, value: applied[axis.name] }));
      setOperation({
        status: "applied",
        action: "DIAL STACK APPLIED",
        detail: `${dirtyAxes.length} ${dirtyAxes.length === 1 ? "AXIS" : "AXES"}`,
      });
    } catch (error) {
      setAxes((current) => current.map((axis) =>
        applied[axis.name] == null ? axis : { ...axis, value: applied[axis.name] }));
      setOperation({
        status: "failed",
        action: "PARTIAL APPLY",
        detail: errorMessage(error),
      });
    }
  }

  async function runAxisPreview() {
    if (!selectedAxis || !previewPrompt.trim()) return;
    setOperation({
      status: "pending",
      action: `PREVIEWING · ${selectedAxis.name.toUpperCase()}`,
      detail: formatValue(selectedDraft),
    });
    try {
      const result = await previewAxis(selectedAxis.name, selectedDraft, previewPrompt.trim());
      setAxisPreview(result);
      setConceptPreview(undefined);
      setOperation({
        status: "applied",
        action: `PREVIEW READY · ${selectedAxis.name.toUpperCase()}`,
        detail: result.warning,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "PREVIEW FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function commitConcept(strength = conceptStrength, conceptOverride?: string) {
    const word = (conceptOverride ?? concept).trim();
    if (!word) return;
    setOperation({
      status: "pending",
      action: strength === 0 ? `REMOVING · ${word.toUpperCase()}` : `APPLYING · ${word.toUpperCase()}`,
      detail: formatValue(strength),
    });
    try {
      const active = await applyConcept(word, strength);
      setActiveConcepts(active);
      setOperation({
        status: strength === 0 ? "reverted" : "applied",
        action: strength === 0 ? `REMOVED · ${word.toUpperCase()}` : `APPLIED · ${word.toUpperCase()}`,
        detail: strength === 0 ? undefined : formatValue(strength),
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "CONCEPT APPLY FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function runConceptPreview() {
    const word = concept.trim();
    if (!word || !previewPrompt.trim()) return;
    setOperation({
      status: "pending",
      action: `PREVIEWING · ${word.toUpperCase()}`,
      detail: formatValue(conceptStrength),
    });
    try {
      const result = await previewConcept(word, conceptStrength, previewPrompt.trim());
      setConceptPreview(result);
      setAxisPreview(undefined);
      setOperation({
        status: "applied",
        action: `PREVIEW READY · ${word.toUpperCase()}`,
        detail: result.note,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "CONCEPT PREVIEW FAILED",
        detail: errorMessage(error),
      });
    }
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
              <b>
                {module.id === "fixes"
                  ? fixRegistry?.actions.length ?? 0
                  : module.id === "dials"
                  ? activeAxes.length
                  : module.id === "concepts"
                    ? Object.keys(activeConcepts).length
                    : sampling ? 1 : 0}
              </b>
            </button>
          ))}
        </nav>
        <section className="behavior-stack-state">
          <header><span>ACTIVE STACK</span><b>{activeAxes.length}</b></header>
          <dl>
            <div><dt>Model</dt><dd>{basename(runtime.engine?.model)}</dd></div>
            <div><dt>Tone dials</dt><dd>{activeAxes.length}</dd></div>
            <div><dt>Concept dials</dt><dd>{Object.keys(activeConcepts).length}</dd></div>
          </dl>
          <div className="behavior-active-dials">
            {sortedActiveAxes.slice(0, 6).map((axis) => (
              <button type="button" onClick={() => {
                setView("dials");
                setSelectedAxisName(axis.name);
              }} key={axis.name}>
                <span>{axis.name}</span>
                <output>{formatValue(axis.value)}</output>
              </button>
            ))}
          </div>
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
        ) : view === "dials" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">MODEL INTERVENTION</span>
                <h1 id="behavior-console-title">Tone dials</h1>
              </div>
              <div className="behavior-head-stats">
                <span><b>AVAILABLE</b>{axes.length}</span>
                <span><b>ACTIVE</b>{activeAxes.length}</span>
                <span><b>DRAFT</b>{dirtyAxes.length}</span>
              </div>
              <div className="behavior-head-actions">
                <button type="button" disabled={!dirtyAxes.length} onClick={revertAxisDrafts}>REVERT</button>
                <button type="button" className="is-primary" disabled={!dirtyAxes.length || operation.status === "pending"} onClick={() => void commitAllAxes()}>
                  APPLY {dirtyAxes.length || ""}
                </button>
              </div>
            </header>
            {errors.axes ? (
              <div className="behavior-unavailable">{errors.axes}</div>
            ) : (
              <div className="behavior-dial-list">
                {axes.map((axis) => {
                  const draft = drafts[axis.name] ?? axis.value;
                  const currentPosition = (axis.value + axis.max) / (axis.max * 2) * 100;
                  return (
                    <div
                      className={[
                        "behavior-dial-row",
                        selectedAxisName === axis.name ? "is-selected" : "",
                        changed(axis.value, draft) ? "is-dirty" : "",
                      ].join(" ")}
                      style={{ "--dial-current": `${currentPosition}%` } as CSSProperties}
                      key={axis.name}
                    >
                      <button type="button" className="behavior-dial-label" onClick={() => setSelectedAxisName(axis.name)}>
                        <strong>{axis.name}</strong>
                        <span>{axis.calibrated ? "CALIBRATED" : axis.custom ? "CUSTOM" : axis.library ? "LIBRARY" : "UNCALIBRATED"}</span>
                      </button>
                      <div className="behavior-dial-control">
                        <div className="behavior-poles">
                          <span>{axis.poles[1]}</span>
                          <b>CURRENT {formatValue(axis.value)}</b>
                          <span>{axis.poles[0]}</span>
                        </div>
                        <input
                          type="range"
                          aria-label={`${axis.name}, ${axis.poles[1]} to ${axis.poles[0]}`}
                          min={-axis.max}
                          max={axis.max}
                          step=".05"
                          value={draft}
                          onChange={(event) => updateDraft(axis.name, Number(event.target.value))}
                        />
                      </div>
                      <output>{formatValue(draft)}</output>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        ) : view === "concepts" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">DIRECTIONAL INTERVENTION</span>
                <h1 id="behavior-console-title">Concept steering</h1>
              </div>
              <span className={`behavior-capability ${runtime.engine?.jlens ? "is-ready" : ""}`}>
                {runtime.engine?.jlens ? "J-LENS READY" : "J-LENS REQUIRED"}
              </span>
            </header>
            <div className="behavior-concept-stage">
              <section className="behavior-concept-form">
                <label>
                  <span>CONCEPT</span>
                  <input value={concept} onChange={(event) => setConcept(event.target.value)} placeholder="word or token" />
                </label>
                <label>
                  <span>STRENGTH</span>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step=".1"
                    value={conceptStrength}
                    onChange={(event) => setConceptStrength(Number(event.target.value))}
                  />
                  <output>{formatValue(conceptStrength)}</output>
                </label>
                <div className="behavior-concept-actions">
                  <button type="button" disabled={!runtime.engine?.jlens || !concept.trim() || operation.status === "pending"} onClick={() => void runConceptPreview()}>PREVIEW</button>
                  <button type="button" className="is-primary" disabled={!runtime.engine?.jlens || !concept.trim() || operation.status === "pending"} onClick={() => void commitConcept()}>APPLY CONCEPT</button>
                </div>
              </section>
              <section className="behavior-session-concepts">
                <header><span>SESSION-OBSERVED CONCEPTS</span><b>{Object.keys(activeConcepts).length}</b></header>
                {Object.entries(activeConcepts).map(([name, strength]) => (
                  <div key={name}>
                    <strong>{name}</strong>
                    <output>{formatValue(strength)}</output>
                    <button type="button" onClick={() => {
                      setConcept(name);
                      void commitConcept(0, name);
                    }}>REMOVE</button>
                  </div>
                ))}
                {!Object.keys(activeConcepts).length && <div className="behavior-empty-row">NONE OBSERVED</div>}
              </section>
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

          {view === "dials" && selectedAxis ? (
            <>
              <section className="behavior-selected-control">
                <header><span>SELECTED AXIS</span><b>{selectedAxis.calibrated ? "CALIBRATED" : "UNCALIBRATED"}</b></header>
                <strong>{selectedAxis.name}</strong>
                <div>
                  <span>{selectedAxis.poles[1]}</span>
                  <output>{formatValue(selectedDraft)}</output>
                  <span>{selectedAxis.poles[0]}</span>
                </div>
                <dl>
                  <div><dt>CURRENT</dt><dd>{formatValue(selectedAxis.value)}</dd></div>
                  <div><dt>DRAFT</dt><dd>{formatValue(selectedDraft)}</dd></div>
                  <div><dt>BOUND</dt><dd>±{selectedAxis.max.toFixed(2)}</dd></div>
                </dl>
                <button type="button" className="is-primary" disabled={!changed(selectedAxis.value, selectedDraft) || operation.status === "pending"} onClick={() => void commitAxis(selectedAxis)}>APPLY AXIS</button>
              </section>
              <section className="behavior-preview-control">
                <label><span>PREVIEW PROMPT</span><textarea value={previewPrompt} onChange={(event) => setPreviewPrompt(event.target.value)} /></label>
                <button type="button" disabled={!previewPrompt.trim() || operation.status === "pending"} onClick={() => void runAxisPreview()}>RUN A/B PREVIEW</button>
              </section>
              <section className="behavior-preview-output">
                <header><span>A/B OUTPUT</span><b>{axisPreview ? axisPreview.axis.toUpperCase() : "NOT RUN"}</b></header>
                {axisPreview ? (
                  <>
                    <div><span>BASELINE</span><p>{axisPreview.baseline}</p></div>
                    <div className="is-steered"><span>DRAFT VALUE</span><p>{axisPreview.steered}</p></div>
                  </>
                ) : <div className="behavior-empty-row">NO PREVIEW RESULT</div>}
              </section>
            </>
          ) : view === "concepts" ? (
            <>
              <section className="behavior-selected-control">
                <header><span>CONCEPT DIRECTION</span><b>{runtime.engine?.jlens ? "READY" : "UNAVAILABLE"}</b></header>
                <strong>{concept || "—"}</strong>
                <div><span>-2.00</span><output>{formatValue(conceptStrength)}</output><span>+2.00</span></div>
              </section>
              <section className="behavior-preview-control">
                <label><span>PREVIEW PROMPT</span><textarea value={previewPrompt} onChange={(event) => setPreviewPrompt(event.target.value)} /></label>
              </section>
              <section className="behavior-preview-output">
                <header><span>A/B OUTPUT</span><b>{conceptPreview ? conceptPreview.concept.toUpperCase() : "NOT RUN"}</b></header>
                {conceptPreview ? (
                  <>
                    <div><span>BASELINE</span><p>{conceptPreview.baseline}</p></div>
                    <div className="is-steered"><span>CONCEPT STEERED</span><p>{conceptPreview.steered}</p></div>
                  </>
                ) : <div className="behavior-empty-row">NO PREVIEW RESULT</div>}
              </section>
            </>
          ) : (
            <section className="behavior-scope-facts">
              <header><span>CURRENT SCOPE</span><b>{view.toUpperCase()}</b></header>
              <dl>
                <div><dt>MODEL</dt><dd>{basename(runtime.engine?.model)}</dd></div>
                <div><dt>PENDING DRAFTS</dt><dd>{pendingCount}</dd></div>
                <div><dt>ACTIVE DIALS</dt><dd>{activeAxes.length}</dd></div>
              </dl>
            </section>
          )}
        </aside>
      )}
    </>
  );
}
