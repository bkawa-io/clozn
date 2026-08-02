import { useEffect, useMemo, useRef, useState } from "react";
import {
  cancelInvestigationExperiment,
  describeInvestigationExperimentError,
  loadInvestigationExperimentJob,
  planInvestigationExperiment,
  startInvestigationExperiment,
  type ExperimentDocument,
  type ExperimentArm,
  type ExperimentIntervention,
  type ExperimentInterventionKind,
  type ExperimentJob,
} from "../../data/investigationExperiment";
import { loadSpanAddresses, type SpanAddress, type SpanAddressDocument } from "../../data/received-context";

export interface InvestigationExperimentProps {
  runId: string;
}

type Resource<T> =
  | { status: "idle" | "loading" }
  | { status: "failed"; message: string }
  | { status: "ready"; value: T };

type Target = { value: string; label: string; detail: string; spanAddressId?: string; sourceId?: string };

const CONTENT_KINDS: ExperimentInterventionKind[] = ["remove_span", "replace_span_neutral"];

function waitMs(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, ms);
    function abort() {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }
    signal.addEventListener("abort", abort, { once: true });
  });
}

function addressLabel(address: SpanAddress): string {
  return address.nativeRef.sourceLabel
    ?? address.nativeRef.clientSourceId
    ?? address.nativeRef.id
    ?? address.addressId.slice(-12);
}

function buildTargets(document: SpanAddressDocument | undefined, mode: ExperimentInterventionKind): Target[] {
  if (!document) return [];
  if (mode === "omit_source") {
    const seen = new Set<string>();
    return document.addresses.flatMap((address) => {
      const sourceId = address.nativeRef.clientSourceId;
      if (!sourceId || seen.has(sourceId)) return [];
      seen.add(sourceId);
      return [{ value: sourceId, sourceId, label: addressLabel(address), detail: `SOURCE ${sourceId}` }];
    });
  }
  return document.addresses
    .filter((address) => address.resolution.state === "exact")
    .filter((address) => address.nativeRef.collection?.includes("prompt") || address.nativeRef.collection?.includes("context"))
    .map((address) => ({
      value: address.addressId,
      spanAddressId: address.addressId,
      label: addressLabel(address),
      detail: `${address.kind.toUpperCase()} · ${address.addressId.slice(-12)}`,
    }));
}

function wireChange(mode: ExperimentInterventionKind, target: Target): ExperimentIntervention {
  return mode === "omit_source"
    ? { kind: mode, sourceId: target.sourceId }
    : { kind: mode, spanAddressId: target.spanAddressId };
}

function armLabel(label: string, arm: ExperimentArm | undefined) {
  if (!arm || arm.available === false) {
    return <li><b>{label}</b><span>UNAVAILABLE</span>{arm?.reason && <small>{arm.reason}</small>}</li>;
  }
  return (
    <li>
      <b>{label}</b>
      <span>{arm.matchesBaseline === undefined ? "RECORDED" : arm.matchesBaseline ? "MATCHES BASELINE" : "DIFFERS"}</span>
      {arm.runId && <a href={`#/runs/${encodeURIComponent(arm.runId)}`}>{arm.runId.slice(-12)}</a>}
    </li>
  );
}

function ExperimentResult({ document }: { document: ExperimentDocument }) {
  const causal = document.causalClaim;
  const observed = document.observed;
  return (
    <div className="investigation-experiment-result">
      {causal && (
        <section className={`investigation-experiment-claim is-${causal.licensed ? "licensed" : "unlicensed"}`}>
          <header><span>CAUSAL CLAIM</span><b>{causal.licensed ? "LICENSED" : "NOT LICENSED"}</b></header>
          <p>{causal.statement}</p>
        </section>
      )}
      {observed && (
        <section className="investigation-experiment-observed">
          <header><span>OBSERVED</span><b>{observed.treatmentReplyDiffersFromBaseline ? "TREATMENT DIFFERED" : "NO TREATMENT DIFFERENCE"}</b></header>
          {observed.note && <p>{observed.note}</p>}
        </section>
      )}
      {document.arms && (
        <ul className="investigation-experiment-arms" aria-label="Experiment arms">
          {armLabel("BASELINE", document.arms.baseline)}
          {armLabel("NO-OP REPLAY", document.arms.noOpReplay)}
          {armLabel("TREATMENT", document.arms.treatment)}
          {armLabel("RANDOM CONTROL", document.arms.randomEqualEffectControl)}
        </ul>
      )}
      {document.error && <p className="investigation-experiment-error">{document.error.message}</p>}
    </div>
  );
}

export function InvestigationExperiment({ runId }: InvestigationExperimentProps) {
  const [spanDocument, setSpanDocument] = useState<Resource<SpanAddressDocument>>({ status: "idle" });
  const [mode, setMode] = useState<ExperimentInterventionKind>("remove_span");
  const [targetValue, setTargetValue] = useState("");
  const [plan, setPlan] = useState<Resource<ExperimentDocument>>({ status: "idle" });
  const [job, setJob] = useState<ExperimentJob | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const actionController = useRef<AbortController | null>(null);
  const requestGeneration = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    requestGeneration.current += 1;
    actionController.current?.abort();
    actionController.current = null;
    setPlan({ status: "idle" });
    setJob(null);
    setActionError(null);
    setTargetValue("");
    if (!runId) {
      setSpanDocument({ status: "idle" });
      return () => controller.abort();
    }
    setSpanDocument({ status: "loading" });
    void loadSpanAddresses(runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setSpanDocument({ status: "ready", value });
    }).catch((error) => {
      if (!controller.signal.aborted) setSpanDocument({ status: "failed", message: describeInvestigationExperimentError(error) });
    });
    return () => controller.abort();
  }, [runId]);

  const targets = useMemo(
    () => buildTargets(spanDocument.status === "ready" ? spanDocument.value : undefined, mode),
    [spanDocument, mode],
  );

  useEffect(() => {
    setTargetValue(targets[0]?.value ?? "");
    setPlan({ status: "idle" });
    setJob(null);
    setActionError(null);
  }, [mode, targets]);

  const target = targets.find((entry) => entry.value === targetValue);
  const currentChange = target ? wireChange(mode, target) : undefined;

  async function makePlan() {
    if (!currentChange || !runId) return;
    const controller = new AbortController();
    actionController.current?.abort();
    actionController.current = controller;
    setPlan({ status: "loading" });
    setJob(null);
    setActionError(null);
    try {
      const document = await planInvestigationExperiment(runId, currentChange, controller.signal);
      if (!controller.signal.aborted) setPlan({ status: "ready", value: document });
    } catch (error) {
      if (!controller.signal.aborted) setPlan({ status: "failed", message: describeInvestigationExperimentError(error) });
    }
  }

  async function execute() {
    if (!currentChange || plan.status !== "ready" || plan.value.phase !== "planned") return;
    const generation = requestGeneration.current + 1;
    requestGeneration.current = generation;
    const controller = new AbortController();
    actionController.current?.abort();
    actionController.current = controller;
    setActionError(null);
    try {
      let next = await startInvestigationExperiment(runId, currentChange, controller.signal);
      if (generation !== requestGeneration.current) return;
      setJob(next);
      while (!["completed", "failed", "cancelled"].includes(next.state)) {
        await waitMs(250, controller.signal);
        next = await loadInvestigationExperimentJob(runId, next.jobId, controller.signal);
        if (generation !== requestGeneration.current) return;
        setJob(next);
      }
    } catch (error) {
      if (!controller.signal.aborted && generation === requestGeneration.current) {
        setActionError(describeInvestigationExperimentError(error));
      }
    }
  }

  async function cancel() {
    if (!job?.jobId || !job.cancellable) return;
    try {
      const cancelled = await cancelInvestigationExperiment(runId, job.jobId);
      setJob(cancelled);
      actionController.current?.abort();
    } catch (error) {
      setActionError(describeInvestigationExperimentError(error));
    }
  }

  const spanUnavailable = spanDocument.status === "ready" && targets.length === 0;
  return (
    <section className="investigation-experiment" aria-labelledby="investigation-experiment-title">
      <header className="investigation-experiment-head">
        <div><span className="eyebrow">CONTROLLED ACTION</span><h3 id="investigation-experiment-title">Did this matter?</h3></div>
        <span className="investigation-experiment-badge">PLAN FIRST · EXECUTE SECOND</span>
      </header>
      <p className="investigation-experiment-boundary">Choose one stable prompt/source address and ask for a controlled child experiment. Planning is model-free; execution creates child runs and never edits this run.</p>

      {spanDocument.status === "loading" && <p className="investigation-experiment-notice" role="status">Loading stable passage addresses…</p>}
      {spanDocument.status === "failed" && <p className="investigation-experiment-notice is-failed" role="alert">Stable passage addresses unavailable: {spanDocument.message}</p>}
      {spanUnavailable && <p className="investigation-experiment-notice" role="status">This run has no exact prompt/source addresses eligible for an on-demand experiment.</p>}

      {spanDocument.status === "ready" && !spanUnavailable && (
        <div className="investigation-experiment-picker">
          <label htmlFor="investigation-experiment-kind"><span>CHANGE</span>
            <select id="investigation-experiment-kind" value={mode} onChange={(event) => setMode(event.target.value as ExperimentInterventionKind)}>
              <option value="remove_span">Remove selected passage</option>
              <option value="replace_span_neutral">Replace selected passage with neutral marker</option>
              <option value="omit_source">Omit attached source</option>
            </select>
          </label>
          <label htmlFor="investigation-experiment-target"><span>PASSAGE / SOURCE</span>
            <select id="investigation-experiment-target" value={targetValue} onChange={(event) => setTargetValue(event.target.value)}>
              {targets.map((entry) => <option key={entry.value} value={entry.value}>{entry.label} · {entry.detail}</option>)}
            </select>
          </label>
          <button type="button" className="is-primary" disabled={!currentChange || plan.status === "loading"} onClick={() => void makePlan()}>PLAN EXPERIMENT</button>
        </div>
      )}

      {plan.status === "loading" && <p className="investigation-experiment-notice" role="status">Checking eligibility without generating…</p>}
      {plan.status === "failed" && <p className="investigation-experiment-notice is-failed" role="alert">Planning failed: {plan.message}</p>}
      {plan.status === "ready" && plan.value.phase === "refused" && <p className="investigation-experiment-notice is-refused" role="status"><b>{plan.value.eligibility.reason?.code ?? "NOT ELIGIBLE"}</b>{plan.value.eligibility.reason?.message ?? "This intervention cannot run for this recorded run."}</p>}
      {plan.status === "ready" && plan.value.phase === "planned" && (
        <section className="investigation-experiment-plan" aria-label="Experiment plan">
          <header><span>ELIGIBLE</span><b>{plan.value.plan?.armOrder.length ?? 0} ARMS</b></header>
          <p>This will create a no-op replay, a treatment child, and any available matched control. The parent stays unchanged.</p>
          <button type="button" className="is-primary" disabled={job?.state === "running" || job?.state === "queued"} onClick={() => void execute()}>RUN CONTROLLED EXPERIMENT</button>
        </section>
      )}

      {job && <section className="investigation-experiment-job" aria-live="polite">
        <header><span>JOB {job.jobId.slice(-10)}</span><b>{job.state.toUpperCase()}</b></header>
        {!['completed', 'failed', 'cancelled'].includes(job.state) && <p>{job.progress.phase.toUpperCase()} · {job.progress.percent}%</p>}
        {job.cancellable && <button type="button" onClick={() => void cancel()}>CANCEL</button>}
        {job.error && <p className="investigation-experiment-error">{job.error.message}</p>}
        {job.result && <ExperimentResult document={job.result} />}
      </section>}
      {actionError && <p className="investigation-experiment-notice is-failed" role="alert">{actionError}</p>}
    </section>
  );
}
