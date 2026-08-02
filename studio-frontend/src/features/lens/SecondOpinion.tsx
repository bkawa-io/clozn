import { useEffect, useRef, useState } from "react";
import {
  describeSecondOpinionError,
  loadSecondOpinionCandidates,
  runSecondOpinion,
  type SecondOpinionCandidates,
  type SecondOpinionComparison,
  type SecondOpinionDocument,
  type SecondOpinionArm,
} from "../../data/secondOpinion";

export interface SecondOpinionProps {
  runId: string;
}

type Resource<T> =
  | { status: "idle" | "loading" }
  | { status: "failed"; message: string }
  | { status: "ready"; value: T };

function ArmIdentity({ arm }: { arm: { modelId?: string; workerIdentity?: SecondOpinionArm["workerIdentity"] } }) {
  const identity = arm.workerIdentity;
  return (
    <dl className="second-opinion-identity">
      <div><dt>MODEL</dt><dd>{arm.modelId ?? "UNKNOWN"}</dd></div>
      {identity?.templateFingerprint && <div><dt>TEMPLATE</dt><dd>{identity.templateFingerprint}</dd></div>}
      {identity?.engineBuild && <div><dt>ENGINE</dt><dd>{identity.engineBuild}</dd></div>}
      {identity?.workerId && <div><dt>WORKER</dt><dd>{identity.workerId}</dd></div>}
    </dl>
  );
}

function Comparison({ comparison }: { comparison: SecondOpinionComparison }) {
  return (
    <section className="second-opinion-comparison" aria-label="Second opinion comparison">
      <header><span>COMPARISON</span><b>{comparison.agreement.lexicalDifferencePercent}% LEXICAL DIFFERENCE</b></header>
      <p>{comparison.agreement.caveat}</p>
      <dl>
        <div><dt>ANCHOR WORDS</dt><dd>{comparison.length.armAWords}</dd></div>
        <div><dt>SECOND WORDS</dt><dd>{comparison.length.armBWords}</dd></div>
        {comparison.formatChanged !== undefined && (
          <div><dt>FORMAT CHANGED</dt><dd>{comparison.formatChanged ? "YES" : "NO"}</dd></div>
        )}
      </dl>
    </section>
  );
}

function ResponseArm({
  label,
  arm,
}: {
  label: string;
  arm: {
    modelId?: string;
    status: string;
    responseText?: string;
    refusal?: { code: string; message: string };
    finishReason?: string;
    latencyMs?: number;
    workerIdentity?: SecondOpinionArm["workerIdentity"];
  };
}) {
  return (
    <article className={`second-opinion-arm is-${arm.status}`}>
      <header>
        <div><span>{label}</span><b>{arm.status.replaceAll("_", " ").toUpperCase()}</b></div>
        <ArmIdentity arm={arm} />
      </header>
      {arm.responseText !== undefined ? (
        <p className="second-opinion-response">{arm.responseText || "(empty response)"}</p>
      ) : arm.refusal ? (
        <p className="second-opinion-refusal"><b>{arm.refusal.code}</b>{arm.refusal.message}</p>
      ) : (
        <p className="second-opinion-refusal">No response text was recorded for this arm.</p>
      )}
      {(arm.finishReason || arm.latencyMs !== undefined) && (
        <footer>
          {arm.finishReason && <span>FINISH {arm.finishReason}</span>}
          {arm.latencyMs !== undefined && <span>{arm.latencyMs} MS</span>}
        </footer>
      )}
    </article>
  );
}

function Compatibility({ document }: { document: SecondOpinionDocument }) {
  const { chatTemplate, contextLimit, toolsOrSchema, qualifiedEvidence } = document.compatibility;
  return (
    <details className="second-opinion-compatibility">
      <summary>COMPATIBILITY AND EVIDENCE CAVEATS</summary>
      <ul>
        <li><b>CHAT TEMPLATE</b><span>{chatTemplate.state.replaceAll("_", " ").toUpperCase()}</span>{chatTemplate.caveat && <small>{chatTemplate.caveat}</small>}</li>
        <li><b>CONTEXT LIMIT</b><span>{contextLimit.state.replaceAll("_", " ").toUpperCase()}</span>{contextLimit.caveat && <small>{contextLimit.caveat}</small>}</li>
        <li><b>TOOLS / SCHEMA</b><span>{toolsOrSchema.state.replaceAll("_", " ").toUpperCase()}</span>{toolsOrSchema.caveat && <small>{toolsOrSchema.caveat}</small>}</li>
        <li><b>QUALIFIED EVIDENCE</b><span>ANCHOR ONLY</span><small>{qualifiedEvidence.note}</small></li>
      </ul>
    </details>
  );
}

function CandidatePicker({
  candidates,
  selectedModel,
  onChange,
  onRun,
  running,
}: {
  candidates: SecondOpinionCandidates;
  selectedModel: string;
  onChange: (modelId: string) => void;
  onRun: () => void;
  running: boolean;
}) {
  const ready = candidates.candidates.filter((candidate) => candidate.ready);
  return (
    <div className="second-opinion-picker">
      <label htmlFor="second-opinion-model"><span>RESIDENT MODEL</span>
        <select id="second-opinion-model" value={selectedModel} onChange={(event) => onChange(event.target.value)} disabled={running}>
          <option value="">Choose a different resident model</option>
          {candidates.candidates.map((candidate) => (
            <option key={candidate.modelId} value={candidate.modelId} disabled={!candidate.ready}>
              {candidate.modelId}{candidate.ready ? "" : " (not ready)"}
            </option>
          ))}
        </select>
      </label>
      <button type="button" className="is-primary" disabled={!selectedModel || running || ready.length === 0} onClick={onRun}>
        {running ? "ASKING…" : "ASK SECOND OPINION"}
      </button>
    </div>
  );
}

export function SecondOpinion({ runId }: SecondOpinionProps) {
  const [candidates, setCandidates] = useState<Resource<SecondOpinionCandidates>>({ status: "idle" });
  const [selectedModel, setSelectedModel] = useState("");
  const [result, setResult] = useState<Resource<SecondOpinionDocument>>({ status: "idle" });
  const actionRequestRef = useRef(0);
  const actionAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    actionRequestRef.current += 1;
    actionAbortRef.current?.abort();
    actionAbortRef.current = null;
    setSelectedModel("");
    setResult({ status: "idle" });
    if (!runId) {
      setCandidates({ status: "idle" });
      return () => {
        controller.abort();
        actionRequestRef.current += 1;
        actionAbortRef.current?.abort();
      };
    }
    setCandidates({ status: "loading" });
    void loadSecondOpinionCandidates(runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setCandidates({ status: "ready", value });
    }).catch((error) => {
      if (!controller.signal.aborted) setCandidates({ status: "failed", message: describeSecondOpinionError(error) });
    });
    return () => {
      controller.abort();
      actionRequestRef.current += 1;
      actionAbortRef.current?.abort();
    };
  }, [runId]);

  function ask() {
    if (!selectedModel || candidates.status !== "ready") return;
    const currentRequest = actionRequestRef.current + 1;
    actionRequestRef.current = currentRequest;
    actionAbortRef.current?.abort();
    const controller = new AbortController();
    actionAbortRef.current = controller;
    setResult({ status: "loading" });
    void runSecondOpinion(runId, selectedModel, controller.signal).then((value) => {
      if (currentRequest === actionRequestRef.current) setResult({ status: "ready", value });
    }).catch((error) => {
      if (currentRequest === actionRequestRef.current) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setResult({ status: "failed", message: describeSecondOpinionError(error) });
      }
    });
  }

  const candidateValue = candidates.status === "ready" ? candidates.value : undefined;
  const readyCount = candidateValue?.candidates.filter((candidate) => candidate.ready).length ?? 0;

  return (
    <section className="second-opinion" aria-labelledby="second-opinion-title">
      <header className="second-opinion-head">
        <div><span className="eyebrow">TRUST / COMPARISON</span><h3 id="second-opinion-title">Would another model disagree?</h3></div>
        <span className="second-opinion-badge">{candidateValue?.managed ? `${readyCount} RESIDENT` : "EXPLICIT RUN ONLY"}</span>
      </header>
      <p className="second-opinion-boundary">The original answer stays the anchor. Choose another already-resident model and press the button to run one fresh answer against the same delivered messages.</p>

      {candidates.status === "loading" && <p className="second-opinion-notice" role="status">Checking for another resident model…</p>}
      {candidates.status === "failed" && <p className="second-opinion-notice is-failed" role="alert">Could not check second-opinion availability: {candidates.message}</p>}
      {candidates.status === "ready" && !candidates.value.managed && <p className="second-opinion-notice" role="status">A second opinion needs a managed gateway with another resident model. This gateway is serving one model only.</p>}
      {candidates.status === "ready" && candidates.value.managed && readyCount === 0 && <p className="second-opinion-notice" role="status">No different resident model is ready. Loading a model is not started by this panel.</p>}
      {candidates.status === "ready" && candidates.value.managed && readyCount > 0 && (
        <CandidatePicker candidates={candidates.value} selectedModel={selectedModel} onChange={setSelectedModel} onRun={ask} running={result.status === "loading"} />
      )}

      {result.status === "loading" && <p className="second-opinion-notice" role="status">Generating the second answer…</p>}
      {result.status === "failed" && <p className="second-opinion-notice is-failed" role="alert">The request could not be started: {result.message}</p>}
      {result.status === "ready" && (
        <div className="second-opinion-result">
          <div className="second-opinion-arms">
            <ResponseArm label="ANCHOR / ORIGINAL RUN" arm={result.value.armA} />
            <ResponseArm label="SECOND OPINION" arm={result.value.armB} />
          </div>
          {result.value.comparison && <Comparison comparison={result.value.comparison} />}
          <Compatibility document={result.value} />
          <p className="second-opinion-receipt">Same delivered input: {result.value.deliveredInput.identicalAcrossArms ? "YES" : "UNKNOWN"} · {result.value.deliveredInput.messageCount} messages · no cross-model token probabilities are shown.</p>
        </div>
      )}
    </section>
  );
}
