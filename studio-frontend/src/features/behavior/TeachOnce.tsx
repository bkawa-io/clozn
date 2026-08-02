import { useEffect, useState } from "react";
import {
  confirmCorrection,
  disableCorrection,
  draftCorrection,
  enableCorrection,
  loadCorrections,
  undoCorrection,
  verifyCorrection,
  type Correction,
  type CorrectionMatchCriterion,
  type CorrectionScopeKind,
  type CorrectionType,
} from "../../data/corrections";

const scopes: Array<{ value: CorrectionScopeKind; label: string }> = [
  { value: "session", label: "THIS SESSION" },
  { value: "client", label: "THIS CLIENT" },
  { value: "project", label: "THIS PROJECT" },
  { value: "model", label: "THIS MODEL (SHA-256)" },
  { value: "global_local", label: "LOCAL DEFAULT" },
];

const types: Array<{ value: CorrectionType; label: string }> = [
  { value: "output_format", label: "OUTPUT FORMAT" },
  { value: "source_requirement", label: "SOURCE REQUIREMENT" },
  { value: "style", label: "STYLE" },
  { value: "forbidden_behavior", label: "FORBIDDEN BEHAVIOR" },
];

const criteria: Array<{ value: CorrectionMatchCriterion; label: string }> = [
  { value: "exact_output", label: "EXACT OUTPUT" },
  { value: "finish_reason", label: "FINISH REASON" },
  { value: "tool_parse", label: "TOOL PARSE" },
  { value: "token_budget", label: "TOKEN BUDGET" },
];

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error || "Operation failed");
}

function scopeText(correction: Correction) {
  return correction.scope.value
    ? `${correction.scope.kind} · ${correction.scope.value}`
    : correction.scope.kind;
}

type VerificationDraft = {
  targetRunId: string;
  childRunId: string;
  criterion: CorrectionMatchCriterion;
};

export function TeachOnce() {
  const [corrections, setCorrections] = useState<Correction[]>([]);
  const [scope, setScope] = useState<CorrectionScopeKind>("session");
  const [scopeValue, setScopeValue] = useState("");
  const [type, setType] = useState<CorrectionType>("style");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("READING CORRECTIONS");
  const [error, setError] = useState("");
  const [verificationDrafts, setVerificationDrafts] = useState<Record<string, VerificationDraft>>({});

  async function refresh() {
    const next = await loadCorrections();
    setCorrections(next.corrections);
  }

  useEffect(() => {
    const controller = new AbortController();
    void loadCorrections(controller.signal).then((next) => {
      if (!controller.signal.aborted) {
        setCorrections(next.corrections);
        setStatus("READY");
      }
    }).catch((reason) => {
      if (!controller.signal.aborted) {
        setStatus("UNAVAILABLE");
        setError(errorMessage(reason));
      }
    });
    return () => controller.abort();
  }, []);

  async function run(action: string, fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    setStatus(action);
    try {
      await fn();
      await refresh();
      setStatus("UPDATED");
    } catch (reason) {
      setStatus("FAILED");
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function draft() {
    const trimmed = content.trim();
    if (!trimmed || (scope !== "global_local" && !scopeValue.trim())) return;
    await run("DRAFTING · CONFIRM REQUIRED", async () => {
      await draftCorrection(scope, scope === "global_local" ? undefined : scopeValue.trim(), type, trimmed);
      setContent("");
    });
  }

  function verificationDraft(id: string): VerificationDraft {
    return verificationDrafts[id] ?? { targetRunId: "", childRunId: "", criterion: "exact_output" };
  }

  function updateVerification(id: string, patch: Partial<VerificationDraft>) {
    setVerificationDrafts((current) => {
      const previous = current[id] ?? { targetRunId: "", childRunId: "", criterion: "exact_output" as const };
      return { ...current, [id]: { ...previous, ...patch } };
    });
  }

  async function verify(correction: Correction) {
    const draft = verificationDraft(correction.id);
    if (!draft.targetRunId.trim() || !draft.childRunId.trim()) return;
    await run("VERIFYING Â· COMPARE FIRST", async () => {
      const result = await verifyCorrection(
        correction.id,
        draft.targetRunId.trim(),
        draft.childRunId.trim(),
        draft.criterion,
      );
      if (!result.promoted) throw new Error(result.reason);
    });
  }

  return (
    <div className="behavior-teach-stage">
      <header className="instrument-head behavior-console-head">
        <div>
          <span className="eyebrow">EXPLICIT · REVERSIBLE · RECEIPTED</span>
          <h1 id="behavior-console-title">Teach Once</h1>
        </div>
        <div className="behavior-head-stats"><span><b>SAVED</b>{corrections.length}</span><span><b>STATE</b>{status}</span></div>
      </header>
      <p className="behavior-teach-note">
        A correction is inert until you confirm it. Confirmed corrections are scoped exactly, appear in
        later run receipts, and never change model weights.
      </p>
      <section className="behavior-teach-draft">
        <header><span>NEW CORRECTION</span><b>DRAFT ONLY</b></header>
        <div className="behavior-teach-fields">
          <label><span>SCOPE</span><select value={scope} onChange={(event) => setScope(event.target.value as CorrectionScopeKind)}>
            {scopes.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
          </select></label>
          {scope !== "global_local" && <label><span>SCOPE VALUE</span><input value={scopeValue} onChange={(event) => setScopeValue(event.target.value)} placeholder={scope === "model" ? "64-hex model_sha256" : "explicit identifier"} /></label>}
          <label><span>TYPE</span><select value={type} onChange={(event) => setType(event.target.value as CorrectionType)}>
            {types.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
          </select></label>
        </div>
        <label className="behavior-teach-content"><span>USER-APPROVED INSTRUCTION</span><textarea value={content} onChange={(event) => setContent(event.target.value)} placeholder="e.g. Answer in short paragraphs." /></label>
        <button type="button" className="is-primary" disabled={busy || !content.trim() || (scope !== "global_local" && !scopeValue.trim())} onClick={() => void draft()}>SAVE DRAFT</button>
      </section>
      {error && <div className="behavior-unavailable">{error}</div>}
      <section className="behavior-teach-list" aria-label="Saved corrections">
        {corrections.map((correction) => (
          <article className={correction.enabled ? "is-enabled" : ""} key={correction.id}>
            <header><div><strong>{correction.type.replaceAll("_", " ")}</strong><span>{scopeText(correction)}</span></div><b>{correction.enabled ? "CONFIRMED" : correction.confirmed_ts ? "DISABLED" : "DRAFT"}</b></header>
            <p>{correction.content ?? "CONTENT REDACTED"}</p>
            {!correction.confirmed_ts && (
              <div className="behavior-teach-verify">
                <header><span>VERIFY BEFORE SAVE</span><b>CHILD RETRY REQUIRED</b></header>
                <div className="behavior-teach-verify-fields">
                  <label><span>TARGET FAILURE RUN ID</span><input value={verificationDraft(correction.id).targetRunId} onChange={(event) => updateVerification(correction.id, { targetRunId: event.target.value })} placeholder="run_…" /></label>
                  <label><span>CHILD RETRY RUN ID</span><input value={verificationDraft(correction.id).childRunId} onChange={(event) => updateVerification(correction.id, { childRunId: event.target.value })} placeholder="run_…" /></label>
                  <label><span>COMPARISON</span><select value={verificationDraft(correction.id).criterion} onChange={(event) => updateVerification(correction.id, { criterion: event.target.value as CorrectionMatchCriterion })}>
                    {criteria.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
                  </select></label>
                </div>
                <button type="button" disabled={busy || !verificationDraft(correction.id).targetRunId.trim() || !verificationDraft(correction.id).childRunId.trim()} onClick={() => void verify(correction)}>VERIFY + PROMOTE</button>
              </div>
            )}
            <footer>
              {!correction.confirmed_ts && <button type="button" disabled={busy} onClick={() => void run("CONFIRMING", () => confirmCorrection(correction.id))}>CONFIRM</button>}
              {correction.enabled && <button type="button" disabled={busy} onClick={() => void run("DISABLING", () => disableCorrection(correction.id))}>DISABLE</button>}
              {!correction.enabled && correction.confirmed_ts && <button type="button" disabled={busy} onClick={() => void run("ENABLING", () => enableCorrection(correction.id))}>ENABLE</button>}
              {correction.confirmed_ts && <button type="button" disabled={busy} onClick={() => void run("UNDOING", () => undoCorrection(correction.id))}>UNDO LAST CHANGE</button>}
            </footer>
          </article>
        ))}
        {!corrections.length && <div className="behavior-empty-row">NO SAVED CORRECTIONS</div>}
      </section>
    </div>
  );
}
