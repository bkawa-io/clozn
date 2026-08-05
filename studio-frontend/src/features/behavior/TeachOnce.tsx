import { useEffect, useMemo, useRef, useState } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import { PairedDelta, type PairedDeltaRow } from "../../components/PairedDelta";
import { TypedActionOffer } from "../../components/TypedActionOffer";
import type { RunSummary } from "../../data/types";
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
  type CorrectionScope,
  type CorrectionScopeKind,
  type CorrectionType,
  type CorrectionVerification,
} from "../../data/corrections";

/**
 * The visual order is intentionally a containment trail rather than the order in which a picker happens
 * to list values. A correction only ever matches an explicit scope key, so this communicates reach
 * without pretending that the store performs relevance matching on its content.
 */
const scopeHierarchy: ReadonlyArray<{
  value: CorrectionScopeKind;
  label: string;
  description: string;
}> = [
  { value: "session", label: "SESSION", description: "one declared conversation" },
  { value: "client", label: "CLIENT", description: "one declared client identity" },
  { value: "model", label: "MODEL", description: "one model SHA-256" },
  { value: "project", label: "PROJECT", description: "one declared project" },
  { value: "global_local", label: "GLOBAL", description: "the local install default" },
];

const correctionTypes: Array<{ value: CorrectionType; label: string }> = [
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

const correctionScopeKinds: readonly CorrectionScopeKind[] = [
  "session",
  "client",
  "model",
  "project",
  "global_local",
];

const correctionTypesByValue: readonly CorrectionType[] = [
  "output_format",
  "source_requirement",
  "style",
  "forbidden_behavior",
];

const NO_RECORDED_RUNS: readonly Pick<RunSummary, "id" | "label">[] = [];

type VerificationDraft = {
  targetRunId: string;
  childRunId: string;
  criterion: CorrectionMatchCriterion;
};

type CorrectionLifecycle = "draft" | "enabled" | "disabled";

interface CorrectionState {
  lifecycle: CorrectionLifecycle;
  label: string;
  confirmation: "CONFIRMED" | "NOT CONFIRMED";
  note: string;
}

interface RecordedResolutionApplied {
  correctionId: string;
  type: CorrectionType;
  scope: CorrectionScope;
  contentHash: string;
}

interface RecordedResolutionConflict {
  type: CorrectionType;
  winnerId: string;
  losingIds: string[];
  rule: string;
}

interface RecordedCorrectionResolution {
  applied: RecordedResolutionApplied[];
  conflicts: RecordedResolutionConflict[];
}

type ResolutionState =
  | { status: "idle"; reason: string }
  | { status: "loading"; runId: string }
  | { status: "available"; runId: string; resolution: RecordedCorrectionResolution }
  | { status: "not_measured"; runId: string; reason: string }
  | { status: "unavailable"; runId: string; reason: string };

export interface TeachOnceProps {
  /** Existing run summaries only populate pickers; verification still asks the server to validate them. */
  runs?: readonly Pick<RunSummary, "id" | "label">[];
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error || "Operation failed");
}

function readableKind(value: string) {
  return value.replaceAll("_", " ");
}

function scopeDetails(scope: CorrectionScope) {
  return scopeHierarchy.find((item) => item.value === scope.kind)
    ?? { value: scope.kind, label: readableKind(scope.kind).toUpperCase(), description: "declared scope" };
}

function scopeText(scope: CorrectionScope) {
  const detail = scopeDetails(scope);
  return scope.value ? `${detail.label} · ${scope.value}` : detail.label;
}

function correctionState(correction: Correction): CorrectionState {
  if (correction.confirmed_ts == null) {
    return {
      lifecycle: "draft",
      label: "DRAFT",
      confirmation: "NOT CONFIRMED",
      note: "Inert until a user confirms this exact content.",
    };
  }
  if (correction.enabled) {
    return {
      lifecycle: "enabled",
      label: "ENABLED",
      confirmation: "CONFIRMED",
      note: "Confirmed and eligible when its declared scope matches.",
    };
  }
  return {
    lifecycle: "disabled",
    label: "DISABLED",
    confirmation: "CONFIRMED",
    note: "Confirmed history remains, but this correction is currently ineligible.",
  };
}

function timestampText(timestamp: number | undefined, fallback = "NOT RECORDED") {
  if (typeof timestamp !== "number" || !Number.isFinite(timestamp)) return fallback;
  const date = new Date(timestamp * 1000);
  if (Number.isNaN(date.getTime())) return fallback;
  return date.toISOString().replace("T", " ").replace(".000Z", "Z");
}

function correctionCriterionLabel(criterion: CorrectionMatchCriterion) {
  return criteria.find((item) => item.value === criterion)?.label ?? readableKind(criterion).toUpperCase();
}

function isCorrectionScopeKind(value: unknown): value is CorrectionScopeKind {
  return typeof value === "string" && correctionScopeKinds.includes(value as CorrectionScopeKind);
}

function isCorrectionType(value: unknown): value is CorrectionType {
  return typeof value === "string" && correctionTypesByValue.includes(value as CorrectionType);
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function parseScope(value: unknown): CorrectionScope | undefined {
  const scope = record(value);
  if (!isCorrectionScopeKind(scope.kind)) return undefined;
  const scopeValue = typeof scope.value === "string" && scope.value ? scope.value : undefined;
  // Global corrections deliberately carry no synthetic value. A fabricated "global" value would turn
  // exact scope matching into a different claim than the persisted record makes.
  if (scope.kind === "global_local") return scopeValue ? undefined : { kind: scope.kind };
  return scopeValue ? { kind: scope.kind, value: scopeValue } : undefined;
}

function parseRecordedResolution(value: unknown):
  | { status: "available"; resolution: RecordedCorrectionResolution }
  | { status: "not_measured"; reason: string }
  | { status: "unavailable"; reason: string } {
  const run = record(value);
  const hasApplied = Object.prototype.hasOwnProperty.call(run, "applied_corrections");
  const hasConflicts = Object.prototype.hasOwnProperty.call(run, "correction_conflicts");

  if (!hasApplied && !hasConflicts) {
    return {
      status: "not_measured",
      reason: "This recorded run has no correction-resolution fields. Resolution was not computed for it.",
    };
  }
  if (!Array.isArray(run.applied_corrections) || !Array.isArray(run.correction_conflicts)) {
    return {
      status: "unavailable",
      reason: "This recorded run carries an incomplete correction-resolution record.",
    };
  }

  const applied = run.applied_corrections.map((item) => {
    const entry = record(item);
    const scope = parseScope(entry.scope);
    if (
      typeof entry.correction_id !== "string"
      || !isCorrectionType(entry.type)
      || !scope
      || typeof entry.content_hash !== "string"
    ) return undefined;
    return {
      correctionId: entry.correction_id,
      type: entry.type,
      scope,
      contentHash: entry.content_hash,
    };
  });
  const conflicts = run.correction_conflicts.map((item) => {
    const entry = record(item);
    const losingIds = Array.isArray(entry.losing_ids)
      ? entry.losing_ids.filter((id): id is string => typeof id === "string" && Boolean(id))
      : [];
    if (
      !isCorrectionType(entry.type)
      || typeof entry.winner_id !== "string"
      || typeof entry.rule !== "string"
      || losingIds.length !== (Array.isArray(entry.losing_ids) ? entry.losing_ids.length : -1)
    ) return undefined;
    return {
      type: entry.type,
      winnerId: entry.winner_id,
      losingIds,
      rule: entry.rule,
    };
  });

  if (applied.some((entry) => entry == null) || conflicts.some((entry) => entry == null)) {
    return {
      status: "unavailable",
      reason: "This recorded run's correction-resolution fields could not be read without guessing.",
    };
  }

  return {
    status: "available",
    resolution: {
      applied: applied.filter((entry): entry is RecordedResolutionApplied => entry != null),
      conflicts: conflicts.filter((entry): entry is RecordedResolutionConflict => entry != null),
    },
  };
}

async function readRecordedResolution(runId: string) {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) throw new Error(`Recorded run request failed (${response.status})`);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error("Recorded run did not return a readable correction-resolution record.");
  }
  return parseRecordedResolution(body);
}

function verificationRows(verification: CorrectionVerification): PairedDeltaRow[] {
  const rows: PairedDeltaRow[] = [
    {
      id: "recorded-run-pair",
      dimension: "Persisted run pair",
      kind: "changed",
      rank: 1,
      valueA: verification.target_run_id,
      valueB: verification.child_run_id,
      note: "These are recorded immutable runs supplied to the comparison; no retry was generated here.",
    },
  ];

  if (!verification.comparison.available) {
    rows.push({
      id: "comparison-evidence",
      dimension: `${correctionCriterionLabel(verification.match_criterion)} comparison`,
      kind: "unavailable",
      rank: 2,
      reason: verification.comparison.reason
        ?? verification.reason
        ?? "The verification receipt did not record comparison evidence.",
    });
    return rows;
  }

  rows.push({
    id: "comparison-evidence",
    dimension: `${correctionCriterionLabel(verification.match_criterion)} comparison`,
    kind: verification.comparison.matched ? "unchanged" : "changed",
    rank: 2,
    valueA: "Target failure",
    valueB: verification.comparison.matched ? "Failure reproduced" : "Failure not reproduced",
    note: verification.reason,
  });
  return rows;
}

function VerificationComparison({ verification }: { verification: CorrectionVerification }) {
  const comparisonStatus = verification.comparison.available
    ? verification.comparison.matched ? "unchanged" : "changed"
    : "unavailable";

  return (
    <section className="behavior-correction-comparison" aria-label="Recorded run verification">
      <header>
        <div>
          <span>RECORDED RUN COMPARISON</span>
          <strong>{verification.verification.toUpperCase()}</strong>
        </div>
        <b>{verification.promoted ? "PROMOTED" : "NOT PROMOTED"}</b>
      </header>
      <p>
        This comparison reads two already-persisted runs. It did not generate a retry or run the model.
      </p>
      <PairedDelta
        title="Verification comparison"
        aLabel="Target (persisted)"
        bLabel="Child (persisted)"
        rows={verificationRows(verification)}
        summaryAxes={[
          {
            id: "run-identities",
            label: "Run identities",
            status: "changed",
            note: "Two distinct persisted run IDs supplied by the user.",
          },
          {
            id: "criterion",
            label: "Criterion",
            status: comparisonStatus,
            note: correctionCriterionLabel(verification.match_criterion),
          },
        ]}
        className="behavior-correction-paired-delta"
      />
    </section>
  );
}

function ScopeHierarchy({ selected }: { selected: CorrectionScopeKind }) {
  return (
    <section className="behavior-scope-hierarchy" aria-labelledby="behavior-scope-hierarchy-title">
      <header>
        <span id="behavior-scope-hierarchy-title">SCOPE CONTAINMENT</span>
        <b>SESSION ⊂ CLIENT ⊂ MODEL ⊂ PROJECT ⊂ GLOBAL</b>
      </header>
      <ol>
        {scopeHierarchy.map((item, index) => (
          <li
            className={item.value === selected ? "is-selected" : undefined}
            aria-current={item.value === selected ? "step" : undefined}
            data-scope={item.value}
            key={item.value}
          >
            <span className="behavior-scope-index" aria-hidden="true">{index + 1}</span>
            <div>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </div>
          </li>
        ))}
      </ol>
      <p>Each outer level contains the level before it; the correction still applies only when its exact declared key matches.</p>
    </section>
  );
}

function CorrectionResolution({
  runs,
  selectedRunId,
  onSelectedRunIdChange,
  state,
  onRead,
}: {
  runs: readonly Pick<RunSummary, "id" | "label">[];
  selectedRunId: string;
  onSelectedRunIdChange: (runId: string) => void;
  state: ResolutionState;
  onRead: () => void;
}) {
  const availability = state.status === "available";
  const reason = state.status === "not_measured" || state.status === "unavailable"
    ? state.reason
    : selectedRunId
      ? "A correction resolution has not been read for this recorded run yet."
      : "No recorded run is available to inspect for correction resolution.";
  const absenceState = state.status === "unavailable" ? "unavailable" as const : "not_measured" as const;

  return (
    <section className="behavior-correction-resolution" aria-labelledby="behavior-correction-resolution-title">
      <header>
        <div>
          <span>APPLIED CORRECTIONS</span>
          <h2 id="behavior-correction-resolution-title">Resolution on a recorded run</h2>
        </div>
        <span className={`behavior-resolution-state is-${state.status}`}>
          {state.status === "available" ? "RECORDED" : state.status.replaceAll("_", " ").toUpperCase()}
        </span>
      </header>
      <p className="behavior-correction-resolution-note">
        This is a stored run receipt, not a projection from the current correction list.
      </p>
      <label className="behavior-resolution-run">
        <span>RECORDED RUN</span>
        <select
          value={selectedRunId}
          disabled={state.status === "loading"}
          onChange={(event) => onSelectedRunIdChange(event.target.value)}
          aria-label="Recorded run for correction resolution"
        >
          {!runs.length && <option value="">NO RECORDED RUNS AVAILABLE</option>}
          {runs.map((run) => <option value={run.id} key={run.id}>{run.label}</option>)}
        </select>
      </label>

      {state.status === "loading" ? (
        <div className="behavior-resolution-loading">READING RECORDED RESOLUTION</div>
      ) : availability ? (
        <div className="behavior-resolution-report">
          <section aria-label="Applied corrections recorded on this run">
            <header><span>APPLIED</span><b>{state.resolution.applied.length}</b></header>
            {state.resolution.applied.length ? (
              <ul>
                {state.resolution.applied.map((entry) => (
                  <li key={entry.correctionId}>
                    <strong>{readableKind(entry.type)}</strong>
                    <span>{scopeText(entry.scope)}</span>
                    <code>{entry.contentHash}</code>
                    <small>{entry.correctionId}</small>
                  </li>
                ))}
              </ul>
            ) : <p>Resolution was recorded: no corrections matched this run.</p>}
          </section>
          <section aria-label="Correction conflicts recorded on this run">
            <header><span>CONFLICTS</span><b>{state.resolution.conflicts.length}</b></header>
            {state.resolution.conflicts.length ? (
              <ul>
                {state.resolution.conflicts.map((conflict) => (
                  <li key={`${conflict.type}-${conflict.winnerId}`}>
                    <strong>{readableKind(conflict.type)}</strong>
                    <span>Winner {conflict.winnerId}</span>
                    <span>Lost {conflict.losingIds.join(", ")}</span>
                    <small>Rule: {readableKind(conflict.rule)}</small>
                  </li>
                ))}
              </ul>
            ) : <p>Resolution was recorded: no conflicts were present.</p>}
          </section>
        </div>
      ) : (
        <TypedActionOffer
          title="Correction resolution is not available yet"
          absence={{ state: absenceState, label: "Resolution absent", reason }}
          cost="Reads one immutable local run record. It does not run a model or mutate a correction."
          preconditions={selectedRunId
            ? ["A recorded run is selected.", "The run must carry stored correction-resolution fields."]
            : ["A recorded run is required before resolution can be read."]}
          action={selectedRunId
            ? { availability: "available", label: "READ RECORDED RESOLUTION", onAction: onRead }
            : {
                availability: "blocked",
                label: "READ RECORDED RESOLUTION",
                blockerReason: "No recorded run is available.",
              }}
          className="behavior-resolution-offer"
        />
      )}
    </section>
  );
}

export function TeachOnce({ runs = NO_RECORDED_RUNS }: TeachOnceProps) {
  const [corrections, setCorrections] = useState<Correction[]>([]);
  const [scope, setScope] = useState<CorrectionScopeKind>("session");
  const [scopeValue, setScopeValue] = useState("");
  const [type, setType] = useState<CorrectionType>("style");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("READING CORRECTIONS");
  const [error, setError] = useState("");
  const [verificationDrafts, setVerificationDrafts] = useState<Record<string, VerificationDraft>>({});
  const [verificationResults, setVerificationResults] = useState<Record<string, CorrectionVerification>>({});
  const [resolutionRunId, setResolutionRunId] = useState(runs[0]?.id ?? "");
  const [resolutionState, setResolutionState] = useState<ResolutionState>({
    status: "idle",
    reason: "No correction resolution has been read yet.",
  });
  // A read receipt belongs to the selected immutable run. Ignore a late reply rather than relabelling
  // evidence for a different run if the picker changes while the request is in flight.
  const resolutionReadVersion = useRef(0);

  const sortedCorrections = useMemo(
    () => [...corrections].sort((left, right) => right.created_ts - left.created_ts),
    [corrections],
  );
  const enabledCount = corrections.filter((correction) => correction.confirmed_ts != null && correction.enabled).length;

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

  useEffect(() => {
    if (resolutionRunId && runs.some((run) => run.id === resolutionRunId)) return;
    resolutionReadVersion.current += 1;
    setResolutionRunId(runs[0]?.id ?? "");
    setResolutionState({ status: "idle", reason: "No correction resolution has been read yet." });
  }, [resolutionRunId, runs]);

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
    const targetRunId = draft.targetRunId.trim();
    const childRunId = draft.childRunId.trim();
    if (!targetRunId || !childRunId || targetRunId === childRunId) return;
    setBusy(true);
    setError("");
    setStatus("COMPARING RECORDED RUNS");
    try {
      // The server validates that both identifiers resolve to already-persisted runs before it compares
      // them. This handler never creates a retry; promotion is the only possible mutation after a pass.
      const result = await verifyCorrection(
        correction.id,
        targetRunId,
        childRunId,
        draft.criterion,
      );
      setVerificationResults((current) => ({ ...current, [correction.id]: result }));
      setCorrections((current) => current.map((item) => item.id === result.correction.id ? result.correction : item));
      try {
        await refresh();
      } catch (refreshError) {
        // The immutable comparison receipt remains useful even if a subsequent list refresh fails.
        setError(`Comparison was recorded, but corrections could not be reloaded: ${errorMessage(refreshError)}`);
      }
      setStatus(result.promoted ? "COMPARISON PASSED · PROMOTED" : "COMPARISON RECORDED · NOT PROMOTED");
    } catch (reason) {
      setStatus("COMPARISON FAILED");
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  function selectResolutionRun(runId: string) {
    resolutionReadVersion.current += 1;
    setResolutionRunId(runId);
    setResolutionState({ status: "idle", reason: "No correction resolution has been read for this run yet." });
  }

  async function readResolution() {
    const runId = resolutionRunId.trim();
    if (!runId) return;
    const requestVersion = resolutionReadVersion.current + 1;
    resolutionReadVersion.current = requestVersion;
    setResolutionState({ status: "loading", runId });
    try {
      const result = await readRecordedResolution(runId);
      if (requestVersion !== resolutionReadVersion.current) return;
      switch (result.status) {
        case "available":
          setResolutionState({ status: "available", runId, resolution: result.resolution });
          break;
        case "not_measured":
          setResolutionState({ status: "not_measured", runId, reason: result.reason });
          break;
        case "unavailable":
          setResolutionState({ status: "unavailable", runId, reason: result.reason });
          break;
        default: {
          const exhaustive: never = result;
          setResolutionState(exhaustive);
        }
      }
    } catch (reason) {
      if (requestVersion !== resolutionReadVersion.current) return;
      setResolutionState({ status: "unavailable", runId, reason: errorMessage(reason) });
    }
  }

  return (
    <div className="behavior-teach-stage">
      <header className="instrument-head behavior-console-head">
        <div>
          <span className="eyebrow">DURABLE · EXPLICIT · REVERSIBLE</span>
          <h1 id="behavior-console-title">Corrections</h1>
        </div>
        <div className="behavior-head-stats">
          <span><b>SAVED</b>{corrections.length}</span>
          <span><b>ENABLED</b>{enabledCount}</span>
          <span><b>STATE</b>{status}</span>
        </div>
      </header>
      <p className="behavior-teach-note">
        Corrections are explicit scoped context, never model training. A draft is inert until confirmed;
        enable, disable, and undo remain user-only reversible changes.
      </p>

      <section className="behavior-teach-draft" aria-labelledby="behavior-new-correction-title">
        <header><span id="behavior-new-correction-title">NEW CORRECTION</span><b>DRAFT ONLY</b></header>
        <div className="behavior-teach-fields">
          <label><span>SCOPE</span><select value={scope} onChange={(event) => setScope(event.target.value as CorrectionScopeKind)}>
            {scopeHierarchy.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
          </select></label>
          {scope !== "global_local" && <label><span>SCOPE VALUE</span><input value={scopeValue} onChange={(event) => setScopeValue(event.target.value)} placeholder={scope === "model" ? "64-hex model_sha256" : "explicit identifier"} /></label>}
          <label><span>TYPE</span><select value={type} onChange={(event) => setType(event.target.value as CorrectionType)}>
            {correctionTypes.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
          </select></label>
        </div>
        <ScopeHierarchy selected={scope} />
        <label className="behavior-teach-content"><span>USER-APPROVED INSTRUCTION</span><textarea value={content} onChange={(event) => setContent(event.target.value)} placeholder="e.g. Answer in short paragraphs." /></label>
        <button type="button" className="is-primary" disabled={busy || !content.trim() || (scope !== "global_local" && !scopeValue.trim())} onClick={() => void draft()}>SAVE DRAFT</button>
      </section>

      {error && status !== "UNAVAILABLE" && <div className="behavior-unavailable" role="alert">{error}</div>}

      <section className="behavior-teach-list" aria-label="Saved corrections">
        <header>
          <div><span>STORED CORRECTIONS</span><h2>Lifecycle and scope</h2></div>
          <b>{sortedCorrections.length}</b>
        </header>
        {sortedCorrections.map((correction) => {
          const state = correctionState(correction);
          const verification = verificationResults[correction.id];
          const verificationInputs = verificationDraft(correction.id);
          const targetRunId = verificationInputs.targetRunId.trim();
          const childRunId = verificationInputs.childRunId.trim();
          const invalidPair = targetRunId === childRunId && Boolean(targetRunId);
          return (
            <article className={`is-${state.lifecycle}`} data-correction-state={state.lifecycle} key={correction.id}>
              <header>
                <div className="behavior-correction-heading">
                  <span className={`behavior-correction-form is-${state.lifecycle}`} aria-hidden="true" />
                  <div>
                    <strong>{readableKind(correction.type)}</strong>
                    <span>{scopeText(correction.scope)}</span>
                  </div>
                </div>
                <div className="behavior-correction-state">
                  <b>{state.label}</b>
                  <span>{state.confirmation}</span>
                </div>
              </header>
              <p className="behavior-correction-state-note">{state.note}</p>
              <dl className="behavior-correction-facts">
                <div><dt>CONTENT HASH</dt><dd><code>{correction.content_hash}</code></dd></div>
                <div><dt>CREATED</dt><dd><time dateTime={correction.created_at}>{correction.created_at}</time></dd></div>
                <div><dt>CONFIRMED</dt><dd>{timestampText(correction.confirmed_ts)}</dd></div>
                {correction.disabled_ts != null && <div><dt>DISABLED</dt><dd>{timestampText(correction.disabled_ts)}</dd></div>}
              </dl>
              <p className="behavior-correction-content">{correction.content ?? "CONTENT REDACTED"}</p>

              {correction.confirmed_ts == null && (
                <section className="behavior-teach-verify" aria-label={`Verify ${correction.id} against recorded runs`}>
                  <header><span>VERIFY A RECORDED PAIR</span><b>COMPARE ONLY</b></header>
                  <p>Choose two already-persisted runs. This comparison does not create a retry; a passing comparison may promote this draft.</p>
                  <div className="behavior-teach-verify-fields">
                    <label><span>TARGET FAILURE RUN</span>
                      {runs.length ? (
                        <select value={verificationInputs.targetRunId} onChange={(event) => updateVerification(correction.id, { targetRunId: event.target.value })}>
                          <option value="">SELECT RECORDED RUN</option>
                          {runs.map((run) => <option value={run.id} key={run.id}>{run.label}</option>)}
                        </select>
                      ) : <input value={verificationInputs.targetRunId} onChange={(event) => updateVerification(correction.id, { targetRunId: event.target.value })} placeholder="run_…" />}
                    </label>
                    <label><span>CHILD RUN</span>
                      {runs.length ? (
                        <select value={verificationInputs.childRunId} onChange={(event) => updateVerification(correction.id, { childRunId: event.target.value })}>
                          <option value="">SELECT RECORDED RUN</option>
                          {runs.map((run) => <option value={run.id} key={run.id}>{run.label}</option>)}
                        </select>
                      ) : <input value={verificationInputs.childRunId} onChange={(event) => updateVerification(correction.id, { childRunId: event.target.value })} placeholder="run_…" />}
                    </label>
                    <label><span>COMPARISON</span><select value={verificationInputs.criterion} onChange={(event) => updateVerification(correction.id, { criterion: event.target.value as CorrectionMatchCriterion })}>
                      {criteria.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
                    </select></label>
                  </div>
                  {invalidPair && <EvidenceMark variant="chip" state="unavailable" label="Pair blocked" reason="Choose two different already-persisted runs." />}
                  <button type="button" disabled={busy || !targetRunId || !childRunId || invalidPair} onClick={() => void verify(correction)}>COMPARE PERSISTED RUNS + PROMOTE IF PASSED</button>
                </section>
              )}

              {verification && <VerificationComparison verification={verification} />}

              <footer>
                {correction.confirmed_ts == null && <button type="button" disabled={busy} onClick={() => void run("CONFIRMING", () => confirmCorrection(correction.id))}>CONFIRM</button>}
                {correction.enabled && <button type="button" disabled={busy} onClick={() => void run("DISABLING", () => disableCorrection(correction.id))}>DISABLE</button>}
                {!correction.enabled && correction.confirmed_ts != null && <button type="button" disabled={busy} onClick={() => void run("ENABLING", () => enableCorrection(correction.id))}>ENABLE</button>}
                {correction.confirmed_ts != null && <button type="button" className="behavior-correction-undo" disabled={busy} onClick={() => void run("UNDOING", () => undoCorrection(correction.id))}>UNDO LAST CHANGE</button>}
              </footer>
            </article>
          );
        })}
        {!sortedCorrections.length && (status === "UNAVAILABLE" ? (
          <div className="behavior-empty-row">
            <EvidenceMark
              variant="chip"
              state="unavailable"
              label="Corrections unavailable"
              reason={error || "The correction store did not return a readable list."}
            />
          </div>
        ) : <div className="behavior-empty-row">NO SAVED CORRECTIONS</div>)}
      </section>

      <CorrectionResolution
        runs={runs}
        selectedRunId={resolutionRunId}
        onSelectedRunIdChange={selectResolutionRun}
        state={resolutionState}
        onRead={() => void readResolution()}
      />
    </div>
  );
}
