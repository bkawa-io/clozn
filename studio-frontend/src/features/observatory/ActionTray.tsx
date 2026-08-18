import type { MechanisticDiffArtifact, TimeTravelArtifact } from "../../data/tokenWorkbench";
import { ACTION_CONFIG, ACTION_ORDER } from "./actionConfig";
import type { CausalTraceEvidence } from "./layerApi";
import type { ActionId, ActionState } from "./useTokenWorkbench";

interface ActionTrayProps {
  actions: Record<ActionId, ActionState>;
  onRun: (id: ActionId) => void;
  onCancel: (id: ActionId) => void;
  /** Whether a fork choice has been made yet -- FORK is disabled without one even when the capability
   * itself is available, since there is nothing to force. */
  forkChoiceLabel?: string;
}

function humanize(value: string): string {
  return value.replace(/_/g, " ").toUpperCase();
}

function phaseLabel(state: ActionState): string {
  switch (state.phase) {
    case "unavailable": return "UNAVAILABLE";
    case "idle": return "READY";
    case "running": return state.job ? `RUNNING · ${humanize(state.job.progress.phase)} ${state.job.progress.percent}%` : "STARTING";
    case "cancelling": return "CANCELLING";
    case "cancelled": return "CANCELLED";
    case "cached": return "CACHED RESULT";
    case "completed": return "COMPLETED";
    case "error": return "REQUEST FAILED";
    default: return state.phase;
  }
}

function pairCompatibilityOperations(report: Record<string, unknown>): Array<{ name: string; permitted: boolean; reason?: string }> {
  const verdict = report.verdict && typeof report.verdict === "object" ? report.verdict as Record<string, unknown> : {};
  const operations = verdict.operations && typeof verdict.operations === "object" ? verdict.operations as Record<string, unknown> : {};
  return Object.entries(operations).flatMap(([name, value]) => {
    if (!value || typeof value !== "object") return [];
    const entry = value as Record<string, unknown>;
    return [{
      name,
      permitted: entry.permitted === true,
      reason: typeof entry.reason === "string" ? entry.reason : undefined,
    }];
  });
}

function ActionResult({ id, state, onMaterialize }: {
  id: ActionId;
  state: ActionState;
  onMaterialize?: (artifact: TimeTravelArtifact) => void;
}) {
  if (id === "exact_fork" && (state.phase === "cached" || state.phase === "completed") && state.artifact) {
    const artifact = state.artifact as TimeTravelArtifact;
    const continuation = artifact.continuation && typeof artifact.continuation === "object"
      ? artifact.continuation : {};
    const materializable = artifact.status === "completed"
      && typeof artifact.experiment_id === "string"
      && typeof artifact.arm_id === "string"
      && typeof artifact.observation_id === "string";
    return (
      <div className="action-result">
        <p>
          {artifact.status === "completed"
            ? `GENERATED OBSERVATION ${String(artifact.observation_id).slice(-8)} · ${artifact.fidelity ?? artifact.resolution ?? "UNAVAILABLE"}`
            : `FORCE-TOKEN ${artifact.status.toUpperCase()} · ${String(artifact.reason ?? "no generated evidence")}`}
        </p>
        {typeof continuation.generated_suffix_text === "string" && (
          <p>GENERATED SUFFIX: {continuation.generated_suffix_text}</p>
        )}
        {materializable && onMaterialize && (
          <button type="button" onClick={() => onMaterialize(artifact)}>MATERIALIZE CHILD RUN</button>
        )}
      </div>
    );
  }
  if (id === "causal_trace" && (state.phase === "cached" || state.phase === "completed") && state.artifact) {
    const evidence = state.artifact as CausalTraceEvidence;
    return (
      <p className="action-result">
        {evidence.ok
          ? `${evidence.survivorCount ?? evidence.nodes.length} / ${evidence.candidateCount} SITES SURVIVED · CONTROL ${evidence.verdict ?? "—"} -- see Layer evidence for the full site list`
          : evidence.blocked ?? evidence.error ?? "the trace completed without a usable result"}
      </p>
    );
  }
  if (id === "source_measurement" && (state.phase === "cached" || state.phase === "completed")) {
    return <p className="action-result">context sources were (re)measured -- Sources reloaded</p>;
  }
  if (id === "mechanistic_diff" && (state.phase === "cached" || state.phase === "completed") && state.artifact) {
    const artifact = state.artifact as MechanisticDiffArtifact;
    const points = artifact.residualPoints.length;
    const measured = artifact.residualPoints.filter((point) => {
      const metrics = point.metrics;
      return Boolean(metrics && typeof metrics === "object" && Object.keys(metrics as object).length);
    }).length;
    return (
      <p className="action-result">
        {measured} / {points} residual points measured across {artifact.layersRequested.length} layers
        {artifact.positionsRequested.length ? ` at ${artifact.positionsRequested.length} position${artifact.positionsRequested.length === 1 ? "" : "s"}` : ""}
        {" -- observational only; see Layers for the metric details"}
      </p>
    );
  }
  if (id === "mechanistic_diff" && state.pairCompatibility) {
    const operations = pairCompatibilityOperations(state.pairCompatibility);
    if (!operations.length) return null;
    return (
      <dl className="action-pair-compatibility">
        {operations.map((operation) => (
          <div key={operation.name}>
            <dt>{humanize(operation.name)}</dt>
            <dd>{operation.permitted ? "PERMITTED" : "REFUSED"}{operation.reason ? ` -- ${operation.reason}` : ""}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return null;
}

/**
 * Milestone F's action tray: one generic operation model instead of a bespoke button per action. Every
 * row is driven by the SAME `ActionState` shape (see useTokenWorkbench.ts), but each row's own result is
 * rendered by its own type-specific branch above (`ActionResult`) -- the four artifacts are never
 * cross-rendered through a shared summary.
 */
export function ActionTray({ actions, onRun, onCancel, forkChoiceLabel, onMaterialize }: ActionTrayProps & {
  onMaterialize?: (artifact: TimeTravelArtifact) => void;
}) {
  return (
    <section className="action-tray" aria-label="Token workbench actions">
      <header className="section-title"><h3>Actions</h3><span>{ACTION_ORDER.length}</span></header>
      <div className="action-tray-list">
        {ACTION_ORDER.map((id) => {
          const config = ACTION_CONFIG[id];
          const state = actions[id];
          const busy = state.phase === "running" || state.phase === "cancelling";
          const canRun = state.phase !== "unavailable" && !busy;
          const canCancel = busy && Boolean(state.job?.cancellable);
          return (
            <article className={`action-row is-${state.phase}`} aria-label={config.label} key={id}>
              <header>
                <div>
                  <strong>{config.label}</strong>
                  <span className="action-native-status">
                    {state.nativeStatus ? humanize(state.nativeStatus) : phaseLabel(state)}
                  </span>
                </div>
                <div className="action-row-buttons">
                  {canCancel && (
                    <button type="button" onClick={() => onCancel(id)}>CANCEL</button>
                  )}
                  <button
                    type="button"
                    disabled={!canRun || (id === "exact_fork" && !forkChoiceLabel)}
                    onClick={() => onRun(id)}
                  >
                    {busy ? phaseLabel(state) : state.phase === "unavailable" ? "UNAVAILABLE" : "RUN"}
                  </button>
                </div>
              </header>
              <p className="action-cost">
                {config.changesExecution ? "RUNS NEW COMPUTATION" : "NO NEW COMPUTATION"} · {config.produces === "child_run" ? "CREATES A CHILD RUN" : "CREATES AN EVIDENCE ARTIFACT"} · {config.cost}
              </p>
              <p className="action-claim">{config.claimBoundary}</p>
              {id === "exact_fork" && forkChoiceLabel && (
                <p className="action-fork-choice"><b>FORCED TOKEN</b>{forkChoiceLabel}</p>
              )}
              {(state.phase === "unavailable" || state.phase === "error" || state.phase === "cancelled") && state.reason && (
                <p className={`action-reason ${state.phase === "error" ? "is-error" : ""}`} role="status">
                  {state.reason}
                </p>
              )}
              <ActionResult id={id} state={state} onMaterialize={onMaterialize} />
              <p className="action-undo">{config.undo}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
