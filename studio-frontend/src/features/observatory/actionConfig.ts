import type { ActionId } from "./useTokenWorkbench";

/**
 * Milestone F's "one generic operation model rather than a bespoke button per action" -- static,
 * reviewable copy for what each of the four token-workbench actions costs, claims, and undoes. ActionTray
 * reads this table for every action instead of writing four near-duplicate JSX blocks; the per-action
 * RESULT rendering (the artifact itself) still stays type-specific in ActionTray.tsx, since the artifacts
 * are four genuinely different evidence shapes that must never be cross-rendered (see
 * data/tokenWorkbench.ts's own doc comment on why the four capabilities stay separate types).
 */
export interface ActionConfig {
  id: ActionId;
  label: string;
  /** Whether running this action performs new engine computation (a forward pass, a checkpoint
   * restore) as opposed to reading something already recorded. */
  changesExecution: boolean;
  /** What the action produces: an immutable new run, or an evidence artifact attached to THIS run. */
  produces: "child_run" | "evidence_artifact";
  cost: string;
  claimBoundary: string;
  undo: string;
}

export const ACTION_CONFIG: Record<ActionId, ActionConfig> = {
  exact_fork: {
    id: "exact_fork",
    label: "FORCE TOKEN",
    changesExecution: true,
    produces: "evidence_artifact",
    cost: "one canonical Generate experiment from the selected answer-token boundary",
    claimBoundary: "records a counterfactual continuation as a GeneratedObservation; exact fidelity is "
      + "reported only after the unchanged control proves it",
    undo: "no undo needed -- this is standalone evidence; use MATERIALIZE CHILD RUN explicitly when "
      + "you want to promote the generated observation into lineage",
  },
  causal_trace: {
    id: "causal_trace",
    label: "CAUSAL TRACE",
    changesExecution: true,
    produces: "evidence_artifact",
    cost: "multiple forward passes across candidate sites plus matched-random controls, on the current "
      + "worker",
    claimBoundary: "reports which layer/position sites survive a matched-random noise floor for this "
      + "token -- controlled intervention evidence, not a full causal proof and not a claim about every "
      + "possible site",
    undo: "no undo needed -- a read-only evidence artifact, cached by run identity and token index; a "
      + "repeat request with the same parameters reuses it rather than recomputing",
  },
  source_measurement: {
    id: "source_measurement",
    label: "SOURCE MEASURE",
    changesExecution: true,
    produces: "evidence_artifact",
    cost: "one counterfactual forward pass per selected context span, on the current worker",
    claimBoundary: "measures whether removing a context span changes this token's probability enough to "
      + "clear the measurement floor -- a span that does not clear is an honest 'no effect detected', "
      + "never proof the span was irrelevant",
    undo: "no undo needed -- persists to this run as its own influence-map record; a repeat request "
      + "with unchanged inputs is a cache hit",
  },
  mechanistic_diff: {
    id: "mechanistic_diff",
    label: "MECHANISTIC DIFF",
    changesExecution: true,
    produces: "evidence_artifact",
    cost: "one bounded teacher-forced capture per model; models are loaded sequentially through the "
      + "managed registry and identical requests are cached",
    claimBoundary: "reports observational residual and token-ranking differences for a compatible pair; "
      + "it does not establish a causal explanation or calibrated confidence",
    undo: "not applicable -- the comparison is immutable evidence attached to the anchor run",
  },
};

export const ACTION_ORDER: ActionId[] = ["exact_fork", "causal_trace", "source_measurement", "mechanistic_diff"];
