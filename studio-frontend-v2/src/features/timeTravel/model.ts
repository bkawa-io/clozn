/**
 * Read model for the token-boundary Time Travel surface.
 *
 * The model intentionally separates recorded history, current execution
 * eligibility, and a completed child's proof. A caller may adapt any gateway
 * response to these types without making the recorded parent mutable.
 */
export type TimeTravelPhase = "loading" | "ready" | "unavailable" | "error" | "stale";

export interface RecordedResponseToken {
  /** Zero-based response-token index; the boundary is immediately before it (0 is prompt boundary). */
  position: number;
  text: string;
}

export interface TimeTravelLocus {
  id: string;
  label: string;
  /** The recorded response-token boundary this locus resolves to. */
  boundaryPosition: number;
}

export type Availability = "available" | "unavailable" | "not_reported";
export type ExactForkAvailability = "requires_live_plan" | "ready_to_execute" | "unavailable" | "not_reported";
export type HistoricalExactness = "verified" | "none" | "not_reported";

export interface ReplayAvailability {
  state: Availability;
  reason?: string;
  /** Only supplied for a reconstructed replay; never inferred by the UI. */
  unavoidableDifferences?: readonly string[];
}

export interface ExactForkAvailabilityState {
  state: ExactForkAvailability;
  reason?: string;
  /** Requirements reported by a live planner, not a UI-generated checklist. */
  requirements?: readonly string[];
}

export interface HistoricalExactProof {
  state: HistoricalExactness;
  verifiedExecutionCount?: number;
  detail?: string;
}

export interface BoundaryFidelity {
  reconstructedReplay: ReplayAvailability;
  exactFork: ExactForkAvailabilityState;
  historicalExactProof: HistoricalExactProof;
}

export interface ProposedIntervention {
  /** Stable ID that a host can pass straight through to its planner/executor. */
  id: string;
  label: string;
  summary?: string;
}

export type ForkExecutionState =
  | "idle"
  | "planning"
  | "control_verifying"
  | "executing"
  | "control_diverged"
  | "failed"
  | "completed";

export interface ForkExecution {
  state: ForkExecutionState;
  detail?: string;
  /** A control mismatch must make this false or absent; the UI never invents a child. */
  interventionRan?: boolean;
}

export interface ChildForkResult {
  runId: string;
  intervention: ProposedIntervention;
  /** Exactness is a completed proof state, never a static capability label. */
  exactness: "verified_exact" | "reconstructed" | "not_reported";
  summary?: string;
}

export interface TimeTravelRun {
  id: string;
  response: string;
  responseTokens?: readonly RecordedResponseToken[];
  model?: string;
  parentRunId?: string;
  sessionKey?: string;
  loci?: readonly TimeTravelLocus[];
  fidelityByBoundary?: Readonly<Record<number, BoundaryFidelity | undefined>>;
  execution?: ForkExecution;
  children?: readonly ChildForkResult[];
}

export interface BoundarySelection {
  runId: string;
  position: number;
  locusId?: string;
}

export interface ForkAction {
  runId: string;
  position: number;
  intervention: ProposedIntervention;
  mode: "exact" | "reconstructed";
}

export function boundaryForLocus(run: TimeTravelRun, locusId?: string): number | undefined {
  return locusId ? run.loci?.find((locus) => locus.id === locusId)?.boundaryPosition : undefined;
}

export function fidelityAt(run: TimeTravelRun, position?: number): BoundaryFidelity | undefined {
  return position === undefined ? undefined : run.fidelityByBoundary?.[position];
}

export function historicalProofAt(run: TimeTravelRun, position: number): HistoricalExactProof | undefined {
  return fidelityAt(run, position)?.historicalExactProof;
}

export function canStageFork(fidelity: BoundaryFidelity | undefined, mode: ForkAction["mode"]): boolean {
  if (!fidelity) return false;
  return mode === "exact"
    ? fidelity.exactFork.state === "ready_to_execute"
    : fidelity.reconstructedReplay.state === "available";
}
