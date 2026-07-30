import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { parseForkArtifact, type ForkArtifact } from "../../data/api";
import type { ObservatoryData, RuntimeState } from "../../data/types";
import {
  cancelWorkbenchJob,
  JOB_TERMINAL_STATES,
  loadTokenWorkbench,
  loadWorkbenchJob,
  postCausalTraceAction,
  postForkAction,
  postMechanisticDiffAction,
  postSourceMeasureAction,
  WorkbenchActionError,
  WorkbenchLoadError,
  type WorkbenchCapabilities,
  type WorkbenchDocument,
  type WorkbenchJob,
} from "../../data/tokenWorkbench";
import { parseCausalTraceEvidence, type CausalTraceEvidence } from "./layerApi";
import type { ScopeSelectionState, ScopeUrlState, ScopeView } from "./urlState";

/**
 * The ONE authority for "which token of which run, seen against which optional reference run, in which
 * view" -- Milestone E's replacement for ObservatoryWorkspace's five previously scattered `useState`
 * calls (selectedToken / selectedLayer / variantReferenceId / view / forkToken), plus Milestone F's
 * generic action-tray state machine (fork / causal-trace / source-measure / mechanistic-diff), all in one
 * hook so a token selection and an action's lifecycle can never drift out of sync with each other.
 *
 * THE CENTRAL RULE THIS FILE ENFORCES
 * ------------------------------------
 * Changing the selection (token, run, reference) fetches EXACTLY ONE thing: GET .../workbench, which the
 * backend guarantees triggers no computation (see clozn/runs/token_workbench.py). No action ever starts
 * from a selection change -- `runAction` is only ever called from an explicit user click in ActionTray.
 * A stale response for a superseded selection is guarded by one monotonic request-id ref, the same
 * pattern scope.tsx already used for run/fork races (see that file's own doc comment).
 */

// ------------------------------------------------------------------------------------------- selection
export interface TokenWorkbenchSelection {
  view: ScopeView;
  token: number;
  layer: number;
  reference: string;
}

type SelectionAction =
  | { type: "view"; value: ScopeView }
  | { type: "token"; value: number }
  | { type: "layer"; value: number }
  | { type: "reference"; value: string };

function selectionReducer(
  state: TokenWorkbenchSelection,
  action: SelectionAction,
): TokenWorkbenchSelection {
  switch (action.type) {
    case "view": return state.view === action.value ? state : { ...state, view: action.value };
    case "token": return state.token === action.value ? state : { ...state, token: action.value };
    case "layer": return state.layer === action.value ? state : { ...state, layer: action.value };
    case "reference": return state.reference === action.value ? state : { ...state, reference: action.value };
    default: return state;
  }
}

export function initialToken(data: ObservatoryData): number {
  if (!data.tokens.length) return 0;
  let weakest = 0;
  for (let index = 1; index < data.tokens.length; index += 1) {
    if ((data.tokens[index].confidence ?? 1) < (data.tokens[weakest].confidence ?? 1)) weakest = index;
  }
  return data.mode === "run" ? weakest : Math.min(7, data.tokens.length - 1);
}

export function clampToken(data: ObservatoryData, requested?: number): number {
  if (!data.tokens.length) return 0;
  return Math.max(0, Math.min(data.tokens.length - 1, requested ?? initialToken(data)));
}

export function clampLayer(runtime: RuntimeState, requested?: number): number {
  const value = Math.max(0, requested ?? 0);
  const count = runtime.engine?.layerCount;
  return count == null || count <= 0 ? value : Math.min(count - 1, value);
}

export function initialView(data: ObservatoryData, requested?: ScopeView): ScopeView {
  if (requested === "layers" && (data.mode !== "run" || !data.response?.trim())) return "trace";
  if (requested === "variants" && data.mode !== "run") return "trace";
  return requested ?? "trace";
}

function initialSelection(
  data: ObservatoryData,
  runtime: RuntimeState,
  requested?: ScopeUrlState,
): TokenWorkbenchSelection {
  return {
    view: initialView(data, requested?.view),
    token: clampToken(data, requested?.token),
    layer: clampLayer(runtime, requested?.layer),
    reference: requested?.reference ?? (data.parentRunId && data.parentRunId !== data.id ? data.parentRunId : ""),
  };
}

// ------------------------------------------------------------------------------------------ workbench doc
export type WorkbenchDocState =
  | { status: "unavailable" }
  | { status: "loading" }
  | { status: "loaded"; doc: WorkbenchDocument }
  | { status: "error"; message: string };

// ---------------------------------------------------------------------------------------- action tray
export const ACTION_IDS = ["exact_fork", "causal_trace", "source_measurement", "mechanistic_diff"] as const;
export type ActionId = (typeof ACTION_IDS)[number];

export interface ActionArtifactMap {
  exact_fork: ForkArtifact;
  causal_trace: CausalTraceEvidence;
  source_measurement: Record<string, unknown>;
  mechanistic_diff: never;
}

/** `artifact` is deliberately typed as the PLAIN union of the four artifact shapes here (not a
 * `Record<ActionId, ActionState<...>>` mapped type narrowed per key) -- a mapped-type-per-key encoding
 * fights TypeScript's variance rules for no real safety gain, since `actions` is keyed by a run-time
 * `ActionId` union everywhere it is read anyway. The actual discipline against cross-rendering one
 * action's artifact as another's lives where it matters: each `runAction` branch below only ever
 * produces its OWN action's artifact shape, and each reader (ActionTray.tsx, LayerScope.tsx) casts to
 * the one type it knows to expect for the specific action id it is rendering -- the same pattern
 * ForkOutcomePanel's own `never`-exhaustiveness switch already uses. */
export interface ActionState {
  phase: "unavailable" | "idle" | "running" | "cancelling" | "cancelled" | "cached" | "completed" | "error";
  /** The capability's OWN native status word (snapshot_state / status), shown as text alongside `phase`
   * -- never replaced by a generic label (see WorkbenchEvidenceSection's doc comment in
   * data/tokenWorkbench.ts). */
  nativeStatus?: string;
  reason?: string;
  job?: WorkbenchJob;
  artifact?: ForkArtifact | CausalTraceEvidence | Record<string, unknown>;
  pairCompatibility?: Record<string, unknown>;
}

export interface ForkOutcomeBanner {
  parentId: string;
  childId: string;
  note?: string;
  artifact: ForkArtifact;
}

function unavailableAction(reason: string, nativeStatus?: string): ActionState {
  return { phase: "unavailable", reason, nativeStatus };
}

function idleAction(nativeStatus?: string): ActionState {
  return { phase: "idle", nativeStatus };
}

function actionsFromCapabilities(capabilities: WorkbenchCapabilities): Record<ActionId, ActionState> {
  return {
    exact_fork: capabilities.exactFork.available
      ? idleAction(capabilities.exactFork.snapshotState)
      : unavailableAction(
        capabilities.exactFork.reason ?? "exact fork is unavailable for this token",
        capabilities.exactFork.snapshotState,
      ),
    causal_trace: capabilities.causalTrace.available
      ? idleAction(capabilities.causalTrace.status)
      : unavailableAction(
        capabilities.causalTrace.reason ?? "causal trace is unavailable for this token",
        capabilities.causalTrace.status,
      ),
    source_measurement: capabilities.sourceMeasurement.available
      ? idleAction(capabilities.sourceMeasurement.status)
      : unavailableAction(
        capabilities.sourceMeasurement.reason ?? "source measurement is unavailable for this token",
        capabilities.sourceMeasurement.status,
      ),
    mechanistic_diff: capabilities.mechanisticDiff.available
      ? idleAction("eligible")
      : unavailableAction(capabilities.mechanisticDiff.reason),
  };
}

const NO_DOC_ACTIONS: Record<ActionId, ActionState> = {
  exact_fork: unavailableAction("no run evidence is loaded yet"),
  causal_trace: unavailableAction("no run evidence is loaded yet"),
  source_measurement: unavailableAction("no run evidence is loaded yet"),
  mechanistic_diff: unavailableAction("no run evidence is loaded yet"),
};

const POLL_INTERVAL_MS = 350;

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      window.clearTimeout(timeout);
      reject(new DOMException("Aborted", "AbortError"));
    }
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function describeActionError(error: unknown): string {
  if (error instanceof WorkbenchActionError) return error.message;
  return error instanceof Error ? error.message : "the action request failed";
}

/** The fork job worker attaches the created child run record itself as `job.result` (see
 * clozn.runs.token_workbench_actions.fork_worker's `control.attach_result(child)`) -- the exact same
 * shape `parseForkArtifact` already decodes for the cached branch and for the legacy POST /runs/<id>/fork
 * response, so a completed job reuses it verbatim. */
function decodeForkJobResult(result: unknown): ForkArtifact | undefined {
  try {
    return parseForkArtifact(result);
  } catch {
    return undefined;
  }
}

/** The causal-trace job worker attaches a full `clozn.token-workbench-action.v1` cache entry as
 * `job.result` (see `causal_trace_worker`'s `control.attach_result(entry)`); `entry.result` is
 * `tracer.trace()`'s own dict, the exact same shape the cached branch already decodes. */
function decodeCausalTraceJobResult(result: unknown): CausalTraceEvidence | undefined {
  const entry = result && typeof result === "object" ? result as Record<string, unknown> : undefined;
  if (!entry) return undefined;
  const inner = entry.result && typeof entry.result === "object" ? entry.result as Record<string, unknown> : {};
  return parseCausalTraceEvidence(inner);
}

export interface UseTokenWorkbenchOptions {
  data: ObservatoryData;
  runtime: RuntimeState;
  initialState?: ScopeUrlState;
  onStateChange?: (state: ScopeSelectionState) => void;
  /** Fork's own navigation IS run selection -- a successful fork just created a new immutable run, so
   * this reuses whatever the caller already uses to load a different run (ScopePanel's `selectRun`)
   * rather than the token workbench inventing a second navigation path. */
  onSelectRun: (runId: string) => void;
  /** Reported once a fork action reaches `cached`/`completed` with a child -- the caller decides how
   * long to keep showing it (Observatory keeps it alive across the run-change remount that
   * `onSelectRun` triggers; see Observatory.tsx). */
  onForkOutcome?: (banner: ForkOutcomeBanner) => void;
}

export function useTokenWorkbench({
  data,
  runtime,
  initialState,
  onStateChange,
  onSelectRun,
  onForkOutcome,
}: UseTokenWorkbenchOptions) {
  const [selection, dispatch] = useReducer(
    selectionReducer,
    undefined,
    () => initialSelection(data, runtime, initialState),
  );
  const [workbench, setWorkbench] = useState<WorkbenchDocState>({ status: "unavailable" });
  const [actions, setActions] = useState<Record<ActionId, ActionState>>(NO_DOC_ACTIONS);
  const [forkChoice, setForkChoice] = useState<{ piece: string; tokenId?: number } | null>(null);

  // One monotonic id per SELECTION (run id + token index + reference id). Every async effect below --
  // the workbench GET, every action's job poll -- captures its own value and checks it before writing
  // state, so a response for a superseded selection can never land on top of a newer one (the same
  // pattern scope.tsx's own requestIdRef already uses for run/fork races).
  const selectionRequestId = useRef(0);
  const actionControllers = useRef<Partial<Record<ActionId, AbortController>>>({});
  // The requestId check above only catches staleness WITHIN one mounted instance of this hook. A run
  // switch remounts the whole workspace (Observatory.tsx's `key={resetKey}`) -- when that happens THIS
  // instance is discarded outright, its own selectionRequestId ref frozen forever at whatever it last
  // was, so the requestId comparison alone could still pass and let an in-flight action's callback (most
  // dangerously `onSelectRun` for a fork) fire into a world that has moved on. `mountedRef` closes that
  // gap: every external callback this hook can invoke asynchronously checks it first.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const activeView: ScopeView = selection.view === "layers" && (data.mode !== "run" || !data.response?.trim())
    ? "trace"
    : selection.view === "variants" && data.mode !== "run"
      ? "trace"
      : selection.view;

  const setView = useCallback((view: ScopeView) => dispatch({ type: "view", value: view }), []);
  const setSelectedToken = useCallback((token: number) => dispatch({ type: "token", value: token }), []);
  const setSelectedLayer = useCallback((layer: number) => dispatch({ type: "layer", value: layer }), []);
  const setVariantReferenceId = useCallback(
    (reference: string) => dispatch({ type: "reference", value: reference }),
    [],
  );

  useEffect(() => {
    onStateChange?.({
      view: activeView,
      token: selection.token,
      reference: selection.reference || undefined,
      layer: selection.layer,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors Observatory's prior effect deps
  }, [activeView, onStateChange, selection.layer, selection.token, selection.reference]);

  // THE single fetch a token/run/reference selection is allowed to trigger.
  useEffect(() => {
    const requestId = ++selectionRequestId.current;
    for (const controller of Object.values(actionControllers.current)) controller?.abort();
    actionControllers.current = {};
    setForkChoice(null);

    if (data.mode !== "run" || !data.tokens.length) {
      setWorkbench({ status: "unavailable" });
      setActions(NO_DOC_ACTIONS);
      return;
    }
    setWorkbench({ status: "loading" });
    const controller = new AbortController();
    void loadTokenWorkbench(data.id, selection.token, selection.reference || undefined, controller.signal)
      .then((doc) => {
        if (selectionRequestId.current !== requestId) return;
        setWorkbench({ status: "loaded", doc });
        setActions(actionsFromCapabilities(doc.capabilities));
        // Backend-composed default reference (clozn.runs.token_workbench's own `_comparison_section`
        // auto-selection), applied exactly once when no explicit reference was requested -- replaces
        // this Studio's former client-side "same prompt in the runtime run list" heuristic entirely
        // (compose the backend's own pick, don't keep a second one).
        if (!doc.referenceRunId && !selection.reference) {
          const autoSelected = doc.comparison.raw.selection;
          const pick = autoSelected && typeof autoSelected === "object"
            ? (autoSelected as Record<string, unknown>).reference_run_id
            : undefined;
          if (typeof pick === "string" && pick) dispatch({ type: "reference", value: pick });
        }
      })
      .catch((error) => {
        if (selectionRequestId.current !== requestId) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        const message = error instanceof WorkbenchLoadError
          ? error.message
          : error instanceof Error ? error.message : "token workbench evidence unavailable";
        setWorkbench({ status: "error", message });
        setActions(NO_DOC_ACTIONS);
      });
    return () => controller.abort();
    // Deliberately keyed on view/layer NOTHING: those two selection fields never change which evidence
    // this fetch needs, only how it's displayed. `selection.reference` IS a dep -- the auto-selection
    // branch above dispatches a new one exactly once, which re-runs this effect with an explicit
    // `reference_run_id`, guarded from double-firing by the requestId check inside `.then()`.
  }, [data.id, data.mode, data.tokens.length, selection.token, selection.reference]);

  const doc = workbench.status === "loaded" ? workbench.doc : undefined;

  const setActionState = useCallback((id: ActionId, next: ActionState) => {
    setActions((current) => ({ ...current, [id]: next }));
  }, []);

  const pollJob = useCallback(async (
    id: ActionId,
    requestId: number,
    runId: string,
    index: number,
    initialJob: WorkbenchJob,
    decodeResult: (result: unknown) => unknown,
    onArtifact: (artifact: unknown) => void,
  ) => {
    const controller = new AbortController();
    actionControllers.current[id] = controller;
    let job = initialJob;
    try {
      while (!JOB_TERMINAL_STATES.includes(job.state)) {
        await wait(POLL_INTERVAL_MS, controller.signal);
        job = await loadWorkbenchJob(runId, index, job.jobId, controller.signal);
        if (selectionRequestId.current !== requestId) return;
        setActionState(id, { phase: job.state === "cancelling" ? "cancelling" : "running", job });
      }
    } catch (error) {
      if (controller.signal.aborted || selectionRequestId.current !== requestId) return;
      setActionState(id, {
        phase: "error",
        reason: error instanceof Error ? error.message : "job status could not be read",
      });
      return;
    }
    if (selectionRequestId.current !== requestId) return;
    if (job.state === "cancelled") {
      setActionState(id, {
        phase: "cancelled",
        job,
        reason: "cancelled -- this Studio stopped reporting on the job, but the server call already in "
          + "flight keeps running to completion in its own thread; its result was never saved",
      });
      return;
    }
    if (job.state === "failed") {
      setActionState(id, { phase: "error", job, reason: job.error?.message ?? "the job failed" });
      return;
    }
    const artifact = job.result != null ? decodeResult(job.result) : undefined;
    setActionState(id, artifact != null
      ? { phase: "completed", job, artifact: artifact as never }
      : { phase: "completed", job });
    // Always reported, even with no decoded artifact -- source-measure's own completion callback
    // (reload the run) has nothing to do with whether an artifact was attached to the job.
    onArtifact(artifact);
  }, [setActionState]);

  const runAction = useCallback((id: ActionId) => {
    // Defense in depth, not just an ActionTray button's `disabled` attribute: this hook itself refuses
    // to start an action the current workbench document did not report available, even if called
    // directly (a test, a keyboard shortcut, a future second caller). Selecting a token never runs an
    // action; this guard makes sure no other path silently can either.
    if (!doc || actions[id].phase === "unavailable") return;
    const requestId = selectionRequestId.current;
    const runId = data.id;
    const index = selection.token;
    setActionState(id, { phase: "running" });

    function reportForkArtifact(artifact: ForkArtifact | undefined) {
      // `mountedRef` (not just the requestId check already applied by every caller of this function) --
      // a run switch can remount this whole hook instance while this fork's request is still in flight;
      // once that happens, this closure must never navigate the NEW instance's world on the old one's
      // behalf. See mountedRef's own doc comment above for why the requestId check alone cannot catch
      // this case.
      if (!mountedRef.current || !artifact?.child) return;
      onForkOutcome?.({
        parentId: artifact.child.parentId || runId,
        childId: artifact.child.id,
        note: artifact.child.note,
        artifact,
      });
      onSelectRun(artifact.child.id);
    }

    if (id === "exact_fork") {
      const choice = forkChoice
        ?? doc.token.alternatives.find((alt) => alt.piece !== doc.token.piece)
        ?? doc.token.alternatives[0];
      if (!choice) {
        setActionState(id, { phase: "error", reason: "no candidate token is available to fork" });
        return;
      }
      void postForkAction(runId, index, choice.piece, choice.tokenId).then((envelope) => {
        if (selectionRequestId.current !== requestId) return;
        if (envelope.outcome === "cached") {
          setActionState(id, { phase: "cached", artifact: envelope.artifact });
          reportForkArtifact(envelope.artifact);
          return;
        }
        if (envelope.outcome === "unavailable") {
          setActionState(id, { phase: "unavailable", reason: envelope.reason.message });
          return;
        }
        setActionState(id, { phase: "running", job: envelope.job });
        void pollJob(id, requestId, runId, index, envelope.job, decodeForkJobResult, (artifact) => {
          reportForkArtifact(artifact as ForkArtifact | undefined);
        });
      }).catch((error) => setActionState(id, { phase: "error", reason: describeActionError(error) }));
      return;
    }

    if (id === "causal_trace") {
      void postCausalTraceAction(runId, index).then((envelope) => {
        if (selectionRequestId.current !== requestId) return;
        if (envelope.outcome === "cached") {
          setActionState(id, { phase: "cached", artifact: envelope.artifact });
          return;
        }
        if (envelope.outcome === "unavailable") {
          setActionState(id, { phase: "unavailable", reason: envelope.reason.message });
          return;
        }
        setActionState(id, { phase: "running", job: envelope.job });
        void pollJob(id, requestId, runId, index, envelope.job, decodeCausalTraceJobResult, () => {});
      }).catch((error) => setActionState(id, { phase: "error", reason: describeActionError(error) }));
      return;
    }

    if (id === "source_measurement") {
      void postSourceMeasureAction(runId, index).then((envelope) => {
        if (selectionRequestId.current !== requestId) return;
        if (envelope.outcome === "cached") {
          setActionState(id, { phase: "cached", artifact: envelope.artifact });
          return;
        }
        if (envelope.outcome === "unavailable") {
          setActionState(id, { phase: "unavailable", reason: envelope.reason.message });
          return;
        }
        setActionState(id, { phase: "running", job: envelope.job });
        // source-measure persists straight to the run (no job.result) -- a completed job reloads the
        // run itself so the newly measured sources reach every surface that reads it, TraceScope
        // included, rather than this hook parsing a second copy of the influence-map artifact.
        void pollJob(id, requestId, runId, index, envelope.job, () => undefined, () => {
          if (mountedRef.current) onSelectRun(runId);
        });
      }).catch((error) => setActionState(id, { phase: "error", reason: describeActionError(error) }));
      return;
    }

    // mechanistic_diff: needs a reference run of a DIFFERENT model, which is exactly what the
    // capability already required to report `available: true` -- the reference already in scope is
    // reused rather than asking the user to pick it twice. Never reaches cached/job this milestone
    // (see data/tokenWorkbench.ts's own doc comment on postMechanisticDiffAction).
    if (!selection.reference) {
      setActionState(id, { phase: "error", reason: "no reference run is selected" });
      return;
    }
    void postMechanisticDiffAction(runId, index, selection.reference).then((envelope) => {
      if (selectionRequestId.current !== requestId) return;
      if (envelope.outcome === "unavailable") {
        setActionState(id, {
          phase: "unavailable",
          reason: envelope.reason.message,
          pairCompatibility: envelope.pairCompatibility,
        });
      }
    }).catch((error) => setActionState(id, { phase: "error", reason: describeActionError(error) }));
  }, [actions, doc, forkChoice, onForkOutcome, onSelectRun, pollJob, selection.reference, selection.token, setActionState, data.id]);

  const cancelAction = useCallback((id: ActionId) => {
    setActions((current) => {
      const job = current[id]?.job;
      if (!job || !job.cancellable) return current;
      void cancelWorkbenchJob(data.id, selection.token, job.jobId).catch(() => {});
      return { ...current, [id]: { ...current[id], phase: "cancelling" } };
    });
  }, [data.id, selection.token]);

  return {
    selection: { ...selection, view: activeView },
    setView,
    setSelectedToken,
    setSelectedLayer,
    setVariantReferenceId,
    workbench,
    doc,
    actions,
    runAction,
    cancelAction,
    forkChoice,
    setForkChoice,
  };
}
