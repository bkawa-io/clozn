import { useCallback, useEffect, useMemo, useState } from "react";
import { studioApi } from "../data/client";
import type { ContextTension, JsonObject, RunRecord, SpanAddressDocument, SuggestedBreakpoints } from "../data/contracts";
import { CompareSurface, projectComparisonSpecimen, recordedStructureFromComparison } from "../features/compare";
import { LinkedEvidenceReader } from "../features/inspect/LinkedEvidenceReader";
import { locusQueryRange, projectDecisionLoci, projectInfluenceSelection, projectLinkedReader, projectTensionSelections } from "../features/inspect/fromContracts";
import type { InfluenceSelection, TextLocus } from "../features/inspect/model";
import { ModelMriSurface, projectRecordedMriSpecimen } from "../features/mri";
import { toJournalRuns } from "../features/runs/fromContracts";
import { RunsJournal } from "../features/runs/RunsJournal";
import { toRuntimeSnapshot } from "../features/runtime/fromContracts";
import type { RuntimeSnapshot, RuntimeSurfacePhase } from "../features/runtime/model";
import { RuntimeSurface } from "../features/runtime/RuntimeSurface";
import { projectTimeTravelRun, TimeTravelSurface, TurnTimeTravelSurface, type ChildForkResult, type ForkAction, type ForkExecution, type ProposedIntervention, type TimeTravelRun } from "../features/timeTravel";
import { AppShell } from "./AppShell";
import { routeHref, type ComparisonRouteSelection, type StudioRoute } from "./router";
import { useHashRoute } from "./useHashRoute";

type RuntimeStatus = "checking" | "ready" | "degraded" | "not-ready" | "unreachable";

function SurfacePlaceholder({ name, question }: { name: string; question: string }) {
  return (
    <section className="surface-placeholder" aria-labelledby="surface-title">
      <span className="eyebrow">GREENFIELD STUDIO</span>
      <h1 id="surface-title">{name}</h1>
      <p>{question}</p>
    </section>
  );
}

function RunsSurface() {
  const [state, setState] = useState<{ phase: "loading" | "ready" | "error"; runs: RunRecord[]; error?: string }>({ phase: "loading", runs: [] });
  useEffect(() => {
    const controller = new AbortController();
    void studioApi.runs(controller.signal).then((runs) => {
      if (!controller.signal.aborted) setState({ phase: "ready", runs: [...runs] });
    }).catch((error) => {
      if (!controller.signal.aborted) setState({ phase: "error", runs: [], error: error instanceof Error ? error.message : "Run journal request failed." });
    });
    return () => controller.abort();
  }, []);
  return <RunsJournal runs={toJournalRuns(state.runs)} phase={state.phase} error={state.error} />;
}

function InspectSurface({ runId, comparison }: Extract<StudioRoute, { surface: "inspect" }>) {
  const [state, setState] = useState<
    | { phase: "loading" }
    | { phase: "error"; error: string }
    | { phase: "ready"; run: RunRecord; addresses: SpanAddressDocument; tension: ContextTension; breakpoints: SuggestedBreakpoints }
  >({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState({ phase: "loading" });
    void Promise.all([studioApi.run(runId, controller.signal), studioApi.spanAddresses(runId, controller.signal), studioApi.contextTension(runId, { limit: 100 }, controller.signal), studioApi.suggestedBreakpoints(runId, controller.signal)])
      .then(([run, addresses, tension, breakpoints]) => {
        if (!controller.signal.aborted) setState({ phase: "ready", run, addresses, tension, breakpoints });
      })
      .catch((error) => {
        if (!controller.signal.aborted) setState({ phase: "error", error: error instanceof Error ? error.message : "Run evidence request failed." });
      });
    return () => controller.abort();
  }, [runId]);

  const projection = useMemo(() => state.phase === "ready" ? projectLinkedReader(state.run, state.addresses) : undefined, [state]);
  const tensionSelections = useMemo(() => state.phase === "ready" && projection ? projectTensionSelections(state.tension, state.run, projection) : {}, [projection, state]);
  const decisionLoci = useMemo(() => state.phase === "ready" ? projectDecisionLoci(state.breakpoints, state.run) : [], [state]);
  const initialLocusId = useMemo(() => {
    const range = comparison?.b;
    if (!range || !projection) return undefined;
    return projection.specimen.answerLoci.find((locus) => locus.start < range.end && locus.end > range.start)?.id;
  }, [comparison?.b, projection]);
  const loadSelection = useCallback(async (locus: TextLocus, signal: AbortSignal): Promise<InfluenceSelection> => {
    if (state.phase !== "ready" || !projection) return { state: "unavailable", reason: "Run evidence is not loaded.", related: [] };
    const range = locusQueryRange(projection.specimen.answer, locus);
    const query = await studioApi.influenceQuery(runId, { ...range, limit: 50 }, signal);
    return projectInfluenceSelection(query, state.run, projection);
  }, [projection, runId, state]);
  const loadTensionSelection = useCallback(async (locus: TextLocus, signal: AbortSignal): Promise<InfluenceSelection> => {
    if (state.phase !== "ready" || !projection) return { state: "unavailable", reason: "Run evidence is not loaded.", related: [] };
    const range = locusQueryRange(projection.specimen.answer, locus);
    const query = await studioApi.contextTension(runId, { ...range, limit: 100 }, signal);
    return projectTensionSelections(query, state.run, projection)[locus.id] ?? { state: query.measurement.state, reason: query.measurement.reason, method: "Recorded context tension", related: [] };
  }, [projection, runId, state]);

  if (state.phase === "loading") return <p className="surface-state" role="status">Reading the recorded run and its stable span addresses…</p>;
  if (state.phase === "error") return <p className="surface-state is-error" role="alert">Investigation unavailable · {state.error}</p>;
  if (!state.run.response) return (
    <section className="surface-placeholder">
      <span className="eyebrow">CONTEXT ↔ ANSWER</span>
      <h1>No readable recorded answer</h1>
      <p>This execution can remain in the journal, but the linked reader will not synthesize answer text that was absent or redacted.</p>
      <a className="primary-action" href="#/runs">Back to Runs</a>
    </section>
  );
  return <LinkedEvidenceReader specimen={projection!.specimen} loadSelection={loadSelection} loadTensionSelection={loadTensionSelection} tensionSelections={tensionSelections} decisionLoci={decisionLoci} initialLocusId={initialLocusId} />;
}

function RuntimeDataSurface() {
  const [revision, setRevision] = useState(0);
  const [state, setState] = useState<{ phase: RuntimeSurfacePhase; snapshot?: RuntimeSnapshot; error?: string }>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState((current) => ({ phase: "loading", snapshot: current.snapshot }));
    void Promise.all([
      studioApi.health(controller.signal),
      studioApi.readiness(controller.signal),
      studioApi.runtimeModels(controller.signal),
    ]).then(([health, readiness, inventory]) => {
      if (!controller.signal.aborted) setState({ phase: "ready", snapshot: toRuntimeSnapshot(health, readiness, inventory) });
    }).catch((error) => {
      if (!controller.signal.aborted) setState((current) => ({
        phase: "error",
        snapshot: current.snapshot,
        error: error instanceof Error ? error.message : "Runtime request failed.",
      }));
    });
    return () => controller.abort();
  }, [revision]);

  return <RuntimeSurface {...state} onRefresh={() => setRevision((current) => current + 1)} />;
}

function recordedRunLabel(run: RunRecord): string {
  const prompt = run.promptSummary?.trim() || "Untitled recorded execution";
  const model = run.model ?? "model not recorded";
  const recorded = run.createdAt ?? (run.createdTs == null ? undefined : new Date(run.createdTs * 1_000).toISOString()) ?? "time not recorded";
  return `${prompt} — ${model} · ${recorded} · ${run.id}`;
}

function CompareDataSurface({ runAId, runBId, selectedDifference }: { runAId?: string; runBId?: string; selectedDifference?: ComparisonRouteSelection }) {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selection, setSelection] = useState({ runAId: runAId ?? "", runBId: runBId ?? "" });
  const [state, setState] = useState<
    | { phase: "idle" }
    | { phase: "loading"; pairKey: string }
    | { phase: "error"; pairKey: string; error: string }
    | { phase: "ready"; pairKey: string; specimen: ReturnType<typeof projectComparisonSpecimen> }
  >({ phase: "idle" });

  useEffect(() => {
    if (!runAId || !runBId) return;
    setSelection((current) => current.runAId === runAId && current.runBId === runBId ? current : { runAId, runBId });
  }, [runAId, runBId]);

  useEffect(() => {
    const controller = new AbortController();
    void studioApi.runs(controller.signal).then((next) => {
      if (!controller.signal.aborted) {
        const all = [...next];
        setRuns(all);
        setSelection((current) => ({
          runAId: current.runAId || all[1]?.id || all[0]?.id || "",
          runBId: current.runBId || all[0]?.id || "",
        }));
      }
    }).catch(() => {
      // Navigation and React's development remount intentionally abort this optional picker load.
      // The pair request below owns the visible error state when the user already supplied ids.
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selection.runAId || !selection.runBId || selection.runAId === selection.runBId) {
      setState({ phase: "idle" });
      return;
    }
    const pairKey = `${selection.runAId}\u0000${selection.runBId}`;
    const controller = new AbortController();
    setState({ phase: "loading", pairKey });
    void Promise.all([
      studioApi.run(selection.runAId, controller.signal),
      studioApi.run(selection.runBId, controller.signal),
      studioApi.compare(selection.runAId, selection.runBId, controller.signal),
    ]).then(([runA, runB, comparison]) => {
      if (!controller.signal.aborted) setState({ phase: "ready", pairKey, specimen: projectComparisonSpecimen(runA, runB, recordedStructureFromComparison(comparison, runA, runB)) });
    }).catch((error) => {
      if (!controller.signal.aborted) setState({ phase: "error", pairKey, error: error instanceof Error ? error.message : "Comparison request failed." });
    });
    return () => controller.abort();
  }, [selection]);

  const selectPair = (side: "runAId" | "runBId", value: string) => {
    const next = { ...selection, [side]: value };
    setSelection(next);
    window.location.hash = routeHref({ surface: "compare", runA: next.runAId || undefined, runB: next.runBId || undefined });
  };
  const pairKey = `${selection.runAId}\u0000${selection.runBId}`;

  return <div className="comparison-host"><form className="comparison-picker" onSubmit={(event) => event.preventDefault()}>
    <label>Reference A<select value={selection.runAId} onChange={(event) => selectPair("runAId", event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>{recordedRunLabel(run)}</option>)}</select></label>
    <span aria-hidden="true">↔</span>
    <label>Candidate B<select value={selection.runBId} onChange={(event) => selectPair("runBId", event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>{recordedRunLabel(run)}</option>)}</select></label>
  </form>
  {selection.runAId === selection.runBId && <p className="surface-state">Choose two different recorded executions.</p>}
  {state.phase === "loading" && state.pairKey === pairKey && <p className="surface-state" role="status">Reading the recorded comparison…</p>}
  {state.phase === "error" && state.pairKey === pairKey && <p className="surface-state is-error" role="alert">Comparison unavailable · {state.error}</p>}
  {state.phase === "ready" && state.pairKey === pairKey && <CompareSurface specimen={state.specimen} initialDifferenceId={selectedDifference?.differenceId} onSelectionChange={(selected) => { window.location.hash = routeHref({ surface: "compare", runA: selected.runAId, runB: selected.runBId, selectedDifference: selected }); }} onInspect={(selected) => { window.location.hash = routeHref({ surface: "inspect", runId: selected.runBId, comparison: selected }); }} onTestThis={(selected) => { window.location.hash = routeHref({ surface: "time-travel", runId: selected.runBId, comparison: selected }); }} />}
  </div>;
}

function asObject(value: unknown): JsonObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as JsonObject : undefined;
}

function tokenPositionForOffset(tokens: readonly string[] | undefined, offset: number | undefined): number | undefined {
  if (!tokens?.length || offset === undefined) return undefined;
  let cursor = 0;
  for (let position = 0; position < tokens.length; position += 1) {
    const next = cursor + tokens[position].length;
    if (offset < next) return position;
    cursor = next;
  }
  return tokens.length - 1;
}

function TimeTravelDataSurface({ runId, mode, tokenPosition, breakpointId, rivalTokenId, answerLocus, sourceLocus, comparison }: Extract<StudioRoute, { surface: "time-travel" }>) {
  const [state, setState] = useState<
    | { phase: "loading" }
    | { phase: "unavailable"; error?: string }
    | { phase: "error"; error: string }
    | { phase: "ready"; recordedRun: RunRecord; family: readonly RunRecord[]; tokenRun: TimeTravelRun }
  >({ phase: runId ? "loading" : "unavailable", error: runId ? undefined : "Select a run from the journal." });
  const [live, setLive] = useState<{ position?: number; plan?: JsonObject; execution?: ForkExecution; children?: ChildForkResult[] }>({});
  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setState({ phase: "loading" });
    setLive({});
    void Promise.all([studioApi.run(runId, controller.signal), studioApi.rewindFidelity(runId, controller.signal), studioApi.family(runId, controller.signal)])
      .then(([recordedRun, fidelity, family]) => { if (!controller.signal.aborted) setState({ phase: "ready", recordedRun, family, tokenRun: projectTimeTravelRun(recordedRun, fidelity) }); })
      .catch((error) => { if (!controller.signal.aborted) setState({ phase: "error", error: error instanceof Error ? error.message : "Time Travel request failed." }); });
    return () => controller.abort();
  }, [runId]);
  if (!runId) return <section className="surface-placeholder"><span className="eyebrow">TIME TRAVEL</span><h1>Select a recorded execution</h1><p>Open a run from the journal, then choose Time Travel to inspect its retained response-token boundaries.</p><a className="primary-action" href="#/runs">Open Runs</a></section>;
  if (state.phase === "error") return <TimeTravelSurface phase="error" error={state.error} />;
  if (state.phase === "unavailable") return <TimeTravelSurface phase="unavailable" />;
  if (state.phase === "loading") return <TimeTravelSurface phase="loading" />;

  const position = tokenPosition ?? tokenPositionForOffset(state.recordedRun.responseTokens, answerLocus?.start ?? comparison?.b?.start) ?? 0;
  const interventions: ProposedIntervention[] = sourceLocus && answerLocus ? [{
    id: `remove:${sourceLocus.id}:${answerLocus.id}`,
    label: "Remove selected context span",
    summary: "Run the recorded answer again with this exact source/answer relationship neutralized by the backend counterfactual primitive.",
  }] : rivalTokenId !== undefined ? [{
    id: `alternative:${rivalTokenId}`,
    label: `Try recorded rival token #${rivalTokenId}`,
    summary: breakpointId ? `Force the retained rival at ${breakpointId}; the parent remains immutable.` : "Force the retained rival at this recorded boundary; the parent remains immutable.",
  }] : comparison ? [{
    id: `fan:${comparison.differenceId}`,
    label: "Explore recorded alternatives",
    summary: "Fan the retained alternatives at the output-difference boundary. Each successful branch remains a separate child for Compare.",
  }] : [];

  const visibleRun: TimeTravelRun = {
    ...state.tokenRun,
    ...(live.execution ? { execution: live.execution } : {}),
    ...(live.children?.length ? { children: [...(state.tokenRun.children ?? []), ...live.children] } : {}),
    ...(live.plan && live.position !== undefined ? {
      fidelityByBoundary: {
        ...state.tokenRun.fidelityByBoundary,
        [live.position]: {
          ...(state.tokenRun.fidelityByBoundary?.[live.position] ?? {
            reconstructedReplay: { state: "not_reported" as const },
            historicalExactProof: { state: "not_reported" as const },
          }),
          exactFork: { state: "ready_to_execute" as const, requirements: ["Live checkpoint and worker identity verified by the exact-resume planner."] },
        },
      },
    } : {}),
  };

  const checkExactFork = rivalTokenId === undefined ? undefined : async (selection: { position: number }) => {
    setLive((current) => ({ ...current, position: selection.position, execution: { state: "planning", detail: "Capturing a live checkpoint and verifying worker identity…", interventionRan: false } }));
    try {
      const checkpoint = await studioApi.captureCheckpoint(state.recordedRun.id);
      const checkpointReference = asObject(checkpoint.checkpoint_reference);
      if (!checkpointReference) throw new Error("The runtime did not return a checkpoint reference.");
      const plan = await studioApi.planExactFork(state.recordedRun.id, { position: selection.position, change: { type: "force_token", token_id: rivalTokenId } }, checkpointReference);
      if (plan.classification !== "exact_execution_fork") {
        const reasons = Array.isArray(plan.reasons) ? plan.reasons.map((reason) => asObject(reason)?.message).filter((value): value is string => typeof value === "string").join(" · ") : undefined;
        throw new Error(reasons || "The live planner did not classify this boundary as exact-executable.");
      }
      setLive({ position: selection.position, plan, execution: { state: "idle", detail: "Live exact-fork plan verified; execution has not started.", interventionRan: false } });
    } catch (error) {
      setLive({ position: selection.position, execution: { state: "failed", detail: error instanceof Error ? error.message : "Exact-fork planning failed.", interventionRan: false } });
    }
  };

  const branch = async (action: ForkAction) => {
    const proposal = interventions.find((candidate) => candidate.id === action.intervention.id) ?? action.intervention;
    setLive((current) => ({ ...current, position: action.position, execution: { state: action.mode === "exact" ? "control_verifying" : "executing", detail: action.mode === "exact" ? "Verifying the unchanged control before the intervention runs…" : "Running the declared reconstructed intervention…" } }));
    try {
      let result: JsonObject;
      if (action.mode === "exact") {
        if (!live.plan || live.position !== action.position) throw new Error("Check exact fork at this boundary before executing it.");
        result = await studioApi.executeExactFork(action.runId, live.plan);
      } else if (sourceLocus && answerLocus) {
        result = await studioApi.testThis(action.runId, { selection: { kind: "context_span", source_span_id: sourceLocus.id, answer_span_id: answerLocus.id }, test: { kind: "remove" } });
      } else if (rivalTokenId !== undefined) {
        result = await studioApi.testThis(action.runId, { selection: { kind: "response_token", position: action.position }, test: { kind: "try_alternative", token_id: rivalTokenId } });
      } else if (comparison) {
        result = await studioApi.testThis(action.runId, { selection: { kind: "response_token", position: action.position }, test: { kind: "fan_alternatives", limit: 4 } });
      } else throw new Error("This boundary has no declared intervention.");
      const nestedResult = asObject(result.result);
      const branchFan = asObject(nestedResult?.branch_fan);
      // Branch Fan is observation-first: it generates each recorded alternative as a
      // GeneratedObservation and creates no child run. Materializing one is a separate choice.
      const fanObservations = Array.isArray(branchFan?.branches) ? branchFan.branches.filter((raw) => {
        const branch = asObject(raw);
        return branch?.state === "completed" && typeof branch?.observation_id === "string";
      }).length : 0;
      const child = asObject(result.child);
      const childRunId = typeof result.child_run_id === "string" ? result.child_run_id : typeof nestedResult?.child_run_id === "string" ? nestedResult.child_run_id : typeof child?.id === "string" ? child.id : undefined;
      if (!childRunId && !fanObservations) {
        const reasons = Array.isArray(nestedResult?.reasons) ? nestedResult.reasons.map((reason) => asObject(reason)?.message).filter((value): value is string => typeof value === "string").join(" · ") : undefined;
        throw new Error(reasons || "The intervention completed without generated evidence.");
      }
      if (!childRunId) {
        setLive((current) => ({ ...current, children: [], execution: { state: "completed", detail: `${fanObservations} alternative${fanObservations === 1 ? " was" : "s were"} generated. Materialize one to compare it as a child run.`, interventionRan: true } }));
        return;
      }
      const createdChildren = [{ runId: childRunId, intervention: proposal, exactness: action.mode === "exact" ? "verified_exact" as const : "reconstructed" as const, summary: "Recorded child created by Test This." }];
      setLive((current) => ({ ...current, children: createdChildren, execution: { state: "completed", detail: `${createdChildren.length} child run${createdChildren.length === 1 ? " is" : "s are"} ready for comparison.`, interventionRan: true } }));
    } catch (error) {
      setLive((current) => ({ ...current, execution: { state: "failed", detail: error instanceof Error ? error.message : "Branch execution failed.", interventionRan: false } }));
    }
  };

  const linkedCoordinate = answerLocus || sourceLocus || comparison ? { answerId: answerLocus?.id, sourceId: sourceLocus?.id, differenceId: comparison?.differenceId, compareRunId: comparison?.runAId } : undefined;
  if (mode !== "token" && !answerLocus && tokenPosition === undefined) return <TurnTimeTravelSurface run={state.recordedRun} family={state.family} linkedSelection={linkedCoordinate} onOpenTokenExecution={(nextPosition) => { window.location.hash = routeHref({ surface: "time-travel", runId, mode: "token", tokenPosition: nextPosition, breakpointId, rivalTokenId, answerLocus, sourceLocus, comparison }); }} onOpenCompare={(parent, child) => { window.location.hash = routeHref({ surface: "compare", runA: parent, runB: child }); }} onInspectRun={(selectedRunId) => { window.location.hash = routeHref({ surface: "inspect", runId: selectedRunId }); }} />;

  return <div className="time-travel-data-host"><nav className="time-travel-view-switcher" aria-label="Time Travel depth"><button type="button" onClick={() => { window.location.hash = routeHref({ surface: "time-travel", runId, mode: "turn", answerLocus, sourceLocus, comparison }); }}>← Conversation strand</button><span>TOKEN EXECUTION</span></nav><TimeTravelSurface run={visibleRun} initialSelection={{ position, locusId: answerLocus?.id }} interventions={interventions} onCheckExactFork={checkExactFork} onBranch={branch} onOpenCompare={(child, parent) => { window.location.hash = routeHref({ surface: "compare", runA: parent, runB: child }); }} onInspectChildRun={(child) => { window.location.hash = routeHref({ surface: "inspect", runId: child }); }} /></div>;
}

function MriDataSurface({ runId }: { runId?: string }) {
  const [state, setState] = useState<
    | { phase: "loading" }
    | { phase: "error"; error: string }
    | { phase: "ready"; specimen: ReturnType<typeof projectRecordedMriSpecimen> }
  >({ phase: "loading" });
  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setState({ phase: "loading" });
    void studioApi.run(runId, controller.signal)
      .then((run) => { if (!controller.signal.aborted) setState({ phase: "ready", specimen: projectRecordedMriSpecimen(run) }); })
      .catch((error) => { if (!controller.signal.aborted) setState({ phase: "error", error: error instanceof Error ? error.message : "Model MRI request failed." }); });
    return () => controller.abort();
  }, [runId]);
  if (!runId) return <section className="surface-placeholder"><span className="eyebrow">MODEL MRI</span><h1>Select a recorded execution</h1><p>Model MRI requires a real run coordinate before it can report which internal instruments were retained or supported.</p><a className="primary-action" href="#/runs">Open Runs</a></section>;
  if (state.phase === "error") return <ModelMriSurface phase="error" error={state.error} />;
  if (state.phase === "loading") return <ModelMriSurface phase="loading" />;
  return <ModelMriSurface specimen={state.specimen} phase="ready" />;
}

function routeSurface(route: StudioRoute) {
  switch (route.surface) {
    case "runs": return <RunsSurface />;
    case "inspect": return <InspectSurface {...route} />;
    case "time-travel": return <TimeTravelDataSurface {...route} />;
    case "compare": return <CompareDataSurface runAId={route.runA} runBId={route.runB} selectedDifference={route.selectedDifference} />;
    case "mri": return <MriDataSurface runId={route.runId} />;
    case "runtime": return <RuntimeDataSurface />;
  }
}

export function App() {
  const route = useHashRoute();
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>("checking");

  useEffect(() => {
    const controller = new AbortController();
    Promise.allSettled([
      studioApi.health(controller.signal),
      studioApi.readiness(controller.signal),
    ]).then(([health, ready]) => {
      if (controller.signal.aborted) return;
      if (health.status === "rejected") return setRuntimeStatus("unreachable");
      if (ready.status === "rejected" || ready.value.status === "not_ready") return setRuntimeStatus("not-ready");
      setRuntimeStatus("ready");
    });
    return () => controller.abort();
  }, []);

  return <AppShell route={route} runtimeStatus={runtimeStatus}>{routeSurface(route)}</AppShell>;
}
