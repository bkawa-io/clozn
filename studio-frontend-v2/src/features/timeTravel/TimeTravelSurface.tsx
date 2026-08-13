import { useEffect, useMemo, useState } from "react";
import {
  boundaryForLocus,
  canStageFork,
  fidelityAt,
  type BoundaryFidelity,
  type BoundarySelection,
  type ChildForkResult,
  type ForkAction,
  type ProposedIntervention,
  type TimeTravelPhase,
  type TimeTravelRun,
} from "./model";
import "./time-travel.css";

export type {
  Availability,
  BoundaryFidelity,
  BoundarySelection,
  ChildForkResult,
  ExactForkAvailability,
  ForkAction,
  ForkExecution,
  ForkExecutionState,
  HistoricalExactProof,
  HistoricalExactness,
  ProposedIntervention,
  RecordedResponseToken,
  ReplayAvailability,
  TimeTravelLocus,
  TimeTravelPhase,
  TimeTravelRun,
} from "./model";

export interface TimeTravelSurfaceProps {
  run?: TimeTravelRun;
  phase?: TimeTravelPhase;
  error?: string;
  /** Controlled current coordinate. Supplying a locus retains a deep-linked investigation context. */
  selection?: BoundarySelection;
  /** Used only while uncontrolled; a locus takes precedence when it resolves in the run. */
  initialSelection?: { position?: number; locusId?: string };
  interventions?: readonly ProposedIntervention[];
  onSelectBoundary?: (selection: BoundarySelection) => void;
  /** Request the live exact-fork plan. It must not execute an intervention. */
  onCheckExactFork?: (selection: BoundarySelection) => void;
  /** Explicit execution boundary after a proposal has been staged in this surface. */
  onBranch?: (action: ForkAction) => void;
  onOpenCompare?: (childRunId: string, parentRunId: string) => void;
  onInspectChildRun?: (childRunId: string) => void;
}

function stateLabel(state: string): string {
  return state.replaceAll("_", " ").toUpperCase();
}

function stateTone(state: string): "success" | "warning" | "danger" | "neutral" | "info" {
  if (state === "available" || state === "verified" || state === "verified_exact" || state === "completed") return "success";
  if (state === "requires_live_plan" || state === "planning" || state === "control_verifying" || state === "executing") return "warning";
  if (state === "unavailable" || state === "control_diverged" || state === "failed") return "danger";
  if (state === "ready_to_execute") return "info";
  return "neutral";
}

function boundaryLabel(position: number): string {
  return position === 0 ? "Prompt boundary · before token 0" : `Before token ${position} · after token ${position - 1}`;
}

function Status({ state }: { state: string }) {
  return <span className={`time-travel-status is-${stateTone(state)}`}>{stateLabel(state)}</span>;
}

function PresentationState({ phase, error }: { phase: TimeTravelPhase; error?: string }) {
  if (phase === "loading") return <p className="time-travel-presentation-state" role="status">Reading the recorded execution…</p>;
  if (phase === "error") return <p className="time-travel-presentation-state is-error" role="alert">Time Travel is unavailable · {error ?? "The recorded run could not be read."}</p>;
  if (phase === "unavailable") return <p className="time-travel-presentation-state">Token-boundary history is not available for this run.</p>;
  if (phase === "stale") return <p className="time-travel-presentation-state" role="status">Showing a recorded snapshot; live exact-fork status may have changed.</p>;
  return null;
}

function FidelityRow({ label, state, detail }: { label: string; state: string; detail?: string }) {
  return <div className="time-travel-fidelity-row"><dt>{label}</dt><dd><Status state={state} />{detail && <small>{detail}</small>}</dd></div>;
}

function FidelityInspector({ fidelity }: { fidelity?: BoundaryFidelity }) {
  if (!fidelity) return <section className="time-travel-card"><h3>Fidelity</h3><p className="time-travel-absence">No rewind-fidelity projection was recorded for this boundary.</p></section>;
  const historicalDetail = fidelity.historicalExactProof.state === "verified"
    ? [fidelity.historicalExactProof.verifiedExecutionCount !== undefined ? `${fidelity.historicalExactProof.verifiedExecutionCount} verified execution${fidelity.historicalExactProof.verifiedExecutionCount === 1 ? "" : "s"}` : undefined, fidelity.historicalExactProof.detail].filter(Boolean).join(" · ")
    : fidelity.historicalExactProof.detail;
  return <section className="time-travel-card" aria-labelledby="fidelity-title"><h3 id="fidelity-title">Fidelity at this boundary</h3><dl className="time-travel-fidelity-list">
    <FidelityRow label="Reconstructed replay" state={fidelity.reconstructedReplay.state} detail={fidelity.reconstructedReplay.reason} />
    <FidelityRow label="Exact fork" state={fidelity.exactFork.state} detail={fidelity.exactFork.reason} />
    <FidelityRow label="Historical exact proof" state={fidelity.historicalExactProof.state} detail={historicalDetail} />
  </dl>
  {fidelity.exactFork.state === "requires_live_plan" && <p className="time-travel-note">Exact rewind may be possible; live verification is still required. A recorded selection never restores live model state.</p>}
  {fidelity.exactFork.state === "unavailable" && <p className="time-travel-note">An unavailable exact attempt does not switch to reconstruction. Reconstructed replay remains a separately labeled experiment.</p>}
  {fidelity.historicalExactProof.state === "verified" && <p className="time-travel-caption">Historical proof records a past completed control match; it does not establish exact execution now.</p>}
  </section>;
}

function ReconstructionDetails({ fidelity }: { fidelity?: BoundaryFidelity }) {
  const differences = fidelity?.reconstructedReplay.unavoidableDifferences;
  if (!differences?.length) return null;
  return <section className="time-travel-card"><h3>Reconstruction differences</h3><ul className="time-travel-differences">{differences.map((difference) => <li key={difference}>{difference.replaceAll("_", " ")}</li>)}</ul></section>;
}

function ExecutionState({ run }: { run: TimeTravelRun }) {
  const execution = run.execution;
  if (!execution || execution.state === "idle") return null;
  const interventionRan = execution.interventionRan;
  return <section className={`time-travel-execution is-${stateTone(execution.state)}`} aria-live="polite"><div><span className="eyebrow">EXECUTION STATUS</span><h3>{stateLabel(execution.state)}</h3></div><Status state={execution.state} />
    {execution.detail && <p>{execution.detail}</p>}
    {execution.state === "control_diverged" && <p><strong>The intervention was not run.</strong> The unchanged control must reproduce token IDs and decoded text before an exact intervention can execute.</p>}
    {interventionRan === false && execution.state !== "control_diverged" && <p>The intervention did not run.</p>}
  </section>;
}

function ChildResults({ children, parentRunId, onOpenCompare, onInspectChildRun }: { children?: readonly ChildForkResult[]; parentRunId: string; onOpenCompare?: TimeTravelSurfaceProps["onOpenCompare"]; onInspectChildRun?: TimeTravelSurfaceProps["onInspectChildRun"] }) {
  if (!children?.length) return null;
  return <section className="time-travel-children" aria-labelledby="time-travel-children-title"><header><div><span className="eyebrow">RECORDED CHILDREN</span><h3 id="time-travel-children-title">Child run results</h3></div><span>{children.length}</span></header><ol>{children.map((child) => <li key={child.runId}><div className="time-travel-child-summary"><code>{child.runId}</code><Status state={child.exactness} /><p>{child.intervention.label}{child.intervention.summary ? ` · ${child.intervention.summary}` : ""}</p>{child.summary && <small>{child.summary}</small>}</div><div className="time-travel-child-actions">{onOpenCompare && <button type="button" className="time-travel-primary" onClick={() => onOpenCompare(child.runId, parentRunId)}>Open in Compare</button>}{onInspectChildRun && <button type="button" onClick={() => onInspectChildRun(child.runId)}>Inspect child</button>}</div></li>)}</ol><p className="time-travel-caption">Compare owns downstream divergence. A child result does not establish that a difference is meaningful.</p></section>;
}

export function TimeTravelSurface({ run, phase = "ready", error, selection, initialSelection, interventions, onSelectBoundary, onCheckExactFork, onBranch, onOpenCompare, onInspectChildRun }: TimeTravelSurfaceProps) {
  const locusBoundary = run ? boundaryForLocus(run, selection?.locusId ?? initialSelection?.locusId) : undefined;
  const defaultPosition = selection?.position ?? locusBoundary ?? initialSelection?.position ?? run?.responseTokens?.[0]?.position;
  const [localPosition, setLocalPosition] = useState<number | undefined>(defaultPosition);
  const [staged, setStaged] = useState<{ intervention: ProposedIntervention; mode: ForkAction["mode"] }>();
  const position = selection?.position ?? localPosition;
  const fidelity = run ? fidelityAt(run, position) : undefined;
  const selectedLocusId = selection?.locusId ?? initialSelection?.locusId;
  const selectedLocus = useMemo(() => run?.loci?.find((locus) => locus.id === selectedLocusId), [run?.loci, selectedLocusId]);

  useEffect(() => { setLocalPosition(defaultPosition); setStaged(undefined); }, [run?.id, defaultPosition]);

  const selectBoundary = (nextPosition: number) => {
    if (!run) return;
    setLocalPosition(nextPosition);
    setStaged(undefined);
    onSelectBoundary?.({ runId: run.id, position: nextPosition, locusId: run.loci?.find((locus) => locus.boundaryPosition === nextPosition)?.id });
  };
  const stage = (intervention: ProposedIntervention, mode: ForkAction["mode"]) => setStaged({ intervention, mode });
  const execute = () => {
    if (!run || position === undefined || !staged) return;
    onBranch?.({ runId: run.id, position, ...staged });
  };

  return <main className="time-travel-surface"><header className="time-travel-heading"><div><span className="eyebrow">RECORDED EXECUTION / TIME TRAVEL</span><h1>Token-boundary investigation</h1><p>{run ? <>Run <code>{run.id}</code>{run.model && <> · {run.model}</>}</> : "Select a recorded run to inspect its response boundaries."}</p></div>{run?.parentRunId ? <p className="time-travel-lineage">Child of <code>{run.parentRunId}</code></p> : run && <p className="time-travel-lineage">Immutable original</p>}</header>
    <PresentationState phase={phase} error={error} />
    {run && phase !== "error" && <div className="time-travel-workbench"><section className="time-travel-recording" aria-labelledby="recorded-answer-title"><header><span className="eyebrow">RECORDED ANSWER</span><h2 id="recorded-answer-title">Original response</h2><p>The original run is preserved. Selecting a boundary only inspects recorded history.</p></header><article className="time-travel-answer">{run.response || <span className="time-travel-absence">No recorded response text is available.</span>}</article>
      {selectedLocus && <p className="time-travel-locus">Linked locus: <strong>{selectedLocus.label}</strong> · boundary {selectedLocus.boundaryPosition}</p>}
      <section className="time-travel-rail" aria-labelledby="token-boundary-title"><header><div><span className="eyebrow">RECORDED RESPONSE TOKENS</span><h3 id="token-boundary-title">Token boundaries</h3></div>{position !== undefined && <span className="time-travel-coordinate">{boundaryLabel(position)}</span>}</header>{run.responseTokens?.length ? <div className="time-travel-token-scroll"><div className="time-travel-token-list" role="listbox" aria-label="Recorded response token boundaries">{run.responseTokens.map((token) => <button type="button" role="option" key={token.position} aria-label={`Boundary ${token.position}, before token ${token.position}: ${token.text || "empty token"}`} aria-selected={position === token.position} className={position === token.position ? "is-selected" : undefined} onClick={() => selectBoundary(token.position)}><span>{token.position}</span><b>{token.text || "∅"}</b></button>)}</div></div> : <p className="time-travel-absence">Recorded token boundaries were not retained for this run.</p>}<p className="time-travel-caption">Blue Ion marks the selected recorded boundary. Historical proof and current exact execution are separate states.</p></section></section>
      <aside className="time-travel-inspector" aria-labelledby="time-travel-inspector-title"><header><span className="eyebrow">TOKEN BOUNDARY</span><h2 id="time-travel-inspector-title">{position === undefined ? "No boundary selected" : boundaryLabel(position)}</h2><p>Use a recorded coordinate to plan a new child; this is not a live rewind control.</p></header>{position !== undefined && <><FidelityInspector fidelity={fidelity} /><ReconstructionDetails fidelity={fidelity} />
        <section className="time-travel-card time-travel-actions"><h3>Branch proposal</h3>{!interventions?.length ? <p className="time-travel-absence">No intervention proposal is available for this boundary.</p> : <><p className="time-travel-caption">Stage a declared intervention before any execution request is made.</p><div className="time-travel-proposals">{interventions.map((intervention) => <div key={intervention.id}><div><strong>{intervention.label}</strong>{intervention.summary && <small>{intervention.summary}</small>}</div><div>{canStageFork(fidelity, "exact") && <button type="button" onClick={() => stage(intervention, "exact")}>Stage exact fork</button>}{canStageFork(fidelity, "reconstructed") && <button type="button" onClick={() => stage(intervention, "reconstructed")}>Stage reconstructed replay</button>}</div></div>)}</div></>}
          {fidelity?.exactFork.state === "requires_live_plan" && onCheckExactFork && <button type="button" className="time-travel-primary" onClick={() => onCheckExactFork({ runId: run.id, position, locusId: selectedLocus?.id })}>Check exact fork</button>}
          {staged && <div className="time-travel-staged"><span className="eyebrow">STAGED — NOT EXECUTED</span><strong>{staged.intervention.label}</strong><p>{staged.intervention.summary ?? "No additional proposal detail was recorded."}</p><p>Mode: <b>{staged.mode === "exact" ? "exact fork" : "reconstructed replay"}</b></p><button type="button" className="time-travel-primary" onClick={execute} disabled={!onBranch}>Branch here</button></div>}
        </section><ExecutionState run={run} /><ChildResults children={run.children} parentRunId={run.id} onOpenCompare={onOpenCompare} onInspectChildRun={onInspectChildRun} /></>}</aside></div>}
  </main>;
}
