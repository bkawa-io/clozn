import { useEffect, useMemo, useRef, useState } from "react";
import {
  differenceTextParts,
  selectedDifference,
  usableDifferences,
  type ComparedExecution,
  type CompareSelection,
  type ComparisonSpecimen,
  type LocalEvidence,
  type StructuralDifference,
} from "./model";
import "./compare.css";

export type { CompareSelection, ComparisonSpecimen, LocalEvidence, PairRelationship, StructuralDifference } from "./model";

export interface CompareSurfaceProps {
  specimen: ComparisonSpecimen;
  initialDifferenceId?: string;
  /** Lets the host retain a user-selected recorded region in route state without re-deriving it. */
  onSelectionChange?: (selection: CompareSelection) => void;
  /** Carries stable region and run coordinates into the recorded evidence reader. */
  onInspect?: (selection: CompareSelection) => void;
  /** Starts an explicit experiment from the currently selected, recorded region. */
  onTestThis?: (selection: CompareSelection) => void;
}

function selectionFor(specimen: ComparisonSpecimen, difference: StructuralDifference): CompareSelection {
  return { runAId: specimen.a.id, runBId: specimen.b.id, differenceId: difference.id, a: difference.a, b: difference.b };
}

function OutputMeta({ execution }: { execution: ComparedExecution }) {
  const facts = [execution.model, execution.recordedAt].filter((value): value is string => Boolean(value));
  return <p className="compare-execution-meta">{facts.length ? facts.join(" · ") : "Recorded execution"}</p>;
}

function OutputReader({ execution, side, differences, selectedId, onSelect }: {
  execution: ComparedExecution;
  side: "a" | "b";
  differences: readonly StructuralDifference[];
  selectedId?: string;
  onSelect: (id: string) => void;
}) {
  if (execution.outputState !== "available" || execution.output === undefined) {
    const message = execution.outputState === "redacted"
      ? "Recorded output is redacted; no replacement text is shown."
      : "Readable output was not retained for this execution.";
    return <p className="compare-output-absence">{message}</p>;
  }
  return <p className="compare-prose">
    {differenceTextParts(execution.output, differences, side).map((part, index) => part.differenceId ? (
      <button
        type="button"
        key={`${part.differenceId}-${index}`}
        className={`compare-prose-difference${selectedId === part.differenceId ? " is-selected" : ""}`}
        aria-pressed={selectedId === part.differenceId}
        aria-label={`Select recorded difference: ${part.text}`}
        onClick={() => onSelect(part.differenceId!)}
      >{part.text}</button>
    ) : <span key={`plain-${index}`}>{part.text}</span>)}
  </p>;
}

function Registration({ specimen }: { specimen: ComparisonSpecimen }) {
  const relation = specimen.relationship;
  return <section className="compare-registration" aria-label="Comparison registration">
    <div><span>PAIR</span><strong>{relation.kind === "related" ? "Related recorded runs" : "Arbitrary A / B runs"}</strong><p>{relation.detail ?? (relation.kind === "related" ? "Recorded lineage relationship." : "No lineage relationship is asserted for this pair.")}</p></div>
    {relation.kind === "related" ? <>
      <div><span>INTERVENTION</span><strong>{relation.intervention?.label ?? "Not recorded"}</strong><p>{relation.intervention?.detail ?? "No intervention coordinate was retained."}</p></div>
      <div><span>CHANGED CONDITIONS</span><strong>{relation.changedConditions?.length ? relation.changedConditions.join(" · ") : "Not recorded"}</strong><p>Conditions and output divergence are separate recorded facts.</p></div>
    </> : <><div><span>CHANGED CONDITIONS</span><strong>{relation.changedConditions?.length ? relation.changedConditions.join(" · ") : "None reported"}</strong><p>Conditions and output divergence are separate recorded facts.</p></div><div className="compare-registration-caveat"><span>LINEAGE</span><strong>Not inferred</strong><p>Similarity of outputs or settings does not establish ancestry.</p></div></>}
  </section>;
}

function Spine({ differences, selectedId, onSelect }: { differences: readonly StructuralDifference[]; selectedId?: string; onSelect: (id: string) => void }) {
  if (!differences.length) return <aside className="compare-spine is-empty" aria-label="Difference navigation">No recorded regions</aside>;
  return <aside className="compare-spine" aria-label="Difference navigation">
    <span className="compare-spine-label">OUTPUT DIFFERENCES</span>
    <div className="compare-spine-track" aria-hidden="true" />
    <ol>
      {differences.map((difference, index) => <li key={difference.id} style={{ top: `${10 + (80 * index) / Math.max(1, differences.length - 1)}%` }}>
        <button type="button" className={selectedId === difference.id ? "is-selected" : undefined} aria-label={`${difference.isFirstOutputDivergence ? "First output divergence, " : ""}${difference.label}`} aria-pressed={selectedId === difference.id} onClick={() => onSelect(difference.id)}>
          <i /> <span>{index + 1}</span>
        </button>
      </li>)}
    </ol>
    <p>Output regions</p>
  </aside>;
}

function EvidencePanel({ evidence }: { evidence?: LocalEvidence }) {
  if (!evidence) return <section className="compare-inspector-section"><span>LOCAL EVIDENCE</span><p>No local evidence was requested for this region.</p></section>;
  if (evidence.state !== "available") return <section className={`compare-inspector-section compare-evidence-absence is-${evidence.state}`}><span>LOCAL EVIDENCE · {evidence.state.replace("_", " ")}</span><p>{evidence.reason ?? "No usable local evidence was returned."}</p></section>;
  return <section className="compare-inspector-section"><span>LOCAL EVIDENCE</span>{evidence.observations?.length ? <dl className="compare-observations">{evidence.observations.map((observation) => <div key={`${observation.label}-${observation.value}`}><dt>{observation.label}</dt><dd>{observation.value}{observation.provenance && <small>{observation.provenance}</small>}</dd></div>)}</dl> : <p>Recorded evidence is available, but this response contains no displayable observations.</p>}<p className="compare-evidence-method">{evidence.method ?? "Recorded local evidence"} · Observations are not causal proof.</p></section>;
}

function Inspector({ specimen, difference, onInspect, onTestThis }: { specimen: ComparisonSpecimen; difference?: StructuralDifference; onInspect?: (selection: CompareSelection) => void; onTestThis?: (selection: CompareSelection) => void }) {
  if (!difference) return <aside className="compare-inspector is-rest" aria-live="polite"><span>DIFFERENCE INSPECTOR</span><h2>Select a recorded difference</h2><p>The spine is a navigation aid. It does not explain why the executions differ.</p></aside>;
  const selection = selectionFor(specimen, difference);
  return <aside className="compare-inspector" aria-live="polite">
    <header><span>DIFFERENCE INSPECTOR</span><h2>{difference.label}</h2>{difference.isFirstOutputDivergence && <em>First recorded output divergence</em>}</header>
    <section className="compare-inspector-section"><span>STRUCTURAL REGISTRATION</span><dl className="compare-structural-facts"><div><dt>Change</dt><dd>{difference.kind}</dd></div><div><dt>Alignment</dt><dd>{difference.alignment}</dd></div></dl><p>{difference.alignmentDetail ?? (difference.alignment === "recorded" ? "Recorded alignment supports navigation between these regions; it is not semantic equivalence." : difference.alignment === "ambiguous" ? "This correspondence is ambiguous and should not be read as one-to-one linguistic matching." : "No usable alignment was recorded for this region.")}</p></section>
    <EvidencePanel evidence={specimen.evidenceByDifferenceId?.[difference.id]} />
    {(onInspect || onTestThis) && <footer className="compare-continuity-actions">{onInspect && <button type="button" onClick={() => onInspect(selection)}>Inspect evidence</button>}{onTestThis && <button type="button" className="compare-test-action" onClick={() => onTestThis(selection)}>Test this</button>}</footer>}
  </aside>;
}

export function CompareSurface({ specimen, initialDifferenceId, onSelectionChange, onInspect, onTestThis }: CompareSurfaceProps) {
  const differences = useMemo(() => usableDifferences(specimen), [specimen]);
  const initial = differences.some((difference) => difference.id === initialDifferenceId) ? initialDifferenceId : differences[0]?.id;
  const [selectedId, setSelectedId] = useState<string | undefined>(initial);
  const previousSpecimen = useRef(specimen);
  useEffect(() => {
    if (previousSpecimen.current !== specimen) { previousSpecimen.current = specimen; setSelectedId(differences.some((difference) => difference.id === initialDifferenceId) ? initialDifferenceId : differences[0]?.id); }
  }, [differences, initialDifferenceId, specimen]);
  const selected = selectedDifference(specimen, selectedId);
  const selectedIndex = differences.findIndex((difference) => difference.id === selectedId);
  const selectDifference = (id: string) => {
    setSelectedId(id);
    const difference = selectedDifference(specimen, id);
    if (difference) onSelectionChange?.(selectionFor(specimen, difference));
  };
  const move = (delta: number) => { if (differences.length) selectDifference(differences[(selectedIndex + delta + differences.length) % differences.length].id); };

  return <main className="compare-surface" aria-labelledby="compare-title">
    <header className="compare-heading"><div><span className="eyebrow">RECORDED EXECUTION PAIR</span><h1 id="compare-title">What changed?</h1><p>Read the two recorded outputs side by side, then inspect only the registered output regions.</p></div><span className="compare-count">{differences.length} output region{differences.length === 1 ? "" : "s"}</span></header>
    <Registration specimen={specimen} />
    <section className="compare-reader" aria-label="Synchronized output reader">
      <article className="compare-output-pane"><header><span>EXECUTION A</span><h2>{specimen.a.label}</h2><OutputMeta execution={specimen.a} /></header><OutputReader execution={specimen.a} side="a" differences={differences} selectedId={selectedId} onSelect={selectDifference} /></article>
      <Spine differences={differences} selectedId={selectedId} onSelect={selectDifference} />
      <article className="compare-output-pane"><header><span>EXECUTION B</span><h2>{specimen.b.label}</h2><OutputMeta execution={specimen.b} /></header><OutputReader execution={specimen.b} side="b" differences={differences} selectedId={selectedId} onSelect={selectDifference} /></article>
    </section>
    <div className="compare-detail-row"><p className="compare-navigation-note">Alignment is navigation evidence, not semantic equivalence or causal evidence.</p>{differences.length > 1 && <nav aria-label="Difference navigation"><button type="button" onClick={() => move(-1)}>Previous</button><span>{selectedIndex + 1} / {differences.length}</span><button type="button" onClick={() => move(1)}>Next</button></nav>}</div>
    <Inspector specimen={specimen} difference={selected} onInspect={onInspect} onTestThis={onTestThis} />
  </main>;
}
