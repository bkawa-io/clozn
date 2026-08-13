import { useEffect, useMemo, useRef, useState } from "react";
import { EvidenceState, ProvenanceCaption, TestThisLauncher } from "../../components/investigation";
import type { EvidenceState as EvidenceStateData } from "../../core/investigation";
import {
  textFragments,
  type ContextDocument,
  type DecisionLocus,
  type InfluenceSelection,
  type LinkedReaderSpecimen,
  type RelatedContextLocus,
  type TextLocus,
} from "./model";
import "./linked-reader.css";

interface LinkedEvidenceReaderProps {
  specimen: LinkedReaderSpecimen;
  loadSelection: (locus: TextLocus, signal: AbortSignal) => Promise<InfluenceSelection>;
  loadTensionSelection?: (locus: TextLocus, signal: AbortSignal) => Promise<InfluenceSelection>;
  tensionSelections?: Readonly<Record<string, InfluenceSelection>>;
  decisionLoci?: readonly DecisionLocus[];
  initialLocusId?: string;
}

type InspectMode = "influence" | "tension" | "decisions";

const EMPTY_TENSION_SELECTIONS: Readonly<Record<string, InfluenceSelection>> = Object.freeze({});

interface ContextLinkTarget {
  answerId: string;
  source: RelatedContextLocus;
}

interface ContextLink extends TextLocus {
  key: string;
  sourceId: string;
  documentId: string;
  effect: RelatedContextLocus["effect"] | "mixed";
  targets: ContextLinkTarget[];
}

function relationshipKey(source: RelatedContextLocus): string {
  return `${source.documentId}\u0000${source.id}\u0000${source.start}\u0000${source.end}`;
}

function contextLinkCatalog(selections: Readonly<Record<string, InfluenceSelection>>): ContextLink[] {
  const links = new Map<string, ContextLink>();
  for (const [answerId, selection] of Object.entries(selections)) {
    if (selection.state !== "available") continue;
    for (const source of selection.related) {
      const key = relationshipKey(source);
      const existing = links.get(key);
      if (existing) {
        const target = existing.targets.find((candidate) => candidate.answerId === answerId);
        if (!target) {
          existing.targets.push({ answerId, source });
        } else if (Math.abs(source.deltaNats) > Math.abs(target.source.deltaNats)) {
          target.source = source;
        }
        if (existing.effect !== source.effect) existing.effect = "mixed";
      } else {
        links.set(key, {
          id: `context-link:${key}`,
          key,
          sourceId: source.id,
          documentId: source.documentId,
          start: source.start,
          end: source.end,
          effect: source.effect,
          targets: [{ answerId, source }],
        });
      }
    }
  }
  return [...links.values()];
}

function SequenceOverview({ specimen, selectedIds }: { specimen: LinkedReaderSpecimen; selectedIds: readonly string[] }) {
  const answerLength = Math.max(1, specimen.answer.length);
  const atomicCount = specimen.answerLoci.reduce((count, locus) => count + (locus.memberIds?.length ?? 1), 0);
  return <section className="sequence-overview" aria-labelledby="sequence-overview-title">
    <header><div><span className="eyebrow">SEQUENCE OVERVIEW</span><h2 id="sequence-overview-title">Prompt to response registration</h2></div><p>{specimen.context.length} context block{specimen.context.length === 1 ? "" : "s"} · {specimen.answerLoci.length} selection phrases · {atomicCount} stable coordinates</p></header>
    <div className="sequence-overview__tracks">
      <div className="sequence-overview__context" aria-label="Recorded context blocks">{specimen.context.map((document, index) => <i key={document.id} title={document.label} style={{ flexGrow: Math.max(1, document.text?.length ?? 1) }}><span>{index + 1}</span></i>)}</div>
      <div className="sequence-overview__flow" aria-hidden="true"><span>RECORDED ORDER</span><b>→</b></div>
      <div className="sequence-overview__answer" aria-label="Measured answer coordinates">{specimen.answerLoci.map((locus) => <i key={locus.id} className={selectedIds.includes(locus.id) ? "is-selected" : undefined} style={{ left: `${Math.min(100, locus.start / answerLength * 100)}%`, width: `${Math.max(.4, (locus.end - locus.start) / answerLength * 100)}%` }} />)}</div>
    </div>
  </section>;
}

function AnswerProse({
  text,
  loci,
  activeIds,
  lockedIds,
  onHover,
  onLock,
}: {
  text: string;
  loci: TextLocus[];
  activeIds: readonly string[];
  lockedIds: readonly string[];
  onHover: (id?: string) => void;
  onLock: (id: string) => void;
}) {
  return (
    <p className="reader-prose answer-prose">
      {textFragments(text, loci).map((part, index) => part.locus ? (
        <button
          type="button"
          key={part.locus.id}
          className={`${activeIds.includes(part.locus.id) ? "is-active" : ""}${lockedIds.includes(part.locus.id) ? " is-locked" : ""}`}
          aria-pressed={lockedIds.includes(part.locus.id)}
          aria-label={`${part.text}, measured answer locus`}
          onMouseEnter={() => onHover(part.locus?.id)}
          onMouseLeave={() => onHover(undefined)}
          onFocus={() => onHover(part.locus?.id)}
          onBlur={() => onHover(undefined)}
          onClick={() => onLock(part.locus!.id)}
        >
          {part.text}
        </button>
      ) : <span key={`plain-${index}`}>{part.text}</span>)}
    </p>
  );
}

function ContextProse({ document, loci, activeKeys, focusedKey, lockedKey, onHover, onLock }: {
  document: ContextDocument;
  loci: ContextLink[];
  activeKeys: ReadonlySet<string>;
  focusedKey?: string;
  lockedKey?: string;
  onHover: (key?: string) => void;
  onLock: (key: string) => void;
}) {
  if (document.state !== "available" || document.text == null) {
    return <p className={`context-document-state is-${document.state}`}>{document.detail ?? `${document.state} context`}</p>;
  }
  return (
    <p className="reader-prose context-prose">
      {textFragments(document.text, loci).map((part, index) => part.locus ? (() => {
        const link = part.locus as ContextLink;
        return <button
          type="button"
          key={link.key}
          className={`context-link is-${link.effect}${activeKeys.has(link.key) ? " is-related" : ""}${focusedKey === link.key ? " is-focused" : ""}${lockedKey === link.key ? " is-locked" : ""}`}
          aria-pressed={lockedKey === link.key}
          aria-label={`${part.text}, linked context span, ${link.targets.length} related answer phrase${link.targets.length === 1 ? "" : "s"}`}
          data-context-link={link.key}
          onMouseEnter={() => onHover(link.key)}
          onMouseLeave={() => onHover(undefined)}
          onFocus={() => onHover(link.key)}
          onBlur={() => onHover(undefined)}
          onClick={() => onLock(link.key)}
        >
          {part.text}
        </button>;
      })() : <span key={`plain-${index}`}>{part.text}</span>)}
    </p>
  );
}

function SelectionInspector({ runId, mode, answerLocus, selection, source }: {
  runId: string;
  mode: Exclude<InspectMode, "decisions">;
  answerLocus?: TextLocus;
  selection?: InfluenceSelection;
  source?: RelatedContextLocus;
}) {
  if (!answerLocus) {
    return (
      <aside className="linked-inspector is-rest">
        <span className="eyebrow">DETAILS ON DEMAND</span>
        <h2>Select either side</h2>
        <p>{mode === "tension" ? "Choose a measured answer phrase or context span to register the opposing relationship in either direction." : "Hover an answer phrase or a subtle measured context span to reveal its links. Click either side to lock the relationship for navigation and evidence detail."}</p>
      </aside>
    );
  }
  const evidence: EvidenceStateData | undefined = selection?.state === "loading" ? undefined : {
    measurement: selection?.state === "available"
      ? { kind: "measured", finding: source?.evidenceState === "causally_supported" ? "supported" : "unsupported", value: source?.deltaNats }
      : selection?.state === "not_measured"
        ? { kind: "not-measured", reason: selection.reason }
        : selection?.state === "unavailable"
          ? { kind: "unavailable", reason: selection.reason }
          : { kind: "failed", reason: selection?.reason },
    artifactMode: "recorded",
    exactness: { kind: "unverified" },
  };
  return (
    <aside className="linked-inspector" aria-live="polite">
      <span className="eyebrow">COMPACT INSPECTOR</span>
      <h2>{selection?.state === "loading" ? "Reading recorded evidence…" : mode === "tension" ? "Selected tension pair" : "Selected relationship"}</h2>
      {selection && selection.state !== "available" ? (
        <div className={`selection-absence is-${selection.state}`}>
          {evidence && <EvidenceState state={evidence} />}
          <p>{selection.reason ?? "No usable recorded measurement was returned for this locus."}</p>
        </div>
      ) : source ? (
        <>
          <dl className="relationship-facts">
            <div><dt>Effect</dt><dd>{source.effect}</dd></div>
            <div><dt>Δ nats</dt><dd>{source.deltaNats >= 0 ? "+" : ""}{source.deltaNats.toFixed(4)}</dd></div>
            <div><dt>Evidence</dt><dd>{source.evidenceState.replace("_", " ")}</dd></div>
            <div><dt>Method</dt><dd>{selection?.method ?? "Recorded influence map"}</dd></div>
          </dl>
          <ProvenanceCaption
            className="linked-provenance"
            method={selection?.method ?? "Recorded influence map"}
            measurementFloor={selection?.floorNats === undefined ? undefined : `${selection.floorNats} nats`}
            artifactMode="recorded"
            exactness={{ kind: "unverified" }}
          />
          <TestThisLauncher
            className="linked-test-launcher"
            locus={{ kind: "answer-span", runId, answerId: source.answerLocusId ?? answerLocus.memberIds?.[0] ?? answerLocus.id, startChar: answerLocus.start, endChar: answerLocus.end }}
            onLaunch={() => {
              // Both loci use the browser's UTF-16 coordinates here. They identify the exact reader
              // selection that initiated an explicit test; Time Travel does not silently guess a token
              // boundary when a text range cannot be resolved against recorded tokens.
              const query = new URLSearchParams({
                mode: "token",
                answer: source.answerLocusId ?? answerLocus.memberIds?.[0] ?? answerLocus.id,
                answerStart: String(answerLocus.start),
                answerEnd: String(answerLocus.end),
                source: source.id,
                sourceStart: String(source.start),
                sourceEnd: String(source.end),
              });
              window.location.hash = `/time-travel/${encodeURIComponent(runId)}?${query.toString()}`;
            }}
          />
        </>
      ) : selection?.state === "available" ? (
        <p>No measured context link was returned for this selected range.</p>
      ) : null}
    </aside>
  );
}

function DecisionLociPanel({ runId, response, loci }: { runId: string; response: string; loci: readonly DecisionLocus[] }) {
  const [selectedId, setSelectedId] = useState(loci[0]?.id);
  const selected = loci.find((locus) => locus.id === selectedId);
  const textLoci: TextLocus[] = loci.flatMap((locus) => locus.start === undefined || locus.end === undefined ? [] : [{ id: locus.id, start: locus.start, end: locus.end }]);
  return <div className="decision-loci-workbench">
    <section className="decision-loci-answer" aria-labelledby="decision-answer-title"><header><span className="eyebrow">RECORDED ANSWER</span><h2 id="decision-answer-title">Close calls in readable context</h2><p>Every mark comes from the backend close-call detector. It is a test location, not a verdict.</p></header><p className="reader-prose">{textFragments(response, textLoci).map((part, index) => part.locus ? <mark key={part.locus.id} className={part.locus.id === selectedId ? "is-selected" : undefined}>{part.text}</mark> : <span key={index}>{part.text}</span>)}</p></section>
    <section className="decision-loci-list" aria-labelledby="decision-list-title"><header><div><span className="eyebrow">DECISION LOCI</span><h2 id="decision-list-title">Recorded close calls</h2></div><strong>{loci.length}</strong></header>{loci.length ? <ol>{loci.map((locus) => <li key={locus.id}><button type="button" className={locus.id === selectedId ? "is-selected" : undefined} onClick={() => setSelectedId(locus.id)}><span>Token {locus.position}</span><strong>{locus.emittedToken?.trim() || `#${locus.position}`}</strong><small>{locus.emittedProbability === undefined || locus.rivalProbability === undefined ? "Recorded near-tie" : `${(locus.emittedProbability * 100).toFixed(1)}% / ${(locus.rivalProbability * 100).toFixed(1)}%`}</small>{locus.meaningful && <em>meaning-changing heuristic</em>}</button></li>)}</ol> : <p className="context-document-state">No close calls were returned for this recorded answer.</p>}</section>
    <aside className="decision-loci-inspector"><span className="eyebrow">COMPACT INSPECTOR</span>{selected ? <><h2>Token boundary {selected.position}</h2><dl className="relationship-facts"><div><dt>Emitted</dt><dd>{selected.emittedToken?.trim() || "retained token"}</dd></div><div><dt>Emitted p</dt><dd>{selected.emittedProbability?.toFixed(3) ?? "not reported"}</dd></div><div><dt>Rival p</dt><dd>{selected.rivalProbability?.toFixed(3) ?? "not reported"}</dd></div><div><dt>Margin</dt><dd>{selected.margin?.toFixed(3) ?? "not reported"}</dd></div></dl><a className="primary-action" href={`#/time-travel/${encodeURIComponent(runId)}?mode=token&position=${selected.position}&breakpoint=${encodeURIComponent(selected.id)}${selected.rivalTokenId === undefined ? "" : `&rival=${selected.rivalTokenId}`}`}>Test this boundary</a></> : <><h2>No close call selected</h2><p>The detector did not return a recorded token decision.</p></>}</aside>
  </div>;
}

export function LinkedEvidenceReader({ specimen, loadSelection, loadTensionSelection, tensionSelections = EMPTY_TENSION_SELECTIONS, decisionLoci = [], initialLocusId }: LinkedEvidenceReaderProps) {
  const [mode, setMode] = useState<InspectMode>("influence");
  const [hoveredId, setHoveredId] = useState<string>();
  const [lockedId, setLockedId] = useState<string | undefined>(initialLocusId);
  const [hoveredContextKey, setHoveredContextKey] = useState<string>();
  const [lockedContextKey, setLockedContextKey] = useState<string>();
  const [selectionById, setSelectionById] = useState<Record<string, InfluenceSelection>>({});
  const [tensionById, setTensionById] = useState<Record<string, InfluenceSelection>>({ ...tensionSelections });
  const [sourceIndex, setSourceIndex] = useState(0);
  const contextViewport = useRef<HTMLDivElement>(null);
  const selectionMap = mode === "tension" ? tensionById : selectionById;
  const contextLinks = useMemo(() => contextLinkCatalog(selectionMap), [selectionMap]);
  const contextLinkByKey = useMemo(() => new Map(contextLinks.map((link) => [link.key, link])), [contextLinks]);
  const contextFocus = contextLinkByKey.get(lockedContextKey ?? hoveredContextKey ?? "");
  const directActiveId = lockedId ?? hoveredId;
  const activeIds = contextFocus ? [...new Set(contextFocus.targets.map((target) => target.answerId))] : directActiveId ? [directActiveId] : [];
  const activeId = activeIds[0];
  const activeLocus = specimen.answerLoci.find((locus) => locus.id === activeId);
  const selection = activeId ? selectionMap[activeId] : undefined;
  const related = selection?.state === "available" ? selection.related : [];
  const contextTarget = contextFocus?.targets.find((target) => target.answerId === activeId) ?? contextFocus?.targets[0];
  const contextSourceIndex = contextTarget ? related.findIndex((source) => relationshipKey(source) === relationshipKey(contextTarget.source)) : -1;
  const currentSourceIndex = contextSourceIndex >= 0 ? contextSourceIndex : sourceIndex;
  const focusedSource = contextTarget?.source ?? related[currentSourceIndex];
  const activeContextKeys = useMemo(() => {
    const ids = new Set(activeIds);
    return new Set(contextLinks.filter((link) => link.targets.some((target) => ids.has(target.answerId))).map((link) => link.key));
  }, [activeIds, contextLinks]);
  const lockedAnswerIds = contextLinkByKey.get(lockedContextKey ?? "")?.targets.map((target) => target.answerId) ?? (lockedId ? [lockedId] : []);

  useEffect(() => {
    setLockedId(initialLocusId);
    setHoveredId(undefined);
    setLockedContextKey(undefined);
    setHoveredContextKey(undefined);
  }, [initialLocusId, specimen.runId]);

  useEffect(() => {
    const controller = new AbortController();
    setSelectionById(Object.fromEntries(specimen.answerLoci.map((locus) => [locus.id, { state: "loading" as const, related: [] }])));
    for (const locus of specimen.answerLoci) {
      void loadSelection(locus, controller.signal).then((next) => {
        if (!controller.signal.aborted) setSelectionById((current) => ({ ...current, [locus.id]: next }));
      }).catch((error) => {
        if (!controller.signal.aborted) setSelectionById((current) => ({
          ...current,
          [locus.id]: { state: "error", reason: error instanceof Error ? error.message : "Evidence request failed.", related: [] },
        }));
      });
    }
    return () => controller.abort();
  }, [loadSelection, specimen.answerLoci, specimen.runId]);

  useEffect(() => setTensionById({ ...tensionSelections }), [tensionSelections]);

  useEffect(() => {
    if (mode !== "tension" || !activeLocus || tensionById[activeLocus.id] || !loadTensionSelection) return;
    const controller = new AbortController();
    setTensionById((current) => ({ ...current, [activeLocus.id]: { state: "loading", related: [] } }));
    void loadTensionSelection(activeLocus, controller.signal).then((next) => {
      if (!controller.signal.aborted) setTensionById((current) => ({ ...current, [activeLocus.id]: next }));
    }).catch((error) => {
      if (!controller.signal.aborted) setTensionById((current) => ({ ...current, [activeLocus.id]: { state: "error", reason: error instanceof Error ? error.message : "Tension request failed.", related: [] } }));
    });
    return () => controller.abort();
  }, [activeLocus, loadTensionSelection, mode]);

  useEffect(() => setSourceIndex(0), [activeId]);

  const relatedByDocument = useMemo(() => {
    const grouped = new Map<string, ContextLink[]>();
    for (const locus of contextLinks) grouped.set(locus.documentId, [...(grouped.get(locus.documentId) ?? []), locus]);
    return grouped;
  }, [contextLinks]);

  function navigateSource(next: number) {
    if (!related.length) return;
    const wrapped = (next + related.length) % related.length;
    setSourceIndex(wrapped);
    if (lockedContextKey) setLockedContextKey(contextLinks.find((link) => link.key === relationshipKey(related[wrapped]))?.key);
    requestAnimationFrame(() => {
      contextViewport.current?.querySelector<HTMLElement>(`[data-context-link="${CSS.escape(relationshipKey(related[wrapped]))}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  }

  return (
    <section className="linked-reader" aria-labelledby="linked-reader-title">
      <header className="surface-heading linked-reader-heading">
        <div>
          <span className="eyebrow">CONTEXT ↔ ANSWER</span>
          <h1 id="linked-reader-title">Linked evidence reader</h1>
          <p>Readable language stays primary. Switch instruments without losing the recorded run coordinate.</p>
        </div>
        <div className="inspect-mode-controls" aria-label="Investigation instrument"><button type="button" aria-pressed={mode === "influence"} onClick={() => setMode("influence")}>Influence</button><button type="button" aria-pressed={mode === "tension"} onClick={() => setMode("tension")}>Context tension</button><button type="button" aria-pressed={mode === "decisions"} onClick={() => setMode("decisions")}>Decision loci <span>{decisionLoci.length}</span></button>{(lockedId || lockedContextKey) && <button type="button" onClick={() => { setLockedId(undefined); setHoveredId(undefined); setLockedContextKey(undefined); setHoveredContextKey(undefined); }}>Clear selection</button>}</div>
      </header>
      <SequenceOverview specimen={specimen} selectedIds={activeIds} />

      {mode === "decisions" ? <DecisionLociPanel runId={specimen.runId} response={specimen.answer} loci={decisionLoci} /> : <><div className="linked-reader-body">
        <section className="reader-pane context-pane" aria-labelledby="context-pane-title">
          <header><span className="eyebrow">WHAT THE MODEL SAW</span><h2 id="context-pane-title">Context</h2></header>
          <div className="reader-scroll" ref={contextViewport}>
            {specimen.context.map((document) => (
              <article key={document.id} data-context-document={document.id}>
                <header><h3>{document.label}</h3><span>{document.text?.length.toLocaleString() ?? document.state.toUpperCase()}</span></header>
                <ContextProse
                  document={document}
                  loci={relatedByDocument.get(document.id) ?? []}
                  activeKeys={activeContextKeys}
                  focusedKey={contextFocus?.key}
                  lockedKey={lockedContextKey}
                  onHover={setHoveredContextKey}
                  onLock={(key) => {
                    setLockedContextKey((current) => current === key ? undefined : key);
                    setLockedId(undefined);
                    setHoveredId(undefined);
                  }}
                />
              </article>
            ))}
          </div>
        </section>

        <aside className="relationship-registration" aria-label="Related context navigation">
          <div className="registration-line" aria-hidden="true">
            {related.map((locus, index) => <i key={`${locus.documentId}:${locus.id}:${locus.effect}:${index}`} className={`${index === currentSourceIndex ? "is-current" : ""} is-${locus.effect}`} style={{ top: `${12 + (index / Math.max(1, related.length - 1)) * 76}%` }} />)}
          </div>
          {related.length > 0 ? (
            <div className="registration-controls">
              <button type="button" onClick={() => navigateSource(currentSourceIndex - 1)} aria-label="Previous related context locus">↑</button>
              <span>{currentSourceIndex + 1} / {related.length}</span>
              <button type="button" onClick={() => navigateSource(currentSourceIndex + 1)} aria-label="Next related context locus">↓</button>
            </div>
          ) : <span className="registration-rest">{mode === "tension" ? "SELECT TENSION" : "SELECT A LOCUS"}</span>}
        </aside>

        <section className="reader-pane answer-pane" aria-labelledby="answer-pane-title">
          <header><span className="eyebrow">WHAT IT ANSWERED</span><h2 id="answer-pane-title">Recorded answer</h2></header>
          <div className="reader-scroll">
            <AnswerProse
              text={specimen.answer}
              loci={specimen.answerLoci}
              activeIds={activeIds}
              lockedIds={lockedAnswerIds}
              onHover={setHoveredId}
              onLock={(id) => {
                setLockedId((current) => current === id ? undefined : id);
                setLockedContextKey(undefined);
                setHoveredContextKey(undefined);
              }}
            />
          </div>
        </section>
      </div>

      <SelectionInspector runId={specimen.runId} mode={mode} answerLocus={activeLocus} selection={selection} source={focusedSource} /></>}
    </section>
  );
}
