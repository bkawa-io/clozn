import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { loadRunConcepts, loadRunInspection } from "../../data/api";
import type {
  ObservatoryData,
  RunConcepts,
  RuntimeState,
  SourceReading,
  TokenReading,
} from "../../data/types";
import { aggregateSources, buildResponseClaims, type SourceAggregate } from "./analysis";
import { LensSelectionInspector, type LensInspectorSelection } from "./LensSelectionInspector";
import { RunWorkspaceHeader } from "./RunWorkspaceHeader";
import "./LensReader.css";

interface LensProps {
  runtime: RuntimeState;
  initialRunId?: string;
}

export type ReaderLensId = "shakiness" | "sources" | "concepts";

type ConceptState =
  | { status: "idle" | "loading" }
  | { status: "done"; data: RunConcepts };

type ReaderDrawer =
  | { kind: "source"; sourceId: string }
  | { kind: "token"; index: number }
  | { kind: "span"; start: number; end: number; claimIndex: number };

/**
 * The one thing currently under the mouse or keyboard focus (hover wins), OR failing that, whatever
 * the drawer is pinned to. Drives both the cross-highlight classes and the connector lines -- there
 * is only ever one active focus, so hovering something else always previews it in place of the pin.
 */
type FocusTarget =
  | { kind: "source"; sourceId: string }
  | { kind: "claim"; index: number };

interface Connector {
  id: string;
  path: string;
  effect: "supports" | "suppresses" | "neutral" | "observed";
}

const LENSES: Array<{ id: ReaderLensId; label: string; description: string }> = [
  { id: "shakiness", label: "Shakiness", description: "Underline uncertain text using rhythm and weight." },
  {
    id: "sources",
    label: "Sources",
    description: "Mark input and output spans with measured source links, including effects too "
      + "small to call on their own. Hover or click a span to see exactly which spans it connects to.",
  },
  { id: "concepts", label: "Concepts", description: "Mark tokens with available internal concept readouts." },
];

function readableToken(value: string) {
  return value.replaceAll("\t", "    ");
}

function inputSources(data: ObservatoryData | null): SourceReading[] {
  if (!data) return [];
  const recorded = data.contextSources ?? data.sources;
  if (recorded.length) return recorded;
  return data.prompt
    ? [{ id: "recorded-prompt", role: "user", kind: "message", label: "Recorded prompt", text: data.prompt }]
    : [];
}

/**
 * The messages to read, in delivery order. Falls back to the measurement's spans only when the run
 * has no message layer at all -- never render both, or a refined span repeats its parent's words.
 */
function inputMessages(data: ObservatoryData | null, sources: readonly SourceReading[]) {
  const messages = data?.contextMessages ?? [];
  return messages.length ? messages : sources;
}

/**
 * Split one message's text at the boundaries of the coarse spans measured inside it. Fine spans
 * (a refinement of a coarse span, so `groupId` names that span rather than the message) are left
 * out: they cover ground their parent already covers, and drawing both would double the prose.
 */
function messagePieces(message: SourceReading, sources: readonly SourceReading[]) {
  const text = message.text ?? "";
  const spans = sources
    .filter((source) => source.groupId === message.id
      && typeof source.start === "number"
      && typeof source.end === "number"
      && source.end > source.start)
    .sort((left, right) => (left.start ?? 0) - (right.start ?? 0));

  const pieces: Array<{ text: string; source?: SourceReading }> = [];
  let cursor = 0;
  for (const span of spans) {
    const start = Math.max(cursor, Math.min(span.start ?? 0, text.length));
    const end = Math.max(start, Math.min(span.end ?? 0, text.length));
    if (start > cursor) pieces.push({ text: text.slice(cursor, start) });
    if (end > start) pieces.push({ text: text.slice(start, end), source: span });
    cursor = Math.max(cursor, end);
  }
  if (cursor < text.length) pieces.push({ text: text.slice(cursor) });
  return pieces.length ? pieces : [{ text }];
}

/** The role, plus a label only when it says something the role does not (the backend sends both). */
function messageLabel(message: SourceReading) {
  const role = (message.role ?? "").toLowerCase();
  const label = message.label;
  if (!label || label.toLowerCase() === role) return undefined;
  return label;
}

function sourceIndex(sources: readonly SourceReading[], sourceId: string) {
  const index = sources.findIndex((source) => source.id === sourceId);
  return index < 0 ? 0 : index;
}

/**
 * Per-token decoration only -- shakiness rhythm and concept marks. Source links are no longer a
 * token-level concern: they mark whole claims/spans (see the output pane), the same granularity the
 * input side already reads at. Rendering a source badge on every one of a long answer's tokens was
 * the exact "hard to reason about" problem this redesign exists to fix.
 */
export function tokenLensClasses(token: TokenReading, activeLenses: ReadonlySet<ReaderLensId>) {
  const classes = ["lens-reader-token"];
  if (activeLenses.has("shakiness") && token.band) classes.push(`is-${token.band}`);
  if (activeLenses.has("concepts")) classes.push("concept-lens-active");
  return classes.join(" ");
}

function initialRun(runtime: RuntimeState, requested?: string) {
  return requested || runtime.runs[0]?.id || "";
}

/** The strongest-magnitude source in an already-sorted aggregate list; undefined for an unlinked claim. */
function dominantAggregate(aggregates: readonly SourceAggregate[]) {
  return aggregates[0];
}

/**
 * Which sources are the single strongest explanation for at least one token in a range -- the basis
 * for cross-highlighting, deliberately stricter than "this source has any nonzero link at all."
 * Over a multi-token claim, nearly every source in a run picks up SOME incidental link below the
 * measurement floor; gating on "any link, however weak" saturates to "every source lights up for
 * every claim," which defeats the point of hovering. "Ever the strongest explanation for one of
 * this claim's tokens" gives the same selectivity a reader gets from reading one token at a time,
 * rolled up across the claim -- confirmed against a live run where it correctly drops both
 * irrelevant distractor sources that the naive "any link" basis wrongly included in every claim.
 */
function dominantSourcesInRange(tokens: readonly TokenReading[], start: number, end: number) {
  const sourceIds = new Set<string>();
  for (let index = Math.max(0, start); index <= Math.min(end, tokens.length - 1); index += 1) {
    const links = [...(tokens[index].sources ?? []), ...(tokens[index].observedSources ?? [])];
    if (!links.length) continue;
    const strongest = links.reduce((best, link) =>
      Math.abs(link.deltaNats) > Math.abs(best.deltaNats) ? link : best);
    sourceIds.add(strongest.sourceId);
  }
  return sourceIds;
}

/** A one-hop cubic bezier from a rect on the left (input) to a rect on the right (output). */
function connectorPath(root: DOMRect, from: DOMRect, to: DOMRect) {
  const x1 = from.right - root.left;
  const y1 = from.top + from.height / 2 - root.top;
  const x2 = to.left - root.left;
  const y2 = to.top + to.height / 2 - root.top;
  const mid = x1 + Math.max(42, (x2 - x1) * .48);
  return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
}

/** Below-floor links draw as `observed` regardless of sign -- the connector's color must never
 * imply a causal claim the measurement didn't clear. */
function connectorEffect(aggregate: SourceAggregate): Connector["effect"] {
  return aggregate.clearTokenCount > 0 ? aggregate.effect : "observed";
}

export function Lens({ runtime, initialRunId }: LensProps) {
  const [runId, setRunId] = useState(() => initialRun(runtime, initialRunId));
  const [data, setData] = useState<ObservatoryData | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [activeLenses, setActiveLenses] = useState<Set<ReaderLensId>>(() => new Set());
  const [drawer, setDrawer] = useState<ReaderDrawer | null>(null);
  const [hover, setHover] = useState<FocusTarget | null>(null);
  const [concepts, setConcepts] = useState<ConceptState>({ status: "idle" });
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const readerSurface = useRef<HTMLDivElement | null>(null);
  const inputPane = useRef<HTMLElement | null>(null);
  const outputPane = useRef<HTMLElement | null>(null);
  const sourceNodes = useRef(new Map<string, HTMLElement>());
  const claimNodes = useRef(new Map<number, HTMLElement>());

  useEffect(() => {
    if (!runId && runtime.runs.length) setRunId(initialRun(runtime, initialRunId));
  }, [initialRunId, runId, runtime]);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setStatus("loading");
    setDrawer(null);
    setConcepts({ status: "idle" });
    void loadRunInspection(runId, controller.signal).then((inspection) => {
      if (controller.signal.aborted) return;
      setData(inspection);
      setStatus("idle");
      history.replaceState(null, "", `#/runs/${encodeURIComponent(runId)}/lens`);
    }).catch(() => {
      if (!controller.signal.aborted) {
        setData(null);
        setStatus("error");
      }
    });
    return () => controller.abort();
  }, [runId]);

  const conceptsActive = activeLenses.has("concepts");
  useEffect(() => {
    if (!conceptsActive || !data || concepts.status !== "idle") return;
    const controller = new AbortController();
    setConcepts({ status: "loading" });
    void loadRunConcepts(data.id, undefined, controller.signal).then((result) => {
      if (!controller.signal.aborted) setConcepts({ status: "done", data: result });
    }).catch(() => {
      if (!controller.signal.aborted) {
        setConcepts({
          status: "done",
          data: { available: false, reason: "Concept evidence is unavailable.", availableLayers: [], tokens: [], readouts: [] },
        });
      }
    });
    return () => controller.abort();
  }, [concepts.status, conceptsActive, data]);

  const sources = useMemo(() => inputSources(data), [data]);
  const messages = useMemo(() => inputMessages(data, sources), [data, sources]);
  const measuredSpanCount = useMemo(
    () => sources.filter((source) => source.measured).length,
    [sources],
  );
  const tokens = data?.tokens ?? [];
  const claims = useMemo(() => buildResponseClaims(tokens), [tokens]);
  const claimAggregates = useMemo(
    () => claims.map((claim) => aggregateSources(tokens, claim.start, claim.end)),
    [claims, tokens],
  );
  // The materiality filter used for cross-highlighting -- see dominantSourcesInRange. Kept separate
  // from claimAggregates (which stays the complete picture for the badge tone and the drawer).
  const claimDominantSources = useMemo(
    () => claims.map((claim) => dominantSourcesInRange(tokens, claim.start, claim.end)),
    [claims, tokens],
  );
  // The reverse index: which claims does a given input source connect to? Built once per map so a
  // hovered input span can look up its related claims in constant time instead of re-scanning them.
  const claimsBySource = useMemo(() => {
    const map = new Map<string, number[]>();
    claimDominantSources.forEach((sourceIds, index) => {
      for (const sourceId of sourceIds) {
        const list = map.get(sourceId);
        if (list) list.push(index);
        else map.set(sourceId, [index]);
      }
    });
    return map;
  }, [claimDominantSources]);

  const selectedRun = runtime.runs.find((run) => run.id === runId) ?? null;
  const conceptsData = concepts.status === "done" ? concepts.data : null;
  const conceptsAligned = Boolean(
    data?.response
    && conceptsData?.available
    && conceptsData.tokens.length === tokens.length
    && conceptsData.tokens.join("") === data.response,
  );

  // Hover always wins; with nothing hovered, whatever the drawer is pinned to stays lit so the
  // connections stay visible while reading its detail. Neither depends on the "Sources" toggle --
  // that toggle only controls the always-on passive marks below, not this per-interaction preview.
  const focus: FocusTarget | null = useMemo(() => {
    if (hover) return hover;
    if (drawer?.kind === "source") return { kind: "source", sourceId: drawer.sourceId };
    if (drawer?.kind === "span") return { kind: "claim", index: drawer.claimIndex };
    return null;
  }, [hover, drawer]);

  const relatedSourceIds = useMemo(() => {
    if (focus?.kind !== "claim") return new Set<string>();
    return claimDominantSources[focus.index] ?? new Set<string>();
  }, [focus, claimDominantSources]);

  const relatedClaimIndexes = useMemo(() => {
    if (focus?.kind !== "source") return new Set<number>();
    return new Set(claimsBySource.get(focus.sourceId) ?? []);
  }, [focus, claimsBySource]);

  useEffect(() => {
    if (!focus || !readerSurface.current) {
      setConnectors([]);
      return;
    }

    let frame = 0;
    const update = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const surface = readerSurface.current;
        if (!surface) return;
        const root = surface.getBoundingClientRect();
        const next: Connector[] = [];
        if (focus.kind === "source") {
          const from = sourceNodes.current.get(focus.sourceId)?.getBoundingClientRect();
          if (from) {
            for (const claimIndex of relatedClaimIndexes) {
              const claimNode = claimNodes.current.get(claimIndex);
              const aggregate = claimAggregates[claimIndex]?.find((item) => item.sourceId === focus.sourceId);
              if (!claimNode || !aggregate) continue;
              next.push({
                id: `claim-${claimIndex}`,
                path: connectorPath(root, from, claimNode.getBoundingClientRect()),
                effect: connectorEffect(aggregate),
              });
            }
          }
        } else {
          const to = claimNodes.current.get(focus.index)?.getBoundingClientRect();
          const dominantSourceIds = claimDominantSources[focus.index] ?? new Set<string>();
          if (to) {
            for (const aggregate of claimAggregates[focus.index] ?? []) {
              if (!dominantSourceIds.has(aggregate.sourceId)) continue;
              const sourceNode = sourceNodes.current.get(aggregate.sourceId);
              if (!sourceNode) continue;
              next.push({
                id: aggregate.sourceId,
                path: connectorPath(root, sourceNode.getBoundingClientRect(), to),
                effect: connectorEffect(aggregate),
              });
            }
          }
        }
        setConnectors(next.slice(0, 24));
      });
    };

    update();
    const panes = [inputPane.current, outputPane.current].filter((pane): pane is HTMLElement => Boolean(pane));
    panes.forEach((pane) => pane.addEventListener("scroll", update, { passive: true }));
    window.addEventListener("resize", update);
    const resizeObserver = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    if (resizeObserver && readerSurface.current) resizeObserver.observe(readerSurface.current);
    return () => {
      cancelAnimationFrame(frame);
      panes.forEach((pane) => pane.removeEventListener("scroll", update));
      window.removeEventListener("resize", update);
      resizeObserver?.disconnect();
    };
  }, [focus, relatedClaimIndexes, claimAggregates, claimDominantSources]);

  function toggleLens(id: ReaderLensId) {
    setActiveLenses((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function openToken(index: number) {
    setDrawer({ kind: "token", index });
  }

  function openSource(sourceId: string) {
    setDrawer({ kind: "source", sourceId });
  }

  function openSpan(start: number, end: number, claimIndex: number) {
    setDrawer({ kind: "span", start, end, claimIndex });
  }

  function focusSource(sourceId: string) {
    setHover({ kind: "source", sourceId });
  }

  function unfocusSource(sourceId: string) {
    setHover((current) => (current?.kind === "source" && current.sourceId === sourceId ? null : current));
  }

  function focusClaim(index: number) {
    setHover({ kind: "claim", index });
  }

  function unfocusClaim(index: number) {
    setHover((current) => (current?.kind === "claim" && current.index === index ? null : current));
  }

  const inspectorSelection: LensInspectorSelection | null = drawer?.kind === "source"
    ? {
      kind: "source",
      sourceId: drawer.sourceId,
      source: sources.find((source) => source.id === drawer.sourceId),
    }
    : drawer?.kind === "span"
      ? { kind: "span", start: drawer.start, end: drawer.end, claimIndex: drawer.claimIndex }
      : drawer?.kind === "token"
        ? { kind: "token", index: drawer.index }
        : null;

  return (
    <section className="lens-reader-page" aria-label="Run Lens">
      <div className="lens-reader-toolbar">
        <div className="lens-reader-toggles" role="group" aria-label="Text lenses">
          <button
            type="button"
            className={activeLenses.size === 0 ? "is-active" : ""}
            aria-pressed={activeLenses.size === 0}
            onClick={() => setActiveLenses(new Set())}
          >Clean</button>
          {LENSES.map((lens) => (
            <button
              type="button"
              className={activeLenses.has(lens.id) ? "is-active" : ""}
              aria-pressed={activeLenses.has(lens.id)}
              title={lens.description}
              onClick={() => toggleLens(lens.id)}
              key={lens.id}
            >{lens.label}</button>
          ))}
        </div>
        <a className="lens-reader-diagnostics-link" href={runId ? `#/runs/${encodeURIComponent(runId)}/diagnostics` : "#/diagnostics"}>
          Diagnostics
        </a>
      </div>

      <div className="lens-reader-notices">
        {activeLenses.has("sources") && !data?.influenceMethod && (
          <p className="lens-reader-availability" role="note">
            Source influence has not been measured for this run. The reader will not infer relationships.
          </p>
        )}
        {conceptsActive && concepts.status === "done" && !conceptsData?.available && (
          <p className="lens-reader-availability" role="note">{conceptsData?.reason ?? "Concept evidence is unavailable."}</p>
        )}
      </div>

      <div className="lens-reader-surface" ref={readerSurface}>
        <section className="lens-reader-pane lens-reader-input" ref={inputPane} aria-labelledby="lens-reader-input-title">
          <header>
            <h2 id="lens-reader-input-title">Input</h2>
            <small>
              {messages.length} {messages.length === 1 ? "message" : "messages"}
              {measuredSpanCount > 0 && ` · ${measuredSpanCount} measured`}
            </small>
          </header>
          <div className="lens-reader-document">
            {status === "loading" && <p className="lens-reader-state">Loading input…</p>}
            {status === "error" && <p className="lens-reader-state is-error">This run could not be loaded.</p>}
            {status === "idle" && !messages.length && <p className="lens-reader-state">No readable input was recorded.</p>}
            {messages.map((message) => {
              const label = messageLabel(message);
              return (
                <article className="lens-reader-message" key={message.id}>
                  <header className="lens-reader-message-meta">
                    <span>{message.role ?? message.kind ?? "Input"}</span>
                    {label && <small>{label}</small>}
                  </header>
                  <p className="lens-reader-message-text">
                    {messagePieces(message, sources).map((piece, pieceIndex) => {
                      if (!piece.source) return <span key={pieceIndex}>{piece.text}</span>;
                      const source = piece.source;
                      const tone = sourceIndex(sources, source.id);
                      const marked = activeLenses.has("sources");
                      const isSelected = drawer?.kind === "source" && drawer.sourceId === source.id;
                      const isFocused = focus?.kind === "source" && focus.sourceId === source.id;
                      const isRelated = relatedSourceIds.has(source.id);
                      return (
                        <button
                          type="button"
                          className={[
                            "lens-reader-input-span",
                            marked ? `source-tone-${tone % 6}` : "",
                            isSelected ? "is-selected" : "",
                            isFocused ? "is-focused" : "",
                            isRelated ? "is-related" : "",
                          ].filter(Boolean).join(" ")}
                          ref={(node) => {
                            if (node) sourceNodes.current.set(source.id, node);
                            else sourceNodes.current.delete(source.id);
                          }}
                          aria-pressed={isSelected}
                          onClick={() => openSource(source.id)}
                          onMouseEnter={() => focusSource(source.id)}
                          onMouseLeave={() => unfocusSource(source.id)}
                          onFocus={() => focusSource(source.id)}
                          onBlur={() => unfocusSource(source.id)}
                          key={source.id}
                        >
                          {piece.text}
                          {marked && <sup className={`source-tone-${tone % 6}`}>{tone + 1}</sup>}
                        </button>
                      );
                    })}
                  </p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="lens-reader-pane lens-reader-output" ref={outputPane} aria-labelledby="lens-reader-output-title">
          <header>
            <h2 id="lens-reader-output-title">Output</h2>
            <small>
              {claims.length} {claims.length === 1 ? "claim" : "claims"} · {tokens.length} tokens
            </small>
          </header>
          <div className="lens-reader-document lens-reader-output-document">
            {status === "loading" && <p className="lens-reader-state">Loading output…</p>}
            {status === "error" && <p className="lens-reader-state is-error">This run could not be loaded.</p>}
            {status === "idle" && !tokens.length && (
              <p className="lens-reader-plain-output">{data?.response || "No readable output was recorded."}</p>
            )}
            {tokens.length > 0 && (
              <div className="lens-reader-output-stream">
                {claims.map((claim, index) => {
                  const aggregates = claimAggregates[index] ?? [];
                  const dominant = dominantAggregate(aggregates);
                  const marked = activeLenses.has("sources") && Boolean(dominant);
                  const dominantClears = dominant ? dominant.clearTokenCount > 0 : false;
                  const tone = dominant ? sourceIndex(sources, dominant.sourceId) : 0;
                  const isSelected = drawer?.kind === "span" && drawer.claimIndex === index;
                  const isFocused = focus?.kind === "claim" && focus.index === index;
                  const isRelated = relatedClaimIndexes.has(index);
                  return (
                    <button
                      type="button"
                      className={[
                        "lens-reader-claim",
                        marked ? (dominantClears ? `has-source-clear source-tone-${tone % 6}` : "has-source-observed") : "",
                        isSelected ? "is-selected" : "",
                        isFocused ? "is-focused" : "",
                        isRelated ? "is-related" : "",
                      ].filter(Boolean).join(" ")}
                      title={[
                        `Claim ${index + 1}`,
                        claim.meanConfidence == null ? null : `mean confidence ${Math.round(claim.meanConfidence * 100)}%`,
                        claim.shakyCount ? `${claim.shakyCount} shaky token${claim.shakyCount === 1 ? "" : "s"}` : null,
                      ].filter(Boolean).join(" · ")}
                      ref={(node) => {
                        if (node) claimNodes.current.set(index, node);
                        else claimNodes.current.delete(index);
                      }}
                      aria-pressed={isSelected}
                      onClick={() => openSpan(claim.start, claim.end, index)}
                      onMouseEnter={() => focusClaim(index)}
                      onMouseLeave={() => unfocusClaim(index)}
                      onFocus={() => focusClaim(index)}
                      onBlur={() => unfocusClaim(index)}
                      key={`claim-${claim.start}`}
                    >
                      {tokens.slice(claim.start, claim.end + 1).map((token, offset) => {
                        const globalIndex = claim.start + offset;
                        const concept = conceptsAligned ? conceptsData?.readouts[globalIndex]?.[0] : undefined;
                        return (
                          <span
                            className={[
                              tokenLensClasses(token, activeLenses),
                              concept ? "has-concept" : "",
                            ].filter(Boolean).join(" ")}
                            key={globalIndex}
                          >
                            {readableToken(token.text)}
                            {conceptsActive && concept && <i className="lens-reader-concept-mark" aria-hidden="true" />}
                          </span>
                        );
                      })}
                      {marked && dominant && <sup className={`source-tone-${tone % 6}`}>{tone + 1}</sup>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        {connectors.length > 0 && (
          <svg className="lens-reader-connectors" aria-hidden="true">
            {connectors.map((connector) => (
              <path className={`effect-${connector.effect}`} d={connector.path} key={connector.id} />
            ))}
          </svg>
        )}
      </div>

      {drawer && (
        <aside className="lens-reader-drawer" aria-label="Selected span details">
          <div className="lens-reader-drawer-bar">
            <span>Selection detail</span>
            <button type="button" onClick={() => setDrawer(null)} aria-label="Close selection drawer">Close</button>
          </div>
          <LensSelectionInspector
            selection={inspectorSelection}
            tokens={tokens}
            sources={sources}
            influenceAbsence={data?.influenceAbsence}
            influenceMethod={data?.influenceMethod}
            influenceThresholds={data?.influenceThresholds}
            contextCoverage={data?.contextCoverage}
            tokenTrace={tokens.length
              ? { state: "available", provenance: "recorded" }
              : { state: "not_captured", detail: "No token trace was retained." }}
            sourceEvidence={data?.influenceMethod
              ? { state: "available", provenance: "measured" }
              : { state: "not_measured", detail: "Source influence has not been measured for this run." }}
            onSelectToken={openToken}
            onSelectSource={openSource}
          />
        </aside>
      )}

      <RunWorkspaceHeader run={selectedRun} active="lens" />
    </section>
  );
}
