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
import { LensSelectionInspector, type LensInspectorSelection } from "./LensSelectionInspector";
import { RunWorkspaceHeader } from "./RunWorkspaceHeader";
import "./LensReader.css";

interface LensProps {
  runtime: RuntimeState;
  initialRunId?: string;
}

export type ReaderLensId = "shakiness" | "sources" | "provenance" | "concepts";

type ConceptState =
  | { status: "idle" | "loading" }
  | { status: "done"; data: RunConcepts };

type ReaderDrawer =
  | { kind: "source"; sourceId: string }
  | { kind: "token"; index: number };

interface Connector {
  id: string;
  path: string;
  effect: "supports" | "suppresses" | "neutral" | "observed";
}

const LENSES: Array<{ id: ReaderLensId; label: string; description: string }> = [
  { id: "shakiness", label: "Shakiness", description: "Underline uncertain text using rhythm and weight." },
  { id: "sources", label: "Sources", description: "Add compact source indexes without recoloring the prose." },
  { id: "provenance", label: "Provenance", description: "Draw measured input-to-output relationships across the reader." },
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

function sourceIndex(sources: readonly SourceReading[], sourceId: string) {
  const index = sources.findIndex((source) => source.id === sourceId);
  return index < 0 ? 0 : index;
}

function dominantLink(token: TokenReading) {
  return [...(token.sources ?? [])].sort(
    (left, right) => Math.abs(right.deltaNats) - Math.abs(left.deltaNats),
  )[0];
}

export function tokenLensClasses(
  token: TokenReading,
  activeLenses: ReadonlySet<ReaderLensId>,
  tone: number,
) {
  const classes = ["lens-reader-token"];
  if (activeLenses.has("shakiness") && token.band) classes.push(`is-${token.band}`);
  if (activeLenses.has("sources") && token.sources?.length) {
    classes.push("has-source-index", `source-tone-${tone % 6}`);
  }
  if (activeLenses.has("provenance") && (token.sources?.length || token.observedSources?.length)) {
    classes.push("has-provenance");
  }
  if (activeLenses.has("concepts")) classes.push("concept-lens-active");
  return classes.join(" ");
}

function initialRun(runtime: RuntimeState, requested?: string) {
  return requested || runtime.runs[0]?.id || "";
}

export function Lens({ runtime, initialRunId }: LensProps) {
  const [runId, setRunId] = useState(() => initialRun(runtime, initialRunId));
  const [data, setData] = useState<ObservatoryData | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [activeLenses, setActiveLenses] = useState<Set<ReaderLensId>>(() => new Set());
  const [drawer, setDrawer] = useState<ReaderDrawer | null>(null);
  const [concepts, setConcepts] = useState<ConceptState>({ status: "idle" });
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const readerSurface = useRef<HTMLDivElement | null>(null);
  const inputPane = useRef<HTMLElement | null>(null);
  const outputPane = useRef<HTMLElement | null>(null);
  const sourceNodes = useRef(new Map<string, HTMLElement>());
  const tokenNodes = useRef(new Map<number, HTMLElement>());

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
  const tokens = data?.tokens ?? [];
  const selectedRun = runtime.runs.find((run) => run.id === runId) ?? null;
  const activeKey = [...activeLenses].sort().join(":");
  const conceptsData = concepts.status === "done" ? concepts.data : null;
  const conceptsAligned = Boolean(
    data?.response
    && conceptsData?.available
    && conceptsData.tokens.length === tokens.length
    && conceptsData.tokens.join("") === data.response,
  );

  useEffect(() => {
    if (!activeLenses.has("provenance") || !data || !readerSurface.current) {
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
        for (const source of sources) {
          const sourceNode = sourceNodes.current.get(source.id);
          if (!sourceNode) continue;
          const links = tokens.flatMap((token, index) => {
            const clear = token.sources?.find((link) => link.sourceId === source.id);
            const observed = token.observedSources?.find((link) => link.sourceId === source.id);
            return clear || observed ? [{ index, link: clear ?? observed, observed: !clear }] : [];
          });
          if (!links.length) continue;
          const targetRects = links
            .slice(0, 24)
            .map(({ index }) => tokenNodes.current.get(index)?.getBoundingClientRect())
            .filter((rect): rect is DOMRect => Boolean(rect));
          if (!targetRects.length) continue;
          const from = sourceNode.getBoundingClientRect();
          const targetY = targetRects.reduce((sum, rect) => sum + rect.top + rect.height / 2, 0) / targetRects.length;
          const x1 = from.right - root.left;
          const y1 = from.top + from.height / 2 - root.top;
          const x2 = Math.min(...targetRects.map((rect) => rect.left)) - root.left;
          const y2 = targetY - root.top;
          const mid = x1 + Math.max(42, (x2 - x1) * .48);
          const strongest = [...links].sort(
            (left, right) => Math.abs(right.link?.deltaNats ?? 0) - Math.abs(left.link?.deltaNats ?? 0),
          )[0];
          next.push({
            id: source.id,
            path: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
            effect: strongest?.observed ? "observed" : strongest?.link?.effect ?? "neutral",
          });
        }
        setConnectors(next.slice(0, 8));
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
  }, [activeKey, data, sources, tokens]);

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

  const inspectorSelection: LensInspectorSelection | null = drawer?.kind === "source"
    ? {
      kind: "source",
      sourceId: drawer.sourceId,
      source: sources.find((source) => source.id === drawer.sourceId),
    }
    : drawer?.kind === "token"
      ? { kind: "token", index: drawer.index }
      : null;

  const selectedToken = drawer?.kind === "token" ? tokens[drawer.index] : undefined;
  const selectedSourceTone = selectedToken && dominantLink(selectedToken)
    ? sourceIndex(sources, dominantLink(selectedToken)!.sourceId)
    : 0;
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
        {(activeLenses.has("sources") || activeLenses.has("provenance")) && !data?.influenceMethod && (
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
            <small>{sources.length} recorded {sources.length === 1 ? "span" : "spans"}</small>
          </header>
          <div className="lens-reader-document">
            {status === "loading" && <p className="lens-reader-state">Loading input…</p>}
            {status === "error" && <p className="lens-reader-state is-error">This run could not be loaded.</p>}
            {status === "idle" && !sources.length && <p className="lens-reader-state">No readable input was recorded.</p>}
            {sources.map((source, index) => (
              <button
                type="button"
                className={[
                  "lens-reader-input-span",
                  (activeLenses.has("sources") || activeLenses.has("provenance")) ? `source-tone-${index % 6}` : "",
                  drawer?.kind === "source" && drawer.sourceId === source.id ? "is-selected" : "",
                ].filter(Boolean).join(" ")}
                ref={(node) => {
                  if (node) sourceNodes.current.set(source.id, node);
                  else sourceNodes.current.delete(source.id);
                }}
                aria-pressed={drawer?.kind === "source" && drawer.sourceId === source.id}
                onClick={() => openSource(source.id)}
                key={source.id}
              >
                <span className="lens-reader-span-meta">
                  {(activeLenses.has("sources") || activeLenses.has("provenance")) && <b>{index + 1}</b>}
                  <span>{source.label ?? source.role ?? source.kind ?? "Input span"}</span>
                  <small>{source.role}</small>
                </span>
                <strong>{source.text}</strong>
              </button>
            ))}
          </div>
        </section>

        <section className="lens-reader-pane lens-reader-output" ref={outputPane} aria-labelledby="lens-reader-output-title">
          <header>
            <h2 id="lens-reader-output-title">Output</h2>
            <small>{tokens.length} recorded tokens</small>
          </header>
          <div className="lens-reader-document lens-reader-output-document">
            {status === "loading" && <p className="lens-reader-state">Loading output…</p>}
            {status === "error" && <p className="lens-reader-state is-error">This run could not be loaded.</p>}
            {status === "idle" && !tokens.length && (
              <p className="lens-reader-plain-output">{data?.response || "No readable output was recorded."}</p>
            )}
            {tokens.length > 0 && (
              <div className="lens-reader-output-stream">
                {tokens.map((token, index) => {
                  const dominant = dominantLink(token);
                  const tone = dominant ? sourceIndex(sources, dominant.sourceId) : selectedSourceTone;
                  const concept = conceptsAligned ? conceptsData?.readouts[index]?.[0] : undefined;
                  return (
                    <span
                      role="button"
                      tabIndex={0}
                      className={[
                        tokenLensClasses(token, activeLenses, tone),
                        drawer?.kind === "token" && drawer.index === index ? "is-selected" : "",
                        concept ? "has-concept" : "",
                      ].filter(Boolean).join(" ")}
                      title={[
                        `Token ${index + 1}`,
                        token.confidence == null ? null : `confidence ${Math.round(token.confidence * 100)}%`,
                        concept ? `concept: ${concept.piece} (${concept.score.toFixed(2)})` : null,
                      ].filter(Boolean).join(" · ")}
                      ref={(node) => {
                        if (node) tokenNodes.current.set(index, node);
                        else tokenNodes.current.delete(index);
                      }}
                      onClick={() => openToken(index)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          openToken(index);
                        }
                      }}
                      key={`${index}-${token.text}`}
                    >
                      {readableToken(token.text)}
                      {activeLenses.has("sources") && dominant && (
                        <sup className={`source-tone-${tone % 6}`}>{tone + 1}</sup>
                      )}
                      {conceptsActive && concept && <i className="lens-reader-concept-mark" aria-hidden="true" />}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        {activeLenses.has("provenance") && connectors.length > 0 && (
          <svg className="lens-reader-provenance" aria-hidden="true">
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
