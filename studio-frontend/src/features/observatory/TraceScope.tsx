import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from "react";
import type {
  ContextCoverage,
  SourceReading,
  TokenReading,
  TokenSourceReading,
} from "../../data/types";
import { ThreadedTrace } from "./ThreadedTrace";

interface TraceScopeProps {
  sources: SourceReading[];
  coverage?: ContextCoverage;
  tokens: TokenReading[];
  selectedToken: number;
  onSelectToken: (index: number) => void;
}

interface OutputChunk {
  start: number;
  end: number;
  text: string;
  linkedTokens: number;
  sourceIds: Set<string>;
  averageConfidence: number;
}

interface ActivePath {
  key: string;
  sourceX: number;
  outputX: number;
  effect: TokenSourceReading["effect"];
  strength: number;
}

interface SourceInfluence {
  source: SourceReading;
  links: number;
  linkedTokens: number;
  totalAbsDelta: number;
  averageAbsDelta: number;
  peakAbsDelta: number;
}

type SourceFilter = "quick" | "all" | "measured" | "omitted";

const SOURCE_ROW_HEIGHT = 62;
const SOURCE_OVERSCAN = 4;
const TOKEN_DETAIL_LIMIT = 384;
const TOKEN_WINDOW_RADIUS = 36;
const THREAD_CONTEXT_WORD_LIMIT = 220;
const THREAD_OUTPUT_WORD_LIMIT = 90;
const THREAD_TOKEN_LIMIT = 140;
const THREAD_SOURCE_LIMIT = 7;
const THREAD_LINK_LIMIT = 240;

function visibleToken(text: string) {
  if (!text) return "∅";
  if (!text.trim()) return text.includes("\n") ? "↵" : text.includes("\t") ? "⇥" : "\u00a0";
  return text.replace(/\r\n|\r|\n/g, "↵").replace(/\t/g, "⇥").replaceAll(" ", "\u00a0");
}

function sourceType(source: SourceReading) {
  if (source.kind === "file") return "FILE";
  if (source.kind === "repository_map") return "REPOSITORY";
  if (source.kind === "retrieval") return "RETRIEVED";
  if (source.kind === "tool_result" || source.role === "tool") return "TOOL RESULT";
  if (source.kind === "tool_call") return "TOOL CALL";
  if (source.kind === "policy" || source.role === "system" || source.role === "developer") return "POLICY";
  return source.role.toUpperCase();
}

function sourceName(source: SourceReading) {
  if (source.label) return source.label;
  if (source.messageIndex != null) return `${source.role} turn ${source.messageIndex + 1}`;
  return source.role || "context";
}

function wordCount(text: string) {
  return text.match(/\S+/gu)?.length ?? 0;
}

function sourceInfluence(sources: SourceReading[], tokens: TokenReading[]) {
  const bySource = new Map(sources.map((source) => [source.id, {
    source,
    links: 0,
    tokenIndexes: new Set<number>(),
    totalAbsDelta: 0,
    peakAbsDelta: 0,
  }]));
  tokens.forEach((token, tokenIndex) => {
    token.sources?.forEach((link) => {
      const aggregate = bySource.get(link.sourceId);
      if (!aggregate) return;
      const magnitude = Math.abs(link.deltaNats);
      aggregate.links += 1;
      aggregate.tokenIndexes.add(tokenIndex);
      aggregate.totalAbsDelta += magnitude;
      aggregate.peakAbsDelta = Math.max(aggregate.peakAbsDelta, magnitude);
    });
  });
  return [...bySource.values()]
    .filter((aggregate) => aggregate.source.measured !== false && aggregate.links > 0)
    .map((aggregate): SourceInfluence => ({
      source: aggregate.source,
      links: aggregate.links,
      linkedTokens: aggregate.tokenIndexes.size,
      totalAbsDelta: aggregate.totalAbsDelta,
      averageAbsDelta: aggregate.totalAbsDelta / aggregate.links,
      peakAbsDelta: aggregate.peakAbsDelta,
    }))
    .sort((a, b) =>
      b.totalAbsDelta - a.totalAbsDelta
      || b.peakAbsDelta - a.peakAbsDelta
      || b.links - a.links,
    );
}

function InfluenceQuickView({
  influence,
  selectedSourceId,
  onSelectSource,
}: {
  influence: SourceInfluence[];
  selectedSourceId: string | null;
  onSelectSource: (sourceId: string | null) => void;
}) {
  const strongest = influence.slice(0, 3);
  const strongestIds = new Set(strongest.map((item) => item.source.id));
  const weakest = [...influence]
    .reverse()
    .filter((item) => !strongestIds.has(item.source.id))
    .slice(0, 3);

  function group(label: string, rows: SourceInfluence[]) {
    return (
      <section>
        <header>
          <strong>{label}</strong>
          <span>Σ |Δ NATS|</span>
        </header>
        {rows.map((row, index) => (
          <button
            type="button"
            className={selectedSourceId === row.source.id ? "is-selected" : ""}
            aria-pressed={selectedSourceId === row.source.id}
            aria-label={`${label.toLowerCase()} rank ${index + 1}: ${sourceName(row.source)}, aggregate absolute influence ${row.totalAbsDelta.toFixed(3)} nats across ${row.links} links`}
            onClick={() => onSelectSource(selectedSourceId === row.source.id ? null : row.source.id)}
            key={row.source.id}
          >
            <span>{String(index + 1).padStart(2, "0")}</span>
            <div>
              <b>{sourceName(row.source)}</b>
              <small>{sourceType(row.source)} · {row.linkedTokens} OUTPUT TOKENS</small>
            </div>
            <strong>{row.totalAbsDelta.toFixed(3)}</strong>
          </button>
        ))}
        {!rows.length && <div className="trace-rank-empty">NO DISTINCT MEASURED SPAN</div>}
      </section>
    );
  }

  return (
    <div className="trace-influence-quick" aria-label="Context influence ranking">
      <div className="trace-rank-method">
        <span>MEASURED LINKS</span>
        <strong>{influence.length} CONTEXT SPANS</strong>
      </div>
      {group("MOST INFLUENTIAL", strongest)}
      {group("LEAST INFLUENTIAL", weakest)}
    </div>
  );
}

function buildChunks(tokens: TokenReading[]): OutputChunk[] {
  if (!tokens.length) return [];
  const target = tokens.length > 2400 ? 120 : tokens.length > 900 ? 88 : 56;
  const max = Math.round(target * 1.5);
  const chunks: OutputChunk[] = [];
  let start = 0;
  let text = "";

  const push = (end: number) => {
    const rows = tokens.slice(start, end);
    const sourceIds = new Set(rows.flatMap((token) => (token.sources ?? []).map((source) => source.sourceId)));
    chunks.push({
      start,
      end,
      text,
      linkedTokens: rows.filter((token) => token.sources?.length).length,
      sourceIds,
      averageConfidence: rows.reduce((total, token) => total + (token.confidence ?? 0), 0)
        / Math.max(1, rows.length),
    });
    start = end;
    text = "";
  };

  for (let index = 0; index < tokens.length; index += 1) {
    text += tokens[index].text;
    const length = index - start + 1;
    const paragraphBoundary = length >= target && /\n\s*\n$/u.test(text);
    const codeBoundary = length >= target && /\n[}\])];?\s*$/u.test(text);
    if (paragraphBoundary || codeBoundary || length >= max) push(index + 1);
  }
  if (start < tokens.length) push(tokens.length);
  return chunks;
}

function chunkIndexForToken(chunks: OutputChunk[], tokenIndex: number) {
  const index = chunks.findIndex((chunk) => tokenIndex >= chunk.start && tokenIndex < chunk.end);
  return Math.max(0, index);
}

function weightedCenters(sources: SourceReading[]): Map<string, number> {
  const weights = sources.map((source) => Math.max(1, Math.min(6, Math.sqrt(source.text.length / 180))));
  const total = weights.reduce((sum, weight) => sum + weight, 0) || 1;
  let cursor = 0;
  return new Map(sources.map((source, index) => {
    const weight = weights[index];
    const center = (cursor + weight / 2) / total * 100;
    cursor += weight;
    return [source.id, center];
  }));
}

function pathEffect(links: TokenSourceReading[]) {
  return [...links].sort((a, b) => Math.abs(b.deltaNats) - Math.abs(a.deltaNats))[0]?.effect ?? "neutral";
}

function tokenTargetForChunk(
  chunk: OutputChunk,
  tokens: TokenReading[],
  selectedSourceId: string | null,
) {
  if (selectedSourceId) {
    const linked = tokens.findIndex((token, index) =>
      index >= chunk.start
      && index < chunk.end
      && token.sources?.some((source) => source.sourceId === selectedSourceId));
    if (linked >= 0) return linked;
  }
  let target = chunk.start;
  for (let index = chunk.start + 1; index < chunk.end; index += 1) {
    if ((tokens[index].confidence ?? 1) < (tokens[target].confidence ?? 1)) target = index;
  }
  return target;
}

export function TraceScope({
  sources,
  coverage,
  tokens,
  selectedToken,
  onSelectToken,
}: TraceScopeProps) {
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("quick");
  const [sourceQuery, setSourceQuery] = useState("");
  const [sourceScrollTop, setSourceScrollTop] = useState(0);
  const [threadQuickOpen, setThreadQuickOpen] = useState(false);
  const [compactLayout, setCompactLayout] = useState(() =>
    typeof window !== "undefined" && window.matchMedia("(max-width: 650px)").matches);
  const sourceListRef = useRef<HTMLDivElement>(null);
  const chunks = useMemo(() => buildChunks(tokens), [tokens]);
  const longOutput = tokens.length > TOKEN_DETAIL_LIMIT;
  const activeChunkIndex = chunkIndexForToken(chunks, selectedToken);
  const activeChunk = chunks[activeChunkIndex];
  const context = coverage ?? {
    totalSources: sources.length,
    measuredSources: sources.filter((source) => source.measured !== false).length,
    omittedSources: sources.filter((source) => source.measured === false).length,
    measuredSpans: sources.filter((source) => source.measured !== false).length,
    complete: sources.length > 0 && sources.every((source) => source.measured !== false),
  };
  const influence = useMemo(() => sourceInfluence(sources, tokens), [sources, tokens]);
  const contextWords = useMemo(
    () => sources.reduce((total, source) => total + wordCount(source.text), 0),
    [sources],
  );
  const outputWords = useMemo(
    () => wordCount(tokens.map((token) => token.text).join("")),
    [tokens],
  );
  const linkCount = useMemo(
    () => tokens.reduce((total, token) => total + (token.sources?.length ?? 0), 0),
    [tokens],
  );
  const threadedMode = !compactLayout
    && sources.length > 0
    && sources.length <= THREAD_SOURCE_LIMIT
    && contextWords <= THREAD_CONTEXT_WORD_LIMIT
    && outputWords <= THREAD_OUTPUT_WORD_LIMIT
    && tokens.length <= THREAD_TOKEN_LIMIT
    && linkCount <= THREAD_LINK_LIMIT;

  useEffect(() => {
    const query = window.matchMedia("(max-width: 650px)");
    const update = () => setCompactLayout(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (selectedSourceId && !sources.some((source) => source.id === selectedSourceId)) {
      setSelectedSourceId(null);
    }
  }, [selectedSourceId, sources]);

  const selectedSource = sources.find((source) => source.id === selectedSourceId);
  const selectedTokenSourceIds = new Set(
    (tokens[selectedToken]?.sources ?? []).map((source) => source.sourceId),
  );
  const sourceTokenIndexes = useMemo(() => new Set(
    selectedSourceId
      ? tokens.flatMap((token, index) =>
          token.sources?.some((source) => source.sourceId === selectedSourceId) ? [index] : [])
      : [],
  ), [selectedSourceId, tokens]);
  const sourceCounts = useMemo(() => {
    const counts = new Map<string, number>();
    tokens.forEach((token) => token.sources?.forEach((source) => {
      counts.set(source.sourceId, (counts.get(source.sourceId) ?? 0) + 1);
    }));
    return counts;
  }, [tokens]);

  const filteredSources = useMemo(() => {
    const query = sourceQuery.trim().toLowerCase();
    return sources.filter((source) => {
      if (sourceFilter === "measured" && source.measured === false) return false;
      if (sourceFilter === "omitted" && source.measured !== false) return false;
      if (!query) return true;
      return `${sourceName(source)} ${source.role} ${source.kind ?? ""} ${source.text}`
        .toLowerCase()
        .includes(query);
    });
  }, [sourceFilter, sourceQuery, sources]);

  const sourceViewportHeight = sourceListRef.current?.clientHeight ?? 260;
  const sourceStart = Math.max(0, Math.floor(sourceScrollTop / SOURCE_ROW_HEIGHT) - SOURCE_OVERSCAN);
  const sourceEnd = Math.min(
    filteredSources.length,
    Math.ceil((sourceScrollTop + sourceViewportHeight) / SOURCE_ROW_HEIGHT) + SOURCE_OVERSCAN,
  );
  const visibleSources = filteredSources.slice(sourceStart, sourceEnd);

  const sourceCenters = useMemo(() => weightedCenters(sources), [sources]);
  const activePaths = useMemo(() => {
    if (!chunks.length || !sources.length) return [];
    const paths: ActivePath[] = [];
    if (selectedSourceId) {
      chunks.forEach((chunk, chunkIndex) => {
        const links = tokens.slice(chunk.start, chunk.end).flatMap((token) =>
          (token.sources ?? []).filter((source) => source.sourceId === selectedSourceId));
        if (!links.length) return;
        paths.push({
          key: `${selectedSourceId}-${chunkIndex}`,
          sourceX: sourceCenters.get(selectedSourceId) ?? 50,
          outputX: (chunk.start + (chunk.end - chunk.start) / 2) / Math.max(1, tokens.length) * 100,
          effect: pathEffect(links),
          strength: Math.max(...links.map((link) => Math.abs(link.deltaNats))),
        });
      });
    } else {
      (tokens[selectedToken]?.sources ?? []).forEach((source) => {
        paths.push({
          key: `${source.sourceId}-${selectedToken}`,
          sourceX: sourceCenters.get(source.sourceId) ?? 50,
          outputX: (selectedToken + .5) / Math.max(1, tokens.length) * 100,
          effect: source.effect,
          strength: Math.abs(source.deltaNats),
        });
      });
    }
    return paths.sort((a, b) => b.strength - a.strength).slice(0, 12);
  }, [chunks, selectedSourceId, selectedToken, sourceCenters, sources.length, tokens]);

  const tokenWindowStart = longOutput
    ? Math.max(0, Math.min(tokens.length - TOKEN_WINDOW_RADIUS * 2 - 1, selectedToken - TOKEN_WINDOW_RADIUS))
    : 0;
  const tokenWindowEnd = longOutput
    ? Math.min(tokens.length, tokenWindowStart + TOKEN_WINDOW_RADIUS * 2 + 1)
    : tokens.length;
  const visibleTokens = tokens.slice(tokenWindowStart, tokenWindowEnd);
  const outputKind = /^\s*[[{]/u.test(tokens.slice(0, 80).map((token) => token.text).join(""))
    ? "STRUCTURED"
    : tokens.some((token) => token.text.includes("```"))
      ? "CODE"
      : "TEXT";

  function selectToken(index: number) {
    setSelectedSourceId(null);
    onSelectToken(index);
  }

  function selectSource(sourceId: string | null) {
    setSelectedSourceId(sourceId);
  }

  function handleTokenKeys(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowRight") next = Math.min(tokens.length - 1, index + 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tokens.length - 1;
    else return;
    event.preventDefault();
    selectToken(next);
    requestAnimationFrame(() => {
      document.querySelector<HTMLButtonElement>(`[data-trace-token="${next}"]`)?.focus();
    });
  }

  const statusBar = (
    <div className="trace-status">
      <span><b>CONTEXT</b>{context.totalSources} sources</span>
      <span><b>MEASURED</b>{context.measuredSources} / {context.totalSources}</span>
      <span><b>OUTPUT</b>{tokens.length} tokens</span>
      {context.promptTokens != null && <span><b>PROMPT</b>{context.promptTokens.toLocaleString()} tokens</span>}
      <strong className={context.complete ? "is-complete" : "is-partial"}>
        {context.complete ? "COMPLETE" : context.measuredSources ? "PARTIAL" : "UNMEASURED"}
      </strong>
    </div>
  );

  if (threadedMode) {
    return (
      <div
        className={`trace-scope is-threaded ${selectedSourceId ? "has-source-focus" : ""}`}
        aria-label="Context and output provenance"
      >
        {statusBar}
        <div className="thread-mode-label">
          <span>WORD-TO-WORD THREADS</span>
          <strong>{linkCount} MEASURED LINKS</strong>
          <button
            type="button"
            aria-expanded={threadQuickOpen}
            onClick={() => setThreadQuickOpen((open) => !open)}
          >
            RANKED SPANS
          </button>
        </div>
        {threadQuickOpen && (
          <div className="thread-rank-overlay">
            <header>
              <strong>CONTEXT INFLUENCE</strong>
              <button type="button" aria-label="Close context influence ranking" onClick={() => setThreadQuickOpen(false)}>×</button>
            </header>
            <InfluenceQuickView
              influence={influence}
              selectedSourceId={selectedSourceId}
              onSelectSource={selectSource}
            />
          </div>
        )}
        <ThreadedTrace
          sources={sources}
          tokens={tokens}
          selectedToken={selectedToken}
          selectedSourceId={selectedSourceId}
          onSelectToken={onSelectToken}
          onSelectSource={selectSource}
        />
      </div>
    );
  }

  return (
    <div
      className={`trace-scope ${longOutput ? "is-chunked" : "is-token-detail"} ${selectedSourceId ? "has-source-focus" : ""}`}
      aria-label="Context and output provenance"
    >
      {statusBar}

      <section className="trace-map" aria-label="Provenance overview">
        <div className="trace-map-row context-row">
          <span>CONTEXT</span>
          <div>
            {sources.map((source) => (
              <button
                type="button"
                className={`${selectedSourceId === source.id ? "is-selected" : ""} ${selectedTokenSourceIds.has(source.id) ? "is-token-source" : ""} ${source.measured === false ? "is-omitted" : ""}`}
                style={{ flexGrow: Math.max(1, Math.min(6, Math.sqrt(source.text.length / 180))) }}
                aria-label={`${sourceName(source)}; ${source.measured === false ? "not measured" : "measured"}`}
                aria-pressed={selectedSourceId === source.id}
                title={`${sourceName(source)} · ${source.text.length.toLocaleString()} characters`}
                onClick={() => setSelectedSourceId((current) => current === source.id ? null : source.id)}
                key={source.id}
              />
            ))}
          </div>
        </div>
        <svg viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true">
          {activePaths.map((path) => (
            <path
              className={`is-${path.effect}`}
              d={`M ${path.sourceX} 1 C ${path.sourceX} 9, ${path.outputX} 15, ${path.outputX} 23`}
              style={{ "--path-strength": Math.min(1, .25 + path.strength * 3) } as CSSProperties}
              key={path.key}
            />
          ))}
        </svg>
        <div className="trace-map-row output-row">
          <span>OUTPUT</span>
          <div>
            {chunks.map((chunk, index) => {
              const sourceMatch = Boolean(selectedSourceId && chunk.sourceIds.has(selectedSourceId));
              return (
                <button
                  type="button"
                  className={`${index === activeChunkIndex ? "is-selected" : ""} ${sourceMatch ? "is-source-match" : ""} ${chunk.linkedTokens ? "has-links" : ""}`}
                  style={{ flexGrow: chunk.end - chunk.start }}
                  aria-label={`Output tokens ${chunk.start + 1} through ${chunk.end}`}
                  aria-pressed={index === activeChunkIndex}
                  title={`${chunk.start + 1}–${chunk.end} · ${chunk.linkedTokens} linked`}
                  onClick={() => onSelectToken(tokenTargetForChunk(chunk, tokens, selectedSourceId))}
                  key={`${chunk.start}-${chunk.end}`}
                />
              );
            })}
          </div>
        </div>
      </section>

      <div className="trace-workspace">
        <section className={`trace-context-browser ${selectedSource ? "has-selection" : ""}`}>
          <header>
            <div>
              <strong>CONTEXT SOURCES</strong>
              <span>
                {sourceFilter === "quick"
                  ? `${influence.length} measured with links · ${context.omittedSources} omitted`
                  : `${filteredSources.length} visible · ${context.omittedSources} omitted from measurement`}
              </span>
            </div>
            <label>
              <span>FILTER</span>
              <input
                type="search"
                value={sourceQuery}
                onChange={(event) => {
                  setSourceQuery(event.target.value);
                  setSourceFilter("all");
                  setSourceScrollTop(0);
                  if (sourceListRef.current) sourceListRef.current.scrollTop = 0;
                }}
              />
            </label>
          </header>
          <nav aria-label="Context source filter">
            {(["quick", "all", "measured", "omitted"] as const).map((filter) => (
              <button
                type="button"
                className={sourceFilter === filter ? "is-active" : ""}
                aria-pressed={sourceFilter === filter}
                onClick={() => {
                  setSourceFilter(filter);
                  if (filter === "quick") setSourceQuery("");
                  setSourceScrollTop(0);
                  if (sourceListRef.current) sourceListRef.current.scrollTop = 0;
                }}
                key={filter}
              >
                {filter.toUpperCase()}
              </button>
            ))}
          </nav>
          {sourceFilter === "quick" ? (
            <InfluenceQuickView
              influence={influence}
              selectedSourceId={selectedSourceId}
              onSelectSource={selectSource}
            />
          ) : (
            <div
              className="trace-source-list"
              ref={sourceListRef}
              onScroll={(event) => setSourceScrollTop(event.currentTarget.scrollTop)}
            >
              <div style={{ height: filteredSources.length * SOURCE_ROW_HEIGHT }}>
                {visibleSources.map((source, offset) => {
                  const index = sourceStart + offset;
                  const linkedTokens = sourceCounts.get(source.id) ?? 0;
                  return (
                    <button
                      type="button"
                      className={`${selectedSourceId === source.id ? "is-selected" : ""} ${selectedTokenSourceIds.has(source.id) ? "is-token-source" : ""} ${source.measured === false ? "is-omitted" : ""}`}
                      aria-pressed={selectedSourceId === source.id}
                      aria-label={`${sourceName(source)}, ${source.measured === false ? "not measured" : "measured"}, linked to ${linkedTokens} output tokens`}
                      onClick={() => selectSource(selectedSourceId === source.id ? null : source.id)}
                      style={{ transform: `translateY(${index * SOURCE_ROW_HEIGHT}px)` }}
                      key={source.id}
                    >
                      <span><b>{sourceType(source)}</b>{source.measured === false ? "NOT MEASURED" : `LINKED ${linkedTokens}`}</span>
                      <strong>{sourceName(source)}</strong>
                      <small>{source.text}</small>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
          {selectedSource && (
            <div className="trace-source-detail">
              <header>
                <span>{sourceType(selectedSource)} · {selectedSource.text.length.toLocaleString()} CHARS</span>
                <strong>{selectedSource.measured === false ? "NOT MEASURED" : `${sourceTokenIndexes.size} LINKED TOKENS`}</strong>
                <button type="button" aria-label="Close context detail" onClick={() => selectSource(null)}>×</button>
              </header>
              <pre>{selectedSource.text}</pre>
            </div>
          )}
          {!sources.length && (
            <div className="trace-empty"><strong>CONTEXT RECORD UNAVAILABLE</strong></div>
          )}
        </section>

        <section className="trace-output-browser">
          <header>
            <div>
              <strong>RECORDED OUTPUT</strong>
              <span>{outputKind} · {longOutput ? `${chunks.length} REGIONS` : "TOKEN DETAIL"}</span>
            </div>
            <span>
              {selectedSource
                ? `${sourceTokenIndexes.size} TOKENS LINKED TO ${sourceName(selectedSource)}`
                : `TOKEN ${selectedToken + 1} OF ${tokens.length}`}
            </span>
          </header>

          <div className="trace-output-scroll">
            {longOutput ? (
              <div className="trace-output-chunks">
                {chunks.map((chunk, index) => {
                  const sourceMatch = Boolean(selectedSourceId && chunk.sourceIds.has(selectedSourceId));
                  return (
                    <button
                      type="button"
                      className={`${index === activeChunkIndex ? "is-selected" : ""} ${sourceMatch ? "is-source-match" : ""} ${selectedSourceId && !sourceMatch ? "is-source-muted" : ""}`}
                      aria-pressed={index === activeChunkIndex}
                      aria-label={`Output region ${index + 1}, tokens ${chunk.start + 1} through ${chunk.end}, ${chunk.sourceIds.size ? `linked to ${chunk.sourceIds.size} context sources` : "no measured source links"}`}
                      onClick={() => onSelectToken(tokenTargetForChunk(chunk, tokens, selectedSourceId))}
                      key={`${chunk.start}-${chunk.end}`}
                    >
                      <span>
                        <b>{String(index + 1).padStart(2, "0")}</b>
                        TOKENS {chunk.start + 1}–{chunk.end}
                        <i>{chunk.linkedTokens} LINKED · {Math.round(chunk.averageConfidence * 100)}% CONF</i>
                      </span>
                      <code>{chunk.text}</code>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="trace-output-tokens" role="listbox" aria-label="Recorded output tokens">
                {visibleTokens.map((token, offset) => {
                  const index = tokenWindowStart + offset;
                  const sourceMatch = sourceTokenIndexes.has(index);
                  return (
                    <button
                      type="button"
                      role="option"
                      aria-selected={index === selectedToken}
                      aria-label={`Token ${index + 1}: ${token.text || "blank"}`}
                      tabIndex={index === selectedToken ? 0 : -1}
                      data-trace-token={index}
                      className={`${index === selectedToken ? "is-selected" : ""} ${sourceMatch ? "is-source-match" : ""} ${selectedSourceId && !sourceMatch ? "is-source-muted" : ""} band-${token.band ?? "none"}`}
                      onClick={() => selectToken(index)}
                      onKeyDown={(event) => handleTokenKeys(event, index)}
                      style={{ "--token-confidence": token.confidence ?? 0 } as CSSProperties}
                      title={`Token ${index + 1} · confidence ${token.confidence?.toFixed(4) ?? "unavailable"}`}
                      key={`${index}-${token.text}`}
                    >
                      {visibleToken(token.text)}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {longOutput && (
            <div className="trace-token-window" role="listbox" aria-label="Selected output token neighborhood">
              <span>{tokenWindowStart + 1}</span>
              <div>
                {visibleTokens.map((token, offset) => {
                  const index = tokenWindowStart + offset;
                  const sourceMatch = sourceTokenIndexes.has(index);
                  return (
                    <button
                      type="button"
                      role="option"
                      aria-selected={index === selectedToken}
                      aria-label={`Token ${index + 1}: ${token.text || "blank"}`}
                      tabIndex={index === selectedToken ? 0 : -1}
                      data-trace-token={index}
                      className={`${index === selectedToken ? "is-selected" : ""} ${sourceMatch ? "is-source-match" : ""} ${selectedSourceId && !sourceMatch ? "is-source-muted" : ""}`}
                      onClick={() => selectToken(index)}
                      onKeyDown={(event) => handleTokenKeys(event, index)}
                      title={`Token ${index + 1}: ${token.text || "blank"}`}
                      key={`${index}-${token.text}`}
                    >
                      {visibleToken(token.text)}
                    </button>
                  );
                })}
              </div>
              <span>{tokenWindowEnd}</span>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
