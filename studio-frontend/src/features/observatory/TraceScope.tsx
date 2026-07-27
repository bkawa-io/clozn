import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type { SourceReading, TokenReading } from "../../data/types";

interface TraceScopeProps {
  sources: SourceReading[];
  tokens: TokenReading[];
  selectedToken: number;
  onSelectToken: (index: number) => void;
}

function sourcePosition(index: number, count: number) {
  return {
    x: count === 1 ? 500 : 180 + index * (640 / Math.max(1, count - 1)),
  };
}

function visibleToken(text: string) {
  if (!text) return "∅";
  if (!text.trim()) return text.includes("\n") ? "↵" : "\u00a0";
  return text
    .replace(/\r\n|\r|\n/g, "")
    .replace(/\t/g, "⇥")
    .replaceAll(" ", "\u00a0");
}

export function TraceScope({ sources, tokens, selectedToken, onSelectToken }: TraceScopeProps) {
  const linkedCount = tokens.filter((token) => token.sources?.length).length;
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const sourceTokenIndexes = new Set(
    selectedSourceId
      ? tokens.flatMap((token, index) =>
          token.sources?.some((source) => source.sourceId === selectedSourceId) ? [index] : [])
      : [],
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const sourceRefs = useRef(new Map<string, HTMLButtonElement>());
  const tokenRefs = useRef(new Map<number, HTMLButtonElement>());
  const [links, setLinks] = useState<Array<{
    key: string;
    d: string;
    effect: string;
    selected: boolean;
    sourceFocused: boolean;
    muted: boolean;
  }>>([]);

  useEffect(() => {
    if (selectedSourceId && !sources.some((source) => source.id === selectedSourceId)) {
      setSelectedSourceId(null);
    }
  }, [selectedSourceId, sources]);

  const measureLinks = useCallback(() => {
    const root = rootRef.current;
    if (!root) return;
    const rootRect = root.getBoundingClientRect();
    const next = tokens.flatMap((token, tokenIndex) => (token.sources ?? []).slice(0, 2).flatMap((source) => {
      const sourceNode = sourceRefs.current.get(source.sourceId);
      const tokenNode = tokenRefs.current.get(tokenIndex);
      if (!sourceNode || !tokenNode) return [];
      const sourceRect = sourceNode.getBoundingClientRect();
      const tokenRect = tokenNode.getBoundingClientRect();
      const sourceX = sourceRect.left + sourceRect.width / 2 - rootRect.left;
      const sourceY = sourceRect.bottom - rootRect.top;
      const tokenX = tokenRect.left + tokenRect.width / 2 - rootRect.left;
      const tokenY = tokenRect.top - rootRect.top;
      const bendY = sourceY + Math.max(28, (tokenY - sourceY) * .46);
      return [{
        key: `${source.sourceId}-${tokenIndex}`,
        d: `M ${sourceX} ${sourceY} C ${sourceX} ${bendY}, ${tokenX} ${bendY}, ${tokenX} ${tokenY}`,
        effect: source.effect,
        selected: selectedSourceId ? source.sourceId === selectedSourceId : tokenIndex === selectedToken,
        sourceFocused: selectedSourceId === source.sourceId,
        muted: Boolean(selectedSourceId && source.sourceId !== selectedSourceId),
      }];
    }));
    setLinks(next);
  }, [selectedSourceId, selectedToken, sources, tokens]);

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const frame = requestAnimationFrame(measureLinks);
    const observer = new ResizeObserver(measureLinks);
    observer.observe(root);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [measureLinks]);

  return (
    <div
      className={`trace-scope ${selectedSourceId ? "has-source-focus" : ""}`}
      aria-label="Token source trace"
      ref={rootRef}
    >
      <div className="trace-status">
        <span><b>SOURCES</b>{sources.length}</span>
        <span>
          <b>LINKED TOKENS</b>
          {selectedSourceId ? sourceTokenIndexes.size : linkedCount}/{tokens.length}
        </span>
      </div>

      <svg className="trace-field" aria-hidden="true">
        <defs>
          <linearGradient id="trace-support" x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="var(--signal-mint)" stopOpacity=".72" />
            <stop offset="1" stopColor="var(--signal-cyan)" stopOpacity=".2" />
          </linearGradient>
          <linearGradient id="trace-suppress" x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="var(--signal-peach)" stopOpacity=".68" />
            <stop offset="1" stopColor="var(--signal-pink)" stopOpacity=".2" />
          </linearGradient>
          <filter id="trace-glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3.4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {links.map((link) => (
          <path
            className={`trace-link is-${link.effect} ${link.selected ? "is-selected" : ""} ${link.sourceFocused ? "is-source-focused" : ""} ${link.muted ? "is-muted" : ""}`}
            d={link.d}
            key={link.key}
          />
        ))}
      </svg>

      <div className="source-row">
        {sources.map((source, index) => {
          const position = sourcePosition(index, sources.length);
          return (
            <button
              type="button"
              className={`source-node ${selectedSourceId === source.id ? "is-selected" : ""} ${selectedSourceId && selectedSourceId !== source.id ? "is-muted" : ""}`}
              key={source.id}
              aria-label={`Context span: ${source.text}`}
              aria-pressed={selectedSourceId === source.id}
              onClick={() => setSelectedSourceId((current) => current === source.id ? null : source.id)}
              ref={(node) => {
                if (node) sourceRefs.current.set(source.id, node);
                else sourceRefs.current.delete(source.id);
              }}
              style={{ "--source-x": `${position.x / 10}%` } as CSSProperties}
              title={source.text}
            >
              <span>{source.role || "context"}</span>
              <strong>{source.text}</strong>
            </button>
          );
        })}
      </div>

      <div className="trace-tokens">
        {tokens.map((token, index) => {
          const hasBreak = token.text.includes("\n");
          const whitespace = Boolean(token.text) && !token.text.trim();
          const sourceMatch = sourceTokenIndexes.has(index);
          const sourceMuted = Boolean(selectedSourceId && !sourceMatch);
          return (
            <Fragment key={`${index}-${token.text}`}>
              <button
                type="button"
                className={`${index === selectedToken ? "is-selected" : ""} ${sourceMatch ? "is-source-match" : ""} ${sourceMuted ? "is-source-muted" : ""} ${whitespace ? "is-whitespace" : ""} ${!token.text ? "is-terminal" : ""} band-${token.band ?? "none"}`}
                aria-label={`Token ${index + 1}: ${token.text || "blank"}`}
                aria-pressed={index === selectedToken}
                onClick={() => {
                  setSelectedSourceId(null);
                  onSelectToken(index);
                }}
                ref={(node) => {
                  if (node) tokenRefs.current.set(index, node);
                  else tokenRefs.current.delete(index);
                }}
                style={{ "--token-confidence": token.confidence ?? 0 } as CSSProperties}
              >
                <span>{visibleToken(token.text)}</span>
              </button>
              {hasBreak && <span className="trace-break" aria-hidden="true" />}
            </Fragment>
          );
        })}
      </div>

      {!sources.length && (
        <div className="trace-empty"><strong>PROVENANCE UNAVAILABLE</strong></div>
      )}
    </div>
  );
}
