import {
  Fragment,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from "react";
import type { SourceReading, TokenReading } from "../../data/types";

interface ThreadedTraceProps {
  sources: SourceReading[];
  tokens: TokenReading[];
  selectedToken: number;
  selectedSourceId: string | null;
  onSelectToken: (index: number) => void;
  onSelectSource: (sourceId: string | null) => void;
}

interface ThreadPath {
  key: string;
  d: string;
  effect: string;
  selected: boolean;
  sourceFocused: boolean;
  muted: boolean;
}

function sourcePosition(index: number, count: number) {
  return count === 1 ? 50 : 12 + index * (76 / Math.max(1, count - 1));
}

function visibleToken(text: string) {
  if (!text) return "∅";
  if (!text.trim()) return text.includes("\n") ? "↵" : text.includes("\t") ? "⇥" : "\u00a0";
  return text.replace(/\r\n|\r|\n/g, "↵").replace(/\t/g, "⇥").replaceAll(" ", "\u00a0");
}

export function ThreadedTrace({
  sources,
  tokens,
  selectedToken,
  selectedSourceId,
  onSelectToken,
  onSelectSource,
}: ThreadedTraceProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const sourceRefs = useRef(new Map<string, HTMLButtonElement>());
  const tokenRefs = useRef(new Map<number, HTMLButtonElement>());
  const [paths, setPaths] = useState<ThreadPath[]>([]);

  const measurePaths = useCallback(() => {
    const root = rootRef.current;
    if (!root) return;
    const rootRect = root.getBoundingClientRect();
    const next = tokens.flatMap((token, tokenIndex) =>
      (token.sources ?? []).slice(0, 2).flatMap((source) => {
        const sourceNode = sourceRefs.current.get(source.sourceId);
        const tokenNode = tokenRefs.current.get(tokenIndex);
        if (!sourceNode || !tokenNode) return [];
        const sourceRect = sourceNode.getBoundingClientRect();
        const tokenRect = tokenNode.getBoundingClientRect();
        const sourceX = sourceRect.left + sourceRect.width / 2 - rootRect.left;
        const sourceY = sourceRect.bottom - rootRect.top;
        const tokenX = tokenRect.left + tokenRect.width / 2 - rootRect.left;
        const tokenY = tokenRect.top - rootRect.top;
        const bendY = sourceY + Math.max(22, (tokenY - sourceY) * .46);
        return [{
          key: `${source.sourceId}-${tokenIndex}`,
          d: `M ${sourceX} ${sourceY} C ${sourceX} ${bendY}, ${tokenX} ${bendY}, ${tokenX} ${tokenY}`,
          effect: source.effect,
          selected: selectedSourceId
            ? source.sourceId === selectedSourceId
            : tokenIndex === selectedToken,
          sourceFocused: selectedSourceId === source.sourceId,
          muted: Boolean(selectedSourceId && source.sourceId !== selectedSourceId),
        }];
      }),
    );
    setPaths(next);
  }, [selectedSourceId, selectedToken, tokens]);

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const frame = requestAnimationFrame(measurePaths);
    const observer = new ResizeObserver(measurePaths);
    observer.observe(root);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [measurePaths]);

  function handleTokenKeys(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowRight") next = Math.min(tokens.length - 1, index + 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tokens.length - 1;
    else return;
    event.preventDefault();
    onSelectSource(null);
    onSelectToken(next);
    requestAnimationFrame(() => tokenRefs.current.get(next)?.focus());
  }

  const selectedSourceTokenIndexes = new Set(
    selectedSourceId
      ? tokens.flatMap((token, index) =>
          token.sources?.some((source) => source.sourceId === selectedSourceId) ? [index] : [])
      : [],
  );

  return (
    <div className="thread-trace" ref={rootRef}>
      <svg className="thread-trace-field" aria-hidden="true">
        <defs>
          <linearGradient id="thread-support" x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="var(--signal-mint)" stopOpacity=".8" />
            <stop offset="1" stopColor="var(--signal-cyan)" stopOpacity=".22" />
          </linearGradient>
          <linearGradient id="thread-suppress" x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="var(--signal-peach)" stopOpacity=".74" />
            <stop offset="1" stopColor="var(--signal-pink)" stopOpacity=".22" />
          </linearGradient>
          <filter id="thread-glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {paths.map((path) => (
          <path
            className={`thread-trace-link is-${path.effect} ${path.selected ? "is-selected" : ""} ${path.sourceFocused ? "is-source-focused" : ""} ${path.muted ? "is-muted" : ""}`}
            d={path.d}
            key={path.key}
          />
        ))}
      </svg>

      <div className="thread-source-row">
        {sources.map((source, index) => (
          <button
            type="button"
            className={`thread-source-node ${selectedSourceId === source.id ? "is-selected" : ""} ${selectedSourceId && selectedSourceId !== source.id ? "is-muted" : ""} ${source.measured === false ? "is-omitted" : ""}`}
            aria-label={`Context span: ${source.text}`}
            aria-pressed={selectedSourceId === source.id}
            onClick={() => onSelectSource(selectedSourceId === source.id ? null : source.id)}
            ref={(node) => {
              if (node) sourceRefs.current.set(source.id, node);
              else sourceRefs.current.delete(source.id);
            }}
            style={{ "--source-x": `${sourcePosition(index, sources.length)}%` } as CSSProperties}
            title={source.text}
            key={source.id}
          >
            <span>{source.measured === false ? "NOT MEASURED" : source.role || "CONTEXT"}</span>
            <strong>{source.label || source.text}</strong>
          </button>
        ))}
      </div>

      <div className="thread-token-row" role="listbox" aria-label="Recorded output tokens">
        {tokens.map((token, index) => {
          const hasBreak = token.text.includes("\n");
          const whitespace = Boolean(token.text) && !token.text.trim();
          const sourceMatch = selectedSourceTokenIndexes.has(index);
          return (
            <Fragment key={`${index}-${token.text}`}>
              <button
                type="button"
                role="option"
                aria-label={`Token ${index + 1}: ${token.text || "blank"}`}
                aria-selected={index === selectedToken}
                tabIndex={index === selectedToken ? 0 : -1}
                className={`${index === selectedToken ? "is-selected" : ""} ${sourceMatch ? "is-source-match" : ""} ${selectedSourceId && !sourceMatch ? "is-source-muted" : ""} ${whitespace ? "is-whitespace" : ""} ${!token.text ? "is-terminal" : ""} band-${token.band ?? "none"}`}
                onClick={() => {
                  onSelectSource(null);
                  onSelectToken(index);
                }}
                onKeyDown={(event) => handleTokenKeys(event, index)}
                ref={(node) => {
                  if (node) tokenRefs.current.set(index, node);
                  else tokenRefs.current.delete(index);
                }}
                style={{ "--token-confidence": token.confidence ?? 0 } as CSSProperties}
              >
                <span>{visibleToken(token.text)}</span>
              </button>
              {hasBreak && <span className="thread-trace-break" aria-hidden="true" />}
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
