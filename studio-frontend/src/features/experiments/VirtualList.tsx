import { useRef, useState, type ReactNode } from "react";

/**
 * Windowed row rendering for a potentially large list, matching the manual scroll-window technique
 * `features/observatory/TraceScope.tsx` already uses for its context source list (row height + overscan,
 * absolute-positioned rows inside a full-height spacer, tracked via `scrollTop` state) -- reused here
 * rather than inventing a second virtualization approach, per the roadmap plan's instruction to follow
 * Model Scope's existing pattern. Generic over row type so the experiment matrix's two suite sections
 * (target rows, guard rows) can each use one instance.
 */
export function VirtualList<T>({
  items,
  rowHeight,
  overscan = 6,
  fallbackViewportHeight = 320,
  renderRow,
  keyFor,
  ariaLabel,
  emptyLabel,
  className,
}: {
  items: T[];
  rowHeight: number;
  overscan?: number;
  /** Used before the container has been laid out (first paint, or under SSR where clientHeight is 0). */
  fallbackViewportHeight?: number;
  renderRow: (item: T, index: number) => ReactNode;
  keyFor: (item: T, index: number) => string;
  ariaLabel: string;
  emptyLabel: string;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);

  if (items.length === 0) {
    return (
      <div className={`virtual-list ${className ?? ""}`} role="list" aria-label={ariaLabel}>
        <div className="virtual-list-empty">{emptyLabel}</div>
      </div>
    );
  }

  const viewportHeight = containerRef.current?.clientHeight || fallbackViewportHeight;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(items.length, Math.ceil((scrollTop + viewportHeight) / rowHeight) + overscan);
  const visible = items.slice(start, end);

  return (
    <div
      className={`virtual-list ${className ?? ""}`}
      ref={containerRef}
      role="list"
      aria-label={ariaLabel}
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <div className="virtual-list-spacer" style={{ height: items.length * rowHeight }}>
        {visible.map((item, offset) => {
          const index = start + offset;
          return (
            <div
              className="virtual-list-row"
              role="listitem"
              style={{ transform: `translateY(${index * rowHeight}px)`, height: rowHeight }}
              key={keyFor(item, index)}
            >
              {renderRow(item, index)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
