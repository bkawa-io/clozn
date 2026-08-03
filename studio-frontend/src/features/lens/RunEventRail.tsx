import type { ReactNode } from "react";
import "./run-frame.css";

/**
 * `token` is deliberately supported only so adapters can pass through raw traces safely. Token-level
 * items are omitted from the rail: this component is for run phases and recorded evidence, not a
 * second response-token viewer.
 */
export type RunEventKind =
  | "run-start"
  | "prompt"
  | "context"
  | "model-load"
  | "generation"
  | "artifact"
  | "warning"
  | "branch"
  | "run-finish"
  | "error"
  | "token"
  | "custom";

export interface RunEventRailEvent {
  id: string;
  label: string;
  kind: RunEventKind;
  detail?: ReactNode;
  timestamp?: string;
  status?: "complete" | "active" | "pending" | "warning" | "error" | "unavailable";
  /** The producer may aggregate many raw events into this marker. */
  count?: number;
  /** Token-granularity events are intentionally suppressed from this semantic rail. */
  granularity?: "run" | "phase" | "artifact" | "token";
  href?: string;
}

export interface RunEventRailProps {
  events: readonly RunEventRailEvent[];
  selectedEventId?: string | null;
  onSelectEvent?: (event: RunEventRailEvent) => void;
  emptyMessage?: string;
  className?: string;
  ariaLabel?: string;
}

function eventClass(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "custom";
}

function eventTime(value?: string) {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

function MarkerContent({ event }: { event: RunEventRailEvent }) {
  return (
    <>
      <i aria-hidden="true" />
      <span className="run-event-rail-copy">
        <strong>{event.label}</strong>
        {(event.detail || event.count !== undefined) && (
          <small>
            {event.detail}
            {event.detail && event.count !== undefined && " · "}
            {event.count !== undefined && `${event.count} ${event.count === 1 ? "ITEM" : "ITEMS"}`}
          </small>
        )}
      </span>
      {event.timestamp && <time dateTime={event.timestamp}>{eventTime(event.timestamp)}</time>}
    </>
  );
}

/**
 * A compact, selectable sequence of meaningful run transitions. Feed it phase, artifact, warning, and
 * lineage events; raw token events are suppressed even if an adapter accidentally includes them.
 */
export function RunEventRail({
  events,
  selectedEventId,
  onSelectEvent,
  emptyMessage = "No semantic run events were recorded.",
  className,
  ariaLabel = "Run events",
}: RunEventRailProps) {
  const seen = new Set<string>();
  let omittedTokenEvents = 0;
  const semanticEvents = events.filter((event) => {
    if (event.kind === "token" || event.granularity === "token") {
      omittedTokenEvents += event.count ?? 1;
      return false;
    }
    if (seen.has(event.id)) return false;
    seen.add(event.id);
    return true;
  });

  return (
    <nav className={["run-event-rail", className].filter(Boolean).join(" ")} aria-label={ariaLabel}>
      {semanticEvents.length > 0 ? (
        <ol>
          {semanticEvents.map((event) => {
            const selected = event.id === selectedEventId;
            const state = event.status ?? (selected ? "active" : "complete");
            const itemClass = [
              "run-event-rail-item",
              `kind-${eventClass(event.kind)}`,
              `status-${eventClass(state)}`,
              selected ? "is-selected" : "",
            ].filter(Boolean).join(" ");
            const content = <MarkerContent event={event} />;

            return (
              <li className={itemClass} key={event.id}>
                {event.href ? (
                  <a
                    href={event.href}
                    aria-current={selected ? "step" : undefined}
                    onClick={() => onSelectEvent?.(event)}
                  >{content}</a>
                ) : onSelectEvent ? (
                  <button
                    type="button"
                    aria-pressed={selected}
                    aria-current={selected ? "step" : undefined}
                    onClick={() => onSelectEvent(event)}
                  >{content}</button>
                ) : (
                  <div aria-current={selected ? "step" : undefined}>{content}</div>
                )}
              </li>
            );
          })}
        </ol>
      ) : <p className="run-event-rail-empty">{emptyMessage}</p>}
      {omittedTokenEvents > 0 && (
        <p className="run-event-rail-omitted">
          {omittedTokenEvents.toLocaleString()} token event{omittedTokenEvents === 1 ? "" : "s"} omitted from this semantic rail.
        </p>
      )}
    </nav>
  );
}
