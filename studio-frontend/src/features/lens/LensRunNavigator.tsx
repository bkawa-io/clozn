import type { RunSummary } from "../../data/types";
import "./LensRunNavigator.css";

export interface LensRunNavigatorProps {
  /** The caller owns ordering and filtering; this rail only presents the supplied run collection. */
  runs: readonly RunSummary[];
  /** Controlled active run id. An unknown or omitted id leaves every row unselected. */
  selectedRunId?: string | null;
  /** Called only from an explicit row selection. */
  onSelectRun: (runId: string) => void;
  title?: string;
}

export type LensRunNavigatorStatus = "error" | "truncated" | "complete" | "recorded";

function shortId(id: string) {
  return id.length > 10 ? id.slice(-8) : id;
}

function compactText(value: string, maxLength = 72) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 1).trimEnd()}…` : normalized;
}

function humanize(value?: string) {
  if (!value) return "NOT RECORDED";
  return value
    .trim()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .toUpperCase();
}

export function lensRunNavigatorStatus(run: RunSummary): LensRunNavigatorStatus {
  if (run.flags.includes("error")) return "error";
  if (run.finishReason === "length" || run.flags.includes("truncated")) return "truncated";
  if (run.finishReason) return "complete";
  return "recorded";
}

export function lensRunNavigatorLabel(run: RunSummary) {
  return compactText(run.prompt) || compactText(run.label) || "Untitled prompt";
}

function durationLabel(run: RunSummary) {
  if (run.duration?.trim()) return run.duration;
  if (run.durationMs != null) return `${Math.round(run.durationMs)} ms`;
  return "DURATION UNAVAILABLE";
}

function statusLabel(status: LensRunNavigatorStatus) {
  switch (status) {
    case "error": return "ERROR";
    case "truncated": return "TRUNCATED";
    case "complete": return "COMPLETE";
    case "recorded": return "RECORDED";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

/**
 * A controlled, presentational run rail for Lens's left column.  It deliberately does not fetch,
 * filter, mutate, or infer a selected run: callers provide the visible list and own selection routing.
 */
export function LensRunNavigator({
  runs,
  selectedRunId = null,
  onSelectRun,
  title = "Runs",
}: LensRunNavigatorProps) {
  return (
    <nav className="lens-run-navigator" aria-labelledby="lens-run-navigator-title">
      <header className="lens-run-navigator-head">
        <div>
          <span className="eyebrow">RUN HISTORY</span>
          <h2 id="lens-run-navigator-title">{title}</h2>
        </div>
        <span>{runs.length}</span>
      </header>

      {runs.length ? (
        <ol className="lens-run-navigator-list">
          {runs.map((run) => {
            const selected = run.id === selectedRunId;
            const status = lensRunNavigatorStatus(run);
            const label = lensRunNavigatorLabel(run);
            return (
              <li key={run.id} className={selected ? "is-selected" : undefined}>
                <button
                  type="button"
                  aria-current={selected ? "page" : undefined}
                  aria-pressed={selected}
                  onClick={() => onSelectRun(run.id)}
                >
                  <span className="lens-run-navigator-row-topline">
                    <span className={`lens-run-navigator-status is-${status}`}>{statusLabel(status)}</span>
                    <time dateTime={run.createdAt}>{run.createdAt || "TIME UNAVAILABLE"}</time>
                  </span>
                  <strong title={run.prompt || run.label}>{label}</strong>
                  <span className="lens-run-navigator-model">{run.model || "MODEL UNAVAILABLE"}</span>
                  <span className="lens-run-navigator-row-bottomline">
                    <span>{durationLabel(run)}</span>
                    <span>FINISH {humanize(run.finishReason)}</span>
                  </span>
                  <small title={run.id}>RUN {shortId(run.id)}</small>
                </button>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="lens-run-navigator-empty">No recorded runs are available.</p>
      )}
    </nav>
  );
}
