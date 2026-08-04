import type { RunPerformance, RunSummary } from "../../data/types";
import "./RunWorkspaceHeader.css";

interface RunWorkspaceHeaderProps {
  run?: RunSummary | null;
  performance?: RunPerformance | null;
  active: "lens" | "diagnostics";
}

function humanize(value?: string | null) {
  return value ? value.replaceAll("_", " ").replaceAll("-", " ") : "Not recorded";
}

function shortId(value?: string) {
  return value ? value.slice(-8) : "—";
}

export function RunWorkspaceHeader({ run, performance, active }: RunWorkspaceHeaderProps) {
  const runId = run?.id;
  return (
    <footer className="run-workspace-header">
      <div className="run-workspace-identity">
        <span className="run-workspace-label">Selected</span>
        <strong>{run?.label || "No run selected"}</strong>
        <p>
          <code>{shortId(runId)}</code>
          <span>{run?.model || "Model unavailable"}</span>
          <span>{run?.duration || performance?.totalDuration?.value || "Duration unavailable"}</span>
          <span>{humanize(run?.finishReason)}</span>
        </p>
      </div>

      {runId && (
        <nav className="run-workspace-switcher" aria-label="Run workspace">
          <a
            className={active === "lens" ? "is-active" : ""}
            aria-current={active === "lens" ? "page" : undefined}
            href={`#/runs/${encodeURIComponent(runId)}/lens`}
          >Lens</a>
          <a
            className={active === "diagnostics" ? "is-active" : ""}
            aria-current={active === "diagnostics" ? "page" : undefined}
            href={`#/runs/${encodeURIComponent(runId)}/diagnostics`}
          >Diagnostics</a>
        </nav>
      )}
    </footer>
  );
}
