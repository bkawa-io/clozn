import { useEffect } from "react";
import { routeHref, type InspectRoute, type InspectView } from "./router";
import { useHashRoute } from "./useHashRoute";
import { InspectSurface } from "../features/inspect/Inspect";
import { RunsSurface } from "../features/runs/Runs";

const WORKSPACE_MODES: ReadonlyArray<{ view: InspectView; label: string }> = [
  { view: "context-effects", label: "Context Effects" },
  { view: "lean-context", label: "Lean Context" },
];

export function App() {
  const route = useHashRoute();
  const surfaceLabel = route.surface === "inspect"
    ? "Message analysis"
    : route.conversationId ? "Conversation" : "Conversations";
  useEffect(() => {
    document.title = `CLOZN · ${surfaceLabel}`;
  }, [surfaceLabel]);
  return (
    <div className="studio-shell studio-shell-v1">
      <header className="studio-topbar">
        <a className="studio-brand" href={routeHref({ surface: "runs" })} aria-label="CLOZN">
          <i aria-hidden="true" /><strong>CLOZN</strong>
        </a>
        <nav className="studio-primary" aria-label="Primary navigation">
          <a href={routeHref({ surface: "runs" })} aria-current={route.surface === "runs" ? "page" : undefined}>Conversations</a>
          {route.surface === "inspect" && WORKSPACE_MODES.map(({ view, label }) => (
            <a
              key={view}
              href={routeHref({
                surface: "inspect",
                runId: route.runId,
                ...(route.conversationId ? { conversationId: route.conversationId } : {}),
                view,
              })}
              aria-current={(route.view ?? "context-effects") === view ? "page" : undefined}
            >
              {label}
            </a>
          ))}
        </nav>
      </header>
      <main className="studio-workspace">
        {route.surface === "inspect"
          ? <InspectSurface route={route} />
          : <RunsSurface route={route} />}
      </main>
    </div>
  );
}
