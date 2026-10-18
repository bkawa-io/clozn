import { useEffect, useState } from "react";
import { routeHref, type InspectRoute, type InspectView } from "./router";
import { useHashRoute } from "./useHashRoute";
import { InspectSurface } from "../features/inspect/Inspect";
import { RunsSurface } from "../features/runs/Runs";

type Theme = "lunar" | "pearl";

const WORKSPACE_MODES: ReadonlyArray<{ view: InspectView; label: string }> = [
  { view: "context-effects", label: "Context Effects" },
  { view: "minimal-context", label: "Minimal Context" },
];

function InspectNavigation({ route }: { route: InspectRoute }) {
  const selected = route.view ?? "context-effects";
  return (
    <nav className="inspect-view-nav" aria-label="Run workspace modes">
      {WORKSPACE_MODES.map(({ view, label }) => (
        <a
          key={view}
          href={routeHref({ surface: "inspect", runId: route.runId, view })}
          aria-current={selected === view ? "page" : undefined}
        >
          {label}
        </a>
      ))}
    </nav>
  );
}

function initialTheme(): Theme {
  return localStorage.getItem("clozn.studio.theme") === "pearl" ? "pearl" : "lunar";
}

export function App() {
  const route = useHashRoute();
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const surfaceLabel = route.surface === "inspect" ? "Run workspace" : "Runs";
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("clozn.studio.theme", theme);
  }, [theme]);
  useEffect(() => {
    document.title = `CLOZN · ${surfaceLabel}`;
  }, [surfaceLabel]);
  return (
    <div className="studio-shell studio-shell-v1">
      <header className="studio-topbar">
        <a className="studio-brand" href={routeHref({ surface: "runs" })} aria-label={`CLOZN ${surfaceLabel}`}>
          <strong>CLOZN</strong><span>{surfaceLabel.toUpperCase()}</span>
        </a>
        <nav className="studio-primary" aria-label="Primary navigation">
          <a href={routeHref({ surface: "runs" })} aria-current={route.surface === "runs" ? "page" : undefined}>Runs</a>
          {route.surface === "inspect" && <span aria-current="page">Run workspace</span>}
        </nav>
        <button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === "lunar" ? "pearl" : "lunar")} aria-label={`Use ${theme === "lunar" ? "Black Pearl" : "Lunar Nacre"} theme`}>
          {theme === "lunar" ? "LUNAR NACRE" : "BLACK PEARL"}
        </button>
      </header>
      <main className="studio-workspace">
        {route.surface === "inspect"
          ? <><InspectNavigation route={route} /><InspectSurface route={route} /></>
          : <RunsSurface />}
      </main>
    </div>
  );
}
