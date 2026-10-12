import { useEffect, useState } from "react";
import { routeHref, type InspectRoute, type InspectView } from "./router";
import { useHashRoute } from "./useHashRoute";
import { InspectSurface } from "../features/inspect/Inspect";
import { SessionsSurface } from "../features/sessions/Sessions";
import { RuntimeSurface } from "../features/runtime/Runtime";
import { CompareSurface } from "../features/compare/Compare";

type Theme = "lunar" | "pearl";

type InspectDestinationView = Exclude<InspectView, "read">;

const INSPECT_DESTINATIONS: ReadonlyArray<{ view: InspectDestinationView; label: string }> = [
  { view: "context-effects", label: "Context effects" },
  { view: "context", label: "Received context" },
  { view: "receipt", label: "Context receipt" },
  { view: "minimal-context", label: "Minimal context" },
  { view: "actions", label: "Actions" },
  { view: "claims", label: "Claims" },
];

function inspectDestination(route: InspectRoute, view: InspectDestinationView): InspectRoute {
  const base = {
    surface: "inspect" as const,
    runId: route.runId,
    ...(route.sessionId ? { sessionId: route.sessionId } : {}),
  };
  return view === "context-effects"
    ? { ...base, view }
    : { ...base, view };
}

function InspectNavigation({ route }: { route: InspectRoute }) {
  const selected = !route.view || route.view === "read" ? "context-effects" : route.view;
  return (
    <nav className="inspect-view-nav" aria-label="Inspect views">
      {INSPECT_DESTINATIONS.map(({ view, label }) => (
        <a
          key={view}
          href={routeHref(inspectDestination(route, view))}
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
  const sessionsHref = route.surface === "inspect" && route.sessionId
    ? routeHref({ surface: "session", sessionId: route.sessionId, runId: route.runId })
    : routeHref({ surface: "sessions" });
  const inspectHref = route.surface === "inspect"
    ? routeHref(route)
    : route.surface === "session" && route.runId
      ? routeHref({ surface: "inspect", runId: route.runId, sessionId: route.sessionId })
      : undefined;
  const surfaceLabel = route.surface === "inspect"
    ? "Inspect"
    : route.surface === "runtime"
      ? "Runtime"
      : route.surface === "compare"
        ? "Compare"
        : "Sessions";
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("clozn.studio.theme", theme);
  }, [theme]);
  useEffect(() => {
    document.title = `CLOZN · ${surfaceLabel}`;
  }, [surfaceLabel]);
  return (
    <div className="studio-shell">
      <header className="studio-topbar">
        <a className="studio-brand" href="#/sessions" aria-label={`CLOZN ${surfaceLabel}`}>
          <strong>CLOZN</strong><span>{surfaceLabel.toUpperCase()}</span>
        </a>
        <span className="studio-topbar-seam" aria-hidden="true" />
        <button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === "lunar" ? "pearl" : "lunar")} aria-label={`Use ${theme === "lunar" ? "Black Pearl" : "Lunar Nacre"} theme`}>
          {theme === "lunar" ? "LUNAR NACRE" : "BLACK PEARL"}
        </button>
      </header>
      <nav className="studio-rail" aria-label="Primary navigation">
        <a href={sessionsHref} className={route.surface === "sessions" || route.surface === "session" ? "is-active" : undefined} aria-current={route.surface === "sessions" || route.surface === "session" ? "page" : undefined}>
          <span aria-hidden="true">S</span><b>Sessions</b>
        </a>
        <a
          href={inspectHref ?? sessionsHref}
          className={route.surface === "inspect" ? "is-active" : undefined}
          aria-current={route.surface === "inspect" ? "page" : undefined}
          aria-disabled={inspectHref ? undefined : "true"}
          onClick={(event) => { if (!inspectHref) event.preventDefault(); }}
        >
          <span aria-hidden="true">I</span><b>Inspect</b>
        </a>
        <a href={routeHref({ surface: "runtime" })} className={route.surface === "runtime" ? "is-active" : undefined} aria-current={route.surface === "runtime" ? "page" : undefined}>
          <span aria-hidden="true">R</span><b>Runtime</b>
        </a>
        <a href={routeHref({ surface: "compare" })} className={route.surface === "compare" ? "is-active" : undefined} aria-current={route.surface === "compare" ? "page" : undefined}>
          <span aria-hidden="true">Δ</span><b>Compare</b>
        </a>
      </nav>
      <main className="studio-workspace">
        {route.surface === "inspect"
          ? <><InspectNavigation route={route} /><InspectSurface route={route} /></>
          : route.surface === "runtime"
            ? <RuntimeSurface />
            : route.surface === "compare"
              ? <CompareSurface route={route} />
              : <SessionsSurface route={route} />}
      </main>
    </div>
  );
}
