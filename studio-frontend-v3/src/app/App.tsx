import { useEffect, useState } from "react";
import { useHashRoute } from "./useHashRoute";
import { SessionsSurface } from "../features/sessions/Sessions";

type Theme = "lunar" | "pearl";

function initialTheme(): Theme {
  return localStorage.getItem("clozn.studio.v3.theme") === "pearl" ? "pearl" : "lunar";
}

export function App() {
  const route = useHashRoute();
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("clozn.studio.v3.theme", theme);
  }, [theme]);
  return (
    <div className="v3-shell">
      <header className="v3-topbar">
        <a className="v3-brand" href="#/sessions" aria-label="CLOZN Sessions">
          <strong>CLOZN</strong><span>SESSIONS</span>
        </a>
        <span className="v3-topbar-seam" aria-hidden="true" />
        <button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === "lunar" ? "pearl" : "lunar")} aria-label={`Use ${theme === "lunar" ? "Black Pearl" : "Lunar Nacre"} theme`}>
          {theme === "lunar" ? "LUNAR NACRE" : "BLACK PEARL"}
        </button>
      </header>
      <nav className="v3-rail" aria-label="Primary navigation">
        <a href="#/sessions" className={route.surface === "sessions" || route.surface === "session" ? "is-active" : undefined} aria-current={route.surface === "sessions" || route.surface === "session" ? "page" : undefined}>
          <span aria-hidden="true">S</span><b>Sessions</b>
        </a>
      </nav>
      <main className="v3-workspace"><SessionsSurface route={route} /></main>
    </div>
  );
}
