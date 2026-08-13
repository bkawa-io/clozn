import { useEffect, useState, type ReactNode } from "react";
import type { StudioRoute } from "./router";

export type StudioTheme = "lunar" | "pearl";

interface AppShellProps {
  route: StudioRoute;
  runtimeStatus: "checking" | "ready" | "degraded" | "not-ready" | "unreachable";
  children: ReactNode;
}

const NAV: ReadonlyArray<{ surface: StudioRoute["surface"]; label: string; href: string; mark: string }> = [
  { surface: "runs", label: "Runs", href: "#/runs", mark: "R" },
  { surface: "inspect", label: "Inspect", href: "#/runs", mark: "I" },
  { surface: "time-travel", label: "Time Travel", href: "#/time-travel", mark: "T" },
  { surface: "compare", label: "Compare", href: "#/compare", mark: "C" },
  { surface: "mri", label: "Model MRI", href: "#/mri", mark: "M" },
  { surface: "runtime", label: "Runtime", href: "#/runtime", mark: "S" },
];

function initialTheme(): StudioTheme {
  return localStorage.getItem("clozn.studio.v2.theme") === "pearl" ? "pearl" : "lunar";
}

export function AppShell({ route, runtimeStatus, children }: AppShellProps) {
  const [theme, setTheme] = useState<StudioTheme>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("clozn.studio.v2.theme", theme);
  }, [theme]);

  return (
    <div className="studio-shell">
      <header className="studio-topbar">
        <a className="studio-brand" href="#/runs" aria-label="CLOZN Studio home">
          <strong>CLOZN</strong>
          <span>STUDIO</span>
        </a>
        <p className="studio-thesis">Precision debugger for local model behavior</p>
        <a className={`runtime-chip is-${runtimeStatus}`} href="#/runtime">
          <i aria-hidden="true" />
          Runtime · {runtimeStatus.replace("-", " ").toUpperCase()}
        </a>
        <button
          className="theme-toggle"
          type="button"
          aria-label={`Use ${theme === "lunar" ? "Black Pearl" : "Lunar Nacre"} theme`}
          onClick={() => setTheme((current) => current === "lunar" ? "pearl" : "lunar")}
        >
          {theme === "lunar" ? "LUNAR NACRE" : "BLACK PEARL"}
        </button>
      </header>

      <nav className="studio-rail" aria-label="Primary navigation">
        {NAV.map((item) => {
          const active = item.surface === route.surface;
          return (
            <a key={item.surface} href={item.href} className={active ? "is-active" : undefined} aria-current={active ? "page" : undefined}>
              <span aria-hidden="true">{item.mark}</span>
              <b>{item.label}</b>
            </a>
          );
        })}
      </nav>

      <main className={`studio-workspace is-${route.surface}`}>
        {children}
      </main>
    </div>
  );
}
