import { useCallback, useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icon";
import { loadRuntimeState } from "../data/api";
import type { RuntimeState, Theme } from "../data/types";
import { panelRegistry, resolveRoute } from "../panels/registry";
import { TopbarProvider } from "../panels/topbar";
import type { TopbarContent } from "../panels/topbar";

/**
 * The shell: theme, runtime status, the inspector toggle, and the hash router. Everything surface-
 * specific lives in `src/panels/*.tsx` and is discovered at build time -- see `docs/SURFACES.md`.
 *
 * This file used to hold a hardcoded `nav` array, a `Route` union, a `readRoute()` chain, a JSX switch,
 * and the Scope surface's entire state. Adding a surface meant editing all five. It now holds none of
 * them, and adding a surface is adding one file.
 */
export function App() {
  const [theme, setTheme] = useState<Theme>(() =>
    localStorage.getItem("clozn.next.theme") === "cathedral" ? "cathedral" : "halo",
  );
  const [runtime, setRuntime] = useState<RuntimeState>({ status: "checking", runs: [] });
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [hash, setHash] = useState(() => location.hash);
  const [published, setPublished] = useState<TopbarContent | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("clozn.next.theme", theme);
  }, [theme]);

  useEffect(() => {
    const controller = new AbortController();
    void loadRuntimeState(controller.signal).then((nextRuntime) => {
      setRuntime(nextRuntime);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const onHashChange = () => setHash(location.hash);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Stable identity: a new object each render would re-run every panel's useTopbar effect forever.
  const publish = useCallback((content: TopbarContent | null) => setPublished(content), []);
  const topbar = useMemo(() => ({ publish }), [publish]);

  const { panels, loadFailures } = panelRegistry;
  const navPanels = panels.filter((item) => !item.hiddenFromNav);
  const resolved = resolveRoute(panels, hash);

  if (!resolved) {
    // Only reachable if the panels directory is empty, which the registry test would have caught.
    return <div className="studio"><main className="workspace">No Studio surfaces are registered.</main></div>;
  }

  const { panel, params } = resolved;
  const ctx = { runtime, inspectorOpen, params };
  const PanelComponent = panel.Component;

  // A panel publishing through useTopbar wins over its own static fields -- a surface whose stats
  // depend on internal state cannot express them statically, so the dynamic value is the truthful one.
  const stats = published?.stats ?? panel.topStats?.(ctx);
  const modeChip = published?.modeChip ?? panel.modeChip?.(ctx);

  return (
    <TopbarProvider value={topbar}>
      <div className="studio">
        <aside className="rail" aria-label="Primary navigation">
          <div className="wordmark" aria-label="Clozn">C</div>
          <nav className="rail-nav">
            {navPanels.map((item) => (
              <a
                className={`rail-item ${item.id === panel.id ? "is-active" : ""}`}
                href={`#/${item.id}`}
                key={item.id}
                aria-current={item.id === panel.id ? "page" : undefined}
              >
                {item.icon()}
                <span>{item.navLabel}</span>
              </a>
            ))}
            {loadFailures.map((failure) => (
              <span className="rail-item is-failed" key={failure.path} title={failure.reason}>
                <span>{failure.id} failed to load</span>
              </span>
            ))}
          </nav>
          <div className="rail-actions">
            {panel.showInspectorToggle !== false && (
              <button
                className="rail-button"
                type="button"
                onClick={() => setInspectorOpen((open) => !open)}
                aria-pressed={inspectorOpen}
                aria-label="Toggle inspector"
              >
                <Icon name="inspector" />
              </button>
            )}
            <button
              className="rail-button"
              type="button"
              onClick={() => setTheme((value) => value === "halo" ? "cathedral" : "halo")}
              aria-label="Toggle theme"
            >
              <Icon name="theme" />
            </button>
          </div>
        </aside>

        <header className="topbar">
          <div className="brand-lockup">
            <strong>CLOZN</strong>
            <span>STUDIO</span>
          </div>
          <div className="top-divider" />
          <span className="route-name">{panel.routeName(params)}</span>
          <div className="topbar-spacer" />
          <span className={`runtime-state is-${runtime.status}`}>
            <i />
            {runtime.status.toUpperCase()}
          </span>
          {stats}
          {modeChip != null && <span className="mode-chip">{modeChip}</span>}
          <div className="topbar-actions">
            {panel.showInspectorToggle !== false && (
              <button
                type="button"
                onClick={() => setInspectorOpen((open) => !open)}
                aria-pressed={inspectorOpen}
                aria-label="Toggle compact inspector"
              >
                <Icon name="inspector" />
              </button>
            )}
            <button
              type="button"
              onClick={() => setTheme((value) => value === "halo" ? "cathedral" : "halo")}
              aria-label="Toggle compact theme"
            >
              <Icon name="theme" />
            </button>
          </div>
        </header>

        <main className={`workspace is-${panel.id} ${inspectorOpen ? "" : "inspector-closed"}`}>
          {/* Keyed by panel id so switching surfaces remounts rather than reusing state across them --
              the old JSX switch got this for free by rendering different component types. */}
          <PanelComponent key={panel.id} {...ctx} />
        </main>
      </div>
    </TopbarProvider>
  );
}
