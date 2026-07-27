import { useEffect, useState } from "react";
import { Icon } from "../components/Icon";
import { createFork, loadRunInspection, loadRuntimeState } from "../data/api";
import { DEMO_OBSERVATORY } from "../data/demo";
import { stressFixture } from "../data/stress";
import type { ForkState, ObservatoryData, RuntimeState, Theme } from "../data/types";
import { Behavior } from "../features/behavior/Behavior";
import { Compare } from "../features/compare/Compare";
import { Lens } from "../features/lens/Lens";
import { Model } from "../features/model/Model";
import { Observatory } from "../features/observatory/Observatory";
import { Runs } from "../features/runs/Runs";

const nav = [
  { id: "runs", label: "Runs", icon: "runs", href: "#/runs" },
  { id: "lens", label: "Lens", icon: "lens", href: "#/lens" },
  { id: "scope", label: "Scope", icon: "observatory", href: "#/scope" },
  { id: "compare", label: "Compare", icon: "compare", href: "#/compare" },
  { id: "behavior", label: "Behavior", icon: "behavior", href: "#/behavior" },
  { id: "model", label: "Model", icon: "model", href: "#/model" },
] as const;

type Route =
  | { kind: "runs" }
  | { kind: "lens"; runId?: string }
  | { kind: "scope"; runId?: string; tokenIndex?: number; fixture?: string }
  | { kind: "compare"; runA?: string; runB?: string }
  | { kind: "behavior" }
  | { kind: "model" };

function readRoute(): Route {
  const compare = location.hash.match(/^#\/compare(?:\/([^/]+)\/([^/]+))?$/);
  if (compare) {
    return {
      kind: "compare",
      runA: compare[1] ? decodeURIComponent(compare[1]) : undefined,
      runB: compare[2] ? decodeURIComponent(compare[2]) : undefined,
    };
  }
  if (/^#\/runs\/?$/.test(location.hash)) return { kind: "runs" };
  const scope = location.hash.match(/^#\/runs\/([^/]+)\/scope(?:\?token=(\d+))?$/);
  const scopeFixture = location.hash.match(/^#\/scope\/?\?fixture=([^&]+)$/);
  if (scope || scopeFixture || /^#\/scope\/?$/.test(location.hash)) {
    return {
      kind: "scope",
      runId: scope ? decodeURIComponent(scope[1]) : undefined,
      tokenIndex: scope?.[2] == null ? undefined : Number(scope[2]),
      fixture: scopeFixture ? decodeURIComponent(scopeFixture[1]) : undefined,
    };
  }
  if (/^#\/lens\/?$/.test(location.hash)) return { kind: "lens" };
  const lens = location.hash.match(/^#\/runs\/([^/]+)$/);
  if (lens) return { kind: "lens", runId: decodeURIComponent(lens[1]) };
  if (/^#\/behavior\/?$/.test(location.hash)) return { kind: "behavior" };
  if (/^#\/model\/?$/.test(location.hash)) return { kind: "model" };
  return {
    kind: "runs",
  };
}

export function App() {
  const [theme, setTheme] = useState<Theme>(() =>
    localStorage.getItem("clozn.next.theme") === "cathedral" ? "cathedral" : "halo",
  );
  const [runtime, setRuntime] = useState<RuntimeState>({ status: "checking", runs: [] });
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [data, setData] = useState<ObservatoryData>(DEMO_OBSERVATORY);
  const [runStatus, setRunStatus] = useState<"idle" | "loading" | "error">("idle");
  const [forkState, setForkState] = useState<ForkState>({ status: "idle" });
  const [route, setRoute] = useState<Route>(readRoute);

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
    const onHashChange = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    if (route.kind !== "scope") return;
    if (route.runId) void selectRun(route.runId);
    else if (route.fixture) {
      setData(stressFixture(route.fixture) ?? DEMO_OBSERVATORY);
      setRunStatus("idle");
      setForkState({ status: "idle" });
    }
    else {
      setData(DEMO_OBSERVATORY);
      setRunStatus("idle");
      setForkState({ status: "idle" });
    }
  }, [
    route.kind,
    route.kind === "scope" ? route.runId : undefined,
    route.kind === "scope" ? route.fixture : undefined,
  ]);

  async function selectRun(runId: string) {
    setForkState({ status: "idle" });
    if (!runId) {
      setData(DEMO_OBSERVATORY);
      setRunStatus("idle");
      history.replaceState(null, "", "#/scope");
      return;
    }
    setRunStatus("loading");
    try {
      const inspection = await loadRunInspection(runId);
      setData(inspection);
      setRunStatus("idle");
      history.replaceState(null, "", `#/runs/${encodeURIComponent(runId)}/scope`);
    } catch {
      setRunStatus("error");
    }
  }

  async function forkRun(position: number, token: string) {
    if (data.mode !== "run") return;
    const parentId = data.id;
    setForkState({ status: "loading", parentId });
    try {
      const child = await createFork(parentId, position, token);
      const [inspection, nextRuntime] = await Promise.all([
        loadRunInspection(child.id),
        loadRuntimeState(),
      ]);
      setData(inspection);
      setRuntime(nextRuntime);
      setRunStatus("idle");
      setForkState({
        status: "success",
        parentId: child.parentId,
        childId: child.id,
        note: child.note,
      });
      history.replaceState(null, "", `#/runs/${encodeURIComponent(child.id)}/scope`);
    } catch (error) {
      setForkState({
        status: "error",
        parentId,
        message: error instanceof Error ? error.message : "Fork failed",
      });
    }
  }

  const activeNav = route.kind;

  return (
    <div className="studio">
      <aside className="rail" aria-label="Primary navigation">
        <div className="wordmark" aria-label="Clozn">C</div>
        <nav className="rail-nav">
          {nav.map((item) => (
            <a
              className={`rail-item ${item.id === activeNav ? "is-active" : ""}`}
              href={item.href}
              key={item.id}
              aria-current={item.id === activeNav ? "page" : undefined}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
        <div className="rail-actions">
          <button
            className="rail-button"
            type="button"
            onClick={() => setInspectorOpen((open) => !open)}
            aria-pressed={inspectorOpen}
            aria-label="Toggle inspector"
          >
            <Icon name="inspector" />
          </button>
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
        <span className="route-name">
          {route.kind === "compare"
            ? "COMPARE"
            : route.kind === "model"
              ? "MODEL"
            : route.kind === "behavior"
              ? "BEHAVIOR"
            : route.kind === "scope"
              ? "MODEL SCOPE"
              : route.kind === "lens"
                ? "LENS"
                : "RUNS"}
        </span>
        <div className="topbar-spacer" />
        <span className={`runtime-state is-${runtime.status}`}>
          <i />
          {runtime.status.toUpperCase()}
        </span>
        {route.kind === "scope" && (
          <span className="top-stat">
            <b>MODEL</b>
            {data.model === "local" ? runtime.engine?.model ?? data.model : data.model}
          </span>
        )}
        {route.kind === "scope" && <span className="top-stat"><b>RUN</b>{data.id}</span>}
        {route.kind === "runs" && <span className="top-stat"><b>RECORDED</b>{runtime.runs.length}</span>}
        {route.kind === "lens" && <span className="top-stat"><b>RECORDED</b>{runtime.runs.length}</span>}
        {route.kind === "behavior" && <span className="top-stat"><b>MODEL</b>{runtime.engine?.model ?? "—"}</span>}
        {route.kind === "model" && <span className="top-stat"><b>MODEL</b>{runtime.engine?.model ?? "—"}</span>}
        <span className="mode-chip">
          {route.kind === "compare"
            ? "A / B"
            : route.kind === "model"
              ? "READOUT"
            : route.kind === "behavior"
              ? "READ / WRITE"
            : route.kind === "runs"
              ? "LEDGER"
            : route.kind === "lens"
              ? "READOUT"
            : forkState.status === "loading"
            ? "FORKING"
            : runStatus === "loading"
              ? "LOADING"
              : runStatus === "error" || forkState.status === "error"
                ? "ERROR"
                : data.mode.toUpperCase()}
        </span>
        <div className="topbar-actions">
          <button
            type="button"
            onClick={() => setInspectorOpen((open) => !open)}
            aria-pressed={inspectorOpen}
            aria-label="Toggle compact inspector"
          >
            <Icon name="inspector" />
          </button>
          <button
            type="button"
            onClick={() => setTheme((value) => value === "halo" ? "cathedral" : "halo")}
            aria-label="Toggle compact theme"
          >
            <Icon name="theme" />
          </button>
        </div>
      </header>

      <main className={`workspace is-${route.kind} ${inspectorOpen ? "" : "inspector-closed"}`}>
        {route.kind === "compare" ? (
          <Compare
            key={`${route.runA ?? ""}:${route.runB ?? ""}`}
            runtime={runtime}
            initialA={route.runA}
            initialB={route.runB}
            inspectorOpen={inspectorOpen}
          />
        ) : route.kind === "behavior" ? (
          <Behavior runtime={runtime} inspectorOpen={inspectorOpen} />
        ) : route.kind === "model" ? (
          <Model inspectorOpen={inspectorOpen} />
        ) : route.kind === "runs" ? (
          <Runs runtime={runtime} inspectorOpen={inspectorOpen} />
        ) : route.kind === "lens" ? (
          <Lens
            key={route.runId ?? "latest"}
            runtime={runtime}
            initialRunId={route.runId}
            inspectorOpen={inspectorOpen}
          />
        ) : (
          <Observatory
            data={data}
            runtime={runtime}
            inspectorOpen={inspectorOpen}
            runStatus={runStatus}
            forkState={forkState}
            onSelectRun={(runId) => void selectRun(runId)}
            onFork={(position, token) => void forkRun(position, token)}
            initialTokenIndex={route.tokenIndex}
          />
        )}
      </main>
    </div>
  );
}
