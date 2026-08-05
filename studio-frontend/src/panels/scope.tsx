import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "../components/Icon";
import { loadRunInspection, loadRuntimeState } from "../data/api";
import { DEMO_OBSERVATORY } from "../data/demo";
import { stressFixture } from "../data/stress";
import type { ObservatoryData } from "../data/types";
import { Observatory } from "../features/observatory/Observatory";
import {
  parseScopeUrl,
  scopeRouteParams,
  scopeStateFromParams,
  serializeScopeUrl,
  type ScopeSelectionState,
  type ScopeUrlState,
} from "../features/observatory/urlState";
import { useTopbar } from "./topbar";
import type { PanelContext, StudioPanel } from "./types";

/**
 * The one panel that owns real state. `data`, `runStatus`, `selectRun` all lived in `App.tsx` before this
 * seam, alongside a topbar that reached into them through a chain of `route.kind === "scope" && ...`
 * conditionals. That is precisely the coupling the seam removes: App would otherwise have had to keep
 * owning one surface's state forever, and every future surface would have been tempted to add its own
 * branch beside it.
 *
 * Milestone F folded fork into the token workbench's own action tray (see
 * features/observatory/useTokenWorkbench.ts) -- this panel no longer owns `forkState`/`forkRun` at all;
 * a completed fork just calls `selectRun` with its new child run's id, exactly like picking a different
 * run from the dropdown would. `selectRun` also refreshes the live runtime snapshot on every call now
 * (not only after a fork) so the run list and engine facts never silently go stale after ANY navigation.
 *
 * The rest of the logic below is moved verbatim, not rewritten. It reports its topbar content through
 * `useTopbar` because App cannot see inside a panel, and should not.
 */
/**
 * The workbench remains independently routable for compatibility, but S2 embeds the exact same
 * instrument under a run section. `embedded` changes only navigation ownership: it never changes the
 * workbench's reads, actions, or evidence vocabulary.
 */
export interface ScopePanelProps extends PanelContext {
  embedded?: boolean;
  initialState?: ScopeUrlState;
  onRunChange?: (runId: string) => void;
  onEmbeddedStateChange?: (state: ScopeSelectionState) => void;
}

export function ScopePanel({
  runtime,
  inspectorOpen,
  params,
  embedded = false,
  initialState: embeddedInitialState,
  onRunChange,
  onEmbeddedStateChange,
}: ScopePanelProps) {
  const [data, setData] = useState<ObservatoryData>(DEMO_OBSERVATORY);
  const [runStatus, setRunStatus] = useState<"idle" | "loading" | "error">("idle");
  const [liveRuntime, setLiveRuntime] = useState(runtime);
  // One monotonic counter for selectRun: whichever call is issued LAST wins. A response for the run
  // displayed when it was requested must never land after the panel has since navigated elsewhere --
  // each async call captures its own id and checks it against this ref before committing state, so a
  // stale response for run A can never paint over run B.
  const requestIdRef = useRef(0);

  useEffect(() => setLiveRuntime(runtime), [runtime]);

  const runId = params.runId;
  const fixture = params.fixture;
  const routeState = embeddedInitialState ?? scopeStateFromParams(params);

  useEffect(() => {
    if (runId) void selectRun(runId);
    else if (fixture) {
      setData(stressFixture(fixture) ?? DEMO_OBSERVATORY);
      setRunStatus("idle");
    } else {
      setData(DEMO_OBSERVATORY);
      setRunStatus("idle");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- matches App.tsx's original dep list
  }, [runId, fixture]);

  async function selectRun(nextRunId: string) {
    const requestId = ++requestIdRef.current;
    if (!nextRunId) {
      setData(DEMO_OBSERVATORY);
      setRunStatus("idle");
      history.replaceState(null, "", "#/scope");
      return;
    }
    setRunStatus("loading");
    try {
      const [inspection, nextRuntime] = await Promise.all([
        loadRunInspection(nextRunId),
        loadRuntimeState(),
      ]);
      if (requestIdRef.current !== requestId) return;   // superseded by a later selection
      setData(inspection);
      setLiveRuntime(nextRuntime);
      setRunStatus("idle");
      if (!embedded) history.replaceState(null, "", `#/runs/${encodeURIComponent(nextRunId)}/scope`);
    } catch {
      if (requestIdRef.current !== requestId) return;
      setRunStatus("error");
    }
  }

  function requestRunSelection(nextRunId: string) {
    if (embedded && onRunChange) {
      // The parent owns the canonical `#/runs/<id>?section=mechanism` address. Keeping the fetch in
      // this workbench while letting it rewrite to `/scope` would make selecting a token appear to leave
      // the run reader, even though nothing about the evidence surface actually changed.
      onRunChange(nextRunId);
      return;
    }
    void selectRun(nextRunId);
  }

  const modelLabel = runStatus === "error"
    ? "UNAVAILABLE"
    : data.model === "local" ? liveRuntime.engine?.model ?? data.model : data.model;
  const runLabel = runStatus === "error" ? runId ?? "—" : data.id;

  useTopbar(
    () => embedded ? {} : ({
      stats: (
        <>
          <span className="top-stat"><b>MODEL</b>{modelLabel}</span>
          <span className="top-stat"><b>RUN</b>{runLabel}</span>
        </>
      ),
      modeChip:
        runStatus === "loading"
          ? "LOADING"
          : runStatus === "error"
            ? "ERROR"
            : data.mode.toUpperCase(),
    }),
    [embedded, modelLabel, data.id, data.mode, runStatus],
  );

  const tokenIndex = params.tokenIndex == null ? undefined : Number(params.tokenIndex);
  const initialState = data.mode === "run" && data.id === runId
    ? routeState
    : undefined;
  const replaceScopeState = useCallback((state: ScopeSelectionState) => {
    if (data.mode !== "run" || runStatus !== "idle") return;
    if (embedded) {
      onEmbeddedStateChange?.(state);
      return;
    }
    history.replaceState(null, "", serializeScopeUrl(data.id, state));
  }, [data.id, data.mode, embedded, onEmbeddedStateChange, runStatus]);

  if (runStatus === "error" && runId) {
    return (
      <section className="instrument scope-load-error" aria-labelledby="scope-load-error-title">
        <header className="instrument-head">
          <div>
            <span className="eyebrow">MODEL SCOPE</span>
            <h1 id="scope-load-error-title">Run not opened</h1>
          </div>
          <span className="mode-chip">ERROR</span>
        </header>
        <p className="scope-load-error-message" role="alert">
          Run &quot;{runId}&quot; could not be loaded. No evidence is available for this deep link.
        </p>
        <p className="scope-load-error-boundary">
          <a href="#/runs">Back to runs</a>
        </p>
      </section>
    );
  }

  return (
    <Observatory
      data={data}
      runtime={liveRuntime}
      inspectorOpen={inspectorOpen}
      runStatus={runStatus}
      onSelectRun={requestRunSelection}
      initialState={initialState ?? (tokenIndex == null ? undefined : { token: tokenIndex })}
      onStateChange={replaceScopeState}
    />
  );
}

const panel: StudioPanel = {
  id: "scope",
  navLabel: "Scope",
  order: 30,
  hiddenFromNav: true,
  icon: () => <Icon name="observatory" />,
  match: (hash) => {
    // `#/runs/<id>/scope?...` -- must be tried before lens's bare `#/runs/<id>`, which nav order
    // (30 < 20 is false, so lens IS tried first) does NOT guarantee. Lens's pattern is anchored to end
    // of string and therefore cannot match a `/scope` suffix, which is what actually keeps them apart.
    const deep = parseScopeUrl(hash);
    if (deep) return scopeRouteParams(deep);
    const withFixture = hash.match(/^#\/scope\/?\?fixture=([^&]+)$/);
    if (withFixture) return { fixture: decodeURIComponent(withFixture[1]) };
    return /^#\/scope\/?$/.test(hash) ? {} : null;
  },
  routeName: () => "MODEL SCOPE",
  Component: ScopePanel,
};

export default panel;
