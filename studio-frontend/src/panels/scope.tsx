import { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "../components/Icon";
import { createFork, loadRunInspection, loadRuntimeState } from "../data/api";
import { DEMO_OBSERVATORY } from "../data/demo";
import { stressFixture } from "../data/stress";
import type { ForkState, ObservatoryData } from "../data/types";
import { Observatory } from "../features/observatory/Observatory";
import {
  parseScopeUrl,
  scopeRouteParams,
  scopeStateFromParams,
  serializeScopeUrl,
  type ScopeSelectionState,
} from "../features/observatory/urlState";
import { useTopbar } from "./topbar";
import type { PanelContext, StudioPanel } from "./types";

/**
 * The one panel that owns real state. `data`, `runStatus`, `forkState`, `selectRun` and `forkRun` all
 * lived in `App.tsx` before this seam, alongside a topbar that reached into them through a chain of
 * `route.kind === "scope" && ...` conditionals. That is precisely the coupling the seam removes: App
 * would otherwise have had to keep owning one surface's state forever, and every future surface would
 * have been tempted to add its own branch beside it.
 *
 * The logic below is moved verbatim, not rewritten. It reports its topbar content through `useTopbar`
 * because App cannot see inside a panel, and should not.
 */
export function ScopePanel({ runtime, inspectorOpen, params }: PanelContext) {
  const [data, setData] = useState<ObservatoryData>(DEMO_OBSERVATORY);
  const [runStatus, setRunStatus] = useState<"idle" | "loading" | "error">("idle");
  const [forkState, setForkState] = useState<ForkState>({ status: "idle" });
  const [liveRuntime, setLiveRuntime] = useState(runtime);
  // One monotonic counter shared by selectRun and forkRun: whichever call is issued LAST wins. A fork
  // response for the run displayed when it was requested must never land after the panel has since
  // navigated elsewhere -- each async function captures its own id and checks it against this ref
  // before committing state, so a stale response for run A can never paint over run B.
  const requestIdRef = useRef(0);

  useEffect(() => setLiveRuntime(runtime), [runtime]);

  const runId = params.runId;
  const fixture = params.fixture;
  const routeState = scopeStateFromParams(params);

  useEffect(() => {
    if (runId) void selectRun(runId);
    else if (fixture) {
      setData(stressFixture(fixture) ?? DEMO_OBSERVATORY);
      setRunStatus("idle");
      setForkState({ status: "idle" });
    } else {
      setData(DEMO_OBSERVATORY);
      setRunStatus("idle");
      setForkState({ status: "idle" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- matches App.tsx's original dep list
  }, [runId, fixture]);

  async function selectRun(nextRunId: string) {
    const requestId = ++requestIdRef.current;
    setForkState({ status: "idle" });
    if (!nextRunId) {
      setData(DEMO_OBSERVATORY);
      setRunStatus("idle");
      history.replaceState(null, "", "#/scope");
      return;
    }
    setRunStatus("loading");
    try {
      const inspection = await loadRunInspection(nextRunId);
      if (requestIdRef.current !== requestId) return;   // superseded by a later selection or fork
      setData(inspection);
      setRunStatus("idle");
      history.replaceState(null, "", `#/runs/${encodeURIComponent(nextRunId)}/scope`);
    } catch {
      if (requestIdRef.current !== requestId) return;
      setRunStatus("error");
    }
  }

  async function forkRun(position: number, token: string, tokenId?: number) {
    if (data.mode !== "run") return;
    const parentId = data.id;
    const requestId = ++requestIdRef.current;
    setForkState({ status: "loading", parentId });
    try {
      const result = await createFork(parentId, position, token, tokenId);
      if (requestIdRef.current !== requestId) return;   // the panel has since moved to a different run
      if (!result.child) {
        // `unavailable` (422): a typed, non-actionable outcome, not a request failure -- no child was
        // created, so there is nothing to load or navigate to.
        setForkState({ status: "unavailable", parentId, outcome: result.outcome });
        return;
      }
      const child = result.child;
      const [inspection, nextRuntime] = await Promise.all([
        loadRunInspection(child.id),
        loadRuntimeState(),
      ]);
      if (requestIdRef.current !== requestId) return;   // superseded again while these were in flight
      setData(inspection);
      setLiveRuntime(nextRuntime);
      setRunStatus("idle");
      setForkState({
        status: "success",
        parentId: child.parentId,
        childId: child.id,
        note: child.note,
        outcome: result.outcome,
      });
      history.replaceState(null, "", `#/runs/${encodeURIComponent(child.id)}/scope`);
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      setForkState({
        status: "error",
        parentId,
        message: error instanceof Error ? error.message : "Fork failed",
      });
    }
  }

  const modelLabel = data.model === "local" ? liveRuntime.engine?.model ?? data.model : data.model;

  useTopbar(
    () => ({
      stats: (
        <>
          <span className="top-stat"><b>MODEL</b>{modelLabel}</span>
          <span className="top-stat"><b>RUN</b>{data.id}</span>
        </>
      ),
      modeChip:
        forkState.status === "loading"
          ? "FORKING"
          : runStatus === "loading"
            ? "LOADING"
            : runStatus === "error" || forkState.status === "error"
              ? "ERROR"
              : forkState.status === "unavailable"
                ? "FORK UNAVAILABLE"
                : data.mode.toUpperCase(),
    }),
    [modelLabel, data.id, data.mode, runStatus, forkState.status],
  );

  const tokenIndex = params.tokenIndex == null ? undefined : Number(params.tokenIndex);
  const initialState = data.mode === "run" && data.id === runId
    ? routeState
    : undefined;
  const replaceScopeState = useCallback((state: ScopeSelectionState) => {
    if (data.mode !== "run" || runStatus !== "idle") return;
    history.replaceState(null, "", serializeScopeUrl(data.id, state));
  }, [data.id, data.mode, runStatus]);

  return (
    <Observatory
      data={data}
      runtime={liveRuntime}
      inspectorOpen={inspectorOpen}
      runStatus={runStatus}
      forkState={forkState}
      onSelectRun={(nextRunId) => void selectRun(nextRunId)}
      onFork={(position, token, tokenId) => void forkRun(position, token, tokenId)}
      initialState={initialState ?? (tokenIndex == null ? undefined : { token: tokenIndex })}
      onStateChange={replaceScopeState}
    />
  );
}

const panel: StudioPanel = {
  id: "scope",
  navLabel: "Scope",
  order: 30,
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
