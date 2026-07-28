import { createContext, useContext, useEffect } from "react";
import type { DependencyList, ReactNode } from "react";

/**
 * Lets a panel publish topbar content derived from its OWN internal state.
 *
 * A panel's static `topStats`/`modeChip` fields cover the common case -- content derivable from the
 * shared context App already has. But the Scope surface's topbar shows the loaded run's id and model,
 * and its mode chip reflects fork/load status, all of which live inside that panel. Before this seam
 * those lived in `App.tsx` as a chain of `route.kind === "scope" && ...` conditionals, which is exactly
 * the coupling the seam exists to remove: App would have had to keep owning one panel's state forever.
 *
 * USAGE -- the factory + deps shape is deliberate:
 *
 *     useTopbar(() => ({ stats: <span>...</span>, modeChip: "LOADING" }), [runId, status]);
 *
 * JSX creates a new element object on every render, so a hook taking the nodes directly would set state
 * on every render and loop forever. Taking a factory and explicit deps puts the caller in control of
 * when the topbar actually changes, matching `useMemo`/`useEffect`'s own contract.
 *
 * Published content is cleared on unmount, so navigating away cannot leave a stale stat from a panel
 * that is no longer mounted.
 */
export interface TopbarContent {
  stats?: ReactNode;
  modeChip?: ReactNode;
}

export interface TopbarPublisher {
  publish(content: TopbarContent | null): void;
}

/** No-op default: a panel rendered outside App (a test, a fixture harness) must not crash. */
const TopbarContext = createContext<TopbarPublisher>({ publish: () => {} });

export const TopbarProvider = TopbarContext.Provider;

export function useTopbar(factory: () => TopbarContent, deps: DependencyList): void {
  const { publish } = useContext(TopbarContext);
  useEffect(() => {
    publish(factory());
    return () => publish(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- deps are the caller's contract; see above
  }, deps);
}
