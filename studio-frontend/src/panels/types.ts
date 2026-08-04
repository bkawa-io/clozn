import type { ComponentType, ReactNode } from "react";
import type { RuntimeState } from "../data/types";

/**
 * A top-level Studio surface.
 *
 * Adding one is adding a file under `src/panels/`. Nothing else is edited -- `App.tsx` discovers
 * panels with `import.meta.glob` and builds the nav rail, the hash router, and the workspace from
 * whatever it finds. See `docs/SURFACES.md`.
 *
 * The interface is shaped by what `App.tsx` actually varied per route before this seam existed: a nav
 * entry, a hash pattern, a route name in the header, an optional stat in the topbar, and the workspace
 * component. A panel that needs to publish topbar content derived from its OWN internal state uses
 * `useTopbar` (see `topbar.tsx`) instead of the static `topStats`/`modeChip` fields, because App cannot
 * see inside a panel's state and should not try to.
 */
export interface PanelContext {
  /** Gateway/runtime status and the recorded-run list. Shared by every panel. */
  runtime: RuntimeState;
  /** Whether the right-hand inspector is expanded. Panels lay out around this. */
  inspectorOpen: boolean;
  /** Whatever this panel's own `match()` pulled out of the hash. */
  params: Record<string, string>;
}

export interface StudioPanel {
  /** Stable id. Also the `workspace is-<id>` CSS hook and the nav's active-state key. */
  id: string;
  /** Rail label. */
  navLabel: string;
  /** Nav rail position; ties broken by id. Existing surfaces are 10..60. */
  order?: number;
  /** Keep a compatibility route available without advertising it as a primary destination. */
  hiddenFromNav?: boolean;
  /** Some focused workspaces own their own drawers and do not use the shell inspector. */
  showInspectorToggle?: boolean;
  /**
   * The rail icon. A function rather than a name so a new panel can bring its own inline SVG without
   * widening `Icon.tsx`'s closed union -- the six original surfaces just return `<Icon name="..." />`.
   */
  icon: () => ReactNode;
  /**
   * Parse `location.hash`. Return the params this route carries, or `null` when the hash is not this
   * panel's. `{}` means "matched, no params" -- which is NOT the same as `null`, so return the empty
   * object rather than a falsy value on a bare match.
   *
   * Panels are tried in `order`, so a panel matching a prefix another panel also claims must sort
   * after it. `App.tsx` falls back to the lowest-ordered panel when nothing matches.
   */
  match(hash: string): Record<string, string> | null;
  /** Header route-name text, e.g. "MODEL SCOPE". */
  routeName(params: Record<string, string>): string;
  /** Optional topbar stat, for values derivable from context alone. */
  topStats?(ctx: PanelContext): ReactNode;
  /** Optional mode chip. Panels whose chip depends on internal state use `useTopbar` instead. */
  modeChip?(ctx: PanelContext): ReactNode;
  /** The workspace body. */
  Component: ComponentType<PanelContext>;
}
