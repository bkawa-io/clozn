import type { StudioPanel } from "./types";

/**
 * Filesystem discovery for top-level Studio surfaces -- the frontend counterpart to
 * `clozn/cli/commands/_autoload.py` and `clozn/server/routes/_autoload.py`, and deliberately the same
 * shape: discover by walking, opt in by an explicit export, record failures rather than swallowing them.
 *
 * `import.meta.glob` is Vite's own build-time mechanism, so this adds no dependency and the panel set is
 * resolved at bundle time rather than by runtime directory reads (which a browser cannot do anyway).
 *
 * WHY FAILURES ARE VALUES, NOT THROWS
 * -----------------------------------
 * A panel whose module fails to load, or whose default export is not shaped like a panel, is collected
 * into `loadFailures` and rendered as a visible placeholder in the nav rail. It does not take down the
 * Studio -- one malformed surface should cost its own tab, not every other tab plus the runtime status
 * the user needs in order to diagnose it. But it is never silent: the rail shows it, and
 * `panelRegistry.loadFailures` is asserted empty by the panel-registry test, which is what turns a
 * broken panel into a build-time failure rather than a UI oddity someone learns to ignore.
 */
export interface PanelLoadFailure {
  /** The module path that failed, e.g. "../panels/experiments.tsx". */
  path: string;
  /** Best-effort id, parsed from the filename when the module could not supply one. */
  id: string;
  reason: string;
}

export interface PanelRegistry {
  panels: StudioPanel[];
  loadFailures: PanelLoadFailure[];
}

function idFromPath(path: string): string {
  const file = path.split("/").pop() ?? path;
  return file.replace(/\.tsx?$/, "");
}

function validate(candidate: unknown, path: string): string | null {
  if (candidate == null || typeof candidate !== "object") {
    return "module has no default export, or it is not an object";
  }
  const panel = candidate as Partial<StudioPanel>;
  for (const field of ["id", "navLabel"] as const) {
    if (typeof panel[field] !== "string" || !panel[field]) return `missing or empty "${field}"`;
  }
  for (const field of ["match", "routeName", "icon"] as const) {
    if (typeof panel[field] !== "function") return `missing "${field}()"`;
  }
  if (panel.Component == null) return 'missing "Component"';
  if (!path.endsWith(`/${panel.id}.tsx`)) {
    // The filename IS the id, the same 1:1 rule the schema seam uses for `schema_version`. Without it
    // a deep link and a nav entry can disagree about which panel owns a route, which is a genuinely
    // confusing bug to chase.
    return `id "${panel.id}" does not match its filename (${idFromPath(path)}.tsx)`;
  }
  return null;
}

export function buildRegistry(modules: Record<string, unknown>): PanelRegistry {
  const panels: StudioPanel[] = [];
  const loadFailures: PanelLoadFailure[] = [];

  for (const path of Object.keys(modules).sort()) {
    // Infrastructure, not a surface. Same convention as clozn/cli/commands/_autoload.py's own scan.
    // (`types.ts` and `registry.ts` escape the glob by being .ts; this covers anything that genuinely
    // needs JSX and still is not a panel.)
    if (idFromPath(path).startsWith("_")) continue;
    const mod = modules[path] as { default?: unknown } | undefined;
    const candidate = mod?.default;
    const problem = validate(candidate, path);
    if (problem) {
      loadFailures.push({ path, id: idFromPath(path), reason: problem });
      continue;
    }
    panels.push(candidate as StudioPanel);
  }

  const seen = new Map<string, string>();
  const unique: StudioPanel[] = [];
  for (const panel of panels) {
    const previous = seen.get(panel.id);
    if (previous) {
      // Two panels claiming one id would make the nav's active state and the router disagree.
      loadFailures.push({ path: panel.id, id: panel.id, reason: `duplicate id, already defined` });
      continue;
    }
    seen.set(panel.id, panel.id);
    unique.push(panel);
  }

  unique.sort((a, b) => (a.order ?? 100) - (b.order ?? 100) || a.id.localeCompare(b.id));
  return { panels: unique, loadFailures };
}

/**
 * Resolve a hash to a panel. Panels are tried in nav order; the first non-null `match()` wins, and a
 * panel whose `match()` throws is skipped rather than being allowed to break routing for every other
 * surface. Falls back to the first panel so an unknown hash lands somewhere real instead of blank.
 */
export function resolveRoute(
  panels: StudioPanel[],
  hash: string,
): { panel: StudioPanel; params: Record<string, string> } | null {
  for (const panel of panels) {
    let params: Record<string, string> | null = null;
    try {
      params = panel.match(hash);
    } catch {
      continue;
    }
    if (params) return { panel, params };
  }
  return panels.length ? { panel: panels[0], params: {} } : null;
}

// `eager: true` means Vite statically imports every match INTO the production bundle at build time --
// unlike the runtime `validate()` above, a negative glob is the only thing that can stop a co-located
// `*.test.tsx` file (e.g. scope.test.tsx next to scope.tsx) from being bundled and executed alongside
// its panel: by the time `validate()` would reject it as "not shaped like a panel", vitest/testing-
// library have already been pulled into the shipped app. Discovered by FORK-02's ScopePanel test:
// without this exclusion, `vite build` silently doubled the bundle and emitted a stray test-tooling
// chunk (magic-string) that no product code ever imports.
const modules = import.meta.glob(["./*.tsx", "!./*.test.tsx"], { eager: true });
export const panelRegistry: PanelRegistry = buildRegistry(modules);
