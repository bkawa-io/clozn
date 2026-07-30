export const SCOPE_VIEWS = ["trace", "variants", "layers"] as const;

export type ScopeView = (typeof SCOPE_VIEWS)[number];

/** Values read from a Scope deep link. Missing fields mean "use the workbench default." */
export interface ScopeUrlState {
  view?: ScopeView;
  token?: number;
  reference?: string;
  layer?: number;
}

/** The fully resolved selection Observatory emits after run-specific bounds are known. */
export interface ScopeSelectionState {
  view: ScopeView;
  token: number;
  reference?: string;
  layer: number;
}

export interface ScopeRoute {
  runId: string;
  state: ScopeUrlState;
}

function isScopeView(value: string | null | undefined): value is ScopeView {
  return value != null && (SCOPE_VIEWS as readonly string[]).includes(value);
}

function nonnegativeInteger(value: string | null | undefined): number | undefined {
  if (value == null || !/^\d+$/.test(value)) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

/** Parse any ordering/subset of the canonical query. Unknown and malformed values fail to absence. */
export function parseScopeUrl(hash: string): ScopeRoute | null {
  const match = hash.match(/^#\/runs\/([^/?]+)\/scope(?:\?(.*))?$/);
  if (!match) return null;

  let runId: string;
  try {
    runId = decodeURIComponent(match[1]);
  } catch {
    return null;
  }
  if (!runId) return null;

  const query = new URLSearchParams(match[2] ?? "");
  const rawReference = query.get("reference");
  return {
    runId,
    state: {
      view: isScopeView(query.get("view")) ? query.get("view") as ScopeView : undefined,
      token: nonnegativeInteger(query.get("token")),
      reference: rawReference ? rawReference : undefined,
      layer: nonnegativeInteger(query.get("layer")),
    },
  };
}

/** Bridge the pure route shape into StudioPanel's string-only params contract. */
export function scopeRouteParams(route: ScopeRoute): Record<string, string> {
  const params: Record<string, string> = { runId: route.runId };
  if (route.state.view != null) params.view = route.state.view;
  if (route.state.token != null) {
    params.token = String(route.state.token);
    // Keep the pre-workbench panel-param name for callers/tests that consume match() directly.
    params.tokenIndex = String(route.state.token);
  }
  if (route.state.reference != null) params.reference = route.state.reference;
  if (route.state.layer != null) params.layer = String(route.state.layer);
  return params;
}

/** Re-read already validated panel params without inventing values for absent/invalid fields. */
export function scopeStateFromParams(params: Record<string, string>): ScopeUrlState {
  return {
    view: isScopeView(params.view) ? params.view : undefined,
    token: nonnegativeInteger(params.token),
    reference: params.reference || undefined,
    layer: nonnegativeInteger(params.layer),
  };
}

/** Emit one stable ordering so equivalent selections always produce the same shareable hash. */
export function serializeScopeUrl(runId: string, state: ScopeSelectionState): string {
  const query = new URLSearchParams();
  query.set("view", state.view);
  query.set("token", String(canonicalIndex(state.token)));
  if (state.reference) query.set("reference", state.reference);
  query.set("layer", String(canonicalIndex(state.layer)));
  return `#/runs/${encodeURIComponent(runId)}/scope?${query.toString()}`;
}

function canonicalIndex(value: number): number {
  return Number.isSafeInteger(value) && value >= 0 ? value : 0;
}
