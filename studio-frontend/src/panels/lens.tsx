import { Icon } from "../components/Icon";
import { RunReader, runReaderSection } from "../features/lens/RunReader";
import { parseScopeUrl, scopeRouteParams, SCOPE_VIEWS } from "../features/observatory/urlState";
import type { PanelContext, StudioPanel } from "./types";

const DIAGNOSTICS_SECTION: Record<string, string> = {
  overview: "read",
  investigate: "read",
  delivery: "what-received",
  rendered: "what-sent",
  context: "what-received",
  influence: "what-mattered",
  generation: "read",
  claims: "claims",
  runtime: "timing",
  events: "record",
  lineage: "record",
  timemachine: "time-machine",
  raw: "record",
};

function decode(value: string): string | null {
  try { return decodeURIComponent(value); } catch { return null; }
}

function nonnegativeInteger(value: string | null): string | undefined {
  return value != null && /^\d+$/.test(value) && Number.isSafeInteger(Number(value)) ? value : undefined;
}

/** Bridges the former top-level run surfaces into one canonical reader. The legacy paths are still
 * parsed exactly, but they now select an instrument inside the same run instead of recreating a menu. */
export function matchRunReaderRoute(hash: string): Record<string, string> | null {
  if (/^#\/lens\/?$/.test(hash)) return { section: "read" };

  const scope = parseScopeUrl(hash);
  if (scope) return { section: "mechanism", ...scopeRouteParams(scope) };

  const diagnostics = hash.match(/^#\/runs\/([^/?]+)\/diagnostics(?:\/([^/?]+))?\/?(?:\?[^#]*)?$/);
  if (diagnostics) {
    const runId = decode(diagnostics[1]);
    const view = diagnostics[2] ? decode(diagnostics[2]) : "overview";
    if (!runId || !view) return null;
    return { runId, section: DIAGNOSTICS_SECTION[view] ?? "read" };
  }

  const legacyLens = hash.match(/^#\/runs\/([^/?]+)\/lens\/?(?:\?[^#]*)?$/);
  if (legacyLens) {
    const runId = decode(legacyLens[1]);
    return runId ? { runId, section: "read" } : null;
  }

  const canonical = hash.match(/^#\/runs\/([^/?]+)\/?(?:\?(.*))?$/);
  if (!canonical) return null;
  const runId = decode(canonical[1]);
  if (!runId) return null;
  const query = new URLSearchParams(canonical[2] ?? "");
  const section = runReaderSection(query.get("section") ?? undefined);
  const params: Record<string, string> = { runId, section };
  const view = query.get("view");
  if (view && (SCOPE_VIEWS as readonly string[]).includes(view)) params.view = view;
  const token = nonnegativeInteger(query.get("token"));
  const layer = nonnegativeInteger(query.get("layer"));
  if (token) params.token = token;
  if (layer) params.layer = layer;
  const reference = query.get("reference");
  if (reference) params.reference = reference;
  return params;
}

const panel: StudioPanel = {
  id: "lens",
  navLabel: "Run",
  order: 20,
  showInspectorToggle: false,
  icon: () => <Icon name="lens" />,
  match: matchRunReaderRoute,
  routeName: () => "RUN",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>RECORDED</b>{runtime.runs.length}</span>
  ),
  modeChip: () => "READER",
  Component: ({ runtime, params }: PanelContext) => (
    <RunReader
      runtime={runtime}
      initialRunId={params.runId}
      initialSection={params.section}
      mechanismState={{
        view: params.view as "trace" | "variants" | "layers" | undefined,
        token: params.token == null ? undefined : Number(params.token),
        reference: params.reference,
        layer: params.layer == null ? undefined : Number(params.layer),
      }}
    />
  ),
};

export default panel;
