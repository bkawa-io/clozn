import { Icon } from "../components/Icon";
import { Lens } from "../features/lens/Lens";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "lens",
  navLabel: "Lens",
  order: 20,
  icon: () => <Icon name="lens" />,
  match: (hash): Record<string, string> | null => {
    if (/^#\/lens\/?$/.test(hash)) return {};
    // `#/runs/<id>` (no suffix) is the Lens deep link. Lens sorts BEFORE scope in nav order, so this is
    // tried first -- what keeps it from stealing `#/runs/<id>/scope` is the end-of-string anchor, not
    // the ordering. Loosening that anchor would silently break the Scope deep link.
    const deep = hash.match(/^#\/runs\/([^/]+)$/);
    return deep ? { runId: decodeURIComponent(deep[1]) } : null;
  },
  routeName: () => "LENS",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>RECORDED</b>{runtime.runs.length}</span>
  ),
  modeChip: () => "READOUT",
  Component: ({ runtime, inspectorOpen, params }: PanelContext) => (
    <Lens
      key={params.runId ?? "latest"}
      runtime={runtime}
      initialRunId={params.runId}
      inspectorOpen={inspectorOpen}
    />
  ),
};

export default panel;
