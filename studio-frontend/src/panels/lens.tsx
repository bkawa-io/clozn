import { Icon } from "../components/Icon";
import { Lens } from "../features/lens/Lens";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "lens",
  navLabel: "Lens",
  order: 20,
  showInspectorToggle: false,
  icon: () => <Icon name="lens" />,
  match: (hash): Record<string, string> | null => {
    if (/^#\/lens\/?$/.test(hash)) return {};
    // Keep the original bare run route as a stable alias while making `/lens` the canonical reader URL.
    const deep = hash.match(/^#\/runs\/([^/]+)(?:\/lens)?\/?$/);
    return deep ? { runId: decodeURIComponent(deep[1]) } : null;
  },
  routeName: () => "LENS",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>RECORDED</b>{runtime.runs.length}</span>
  ),
  modeChip: () => "READER",
  Component: ({ runtime, params }: PanelContext) => (
    <Lens
      key={params.runId ?? "latest"}
      runtime={runtime}
      initialRunId={params.runId}
    />
  ),
};

export default panel;
