import { Icon } from "../components/Icon";
import { Compare } from "../features/compare/Compare";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "compare",
  navLabel: "Compare",
  order: 40,
  icon: () => <Icon name="compare" />,
  match: (hash) => {
    const m = hash.match(/^#\/compare(?:\/([^/]+)\/([^/]+))?$/);
    if (!m) return null;
    const params: Record<string, string> = {};
    if (m[1]) params.runA = decodeURIComponent(m[1]);
    if (m[2]) params.runB = decodeURIComponent(m[2]);
    return params;
  },
  routeName: () => "COMPARE",
  modeChip: () => "A / B",
  Component: ({ runtime, inspectorOpen, params }: PanelContext) => (
    <Compare
      key={`${params.runA ?? ""}:${params.runB ?? ""}`}
      runtime={runtime}
      initialA={params.runA}
      initialB={params.runB}
      inspectorOpen={inspectorOpen}
    />
  ),
};

export default panel;
