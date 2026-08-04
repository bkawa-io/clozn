import { Icon } from "../components/Icon";
import { RunDiagnostics } from "../features/diagnostics/RunDiagnostics";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "diagnostics",
  navLabel: "Diagnostics",
  order: 25,
  showInspectorToggle: false,
  icon: () => <Icon name="observatory" />,
  match: (hash): Record<string, string> | null => {
    const canonical = hash.match(/^#\/runs\/([^/]+)\/diagnostics(?:\/([^/?]+))?\/?$/);
    if (canonical) {
      return {
        runId: decodeURIComponent(canonical[1]),
        view: canonical[2] ? decodeURIComponent(canonical[2]) : "overview",
      };
    }

    // Compatibility routes lead into the read-only replacement instead of stranding old bookmarks.
    const oldScope = hash.match(/^#\/runs\/([^/]+)\/scope(?:\?[^#]*)?$/);
    if (oldScope) return { runId: decodeURIComponent(oldScope[1]), view: "generation" };
    const oldInvestigation = hash.match(/^#\/sessions\/([^/]+)\/investigate\/?$/);
    if (oldInvestigation) return { sessionId: decodeURIComponent(oldInvestigation[1]), view: "overview" };
    if (/^#\/(?:diagnostics|investigation|scope)\/?$/.test(hash)) return { view: "overview" };
    return null;
  },
  routeName: () => "DIAGNOSTICS",
  modeChip: () => "READ ONLY",
  Component: ({ runtime, params }: PanelContext) => (
    <RunDiagnostics
      runtime={runtime}
      initialRunId={params.runId}
      initialView={params.view}
      sessionId={params.sessionId}
    />
  ),
};

export default panel;
