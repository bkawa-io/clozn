import { Compare } from "../features/compare/Compare";
import type { PanelContext, StudioPanel } from "./types";

function ExperimentsIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" aria-hidden="true">
      <path d="M9 3v5.2L4.5 16a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L15 8.2V3" />
      <path d="M8.5 3h7M6.5 14h11" />
    </svg>
  );
}

const panel: StudioPanel = {
  id: "experiments",
  navLabel: "Experiments",
  order: 70,
  hiddenFromNav: true,
  icon: () => <ExperimentsIcon />,
  match: (hash) => {
    // Kept as a defensive compatibility owner for consumers that resolve this panel in isolation. The
    // Compare panel claims these routes first in the registry, so normal navigation stays on Compare.
    const withId = hash.match(/^#\/experiments\/([^/?]+)(?:\?(.*))?$/);
    if (withId) {
      const params: Record<string, string> = { id: decodeURIComponent(withId[1]) };
      if (withId[2] != null) params.q = withId[2];
      return params;
    }
    const bare = hash.match(/^#\/experiments\/?(?:\?(.*))?$/);
    if (bare) {
      const params: Record<string, string> = {};
      if (bare[1] != null) params.q = bare[1];
      return params;
    }
    return null;
  },
  routeName: () => "COMPARE MATRIX",
  modeChip: () => "MATRIX",
  Component: ({ runtime, inspectorOpen, params }: PanelContext) => (
    <Compare
      key={`legacy-matrix:${params.id ?? ""}:${params.q ?? ""}`}
      runtime={runtime}
      inspectorOpen={inspectorOpen}
      mode="matrix"
      initialExperimentId={params.id}
      rawExperimentQuery={params.q}
      experimentRouteBase="#/experiments"
    />
  ),
};

export default panel;
