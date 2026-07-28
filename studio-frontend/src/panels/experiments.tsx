import { Experiments } from "../features/experiments/Experiments";
import type { PanelContext, StudioPanel } from "./types";
// Self-imported rather than added to `src/main.tsx`'s hand-listed stylesheet chain: `main.tsx` is a
// shared file (docs/SURFACES.md notes this is avoidable today), and several other panels are landing in
// this same window, so a panel that needs its own CSS pulls it in directly instead of contending for an
// edit to a file it does not own.
import "../styles/experiments.css";

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
  icon: () => <ExperimentsIcon />,
  match: (hash) => {
    // `#/experiments/<id>[?...]` -- tried first, same reasoning as scope.tsx's deep-link pattern: an
    // end-anchored regex without an id group would happily swallow this route too if tried first.
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
  routeName: (params) => (params.id ? "EXPERIMENT" : "EXPERIMENTS"),
  Component: ({ params }: PanelContext) => <Experiments id={params.id} rawQuery={params.q} />,
};

export default panel;
