import { Icon } from "../components/Icon";
import { Runs } from "../features/runs/Runs";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "runs",
  navLabel: "Runs",
  order: 10,
  icon: () => <Icon name="runs" />,
  // Also the fallback surface: registry.resolveRoute() lands on the first panel in nav order when no
  // panel claims the hash, which preserves App's original "default to runs" behavior.
  match: (hash) => (/^#\/runs\/?$/.test(hash) || hash === "" || hash === "#" ? {} : null),
  routeName: () => "RUNS",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>RECORDED</b>{runtime.runs.length}</span>
  ),
  modeChip: () => "LEDGER",
  Component: ({ runtime, inspectorOpen }: PanelContext) => (
    <Runs runtime={runtime} inspectorOpen={inspectorOpen} />
  ),
};

export default panel;
