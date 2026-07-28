import { Icon } from "../components/Icon";
import { Behavior } from "../features/behavior/Behavior";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "behavior",
  navLabel: "Behavior",
  order: 50,
  icon: () => <Icon name="behavior" />,
  match: (hash) => (/^#\/behavior\/?$/.test(hash) ? {} : null),
  routeName: () => "BEHAVIOR",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>MODEL</b>{runtime.engine?.model ?? "—"}</span>
  ),
  modeChip: () => "READ / WRITE",
  Component: ({ runtime, inspectorOpen }: PanelContext) => (
    <Behavior runtime={runtime} inspectorOpen={inspectorOpen} />
  ),
};

export default panel;
