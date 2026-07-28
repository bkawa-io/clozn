import { Icon } from "../components/Icon";
import { Model } from "../features/model/Model";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "model",
  navLabel: "Model",
  order: 60,
  icon: () => <Icon name="model" />,
  match: (hash) => (/^#\/model\/?$/.test(hash) ? {} : null),
  routeName: () => "MODEL",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>MODEL</b>{runtime.engine?.model ?? "—"}</span>
  ),
  modeChip: () => "READOUT",
  Component: ({ inspectorOpen }: PanelContext) => <Model inspectorOpen={inspectorOpen} />,
};

export default panel;
