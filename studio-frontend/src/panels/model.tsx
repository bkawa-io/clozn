import { Icon } from "../components/Icon";
import { Model } from "../features/model/Model";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "model",
  navLabel: "Runtime",
  order: 60,
  icon: () => <Icon name="model" />,
  // The shell derives its rail URL from this file-backed id, so `#/model` remains a supported bookmark.
  // Recognizing `#/runtime` gives new deep links the product name without changing shared route startup.
  match: (hash) => (/^#\/(?:runtime|model)\/?$/.test(hash) ? {} : null),
  routeName: () => "RUNTIME",
  topStats: ({ runtime }: PanelContext) => (
    <span className="top-stat"><b>ENGINE</b>{runtime.engine?.model ?? "—"}</span>
  ),
  modeChip: () => "INSTALLATION",
  Component: ({ inspectorOpen }: PanelContext) => <Model inspectorOpen={inspectorOpen} />,
};

export default panel;
