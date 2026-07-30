import type { SlotPanel } from "../../components/SlotHost";
import { WhatMattered } from "../../features/lens/WhatMattered";
import type { LensEvidenceData } from "./context-receipt";

/** The cross-linked "which prompt spans measurably moved which answer spans" panel. Ordered after
 * context-receipt (10): delivery evidence reads first, attribution evidence reads second -- "what
 * reached the model" before "what moved the answer". */
const panel: SlotPanel<LensEvidenceData> = {
  id: "what-mattered",
  slot: "lens.evidence",
  title: "What mattered?",
  order: 20,
  Component: ({ data }) => <WhatMattered runId={data.runId} />,
};

export default panel;
