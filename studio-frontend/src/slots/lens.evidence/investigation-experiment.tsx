import type { SlotPanel } from "../../components/SlotHost";
import { InvestigationExperiment } from "../../features/lens/InvestigationExperiment";
import type { LensEvidenceData } from "./context-receipt";

/** C3 -- explicit "did this matter?" plan/execute surface over stable prompt/source addresses. */
const panel: SlotPanel<LensEvidenceData> = {
  id: "investigation-experiment",
  slot: "lens.evidence",
  title: "Did this matter?",
  order: 22,
  Component: ({ data }) => <InvestigationExperiment runId={data.runId} />,
};

export default panel;
