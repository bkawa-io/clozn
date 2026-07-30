import type { SlotPanel } from "../../components/SlotHost";
import { DiagnosisRepair } from "../../features/lens/DiagnosisRepair";
import type { LensEvidenceData } from "./context-receipt";

/** D5 -- the guided corrective-retry panel: D1/D2's rule findings + narrative, plus D3's real
 * preview -> confirm -> keep corrective-retry mechanics, for the currently-selected Lens run. Ordered
 * after context-receipt (10) and what-mattered (20): delivery evidence, then attribution evidence, then
 * "why, and what to try" -- each register builds on what the reader has already seen above it. */
const panel: SlotPanel<LensEvidenceData> = {
  id: "diagnosis-repair",
  slot: "lens.evidence",
  title: "Why, and what to try",
  order: 30,
  Component: ({ data }) => <DiagnosisRepair runId={data.runId} />,
};

export default panel;
