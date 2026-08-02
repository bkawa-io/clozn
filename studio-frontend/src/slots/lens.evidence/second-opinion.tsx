import type { SlotPanel } from "../../components/SlotHost";
import { SecondOpinion } from "../../features/lens/SecondOpinion";
import type { LensEvidenceData } from "./context-receipt";

/** E4 -- explicit cross-model answer comparison. Candidate discovery is read-only; generation only
 * begins after the user chooses a resident model and presses ASK SECOND OPINION. The panel sits after
 * claim verification so the anchor's own evidence is available before a second answer is compared. */
const panel: SlotPanel<LensEvidenceData> = {
  id: "second-opinion",
  slot: "lens.evidence",
  title: "Would another model disagree?",
  order: 28,
  Component: ({ data }) => <SecondOpinion runId={data.runId} />,
};

export default panel;
