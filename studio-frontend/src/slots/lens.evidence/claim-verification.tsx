import type { SlotPanel } from "../../components/SlotHost";
import { ClaimVerification } from "../../features/lens/ClaimVerification";
import type { LensEvidenceData } from "./context-receipt";

/** E3 -- inline claim markers over the answer text, each carrying E1/E2's deterministic verification
 * status. Ordered after what-mattered (20), before diagnosis-repair (30): delivery evidence, then
 * attribution evidence, then "are the individual claims supported?", then "why, and what to try" --
 * each register builds on the one above it. */
const panel: SlotPanel<LensEvidenceData> = {
  id: "claim-verification",
  slot: "lens.evidence",
  title: "Are the claims supported?",
  order: 25,
  Component: ({ data }) => <ClaimVerification runId={data.runId} />,
};

export default panel;
