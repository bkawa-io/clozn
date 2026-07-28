import type { SlotPanel } from "../../components/SlotHost";
import { ContextReceipt } from "../../features/lens/ContextReceipt";

/** Feature 06's Studio surface: the compact + detailed context-receipt view, inside Lens's "context"
 * card. See docs/SURFACES.md's "Registered slots" table for the `lens.evidence` contract. */
export interface LensEvidenceData {
  runId: string;
}

const panel: SlotPanel<LensEvidenceData> = {
  id: "context-receipt",
  slot: "lens.evidence",
  title: "Context receipt",
  order: 10,
  Component: ({ data }) => <ContextReceipt runId={data.runId} />,
};

export default panel;
