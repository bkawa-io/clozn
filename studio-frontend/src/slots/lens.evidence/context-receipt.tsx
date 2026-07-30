import type { SlotPanel } from "../../components/SlotHost";
import { ReceivedContext } from "../../features/lens/ReceivedContext";

/** The primary delivery investigation panel. Exact prompt text remains in the existing ContextReceipt
 * disclosure, mounted by ReceivedContext only after an explicit user action. */
export interface LensEvidenceData {
  runId: string;
}

const panel: SlotPanel<LensEvidenceData> = {
  id: "context-receipt",
  slot: "lens.evidence",
  title: "What did the model receive?",
  order: 10,
  Component: ({ data }) => <ReceivedContext runId={data.runId} />,
};

export default panel;
