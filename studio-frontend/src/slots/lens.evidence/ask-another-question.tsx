import type { SlotPanel } from "../../components/SlotHost";
import { AskAnotherQuestion } from "../../features/lens/AskAnotherQuestion";
import type { LensEvidenceData } from "./context-receipt";

/** C4 -- the investigation directory. Ordered FIRST (5, before context-receipt's 10) on purpose: this
 * panel is the entry point that routes to every other lens.evidence panel below it (context-receipt,
 * what-mattered, claim-verification, diagnosis-repair), so it reads before the evidence it points at,
 * not after. See `src/data/askAnotherQuestion.ts` for the capability audit behind each question and
 * `src/features/lens/AskAnotherQuestion.tsx` for why this renders nothing chat-shaped. */
const panel: SlotPanel<LensEvidenceData> = {
  id: "ask-another-question",
  slot: "lens.evidence",
  title: "Ask another question",
  order: 5,
  Component: ({ data }) => <AskAnotherQuestion runId={data.runId} />,
};

export default panel;
