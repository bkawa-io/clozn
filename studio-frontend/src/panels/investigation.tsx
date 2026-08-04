import { Icon } from "../components/Icon";
import { ConversationInvestigation } from "../features/investigation/ConversationInvestigation";
import { SessionPicker } from "../features/investigation/SessionPicker";
import "../styles/investigation.css";
import type { PanelContext, StudioPanel } from "./types";

/**
 * F3 -- the conversation investigation view. `#/investigation` and `#/sessions` (bare) are the session
 * picker (F1's own `GET /sessions` list plus an "open by id" router); `#/sessions/<id>/investigate` is the
 * investigation view itself. Deliberately its own top-level nav entry, not a tab bolted onto Runs or Lens -- the F3
 * brief's own acceptance criterion ("investigation vs chat visually separate") reads as a surface-level
 * requirement, and C4's `AskAnotherQuestion.tsx` already established the precedent of a dedicated,
 * non-chat-shaped panel for evidence review rather than folding it into an existing one.
 */
const panel: StudioPanel = {
  id: "investigation",
  navLabel: "Investigate",
  order: 25,
  hiddenFromNav: true,
  icon: () => <Icon name="investigation" />,
  match: (hash): Record<string, string> | null => {
    const deep = hash.match(/^#\/sessions\/([^/]+)\/investigate\/?$/);
    if (deep) return { sessionId: decodeURIComponent(deep[1]) };
    // `#/investigation` is the canonical rail destination (the panel id is part of the public shell
    // contract). Keep the older `#/sessions` form as a stable deep-link alias because existing session
    // links and bookmarks use it.
    return /^#\/(?:investigation|sessions)\/?$/.test(hash) ? {} : null;
  },
  routeName: (params) => (params.sessionId ? "SESSION INVESTIGATION" : "SESSIONS"),
  Component: ({ params }: PanelContext) => (
    params.sessionId
      ? <ConversationInvestigation key={params.sessionId} sessionId={params.sessionId} />
      : <SessionPicker />
  ),
};

export default panel;
