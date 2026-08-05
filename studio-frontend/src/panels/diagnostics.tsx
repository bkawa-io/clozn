import { Icon } from "../components/Icon";
import { RunDiagnostics } from "../features/diagnostics/RunDiagnostics";
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "diagnostics",
  navLabel: "Diagnostics",
  order: 25,
  // Run-specific diagnostics now select S2's Timing/Record instruments through lens.tsx. Keep this
  // panel only as an addressable compatibility fallback while the older URLs remain in circulation.
  hiddenFromNav: true,
  showInspectorToggle: false,
  icon: () => <Icon name="observatory" />,
  match: (hash): Record<string, string> | null => {
    const canonical = hash.match(/^#\/runs\/([^/]+)\/diagnostics(?:\/([^/?]+))?\/?$/);
    if (canonical) {
      return {
        runId: decodeURIComponent(canonical[1]),
        view: canonical[2] ? decodeURIComponent(canonical[2]) : "overview",
      };
    }

    // Compatibility routes lead into the read-only replacement instead of stranding old bookmarks.
    const oldScope = hash.match(/^#\/runs\/([^/]+)\/scope(?:\?[^#]*)?$/);
    if (oldScope) return { runId: decodeURIComponent(oldScope[1]), view: "generation" };

    // NO LONGER claims the session routes. It used to take both `#/investigation` and
    // `#/sessions/<id>/investigate`, which were written as retirements before F3 existed -- but F3
    // shipped panels/investigation.tsx with a real ConversationInvestigation and SessionPicker, and
    // this panel was silently winning both: registry.ts breaks an `order` tie with
    // `id.localeCompare(id)`, and "diagnostics" sorts before "investigation". A live surface was
    // unreachable because of a string comparison.
    //
    // The session case was the damaging one. Clicking a session handed this panel a `sessionId` and
    // no `runId`; RunDiagnostics then scanned only the LOADED page of runs for a matching sessionKey
    // and fell back to a default run when it missed. You asked for one session and got an unrelated
    // run's overview, with nothing on screen admitting the substitution (STUDIO_UI_AUDIT 3.6, 5).
    //
    // `#/scope` IS still claimed: scope.tsx is hidden from nav and its surface was folded in here, so
    // that redirect looks deliberate rather than accidental. Left alone rather than guessed at.
    if (/^#\/(?:diagnostics|scope)\/?$/.test(hash)) return { view: "overview" };
    return null;
  },
  routeName: () => "DIAGNOSTICS",
  modeChip: () => "READ ONLY",
  Component: ({ runtime, params }: PanelContext) => (
    <RunDiagnostics
      runtime={runtime}
      initialRunId={params.runId}
      initialView={params.view}
      sessionId={params.sessionId}
    />
  ),
};

export default panel;
