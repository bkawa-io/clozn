import type {
  ForkIntervention,
  ForkOutcome,
  ForkOutcomeKind,
  ForkUnchangedControl,
} from "../../data/types";

/**
 * Renders FORK-02's `outcome` (clozn.replay.fork.compat_fork / docs/EXECUTION_FORK_CONTRACT.md) as
 * visible, distinguishable copy -- never a color or icon alone (see docs/STUDIO_NEXT_HANDOFF_2026-07-26.md's
 * copy rules). `exact_execution_fork` reads as the strong result; `reconstructed_replay` is visibly
 * weaker and says outright that BPE re-tokenization means the continuation is not guaranteed to run on
 * the exact recorded token ids; `unavailable` shows the gateway's own typed reason instead of a generic
 * failure sentence.
 *
 * Every switch below ends in a `never` assignment (the same convention as EvidenceCaveat.tsx): a value
 * this module hasn't been taught about fails the TypeScript build instead of silently falling through to
 * whatever branch happens to run last.
 */

function humanize(value?: string): string | undefined {
  if (!value) return undefined;
  return value.replace(/_/g, " ").toUpperCase();
}

export function forkOutcomeBadge(kind: ForkOutcomeKind): { label: string; className: string } {
  switch (kind) {
    case "exact_execution_fork":
      return { label: "EXACT EXECUTION FORK", className: "is-fork-exact" };
    case "reconstructed_replay":
      return { label: "RECONSTRUCTED REPLAY", className: "is-fork-reconstructed" };
    case "unavailable":
      return { label: "FORK UNAVAILABLE", className: "is-fork-unavailable" };
    default: {
      const exhaustive: never = kind;
      return exhaustive;
    }
  }
}

function forkOutcomeSummary(outcome: ForkOutcome): string {
  switch (outcome.kind) {
    case "exact_execution_fork":
      return "The worker restored its exact recorded KV state and applied the forced token directly "
        + "on its token id -- no text splice, nothing to retokenize.";
    case "reconstructed_replay":
      return "Legacy text splice: the kept prefix and the forced token were concatenated as text and "
        + "re-tokenized by the engine. BPE token boundaries can shift at that junction, so this "
        + "continuation is NOT guaranteed to run on the exact recorded token ids.";
    case "unavailable":
      return "Neither the exact execution-fork path nor the reconstructed text splice could run "
        + "honestly for this request.";
    default: {
      const exhaustive: never = outcome;
      return exhaustive;
    }
  }
}

function interventionText(intervention?: ForkIntervention): string {
  if (!intervention) return "—";
  const kind = humanize(intervention.type) ?? intervention.type.toUpperCase();
  if (intervention.tokenPiece && intervention.tokenId != null) {
    return `${kind} → "${intervention.tokenPiece}" (id ${intervention.tokenId})`;
  }
  if (intervention.tokenPiece) return `${kind} → "${intervention.tokenPiece}"`;
  if (intervention.tokenId != null) return `${kind} → id ${intervention.tokenId}`;
  return kind;
}

function controlText(control?: ForkUnchangedControl): string {
  if (!control) return "—";
  const status = humanize(control.status) ?? control.status.toUpperCase();
  const match = control.result?.exactMatch;
  return match == null ? status : `${status} · ${match ? "EXACT MATCH" : "DIVERGED"}`;
}

export function ForkOutcomePanel({ outcome, note }: { outcome: ForkOutcome; note?: string }) {
  const badge = forkOutcomeBadge(outcome.kind);
  return (
    <div className={`fork-outcome ${badge.className}`} role="status">
      <header>
        <span className="fork-outcome-badge">{badge.label}</span>
        {outcome.kind === "reconstructed_replay" && (
          <span className="fork-outcome-flag">
            {outcome.retokenized ? "RETOKENIZED" : "PREFIX VERIFIED TOKEN-EXACT"}
          </span>
        )}
      </header>
      <p className="fork-outcome-summary">{forkOutcomeSummary(outcome)}</p>
      {note && <p className="fork-outcome-note">{note}</p>}

      {outcome.kind === "exact_execution_fork" && (
        <dl className="metric-list fork-outcome-facts">
          <div><dt>Truncation regime</dt><dd>{humanize(outcome.exactness.regime) ?? "—"}</dd></div>
          <div><dt>Restore mode</dt><dd>{humanize(outcome.intervention?.restoreMode) ?? "—"}</dd></div>
          <div><dt>Response boundary</dt><dd>{outcome.exactness.truncateTo ?? "—"}</dd></div>
          <div><dt>Intervention applied</dt><dd>{interventionText(outcome.intervention)}</dd></div>
          <div><dt>Unchanged control</dt><dd>{controlText(outcome.unchangedControl)}</dd></div>
          <div><dt>Proof status</dt><dd>{humanize(outcome.exactness.proofStatus) ?? "—"}</dd></div>
        </dl>
      )}

      {outcome.kind === "reconstructed_replay" && outcome.unavoidableDifferences.length > 0 && (
        <div className="fork-outcome-differences">
          <span>NEVER REPRODUCED BY THIS PATH</span>
          <ul>
            {outcome.unavoidableDifferences.map((difference) => (
              <li key={difference}>{humanize(difference) ?? difference}</li>
            ))}
          </ul>
        </div>
      )}

      {outcome.kind === "unavailable" && (outcome.exactness?.regime || outcome.unchangedControl) && (
        <dl className="metric-list fork-outcome-facts">
          {outcome.exactness?.regime && (
            <div><dt>Attempted regime</dt><dd>{humanize(outcome.exactness.regime)}</dd></div>
          )}
          {outcome.unchangedControl && (
            <div><dt>Unchanged control</dt><dd>{controlText(outcome.unchangedControl)}</dd></div>
          )}
        </dl>
      )}

      {outcome.reasons.length > 0 && (
        <ul className="fork-outcome-reasons">
          {outcome.reasons.map((reason, index) => (
            <li key={`${reason.code}-${index}`}>
              <b>{humanize(reason.code) ?? reason.code.toUpperCase()}</b>
              {reason.message && <span>{reason.message}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
