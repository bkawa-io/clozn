import type { CandidateReading } from "../../data/types";
import type { WorkbenchTokenSection } from "../../data/tokenWorkbench";

interface DistributionRow {
  token: string;
  score: number;
  tokenId?: number;
  isRecorded: boolean;
}

function formatPercent(value: number) {
  return `${Math.round(Math.max(0, value) * 100)}%`;
}

/** From the workbench `token` section, when loaded -- the same recorded alternatives previously read off
 * the full run's `ObservatoryData.candidates`, now sourced from the pure per-token projection instead
 * (compose the backend's own token section, don't keep a parallel client copy). */
function rowsFromWorkbench(token: WorkbenchTokenSection): DistributionRow[] {
  const rows = token.alternatives.map((alt) => ({
    token: alt.piece,
    score: alt.prob ?? 0,
    tokenId: alt.tokenId,
    isRecorded: alt.piece === token.piece,
  }));
  if (rows.some((row) => row.isRecorded)) return rows;
  // The recorded piece is not among the recorded alternatives on this run (older traces sometimes
  // stored only the runner-up candidates) -- still show it, first, so "what was actually produced"
  // is never missing from its own distribution card.
  return [{ token: token.piece, score: 1, tokenId: token.tokenId, isRecorded: true }, ...rows];
}

function rowsFromFallback(candidates: CandidateReading[]): DistributionRow[] {
  return candidates.map((candidate, index) => ({
    token: candidate.token,
    score: candidate.score,
    tokenId: candidate.tokenId,
    isRecorded: index === 0,
  }));
}

export interface TokenDistributionCardProps {
  token?: WorkbenchTokenSection;
  fallbackCandidates: CandidateReading[];
  canChoose: boolean;
  choice: { piece: string; tokenId?: number } | null;
  onChoose: (choice: { piece: string; tokenId?: number }) => void;
}

/** Milestone E's "candidate list -> token-distribution card": the same recorded top-k readout the old
 * Observatory inspector rendered inline, now its own component so the action tray's fork row can read
 * the SAME chosen candidate this card reports, rather than each owning separate state. */
export function TokenDistributionCard({
  token,
  fallbackCandidates,
  canChoose,
  choice,
  onChoose,
}: TokenDistributionCardProps) {
  const rows = token ? rowsFromWorkbench(token) : rowsFromFallback(fallbackCandidates);
  return (
    <section className="inspector-section token-distribution-card">
      <div className="section-title">
        <h3>Token distribution</h3>
        <span>TOP-K</span>
      </div>
      <div className="candidate-list">
        {rows.map((row, index) => (
          <button
            type="button"
            className={`${row.isRecorded ? "candidate is-leading" : "candidate"} ${choice?.piece === row.token && choice.tokenId === row.tokenId ? "is-fork-choice" : ""}`}
            disabled={row.isRecorded || !canChoose}
            aria-pressed={row.isRecorded ? undefined : choice?.piece === row.token && choice.tokenId === row.tokenId}
            onClick={() => onChoose({ piece: row.token, tokenId: row.tokenId })}
            key={`${row.token}-${row.tokenId ?? index}`}
          >
            <span>{row.token || "∅"}</span>
            <i><b style={{ width: formatPercent(row.score) }} /></i>
            <output>{row.score.toFixed(4)}</output>
          </button>
        ))}
        {!rows.length && <span className="unavailable">NO RECORDED DISTRIBUTION</span>}
      </div>
    </section>
  );
}
