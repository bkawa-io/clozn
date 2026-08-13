import type { EvidenceState as EvidenceStateData } from "../../core/investigation";
import { evidenceExactnessLabel, evidenceStateLabel } from "../../core/investigation";
import "./investigation.css";

export interface EvidenceStateProps {
  readonly state: EvidenceStateData;
  readonly className?: string;
}

/** A compact, textual status that preserves epistemic state rather than implying a value. */
export function EvidenceState({ state, className }: EvidenceStateProps) {
  const exactness = evidenceExactnessLabel(state.exactness);
  const classes = ["investigation-evidence", `investigation-evidence--${state.measurement.kind}`, className]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={classes} aria-label={exactness ? `${evidenceStateLabel(state)}; ${exactness}` : evidenceStateLabel(state)}>
      <span className="investigation-evidence__mark" aria-hidden="true" />
      <span>{evidenceStateLabel(state)}</span>
      {exactness ? <span>({exactness})</span> : null}
    </span>
  );
}
