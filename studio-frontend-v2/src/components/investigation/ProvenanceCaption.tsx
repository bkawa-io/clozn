import type { EvidenceArtifactMode, EvidenceExactness } from "../../core/investigation";
import { evidenceExactnessLabel } from "../../core/investigation";
import "./investigation.css";

export interface ProvenanceCaptionProps {
  readonly method?: string;
  readonly measurementFloor?: string;
  readonly artifactId?: string;
  readonly qualification?: string;
  readonly artifactMode?: EvidenceArtifactMode;
  readonly exactness?: EvidenceExactness;
  readonly className?: string;
}

/** Instrument-style caption for the provenance and limits of a displayed finding. */
export function ProvenanceCaption(props: ProvenanceCaptionProps) {
  const exactness = evidenceExactnessLabel(props.exactness);
  const items = [
    props.method && `Method: ${props.method}`,
    props.measurementFloor && `Floor: ${props.measurementFloor}`,
    props.artifactId && `Artifact: ${props.artifactId}`,
    props.artifactMode && `${props.artifactMode[0].toUpperCase()}${props.artifactMode.slice(1)}`,
    exactness,
    props.qualification,
  ].filter((item): item is string => Boolean(item));

  if (!items.length) return null;
  return (
    <p className={["investigation-provenance", props.className].filter(Boolean).join(" ")}>
      {items.map((item) => <span className="investigation-provenance__item" key={item}>{item}</span>)}
    </p>
  );
}
