import { useId, useMemo, useRef } from "react";
import { locusKey, type InvestigationLocus } from "../../core/investigation";
import "./investigation.css";

export interface RegistrationMark {
  readonly locus: InvestigationLocus;
  readonly label: string;
  /** Ordered coordinate normalized to the range 0…1. */
  readonly position: number;
  readonly related?: boolean;
}

export interface RegistrationRailProps {
  readonly marks: readonly RegistrationMark[];
  readonly selected?: InvestigationLocus;
  readonly onSelect?: (mark: RegistrationMark) => void;
  readonly ariaLabel?: string;
  readonly className?: string;
}

interface PositionedMark extends RegistrationMark { lane: number; }

/** Allocates a separate visual lane for nearby marks at the same rendered coordinate. */
export function assignCollisionLanes(marks: readonly RegistrationMark[], minimumGap = 0.055): PositionedMark[] {
  const laneEnds: number[] = [];
  return marks
    .map((mark, originalIndex) => ({ mark, originalIndex }))
    .sort((a, b) => a.mark.position - b.mark.position || a.originalIndex - b.originalIndex)
    .map(({ mark }) => {
      const position = Math.max(0, Math.min(1, mark.position));
      let lane = laneEnds.findIndex((end) => position - end >= minimumGap);
      if (lane < 0) lane = laneEnds.length;
      laneEnds[lane] = position;
      return { ...mark, position, lane };
    });
}

export function RegistrationRail({ marks, selected, onSelect, ariaLabel = "Related loci", className }: RegistrationRailProps) {
  const railId = useId();
  const buttons = useRef<Array<HTMLButtonElement | null>>([]);
  const positioned = useMemo(() => assignCollisionLanes(marks), [marks]);
  const selectedKey = selected && locusKey(selected);

  const selectAt = (index: number) => {
    const mark = positioned[index];
    if (!mark) return;
    buttons.current[index]?.focus();
    onSelect?.(mark);
  };

  return (
    <div className={["registration-rail", className].filter(Boolean).join(" ")} role="listbox" aria-label={ariaLabel} id={railId}>
      {positioned.map((mark, index) => {
        const key = locusKey(mark.locus);
        const active = selectedKey === key;
        return (
          <button
            aria-label={mark.label}
            aria-selected={active}
            className="registration-rail__mark"
            data-related={mark.related || undefined}
            key={key}
            onClick={() => onSelect?.(mark)}
            onKeyDown={(event) => {
              if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); selectAt(Math.min(index + 1, positioned.length - 1)); }
              if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); selectAt(Math.max(index - 1, 0)); }
              if (event.key === "Home") { event.preventDefault(); selectAt(0); }
              if (event.key === "End") { event.preventDefault(); selectAt(positioned.length - 1); }
            }}
            ref={(element) => { buttons.current[index] = element; }}
            role="option"
            style={{ "--registration-position": mark.position, "--registration-lane": mark.lane } as React.CSSProperties}
            type="button"
          >
            <span className="visually-hidden">{mark.label}</span>
          </button>
        );
      })}
    </div>
  );
}
