/**
 * The one shared primitive for "does evidence exist here, and if not, why". Four surfaces already
 * render some version of this: EvidenceCaveat's `evidenceStateBadge`, LensSelectionInspector's
 * `availabilityView`/`absenceView` (see its `LensInspectorAvailability` union), WhatMattered's
 * `CellState`, and the `.lens-selection-availability` treatments in lens-selection-inspector.css. Each
 * drew its own version of the same four states. EvidenceMark is what the rest of the component system
 * should compose on top of instead of inventing a fifth rendering; it does not replace those call
 * sites (that refactor is bigger than this component).
 *
 * The Studio's categorical palette fails colourblind separation (measured with a standard CVD
 * simulation/deltaE validator: worst pair deltaE 3.6 for deuteranopia, 5.7 even for normal vision --
 * against the usual separability floor of 15). So FORM carries the state -- solid fill, diagonal
 * hatch, dashed outline, solid outline -- and colour is layered on top only as a secondary cue. A
 * greyscale screenshot must still separate all four; see EvidenceMark.css for the four silhouettes.
 *
 * The type below is the part that matters most. `reason` is required on the two states that mean "no
 * measurement exists" (`not_measured`, `unavailable`) and is structurally impossible to pass on the
 * two that mean a measurement actually happened (`measured`, `below_floor`) -- TypeScript rejects a
 * `reason` on those members outright, it is not just an unused optional. That is the type-level form
 * of this product's core invariant: missing evidence is not a zero, it is always an explained absence.
 */

export type EvidenceState = "measured" | "below_floor" | "not_measured" | "unavailable";

export type EvidenceMarkVariant = "dot" | "chip";

interface EvidenceMarkCommonProps {
  /** "dot" is a compact inline glyph for table cells / list rows. "chip" adds a visible label (and,
   * for the absence states, the reason) for standalone use. Defaults to "dot". */
  variant?: EvidenceMarkVariant;
  /** Overrides the default per-state label ("Measured", "Below floor", "Not measured", "Unavailable"). */
  label?: string;
  className?: string;
}

export type EvidenceMarkProps =
  | (EvidenceMarkCommonProps & { state: "measured"; reason?: never })
  | (EvidenceMarkCommonProps & { state: "below_floor"; reason?: never })
  | (EvidenceMarkCommonProps & { state: "not_measured"; reason: string })
  | (EvidenceMarkCommonProps & { state: "unavailable"; reason: string });

const DEFAULT_LABEL: Record<EvidenceState, string> = {
  measured: "Measured",
  below_floor: "Below floor",
  not_measured: "Not measured",
  unavailable: "Unavailable",
};

/** Exhaustive on `EvidenceState`, matching the `never`-assignment pattern already used for this exact
 * kind of switch (EvidenceCaveat.tsx, WhatMattered.tsx) -- a fifth state fails `tsc -b`, not the render. */
function stateClassName(state: EvidenceState): string {
  switch (state) {
    case "measured": return "is-mark-measured";
    case "below_floor": return "is-mark-below-floor";
    case "not_measured": return "is-mark-not-measured";
    case "unavailable": return "is-mark-unavailable";
    default: {
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}

/** Narrows on `props.state` rather than a loose `"reason" in props` check, so it is the compiler --
 * not a runtime guess -- that proves `reason` cannot leak out of the two presence states. */
function reasonOf(props: EvidenceMarkProps): string | undefined {
  switch (props.state) {
    case "measured":
    case "below_floor":
      return undefined;
    case "not_measured":
    case "unavailable":
      return props.reason;
    default: {
      const exhaustive: never = props;
      return exhaustive;
    }
  }
}

export function EvidenceMark(props: EvidenceMarkProps) {
  const { state, variant = "dot", label, className } = props;
  const reason = reasonOf(props);
  const stateLabel = label ?? DEFAULT_LABEL[state];
  // A dot has no visible text of its own, so the reason has to travel through the accessible name
  // (and title) or it is lost to assistive tech -- rule 3 applies to screen readers too.
  const accessibleName = reason ? `${stateLabel} -- ${reason}` : stateLabel;
  const glyphClassName = ["evidence-mark-glyph", stateClassName(state)].join(" ");

  if (variant === "chip") {
    return (
      <span
        className={["evidence-mark", "evidence-mark-chip", stateClassName(state), className].filter(Boolean).join(" ")}
        role="img"
        aria-label={accessibleName}
        title={reason}
      >
        <span className={glyphClassName} aria-hidden="true" />
        {/* Visible for sighted users; aria-hidden because the parent's aria-label above already says
            the same thing once -- without this, screen readers would read the state twice. */}
        <span className="evidence-mark-copy" aria-hidden="true">
          <strong>{stateLabel}</strong>
          {reason && <small>{reason}</small>}
        </span>
      </span>
    );
  }

  return (
    <span
      className={["evidence-mark", "evidence-mark-dot", glyphClassName, className].filter(Boolean).join(" ")}
      role="img"
      aria-label={accessibleName}
      title={reason ?? stateLabel}
    />
  );
}
