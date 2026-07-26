# Clozn Visualization Guide

## Data contract first

Every visualization documents:

- the field driving each mark
- evidence class: measured, derived, estimated, illustrative, decorative
- degraded behavior when data is absent
- reduced-motion behavior
- textual equivalent

## Encoding preferences

- Position: ordered quantitative comparison
- Length: magnitude
- Area: approximate magnitude only
- Opacity: supporting cue, not sole encoding
- Glow: activity or selection
- Motion: change over time

## Canvas/WebGL requirements

- deterministic resizing
- Canvas2D or textual fallback
- capped device pixel ratio
- explicit teardown
- seeded randomness for demos
- no random flicker in reduced-motion/static state
- DOM-accessible controls and summaries

## Missing data

Do not replace missing values with plausible curves. Use “not recorded,” muted structure, and an explanation of the required capability.

## Decorative framing

Decorative planes and filaments are allowed only behind measured signals, without exact labels, and without suggesting active computation while idle.
