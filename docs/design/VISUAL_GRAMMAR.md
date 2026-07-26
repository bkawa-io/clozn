# Clozn Studio Visual Grammar

Status: draft v1

## Product stance

Clozn Studio is an instrument for observing, comparing, and intervening in model behavior. It is not a generic analytics dashboard. Visual language must never imply measurements the runtime does not provide.

Every visual element belongs to one class:

1. **Measured** — rendered directly from recorded runtime data.
2. **Derived** — calculated deterministically from measured data.
3. **Estimated** — an approximation with a named method and limitations.
4. **Illustrative** — explanatory structure that is not itself a measurement.
5. **Decorative** — atmosphere with no semantic claim.

## Core principles

### Observation before decoration

Evidence receives the strongest contrast. Decorative atmosphere stays behind the evidence plane and never obscures text, hit targets, or quantitative marks.

### Light has meaning

Glow is reserved for active signals, selections, and computation in progress.

- Evidence: mint
- Inference path: periwinkle
- Alternative or fork: pink/lilac
- Intervention: gold
- Instability: warm coral
- Selection: near-white
- Inactive structure: neutral ink and line tokens

### Motion represents change

Motion should explain propagation, state transition, provenance, replay, or navigation. Idle instruments remain nearly still.

### Space represents abstraction

- Top: prompt, context, or causes
- Middle: model interior, transformation, competition
- Bottom: emitted answer or consequence
- Left-to-right: progression through depth or time
- Foreground: current selection
- Background: context and illustrative structure

### One hero per page

- Lens: response and evidence
- Observatory: Casting
- Compare: divergence
- Behavior: intervention and consequence
- Runs: run history
- Models: identity and capability evidence

## Material classes

### Environment
Application chassis and background. Quiet and low contrast.

### Instrument
Controls, navigation, inspectors, ledgers, and toolbars.

### Viewport
Observation surface containing model phenomena.

### Signal
Measured or derived phenomenon. Highest semantic contrast and the only class entitled to strong glow.

## Evidence treatments

| Class | Treatment | Disclosure |
|---|---|---|
| Measured | solid line, stable label, semantic color | source run/field available |
| Derived | solid or lightly dashed | method named |
| Estimated | dashed, reduced opacity, uncertainty cue | method and limitations |
| Illustrative | neutral/lilac scaffold, low contrast | label where ambiguity is possible |
| Decorative | background only, no quantitative labels | none if unmistakably atmospheric |

Decorative layers must not use exact layer numbers, probabilities, confidence values, or calibrated axes.

## Typography voices

### Machine
Metadata, state, IDs, layers, shortcuts, compact controls. Monospace, small, tracked, tabular.

### Instrument
Page titles and instrument names. Restrained display face, compact line height.

### Editorial
Responses and explanations. Readable, normal casing, generous line height.

## Interaction grammar

- **Hover:** reveals affordance; does not commit selection.
- **Selection:** persists, updates the inspector, and is identifiable without color alone.
- **Focus:** always visible and stronger than hover.
- **Replay:** follows recorded event order; illustrative interpolation is visually distinct.
- **Fork:** bifurcating geometry; original path remains visible.
- **Intervention:** gold stitching, brackets, or splice marks applied to an existing run.

## Page archetypes

- Lens: microscope
- Observatory: observatory
- Compare: interferometer
- Behavior: intervention console
- Runs/Models: evidence ledger

## Accessibility

- Never encode meaning by color alone.
- Respect `prefers-reduced-motion`.
- Decorative layers use `aria-hidden`.
- Canvas visualizations need textual summaries and keyboard-accessible selections.
- Focus order follows reading order.
- Resizable layouts remain usable at 200% zoom.

## Review checklist

1. What class is each new visual element?
2. What data field drives it?
3. Does decoration imply precision?
4. Is the page hero dominant?
5. Does motion explain change?
6. Is selection readable without color?
7. Is reduced motion complete?
8. Does the view work without WebGL?
9. Can users distinguish measured, derived, and illustrative content?
10. Does the implementation reuse tokens rather than hardcoded colors?
