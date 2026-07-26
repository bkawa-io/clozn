# RFC 003 — Observatory Workspace Integration

Status: implementation draft

## Goal

Make the Casting the dominant Observatory instrument while moving selection details and replay controls into the shared Studio inspector and timeline. This RFC does not change the measurement contract described in `observatory.mjs` or `cast-feed.mjs`.

## Data boundary

The workspace adapter receives the same cast object already passed to `casting.update()`.

It must not infer missing probabilities, entropy, provenance, commit layers, or alternatives. A missing field renders as “not recorded” or is omitted.

The adapter classifies its content as follows:

- token text and token order: measured when sourced from a real cast
- entropy: measured/approximated exactly as labeled by the cast producer
- source links: measured only when the provenance gate passed
- alternatives: measured recorded alternatives
- event-strip geometry: illustrative navigation over measured token order
- demo cast values: scripted demo and visibly labeled as such

## Selection contract

The Observatory owns one selected token index.

Selection can originate from:

1. the Casting canvas
2. the timeline token strip
3. keyboard stepping
4. an external route action

All sources call one method:

```js
controller.selectToken(index, { source: "casting" | "timeline" | "keyboard" | "route" })
```

The controller then:

- clamps the index
- updates the inspector
- updates timeline selection
- asks Casting to focus the token if that optional API exists
- dispatches `clozn:observatory-selection`

## Casting API extension

The preferred addition to `mountCasting` is backward-compatible:

```js
mountCasting(el, {
  onFork(tokenIndex, alternativeText),
  onSelect(tokenIndex),
})
```

Optional returned methods:

```js
{
  update(cast),
  destroy(),
  setNight(value),
  selectToken(index),
  getSelectedToken(),
}
```

Only `onSelect` is required for the first integration. The workspace controller feature-detects `selectToken`.

## Inspector sections

### Identity

- token index
- token piece
- real or demo cast

### Recorded signal

- entropy, preserving the producer's approximation label
- commit layer
- confidence/rank only when present

### Provenance

- verified source spans or words
- explicit “from weights” only when the producer emitted that claim
- otherwise “not available for this cast”

### Alternatives

- recorded alternatives
- fork action remains owned by Casting/Observatory

## Timeline

The timeline is a token event strip, not a quantitative chart.

- token order is measured
- width is uniform by default
- optional entropy changes vertical intensity only when present
- selected token has a non-color outline and `aria-current="true"`
- arrow keys step selection
- Home/End jump to boundaries
- Space is reserved for future replay and is not intercepted yet

## Degraded behavior

Without the shared workspace, the Observatory continues to operate exactly as it does today. The adapter returns a no-op controller.

Without a Casting `onSelect` callback, timeline and keyboard selection still update the inspector, but canvas clicks do not drive it.

## Accessibility

The token strip is a `listbox`; tokens are `option` elements. The selected token is announced with its index, text, and available entropy. Inspector updates use `aria-live="polite"`.
