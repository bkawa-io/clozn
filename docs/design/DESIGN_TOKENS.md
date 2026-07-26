# Clozn Studio Design Tokens

Status: draft v1

Semantic tokens should be consumed instead of page-specific hardcoded values.

## Layer tokens

```css
--chassis-0; --chassis-1;
--instrument-0; --instrument-1;
--viewport-0; --viewport-1;
--surface-raised;
```

## Signal aliases

```css
--signal-evidence: var(--evidence);
--signal-path: var(--iri-peri);
--signal-alternative: var(--almost);
--signal-fork: var(--fork);
--signal-intervention: var(--iri-gold);
--signal-instability: var(--shaky);
--signal-selection: var(--ink);
```

## Borders

```css
--edge-subtle;
--edge-instrument;
--edge-active;
--edge-selection;
```

## Shadows and glow

```css
--shadow-chassis;
--shadow-instrument;
--shadow-overlay;
--glow-evidence;
--glow-path;
--glow-fork;
--glow-intervention;
--glow-selection;
```

Glow must use semantic colors. Generic purple glow is prohibited.

## Spacing

Use a 4px rhythm: 4, 8, 12, 16, 20, 24, 32, and 40px.

## Default workspace geometry

```css
--rail-width: 72px;
--inspector-width: 320px;
--timeline-height: 148px;
--toolbar-height: 44px;
--panel-min: 220px;
```

## Radius

- 6px: controls
- 10px: panels
- 14px: major viewport
- pill: status chips only

## Motion

```css
--motion-instant: 80ms;
--motion-fast: 140ms;
--motion-standard: 220ms;
--motion-slow: 420ms;
--motion-ambient: 2800ms;
--ease-instrument: cubic-bezier(.2,.8,.2,1);
--ease-settle: cubic-bezier(.16,1,.3,1);
```
