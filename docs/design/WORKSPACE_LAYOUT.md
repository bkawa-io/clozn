# Clozn Studio Workspace Layout

Status: implementation draft

```text
┌────────┬─────────────────────────────────────┬──────────────┐
│ rail   │ toolbar                             │ inspector    │
│        ├─────────────────────────────────────┤              │
│        │ route viewport                      │              │
│        ├─────────────────────────────────────┴──────────────┤
│        │ timeline / event strip                              │
└────────┴─────────────────────────────────────────────────────┘
```

## Regions

- **Rail:** persistent primary navigation; serving status at the bottom.
- **Toolbar:** route identity, current run/model context, global actions.
- **Viewport:** the page hero and majority of available area.
- **Inspector:** contextual and selection-driven.
- **Timeline:** present where ordered inference events exist.

## Responsive behavior

- `>=1180px`: full rail, inspector, and timeline.
- `860–1179px`: inspector becomes a drawer.
- `<860px`: rail becomes horizontal navigation; timeline scrolls horizontally.

## Keyboard model

- `Mod+K`: command palette
- `1–6`: primary routes when not typing
- `[` / `]`: previous/next layer or event
- `Space`: replay play/pause
- `J` / `L`: step backward/forward
- `I`: inspector
- `T`: timeline
- `Esc`: clear selection or close transient UI
