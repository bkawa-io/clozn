# Clozn Studio

Studio is the browser interface served by the same gateway that records runs.

```sh
clozn serve MODEL --port 8080
clozn studio --port 8080 --open
```

## Conversations

The landing view groups recorded conversations by day. Each row previews the opening message and latest
reply; opening it shows the transcript, where any request or response can be selected for analysis.
Chats originate from the OpenAI-compatible API rather than a separate Studio composer.

Clozn uses headings, fenced blocks, document markers, and paragraph boundaries to derive a bounded flat
source partition automatically.

The adaptive partition target is 16 and can be set from 1 through 64 with
`CLOZN_MAX_CONTEXT_ARMS`. The independent exact-unit safety ceiling defaults to 64 and can be lowered
with `CLOZN_MAX_CONTEXT_UNITS`; it does not increase Lean Context's separate 16-new-execution budget.
For total removable length `L` and arm limit `A`, v1 uses a target of
`ceil(L/A)`, a preferred minimum of `max(40, ceil(target/2))`, and a preferred maximum of
`max(240, target*2)`. It packs adjacent Markdown pieces toward those sizes, then merges the smallest
adjacent chunks inside each message until it reaches the cap. Every non-empty input message is eligible, regardless of role or
turn: system instructions, conversation history, assistant prefills, and the latest user message all belong
to the same analysis universe. Chat-template syntax remains structural and is never treated as source text.

One message is the smallest cross-turn unit in v1: exact receipt sources do not span message
boundaries. If the number of non-empty messages exceeds the 64-unit safety ceiling, Clozn preserves every message
and reports bounded context search as unavailable rather than silently dropping input. Direct
Context Effects can still measure the full source set. On larger local models, that measurement can take
several minutes because every source is a separate teacher-forced scoring arm.

## Run workspace

The analysis workspace has two modes:

- **Context Effects** opens on the linked input/output reader. The answer side groups exact token
  evidence into readable sentence/clause-sized links; expanding Answer restores the exact recorded
  evidence spans in a full-width heatmap. Support and suppression have distinct colors, and intensity
  reflects the magnitude of each span's token-level effect. Expanding Context likewise turns the source
  prose into a heatmap. Both expanded views keep a corresponding-text detail row visible: hover or
  keyboard focus previews the relationship, while click pins it. Overall combines support and
  suppression; the Tension and Suppression lenses isolate those relationships.
- **Lean Context** runs a teacher-forced loss search with at most 16 **new** model executions;
  compatible Context Effects measurements are reused without consuming that budget. It reports only the
  best candidate found within the budget and never claims a proven minimum. The result is shown as the
  original prose with removed ranges struck out, plus a kept-only view and fresh generations from the
  original and reduced contexts.

Context, provenance, and runtime identity live on the Highlights view rather than being repeated inside
every analysis view. There is no separate runtime dashboard, experiment matrix, or general comparison
workbench in v1.

## Evidence rules

- Delivered context is not automatically influential.
- Lean Context uses Context Effects scoring with fixed balanced loss tolerances. It permits bounded
  probability loss and does not create a reduced child.
- Low effect, unmeasured, unavailable, failed, and cancelled are distinct.
- “Best found within budget” is the normal bounded-search result.
- Deletion is explicit and refuses unsafe parent removal while children remain.

The backend routes used by these views are listed in
[STUDIO_SURFACE_CONTRACT.md](STUDIO_SURFACE_CONTRACT.md).
