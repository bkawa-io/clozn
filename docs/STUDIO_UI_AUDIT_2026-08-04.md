# Clozn Studio UI audit

- **Date:** 2026-08-04
- **Status:** current implementation audit
- **Companion proposal:** [Studio UI redesign](STUDIO_UI_REDESIGN_2026-08-04.md)
- **Related foundations:** [UX foundation](design/UX_FOUNDATION.md),
  [visual grammar](design/VISUAL_GRAMMAR.md), and
  [visualization guide](design/VISUALIZATION_GUIDE.md)

## 1. Scope and method

This is a hands-on audit of the running Studio rather than a review of screenshots or component
specifications. The UI was inspected at 1280 × 720 in its dark theme against the live local gateway.
The audit used the evidence-rich run `run_0019fca9a499b_b31221` and the comparison pair
`run_0019fca9a499b_b31221` / `run_0019fc490f82d_b64c9b`.

The review covered:

- Every primary workspace: Runs, Run, Compare, Behavior, and Runtime.
- Every Run section and each Read representation.
- Both Compare object types and the token-alignment disclosure.
- All six Behavior modules.
- Runtime from its first summary through installation controls.
- Safe read-only actions and disclosures, including comparison planning, correction resolution,
  snapshot preview, and retained run fields.
- Reachable compatibility routes for Diagnostics, Experiments, Sessions, Scope, and Snapshots.

No model-generating action, mutation, or application-code change was performed during the audit.

For each surface the review asked:

1. Is the first impression overwhelming, and is there a clear place to begin?
2. Does the surface foreground the information a user cares about?
3. Is the hierarchy between primary and secondary information legible?
4. Is the purpose of the page distinct from neighboring pages?
5. Is the surface visually interesting and suitable for a product demonstration?
6. Are its visualization techniques appropriate to the data and evidence class?

## 2. Executive assessment

Clozn Studio is visually polished but cognitively hostile. It frequently presents the data model,
evidence machinery, and internal identifiers more prominently than the user's question.

The central problem is not only small type. Almost everything receives similar visual weight:

- Borders, labels, IDs, evidence states, headings, and findings compete with each other.
- Uppercase monospace metadata appears more authoritative than readable prose.
- Missing evidence is represented honestly, but repeated so often that absence becomes the main content.
- Fixed sidebars and an open inspector consume substantial width by default.
- Several sophisticated visualizations encode data the user cannot identify.

The strongest surfaces are Read, Timing, and the standalone Diagnostics overview. These surfaces
foreground recognizable content or familiar measurements. The weakest are What mattered, Why, Runs,
Runtime, and the raw portion of Compare.

## 3. Cross-product findings

### 3.1 There is no reliable visual hierarchy

Most pages need one clear first result. Instead, the first viewport often contains a large shell header,
a run selector, a section sidebar, an inspector, multiple status registers, tiny technical labels, and
only the beginning of the actual content.

The product should consistently answer, in this order:

1. What happened?
2. Why should I care?
3. What evidence supports that?
4. What can I do?
5. What are the technical details?

Most current pages start at steps three through five.

### 3.2 Meaningful text is below reading size

The problem is most severe in explanatory copy, legends, chart metadata, source paths, and findings.
Much of it reads at approximately 8–11px in the audited viewport.

This contradicts the existing [UX foundation](design/UX_FOUNDATION.md#11-density-and-typography), which
sets these floors:

- Prompt and response prose: 15–16px.
- Structured evidence and inspector values: 13–14px.
- Timestamps, badges, and tertiary labels: 12px minimum.

The inspector can be collapsed, and doing so materially improves Behavior and Compare, but the control
is an unlabeled icon near the bottom of the rail. It is not a credible remedy for the default layout.

### 3.3 Human-readable text disappears where it matters most

Read proves that Studio possesses useful prompt and response text. Yet:

- What was sent does not show the rendered prompt.
- What mattered uses `T1`–`T58` and coarse/fine row IDs.
- Mechanism shows tokens and generic `user` sources without source excerpts.
- Compare shows context hashes and segment IDs instead of changed passages.
- Why shows JSON offsets instead of the cited instruction.

For the audited run, the investigation artifact is `metadata_only`: its prompt and answer spans contain
offsets and hashes rather than text. That is a legitimate evidence constraint. It means a textless
matrix should not be the primary presentation. Studio should reconstruct excerpts from retained text
and offsets only when the privacy contract permits it; otherwise it should provide an evidence-limited
summary rather than promote opaque IDs to the page hero.

### 3.4 Similar pages are not differentiated clearly enough

Notable overlaps include:

- What it received versus What was sent.
- What mattered versus Mechanism.
- Why versus Claims.
- One-shot retries under Behavior versus corrective retries under Why.
- Runtime defaults under Behavior versus the Runtime workspace.
- Runtime snapshots versus the standalone Snapshots route.
- Compare matrix versus the legacy Experiments route.

The product exposes implementation boundaries more clearly than user-task boundaries.

### 3.5 Sophisticated visualizations lack semantic anchors

Studio uses many appropriate techniques:

- Composition bars.
- Diverging influence matrices.
- Dumbbells.
- Ordinal evidence ladders.
- Waterfalls.
- Confidence traces.
- Token tapes.
- Bipolar sliders.
- Inline claim highlighting.

The labels around them are often raw IDs, field paths, or unexplained numbers. The charts therefore look
technical without helping the reader reach a conclusion.

### 3.6 There are functional layout and routing defects

- Runtime's bottom evidence cards overlap and truncate.
- The Sessions index reports 10 sessions but only eight are visible, and the page does not scroll.
- Selecting a session is intercepted by the Diagnostics compatibility route and opens an unrelated run
  overview.
- The standalone Snapshots page clips its lower card at the audited viewport.
- Several legacy URLs resolve to unexpected destinations.
- The Read lenses visually accumulate despite looking like mutually exclusive tabs.

## 4. Workspace and page audit

### 4.1 Runs

**First impression:** Very overwhelming. Filters, dense rows, evidence columns, comparison staging,
selection details, and the run-family canvas all compete.

**Important information:** Prompt or task label, output/result, model, status, duration, finish reason,
time, and whether deeper evidence is available.

**Low-value foreground information:** Repeated `NOT RECORDED`, CTX/INF/PERF/CLM marks, adapter absence,
short IDs, provenance labels, and a large family canvas when lineage is trivial.

**Hierarchy:** Poor. The run title is only slightly more prominent than its metadata.

**Distinct purpose:** Clear: find, select, and stage runs.

**Demo and visualization:** It resembles a serious lab console, but is difficult to explain in a demo.
The family visualization wastes space for a single unbranched run. Evidence absence should be summarized
once rather than becoming the dominant content of every row.

### 4.2 Run — Read / Clean

This is the strongest primary surface.

**First impression:** Calm once past the oversized run frame.

**Important information:** Actual input messages and output text are prominent and readable.

**Low-value foreground information:** The run selector and model/client/entry-point metadata collide
visually in the header.

**Hierarchy and purpose:** Strong. The surface clearly answers what went in and what came out.

**Demo and visualization:** Strong. It communicates the product immediately.

### 4.3 Read — Shakiness

The output receives widespread dotted underlining without an obvious legend or summary explaining what
is shaky, how severe it is, or where to begin. It resembles spellcheck decoration more than confidence
evidence. The most uncertain span should be called out directly, with the full overlay secondary.

### 4.4 Read — Sources

Sources adds superscript numbers to input and output spans, but the numbers are not self-explanatory. It
also remains combined with the Shakiness overlay. The reader cannot tell whether a number is a rank,
source ID, token group, or citation. A source view should visibly connect an answer claim to the exact
supporting passage.

### 4.5 Read — Concepts

Selecting Concepts produced no discernible new visual encoding, while Shakiness and Sources remained
active. If these controls are composable overlays, they look like tabs and therefore communicate the
wrong interaction model. If they are intended to be exclusive, the state is broken.

### 4.6 What it received

**First impression:** Dense and technical.

**Important information:** What exact content reached prompt assembly, what was omitted, and how much
context was consumed.

**Low-value foreground information:** Byte counts, segment IDs, span IDs, repeated delivery states, and
unaccounted-byte explanations.

**Hierarchy:** The composition bar is useful, but the meaningful question—what did the model actually
receive?—is unanswered.

**Distinct purpose:** Technically a delivery receipt, but insufficiently differentiated from What was
sent.

**Demo and visualization:** Weak. The composition bar is sound, but rows showing `UNAVAILABLE`, hashes,
and bytes feel like an API debugger. The page simultaneously says the stored receipt is `FULL` and the
view is `METADATA ONLY`, which is especially confusing.

### 4.7 What was sent

**First impression:** Dense and schema-oriented.

**Important information:** The final rendered prompt, included instructions, omitted history,
transformations, token budget, and termination.

**Low-value foreground information:** Template fingerprints, hashes, backend names, raw request fields,
and privacy metadata.

**Hierarchy:** Poor. The exact rendered prompt should dominate, but is absent.

**Distinct purpose:** Not clear enough relative to What it received.

**Demo and visualization:** Weak. This is mostly a schema report rather than an explanation of the final
model input.

### 4.8 What mattered

This is the largest usability failure.

**First impression:** Extremely overwhelming.

**Important information:** Which prompt passage affected which part of the answer, whether it supported
or suppressed it, how strong the effect was, and how the effect was measured.

**Low-value foreground information:** `T1`–`T58`, coarse/fine rows, duplicate matrices, plus/minus glyphs,
and dense hatch patterns.

**Hierarchy:** None. The user cannot identify either axis without hovering and understanding internal
IDs.

**Distinct purpose:** Conceptually clear, but it overlaps heavily with Mechanism.

**Demo and visualization:** Visually striking but not interpretable. The diverging scale and below-floor
hatching are appropriate. The primary view should instead be a ranked set of source-passage →
answer-passage relationships, with the matrix retained as an expert disclosure.

### 4.9 Why

This page presents a backend rule ledger as if it were a user explanation.

**First impression:** Extremely dense.

**Important information:** The one or two findings that plausibly explain the result, the actual evidence
excerpt, confidence and limitations, and a relevant action.

**Low-value foreground information:** `R01`–`R12`, negative checks, raw field paths, JSON offsets,
heuristic implementation notes, and unrelated retry cards.

**Hierarchy:** The one real finding is buried among mostly unavailable or negative rules.

**Distinct purpose:** The intention is clear, but the page does not answer “why” in natural language.

**Demo and visualization:** Poor. The page is an audit table with no causal or explanatory path.

The R07 finding should effectively read:

> A relevant instruction—“Always name the policy ID…”—appeared four messages before the final request.
> Clozn currently measures message count, not token or character distance, so this is a possible
> placement issue rather than proof. Try repeating the instruction near the final question.

That content already exists, but it is divided into equally weighted technical fragments.

### 4.10 Claims

**First impression:** Moderately dense.

**Important information:** Two claims could not be verified from retained evidence, and these are the
exact claims.

**Low-value foreground information:** A six-category legend where five categories have zero items.

**Hierarchy:** Better than Why, but status and evidence are still too small.

**Distinct purpose:** Clear: verify claims in the answer.

**Demo and visualization:** Moderate. Inline highlighting is useful, but every claim uses the same
highlight and no source passage is paired with it. Empty status classes should collapse.

### 4.11 Second opinion

The state is honest but nearly empty: only one resident model is available. The page is a dead end. It
needs a route to Runtime or an explanation of how to add or activate another model. Its purpose is clear,
but it is not useful as a standalone navigation destination or demonstration surface.

### 4.12 Without this passage

This is also honest but nearly empty: exact source addressing was not retained, so a removal experiment
cannot run. There is no route to inspect or enable the missing prerequisite. The action belongs next to
the passage it would change rather than in a permanent page.

### 4.13 Timing

This is one of the strongest pages.

**First impression:** Moderately dense but comprehensible.

**Important information:** 945ms total, 434ms generation, 131.3 tokens/s, prompt/output tokens, context,
and the phase breakdown.

**Low-value foreground information:** Tiny raw source paths and the complete diagnostic rule list below
the waterfall.

**Hierarchy and purpose:** Good at the top and weaker lower down. The purpose is clear.

**Demo and visualization:** Strong potential. KPI cards and a phase waterfall are appropriate. The page
needs a plain-language conclusion such as “prefill and decode account for most recorded time,” with
technical findings collapsed.

### 4.14 Time machine

**First impression:** Moderately overwhelming.

**Important information:** Which replay or branch modes are available, what each changes, cost, and
confidence.

**Low-value foreground information:** Long caveats, disabled controls, and repeated qualification
language.

**Hierarchy:** `STRUCTURALLY REPRODUCIBLE` dominates without telling the user what to do next.

**Distinct purpose:** Mostly clear.

**Demo and visualization:** Moderate. Three mode cards with outcome, availability, cost, and recommended
use would be more effective than the current instrument panel.

### 4.15 Mechanism

**First impression:** Dense but visually interesting.

**Important information:** The selected part of the answer, alternative tokens, confidence, and the exact
source passages influencing it.

**Low-value foreground information:** Raw token positions, punctuation-only selections, repeated `user`
source labels, and tiny chart/token metadata.

**Hierarchy:** Weak. Selecting comma token number 16 provides no useful context.

**Distinct purpose:** Unclear relative to What mattered.

**Demo and visualization:** High potential. The top-k chart, confidence trace, and token tape are good
techniques, but they need to be anchored to the answer text and source excerpts.

### 4.16 The record

This is a successful restrained page.

**First impression:** Calm.

**Important information:** Run identity, provenance, finish state, lineage, and event history.

**Low-value foreground information:** Minimal.

**Hierarchy and purpose:** Good and clear. It is an immutable ledger rather than an explanation.

**Demo and visualization:** Neutral but appropriate. The raw retained-fields JSON is correctly hidden
behind a disclosure.

### 4.17 Compare — recorded runs

**First impression:** Attractive at the top and increasingly overwhelming below it.

**Important information:** What meaningfully changed between A and B, and which changes plausibly explain
the different output.

**Low-value foreground information:** Raw field paths, segment hashes, JSON-shaped values, repeated rank
labels, and a complete five-step ladder for every observed finding.

**Hierarchy:** Axis chips provide a useful overview, but the meaningful findings are buried below dozens
of rows.

**Distinct purpose:** Clear.

**Demo and visualization:** Strong-looking initially, then weak once the raw diff begins. The dumbbell is
misleading for categorical and unrelated values: a connector implies a common quantitative scale that
does not exist. Dumbbells should be reserved for comparable numeric values. Text, template, identity,
and context changes need direct before/after diffs.

The safe change-test preview is clearer: it reports that context and sampling are replayable and states
a maximum of four child runs and 120 seconds. That is useful information, though still too small.

### 4.18 Compare — token alignment

The disclosure adds A/B token inspection, confidence and entropy deltas, a confidence comparison chart,
and matched-token counts. These are useful expert tools, but remain disconnected from surrounding answer
text. The initial aligned pair was `We` versus `**`, which is not a useful demonstration selection.
Token-diagnostic links also resolve through compatibility routing in surprising ways.

### 4.19 Compare — experiment matrix

The page is not overwhelming because it is empty. It repeats the headings “Experiment matrix” and
“Experiment matrices,” then offers no path explaining how to create or import a result. Its role as an
alternate Compare object is plausible, but it is not a useful empty state or demonstration surface.

### 4.20 Behavior — general frame

The default three-column layout is a major contributor to unreadability. Collapsing the Consequence
inspector makes the center substantially better, but the control is difficult to discover.

### 4.21 Behavior — One-shot retries

The action-card model is understandable, though descriptions are long and small. The module also
overlaps with corrective retries shown on Why.

### 4.22 Behavior — Runtime defaults

The content is comparatively simple: sampling parameters and a disposition guard. The main issue is
information architecture. `Runtime` exists both here and as a top-level workspace. This module should be
named for its task, such as Generation defaults, and belong to Runtime.

### 4.23 Runtime

**First impression:** Extremely overwhelming. It combines health, capability flags, serving model,
local inventory, omitted records, snapshots, adoption, privacy controls, and a source
ledger in one long internal scroll.

**Important information:** Is the runtime healthy? What model is serving? What capabilities work? What
action is needed?

**Low-value foreground information:** Endpoint paths, repeated evidence-boundary prose, unavailable
installation features, model hashes, and full snapshot management.

**Hierarchy:** Weak. `Healthy` competes with four unavailable capabilities and many caveats.

**Distinct purpose:** Clear: installation state.

**Demo and visualization:** Weak despite useful data. The bottom Installation controls visibly overflow
and overlap. Evidence text repeatedly truncates. Snapshots are much more readable on their standalone
route than in this three-column page.

Runtime should begin with a compact health summary and serving-model card. Inventory, capabilities,
snapshots, and unsupported controls should become separate sections or disclosures.

## 5. Compatibility route audit

### Diagnostics

The standalone Diagnostics overview is cleaner and more legible than several newer Run sections. It
provides six availability cards followed by readable input and output text. Individual diagnostic links
are intercepted and mapped into the new Run reader, preserving URLs while making the compatibility
information architecture inconsistent.

### Scope and Investigation aliases

`#/scope` and `#/investigation` both resolve to the same Diagnostics overview rather than their nominal
surfaces.

### Sessions

The Sessions index is reachable at `#/sessions`, but is made almost entirely of raw session IDs. Two
sessions are clipped, the page does not scroll, and selecting a session opens an unrelated Diagnostics
run overview.

### Snapshots

The standalone Snapshots page is clearer than Runtime's embedded version, but its final card is clipped.
The safe pin preview returned the requirement “a ready identity-qualified product worker” as a small
cyan status line rather than an integrated prerequisite state.

### Experiments

The legacy Experiments route is effectively the same empty Compare matrix.

## 6. Recommended priority

1. Fix routing and clipping defects: Sessions navigation, inaccessible rows and cards, and Runtime
   overflow.
2. Enforce the documented typography floor and default the inspector closed at this viewport.
3. Redesign What mattered around source and answer excerpts; demote the raw matrix.
4. Redesign Why around one plain-language finding, evidence quote, limitation, and action.
5. Simplify Runs to the small set of fields needed for triage and selection.
6. Group Compare by semantic axis, move findings above raw differences, and use dumbbells only for
   quantitative measures.
7. Separate Runtime into health, models, capabilities, configuration, and snapshots.
8. Clarify or remove overlapping destinations and duplicate compatibility surfaces.

The product has strong evidence primitives and several good visualization ideas. The current interface
exposes too many of them simultaneously, before establishing what the user is supposed to learn.
