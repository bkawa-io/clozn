# Clozn Studio UI redesign proposal

- **Date:** 2026-08-04
- **Status:** recommendation
- **Evidence base:** [Studio UI audit](STUDIO_UI_AUDIT_2026-08-04.md)
- **Capability boundary:** [Capability and release matrix](CAPABILITIES.md)
- **Design foundations:** [UX foundation](design/UX_FOUNDATION.md),
  [visual grammar](design/VISUAL_GRAMMAR.md), and
  [visualization guide](design/VISUALIZATION_GUIDE.md)

## 1. Outcome

Studio should retain five primary workspaces, but organize them around the user's investigation loop
rather than backend capability boundaries:

1. **Runs** — find a run worth investigating.
2. **Inspect** — understand one run and its available evidence.
3. **Compare** — understand what moved between runs or across an experiment.
4. **Improve** — test and retain a behavior change.
5. **Runtime** — manage the environment that makes evidence and interventions possible.

`Run` becomes `Inspect` to name the user's task. `Behavior` becomes `Improve` to include both temporary
tests and durable changes. Forensic token and mechanism tools remain a deep mode of Inspect rather than a
sixth permanent destination. Experiment matrices remain an alternate Compare object rather than gaining
another top-level workspace.

This preserves the compact five-item rail while creating a coherent loop:

`Runs → Inspect → Compare → Improve`

Runtime supports the loop without competing with it.

## 2. Design constraints

### 2.1 One conclusion before one instrument

Every page begins with one primary statement or selection, followed by the visual instrument that
supports it. The interface should never require the user to interpret a chart before knowing what
question the chart answers.

### 2.2 Progressive evidence disclosure

Every evidence-bearing page follows four layers:

1. **Conclusion:** a plain-language answer to the page's question.
2. **Readable evidence:** exact prompt, answer, passage, phase, or changed value.
3. **Method and limitations:** provenance, availability, floor, clamp, clock owner, cost, or confidence.
4. **Raw record:** IDs, hashes, field paths, JSON, and endpoint provenance.

Raw records remain available, but never occupy the default reading path.

### 2.3 Missing evidence remains explicit but summarized

The invariant “missing evidence is not zero” remains non-negotiable. The redesign changes repetition,
not honesty:

- Summarize repeated absence once at the page or group level.
- Show an EvidenceMark on the selected object or attempted action.
- Expand the exact reason, preconditions, and source field on demand.
- Do not render an unavailable capability as an empty quantitative mark.

### 2.4 Text has a documented floor

Use the sizes already specified by the UX foundation:

- Prompt, response, findings, and explanatory prose: 15–16px with approximately 1.5 line height.
- Tables, structured evidence, chart labels, and inspector values: 13–14px.
- Timestamps, badges, shortcuts, and tertiary labels: 12px minimum.

Monospace is reserved for exact prompts, code, IDs, hashes, tokens, and tabular measurements. A finding
or explanation is editorial prose, not machine metadata.

### 2.5 Responsive panels collapse before text shrinks

At 1,000–1,599px, including the audited 1280px viewport:

- The inspector is closed by default and opens as a drawer.
- The run browser is independently collapsible.
- Prompt and output remain side by side while each has a readable line length.
- The evidence deck may collapse vertically but remains available.

Only at 1,600px and above should the run browser, main stage, inspector, and evidence deck all be pinned
simultaneously.

## 3. Proposed information architecture

```text
Runs
├── Needs attention
├── All runs
├── Sessions and lineage
└── Saved views

Inspect
├── Read
├── Input
├── Evidence
├── Performance
├── Replay
└── Record

Compare
├── Run comparison
└── Experiment matrix

Improve
├── Quick tests
├── Corrections
├── Controls
└── Profiles

Runtime
├── Overview
├── Models
├── Capabilities
├── Defaults and capture
└── Snapshots
```

### 3.1 Current-to-proposed placement

| Current surface | Proposed placement | Treatment |
|---|---|---|
| Runs | Runs | Simplify into triage ledger and saved views |
| Read | Inspect → Read | Preserve as the stable reading surface |
| Shakiness | Inspect → Read overlay | One overlay at a time; add summary and legend |
| Sources | Inspect → Read/Evidence overlay | Connect answer text to exact source excerpts |
| Concepts | Inspect → Evidence forensic mode | Show only when qualified evidence exists |
| What it received | Inspect → Input → Delivery | Merge with What was sent as one stage trace |
| What was sent | Inspect → Input → Rendered | Show exact model-facing prompt when permitted |
| What mattered | Inspect → Evidence → Sources | Replace textless matrix as primary view |
| Why | Inspect → Read findings / Evidence detail | Findings become summary cards, not a page of rules |
| Claims | Inspect → Read overlay | Pair each claim with evidence and status |
| Second opinion | Inspect contextual action | Remove as permanent page; track results in Improve |
| Without this passage | Inspect source action | Attach to selected passage; track result in Compare |
| Timing | Inspect → Performance | Preserve and simplify |
| Time machine | Inspect → Replay | Present modes as availability cards |
| Mechanism | Inspect → Evidence → Tokens | Deep forensic representation, not sibling of What mattered |
| The record | Inspect → Record | Preserve; keep raw fields disclosed |
| Compare runs | Compare → Run comparison | Foreground semantic findings and readable diffs |
| Experiment matrix | Compare → Experiment matrix | Keep, add creation/import guidance and result summary |
| One-shot retries | Improve → Quick tests | Also launched contextually from Inspect |
| Corrections | Improve → Corrections | Preserve durable lifecycle |
| Tone dials | Improve → Controls | Group and show changed axes first |
| Concept steering | Improve → Controls | Capability-gated section, not an empty module |
| Runtime defaults | Runtime → Defaults and capture | Move out of Behavior/Improve |
| Profiles | Improve → Profiles | Add bundle preview and active-state summary |
| Runtime installation state | Runtime → Overview | Compact health and action summary |
| Runtime inventory | Runtime → Models | Full-width model table/cards |
| Runtime capability flags | Runtime → Capabilities | Group by user task and availability |
| Runtime snapshots | Runtime → Snapshots | Use the readable standalone layout |
| Sessions compatibility page | Runs → Sessions and lineage | Use readable session labels and branch structure |
| Diagnostics compatibility page | Inspect | Redirect visibly and preserve the chosen run/section |

## 4. Global workspace frame

### 4.1 Desktop at 1280px

```text
┌────────┬─────────────────────────────────────────────────────────────┐
│ rail   │ compact context bar                                         │
│        ├─────────────────────────────────────────────────────────────┤
│        │ page hero: one conclusion, selection, or primary action     │
│        ├─────────────────────────────────────────────────────────────┤
│        │ main readable stage                                         │
│        │                                                             │
│        │                                              inspector ▸     │
│        ├─────────────────────────────────────────────────────────────┤
│        │ collapsible evidence deck                                   │
└────────┴─────────────────────────────────────────────────────────────┘
```

The inspector is a drawer, not a permanently allocated third column. Its rail control needs a text label
or tooltip and a conventional icon placement near the current selection.

### 4.2 Compact context bar

The large run header becomes a one-line or two-line persistent frame:

```text
Refund request after renewal                 LOW CONFIDENCE
Qwen2.5 0.5B · Q4_K_M · openai_api · 945ms        run…b31221 ▾
```

The human-readable title, warning, model, status, and duration are primary. Client, full run ID, hashes,
and exact artifact identity move to the inspector or Record.

### 4.3 Inspector contract

The inspector answers only four questions about the current selection:

1. What is selected?
2. What is known about it?
3. What evidence supports that statement?
4. What can be done next?

It does not remain open merely to repeat global run or model metadata.

## 5. Runs redesign

### 5.1 Page purpose

Answer: **Which run should I inspect next?**

### 5.2 Recommended layout

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Runs                                      80 recorded · 3 need review│
│ [Needs attention] [All] [Sessions] [Saved views]       Search…  Filter│
├──────────────────────────────────────────────────────────────────────┤
│ ! Refund request after renewal                    945ms · STOP · 19:28│
│   Qwen2.5 0.5B · low confidence        “We should inform…”            │
│   Evidence: context, influence, timing, claims                        │
├──────────────────────────────────────────────────────────────────────┤
│   Extract fields from support note                619ms · STOP · 15:21│
│   qwen-0.5b                           “**Customer Name:** …”           │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.3 Hierarchy

- Primary: task/prompt title and attention reason.
- Secondary: response excerpt, model, finish state, duration, and time.
- Tertiary: source/client, short ID, and compact evidence availability.
- Hidden until selected: full provenance, hashes, all evidence reasons, lineage, and comparison staging.

Evidence availability should be one compact summary such as `4/4 artifacts available` or
`Timing available · influence not measured`. It should not render four full absence explanations in
every row.

### 5.4 Visualization

Use visualization only for aggregate triage:

- A small latency distribution for the active filtered set.
- Counts for error, truncated, slow, and low-confidence runs.
- A session/lineage tree only after the user selects Sessions or a run with branches.

Do not reserve a large node canvas for an unbranched run.

## 6. Inspect redesign

Inspect is the durable center of Studio. It keeps readable prompt and response text available while the
user changes evidence representations.

### 6.1 Inspect → Read

#### Purpose

Answer: **What happened, and what deserves attention?**

#### Layout

```text
┌───────────────────────────────┬──────────────────────────────────────┐
│ Input                         │ Output                               │
│                               │                                      │
│ SYSTEM                        │ We should inform the customer…        │
│ You are Northwind Cloud…      │                                      │
│                               │ [Claim 1: unverifiable]               │
│ USER                          │ [Claim 2: unverifiable]               │
│ POLICY REFUND-2024-11…        │                                      │
│                               │                                      │
├───────────────────────────────┴──────────────────────────────────────┤
│ Findings: 1 likely issue · instruction may be too far from request  │
│ Evidence availability: influence measured · second opinion blocked  │
└──────────────────────────────────────────────────────────────────────┘
```

Keep Clean as the default. Replace the current tab-like overlay controls with an explicit single-choice
control:

`View: Clean | Claims | Confidence | Sources`

Only one overlay is active at a time. Concepts appears only in qualified forensic mode.

#### Claims

- Highlight only claims with a non-neutral status.
- Place the status label beside the claim, not in a six-category global legend.
- Selecting a claim opens exact supporting/contradicting passages and evidence limitations.
- Hide status categories with zero claims behind a legend disclosure.

#### Shakiness

- Start with a summary: `12 of 58 tokens are below the confidence threshold; the first answer sentence
  contains the weakest region.`
- Highlight contiguous spans, not every token independently.
- Show confidence values on selection, not as a wall of numbers.

#### Findings / Why

Move high-value findings below the readable answer. Each finding uses this contract:

```text
Instruction may be too far from the request                   POSSIBLE
“Always name the policy ID you relied on.” appeared four messages earlier.

Limitation: distance is based on message count, not token or character distance.
[Show evidence] [Test by moving instruction]
```

Negative checks belong in a collapsed `11 other checks` disclosure. Rule IDs and field paths belong in
the evidence detail.

### 6.2 Inspect → Input

#### Purpose

Answer: **What reached the model, and how did it change along the way?**

Merge What it received and What was sent into one trace:

`Requested → Delivered → Assembled → Rendered → Limits → Termination`

#### Layout and visualization

Use a stage trace where each message or source is a selectable horizontal clip:

```text
                    REQUESTED  DELIVERED  ASSEMBLED  RENDERED
System instructions     ●──────────●──────────●──────────●
Refund policy           ●──────────●──────────●──────────●
Retention policy        ●──────────●──────────●──────────●
SLA policy              ●──────────●──────────●──────────●
Final question          ●──────────●──────────●──────────●
Unaccounted bytes                  ▒▒▒▒▒▒
```

Position communicates progression; line continuity communicates survival. A break or explicit omission
mark communicates exclusion. Hatching remains reserved for a recorded gap or below-floor state.

Above the trace, retain the composition bar for proportion. Below or beside it, show the selected
message's actual text. Bytes, span IDs, hashes, and exact offsets live in the inspector.

Provide three readable representations:

- **Conversation:** messages and sources as authored.
- **Delivery:** the stage trace.
- **Rendered:** the exact model-facing prompt with section boundaries when privacy permits it.

When text is privacy-limited, replace it with one explicit privacy state and available metadata. Do not
show `FULL` and `METADATA ONLY` without explaining that one describes the stored receipt and the other
describes the authorized view.

### 6.3 Inspect → Evidence

#### Purpose

Answer: **What evidence connects this input to this output?**

This consolidates What mattered and Mechanism while preserving multiple depths.

#### Default: ranked source-to-answer evidence

```text
┌───────────────────────────────┬──────────────────────────────────────┐
│ Source passages               │ Answer spans                         │
│                               │                                      │
│ 1  REFUND-2024-11             │ A  “14-day refund period…”           │
│    “After 14 days no refund…” ├──────── supports +0.82 nats ────────┤
│                               │                                      │
│ 2  System instruction         │ B  “which policy applies…”           │
│    “Always name the policy…”  ├──── below floor / not established ──┤
└───────────────────────────────┴──────────────────────────────────────┘
```

The default ordering is by strongest measured absolute effect, grouped into answer claims or readable
answer spans. Each relationship carries:

- Supports or suppresses.
- Signed and absolute effect on selection.
- Evidence state.
- Measurement method and floor.
- Exact source and answer excerpts when authorized.

Color is secondary. Direction, labels, connector style, and lightness carry the meaning.

#### Expert matrix disclosure

Retain the diverging influence matrix under `Open full matrix` for users who need complete topology.

- Use readable truncated excerpts for row and column headers.
- If text is not authorized, use stable semantic labels such as `Refund policy, sentence 2` and
  `Answer claim 1`, not `T17`.
- Keep symmetric clamping, both poles, zero midpoint, floor value, hatch, and all four evidence states.
- Never render two matrices for the same links on one page.

#### Token mode

Token mode replaces the current Mechanism page:

- The answer remains visible and the selected token is highlighted in context.
- Default selection is the first meaningful low-confidence or source-linked token, never punctuation.
- Top-k alternatives appear as a horizontal bar chart.
- Confidence/entropy appears in the evidence deck synchronized to answer positions.
- Source links use passage excerpts rather than repeated `user` labels.
- J-lens and concept readouts appear only when their artifact is qualified for the selected model.

### 6.4 Inspect → Performance

#### Purpose

Answer: **Where did the run spend time?**

#### Layout

1. One conclusion: `945ms total; prefill and decode account for most recorded worker time.`
2. Four to six primary metrics.
3. The span waterfall.
4. A collapsed list of diagnostic rules and raw provenance.

#### Visualization

Keep the current waterfall invariants:

- One lane per clock owner.
- Never align cross-process offsets.
- Show unaccounted time as a visible hatched gap.
- Exclude overlap and startup from known in-request arithmetic while still rendering them.
- Use an EvidenceMark for absent duration.

Add an axis and a direct callout for the largest phase. Source field paths belong in a disclosure, not
under every KPI.

### 6.5 Inspect → Replay

#### Purpose

Answer: **What can be replayed or branched faithfully from this run?**

Present replay modes as comparable cards:

| Mode | Result | Availability | Cost | Best use |
|---|---|---|---|---|
| Structural replay | Rebuilds transcript structure | Available | One child run | Reproduce input shape |
| Same-prompt exact child | Restores verified boundary | Blocked or available | Verification + child | Test runtime stability |
| Appended-turn continuation | Continues session history | Blocked or available | One child | Ask the next question |

Select one mode before showing its controls. Show the recommended mode first. Long qualification details
belong under `Why is this blocked?`.

Snapshots are a prerequisite or durability action within Replay, but their inventory and management
remain in Runtime → Snapshots.

### 6.6 Inspect → Record

Preserve the current restrained ledger. Keep identity, recorded time, finish, lineage, and event history
visible. Keep retained fields and raw artifacts behind disclosures. Merge compatibility Diagnostics
`Events`, `Lineage`, and `Raw artifacts` here rather than maintaining parallel destinations.

### 6.7 Contextual actions removed from navigation

Second opinion, passage removal, claim retry, sampler retry, and controlled source tests attach to the
object they affect:

- Claim selected → `Retry unsupported claim` or `Ask second model`.
- Source selected → `Run without this passage` or `Measure influence`.
- Finding selected → `Test suggested change`.
- Token selected → `Open forensic token tools`.

Unavailable actions remain visible in the inspector with their reason and route to the missing Runtime
prerequisite. They do not require permanent empty pages.

## 7. Compare redesign

### 7.1 Compare → Run comparison

#### Purpose

Answer: **What changed, and what does the evidence license us to conclude?**

#### Layout

```text
┌──────────────────────────────────────────────────────────────────────┐
│ A Refund request                         B Corrective retry           │
│ Relationship: parent → child                    3 meaningful changes │
├──────────────────────────────────────────────────────────────────────┤
│ Finding: B delivered 3 fewer messages and changed 6 sampler values  │
│ Evidence level: observed; causation not established                  │
├──────────────────────────────────────────────────────────────────────┤
│ [Input & context] [Settings] [Output] [Performance] [Raw]            │
│                                                                      │
│ Context: 5 messages → 2 messages                                     │
│ Removed: “POLICY DATA-2023-04…”                                      │
│ Removed: “POLICY SLA-2025-01…”                                       │
│                                                                      │
│ Temperature                  0.8 ●──────────────● 0.2                 │
└──────────────────────────────────────────────────────────────────────┘
```

Move findings and the evidence license above individual rows. Group differences by the eight summary
axes rather than rendering one unbroken field ledger.

#### Encoding by data type

| Difference type | Recommended encoding |
|---|---|
| Comparable number | Paired dot/dumbbell with axis and direct selected value |
| Ordered status | Position on an ordinal ladder |
| Category or identity | Labeled A → B chips or two-column values |
| Boolean | Changed/unchanged state with explicit labels |
| Text | Inline or side-by-side diff with excerpts |
| Set of sources/messages | Added/removed list grouped by readable source |
| Missing/failed comparison | EvidenceMark with reason |

Do not draw a quantitative connector between arbitrary strings, JSON objects, hashes, or unrelated
categorical values.

Unchanged axes remain visible in the summary as quiet checks but collapse in the detail. Raw field paths
and JSON remain under Raw.

### 7.2 Evidence ladder

Render the causal ladder once for the selected finding, not five times on every finding card:

`Observed → Eliminated → Reproduced → Correlated → Causally supported`

The current position uses length and placement, not five unrelated colors. A finding at Observed should
read as the beginning of the ladder, not a peer of Causally supported.

### 7.3 Token alignment

Token alignment is a representation under Output, not a detached lower dashboard.

- Display both output texts with aligned spans.
- Selecting a span synchronizes the confidence comparison plot.
- Start at the first meaningful divergence, not the first technical alignment hunk.
- Place matched-token count and alignment method beside the control.
- Keep confidence and entropy values in the inspector or on selected hover.

### 7.4 Controlled change test

Attach `Preview test` to a selected finding. Preview reports:

- Which changes can be isolated.
- What remains locked.
- Maximum child runs and time.
- Worker and identity preconditions.
- The result that will open in Compare.

The execution action remains a separate explicit confirmation.

### 7.5 Compare → Experiment matrix

Keep experiments within Compare using `Object: Runs | Experiment`.

When results exist:

- Hero: pass/fail/changed summary and the most important variant effect.
- Matrix: cases × variants, with seed or repetition summarized in each cell.
- Cell encoding: status by form plus color, with unavailable distinct from failure.
- Detail: selected case, baseline/candidate outputs, assertions, uncertainty, and provenance.
- Filters: suite, model, variant, status, and date.

When empty, explain how results enter Studio and link to the relevant CLI/API documentation. Avoid two
nearly identical headings around an empty table.

## 8. Improve redesign

Improve contains interventions and durable behavior configuration. It does not become the place where
run evidence is first discovered; those actions originate from Inspect.

### 8.1 Improve → Quick tests

Combine one-shot retries, second opinion, passage removal, and bounded finding tests into a history of
temporary interventions.

Each card states:

- Source run and selected object.
- Exactly one proposed change.
- What stays locked.
- Cost and prerequisites.
- Preview/result status.
- Link to the resulting Compare pair.

Start-new-test controls may live here, but contextually launched tests arrive prefilled.

### 8.2 Improve → Corrections

Use a clear four-step lifecycle:

`Draft → Confirm → Verify → Enable`

The first viewport foregrounds the instruction and scope selector. Scope containment is a help disclosure,
not the largest initial visualization.

Saved corrections appear as readable cards with instruction, exact scope, status, last verification,
and undo/disable actions. Recorded resolution appears on the selected correction or run, not as an empty
global panel.

### 8.3 Improve → Controls

Combine Tone dials and Concept steering as qualified controls.

#### Tone

- Group axes into voice, density, certainty, and formatting where the capability metadata allows it.
- Show changed/active axes first.
- Collapse neutral axes into `7 unchanged controls`.
- Keep bipolar sliders because position around zero is the correct encoding.
- Show a readable A/B response preview for the selected change.
- Display `uncalibrated` once at the group level when all axes share that state.

#### Concepts

- If J-lens is unavailable, show one prerequisite card and route to Runtime → Capabilities.
- Do not show active-looking Apply controls before the prerequisite is satisfied.
- When available, show the concept, direction, strength, method, artifact identity, and an A/B preview.

### 8.4 Improve → Profiles

Before save, show the bundle contents:

- Active tone axes and values.
- Active concept directions.
- Guard and generation defaults referenced by the profile.
- Model/artifact compatibility.
- Description and intended use.

Saved profiles use cards with active/inactive state, last changed time, compatibility, and preview. The
user should never have to infer what `10 DIALS` means.

## 9. Runtime redesign

Runtime is a supporting utility workspace. Its first question is: **Can this installation perform the
next operation?**

### 9.1 Runtime → Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Runtime healthy                                      1 action needed │
│ Qwen2.5-0.5B serving on CUDA · Q4_K_M · context 4,096               │
├───────────────────────┬───────────────────────┬──────────────────────┤
│ Serving model         │ Evidence capabilities │ Storage / snapshots  │
│ Ready                 │ 4 available · 4 absent│ 3 pinned              │
├───────────────────────┴───────────────────────┴──────────────────────┤
│ Attention: J-lens unavailable; concept steering cannot run.         │
└──────────────────────────────────────────────────────────────────────┘
```

Endpoint paths and source ledgers move to a `Sources` disclosure. Repeated capability caveats collapse
into grouped counts and a selected detail.

### 9.2 Runtime → Models

Use the full width for local inventory:

| Model | Status | Quant | Size | Qualification | Available actions |
|---|---|---|---|---|---|

The serving model is clearly marked. Hashes are copyable secondary metadata. Do not show a residency or
capacity meter until the runtime reports those values.

### 9.3 Runtime → Capabilities

Group capabilities by user task:

- Generation and streaming.
- Replay and revision.
- Source measurement.
- Forensic/readout tools.
- Steering and intervention.

Each row shows Available, Unsupported, Not configured, or Not reported. Selecting an unavailable
capability explains which operation it blocks and what is required. Availability does not imply that the
capability ran for a particular run.

### 9.4 Runtime → Defaults and capture

Move sampling/runtime defaults here. Also place capture tier, privacy defaults, and guard configuration
here only when backed by real routes.

When privacy or adoption controls are not exposed, show one compact `Not configurable in Studio` group
with documentation, rather than two large disabled action cards with unknown costs.

### 9.5 Runtime → Snapshots

Use the standalone Snapshots layout rather than embedding it in the narrow Runtime column.

- Hero: `3 pinned · 3.3 MB` when total storage is reported or safely derived.
- Pin flow: select run → preview identity/storage/preconditions → explicit pin.
- Cards: readable run label first; run ID, checkpoint ID, and SHA secondary.
- Show child dependencies before offering unpin.
- Ensure the collection scrolls and remains usable at the target viewport.

## 10. Sessions and lineage

Sessions belong under Runs because they are a way to find and contextualize runs.

Replace raw session-ID rows with:

- First user message or a derived readable session title.
- Turn count and last activity.
- Model and warning summary.
- Branch/retry count.
- Session ID as secondary copyable metadata.

For a selected session, use an ordered turn timeline with branches:

```text
Turn 1 ── Turn 2 ── Turn 3 ── Turn 4
                    ├── retry A
                    └── corrected child
```

Selecting a turn opens that exact run in Inspect. Compatibility routing must preserve the selected
session and run rather than substitute the first available run.

## 11. Visualization decision table

| User question | Primary technique | Secondary/expert technique | Avoid |
|---|---|---|---|
| Which run needs attention? | Sorted ledger + status counts | Filtered latency distribution | Full evidence explanation per row |
| What reached the model? | Stage trace + composition bar | Exact metadata ledger | Byte/ID list as hero |
| Which passage affected the answer? | Ranked passage-to-answer links | Diverging matrix | Textless token IDs as primary axes |
| Is a claim supported? | Inline claim status + paired source excerpt | Claim evidence table | Empty six-state legend |
| Why might this have happened? | Finding card with evidence/limitation/action | Collapsed rule ledger | Flat rule-engine dump |
| Where was time spent? | Clock-owner waterfall | Rule/provenance table | Cross-process aligned offsets |
| What changed between runs? | Semantic groups and data-type-specific diffs | Raw field diff | Dumbbells for categories and JSON |
| How far is a finding from causal support? | One ordinal ladder | Detailed receipts | Five peer colors |
| How did confidence change? | Text-aligned line/small-multiple plot | Token table | Plot detached from output text |
| How is behavior configured? | Bipolar sliders + A/B preview | Full axis table | Ten identical neutral rows |
| Is the runtime ready? | Status summary + capability groups | Endpoint/source ledger | Large repeated absence cards |

## 12. Empty, blocked, and unavailable states

An empty state should occupy only the space required to answer:

1. What is absent?
2. Why is it absent?
3. Is this expected, blocked, or actionable?
4. Where can the user satisfy the prerequisite?

Examples:

- **Second model unavailable:** `Only one model is resident. Add or activate another model in Runtime →
  Models.`
- **Passage removal unavailable:** `This run retained influence metadata but not an exact removable span.
  Inspect the capture tier in Runtime → Defaults and capture.`
- **No experiment results:** `No recorded experiment result bundles. Run clozn experiment … or import a
  result bundle.`
- **Concept steering unavailable:** `J-lens is not configured for this model. See Runtime →
  Capabilities.`

Do not allocate a permanent full page to a single unavailable action.

## 13. Demonstration path

The redesigned product should support a coherent five-minute demonstration:

1. **Runs:** Open a low-confidence refund-policy run from Needs attention.
2. **Inspect / Read:** Read the prompt and answer; see two unverifiable claims and one likely instruction
   placement finding.
3. **Inspect / Evidence:** Select the refund claim and see the exact policy passage that supports or
   suppresses the answer span, including the measurement limitation.
4. **Compare:** Open a bounded retry and see the three meaningful changes before any raw fields.
5. **Improve:** Save the successful instruction change as a scoped correction and verify its state.

Timing can provide a second compact demo: open Performance and immediately see where 945ms was spent.

The demo should not require explaining a raw ID, field path, schema enum, or compatibility route unless
the viewer explicitly opens technical details.

## 14. Delivery sequence

### Phase 0: correctness and access

- Fix Sessions routing and scroll containment.
- Fix Runtime evidence-card overflow.
- Fix standalone Snapshots clipping.
- Make overlay state semantics explicit.

### Phase 1: readable shell

- Enforce typography floors.
- Default the inspector closed below 1,600px.
- Compact the run frame.
- Establish one hero and one primary conclusion per page.

### Phase 2: consolidate Inspect

- Merge What it received and What was sent into Input.
- Move Claims and Why summaries into Read.
- Merge What mattered and Mechanism into Evidence.
- Convert Second opinion and Without this passage into contextual actions.
- Preserve Timing, Replay, and Record as focused sections.

### Phase 3: semantic comparison

- Move findings above rows.
- Group rows by summary axis.
- Use visualization by data type.
- Integrate token alignment with output text.

### Phase 4: reorganize Improve and Runtime

- Move Runtime defaults to Runtime.
- Combine qualified behavior controls.
- Add profile bundle previews.
- Split Runtime into focused subpages and use the full-width Snapshots layout.

### Phase 5: remove compatibility ambiguity

- Redirect legacy routes visibly to their canonical destination.
- Preserve selected run, session, section, and object.
- Remove unreachable duplicate panels after the compatibility window.

## 15. Acceptance criteria

The redesign is successful when, at 1280px and 200% zoom where applicable, a developer can:

- Identify the primary conclusion of every page within five seconds.
- Read prompt, answer, findings, and limitations without zooming.
- Explain why each primary workspace exists and how it differs from its neighbors.
- See actual source and answer text before internal span or token IDs.
- Distinguish unavailable, not captured, not measured, failed, and below-floor evidence.
- Understand a run comparison without opening the raw diff.
- Use quantitative charts only where their axes and values are commensurate.
- Reach a relevant next action from the object it affects.
- Navigate sessions, snapshots, and long Runtime content without clipping.
- Complete the demonstration path without explaining backend implementation details.

The proposal does not relax Clozn's evidence boundaries. It changes the order and form in which those
boundaries are communicated so that honesty supports comprehension rather than competing with it.
