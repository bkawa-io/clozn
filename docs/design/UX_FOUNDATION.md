# CLOZN UX foundation

**Status:** decision draft

**Capability source:** [CLOZN feature map](../CLOZN_FEATURE_MAP.md)

This document defines the product information architecture and interaction model that should organize
CLOZN's capabilities. It is intentionally upstream of visual styling and component implementation.

## 1. Product model

CLOZN is a local-LLM debugging instrument organized around a selected run. The main investigation loop
is:

1. **Find** a run that deserves attention.
2. **Understand** what was delivered, what the model produced, and what evidence is available.
3. **Locate** the suspicious source, section, claim, span, token, event, or runtime phase.
4. **Test** one bounded hypothesis through measurement, replay, or intervention.
5. **Compare** the source run with the resulting child or reference run.
6. **Generalize** a useful result into an evaluation case, regression suite, profile, or correction.

The selected run is the durable center of this loop. Evidence views change how the run is inspected; they
do not silently replace it with another run.

## 2. Primary users and jobs

The primary user is a technically capable developer running or integrating local language models. They
may understand prompts, model settings, and application code without being a mechanistic-interpretability
specialist.

The principal jobs are:

- Find failed, truncated, slow, unstable, or otherwise suspicious runs.
- Determine exactly what reached the model.
- Separate delivery/context problems from generation/model problems.
- Inspect response claims, token uncertainty, tool calls, and structured-output failures.
- Determine whether supplied context affected a particular answer span.
- Compare a run with a retry, branch, model, adapter, prompt, or sampler variant.
- Test a specific causal hypothesis with controls.
- Replay or branch from a reproducible point.
- Turn a successful fix into a durable correction or regression case.
- Inspect runtime/model identity and evidence availability.

## 3. Core objects

CLOZN's navigation and selection model should use these objects directly:

- **Run:** one recorded model execution and its evidence.
- **Session:** an ordered conversation plus branches and retries.
- **Input object:** message, delivered segment, source, prompt section, rendered-prompt span, correction, or
  omission.
- **Output object:** claim, answer span, token, tool call, structured result, or finish marker.
- **Evidence object:** warning, diagnostic finding, influence link, receipt, event, timing phase, readout,
  activation site, or comparison delta.
- **Relationship:** parent/child, replay, retry, branch, intervention, baseline/candidate, or experiment
  membership.
- **Model/runtime:** model artifact, adapter, template, worker, machine, capabilities, and qualification.
- **Experiment:** suite, case, variant, seed, result cell, assertion, and promotion target.

The current selection can therefore be expressed as:

`workspace + run or run pair + selected object + evidence representation`

This state should be durable in the URL. Changing workspaces must preserve the run and selected object
when the destination supports them.

## 4. Evidence depth

The feature map naturally forms four levels of disclosure.

### Level 1: recorded facts

Available without new model work:

- Request, messages, sources, prompt, and response
- Delivery and assembly receipt
- Model/runtime identity
- Sampling and limits
- Tool/structured-output result
- Finish reason and errors
- Token trace, when captured
- Timing and recorded events
- Lineage and session position

This is the default investigation surface.

### Level 2: deterministic interpretation

Computed from stored evidence without a model execution:

- Claim boundaries
- Confidence and entropy regions
- Degeneration and truncation signals
- Context-pressure and prompt diagnostics
- Performance aggregation
- Run and session diffs
- Evidence availability

This level should guide attention while retaining its derived provenance.

### Level 3: measured evidence

Requires scoring, ablation, replay, or a controlled experiment:

- Source-to-answer influence
- Section influence and sentence drill-down
- Claim support measurement
- Leave-one-out and forced-score receipts
- Controlled span/source/sampler experiments
- Attention knockout
- Second-model opinion

These operations need explicit method, cost, progress, controls, and measurement-floor states.

### Level 4: forensic evidence

Requires compatible white-box engine capabilities:

- Token workbench actions
- J-lens and concept readouts
- Residual, head, and FFN evidence
- Mechanistic diff
- Causal trace
- Activation transplant
- Causal bisect
- Exact checkpoint fork

This level receives its own workspace because its spatial and conceptual needs are different from ordinary
run debugging.

## 5. Workspace architecture

### Runs

Purpose: find and triage work.

Contains:

- Historical and live run modes
- Search and structured filters
- Sessions
- Status, finish reason, warnings, model, duration, and time
- Saved investigative views such as errors, truncated, slow, low-confidence, corrected, and unreviewed
- Keyboard selection and open-to-side behavior

Selecting a run opens it in Debug without losing the list's filters or scroll position.

### Debug

Purpose: understand one run and decide what to inspect or test next.

Contains:

- Persistent run frame
- Prompt and output side-by-side
- Delivery receipt and exact rendered prompt
- Claims, sources, confidence, and diagnostics
- Tool and structured-output evidence
- Performance and runtime facts
- Session and lineage context
- Selection-driven inspector
- Evidence deck
- Retry, branch, measurement, annotation, export, and promotion actions

This is the default workspace for a selected run.

### Compare

Purpose: understand a controlled difference between two runs.

Contains:

- Persistent baseline and candidate
- Identity and configuration delta
- Prompt, context, and delivery diff
- Output and token diff
- Finish, warning, evaluation, latency, and token deltas
- Parent/child and intervention provenance
- Controlled change tests
- Second-model opinion
- Promotion into an evaluation or regression case

Identical metadata is hidden by default. The comparison should foreground what changed and whether the
evidence licenses attributing the result to that change.

### Forensic

Purpose: investigate one selected source, span, claim, token, layer, or mechanism at full depth.

Contains:

- Token workbench
- Source and section influence detail
- Causal receipts
- Layer/readout evidence
- Mechanistic diff
- Causal trace
- Attention provenance
- Activation transplant and causal bisect
- Exact token fork

The entry selection remains visible, and the workspace provides a direct return to the corresponding
prompt or response position in Debug.

### Evaluate

Purpose: generalize from individual investigations to repeatable model-quality decisions.

Contains:

- Experiment matrices
- Suites, cases, variants, and seeds
- Assertions and result evidence
- Paired statistics and uncertainty
- History and trends
- Calibration and selective-answer policies
- Regression promotion
- CI configuration and evidence bundles

### Runtime

Purpose: manage the environment that makes evidence possible.

Contains:

- Local model inventory
- Model and adapter identity
- Worker lifecycle and queue state
- Runtime capabilities
- Qualification results
- Capture tier and receipt privacy defaults
- Engine install, upgrade, rollback, and smoke evidence
- Behavior profiles and guard calibration

Runtime is a supporting utility area rather than another way of inspecting the selected run.

## 6. The persistent run frame

Debug, Compare, and Forensic share a compact run frame containing:

- Human-readable run title
- Status and finish reason
- Model, adapter, quantization, and template identity
- Time and duration
- Prompt/generated token counts
- Important warnings
- Session and parent/child position
- Current comparison or intervention relationship
- Copyable full run ID as secondary metadata

The run frame never falls back to a different run while loading. If an artifact or representation is not
available, the frame remains anchored and the unavailable state belongs to that artifact.

## 7. Debug workspace frame

The default desktop frame borrows the spatial logic of media-editing tools: a browser/bin, a central
arrangement, a selection inspector, and a lower detail/evidence deck.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Run frame · identity · warnings · lineage · run actions                     │
├──────────────┬───────────────────────────────────────┬───────────────────────┤
│ Run browser  │ Main stage                            │ Selection inspector   │
│              │ ┌─────────────────┬─────────────────┐ │                       │
│ history      │ │ Prompt/context  │ Output          │ │ selected source,     │
│ sessions     │ │                 │                 │ │ section, claim,      │
│ filters      │ │ delivery view   │ readable answer │ │ token, finding,      │
│              │ │ exact prompt    │ overlays        │ │ event, or phase      │
│              │ └─────────────────┴─────────────────┘ │                       │
├──────────────┴───────────────────────────────────────┴───────────────────────┤
│ Evidence deck · synchronized lanes · events · timing · lineage · jobs       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Main stage

Prompt/context and output are side-by-side at desktop widths. They are the stable reading surfaces; other
evidence changes their representation or overlays rather than replacing them.

The prompt/context side has three primary representations:

1. **Conversation:** readable system, user, tool, and source messages.
2. **Delivery:** requested → delivered → assembled → rendered, including omitted material, corrections,
   transformations, limits, and termination evidence.
3. **Rendered:** exact model-facing prompt with section and source boundaries.

Delivery is a primary representation, not an advanced disclosure area.

The output side supports:

- Readable response
- Claim boundaries
- Source/support overlay
- Confidence/shakiness overlay
- Influence overlay
- Tool/structured-output representation
- Finish/truncation marker

Selecting an object on either side synchronizes the inspector and evidence deck.

### Selection inspector

The inspector answers four questions for the selected object:

1. What is selected?
2. What is known about it?
3. What evidence supports that information?
4. What can be done next?

It may show exact address, hashes, provenance, measurement method, limitations, linked objects, and
contextual actions. It should not become a permanent catalog of unrelated run metadata.

### Evidence deck

The lower deck holds representations that benefit from horizontal space or synchronized position:

- Confidence and entropy lanes
- Source support/influence lanes
- Semantic events
- Finish and truncation markers
- Runtime waterfall
- Session and lineage strip
- Background job progress

Selecting a position in a lane selects the associated token/span/object above. The deck is resizable,
collapsible, and can temporarily become the dominant surface.

## 8. Delivery representation

Delivery should be treated as a trace with distinct stages:

`requested → delivered → assembled → rendered → model limits → termination`

Each source or message behaves like a selectable clip traveling through that trace. A segment can be:

- Delivered and assembled
- Delivered but omitted
- Transformed
- Corrected
- Redacted or metadata-only
- Included in the rendered prompt
- Selected for measurement
- Measured above or below the effect floor
- Linked to answer spans

The representation must keep three statements separate:

- **It reached the server.**
- **It reached the model.**
- **It measurably affected this answer.**

Clicking a segment should reveal its exact content or privacy state, hashes, offsets, assembly status,
omission reason, prompt location, influence state, linked answer spans, and available measurements.

## 9. Evidence and availability language

Every evidence-bearing object exposes both provenance and availability.

### Provenance

- Recorded
- Derived
- Measured
- Evaluated
- User-authored

### Availability

- **Available:** the artifact exists and can be inspected.
- **Not captured:** the run completed without the required evidence.
- **Not measured:** the run is eligible, but the measurement has not been run.
- **Computing:** an asynchronous artifact is in progress.
- **Failed:** an attempted artifact has an error receipt.
- **Unsupported:** the runtime/model cannot perform the operation.
- **Privacy-limited:** content is intentionally unavailable while metadata remains.

Only `Not measured` should normally offer an immediate measurement action. The action should state its
method, approximate cost, model/runtime requirement, and whether it creates a new run.

## 10. Action model

Run-changing actions follow one interaction contract:

1. Start from a persisted source run.
2. Select the evidence object or condition to change.
3. Preview one explicit intervention.
4. Show what remains locked.
5. Execute a child run or controlled measurement.
6. Open the result as source-versus-child comparison.
7. Keep, undo, annotate, or promote the result.

Primary actions include:

- Measure source or section influence
- Verify claim support
- Retry with one change
- Remove or replace a span
- Omit a source
- Change a sampler setting
- Fork at a token
- Replay or continue from a checkpoint
- Run a second model
- Open a causal or mechanistic test
- Promote to evaluation/regression
- Apply or retain a correction
- Export or share evidence

Actions are attached to the object they affect. A global command surface may expose the same actions for
experts, but it should resolve to the identical preview and receipt flow.

## 11. Density and typography

Density comes from allocation of space, compact controls, synchronized views, and fast disclosure—not
from reducing meaningful text below reading size.

- Prompt and response prose: **15–16px**, approximately 1.5 line height.
- Tables, structured evidence, and inspector values: **13–14px**.
- Timestamps, compact badges, and tertiary labels: **12px minimum**.
- Monospace is reserved for exact prompts, code, IDs, hashes, tokens, and tabular measurements.
- Readable response text uses an editorial face or the system sans rather than machine metadata styling.
- Compact and comfortable density modes change padding and row height, not type size.

## 12. Responsive behavior

### 1,600px and above

- Run browser, side-by-side stage, inspector, and evidence deck may all remain visible.
- A comparison or reference pane may be pinned temporarily.

### 1,000–1,599px

- Prompt and output remain side-by-side.
- Run browser and inspector become independently collapsible drawers.
- Evidence deck remains available at the bottom and can expand.

### Below 1,000px

- Run browser becomes a drawer.
- Inspector becomes a drawer or bottom sheet.
- Prompt/output may use a two-tab representation when each column would fall below a readable measure.
- The selected run, selected object, and evidence state remain unchanged while switching representations.

At 200% zoom, the product may collapse panels but must not shrink type or require simultaneous horizontal
scrolling of the two primary reading surfaces.

## 13. Navigation rules

- Selecting another workspace preserves the selected run.
- Selecting another run resets only selections that cannot exist on the new run.
- Compare requires an explicit baseline and candidate; it never silently chooses one.
- Forensic opens with the originating source/span/claim/token selection preserved.
- Returning from Forensic restores the exact position in Debug.
- Back/forward navigation restores run, workspace, representation, selected object, filters, and relevant
  scroll position.
- An unavailable representation does not substitute another run or synthetic data.
- Sessions and lineages are relationships among runs, not separate copies of run evidence.

## 14. Feature placement summary

| Capability family | Primary workspace | Secondary access |
|---|---|---|
| Run history, watch, sessions, filters | Runs | Debug run browser |
| Prompt, response, delivery receipt, claims, warnings | Debug | Compare |
| Source/section influence | Debug | Forensic |
| Diagnostics, performance, output contracts, tools | Debug | Compare |
| Parent/child, retry, replay, time machine | Debug | Compare |
| Run, prompt, context, output, and identity diff | Compare | Debug lineage |
| Token workbench and causal receipts | Forensic | Debug inspector |
| J-lens, mechanistic diff, trace, transplant, bisect | Forensic | Compare |
| Experiment matrices, calibration, regression, CI | Evaluate | Compare promotion |
| Models, workers, qualification, capture, guard/profile setup | Runtime | Run frame and inspector |
| Corrections and bounded retries | Debug | Compare and Runtime profile setup |
| Privacy, export, sharing | Object-level actions | Runtime defaults |

## 15. First prototype scope

The first prototype should cover one evidence-rich run in Debug and validate the architecture before
expanding other workspaces.

Required structure:

- Persistent run frame
- Collapsible run browser
- Prompt/context and output side-by-side
- Conversation, Delivery, and Rendered input representations
- Readable response plus one overlay at a time
- Selection inspector
- Resizable evidence deck
- Durable selection in the URL

Required selection paths:

- Source/segment → delivery evidence → linked prompt region → linked answer spans
- Claim → source support → influence measurement availability
- Token → confidence/entropy → alternatives → Forensic entry
- Warning/finding → exact evidence → suggested measurement or bounded action
- Parent/child relationship → Compare

Required unavailable states:

- Delivery receipt not captured
- Token trace not captured
- Source evidence not measured
- Measurement unsupported
- Artifact failed
- Privacy-limited content

The prototype should be validated at approximately 1,000px, 1,440px, and 1,920px widths, plus 200% zoom.

## 16. Success criteria

The architecture succeeds when a developer can:

- Identify what reached the model without searching through unrelated metadata.
- Read prompt and output comfortably at the same time.
- Distinguish delivery from measured influence.
- Select any highlighted answer object and understand its provenance.
- Find the deepest available evidence without knowing CLOZN's internal feature names.
- See immediately when evidence was not captured, not measured, failed, or unsupported.
- Test one hypothesis and understand exactly what changed.
- Compare the result without losing the source-run context.
- Promote a useful result into repeatable evaluation work.

