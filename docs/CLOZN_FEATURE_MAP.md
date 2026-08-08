# CLOZN capability and information map

**Code audit date:** 2026-08-02

This document inventories what CLOZN can record, derive, measure, execute, persist, and export. It is a
capability and information map rather than an interface specification.

## Evidence vocabulary

CLOZN produces several fundamentally different kinds of information:

- **Recorded:** Captured directly during a request or model execution.
- **Derived:** Deterministically computed from stored evidence without running the model again.
- **Measured:** Produced by scoring, ablation, replay, activation capture, or another controlled engine
  operation.
- **Evaluated:** Produced by a diagnostic rule, evaluator, classifier, or calibration model.
- **Generated:** Requires a new model execution.
- **Operational:** Runtime, worker, queue, model-loading, storage, or checkpoint state.
- **User-authored:** Feedback, annotations, corrections, profiles, experiment definitions, and decisions.

Unavailable evidence is represented explicitly. Missing measurement does not mean “no effect,” and a
failed measurement does not become a zero.

## 1. Prompt delivery and context evidence

CLOZN can reconstruct the progression from caller input to the exact model input.

### Delivered input

For every delivered message or source segment, CLOZN can retain:

- Stable content-derived segment ID
- Caller-supplied source ID
- Source type and label
- Message role
- Original ordering
- Delivered byte and token counts, when available
- Content SHA-256
- Redaction state
- Exact content, subject to receipt privacy settings

Callers can attach source identity separately from message text, so source metadata does not alter the
rendered prompt.

### Assembled input

CLOZN distinguishes delivered content from the content selected for assembly:

- Included delivered segments
- Assembly order
- Content hashes
- Propagated source identity
- Applied prompt policies
- Applied corrections
- Correction conflicts
- Structured transformations, when known
- Explicitly omitted segments and omission reasons

### Exact rendered prompt

CLOZN can retain:

- Exact final prompt text
- Rendered-prompt SHA-256
- Prompt byte count
- Prompt token count
- Chat-template fingerprint
- Rendered conversation structure
- Model/template preparation evidence
- Prompt boundaries used by checkpoint and continuation verification

### Context limits and termination

Delivery evidence can include:

- Prompt tokens
- Context-window capacity
- Reserved or requested output tokens
- Generated tokens
- Normalized finish reason
- Raw backend finish reason
- Finish-reason source

Normalized termination states include:

- EOS
- Maximum output tokens
- Context limit
- Tool call
- Client cancellation
- Worker error
- Unknown

Overlong requests are rejected explicitly instead of being silently trimmed by the normal generation
path.

### Prompt section manifest

CLOZN can decompose a prompt into stable semantic sections using:

- Explicit caller-provided section tags
- Native character ranges
- Markdown headings
- Document separators and markers
- Code fences
- XML-style wrappers
- Paragraph boundaries

Each section can include:

- Stable ID and name
- Source
- One or more prompt-coordinate parts
- Message index
- Start/end offsets
- Character count
- Preview
- Resolved text

The final user request is kept separate from surrounding context sections.

### Privacy levels

Context receipts support four independent privacy levels:

- `full`
- `metadata_only`
- `hashes_only`
- `off`

Changing receipt privacy supports preview, compare-and-swap application, and undo. Run-level redaction
and deletion are separate operations.

Primary implementations: [`context_receipt.py`](../clozn/runs/context_receipt.py),
[`sections.py`](../clozn/runs/sections.py), [`openai_compat.py`](../clozn/server/openai_compat.py), and
[`context_receipt.py` routes](../clozn/server/routes/context_receipt.py).

## 2. Stable text provenance and span addresses

CLOZN can identify and resolve text spans without relying on brittle display strings.

Addressable objects include:

- Delivered messages
- Rendered-prompt segments
- Attached source spans
- Answer spans
- Claims

Each address may carry:

- Run-scoped address ID
- Native artifact reference
- Relation key
- Unicode code-point offsets
- UTF-8 byte count
- Basis artifact SHA-256
- Span SHA-256
- Resolution state
- Parent/child lineage mappings

Resolution states include:

- Exact
- Metadata-only
- Drifted
- Redacted
- Unavailable

The span bridge verifies basis and span hashes before editing. It can:

- Resolve an address into real message coordinates
- Remove a span
- Replace it with neutral text
- Resolve every span belonging to a source
- Select a matched random control span
- Derive deterministic experiment seeds
- Carry addresses into descendant runs

Primary implementations: [`text_span_addresses.py`](../clozn/runs/text_span_addresses.py) and
[`span_bridge.py`](../clozn/replay/span_bridge.py).

## 3. Run records

A run is CLOZN’s central evidence object.

### Request and response

A run may contain:

- Delivered messages
- Assembled messages
- Exact rendered prompt
- Public response
- Separately emitted model reasoning content, when the backend provides it
- Structured-output contract
- Tool declarations and tool history
- Parsed tool call or structured result
- Raw parser input
- Parsing and validation outcome
- Finish reason and error
- Warnings and diagnostic flags

### Run identity

CLOZN records immutable execution identity:

- Absolute model path
- Exact model SHA-256
- Model file size
- Model name
- Quantization metadata
- Template fingerprint
- Engine build
- CLOZN version
- Adapter identity
- Machine identity
- Engine artifact identity
- Behavior-intervention identity

Identity comparison walks extensible facets rather than dropping unfamiliar fields.

### Timing and usage

A run may carry:

- Creation and recording time
- Total duration
- Prompt and generated token counts
- Model-load time
- Prefill time
- Decode time
- Per-token timing
- Queue wait
- Client-delivery timing
- Context allocation and pressure
- Runtime and hardware facts
- Throughput

### Relationships

Runs support:

- Parent and child links
- Retries
- Replays
- Branches
- Interventions
- Corrected children
- Exact execution forks
- Time-machine continuations
- Experiment-cell relationships
- Snapshot dependencies
- Session membership

### Storage integrity

Run metadata is indexed in SQLite. Larger evidence artifacts use content-addressed SHA-256 blobs. Digests
are verified when loading. Missing, unreadable, or corrupt evidence is surfaced as unavailable rather
than converted to empty data.

Primary implementations: [`store.py`](../clozn/runs/store.py), [`identity.py`](../clozn/runs/identity.py),
and [`trace.py`](../clozn/runs/trace.py).

## 4. Token-level generation evidence

With standard capture enabled, CLOZN can retain for each committed output token:

- Position
- Text or token piece
- Token ID
- Probability/confidence
- Log probability
- Top-k entropy
- Candidate alternatives
- Alternative token IDs, probabilities, and log probabilities
- Wall-clock timestamp
- Time since previous token
- Associated engine events
- Workspace/J-lens readouts

The trace can also preserve the backend’s richer committed generation step and distinguish public finish
reasons from raw engine reasons.

### Token workbench

For an individual token, CLOZN can synthesize:

- Token identity
- Confidence and entropy
- Alternatives
- Source-influence links
- Comparison evidence
- Layer/readout evidence
- Available and unavailable follow-up operations

Supported token actions are:

1. Fork at this token
2. Run a causal trace
3. Measure source influence
4. Run a mechanistic diff against a reference

Each action returns either a cached artifact, an asynchronous job, or a typed unavailable result.

Primary implementations: [`token_workbench.py`](../clozn/runs/token_workbench.py) and
[`token_workbench_actions.py`](../clozn/runs/token_workbench_actions.py).

## 5. Structured outputs and tool calls

CLOZN supports:

- Plain text
- JSON object output
- Strict JSON Schema output
- OpenAI-style function tools
- Native template-specific tool rendering/parsing
- Instruction-based fallback structured generation
- Tool-call argument validation
- Canonical JSON serialization
- Parser and runtime qualification evidence

Structured-output evidence can include:

- Requested contract
- Tool schema
- Parser selected
- Raw and sanitized parser input
- Native parser result
- JSON normalization
- Validation result
- Error code and message
- Public and substrate finish reasons
- Exact qualification state

The JSON Schema validator supports a bounded subset, including objects, arrays, scalar types, enums,
required properties, and nesting limits. Tool execution currently supports one active tool call rather
than parallel tool calls.

OpenAI-compatible streaming is supported; structured results are validated before final compatible
stream emission. Ollama-compatible chat and generation formats are also supported.

Primary implementations: [`structured_io.py`](../clozn/server/structured_io.py),
[`openai.py`](../clozn/server/routes/openai.py), and [`ollama.py`](../clozn/server/routes/ollama.py).

## 6. Source-to-answer influence

CLOZN can measure which supplied context spans affected which answer spans.

The principal influence-map method is teacher-forced scoring with source/context spans removed or
replaced. It is an intervention-based score, not an attention visualization.

### Influence-map data

An influence artifact can contain:

- Prompt sources
- Prompt spans
- Answer spans
- Exact offsets and hashes
- Baseline continuation score
- Context-span × answer-span matrix
- Per-link log-probability delta
- Absolute effect magnitude
- Supports/suppresses/neutral classification
- Measurement-floor result
- Evidence state
- Selected and unselected sources
- Measurement coverage
- Matrix completeness
- Failed span, if any
- Method and continuation identity
- Artifact and cache identity
- Timing and progress
- Redundancy controls

Evidence states distinguish:

- Causally supported
- Observed but below the measurement floor
- Delivered but unmeasured
- Omitted
- Unavailable
- Failed
- Inconclusive

Influence calculations support asynchronous execution, progress, cancellation, caching, and portable
metadata-only export.

### Section influence

For each prompt section, CLOZN can calculate:

- Total log-probability change when removed
- Per-token change
- Share of total measured absolute effect
- Whether any section clears the noise floor
- A bounded effect summary

A section can then be split into sentence- or line-level pieces and measured again within that section.
These shares are local to the selected parent section.

Primary implementations: [`context_answer_influence.py`](../clozn/receipts/context_answer_influence.py),
[`section_influence.py`](../clozn/server/routes/section_influence.py), and
[`section_drill.py`](../clozn/server/routes/section_drill.py).

## 7. Claims and source support

CLOZN deterministically segments answer text into addressable claims.

Claim categories include:

- Factual claim
- Recommendation
- Uncertainty statement
- Instruction or procedure
- Non-verifiable prose

Each claim can include:

- Exact answer offsets
- Text and basis hashes
- Stable span address
- Structural type such as sentence, list item, or code block
- Source-support links
- Support status
- Measurement method and limitations

Support statuses include:

- Supported
- Weakly supported
- Contradicted
- Unsupported by supplied materials
- Unverifiable from available evidence
- Measurement unavailable

A positive causal influence link is required for the strongest “supported” label. Textual overlap only
produces weak support. Contradiction detection is intentionally narrow, using evidence such as mismatched
numbers/dates or direct negation. “Unsupported” does not mean false; it only describes the supplied
materials.

An optional NLI evaluator can assess entailment against recorded evidence, with evaluator provenance
retained.

Primary implementations: [`claims.py`](../clozn/runs/claims.py) and
[`claim_support.py`](../clozn/runs/claim_support.py).

## 8. Diagnostics, warnings, and trust signals

### Direct generation signals

CLOZN detects:

- Worker and generation errors
- Token-limit truncation
- Empty responses
- Exact repetition loops
- Cyclic degeneration
- Malformed fenced JSON
- Low-confidence token regions
- High-entropy regions
- Close token decisions
- Digit or polarity forks
- Output-contract parse failures

Confidence spans can include:

- Strong/okay/shaky bands
- Mean and minimum confidence
- Token count
- Hesitation count
- Start/end positions

### Diagnostic findings

The diagnostic rule engine produces explicit `finding`, `not_observed`, `unavailable`, `pending`, or
`suppressed` states for:

- Input omitted or rejected
- Context-budget pressure
- Conflicting instructions
- Duplicate instructions
- Repeated source material
- Missing requested format
- Instructions far from the final request
- Source below measurement floor
- Source having little measured effect
- Final request conflicting with earlier instructions
- Output stopped by length
- Run-to-run identity or configuration drift

A finding may include:

- Severity
- Exact, pattern-derived, or inferred confidence class
- Evidence fields and spans
- Limitations
- Suggested follow-up action

### Performance attribution

Performance evidence supports phases with:

- Owner
- Clock domain
- Start and duration
- Measured versus estimated state
- Exclusive, overlapping, or context-only semantics
- Coverage and unaccounted duration

Named performance diagnoses include:

- Large context
- Slow decode relative to comparable runs
- Cold model load
- Client backpressure
- Gateway queue contention
- Adapter reload
- Memory pressure

Every rule reports fired, not fired, or unavailable.

Primary implementations: [`diagnosis.py`](../clozn/runs/diagnosis.py),
[`diagnosis_rules.py`](../clozn/runs/diagnosis_rules.py), and
[`perf_diagnosis.py`](../clozn/runs/perf_diagnosis.py).

## 9. Calibrated trust and selective answering

CLOZN can calculate:

- Reliability curves over organic runs
- Confidence-bin sample sizes
- Mean/minimum-confidence failure signatures
- Entropy and low-confidence fractions
- Prompt-class drift
- Brier score
- Expected calibration error
- Risk-coverage curves
- Area under the risk-coverage curve
- Temperature-scaled confidence
- Small-sample caveats
- Ask/answer/abstain policy outcomes

Calibration can be gated to an exact model identity and aggregate. Trust labels retain provenance and
sample-size limitations.

OpenAI-compatible responses can optionally include:

- Raw confidence spans
- Calibrated ask/abstain metadata
- Persisted policy verdict
- Clarification or abstention text
- Original raw answer and policy provenance

Primary implementations: [`actuary.py`](../clozn/runs/actuary.py),
[`calibrated_trust.py`](../clozn/runs/calibrated_trust.py), and
[`calibration.py`](../clozn/eval/calibration.py).

## 10. Run comparison

CLOZN can compare two runs across several independent axes.

### Identity and configuration

- Model hash and path
- Adapter
- Template fingerprint
- Engine build
- Machine
- Behavior interventions
- Temperature
- Top-p and top-k
- Repetition penalty
- No-repeat n-gram
- Maximum tokens
- Seed
- Context size
- Sampler mode
- Stop conditions

### Context delivery

- Delivered, assembled, and omitted segments
- Added and removed content
- Reordering
- Content-hash changes
- Source-label changes
- Rendered-prompt hash
- Limit changes
- Exact prompt equality

### Output

- Finish reason
- Response length
- Text similarity
- Token diff
- Longest common prefix
- First divergence
- Per-position confidence
- Whether one run’s chosen token appeared in the other run’s alternatives
- Output-contract and parse differences

### Controlled change tests

CLOZN can test bounded context, template, or sampling changes against criteria such as:

- Exact output
- Tool parse success
- Finish reason
- Token budget

Arms report planned, available, unavailable, or completed states.

### Second-model opinion

A stored run can be used as an anchor and compared with a new execution from another already-ready model.
Compatibility checks include:

- Chat template
- Context limit
- Tools and schema contract
- Qualified evidence availability

Cross-model log probabilities are not treated as directly comparable.

Primary implementations: [`run_diff.py`](../clozn/analysis/run_diff.py),
[`controlled.py`](../clozn/replay/controlled.py), and
[`second_opinion.py`](../clozn/runs/second_opinion.py).

## 11. Sessions and lineage

Sessions are persisted opaque identities that group runs across clients.

Session information includes:

- Session ID/key
- Optional title
- Client key
- Visible or hidden metadata
- Ordered turns
- Branches and retries
- Aggregate duration and token use
- Turn-level diagnostics
- Adjacent-turn configuration and context changes

The session trace derives:

- Linear turn history
- Branch children
- New, dropped, or carried context segments
- Settings drift
- First diagnostic finding
- First failed or cancelled turn
- First configuration change

Run family data supports complete parent-child lineage and dependency checks for snapshots and deletion.

Primary implementations: [`sessions.py`](../clozn/runs/sessions.py) and
[`session_trace.py`](../clozn/runs/session_trace.py).

## 12. White-box and mechanistic evidence

When the native engine and model architecture expose the required hooks, CLOZN supports substantially
deeper analysis.

### Teacher-forced scoring

The engine can score an exact continuation against:

- Prompt text or exact prompt tokens
- Continuation text or exact continuation tokens
- Steering vectors
- Residual writes
- Head writes
- FFN writes
- Attention-edge knockout

Results can include per-token log probability, top-k candidates, total log probability, and explicit
missing-capture fields.

### Activation capture

Available evidence can include:

- Layer × token residual norms
- Residual vectors
- Layer means
- Per-head capture norms
- Full head rows
- FFN rows
- J-lens token candidates
- Concept/direction decomposition
- Reconstruction cosine
- Explained variance
- Residual norm
- Top concept labels or words

Hooks are capability-reported and architecture-gated. Residual, attention-head, and FFN writes have
different valid layer ranges.

### Mechanistic diff

For pair-compatible models, CLOZN can compare the same teacher-forced continuation across models:

- Per-token rank and log probability
- Layer capture availability
- Residual cosine similarity
- Normalized L2 difference
- Layer-change summaries
- Optional tensor references

Exact tokenizer compatibility is required for per-token comparison. Template, architecture, hidden size,
layer count, vocabulary, and head count are checked independently.

### Causal trace

CLOZN can:

- Select a target token
- Screen candidate layers and directions
- Apply directional ablations
- Run noise-floor controls
- Measure solo and combined deltas
- Construct candidate paths and edges
- Report accounting and control verdicts

### Attention provenance

Attention-edge knockout can test dependence on selected source positions. This requires compatible
attention hooks and flash attention to be disabled.

### Activation transplant

A reference model’s activation can be transplanted into a candidate model at a residual, head, or FFN
site. The harness includes:

- Reference arm
- Candidate self-transplant/no-op arm
- Random equal-norm control
- Target log-probability and rank
- NLL movement
- Greedy-suffix restoration
- Structured/assertion restoration
- Instrument sanity
- Reference-specific effect analysis

### Causal bisect

CLOZN can search coarse-to-fine across layer windows, heads, and FFN sites. Verdicts distinguish:

- Localized individual site
- Localized window
- Distributed restoration
- Inconclusive
- Unavailable

Primary implementations: [`clozn_engine.py`](../engine/client/clozn_engine.py),
[`routes_whitebox.cpp`](../engine/core/serve/routes_whitebox.cpp),
[`mechanistic_diff.py`](../clozn/analysis/mechanistic_diff.py), [`tracer.py`](../clozn/analysis/tracer.py),
[`transplant.py`](../clozn/analysis/transplant.py), and [`causal_bisect.py`](../clozn/analysis/causal_bisect.py).

## 13. Replay, branching, and intervention

### Ordinary replay

A recorded run can be re-executed with explicit changes to:

- Prompt instructions
- Included or excluded sections
- Specific text spans
- Sampling
- Greedy mode
- Context
- Template
- Behavior dials
- Output budget

The child records its parent and exact applied changes.

### Token forks

CLOZN can force:

- A recorded alternative token
- An exact token ID
- A branch at a selected token position

Two execution regimes are distinguished:

- Exact checkpoint continuation
- Reconstructed text splice with retokenization differences

### Exact execution forks

When eligible, CLOZN records:

- Parent fingerprint
- Runtime, worker, model, and adapter identity
- Checkpoint reference
- Intervention
- Sampler restoration
- Unchanged-control proof
- Exactness proof
- Child lineage
- Execution receipt

### Checkpoints and snapshots

Checkpoint data can include:

- Token history
- Prompt/generated boundary
- RNG and sampler state
- Temperature, top-k/top-p, repetition penalty
- Steering state
- Worker generation identity
- Lifecycle and size

Checkpoints can be exported into hashed durable envelopes, pinned, imported, resolved, listed, and
dependency-safely unpinned.

### Time Machine

CLOZN supports:

- Turn extraction
- Per-turn exact eligibility
- Prompt-boundary verification
- Exact same-prompt child replay
- Alternate-user branch
- Appended-turn continuation
- Snapshot/live/durable checkpoint sources
- Sampler and worker proof
- Typed unavailable and failed states

It does not silently downgrade an invalid exact continuation into ordinary text replay.

Primary implementations: [`replay.py`](../clozn/replay/replay.py),
[`execution_fork.py`](../clozn/replay/execution_fork.py),
[`checkpoint_pin_store.py`](../clozn/replay/checkpoint_pin_store.py),
[`timetravel.py`](../clozn/replay/timetravel.py), and
[`time_machine_continuation.py`](../clozn/replay/time_machine_continuation.py).

## 14. Controlled investigation experiments

CLOZN can run bounded “did this matter?” experiments using:

- Removal of a stable text span
- Neutral replacement of a stable text span
- Omission of an attached source
- A sampler-setting change

Where possible, the experiment contains:

- Baseline arm
- Treatment arm
- Matched random or no-op control
- Pinned seed
- Eligibility and refusal evidence
- Instrument-sanity result
- Whether treatment moved
- Whether control moved
- Whether the effect was specific
- Observed differences
- Separately licensed causal statement

A causal claim is only licensed when the control structure supports effect specificity. Adapter-scale
experiments are recognized by the contract but currently return a typed unsupported result.

Experiments run asynchronously and support progress and cancellation.

Primary implementations: [`investigation_experiment.py`](../clozn/runs/investigation_experiment.py) and
[`investigation_experiment.py` receipts](../clozn/receipts/investigation_experiment.py).

## 15. Causal receipts and answer explanations

CLOZN supports several explanation artifacts with different cost and rigor.

### Leave-one-out receipts

For an active dial or prompt section, CLOZN can compare:

- Greedy execution with the influence
- Greedy execution without it
- Text and metric change
- Whether the answer changed
- Applied ablation
- Child runs

A pairwise redundancy guard checks whether two individually non-load-bearing influences matter together.

### Teacher-forced receipts

A faster alternative scores the stored answer:

- With the influence
- Without the influence
- Per-token log-probability deltas
- Total and mean effect
- Most dependent tokens
- Noise-floor control
- Caveat and method provenance

### Arbitrary span receipts

A selected text span can be ablated and compared against a matched-length filler control.

### Coalition credit

For multiple influences, CLOZN can calculate:

- Solo effects
- Pair effects
- Joint effect
- Interaction gaps
- Exact Shapley values for bounded small sets
- Second-order Shapley–Taylor approximation for larger sets
- Per-influence interaction confidence intervals
- Approximation and coverage notes

### Concept swap receipts

A concept/direction can be swapped and evaluated against effect and coherence controls.

### Accountable narration

CLOZN can generate:

- A constrained explanation assembled from citable evidence
- Receipt IDs supporting the explanation
- A separate unconstrained self-report
- A diff between self-report claims and actual receipts
- Unsupported-claim flags

Self-report reliability categories include:

- Faithful credit
- Confabulated credit
- Unattributed claim
- Missed driver
- Correct silence

These categories assess agreement with recorded causal evidence; they do not prove introspective access.

Primary implementations: [`core.py`](../clozn/receipts/core.py),
[`forced.py`](../clozn/receipts/forced.py), [`span_receipt.py`](../clozn/receipts/span_receipt.py),
[`coalition.py`](../clozn/receipts/coalition.py), [`narrate.py`](../clozn/receipts/narrate.py), and
[`self_report_reliability.py`](../clozn/receipts/self_report_reliability.py).

## 16. Behavior controls, corrections, and feedback

### Behavior profiles

Profiles can contain:

- Named dials
- Custom dial values
- Profile-supplied facts
- Active profile identity

Profiles support save, switch, import, export, and delete. Older profile bundles may still carry a
`response_policies` field from the retired durable-correction system below; it is preserved verbatim
on load/save but never applied to a generation.

### Corrective actions (one-shot, request-local)

Built-in actions include:

- Less verbose
- More concrete
- Use context
- Ask before guessing
- Preserve formatting
- Stop repeating

Actions have:

- Label and description
- Conflicts
- Scope (always `once` -- see "Durable corrections were retired" below)
- Available backend
- Evaluation metrics
- Fallback
- Fingerprint
- Expiration

A preview/confirm generates a matched baseline and corrected child; "keeping" a result selects the
corrected child as that one run's own revision. Nothing here persists beyond the run it was generated
from -- see [CAPABILITIES.md](CAPABILITIES.md) for the request-local vs. durable distinction.

### Durable corrections were retired

CLOZN used to let a correction be drafted, confirmed, scoped to a session/client/model/project, and
then auto-applied to every future matching request until explicitly disabled or deleted -- "Teach
Once." That entire durable, auto-applying lifecycle (draft/confirm/enable/disable/delete/undo/verify
/export/applicability-resolution/event-ledger, plus session/profile-scoped corrective retries) was
removed: see [CAPABILITIES.md](CAPABILITIES.md)'s Removed/Retired section for what changed and why.

`GET`/`POST /corrections` and `/corrections/*` now return a typed HTTP 410
(`durable_corrections_retired`) rather than a plain 404, so an old caller can tell "this used to work
and no longer applies" apart from a route miss. Studio no longer exposes a Corrections/Teach Once
surface. Runs recorded before the retirement may still carry `applied_corrections`/
`correction_conflicts` receipt fields or `corrective_retry.scope` metadata; those remain readable as
historical evidence but are never produced by, or re-applied through, a new generation.

### Feedback and preferences

Run-linked feedback can record:

- Kind
- Dial
- Direction
- Metadata

Feedback is aggregated into pending preference proposals. Proposals require explicit approval or
dismissal.

Primary implementations: [`registry.py`](../clozn/behavior/registry.py),
[`store.py`](../clozn/profiles/store.py), and [`preferences.py`](../clozn/behavior/preferences.py).

## 17. Guarded generation

CLOZN supports an opt-in detect-and-correct generation loop for calibrated concepts.

A guard specification can include:

- Concepts
- Activation thresholds
- Layer
- Top-k
- Counter-steering strength
- Maximum correction count
- Chunk size

For each concept, evidence can contain:

- Calibration state
- Threshold and layer source
- Trigger token IDs or pieces
- Maximum activation
- Fire positions
- Pre/post activation
- Counter direction and strength
- Whether counter-steering was applied
- Application error
- Maximum-fires state
- Trace-capture state
- Coherence note

The guard generates in chunks, inspects J-lens readouts, discards a triggering chunk, and regenerates it
with calibrated counter-steering. Uncalibrated concepts are annotation-only.

This currently applies to non-streaming OpenAI-compatible generation and requires compatible J-lens
support.

Primary implementation: [`generation_guard.py`](../clozn/server/generation_guard.py).

## 18. Experiments, evaluation, and Model CI

### Experiment definitions

Experiments support:

- Case × variant × seed matrices
- Target and guard suites
- Base, tuned, quantized, prompt, and dial variants
- Message inputs
- Assertions
- Exact manifest digest
- Suite fingerprint

Each cell can contain:

- Coordinate
- Pass/fail/error/unscored state
- Run ID
- Evidence
- Assertions
- Comparisons
- Replay class

### Statistics

CLOZN can calculate:

- Seed-collapsed case outcomes
- Pass rates
- Paired case deltas
- Deterministic paired-bootstrap confidence intervals
- Bonferroni-adjusted alpha
- Instability counts
- Identity-compatible history and trends

Replay classes include bit-identical greedy, re-prefilled, sampled stochastic, and unknown.

### Regression promotion

A result can be promoted into a regression suite using:

- Preview
- Secret/sensitive-text scanning
- Redaction
- Exact destination and hash
- Drift-safe application

### Evaluation

Supported evaluation data includes:

- Exact-match grading
- Numeric grading
- Multiple-choice grading
- Arithmetic and curated probes
- Golden result save/diff
- Calibration metrics
- Selective policy metrics
- Risk/coverage analysis
- Quantization checks
- Adapter validation and equivalence receipts

### CI and interoperability

CLOZN can produce:

- GitHub Actions workflow configuration
- Baseline and check reports
- Deterministic exit codes
- Metadata-only evidence bundles
- EEE/Open Evaluation exports
- Promptfoo exports
- Inspect exports
- Lighteval exports
- Export validation

Primary implementations: [`experiment.py`](../clozn/experiments/experiment.py),
[`stats.py`](../clozn/experiments/stats.py), [`promotion.py`](../clozn/experiments/promotion.py), and
[`ci_check.py`](../clozn/cli/commands/ci_check.py).

## 19. Models, routing, runtime, and qualification

### Local model inventory

CLOZN can enumerate local GGUF models with:

- Path and filename
- Size
- Best-effort quantization tag
- Cached SHA-256
- GGUF header information
- Resource-fit estimate

It supports model download, lockfile verification, and exact-hash fetching.

### Managed routing

Routing evidence can include:

- Literal requested model
- Routing policy
- Canonical resolved model
- Artifact and runtime key
- Adapter
- Worker
- Model hash
- Context limit
- Backend
- Build
- Template
- Capabilities
- Typed mismatch or load failure

Worker lifecycle states include:

- Unloaded
- Loading
- Ready
- Evicting
- Failed

The router supports ready-worker selection, queue limits, timeouts, cancellation, identity handshakes, and
typed cold-load outcomes.

### Model qualification

Qualification can record:

- GGUF architecture
- Hidden size
- Layer count
- Vocabulary
- Tokenizer
- Template
- Quantization
- File size and SHA
- Resource plan
- Deterministic generation smoke test
- Context receipt
- Performance check
- Structured I/O qualification
- White-box qualification
- Dial qualification
- J-lens qualification
- Acceptance batteries

Every qualification step reports its own passed, failed, unavailable, or skipped state.

### Engine management

Native engine setup supports:

- Platform-matched artifact selection
- Checksum verification
- Install/status
- Upgrade
- Rollback
- Supervisor process listing
- Stop/restart
- Smoke testing

Primary implementations: [`inventory.py`](../clozn/models/inventory.py),
[`model_routing.py`](../clozn/server/model_routing.py),
[`pipeline.py`](../clozn/qualification/pipeline.py), and [`install.py`](../clozn/setup/install.py).

## 20. Privacy, retention, sharing, and export

### Run privacy

CLOZN supports:

- Literal-string redaction across inline run payloads
- Full tombstone redaction
- Removal of trace and influence references
- Dependency-aware deletion
- Optional descendant cascade
- Retention planning by count or age
- Dry-run retention
- Broken-lineage reporting
- Separate orphaned-blob garbage collection

### Export formats

Supported evidence exports include:

- Context receipt
- Influence map
- OpenTelemetry/OpenInference JSONL
- Run evidence bundle
- Trace
- Receipt bundle
- Tensor arrays
- Manifest with hashes
- Verification notebook
- CI metadata bundle
- Correction ledger
- Experiment/community formats

Open evidence bundles can be hash-verified offline. Generated notebooks reconstruct evidence and can
optionally attempt live reproduction against the exact model hash.

### Sharing

CLOZN can create:

- Receipt permalinks
- Optional response receipt footer
- Metadata-limited receipt views
- Run cards
- Evidence summaries

Primary implementations: [`mutations.py`](../clozn/runs/mutations.py),
[`telemetry.py`](../clozn/runs/telemetry.py), [`bundle_export.py`](../clozn/runs/bundle_export.py), and
[`gc.py`](../clozn/runs/gc.py).

## 21. Integrations and compatibility

CLOZN exposes:

- Native CLOZN generation API
- OpenAI-compatible chat completions and model listing
- OpenAI-compatible streaming
- Ollama-compatible chat, generation, tags, and version
- Aider connector
- Open WebUI connector
- OpenAI environment configuration
- GitHub Actions integration

Connector operations support:

- Detection
- Plan
- Dry run
- Application
- Backup
- Drift detection
- Undo

Ollama adoption can discover an existing local model, resolve its underlying GGUF artifact, register or
copy it without downloading it again, qualify it, and optionally configure connectors.

Primary implementations: [`_connector.py`](../clozn/cli/commands/_connector.py),
[`adopt.py`](../clozn/cli/commands/adopt.py),
[`ollama_discovery.py`](../clozn/adopt/ollama_discovery.py), and
[`ollama_resolver.py`](../clozn/adopt/ollama_resolver.py).

## 22. Capture and availability boundaries

- `light` capture retains text, finish state, and metadata without a stored per-token trace.
- `standard` capture retains the token trace and is the normal evidence-rich tier.
- `deep` and `lab` are accepted and recorded capture labels, but generation capture currently stores the
  same trace class as `standard`; they do not automatically persist raw activations or SAE artifacts.
- Raw activation evidence is acquired through explicit white-box operations.
- Influence, causal trace, transplants, and causal bisect require a compatible live model/runtime.
- Exact execution forks and appended-turn continuations require a valid compatible checkpoint.
- Cross-model token comparison requires exact tokenizer compatibility.
- Attention knockout requires the appropriate hooks and non-flash attention.
- Missing or unavailable evidence remains distinct from a negative measurement.
- Heuristic findings, evaluator results, measured interventions, and recorded facts retain separate
  provenance.
- Some measurements are cached; others create new child runs or asynchronous jobs.
- Mutating operations generally use previews, fingerprints, compare-and-swap checks, or explicit
  confirmation.

This inventory defines the design inputs: the central data objects, the information available on each,
the analyses CLOZN can produce, the actions users can take, and the conditions under which each artifact
is trustworthy.
