"""generation_guard.py -- closed-loop disposition guardrails, an OPT-IN, DEFAULT-OFF generation mode
(FRONTIER_BETS section 9.1 / experiment A1.1, docs/RESEARCH_ROADMAP.md's A1.1 row + docs/PRODUCT_ROADMAP.md's
R3 lane).

===============================================================================================
HONESTY LAW -- read this before touching ANY user-facing string in this module. It governs every label.
===============================================================================================
A1.1's verdict was INCONCLUSIVE on its own headline thesis, not a clean win. What LIVES, measured on a
20-prompt banned-topic battery (Qwen3.5-9B, J-lens layer 16, counter_strength -0.5):
  * catch-rate 100% (10/10 banned-content cases caught)
  * false-positive-rate 5% (1/20 clean prompts flagged)
What FAILED: the "intent-BEFORE-speech" lead-time thesis. median_lead_time_tokens == 0, and only 4/10
cases showed ANY positive lead. The lens flags the guarded disposition AT the token where it is realized,
NOT before it. So this is a PRESENT-TENSE DETECT-AND-CORRECT guard, not a predictive/precognitive one.

Every user-facing string, docstring, and receipt field in this module MUST say "detects and corrects
during generation" (or an equivalent present-tense framing) and MUST NOT claim lead-time, prediction,
"before it's said", or "acts on intent". GUARD_CAVEAT below is the one canonical copy of this framing --
every surface that reports on a firing guard reuses it verbatim rather than re-deriving its own wording.
This mirrors the house rule that killed every predictive-interior claim (docs/RESEARCH_ROADMAP.md's Killed
list: "White-box risk controller advantage" -- the deployed selective-generation score turned out
bit-identical to a black-box logprob; internal state added nothing there either).

===============================================================================================
2026-07-23 UPDATE: the 9B defaults do not transfer -- two model-specific assumptions, now fixed
===============================================================================================
Running scripts/calibration/guard_signal_calibrate.py against Qwen2.5-7B found the guard's SIGNAL works,
but two of its defaults were silently A1.1/9B-specific (receipt: runs/experiments/guard_signal_qwen2.5-7b
.json):
  1. DEFAULT_LAYER=16 is not even a FITTED J-lens layer on every model (this 7B's fitted layers are
     [2, 14, 21, 25] -- polling layer 16 is a hard 400, not a degraded reading). See LAYER SELECTION below.
  2. The concept-WORD token (e.g. " violence") is the wrong signal: the lens top-k at these layers shows
     immediate NEXT-tokens (format/verb candidates), not abstract concept words -- " violence" essentially
     never appears. A TRIGGER-TOKEN SET (concrete words the model would actually emit while drifting toward
     the concept, e.g. " kill", " knife", " blood", " stab", " gun" for "violence") DOES appear: on a small
     6-banned/6-clean battery at layer 14, a trigger surfaced in 100% of banned generations and 0% of clean
     ones. See SIGNAL below.

===============================================================================================
PER-MODEL CALIBRATION (mirrors concept_dir.py's own per-model concept-dial calibration exactly)
===============================================================================================
`load_guard_calibration(model_sha256)` reads ~/.clozn/models/<sha256>/guard_threshold_calibration.json --
written by scripts/calibration/guard_signal_calibrate.py's live sweep, schema
"clozn.guard_threshold_calibration.v1": {model_sha256, default_layer, concepts: {name: {layer, threshold,
trigger_ids, trigger_pieces, catch, fp, n_battery, note}}}. Missing/unreadable/wrong-schema file -> None,
never a crash -- calibration is optional enrichment, mirroring concept_dir.load_concept_dial_calibration's
missing-file=no-op discipline exactly (same ~/.clozn/models/<sha>/ per-exact-GGUF scoping convention
clozn.server.app._model_scoped_path also uses).

SIGNAL: a guarded concept found in the calibration (AT THE LAYER THIS REQUEST IS ACTUALLY POLLING -- see
LAYER SELECTION) is polled as its TRIGGER-TOKEN SET, not the bare concept word: concept_activation takes
the MAX lens-logit score over every trigger id present in one readout position's top-k. A concept NOT in
the calibration (or calibrated at a DIFFERENT layer than this request polls) falls back to the OLD
concept-WORD-token signal with a placeholder threshold -- see UNCALIBRATED CONCEPTS DO NOT FIRE below for
why that fallback can still ANNOTATE but never CORRECT.

===============================================================================================
LAYER SELECTION -- stop hardcoding 16
===============================================================================================
A hardcoded hardcoded default layer 400s on any model whose fitted J-lens layers don't include it (exactly
what broke on the 7B). `resolve_guard_layer` picks ONE layer for the whole guarded generation, in this
precedence:
  1. `spec['layer']` -- the CALLER explicitly set it. Honored as-is, no second-guessing: an explicit
     request for an invalid layer still surfaces the engine's own 400 (this is a deliberate request for a
     SPECIFIC layer, not a default that needs validating).
  2. the matched calibration's `default_layer`, IF the engine actually reports that layer as available
     right now (`engine_jlens_layers`, straight off /health's `jlens.layers` -- the one source of truth for
     "is this layer even fitted on this engine").
  3. the AVAILABLE layer numerically nearest to the legacy default (DEFAULT_LAYER=16) -- `_nearest_layer`.
  4. (None, "unavailable") when the engine reports no J-lens layers at all -- the caller must fail closed;
     there is no hardcoded fallback left that could silently 400.
`layer_source` ("explicit" | "calibration_default" | "discovered_valid") rides the receipt so a caller can
always tell which rule actually picked the layer.

===============================================================================================
UNCALIBRATED CONCEPTS DO NOT FIRE (documented decision, per BK's stated lean)
===============================================================================================
A concept with no matching calibration entry (at the polled layer) is still POLLED and ANNOTATED in the
receipt (`calibrated: false`, its observed activation reported) -- but it can NEVER trigger a correction.
Reasoning: /jlens's score is a raw, uncalibrated lens logit (routes_jlens.cpp's own docstring: "score = the
raw lens logit"); DEFAULT_THRESHOLD=0.0 is a placeholder with no measured meaning on an arbitrary model/
layer/token. Acting on "score >= 0.0" for a concept nobody has ever measured a real catch/FP rate for would
be closer to noise than to a safety mechanism, and firing corrections on noise is worse than not firing at
all (spurious re-steers degrade fluency for no protective benefit -- see the coherence note). So an
uncalibrated concept is EXCLUDED from `correctable_concepts` unconditionally: annotate, never correct.
Calibrate it (scripts/calibration/guard_signal_calibrate.py) to make it correctable.

===============================================================================================
TOPK FLOOR -- a trigger outside the polled top-k is an invisible, SILENT miss
===============================================================================================
scripts/calibration/guard_signal_calibrate.py measured its separation at topk=64. A trigger token that
would have appeared at, say, rank 70 of the true distribution is simply never seen by a poll asking for
only the top 8 -- concept_activation reports None ("not observed"), which reads as an honest "clean so
far" but is actually a silent MISS caused by too narrow a window, not a real absence. `resolve_guard_topk`
enforces a floor: any request with at least one CALIBRATED concept polls at >= CALIBRATION_TOPK_FLOOR (64,
matching the calibration tool's own default), regardless of what `spec['topk']` asked for; a per-concept
"topk" key in the calibration file (not present in today's artifact, but read defensively for forward
compatibility) raises the floor further if larger.

===============================================================================================
MECHANISM (the loop itself, unchanged in shape)
===============================================================================================
Every engine primitive here already ships: /jlens (a deterministic linear read), /harvest, and /intervene
(steer_vec + coef, the same wire dir(c) already rides -- concept_dir.ConceptSteer.steer_toward). What's new
is the LOOP: the engine steers a WHOLE generation call; it has no notion of "steer only after token 40". So
this module is a GATEWAY control loop over the engine, not an engine feature:
  1. Generate a short CHUNK of tokens (chunk_tokens, default 24) with no steering.
  2. Poll: read the J-lens disposition toward every guarded concept's trigger set over the text generated
     so far, at the ONE resolved poll layer.
  3. If a CORRECTABLE (calibrated) concept's disposition crosses its OWN threshold, DISCARD that chunk and
     regenerate it with the corrective dir(c) counter-direction (concept_dir.ConceptSteer.steer_toward)
     injected via /intervene's steer_vec, at the SAME token budget.
  4. Continue from the (possibly corrected) chunk. Repeat until `max_tokens` is produced or the
     `max_fires` re-steer cap is reached (see RE-STEER CAP below).

`run_guarded_generation` below is the pure, engine-agnostic control loop (generate_chunk/
read_disposition/build_counter are injected callables) -- fully unit-testable with fakes, no socket, no
GPU. `guarded_chat_completion` is the thin production adapter that wires it to a real EngineSubstrate's
engine client + a fresh concept_dir.ConceptSteer; it is exercised in tests/test_generation_guard_server.py
via a fake substrate/engine (never a live engine), matching this codebase's "live path deferred, the
pure/wiring logic is model-free tested" split (see clozn.server.generation_gateway's
selective_generation_action, or scripts/calibration/concept_dial_autocalibrate.py's own sweep).

===============================================================================================
RE-STEER CAP
===============================================================================================
`max_fires` (default 3) bounds how many chunks this loop will ever re-steer. Once the cap is spent, the
guard STOPS watching for the remainder of the generation (no more polling, no more correction) and the
receipt says so plainly (`cap_reached: true`) -- never an infinite re-steer loop, and never a silent claim
that the rest of the output was checked when it wasn't.

===============================================================================================
FAIL-CLOSED DECISION (documented here, not just in code)
===============================================================================================
If a CORRECTABLE (calibrated) concept cannot be resolved to a working dir(c) (no unembed/engine support, an
unfitted J-lens layer, ...), this module REFUSES the request outright -- it does not silently generate an
unguarded reply, even flagged. Rationale: `clozn_guard` is a safety request ("please steer this generation
away from X"); a caller who explicitly asked for that guarantee and silently got ordinary, unguarded
generation back (even with a flag buried in the metadata) may never notice the flag and may treat the reply
as guarded when it never was -- exactly the "silent pass" this feature exists to prevent. Refusing outright
(a clear 4xx with a stated reason) is the safer default. This mirrors selective_generation_action's
fail-closed pattern (calibration backlog #10) but goes one step further: there, the safe fallback
(annotate-only) was itself harmless; here, the "fallback" would be exactly the ungated content the guard
was supposed to prevent, so refusing is the only honest option.

An UNCALIBRATED concept that can't even be tokenized for annotation is NOT treated this strictly: since it
was never going to fire a correction anyway (see UNCALIBRATED CONCEPTS DO NOT FIRE), a resolution failure
there degrades that ONE concept's annotation quietly (still reported in the receipt, with a note explaining
why it has no trigger ids) rather than refusing the whole request over a concept that carried no safety
promise to begin with. Also refused outright: no valid J-lens layer at all on this engine (LAYER SELECTION
step 4), and `clozn_guard` together with `stream: true` (see SCOPE LIMITS).

===============================================================================================
2026-07-25 UPDATE: a firing guard's counter-injection can fail MID-GENERATION -- degrade honestly,
never crash, never refuse an in-flight reply
===============================================================================================
A hostile pressure pass found a gap THIS section's own check doesn't close: resolve_guard_signals's
fail-closed gate (above) calls concept_steer.compute() to prove a calibrated concept's UNIT dir(c) can be
built (J_l + W_U resolve) -- it does NOT call steer_toward(), which ALSO needs a per-model/per-layer
median-residual-norm (concept_dir.py's own VALIDATED_MEDIAN_RESID_NORM / concept_dial_calibration.json) to
scale that unit vector into a real `coef`. So a concept can pass setup clean and still fail the FIRST time
it actually fires, when build_counter() finally calls steer_toward() and hits ConceptSteer.steer_toward's
own "no_norm_calibration" block. Confirmed live: Qwen2.5-7B-Instruct Q4_K_M's guard-threshold calibration
fires "violence" at layer 14 (guard_threshold_calibration.json), but neither VALIDATED_MEDIAN_RESID_NORM
(only 16/21/25) nor a per-model concept_dial_calibration.json (none exists at all under this model's
~/.clozn/models/<sha>/ as of this pass) covers layer 14 on this model -- so every real fire on this model/
concept hits this exact gap. It closes the moment someone runs scripts/calibration/
concept_dial_autocalibrate.py (--layers 14, or the model's other fitted layers) against a live engine with
this model loaded: it measures median_resid_norm (and a usable dir(c) scale range) per layer and writes
~/.clozn/models/<model_sha256>/concept_dial_calibration.json via concept_dir.save_concept_dial_calibration
-- ConceptSteer reads it automatically the next time it's constructed, no code change needed.

Before this update, that failure was an uncaught RuntimeError inside build_counter(), propagating out of
run_guarded_generation and guarded_chat_completion into clozn.server.app's top-level dispatcher guard
(commit 509dc2b) as a bare 500 -- the WHOLE reply lost (already-generated chunks discarded) over one
concept's missing SCALE calibration, even though its dir(c) DIRECTION was perfectly fine.

RESOLUTION: this is a MID-GENERATION failure, not a setup-time one -- by the time it happens, real tokens
already exist, so refusing the whole request is no longer free the way it is at setup. GUARD_CAP_NOTE
already establishes this module's answer to "the guard can't do everything it promised for the rest of
this reply": say so honestly and keep going, never silently and never by crashing. run_guarded_generation
now catches build_counter()'s exception PER FIRE: the fire is still recorded (never dropped from the
receipt), with `counter_applied: false` and `counter_error` set to the real underlying message, and the
flagged chunk rides UNCORRECTED into the reply exactly as it would without a guard at all -- the caller
learns the guard SAW the content and could NOT counter-steer it, and why. This does NOT contradict
FAIL-CLOSED DECISION's rationale above: that section's concern is a SILENT pass (a caller believing they
got a guarded reply when they actually got an ordinary one, with nothing telling them so);
`counter_applied: false` is the opposite of silent. FAIL-CLOSED DECISION's outright-refusal still governs
every SETUP-time resolution failure (dir(c) itself unbuildable, no valid layer at all) -- those refuse
before any token is generated, where refusing costs nothing; this is the one narrow case that cannot be
caught at setup (see above) and must be handled once generation is already in flight.

===============================================================================================
SCOPE LIMITS OF THIS FIRST CUT (said honestly, not silently)
===============================================================================================
  * STREAMING IS DEFERRED. Guarded generation only supports the non-streaming
    POST /v1/chat/completions path; requesting `clozn_guard` together with `stream: true` is refused
    (fail-closed for the same reason as above -- a streamed reply's early tokens are already delivered to
    the client before any correction could happen, so silently ignoring the guard on a streaming request
    would be exactly the silent pass this module exists to prevent).
  * NOT YET COMPOSED with prompt-card memory, tone-dial steering, corrective-retry policies, or structured
    output -- a guarded generation renders the chat messages through the model's own template
    (clozn.server.app._engine_tmpl, the same renderer chat() uses) and generates directly against the raw
    engine client, bypassing that machinery entirely for this first cut. Composing them is future work,
    not silently dropped.
  * Coherence is a light, cheap NOTE (clozn.replay.counterfactual._coherence's pure text-counting check,
    over the corrected chunks only), not a hard gate -- see `_coherence_note`.
  * The calibration itself is a SMALL-BATTERY presence separation (6 banned + 6 clean prompts on Qwen2.5-
    7B) -- a real, honest signal-design fix, but not a public reliability claim at any scale. GUARD_CAVEAT
    says so explicitly; re-run scripts/calibration/guard_signal_calibrate.py with a larger battery before
    treating a "calibrated" concept's catch/fp numbers as anything more than "this small battery separated
    cleanly."
  * A calibration entry measured at layer L is applied ONLY when this request is actually polling layer L
    (see resolve_concept_signal) -- never misapplied to a different layer's readout, even when a
    calibration file exists for the concept. This can, in principle, leave one concept calibrated and
    another uncalibrated within the SAME request if their calibrated layers differ from the resolved poll
    layer or from each other; this module polls exactly ONE layer per request (no per-concept multi-layer
    polling yet) -- a documented limitation, not a silent one.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

# =================================================================================================
# THE HONEST CAVEAT -- the one canonical copy. See the HONESTY LAW section above before editing this.
# =================================================================================================
GUARD_CAVEAT = (
    "closed-loop disposition guardrail (FRONTIER_BETS section 9.1 / experiment A1.1). This guard "
    "detects and corrects during generation -- present tense only: it reads a guarded concept's "
    "disposition at the token where it is realized and steers the continuation away from it, never "
    "earlier. A1.1's measured numbers (20-prompt banned-topic battery, Qwen3.5-9B, J-lens layer 16, "
    "counter_strength -0.5): catch-rate 100% (10/10 banned-content cases caught), false-positive-rate "
    "5% (1/20 clean prompts flagged). Its foresight thesis FAILED: median_lead_time_tokens = 0 and only "
    "4/10 cases showed any positive lead -- so this is not a foreknowledge mechanism and must never be "
    "described as one. Per-concept calibration (when present -- see the receipt's 'calibrated' field) is "
    "a SMALL-BATTERY presence separation (e.g. 6 banned + 6 clean prompts), a real signal-design result, "
    "NOT a public reliability claim at any scale; an uncalibrated concept only annotates and never fires "
    "a correction."
)

GUARD_CAP_NOTE = (
    "guard cap reached: the re-steer budget (max_fires) was spent before generation finished, so the "
    "remainder of this reply was produced WITHOUT further guard monitoring or correction -- it is "
    "honest, not silent, about that gap."
)

UNCALIBRATED_NOTE = (
    "no per-model calibration for this concept at the layer this request is polling -- falling back to "
    "the concept-WORD token as the signal with a placeholder threshold (0.0 on a raw, uncalibrated lens "
    "logit is not a meaningful cutoff). DECISION: an uncalibrated concept never fires a correction -- it "
    "is annotated (its observed activation is still reported) but cannot trigger a re-steer, since acting "
    "on an arbitrary threshold over a raw logit would be closer to noise than signal. Run "
    "scripts/calibration/guard_signal_calibrate.py against this exact model to calibrate it."
)

# See the module docstring's 2026-07-25 "counter-injection can fail MID-GENERATION" section: a calibrated
# concept's dir(c) DIRECTION can build fine at setup while its SCALE (median-residual-norm) is missing for
# this layer -- discovered only when build_counter() actually calls steer_toward() for a real fire. The
# fire is still recorded (never dropped); this canned note is the RECEIPT-LEVEL flag that at least one fire
# below has counter_applied: false -- the per-fire counter_error carries the real, specific message.
GUARD_COUNTER_FAILURE_NOTE = (
    "at least one fire below could not be counter-steered: the guard detected a guarded concept's "
    "disposition crossing its calibrated threshold, but building the corrective counter-direction failed "
    "for that specific fire (see each fire's own 'counter_error') -- that flagged chunk was NOT corrected "
    "and rode into the reply exactly as originally generated. This is a per-fire degrade, not a guard "
    "failure or a silent pass: the request was not refused and the reply was not aborted; run "
    "scripts/calibration/concept_dial_autocalibrate.py against this model/layer to close the gap (see the "
    "module docstring)."
)

# -- opt-in wiring -----------------------------------------------------------------------------------
GUARD_FIELD = "clozn_guard"                 # request-body extension field (an object, not a boolean)
GUARD_SETTING = "generation_guard"          # server-wide default spec (clozn.memory.mode's settings store)

# -- defaults, documented as placeholders where calibration doesn't exist yet ------------------------
DEFAULT_COUNTER_STRENGTH = -0.5             # A1.1's own validated counter_strength (steers AWAY -- negative)
DEFAULT_MAX_FIRES = 3
DEFAULT_LAYER = 16                          # the LEGACY target for nearest-valid-layer discovery only --
                                            # NEVER used directly as a poll layer anymore; see LAYER SELECTION.
DEFAULT_CHUNK_TOKENS = 24
DEFAULT_TOPK = 8
# See the module docstring's TOPK FLOOR section: guard_signal_calibrate.py measured its separation at
# topk=64, so any request with at least one calibrated concept polls at least this wide, regardless of
# `spec['topk']` -- a narrower window would silently miss a real trigger sitting just outside it.
CALIBRATION_TOPK_FLOOR = 64
# /jlens's "score" is the RAW LENS LOGIT (engine/core/serve/routes_jlens.cpp's own docstring: "score = the
# raw lens logit"), not a probability -- there is no universal safe cutoff. 0.0 is an UNCALIBRATED
# placeholder (the neutral point where the readout turns positive), shown for an uncalibrated concept's
# annotation only -- it is NEVER used to decide whether an uncalibrated concept fires (see
# UNCALIBRATED CONCEPTS DO NOT FIRE). Calibrate per model/layer via scripts/calibration/
# guard_signal_calibrate.py to get a real, measured threshold.
DEFAULT_THRESHOLD = 0.0

GUARD_CALIBRATION_SCHEMA = "clozn.guard_threshold_calibration.v1"


# =================================================================================================
# opt-in spec parsing (mirrors generation_gateway.selective_generation_enabled's precedence exactly)
# =================================================================================================

def _normalize_guard_spec(raw: Any) -> Optional[dict]:
    """Validate one `clozn_guard` (or server-default) payload into a fully-defaulted spec dict, or None
    when there is genuinely nothing to guard (no concepts -- "empty" is OFF, not an error, per the opt-in
    contract). Raises ValueError on a STRUCTURALLY malformed non-empty value (wrong types) -- an explicit,
    broken guard request should fail loudly (400), not be silently ignored; that silence is exactly what
    this whole feature exists to avoid.

    `spec['layer']` is None unless the CALLER explicitly set it (no default filled in here anymore --
    see the module docstring's LAYER SELECTION section for why a bare hardcoded default is exactly the
    bug this fixes). `spec['threshold']` keeps its old default and is used ONLY as the displayed/
    annotation threshold for an UNCALIBRATED concept -- a calibrated concept always uses its OWN
    calibrated threshold, never this value (see UNCALIBRATED CONCEPTS DO NOT FIRE)."""
    if not isinstance(raw, Mapping):
        raise ValueError("clozn_guard must be an object")
    concepts_raw = raw.get("concepts")
    if concepts_raw is None:
        concepts_raw = []
    if not isinstance(concepts_raw, (list, tuple)):
        raise ValueError("clozn_guard.concepts must be a list of concept words")
    concepts = []
    for c in concepts_raw:
        if isinstance(c, bool) or not isinstance(c, str) or not c.strip():
            raise ValueError("clozn_guard.concepts must be a list of non-empty strings")
        concepts.append(c.strip())
    if not concepts:
        return None   # nothing to guard against -- equivalent to off, not a misconfiguration

    def _num(key, default):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"clozn_guard.{key} must be a number")
        return float(value)

    def _pos_int(key, default):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"clozn_guard.{key} must be a positive integer")
        return int(value)

    layer_raw = raw.get("layer")
    if layer_raw is not None:
        if isinstance(layer_raw, bool) or not isinstance(layer_raw, int):
            raise ValueError("clozn_guard.layer must be an integer")
        layer_raw = int(layer_raw)

    return {
        "concepts": concepts,
        "threshold": _num("threshold", DEFAULT_THRESHOLD),
        "counter_strength": _num("counter_strength", DEFAULT_COUNTER_STRENGTH),
        "max_fires": _pos_int("max_fires", DEFAULT_MAX_FIRES),
        "layer": layer_raw,   # None unless the caller set it explicitly -- see resolve_guard_layer
        "chunk_tokens": _pos_int("chunk_tokens", DEFAULT_CHUNK_TOKENS),
        "topk": _pos_int("topk", DEFAULT_TOPK),
    }


def parse_guard_spec(body: Any) -> Optional[dict]:
    """The request's guard spec, or None (OFF -- byte-identical to today). An explicit `clozn_guard` on
    the request always wins (including an explicit falsy value, which means "opted out" even when the
    server default is on); only when the request omits the field entirely does the server-wide
    `generation_guard` setting (GUARD_SETTING, clozn.memory.mode's generic settings store) apply. Raises
    ValueError on a structurally malformed explicit value (see _normalize_guard_spec) -- callers should
    turn that into an HTTP 400, never swallow it."""
    if isinstance(body, Mapping) and GUARD_FIELD in body:
        raw = body.get(GUARD_FIELD)
        return _normalize_guard_spec(raw if raw else {})
    try:
        from clozn.memory import mode as memory_mode
        saved = memory_mode.get_setting(GUARD_SETTING, None)
    except Exception:
        saved = None
    if not saved:
        return None
    return _normalize_guard_spec(saved)


# =================================================================================================
# PERSISTED SERVER-WIDE DEFAULT -- clozn.server.routes.guard's GET/POST /guard/mode reads and writes
# through these two functions. Added so the guard has a toggle that STICKS (previously GUARD_SETTING was
# only ever read here, never written by any HTTP route -- see studio/app/behavior.mjs's prior "declared
# skeleton" note). Nothing about parse_guard_spec's own precedence changes: an explicit per-request
# `clozn_guard` field on /v1/chat/completions ALWAYS wins over whatever is persisted here, including an
# explicit falsy value meaning "opted out this one call even though the server default is on" -- this
# persisted default only ever governs a request that says nothing about clozn_guard at all.
# =================================================================================================

def get_persisted_guard_spec() -> Optional[dict]:
    """The currently persisted server-wide guard default, fully normalized (same shape parse_guard_spec
    returns for a request-level spec), or None when off/absent/corrupt. A corrupt or stale persisted value
    (e.g. hand-edited settings file) degrades to None -- treated as off -- rather than raising, mirroring
    clozn.memory.mode's own never-raise-on-load discipline; the only place a malformed guard spec is ever
    surfaced as an error is set_persisted_guard_spec, at the moment someone tries to WRITE it."""
    try:
        from clozn.memory import mode as memory_mode
        saved = memory_mode.get_setting(GUARD_SETTING, None)
    except Exception:
        saved = None
    if not saved:
        return None
    try:
        return _normalize_guard_spec(saved)
    except ValueError:
        return None


def set_persisted_guard_spec(raw: Any) -> Optional[dict]:
    """Validate and persist the server-wide guard default via clozn.memory.mode.set_setting -- which
    writes through clozn._io.atomic_write_json (temp-file-then-rename), never a bare open().write() that
    could truncate the settings file on a bad value (see clozn._io's own module docstring for the bug this
    closes). `raw` uses the exact same shape as a request's `clozn_guard` field (concepts + optional
    threshold/counter_strength/max_fires/layer/chunk_tokens/topk) -- this function invents no new knobs,
    it validates through the SAME _normalize_guard_spec every live per-request clozn_guard value goes
    through. An empty/falsy `raw` (or one with no concepts) persists OFF (None), matching the opt-in
    contract "empty is off, not an error" documented on _normalize_guard_spec.

    Raises ValueError on a structurally malformed non-empty value (bad types, e.g. concepts not a list) --
    the caller (clozn.server.routes.guard) must turn that into an HTTP 400, never swallow it, exactly like
    parse_guard_spec's own contract for a live request. Returns the normalized spec that was persisted
    (None when turned off)."""
    from clozn.memory import mode as memory_mode
    spec = _normalize_guard_spec(raw if raw else {})
    memory_mode.set_setting(GUARD_SETTING, spec)
    return spec


# =================================================================================================
# per-model guard-threshold calibration (mirrors concept_dir.py's per-model concept-dial calibration)
# =================================================================================================

def guard_calibration_path(model_sha256: Optional[str] = None) -> str:
    """~/.clozn/models/<model_sha256>/guard_threshold_calibration.json -- the SAME per-exact-GGUF
    ~/.clozn/models/<sha>/ scoping convention clozn.server.app._model_scoped_path uses (and
    concept_dir.concept_calibration_path mirrors for the concept dial's own per-model calibration). An
    explicit argument rather than a live-substrate read, so this stays testable/reusable outside a
    request context, exactly like concept_dir's equivalent."""
    base = os.path.join(os.path.expanduser("~"), ".clozn")
    if model_sha256:
        return os.path.join(base, "models", str(model_sha256), "guard_threshold_calibration.json")
    return os.path.join(base, "guard_threshold_calibration.json")


def load_guard_calibration(model_sha256: Optional[str], *, path: Optional[str] = None) -> Optional[dict]:
    """This exact model's guard-threshold calibration (scripts/calibration/guard_signal_calibrate.py's
    artifact), or None when there is nothing usable to read -- missing file, unreadable JSON, wrong/
    missing schema, or no concept entries at all. NEVER raises -- calibration is optional enrichment;
    every caller must fall back to the uncalibrated (annotate-only) path exactly as if this file didn't
    exist, mirroring concept_dir.load_concept_dial_calibration's own missing-file=no-op discipline.

    Returns {"model_sha256": str | None, "default_layer": int | None,
    "concepts": {name: {"layer", "threshold", "trigger_ids": list[int], "trigger_pieces": list[str],
    "catch", "fp", "n_battery", "note", "topk": int | None}}}, or None. A malformed individual concept
    entry is skipped (not fatal to the rest of the file); a malformed `default_layer` degrades to None
    (LAYER SELECTION falls through to layer discovery) rather than poisoning the whole read."""
    p = path or guard_calibration_path(model_sha256)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or raw.get("schema_version") != GUARD_CALIBRATION_SCHEMA:
            return None
        concepts = raw.get("concepts")
        if not isinstance(concepts, dict) or not concepts:
            return None
        out_concepts: dict = {}
        for name, entry in concepts.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                continue
            layer = entry.get("layer")
            threshold = entry.get("threshold")
            trigger_ids_raw = entry.get("trigger_ids")
            if (isinstance(layer, bool) or not isinstance(layer, int)
                    or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                    or not isinstance(trigger_ids_raw, list) or not trigger_ids_raw):
                continue
            clean_ids = [int(tid) for tid in trigger_ids_raw
                        if not isinstance(tid, bool) and isinstance(tid, int)]
            if not clean_ids:
                continue
            topk_raw = entry.get("topk")
            out_concepts[name] = {
                "layer": int(layer), "threshold": float(threshold), "trigger_ids": clean_ids,
                "trigger_pieces": [str(x) for x in (entry.get("trigger_pieces") or [])],
                "catch": entry.get("catch"), "fp": entry.get("fp"),
                "n_battery": entry.get("n_battery"), "note": entry.get("note"),
                # Not present in today's artifact (guard_signal_calibrate.py doesn't write one) -- read
                # defensively for forward compatibility; resolve_guard_topk treats a missing/invalid value
                # as "no extra floor beyond CALIBRATION_TOPK_FLOOR" rather than raising.
                "topk": int(topk_raw) if isinstance(topk_raw, int) and not isinstance(topk_raw, bool) else None,
            }
        if not out_concepts:
            return None
        default_layer = raw.get("default_layer")
        if isinstance(default_layer, bool) or not isinstance(default_layer, int):
            default_layer = None
        return {"model_sha256": raw.get("model_sha256"), "default_layer": default_layer,
               "concepts": out_concepts}
    except Exception:
        return None


# =================================================================================================
# layer selection -- see the module docstring's LAYER SELECTION section
# =================================================================================================

def engine_jlens_layers(engine) -> list:
    """The J-lens layers this engine actually has fitted/loaded, straight off /health's `jlens.layers` --
    the ONE source of truth for "is layer N even valid to poll on this engine" (a stale hardcoded default
    silently 400s otherwise -- exactly what broke moving from the 9B to the 7B). Never raises: a down
    engine, or a build with no jlens block at all, reports [] (no valid layers), which the caller must
    treat as guard-unavailable."""
    try:
        health = engine.health() if hasattr(engine, "health") else {}
    except Exception:
        return []
    jl = (health or {}).get("jlens") or {}
    try:
        return sorted({int(x) for x in (jl.get("layers") or [])})
    except Exception:
        return []


def _nearest_layer(preferred: int, available: list) -> Optional[int]:
    """The AVAILABLE layer numerically closest to `preferred` (ties broken toward the smaller/lower layer,
    deterministic) -- pure, no I/O. None when `available` is empty."""
    if not available:
        return None
    return min(available, key=lambda layer: (abs(layer - preferred), layer))


def resolve_guard_layer(spec: dict, calibration: Optional[dict],
                        available_layers: list) -> tuple[Optional[int], str]:
    """The ONE J-lens layer this guarded generation polls, (layer, source) -- see the module docstring's
    LAYER SELECTION section for the full precedence + rationale. `source` is
    "explicit" | "calibration_default" | "discovered_valid", always surfaced on the receipt so a caller
    can tell which rule actually picked the layer. Returns (None, "unavailable") when nothing above yields
    a valid layer -- the caller must refuse the request; there is no hardcoded fallback left."""
    if spec.get("layer") is not None:
        return int(spec["layer"]), "explicit"
    if (calibration and calibration.get("default_layer") is not None
            and calibration["default_layer"] in available_layers):
        return int(calibration["default_layer"]), "calibration_default"
    nearest = _nearest_layer(DEFAULT_LAYER, available_layers)
    if nearest is not None:
        return nearest, "discovered_valid"
    return None, "unavailable"


# =================================================================================================
# per-concept signal resolution -- trigger SET (calibrated) vs concept-word fallback (uncalibrated)
# =================================================================================================

def resolve_concept_signal(concept: str, calibration: Optional[dict], poll_layer: int,
                           fallback_threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Per-concept polling signal: which trigger token ids to watch for, at what threshold, and whether
    this concept is CALIBRATED for the layer THIS request is actually polling (see the module docstring's
    LAYER SELECTION + UNCALIBRATED CONCEPTS DO NOT FIRE sections). A calibration entry measured at a
    DIFFERENT layer than `poll_layer` is NEVER applied -- scores from different layers/readouts are not
    comparable -- and is reported uncalibrated with a note explaining why, rather than silently misapplied.

    Returns {"calibrated": bool, "threshold": float, "trigger_ids": set[int] | None,
    "trigger_pieces": list[str] | None, "topk": int | None, "note": str | None, ...calibration
    provenance fields ("catch"/"fp"/"n_battery"/"calibration_note") when calibrated}. `trigger_ids` is
    None until a concrete token id is resolved (the caller fills it in for the uncalibrated fallback via
    ConceptSteer.resolve_token_id -- see resolve_guard_signals)."""
    entry = (calibration or {}).get("concepts", {}).get(concept) if calibration else None
    if entry is not None and int(entry["layer"]) == int(poll_layer):
        return {
            "calibrated": True, "threshold": float(entry["threshold"]),
            "trigger_ids": set(entry["trigger_ids"]), "trigger_pieces": list(entry.get("trigger_pieces") or []),
            "topk": entry.get("topk"), "catch": entry.get("catch"), "fp": entry.get("fp"),
            "n_battery": entry.get("n_battery"), "calibration_note": entry.get("note"), "note": None,
        }
    if entry is not None:
        return {
            "calibrated": False, "threshold": float(fallback_threshold), "trigger_ids": None,
            "trigger_pieces": None, "topk": None,
            "note": (f"a calibration exists for {concept!r} at layer {entry['layer']}, but this request "
                    f"is polling layer {poll_layer} -- a threshold/trigger-set measured at one layer is "
                    "never applied to another layer's readout, so this concept is uncalibrated for this "
                    "poll (annotate-only, cannot trigger a correction)."),
        }
    return {"calibrated": False, "threshold": float(fallback_threshold), "trigger_ids": None,
           "trigger_pieces": None, "topk": None, "note": UNCALIBRATED_NOTE}


def resolve_guard_signals(concept_steer, spec: dict, calibration: Optional[dict],
                          poll_layer: int) -> tuple[Optional[dict], Optional[str]]:
    """Resolve every guarded concept's polling signal, FAIL CLOSED only for concepts that would actually
    be relied upon to fire a correction (calibrated at `poll_layer`) -- see the module docstring's
    FAIL-CLOSED DECISION. If a calibrated concept's dir(c) counter-direction can't be built (no unembed/
    engine support, an unfitted local J-lens sidecar layer, ...), returns (None, reason): the caller must
    refuse the WHOLE request rather than generate under a safety guarantee it cannot back for even one
    concept. An UNCALIBRATED concept degrades independently: if even `resolve_token_id` fails for it, that
    ONE concept's signal just carries no trigger ids (annotation unavailable, noted why) -- never fatal to
    the rest of the request, since it was never eligible to fire in the first place.

    Returns ({concept: resolve_concept_signal()-shaped dict}, None) on success, or (None, reason) on a
    calibrated concept's resolution failure."""
    out = {}
    for concept in spec["concepts"]:
        signal = resolve_concept_signal(concept, calibration, poll_layer, spec["threshold"])
        if signal["calibrated"]:
            built = concept_steer.compute(concept, layer=poll_layer)
            if not built.get("ok"):
                return None, f"concept {concept!r} is calibrated but unavailable: {built.get('note')}"
            out[concept] = signal
            continue
        resolved = concept_steer.resolve_token_id(concept)
        if resolved.get("ok"):
            signal = dict(signal)
            signal["trigger_ids"] = {resolved["token_id"]}
            signal["trigger_pieces"] = [resolved.get("piece") or concept]
        else:
            signal = dict(signal)
            signal["note"] = (
                (signal.get("note") or "") +
                f" Additionally, this concept could not be resolved to any token for annotation either: "
                f"{resolved.get('note')}."
            )
        out[concept] = signal
    return out, None


def resolve_guard_topk(spec: dict, resolved_signals: dict) -> int:
    """The jlens poll topk actually used this request -- see the module docstring's TOPK FLOOR section.
    Never smaller than `spec['topk']`; raised to CALIBRATION_TOPK_FLOOR (64) the moment ANY guarded
    concept is calibrated, and raised further by a per-concept calibrated "topk" value when one is
    present (forward-compatible; today's calibration artifact carries none)."""
    floor = int(spec["topk"])
    for signal in resolved_signals.values():
        if signal.get("calibrated"):
            floor = max(floor, CALIBRATION_TOPK_FLOOR)
            cal_topk = signal.get("topk")
            if isinstance(cal_topk, int) and not isinstance(cal_topk, bool):
                floor = max(floor, cal_topk)
    return floor


# =================================================================================================
# disposition read (jlens-based) -- generalized to a TRIGGER SET, not one bare token id
# =================================================================================================

def concept_activation(jlens_result: Optional[dict], token_ids, position: int = -1) -> Optional[float]:
    """The MAX raw lens-logit score, over every id in `token_ids`, appearing in one jlens readout
    position's top-k (default: the LAST position -- the most recently generated token), or None when NONE
    of them appear in that position's top-k at all (never a fabricated 0.0 -- "not in the top-k" and
    "scored exactly 0" are different facts, and only the first is what usually happens).

    `token_ids` may be a single int (one concept-word token -- the uncalibrated fallback) or any iterable
    of ints (a calibrated TRIGGER SET -- see the module docstring's SIGNAL section: "MAX lens-score over
    any trigger id present in the readout top-k" is the exact recipe
    scripts/calibration/guard_signal_calibrate.py's own max_trigger_score used to find the signal that
    actually separates banned from clean generation)."""
    if token_ids is None:
        return None
    ids = {int(token_ids)} if isinstance(token_ids, int) else {int(t) for t in token_ids}
    if not ids:
        return None
    if not isinstance(jlens_result, Mapping):
        return None
    readouts = jlens_result.get("readouts") or []
    if not readouts:
        return None
    if not (-len(readouts) <= position < len(readouts)):
        position = -1
    row = readouts[position] or []
    best = None
    for entry in row:
        if not isinstance(entry, Mapping) or entry.get("id") not in ids:
            continue
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        score = float(score)
        best = score if best is None else max(best, score)
    return best


# =================================================================================================
# the pure control loop -- engine-agnostic, fully unit-testable with fakes
# =================================================================================================

@dataclass
class GuardFiring:
    chunk_index: int
    token_position: int
    concept: str
    pre_activation: Optional[float]
    post_activation: Optional[float]
    counter_strength: float
    layer: int
    threshold: float
    calibrated: bool = True   # only correctable (calibrated) concepts can ever fire -- see the module
                             # docstring's UNCALIBRATED CONCEPTS DO NOT FIRE section
    # See the module docstring's 2026-07-25 "counter-injection can fail MID-GENERATION" section: the guard
    # DETECTED this fire either way (it is always recorded), but the corrective counter-direction may have
    # failed to build for THIS specific fire -- counter_applied=False + counter_error is the honest record
    # of that, instead of a crash or a silently-uncorrected chunk. counter_applied defaults True (a normal
    # fire) so every existing caller/test that only ever saw successful fires is unaffected.
    counter_applied: bool = True
    counter_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index, "token_position": self.token_position,
            "concept": self.concept, "pre_activation": self.pre_activation,
            "post_activation": self.post_activation, "counter_strength": self.counter_strength,
            "layer": self.layer, "threshold": self.threshold, "calibrated": self.calibrated,
            "counter_applied": self.counter_applied, "counter_error": self.counter_error,
        }


def run_guarded_generation(
    *, generate_chunk: Callable[..., str], read_disposition: Callable[[str], dict],
    build_counter: Callable[[str], Any], base_text: str, max_tokens: int, chunk_tokens: int,
    concepts: list, correctable_concepts, thresholds: dict, counter_strength: float, max_fires: int,
    layer: int,
) -> dict:
    """THE control loop (see the module docstring's MECHANISM section). Engine-agnostic: every engine
    round trip is behind an injected callable, so this function itself never talks to a socket and is
    fully deterministic to unit-test.

    generate_chunk(prompt_so_far: str, max_new: int, *, counter=None) -> str
        `counter`, when given, is whatever `build_counter()` returned for one concept (opaque to this
        loop) -- the production adapter turns it into an /intervene call; None means "generate
        uncorrected" (a plain /v1/completions-equivalent call).
    read_disposition(text_so_far: str) -> {concept: float | None}
        per-concept activation after the text generated so far, for EVERY name in `concepts` (calibrated
        and uncalibrated alike -- uncalibrated ones are polled purely for annotation, see
        `max_observed_activation` in the return value).
    build_counter(concept: str) -> Any
        the corrective payload for one concept (production: ConceptSteer.steer_toward's returned dict) --
        opaque here, passed straight through to `generate_chunk`'s `counter=`. Only ever called for a
        concept in `correctable_concepts`. MAY RAISE (production: a calibrated concept's dir(c) direction
        built fine at setup but its median-residual-norm SCALE is missing for this layer -- see the module
        docstring's 2026-07-25 "counter-injection can fail MID-GENERATION" section): this loop catches
        that per fire rather than letting it escape -- see the fire-handling below.

    `correctable_concepts`: the subset of `concepts` eligible to actually FIRE a correction (calibrated at
    the polled layer -- see the module docstring's UNCALIBRATED CONCEPTS DO NOT FIRE section). A concept
    outside this set can cross its `thresholds` entry and still never fire -- it is annotate-only.
    `thresholds`: {concept: float}, per-concept (a calibrated concept's OWN measured threshold, or the
    uncalibrated placeholder for the rest -- moot for firing since only `correctable_concepts` can fire).
    `layer`: the single resolved poll layer, folded into every GuardFiring record purely as descriptive
    metadata (this loop always polls one layer for the whole request).

    Returns {"text": str, "fires": [GuardFiring.as_dict(), ...], "n_fires": int, "cap_reached": bool,
    "n_chunks": int, "max_observed_activation": {concept: float | None}}. Once `max_fires` re-steers have
    happened, the loop stops POLLING entirely for the rest of generation (not just stops correcting) and
    generates the remainder in one plain, uncorrected call -- `cap_reached` says so; see GUARD_CAP_NOTE
    for the receipt-facing wording. Never raises on its own control flow; a failure inside an injected
    callable propagates to the caller unchanged."""
    text = ""
    fires: list[GuardFiring] = []
    cap_reached = False
    chunk_index = 0
    remaining = int(max_tokens)
    token_position = 0
    max_observed: dict = {c: None for c in concepts}

    def _note_observed(activations: dict) -> None:
        for c in concepts:
            v = activations.get(c)
            if v is not None:
                max_observed[c] = v if max_observed[c] is None else max(max_observed[c], v)

    while remaining > 0:
        this_chunk = min(int(chunk_tokens), remaining)
        prompt_so_far = base_text + text

        if cap_reached:
            # The re-steer budget is spent -- generate every remaining token in one plain, unwatched call
            # (honest: no further polling, not a silent "still checking" claim).
            text += generate_chunk(prompt_so_far, remaining, counter=None)
            remaining = 0
            break

        piece = generate_chunk(prompt_so_far, this_chunk, counter=None)
        activations = read_disposition(prompt_so_far + piece) or {}
        _note_observed(activations)
        fired_concept = next(
            (c for c in concepts if c in correctable_concepts and activations.get(c) is not None
             and activations[c] >= thresholds.get(c, DEFAULT_THRESHOLD)),
            None,
        )
        if fired_concept is not None:
            pre = activations.get(fired_concept)
            if len(fires) < max_fires:
                # HONEST DEGRADE (module docstring, 2026-07-25 "counter-injection can fail MID-GENERATION"):
                # build_counter() can raise even for a concept that resolved fine at setup (its dir(c)
                # DIRECTION was buildable; its scale/coef was not). The fire is ALWAYS recorded below --
                # only whether a correction was actually applied differs. Never a crash, never a silent
                # drop, never a fabricated correction.
                counter_error = None
                try:
                    counter = build_counter(fired_concept)
                except Exception as exc:
                    counter = None
                    counter_error = str(exc)
                if counter is not None:
                    corrected = generate_chunk(prompt_so_far, this_chunk, counter=counter)
                    post_activations = read_disposition(prompt_so_far + corrected) or {}
                    _note_observed(post_activations)
                    post = post_activations.get(fired_concept)
                    piece = corrected
                    counter_applied = True
                else:
                    post = None            # no corrected regeneration happened -- nothing new to read
                    counter_applied = False
                fires.append(GuardFiring(
                    chunk_index=chunk_index, token_position=token_position, concept=fired_concept,
                    pre_activation=pre, post_activation=post, counter_strength=counter_strength,
                    layer=layer, threshold=thresholds.get(fired_concept, DEFAULT_THRESHOLD),
                    counter_applied=counter_applied, counter_error=counter_error,
                ))
            else:
                cap_reached = True

        text += piece
        chunk_index += 1
        token_position += this_chunk
        remaining -= this_chunk

    return {
        "text": text,
        "fires": [f.as_dict() for f in fires],
        "n_fires": len(fires),
        "cap_reached": cap_reached,
        "n_chunks": chunk_index,
        "max_observed_activation": max_observed,
    }


# =================================================================================================
# light coherence note (A1.1's own "n_guarded_outputs_looking_incoherent" axis, cheap version)
# =================================================================================================

def _coherence_note(corrected_texts: list) -> dict:
    """Cheap coherence check over the CORRECTED chunks only (pure text counting, no model call -- reuses
    clozn.replay.counterfactual._coherence, the same mandatory degenerate-output gate
    research/dial_autocalibrate_engine.py's own sweep uses). A light note, not a hard gate: A1.1's own
    protocol counted "outputs looking incoherent" as a check, not a kill condition on its own. On any
    import/scan failure this degrades to {"checked": False, ...} rather than breaking the guard result."""
    if not corrected_texts:
        return {"checked": True, "n_checked": 0, "any_degenerate": False}
    try:
        from clozn.replay.counterfactual import _coherence
        flags = [_coherence(t) for t in corrected_texts]
        return {
            "checked": True, "n_checked": len(flags),
            "any_degenerate": any(f.get("degenerate") for f in flags),
        }
    except Exception as exc:
        return {"checked": False, "reason": str(exc)}


def build_receipt(result: dict, spec: dict, *, layer: int, layer_source: str, topk: int,
                  resolved_signals: dict) -> dict:
    """The public `clozn_guard_receipt` shape -- one canonical builder so every caller (the production
    route, tests) constructs the identical fields. Only present on the response when the guard actually
    ran (opted in AND every calibrated concept resolved); see the module docstring's HONESTY LAW for
    GUARD_CAVEAT's wording rules.

    `receipt['concepts']` is now a per-concept BREAKDOWN (not a bare name list): {name: {"calibrated",
    "layer", "threshold", "trigger_ids", "trigger_pieces", "max_observed_activation", "note", and (when
    calibrated) "catch"/"fp"/"n_battery"/"calibration_note"}} -- exactly the "calibrated: bool, the layer
    actually used, the threshold used, and (when calibrated) the trigger representation" this receipt is
    required to carry per concept.

    `counter_failure_note` (2026-07-25) rides whenever any fire below has counter_applied: false -- see
    the module docstring's "counter-injection can fail MID-GENERATION" section; each fire's own
    counter_error still carries the specific message, this is just the receipt-level flag that at least
    one exists, so a caller scanning only top-level fields still can't miss it."""
    max_observed = result.get("max_observed_activation") or {}
    concepts_receipt = {}
    for concept in spec["concepts"]:
        signal = resolved_signals.get(concept) or {
            "calibrated": False, "threshold": DEFAULT_THRESHOLD, "trigger_ids": None,
            "trigger_pieces": None, "note": "concept could not be resolved at all",
        }
        entry = {
            "calibrated": bool(signal.get("calibrated")),
            "layer": layer,
            "threshold": signal.get("threshold"),
            "trigger_ids": sorted(signal["trigger_ids"]) if signal.get("trigger_ids") else None,
            "trigger_pieces": signal.get("trigger_pieces"),
            "max_observed_activation": max_observed.get(concept),
            "note": signal.get("note") or signal.get("calibration_note"),
        }
        if signal.get("calibrated"):
            entry["catch"] = signal.get("catch")
            entry["fp"] = signal.get("fp")
            entry["n_battery"] = signal.get("n_battery")
        concepts_receipt[concept] = entry

    # run_guarded_generation doesn't retain per-chunk text separately, so the light coherence check runs
    # over the whole final text -- cheap (pure text counting) and still catches the case A1.1's own
    # protocol watched for (a correction that derails into repeat-loop/degenerate output).
    note = _coherence_note([result["text"]]) if result.get("text") else _coherence_note([])
    receipt = {
        "concepts": concepts_receipt,
        "layer": layer,
        "layer_source": layer_source,
        "topk": topk,
        "counter_strength": spec["counter_strength"],
        "max_fires": spec["max_fires"],
        "n_fires": result["n_fires"],
        "fires": result["fires"],
        "cap_reached": result["cap_reached"],
        "coherence": note,
        "caveat": GUARD_CAVEAT,
    }
    if result["cap_reached"]:
        receipt["cap_note"] = GUARD_CAP_NOTE
    if any(not f.get("counter_applied", True) for f in result["fires"]):
        receipt["counter_failure_note"] = GUARD_COUNTER_FAILURE_NOTE
    return receipt


# =================================================================================================
# production adapter -- wires the pure loop to a real EngineSubstrate + ConceptSteer.
# Exercised in tests/test_generation_guard_server.py via a FAKE substrate/engine (never a live one) --
# see the module docstring's MECHANISM section for why the pure loop above, not this function, carries
# the bulk of the unit-test weight.
# =================================================================================================

def _sampling_params(sub) -> dict:
    """The raw engine-call sampling kwargs (temperature/top_k/top_p/rep_penalty/seed), resolved the SAME
    way EngineSubstrate._complete_chat_native does -- see clozn.server.app._resolve_sampling. Reused here
    rather than re-derived so a guarded generation samples under the identical regime an ordinary chat
    call would (S5), not some bespoke guard-only default."""
    from clozn.server import app as ctx
    samp = ctx._resolve_sampling(True)
    if samp and samp.get("on"):
        return {"temperature": float(samp["temperature"]), "rep_penalty": float(samp["repeat_penalty"]),
               "top_k": int(samp["top_k"]), "top_p": float(samp["top_p"]), "seed": int(samp["seed"])}
    return {"temperature": 0.0, "rep_penalty": 1.0, "top_k": 0, "top_p": 1.0, "seed": 0}


def guarded_chat_completion(handler, messages: list, *, model: str, max_tokens: int, sample=True,
                           spec: dict, source: str = "openai_api", extra_meta: Optional[dict] = None) -> dict:
    """The production entry point routes/openai.py calls when `clozn_guard` is opted in on a non-streaming
    /v1/chat/completions request.

    Order of operations (see the module docstring's LAYER SELECTION / SIGNAL / FAIL-CLOSED sections):
      1. Load this exact model's guard calibration (load_guard_calibration), if any.
      2. Resolve the ONE poll layer (resolve_guard_layer) -- refuse if the engine has no valid J-lens
         layer at all.
      3. Build a fresh concept_dir.ConceptSteer at that layer, and resolve every concept's polling signal
         (resolve_guard_signals) -- refuse if any CALIBRATED concept's dir(c) can't be built.
      4. Resolve the effective topk (resolve_guard_topk) -- never below CALIBRATION_TOPK_FLOOR once any
         concept is calibrated.
      5. Drive run_guarded_generation with adapters over the engine's real /v1/completions ("complete"),
         /intervene, and /jlens calls.

    Returns one of:
      {"ok": False, "reason": "..."} -- layer or concept resolution failed; the caller must refuse the
        whole request (see the module docstring's FAIL-CLOSED DECISION), never fall back to an unguarded
        reply.
      {"ok": True, "reply": str, "run_id": str | None, "receipt": {...}, "finish_reason": str | None} --
        the guard ran (whether or not it ever actually fired); `receipt` is build_receipt's shape.
    Never raises on the resolution path; a genuine engine failure during generation propagates (the
    caller's own error handling applies, matching every other engine-touching route)."""
    import time
    from clozn.server import app as ctx
    from clozn.behavior.steering.concept_dir import ConceptSteer, _text_of as _engine_text_of

    started = time.time()
    sub = ctx.active_sub(handler)
    model_sha256 = getattr(sub, "model_sha256", None)
    calibration = load_guard_calibration(model_sha256) if model_sha256 else None

    available_layers = engine_jlens_layers(sub.engine)
    layer, layer_source = resolve_guard_layer(spec, calibration, available_layers)
    if layer is None:
        return {"ok": False,
               "reason": "no valid J-lens layer is available on this engine (jlens not loaded, or the "
                         "engine reports no fitted layers) -- cannot poll disposition at any layer"}

    concept_steer = ConceptSteer(sub.engine, layer=layer)
    resolved_signals, reason = resolve_guard_signals(concept_steer, spec, calibration, layer)
    if resolved_signals is None:
        return {"ok": False, "reason": reason}

    correctable_concepts = {c for c, s in resolved_signals.items() if s["calibrated"]}
    thresholds = {c: s["threshold"] for c, s in resolved_signals.items()}
    topk = resolve_guard_topk(spec, resolved_signals)

    base_text = ctx._engine_tmpl(sub.engine, messages)
    engine_kw = _sampling_params(sub)
    last_finish = {"reason": None}

    def generate_chunk(prompt_so_far: str, max_new: int, *, counter=None) -> str:
        if counter is None:
            resp = sub.engine.complete(prompt_so_far, max_tokens=int(max_new), **engine_kw)
        else:
            resp = sub.engine.intervene(prompt_so_far, vector=counter["vector"], coef=counter["coef"],
                                        layer=layer, max_tokens=int(max_new), **engine_kw)
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if choices:
            last_finish["reason"] = choices[0].get("finish_reason")
        return _engine_text_of(resp)

    def read_disposition(text_so_far: str) -> dict:
        jl = sub.engine.jlens(text_so_far, layer=layer, topk=topk)
        return {concept: concept_activation(jl, resolved_signals[concept].get("trigger_ids"))
               for concept in spec["concepts"]}

    def build_counter(concept: str):
        """Raises on failure -- CAUGHT PER FIRE by run_guarded_generation, never crashes the request (see
        the module docstring's 2026-07-25 "counter-injection can fail MID-GENERATION" section). A concept
        that resolved fine at setup (resolve_guard_signals only proves its dir(c) DIRECTION builds, via
        compute()) can still fail HERE if its median-residual-norm SCALE is missing for this layer --
        ConceptSteer.steer_toward's own "no_norm_calibration" block. That is a real, foreseeable gap on a
        model/layer with no concept_dial_calibration.json (see the module docstring), not a sign of
        internal corruption -- it no longer needs to look like one."""
        corrected = concept_steer.steer_toward(concept, spec["counter_strength"], layer=layer)
        if not corrected.get("ok"):
            raise RuntimeError(
                f"counter-direction for {concept!r} unavailable at layer {layer}: {corrected.get('note')}"
            )
        return corrected

    result = run_guarded_generation(
        generate_chunk=generate_chunk, read_disposition=read_disposition, build_counter=build_counter,
        base_text=base_text, max_tokens=int(max_tokens), chunk_tokens=spec["chunk_tokens"],
        concepts=spec["concepts"], correctable_concepts=correctable_concepts, thresholds=thresholds,
        counter_strength=spec["counter_strength"], max_fires=spec["max_fires"], layer=layer,
    )
    receipt = build_receipt(result, spec, layer=layer, layer_source=layer_source, topk=topk,
                           resolved_signals=resolved_signals)

    meta = dict(extra_meta or {})
    meta["clozn_guard"] = receipt
    rid = handler._log_run(source, messages, result["text"], model, started, extra_meta=meta)

    return {"ok": True, "reply": result["text"], "run_id": rid, "receipt": receipt,
           "finish_reason": last_finish["reason"]}
