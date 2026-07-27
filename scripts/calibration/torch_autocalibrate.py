"""dial_autocalibrate.py -- per-dial, per-model COHERENT OPERATING RANGE calibration.

Antecedent: parliament.py's `calibrate_and_check_liveness` (per-axis dose sweep + a shuffled-direction
liveness check + a coherence gate, on 5 fixed stances, choosing ONE operating dose). This module generalizes
that pattern from "pick one dose for these 5 stances" to "for ANY dial (the full steering.AXES tone-dial
set, plus parliament's skeptical/plain customs, or an arbitrary --dials list), report the FULL usable dose
RANGE" -- Law #6's studio-facing need: a 7B-calibrated dial derails a 1.5B, so the slider must show
0-to-where-it-actually-works, PER MODEL, not one asserted number.

WHAT THIS DOES NOT DO (read before trusting a number out of this file): it does NOT pick a single "best"
dose. Ranking doses against each other needs a USER-PREFERENCE signal (which dose's warmth/candor/whatever
the user actually likes) that nothing in this codebase collects yet. What IS measurable without that
signal: (a) where the dial provably breaks the model (derail_point, via counterfactual._coherence -- the
same mandatory degeneration gate every dial/steering receipt in this codebase already uses), and (b) where
the dial provably moves the reply toward its OWN pole, attributably -- not any perturbation of that size,
and not just "the wording changed" (usable_max, gated on beating a matched-norm SHUFFLED-direction null at
the identical dose -- the same null parliament.py's shuffled-dial-null arm uses). The output is a RANGE
plus that derail point, never a recommended single setting.

THE EFFECT MEASURE -- REWRITTEN, on a CONFIRMED real-run failure mode (stated loud, not buried). A prior
version of this module measured "effect" as word-type-Jaccard CHANGE vs baseline
(receipts.receipt_metrics(...)["changed"] / 100) -- "how much did the wording move", full stop. That
measure cannot tell a dial that genuinely produces its trait from a dial that merely REFORMATS the answer
(headers, bullet points, a worked example, a different opening line) while never actually moving toward its
own pole: on a real Qwen2.5-7B-Instruct nf4 run (research/runs/dial_autocalibrate.json), "skeptical" at dose
1.0 turned a plain TCP/UDP explainer into one with "### Key Differences" headers and a worked example --
genuinely different wording -- but not one sentence of actual doubt, hedging, or challenged claims, while
"warm" at the SAME dose produced "Hey there! I'd love to explain...", an actually warmer reply. The OLD
metric scored both "usable", identically; a reformat is not steering.

THE FIX -- direction, not magnitude. A dial IS a direction in the residual stream at its steering layer
(sc.vecs[dial]: the UNIT diff-of-means of its pos/neg poles, computed once by SteeringControl.compute /
add_custom). So instead of asking "how different is the wording", this module now asks "did the reply's
OWN representation move further toward that direction" -- a white-box projection, no LLM judge, one extra
forward pass per reply (no generation):

  directional_alignment(sc, reply_text, dial) -- encode `reply_text` RAW (no chat template, no added
      generation prompt: the tokenizer sees exactly the reply's own tokens and nothing else -- see the
      function's own docstring for why raw rather than chat-wrapped), one forward pass
      (output_hidden_states=True), MEAN-POOL the layer-`sc.layer` hidden state over every token position
      (hidden_states[sc.layer + 1], the same layer/indexing convention _last_resid already uses elsewhere
      in this codebase), then take the SIGNED scalar projection of that pooled vector onto
      unit(sc.vecs[dial]). Higher = further toward the dial's POSITIVE pole. This is the piece that
      changed; HOW the direction itself is computed (SteeringControl.compute/add_custom) is untouched.

  directional_effect(sc, dial, baseline_texts, steered_texts) -- THE new `effect` / `shuffled_effect` curve
      field: mean over the prompt sample of [directional_alignment(steered) - directional_alignment(that
      SAME prompt's own dose-0 baseline reply)]. Positive = the steered reply sits further toward the pole
      than that prompt's own unsteered reply did; ~0 = no net directional movement (what a mere-reformat
      dose should show, however much its wording changed); negative = moved toward the OPPOSITE pole -- a
      real, reportable finding, never clamped away (see _compute_calibration, which correctly never credits
      a negative number as "real": this sweep only ever engages the positive-pole direction of a dose).

THE OLD MEASURE IS KEPT, DEMOTED TO A DIAGNOSTIC (`change_magnitude`, still computed by
effect_vs_baseline/receipts.receipt_metrics, unchanged) precisely so a reader can see change-vs-direction
side by side in the same curve row -- a dose with a big change_magnitude and a near-zero effect IS the
reformat-not-steering signature this fix exists to catch, and the JSON/console output should make that
legible, not hide it. change_magnitude no longer feeds usable_max / dead_below / derail_point; only the
direction-aware `effect` does.

CAVEATS ON THE NEW MEASURE (Law #6, stated loud, not hidden -- this is a DIFFERENT metric with ITS OWN
honest limitations, not a solved problem):
  * MEAN-POOL is a CHOICE, not a law: pooling over every token treats a reply's opening line and its last
    clause as equally informative about "did this move toward the pole". A trait that shows up in only
    part of a long reply (one warm closing line on an otherwise neutral answer) gets diluted by the tokens
    around it. A last-token or an attention-weighted pool could disagree with this one on some replies.
  * SINGLE LAYER: only sc.layer (the SAME mid layer steering itself pushes on) is read. A trait could be
    more legible earlier or later in the stack; this module does not sweep layers to check.
  * SELF-REFERENTIAL: the ruler is made of the same stuff as what it measures -- sc.vecs[dial] is the
    diff-of-means of the SAME pos/neg instruction pair used to steer, so a reply that merely echoes the
    pole's own vocabulary (e.g. says "I doubt this claim" without actually scrutinizing anything) could
    still project toward the direction for surface-lexical reasons correlated with, but not identical to,
    genuinely exhibiting the trait. This is a narrower, subtler version of the OLD metric's gameability, not
    an elimination of it -- sample_replies are kept for exactly this reason: a human should still eyeball
    any dose flagged "usable".
  * ONE MODEL, ONE RUN: like every other number in this file, a raw projection value carries no meaning
    transferred to a different model, layer, or quantization -- see _EFFECT_EPS below and "SINGLE MODEL
    ONLY" further down.

THE SHUFFLED NULL: for each dial, ONE random UNIT direction is drawn once (make_shuffle_unit_vector, seeded
deterministically from (--seed, dial name) via _dial_seed -- pure integer arithmetic, not Python's
process-randomized hash()) and reused at EVERY dose of that dial's sweep, written at the SAME magnitude as
the real direction at that dose -- UNCHANGED. What changed is how the null is SCORED: its directional_effect
is directional_alignment(shuffled_steered_reply) - directional_alignment(baseline), projected onto THAT SAME
REAL dial's unit(sc.vecs[dial]) -- never onto the random direction itself. The question the null answers is
"does a random push of this size happen to move the reply along THIS dial's own axis" -- projecting onto
the random direction's own axis would answer a different, uninteresting question (a random direction
trivially aligns with itself). A real, working dial should clear this null comfortably; a random direction
of matched magnitude should not, and should usually score close to zero.

THRESHOLDS ARE CHOSEN CUTS, NOT DERIVED (stated loud, not hidden): _DEGEN_THRESHOLD=0.34 is UNCHANGED (a
dose is flagged derailing once MORE than ~1-in-3 sampled prompts come back degenerate -- copied from
parliament.py's _DEGEN_OK; coherence has nothing to do with the effect measure and needed no re-tuning).
_EFFECT_EPS changed VALUE AND UNITS along with the effect measure: it used to be 0.03 on the old metric's
native [0,1] Jaccard-distance scale (3% of the full dynamic range). The new metric's native scale is a raw
dot product -- a projection of a mean-pooled residual vector onto a UNIT direction at sc.layer, unbounded,
and NOT comparable across models/layers/quantizations the way a 0-1 ratio incidentally was. _EFFECT_EPS is
now 2.0, PICKED (not derived) from this exact rig's own most recent real run against Qwen2.5-7B-Instruct
nf4 at layer 14 (research/runs/dial_autocalibrate.json's recorded steer_info.resid_norm = 68.7 -- the
average norm of a SINGLE token's layer-14 residual over the contrastive seed prompts; hidden_size for this
model is 3584, confirmed from its config.json, so 28 layers -> mid-layer 14 matches SteeringControl's own
comment). A generic, uncorrelated vector of that norm projected onto an arbitrary fixed unit direction in a
3584-dim space would be expected to land, VERY roughly, around resid_norm / sqrt(hidden_size) =~
68.7 / 59.9 =~ 1.15 -- the textbook scaling for an unrelated high-dimensional vector against a fixed axis, a
rough sanity check and NOT a rigorous bound (real activations are anisotropic, not isotropic, so this could
be off in either direction). 2.0 sits comfortably above that rough noise-floor estimate, and well below the
scale of an obvious, human-legible shift (this run's warm-dial sample_replies visibly change register by
frac 1.0, under a raw steering push of base=58.42 per unit strength). Like _DEGEN_THRESHOLD, this is an
eyeballed cut a different, equally defensible choice could move -- a module constant, not learned or fit --
but it carries a SHARPER caveat than the old dimensionless epsilon did: because it is expressed in this
model/layer's own raw units rather than a portable 0-1 ratio, re-running this rig against a very
differently-scaled model, quantization, or steering layer should come with re-EYEBALLING this constant
against THAT run's own curve (not just trusting 2.0 to still mean "small" there) -- Law #6 applies to this
number harder than it did to its predecessor.

THE PROMPT SAMPLE SHAPES THE RESULT (stated loud): by default this pulls the N most recent DISTINCT user
turns from runlog (runlog.list_runs + runlog.get_run -- real, unmodified text, on the theory that "does this
dial hold up on THIS user's real prompts" is the actual question), falling back to a small built-in neutral
set (NEUTRAL_PROMPTS) when the runlog is empty (a fresh install, or a machine with nothing logged yet).
Every run records which source was actually used (prompt_source) and the literal prompts (prompts) -- a
different sample, or a different day's runlog, can shift the calibrated range; this is a calibration against
A sample, not THE truth.

SINGLE MODEL ONLY: every number in one run's JSON is specific to (this checkpoint, this quantization, this
prompt sample, this layer). Nothing here is portable to a different model size -- that is the entire point
(Law #6) and also the whole reason no single number here should ever be read as a universal constant.

GREEDY DECODING throughout (matching receipts.py/counterfactual.py's own convention): a diff between two
arms must be attributable to the dose/direction change, not to sampling dice.

Cost: O(n_dials x n_doses x n_prompts x ~2) greedy generations (~2 = one real-direction decode + one
shuffled-direction decode per prompt per nonzero dose; dose 0 needs only 1, shared as everyone's baseline),
PLUS one cheap forward pass (no generation, output_hidden_states only) per reply for directional_alignment.
Generation and alignment are chunked by --batch-size (default 8), and the unsteered baseline replies are
generated once and shared across all dials in the sweep. Lower --batch-size after an OOM; raise it when
VRAM headroom remains. Every completed dial is checkpointed, and --resume continues a compatible --out
without recomputing those dials. Default config (12 dials x 7 doses x 6 prompts) is a few hundred short
greedy decodes -- --smoke (1-2 dials, 3 doses, 2 prompts) proves the wiring cheaply and is NOT a finding.

--exemplars MODE -- A/B the DIRECTION RECIPE, not just the dose (motivated by a real, measured failure: on
Qwen3.5-9B, every dial under the instruction recipe below came back near-zero effect except `poetic`, while
the SAME harness steers Qwen2.5-14B strongly -- see dial_exemplars.py's own module docstring for the full
story). The DEFAULT recipe this module has always used (compute_dials -> SteeringControl.compute/add_custom)
contrasts ONE instruction pair ("Respond warmly" vs "Respond coldly") across a handful of SEED_PROMPTS and
diff-of-means the last prompt token's residual -- cheap, and it works on some model families, but it carves a
much fainter axis on others. `--exemplars [BANK.json]` swaps in a STRONGER recipe for any dial the bank
covers: derive_caa_directions reads dial_exemplars.py's matched-pair bank (many {prompt, pos reply, neg
reply} triples per dial, the standard contrastive-activation-steering construction) and, for each pair, reads
the residual IN CHAT CONTEXT -- render the chat template with `prompt` as the user turn and the reply
appended as the assistant turn, then MEAN-POOL sc.layer's hidden state over the reply's OWN token span only
-- rather than a single instruction's last prompt token. This matters for two independent reasons: (1) real
styled TEXT captures the actual output distribution a working dial needs to steer toward, not the model's
(possibly weak) READING of a command; (2) many matched pairs, mean-differenced, is a more robust estimate
than one pair x a few seeds. Both knobs are stated loud so a reader can tell which one (if either) mattered.

ORDERING IS LOAD-BEARING: compute_dials() ALWAYS runs first, unchanged, computing every requested dial's
instruction-derived direction AND calibrating sc.base/sc.resid_norm (the per-model dose SCALE -- see Law #6
in the docstring above). Only THEN, if --exemplars was passed, does derive_caa_directions/apply_caa_directions
REPLACE sc.vecs[name] for whichever dials the bank covers with >= dial_exemplars.MIN_RECOMMENDED_PAIRS pairs
-- sc.base is never recomputed from the exemplar bank, so the two recipes are compared at the IDENTICAL dose
scale, and the only thing that differs between an "exemplars" dial and an "instructions" dial in the same run
is the ORIENTATION of the unit direction being swept, never the strength units it's swept in. A dial the bank
doesn't cover (or covers too thinly) keeps its instruction-derived vector -- an honest partial swap, recorded
per-dial in the saved report's dial_source/caa_pairs_used/caa_pairs_skipped fields, never silently assumed.

Run (CUDA venv):
    PY=C:/Users/brigi/src/clozn/.venv/Scripts/python.exe
    $PY scripts/calibration/torch_autocalibrate.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 8 --out research/runs/dial_autocalibrate.json
Smoke first (prove the wiring cheaply -- NOT a finding):
    $PY scripts/calibration/torch_autocalibrate.py --smoke --out research/runs/dial_autocalibrate_smoke.json
A subset of dials:
    $PY scripts/calibration/torch_autocalibrate.py --dials warm candid concrete --n-prompts 10
Resume an interrupted compatible checkpoint (same model/prompts/dials/settings, including batch size):
    $PY scripts/calibration/torch_autocalibrate.py --batch-size 8 --resume --out research/runs/dial_autocalibrate.json

--library MODE: sweep an entire CANDIDATE LIBRARY (research/dial_library_candidates.json's ~70-dial,
15-category {"dials":[{name,category,pos,neg,predict}]} format) instead of steering.AXES/--dials -- every
entry registered as a custom dial (register_library_dials) and swept through the IDENTICAL calibrate_dial
path as everything else in this file. Checkpoint-saved after EVERY dial (see run_library's docstring), so a
kill/OOM partway through this much bigger run keeps every dial finished so far and --resume can continue it:
    $PY scripts/calibration/torch_autocalibrate.py --library research/dial_library_candidates.json --out research/runs/dial_library_sweep.json
--report MODE: pure analysis, NO model/GPU (still imports torch/transformers at module level like every mode
here, but loads no model and touches no CUDA) -- reads a completed --library sweep JSON and prints the
per-category summary, the surface-vs-cognitive hypothesis verdict, and the curated shippable list, and writes
that curated list to --curated-out (default research/runs/dial_library_curated.json):
    $PY scripts/calibration/torch_autocalibrate.py --report research/runs/dial_library_sweep.json
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root (clozn/ package)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import clozn.behavior.steering.axes as steering_mod
from clozn.behavior.steering import SteeringControl
from clozn.replay.counterfactual import _coherence   # {"degenerate": bool, "reason": str} -- the mandatory coherence axis
from clozn import receipts                          # receipt_metrics -- the word-type-Jaccard "%changed" DIAGNOSTIC
import clozn.runs.store as runlog
import dial_exemplars   # sibling module (scripts/calibration/) -- the matched-pair CAA exemplar bank loader

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================================================ helpers
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _positive_int(value: str) -> int:
    """argparse type for knobs where zero would otherwise create a silent no-op/infinite loop."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _chunks(items, size: int):
    """Yield stable, order-preserving list chunks. `size` is validated at both CLI and API boundaries."""
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _ensure_padding_token(tok) -> None:
    """Decoder-only batch generation needs a real pad id; Llama-family tokenizers often omit one."""
    if getattr(tok, "pad_token_id", None) is not None:
        return
    if getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    elif getattr(tok, "unk_token", None) is not None:
        tok.pad_token = tok.unk_token
    else:
        raise RuntimeError("tokenizer has no pad/eos/unk token; cannot batch safely")


def _file_sha256(path: str | None) -> str | None:
    """Fingerprint result-shaping input files so resume cannot mix changed direction recipes."""
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def degenerate_rate(texts: list[str]) -> float:
    return round(_mean(_coherence(t)["degenerate"] for t in texts), 3) if texts else 0.0


def effect_vs_baseline(baseline_texts: list[str], steered_texts: list[str]) -> float:
    """THE OLD effect measure -- KEPT, but DEMOTED to a reported diagnostic field (`change_magnitude`) as of
    the direction-aware rewrite (see the module docstring's THE EFFECT MEASURE section): 1 minus the
    word-type-Jaccard-similarity between a steered reply and that SAME prompt's dose-0 (unsteered) reply --
    concretely, receipts.receipt_metrics(baseline, steered)["changed"] / 100, averaged over the prompt
    sample. This is deliberately the SAME pure-counting "%-of-wording-changed" metric heavn Replay's delta strip /
    receipts.py / counterfactual.py already compute for every other receipt in this codebase, so "how much
    did this dose move the text, lexically" is never a second, silently-different definition of that
    question.

    Computed PER PROMPT (steered_texts[i] vs baseline_texts[i] -- that SAME prompt's own dose-0 reply,
    never a cross-prompt comparison) and averaged over the sample. 0.0 = identical word types (no
    detectable change); 1.0 = completely disjoint vocabularies.

    NO LONGER DRIVES usable_max / dead_below / derail_point (directional_effect does -- see the module
    docstring): this function's own CONFIRMED failure mode on a real run is EXACTLY why -- a dial that only
    reformats the answer (new headers, a different opening line, a worked example) scores a large "changed"
    number here with zero genuine movement toward its pole, and a degenerate reply (repetition loop /
    script-switch) very often ALSO scores large here despite being worthless. Kept and reported
    (`change_magnitude`) specifically so a reader can see change-vs-direction side by side in the same curve
    row -- a big change_magnitude next to a near-zero effect IS the reformat-not-steering signature this
    module was rewritten to catch, and hiding that comparison would be less honest than showing it."""
    if not baseline_texts or not steered_texts or len(baseline_texts) != len(steered_texts):
        return 0.0
    vals = [receipts.receipt_metrics(b, s)["changed"] / 100.0 for b, s in zip(baseline_texts, steered_texts)]
    return round(_mean(vals), 4)


def _project_onto_unit(vec: torch.Tensor, direction: torch.Tensor) -> float:
    """Pure tensor math, no model, no I/O: the SIGNED scalar projection of `vec` onto `direction`,
    normalizing `direction` to a unit vector first regardless of whether it already is one (sc.vecs[dial]
    entries already ARE unit per SteeringControl.compute/add_custom, but this makes the projection correct
    even if handed a raw, non-unit vector -- defensive, not load-bearing at the current call site).
    Unit-testable directly on fabricated CPU tensors, exactly as make_shuffle_unit_vector already is."""
    unit = direction / (direction.norm() + 1e-8)
    return float(torch.dot(vec.float(), unit.float().to(vec.device)))


@torch.no_grad()
def directional_alignment(sc, reply_text: str, dial: str) -> float:
    """White-box directional-alignment score for ONE generated reply against ONE dial's own axis -- the
    primitive the new effect measure (directional_effect, below) is built from: does this reply's own
    representation sit further toward the dial's pole, not just "does it look different" (see the module
    docstring's THE EFFECT MEASURE section for why this replaced the old word-Jaccard measure).

    ENCODING CHOICE (documented, not accidental): `reply_text` is tokenized RAW -- sc.tok(reply_text), with
    NO chat template wrapped around it and no added generation prompt -- so every token fed to the model is
    reply CONTENT, nothing else. Two reasons, not one: (1) a chat-wrapped encoding (role="user", say) would
    introduce role/special-token scaffolding (<|im_start|>user, <|im_end|>, ...) that a mean-pool would need
    to explicitly carve back out to avoid diluting the signal with template tokens, and getting that
    carve-out wrong would silently leak scaffolding into the score; (2) a reply is never itself a "user
    turn" -- it is the model's OWN assistant output, generated as raw continuation tokens with no wrapper of
    its own -- so feeding it back in exactly that raw form, rather than re-framing the model's own words as
    if a user had said them, is the less distorting choice. The cost (an honest limitation, not hidden):
    this reads the model in a slightly different "mode" than the chat-templated, last-token instruction
    contrasts that compute sc.vecs[dial] itself (SingleTurnSteer._last_resid / SteeringControl._last_resid)
    -- an apples-to-oranges wrinkle inherent to comparing a direction derived from INSTRUCTIONS against
    content read from a finished REPLY. One forward pass, no generation.

    POOLING CHOICE (documented, not accidental): MEAN over every token position of hidden_states[sc.layer +
    1] (the output of decoder block sc.layer -- identical indexing convention to every other _last_resid in
    this codebase), not just the last token. A reply is being read as a finished utterance to locate in
    activation space, not as a not-yet-answered prompt whose LAST token is about to decide the next one
    (that is _last_resid's own, different, job: computing a direction FROM instruction contexts, never a
    question this function needs to answer). Mean-pooling treats the reply's opening line and its last
    clause as equally informative -- a trait concentrated in only part of a long reply is diluted by the
    tokens around it; a different pooling choice (last-token, attention-weighted) could disagree with this
    one on some replies. See the module docstring's CAVEATS for this and the metric's other limitations.

    Empty text / zero tokens -> 0.0 (nothing to project). Returns the SIGNED scalar projection of the
    pooled residual onto unit(sc.vecs[dial]) via _project_onto_unit -- a raw dot product, NOT centered or
    scaled against anything of its own; callers always compare it against the SAME prompt's own
    baseline-reply alignment (directional_effect = align(steered) - align(baseline)), never read as an
    absolute in isolation."""
    if not reply_text:
        return 0.0
    ids = sc.tok(reply_text, return_tensors="pt").input_ids.to(DEV)
    if ids.shape[1] == 0:
        return 0.0
    hs = sc.model(ids, output_hidden_states=True).hidden_states[sc.layer + 1]
    pooled = hs[0].float().mean(dim=0)          # [H] -- mean over every (content) token position
    return _project_onto_unit(pooled, sc.vecs[dial])


@torch.no_grad()
def directional_alignments(sc, reply_texts: list[str], dial: str, batch_size: int = 1) -> list[float]:
    """Batch-aware form of directional_alignment, preserving input order and masking padding from pools.

    batch_size=1 deliberately calls the scalar primitive so existing callers/tests that replace
    directional_alignment keep working. Real calibration runs pass their --batch-size here and amortize
    tokenization/model-forward overhead across multiple replies. Empty replies retain the scalar API's
    documented 0.0 result without feeding synthetic padding/BOS content through the model.
    """
    texts = list(reply_texts or [])
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_size == 1:
        return [directional_alignment(sc, text, dial) for text in texts]

    scores = [0.0] * len(texts)
    nonempty = [(i, text) for i, text in enumerate(texts) if text]
    if not nonempty:
        return scores

    _ensure_padding_token(sc.tok)
    unit = sc.vecs[dial].float()
    unit = unit / (unit.norm() + 1e-8)
    for rows in _chunks(nonempty, batch_size):
        indices = [i for i, _ in rows]
        batch_texts = [text for _, text in rows]
        enc = sc.tok(batch_texts, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(DEV)
        mask = enc.get("attention_mask")
        mask = (mask.to(DEV) if mask is not None else torch.ones_like(ids)).bool()
        hs = sc.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[
            sc.layer + 1
        ].float()
        weights = mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        values = torch.matmul(pooled, unit.to(pooled.device)).detach().cpu().tolist()
        for index, value in zip(indices, values):
            scores[index] = float(value)
        del ids, mask, hs, weights, pooled
    return scores


def directional_effect(sc, dial: str, baseline_texts: list[str], steered_texts: list[str], *,
                       batch_size: int = 1, baseline_alignments: list[float] | None = None) -> float:
    """THE new effect measure (replaces the old word-Jaccard change_magnitude as what usable_max/dead_below
    are actually gated on -- see the module docstring's THE EFFECT MEASURE section): mean over the prompt
    sample of [directional_alignment(steered) - directional_alignment(baseline)], each alignment a
    projection onto the dial's OWN unit direction (sc.vecs[dial]).

    Computed PER PROMPT (steered_texts[i] against that SAME prompt's own baseline_texts[i], never a
    cross-prompt comparison) and averaged over the sample -- the identical pairing discipline
    effect_vs_baseline already used. Positive = the steered reply sits further toward the dial's positive
    pole than that prompt's own unsteered reply did; ~0 = no net movement along the dial's axis (what a
    mere-REFORMAT dose should show, however much its wording changed); negative = moved toward the dial's
    NEGATIVE pole -- also a real, reportable finding, never clamped away (a caller comparing against
    _EFFECT_EPS with a plain `>` naturally excludes it from "real", which is correct: this sweep only ever
    engages the POSITIVE-pole direction of a dose -- see calibrate_dial).

    Called with the IDENTICAL `dial` name for both the real-direction arm and the shuffled-direction null --
    only `steered_texts` differs between the two calls a sweep makes per dose. That is deliberate, not an
    oversight: see the module docstring's THE SHUFFLED NULL section for why the null's replies are
    projected onto the SAME real dial axis rather than onto the random direction's own axis.

    baseline_alignments is an optional per-dial cache populated once by calibrate_dial. Direct callers can
    omit it and retain the original self-contained behavior. Both baseline and steered scoring honor
    batch_size, with batch_size=1 preserving the original scalar execution path."""
    if not baseline_texts or not steered_texts or len(baseline_texts) != len(steered_texts):
        return 0.0
    baseline_scores = (list(baseline_alignments) if baseline_alignments is not None
                       else directional_alignments(sc, baseline_texts, dial, batch_size=batch_size))
    if len(baseline_scores) != len(baseline_texts):
        return 0.0
    steered_scores = directional_alignments(sc, steered_texts, dial, batch_size=batch_size)
    vals = [steered - baseline for baseline, steered in zip(baseline_scores, steered_scores)]
    return round(_mean(vals), 4)


# nf4 for anything that won't fit bf16 comfortably on the 16GB card -- copied verbatim from
# parliament.py/mirror_bench.py (not imported): this codebase's own precedent is that each experiment
# script owns its small model-loading helpers rather than importing a sibling script.
_SMALL = ("0.5b", "1.5b", "-1b", "1b-", "2b", "3b", "-1.7b")
def wants_four_bit(name: str, override: str) -> bool:
    if override == "yes":
        return True
    if override == "no":
        return False
    return not any(s in name.lower() for s in _SMALL)


def axis_max_of(sc, name: str) -> float:
    """Per-dial calibrated ceiling -- sc.custom's own "max" FIRST if `name` has been explicitly registered
    as a custom dial (via add_custom), else steering.AXES' own "max" for a plain built-in, else
    SteeringControl.set's own default (1.5) if neither declares one.

    PRECEDENCE, AND WHY IT CHANGED FROM parliament.py's IDENTICAL-LOOKING HELPER: this used to check AXES
    first (parliament.py's own axis_max_of, copied verbatim, still does -- untouched, out of scope here).
    That was harmless there because parliament.py never registers a custom dial under a name that ALSO
    exists in steering.AXES. This module's --library mode can: the candidate library (research/
    dial_library_candidates.json) independently invented dial names like "warm"/"concise"/"formal"/
    "playful"/"poetic"/"concrete"/"confident" that happen to collide with steering.AXES' own built-in keys.
    add_custom already OVERWRITES sc.vecs[name] on such a collision (the library's own pos/neg direction
    wins, unconditionally -- see SteeringControl.add_custom), so a library dial's DIRECTION is already the
    library's, not the built-in's; axis_max_of checking AXES first would silently give that SAME dial the
    built-in's max (often 1.5, since most built-ins don't declare one) instead of the library's registered
    ceiling (_LIBRARY_DEFAULT_MAX) -- a real, silent mismatch between which pos/neg pair defines the swept
    axis and which ceiling bounds the sweep of it. Checking sc.custom first keeps both in lock-step:
    whichever definition is actually live in sc.vecs[name] is also the one whose max is read. Safe for every
    OTHER existing call site: a name only ever lives in ONE of {sc.custom, steering.AXES} everywhere else in
    this codebase (see test_axis_max_of_builtin_caps / test_axis_max_of_custom_axis, both still passing
    unchanged), so this precedence flip only ever changes behavior on the new collision case (see
    test_axis_max_of_custom_overrides_builtin_on_name_collision)."""
    return (sc.custom.get(name) or steering_mod.AXES.get(name) or {}).get("max", 1.5)


def _dial_seed(base_seed: int, name: str) -> int:
    """Deterministic per-(run-seed, dial-name) integer seed for the shuffled-direction generator -- pure
    integer arithmetic over the dial name's character codes, NOT Python's hash() (string hashing is
    process-randomized unless PYTHONHASHSEED is pinned, which would silently break --seed reproducibility).
    Generalizes parliament.py's _axis_seed (which indexes into a fixed 5-item STANCES list) to an arbitrary
    dial-name string, since this module's dial set is open-ended (--dials). Position-weighted so an
    anagram-like pair of names doesn't collide."""
    name_val = sum((i + 1) * ord(c) for i, c in enumerate(name))
    return (int(base_seed) * 1_000_003 + name_val * 97 + 13) & 0xFFFFFFFF


def make_shuffle_unit_vector(ref: torch.Tensor, seed: int) -> torch.Tensor:
    """A fresh random UNIT direction with the same shape/device/dtype as `ref`, seeded reproducibly on CPU
    (so the same --seed gives the same shuffled directions regardless of CUDA's own RNG state). Pure tensor
    math -- no model -- so this is unit-testable on any CPU tensor. Copied verbatim from parliament.py."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    v = torch.randn(ref.shape, generator=gen).to(ref.device, ref.dtype)
    return v / (v.norm() + 1e-8)


def _free_cuda():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


# ============================================================================================ prompt sample
# Neutral fallback -- deliberately original text, disjoint from the steering SEED_PROMPTS (used only to
# compute the diff-of-means directions themselves) and from parliament.py's CALIB_PROBES, so evaluating on
# it is never circular with either.
NEUTRAL_PROMPTS = [
    "What's a good way to spend a rainy afternoon?",
    "Can you help me plan a small dinner party?",
    "I'm not sure what to do about a noisy neighbor.",
    "What should I keep in mind before starting a garden?",
    "Tell me about a topic you find interesting.",
    "I'm nervous about an upcoming presentation at work.",
    "What's the best way to organize a messy closet?",
    "How do I get better at sticking to a morning routine?",
    "My phone battery drains really fast lately, any ideas?",
    "What are some good conversation starters for a first date?",
]


def sample_prompts(n: int, seed: int = 0) -> tuple[list[str], str]:
    """(prompts, source) -- source is "runlog" or "neutral-fallback". Pulls the n most recent DISTINCT
    user turns from runlog (runlog.list_runs() for recency-ordered ids, runlog.get_run() for the full,
    un-truncated `messages` -- list_runs()'s own prompt_summary is clipped to 90 chars, too short to trust
    as an actual generation prompt), else NEUTRAL_PROMPTS[:n] when the runlog is empty/unavailable (a fresh
    install, or a machine whose ~/.clozn/runs has nothing logged yet). `seed` is accepted for interface
    symmetry with this module's other seeded calls but unused: sampling takes "the N most recent", not a
    random draw, so there is nothing to seed here -- kept so callers never need a special case.

    Deliberately real, unmodified user text: the entire point of calibrating against "the user's real
    prompts" is to catch a dial that behaves differently on this user's actual distribution than on a bank
    of neutral test sentences -- see the module docstring's "the prompt sample shapes the result" caveat.
    """
    del seed  # unused -- see docstring
    try:
        rows = runlog.list_runs(limit=max(200, n * 20))   # already newest-first
    except Exception:
        rows = []
    seen: set = set()
    prompts: list[str] = []
    for row in rows:
        rid = row.get("id") if isinstance(row, dict) else None
        if not rid:
            continue
        rec = runlog.get_run(rid)
        if not rec:
            continue
        msgs = rec.get("messages") or []
        user_text = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        user_text = (user_text or "").strip()
        if not user_text or user_text in seen:
            continue
        seen.add(user_text)
        prompts.append(user_text)
        if len(prompts) >= n:
            break
    if prompts:
        return prompts, "runlog"
    return list(NEUTRAL_PROMPTS[:n]), "neutral-fallback"


def load_prompts_file(path: str) -> tuple[list[str], str]:
    """(prompts, source) for an explicit --prompts-file: a CURATED, model-independent bank, used verbatim
    and in full (n_prompts does NOT cap it -- the whole point of a hand-picked set is that every prompt
    earns its place). This is the reusable knob: the SAME file, swept against any --model, makes two
    models' calibrations comparable because they were measured on the identical prompt distribution (a
    runlog sample is per-machine and drifts; a file is fixed). Accepts a bare JSON list of strings, or a
    {"prompts": [...]} object (extra keys like "description" ignored), or -- if the file isn't JSON -- one
    non-empty, non-`#`-comment prompt per line. Source is tagged file:<basename> so it's self-documenting
    in the saved report's prompt_source."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        obj = json.loads(raw)
        items = obj.get("prompts", []) if isinstance(obj, dict) else obj
    except json.JSONDecodeError:
        items = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    prompts = [str(p).strip() for p in items if isinstance(p, (str, int, float)) and str(p).strip()]
    if not prompts:
        raise SystemExit(f"--prompts-file {path!r} yielded no usable prompts "
                         "(expected a JSON list of strings, a {'prompts':[...]} object, or one per line)")
    return prompts, f"file:{os.path.basename(path)}"


# =========================================================================== the backbone + dial machinery
class SingleTurnSteer(SteeringControl):
    """SteeringControl, but every contrast prompt used to COMPUTE a direction is folded into a single USER
    turn (no system role). Copied from parliament.py's class of the same name: some chat templates (Gemma-2)
    reject a system role outright, and using the identical single-user-turn recipe UNCONDITIONALLY (not
    just when --model happens to be one of those) keeps the direction-computation recipe identical no
    matter which checkpoint --model points at. compute()/add_custom() are inherited unchanged from
    SteeringControl and call this override polymorphically -- nothing in the adapter itself needs touching.
    """

    @torch.no_grad()
    def _last_resid(self, system: str, user: str) -> torch.Tensor:
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            add_generation_prompt=True, return_tensors="pt")
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(DEV)  # 5.x returns a BatchEncoding
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1]
        return hs[0, -1].float()


class Rig:
    """Loads one model. Local-cache-first path lookup and the nf4-vs-bf16 choice follow parliament.py's
    Rig (itself following steer_vs_prompt.py's), except four_bit uses this module's own wants_four_bit."""

    def __init__(self, name: str, four_bit_override: str = "auto"):
        from transformers import AutoConfig
        path = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else name
        cfg = AutoConfig.from_pretrained(path)
        # Wave-1 note: some checkpoints ship as MULTIMODAL wrappers (Gemma-4-E4B = Gemma4ForConditionalGeneration,
        # Ministral-3 = Mistral3ForConditionalGeneration) whose text decoder lives at .model.language_model, and
        # some ship PRE-QUANTIZED (Ministral-3 in FineGrainedFP8) so bitsandbytes must NOT stack on top. The seam
        # everything downstream needs is unchanged: a `self.model` that generates and returns text hidden_states,
        # with `self.model.model.layers` = the decoder stack (SteeringControl hooks there).
        text_cfg = getattr(cfg, "text_config", None)            # non-None => multimodal wrapper
        prequant = getattr(cfg, "quantization_config", None)    # non-None => already quantized on disk
        self.four_bit = False if prequant is not None else wants_four_bit(name, four_bit_override)
        kind = "native-prequant" if prequant is not None else ("nf4" if self.four_bit else "bf16")
        print(f"[load] {name} ({kind}{', multimodal->text' if text_cfg else ''}, {DEV}) ...", flush=True)

        is_mistral = "mistral" in str(getattr(cfg, "model_type", "")).lower() or "mistral" in name.lower()
        self.tok = AutoTokenizer.from_pretrained(path, **({"fix_mistral_regex": True} if is_mistral else {}))
        if not getattr(self.tok, "chat_template", None):        # Mistral ships a standalone chat_template.jinja
            try:
                from transformers.utils import cached_file
                jinja = cached_file(path, "chat_template.jinja", _raise_exceptions_for_missing_entries=False)
                if jinja:
                    self.tok.chat_template = open(jinja, encoding="utf-8").read()
            except Exception:
                pass

        loader = AutoModelForCausalLM
        if text_cfg is not None:
            from transformers import AutoModelForImageTextToText
            loader = AutoModelForImageTextToText
        if prequant is not None:
            self.model = loader.from_pretrained(path, device_map={"": 0}).eval()
        elif self.four_bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = loader.from_pretrained(path, quantization_config=bnb, device_map={"": 0}).eval()
        else:
            self.model = loader.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()

        if text_cfg is not None:
            # Point the decoder-stack seam at the text backbone and make num_hidden_layers the TEXT count, so
            # SteeringControl (model.model.layers[L], model.config.num_hidden_layers) resolves unchanged.
            self.model.model.layers = self.model.model.language_model.layers
            if getattr(self.model.config, "num_hidden_layers", None) is None:
                self.model.config.num_hidden_layers = text_cfg.num_hidden_layers

        _ensure_padding_token(self.tok)
        # Decoder-only generation must left-pad: the final input position for every row must be a real
        # prompt token, not padding, because generate() selects next-token logits from that position.
        self.tok.padding_side = "left"

    @torch.no_grad()
    def gen_batch(self, users: list[str], max_new: int = 100, sample: bool = False,
                  temperature: float = 0.9) -> list[str]:
        """Generate one ordered batch of single-USER turns with padding masked correctly.

        Chat templates are rendered to text first, then tokenized together with add_special_tokens=False;
        this works across Transformers versions whose apply_chat_template return type differs and follows
        HF's rule not to duplicate special tokens already emitted by the template.
        """
        users = list(users)
        if not users:
            return []
        rendered = [self.tok.apply_chat_template([{"role": "user", "content": user}],
                                                 add_generation_prompt=True, tokenize=False)
                    for user in users]
        enc = self.tok(rendered, padding=True, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(DEV)
        attention_mask = enc.get("attention_mask")
        attention_mask = (attention_mask.to(DEV) if attention_mask is not None else torch.ones_like(ids))
        kw = dict(max_new_tokens=max_new, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=(self.tok.pad_token_id if self.tok.pad_token_id is not None else 0))
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        out = self.model.generate(input_ids=ids, attention_mask=attention_mask, **kw)
        replies = self.tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        return [reply.strip() for reply in replies]

    @torch.no_grad()
    def gen(self, user: str, max_new: int = 100, sample: bool = False, temperature: float = 0.9) -> str:
        """Scalar compatibility wrapper over gen_batch."""
        return self.gen_batch([user], max_new=max_new, sample=sample, temperature=temperature)[0]

    def free(self):
        self.model = None
        self.tok = None


def _generate_prompts(rig, prompts: list[str], max_new: int, batch_size: int) -> list[str]:
    """Generate prompts in stable chunks, falling back to scalar fake/legacy rigs used by pure tests."""
    replies: list[str] = []
    for batch in _chunks(prompts, batch_size):
        if hasattr(rig, "gen_batch"):
            rows = rig.gen_batch(batch, max_new=max_new)
        else:
            rows = [rig.gen(prompt, max_new=max_new) for prompt in batch]
        if len(rows) != len(batch):
            raise RuntimeError(f"generation returned {len(rows)} replies for a batch of {len(batch)} prompts")
        replies.extend(rows)
    return replies


# ---- custom dial defs: parliament.py's skeptical/plain, copied verbatim (not imported -- this codebase's
# stated precedent for a small definition shared between sibling experiment scripts; see wants_four_bit
# above, copied the same way from parliament.py/mirror_bench.py). ----
_SKEPTICAL_POS = ("Respond with skeptical, critical scrutiny: question the claims involved, flag what is "
                  "unproven, uncertain, or unverified, and do not accept assertions at face value.")
_SKEPTICAL_NEG = ("Respond with complete trust and acceptance: take all claims at face value and do not "
                  "question or doubt anything.")
_PLAIN_POS = ("Respond in plain, unembellished language: state things simply and directly, with no "
              "metaphor, no rhetorical flourish, and no stylistic decoration.")
_PLAIN_NEG = ("Respond in a highly stylized, embellished, decorative way, full of rhetorical flourish, "
              "vivid metaphor, and elaborate language.")
_CUSTOM_DIAL_DEFS = {                          # name -> (pos, neg, max)
    "skeptical": (_SKEPTICAL_POS, _SKEPTICAL_NEG, 0.5),
    "plain": (_PLAIN_POS, _PLAIN_NEG, 0.5),
}

# Captured NOW, at import time -- compute_dials() mutates the module-global steering_mod.AXES later, and
# list(...) of a dict's keys copies them, so this stays the FULL original set regardless of later narrowing.
DEFAULT_DIALS = list(steering_mod.AXES) + list(_CUSTOM_DIAL_DEFS)


def compute_dials(sc, dial_names: list[str]) -> dict:
    """Compute every requested dial's direction on sc's backbone. Built-ins (steering.AXES keys) go through
    sc.compute() -- narrowed FIRST to just the requested built-in names (parliament.py's compute_stances
    trick) so forward passes aren't burned on axes nobody asked for. sc.compute() is ALWAYS called at least
    once, even when every requested dial is a non-built-in custom one, because it is what calibrates
    sc.base/sc.resid_norm (Law #6: per-model, not a fixed default) -- skipping it entirely would leave
    add_custom() silently reusing SteeringControl.__init__'s uncalibrated base=1.0. When no built-ins were
    requested, steering_mod.AXES is left UNNARROWED so compute() still has its full default set to
    calibrate against.

    Non-built-in names are registered via sc.add_custom() if recognized (currently: skeptical, plain --
    parliament.py's two custom stances); an unrecognized name is neither a steering.AXES key nor a known
    custom and is reported in info["unknown_dials"] rather than raising, so a typo in --dials degrades to
    an honest warning + a shorter dial list, not a crash mid-sweep.

    NOTE: mutates the process-global steering_mod.AXES (never restores it) -- a one-shot script, exactly
    matching parliament.py's own compute_stances. Callers that need the ORIGINAL full AXES afterward (e.g.
    tests sharing a process with other suites) must snapshot/restore it themselves."""
    builtin_req = [d for d in dial_names if d in steering_mod.AXES]
    custom_req = [d for d in dial_names if d not in steering_mod.AXES]
    if builtin_req:
        steering_mod.AXES = {k: v for k, v in steering_mod.AXES.items() if k in builtin_req}
    info = sc.compute()
    info["custom_axes"] = {}
    unknown = []
    for dname in custom_req:
        if dname in _CUSTOM_DIAL_DEFS:
            pos, neg, mx = _CUSTOM_DIAL_DEFS[dname]
            sc.add_custom(dname, pos, neg, mx=mx)
            info["custom_axes"][dname] = {"max": mx}
        else:
            unknown.append(dname)
    info["unknown_dials"] = unknown
    return info


# =========================================================================== CAA exemplar-bank derivation
# See the module docstring's "--exemplars MODE" section for the why + the ordering requirement (compute_dials
# must run BEFORE any of this touches sc.vecs). This section is the derivation itself.

@torch.no_grad()
def _pooled_reply_resid(sc, prompt: str, reply: str) -> torch.Tensor | None:
    """The CAA primitive: read ONE reply's residual at sc.layer, IN CHAT CONTEXT, mean-pooled over the
    reply's OWN tokens only -- never the prompt's. This is deliberately a DIFFERENT read than
    SingleTurnSteer._last_resid (which reads the LAST token of an instruction+seed PROMPT, never a reply) --
    the whole point of the exemplar-bank recipe is to read the residual a finished, styled REPLY actually
    occupies, in the same chat-templated geometry steering is later applied in (see the module docstring's
    "--exemplars MODE" section for why bare-text reading would misalign with that geometry).

    MECHANICS: render two chat-template encodings --
      full_ids   = template([user: prompt, assistant: reply])                   (no add_generation_prompt)
      prefix_ids = template([user: prompt], add_generation_prompt=True)
    Both use the SAME `(enc if isinstance(enc, torch.Tensor) else enc["input_ids"])` normalization idiom
    _last_resid/Rig.gen already use elsewhere in this file (transformers 5.x returns a BatchEncoding here,
    4.x a bare tensor). The reply's own tokens are then exactly full_ids[0, prefix_ids.shape[1]:] -- for
    every chat template this codebase targets, `template([user], add_generation_prompt=True)` IS the literal
    token prefix of `template([user, assistant])` (the generation-prompt scaffolding -- e.g. an opened
    "<|im_start|>assistant\\n" -- is exactly what add_generation_prompt=True inserts, and exactly what the
    non-generation-prompt full render also inserts before the assistant's own content). Two forward passes'
    worth of tokenization, ONE forward pass (only `full_ids` is ever run through the model).

    GUARD (stated loud, not silently swallowed): if that span is EMPTY (an empty/whitespace-only reply) OR
    full_ids is no LONGER than prefix_ids (a template that does not nest the way assumed above -- has not
    been observed on any model this rig targets, but would silently pool garbage -- prompt tokens, or
    nothing -- as if it were the reply's own signal if not caught), this returns None and the CALLER
    (derive_caa_directions) is responsible for counting that as a SKIPPED pair rather than pooling it. This
    function never raises on that condition; a malformed/unusual pair degrading a whole calibration run
    would be a worse failure mode than silently, honestly dropping one pair and saying so in the count.

    Pooling is MEAN over hidden_states[sc.layer + 1][0, prefix_len:, :] -- the SAME layer-indexing
    convention _last_resid/directional_alignment already use (decoder block sc.layer's output), and the SAME
    mean-pool CHOICE directional_alignment already makes for a different reason (there: pooling a whole
    generated reply being read as a finished utterance; here: pooling a whole EXEMPLAR reply for the same
    reason) -- see directional_alignment's own docstring for the caveats that choice carries (a trait
    concentrated in one clause of a longer reply gets diluted by the tokens around it)."""
    full_text = sc.tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}], tokenize=False)
    # Locate the reply's OWN characters in the rendered text, then map that char span to tokens via the
    # fast tokenizer's offset mapping. This replaces the old prefix-length subtraction
    # (len(template([user], add_generation_prompt=True)) as the reply's start), which ASSUMED the
    # generation prompt is a literal token-prefix of the full render. MEASURED FALSE on Qwen3.5-9B, a
    # REASONING model: the generation prompt ends inside an OPEN think block ("...assistant\n<think>\n"),
    # while the full render emits a CLOSED, empty one ("...<think>\n\n</think>\n\n" + reply + "<|im_end|>").
    # Subtracting the prefix length there starts the pool on "</think>" and also swallows the trailing
    # <|im_end|>, i.e. it pools scaffold+reply, not the reply. That scaffold does NOT cancel in the pos-neg
    # difference: its hidden states are identical across the pair (same prompt, causal attention), but the
    # replies differ in LENGTH, so the shared scaffold is averaged in with different weights on each side --
    # a quiet, asymmetric contamination of exactly the vector this recipe exists to measure. Char-span
    # location is template-agnostic: think block or not, it pools the reply's tokens and nothing else.
    char_start = full_text.rfind(reply)
    if char_start < 0:
        return None   # the template normalized/mangled the reply text -- SKIP, never pool garbage
    char_end = char_start + len(reply)
    enc = sc.tok(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and b > char_start and a < char_end]
    if not idx:
        return None   # empty/whitespace reply, or no token overlapped the span -- SKIP
    ids = enc["input_ids"].to(DEV)
    hs = sc.model(ids, output_hidden_states=True).hidden_states[sc.layer + 1]
    pooled = hs[0, idx, :].float().mean(dim=0)
    del ids, hs   # free the (batch, seq, hidden) activation promptly -- see derive_caa_directions
    return pooled


@torch.no_grad()
def derive_caa_directions(sc, bank: dict, dial_names: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """Derive a CAA (contrastive-activation-addition) direction per dial from the matched-pair exemplar
    bank, for every name in `dial_names` the bank covers with >= dial_exemplars.MIN_RECOMMENDED_PAIRS pairs
    (dial_exemplars.ready(bank) is the SAME gate the bank's own --list/--validate CLI reports) -- a dial
    below that bar is simply absent from the returned dict, left for the caller (apply_caa_directions) to
    fall back to its instruction-derived vector, honestly, not silently under-averaged.

    DIVERGENCE FROM dial_exemplars.pairs_for(): that helper returns [(pos_text, neg_text), ...] tuples --
    exactly the shape a BARE-TEXT CAA derivation would consume, but it drops each pair's `prompt` field,
    which THIS derivation needs (chat-context reading is the entire point -- see the module docstring's
    "--exemplars MODE" section for why bare-text reading was rejected). So this function reads
    bank["dials"][name]["pairs"] directly (each a raw {prompt, pos, neg} dict) instead of going through
    pairs_for -- dial_exemplars.load()/.ready() are still used exactly as the brief for this feature
    specified; only pairs_for was inapplicable, because it discards the one field this recipe is built on.

    PER PAIR: pooled(pos) = _pooled_reply_resid(sc, prompt, pos_reply); pooled(neg) likewise; diff = pooled
    (pos) - pooled(neg), float32. A pair is SKIPPED (counted, not silently dropped) if `prompt` is missing/
    blank, `pos`/`neg` aren't strings, or _pooled_reply_resid returns None for EITHER pole (a partially-
    computable pair is not half-pooled -- the whole pair is excluded, so every contributing diff is a clean,
    matched pos-vs-neg contrast).

    PER DIAL: direction = mean of every usable pair's diff, then UNIT-NORMALIZED as `v / (v.norm() + 1e-8)``
    -- the IDENTICAL formula SteeringControl.compute/add_custom already use to build sc.vecs[name] (see
    hf_adapter.py: `d / (d.norm() + 1e-8)`), so a CAA-derived direction and an instruction-derived one are
    stored in EXACTLY the same convention (float32, device=DEV, shape [hidden_size], unit norm) -- the A/B
    this feature exists for is only fair if the only thing that differs is the direction's ORIENTATION, never
    its scale convention. If a dial ends up with zero usable pairs (every pair skipped -- e.g. a template
    that never nests, or an all-blank-prompt dial), it is simply absent from the returned dict, same as a
    dial that never cleared MIN_RECOMMENDED_PAIRS in the first place.

    Returns (directions, stats): `directions` is {dial_name: unit_tensor} for every dial that produced at
    least one usable pair; `stats` is {dial_name: {"pairs_used", "pairs_skipped"}} for every dial that was
    AT LEAST ATTEMPTED (i.e. cleared the MIN_RECOMMENDED_PAIRS gate), whether or not it ended up producing a
    direction -- a two-value return, one more than the brief's originally-sketched `dict[str, torch.Tensor]`
    signature, because the self-documenting run() output this feature also requires (dial_source AND a
    per-dial pairs-used/skipped count) needs both, and threading the counts back out through a second,
    mutated argument would be a worse API than just returning them.

    Runs under @torch.no_grad() throughout; intermediates (`hs`/`reply_hs`/`full_ids` inside
    _pooled_reply_resid, `diffs` here) are `del`eted as soon as they're no longer needed and _free_cuda() is
    called once per dial (not once per pair -- torch.cuda.empty_cache() is itself not free, and a per-dial
    cadence is the same granularity run()'s own per-dial checkpointing already uses) so a 9B nf4 model
    sweeping a dozen dials x ~10 pairs x 2 poles doesn't accumulate enough live activation tensors to OOM a
    16GB card."""
    directions: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}
    ready = set(dial_exemplars.ready(bank))
    for name in dial_names:
        if name not in ready:
            continue
        pairs = (bank.get("dials", {}).get(name) or {}).get("pairs", [])
        diffs: list[torch.Tensor] = []
        used = skipped = 0
        for p in pairs:
            if not isinstance(p, dict):
                skipped += 1
                continue
            prompt, pos_reply, neg_reply = p.get("prompt"), p.get("pos"), p.get("neg")
            if not (isinstance(prompt, str) and prompt.strip()
                    and isinstance(pos_reply, str) and isinstance(neg_reply, str)):
                skipped += 1
                continue
            pos_pooled = _pooled_reply_resid(sc, prompt, pos_reply)
            neg_pooled = _pooled_reply_resid(sc, prompt, neg_reply)
            if pos_pooled is None or neg_pooled is None:
                skipped += 1
                continue
            diffs.append(pos_pooled - neg_pooled)
            used += 1
        if diffs:
            mean_diff = torch.stack(diffs).mean(0)
            directions[name] = mean_diff / (mean_diff.norm() + 1e-8)
        stats[name] = {"pairs_used": used, "pairs_skipped": skipped}
        del diffs
        _free_cuda()
    return directions, stats


def apply_caa_directions(sc, dial_names: list[str],
                         directions: dict[str, torch.Tensor]) -> dict[str, str]:
    """Swap sc.vecs[name] for every dial `directions` covers, leaving every OTHER requested dial's
    instruction-derived vector (already computed by compute_dials, which MUST have run first -- see the
    module docstring's "--exemplars MODE" ordering note) untouched. This is the ONLY place sc.vecs is
    mutated for the CAA path -- derive_caa_directions itself never touches `sc` beyond reading sc.tok/
    sc.model/sc.layer, so this function is pure enough to unit-test with a bare object exposing `.vecs`.

    Returns the `dial_source` map ({name: "exemplars" | "instructions"}) for EVERY name in `dial_names`, in
    order -- run()'s self-documenting output stamps this directly into its saved report so a reader can
    never mistake which recipe produced which dial's swept direction, without re-deriving it from whether
    the dial happened to appear in `directions`."""
    source: dict[str, str] = {}
    for name in dial_names:
        if name in directions:
            sc.vecs[name] = directions[name]
            source[name] = "exemplars"
        else:
            source[name] = "instructions"
    return source


# ======================================================================================= the candidate library
def load_dial_library(path: str) -> list[dict]:
    """Pure I/O + validation, NO model: load a candidate-dial library JSON in research/dial_library_
    candidates.json's own format ({"dials": [{"name", "category", "pos", "neg", "predict"}, ...]}) and
    return its "dials" list, in file order. Raises ValueError -- fast, before any model loads -- on a
    structurally broken file: not a {"dials": [...]} shape, an entry missing a required field, or two
    entries sharing the same `name` (register_library_dials registers each by name; a silent collision
    there would make the SECOND add_custom call silently overwrite the FIRST's direction under sc.vecs/
    sc.custom, and this module would then report calibration results for only one of the two intended
    dials without ever saying so -- caught here instead, loudly, before any GPU work starts)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dials = data.get("dials") if isinstance(data, dict) else None
    if not isinstance(dials, list) or not dials:
        got = type(dials).__name__ if dials is not None else type(data).__name__
        raise ValueError(f"{path}: expected a top-level {{'dials': [...]}} non-empty list, got {got}")
    required = ("name", "category", "pos", "neg", "predict")
    for i, d in enumerate(dials):
        if not isinstance(d, dict):
            raise ValueError(f"{path}: dials[{i}] is not an object: {d!r}")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"{path}: dials[{i}] missing required field(s) {missing}: {d}")
    names = [d["name"] for d in dials]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate dial name(s) in library: {dupes}")
    return dials


_LIBRARY_DEFAULT_MAX = 1.5   # the SWEEP CEILING for library candidates (strength = frac*axis_max, frac up
                              # to 1.5 -> strength up to 2.25). Set to 1.5 (NOT add_custom's 0.5 default) so
                              # the library sweeps the SAME strength regime where the built-in tonal dials
                              # actually showed effect AND reached derail in the 12-dial run (warm/concise/etc.
                              # at max 1.5). A 0.5 ceiling caps exploration at strength 0.75 -- too weak to
                              # reach many dials' effect/derail regime, so it would silently under-dose and
                              # inflate false-"dead" verdicts. The coherence gate catches over-injection, so a
                              # wide ceiling is strictly more informative; the per-dial USABLE range the sweep
                              # discovers within [0, this] is still the output, this just sets how far it looks.


def register_library_dials(sc, library: list[dict]) -> dict:
    """Register EVERY dial spec in `library` (as returned by load_dial_library) as a custom dial on `sc`,
    via the exact same sc.add_custom(name, pos, neg, mx=...) recipe this module's own skeptical/plain
    customs and parliament.py's identical pattern already use -- just batch-driven from a big JSON file
    instead of a hand-written dict. Returns {name: {"category", "predict", "pos", "neg", "max",
    "shadows_builtin"}} in the library's own order -- list(the return value) becomes the sweep's dial_order,
    and every other field rides along so run_library can stamp it onto that dial's calibration report
    without re-reading the library file later (--report mode never needs the original library JSON at all).

    CALLER MUST have already called compute_dials/sc.compute() at least once before this (to calibrate
    sc.base/sc.resid_norm) -- add_custom computes a direction but never touches sc.base itself; see
    compute_dials's own docstring for why sc.compute() must run even when zero steering.AXES built-ins are
    requested.

    shadows_builtin=True flags a library dial whose name ALSO exists in steering.AXES (this library
    independently invented names like "warm"/"concise"/"formal"/"playful"/"poetic"/"concrete"/"confident"
    that collide with the built-in steering axes) -- add_custom overwrites sc.vecs[name] unconditionally on
    such a collision (the library's own pos/neg wins, not the built-in's), and axis_max_of's custom-first
    precedence (see its own docstring) keeps the swept ceiling consistent with that same override. Flagged,
    not prevented: a collision is not an error, just a fact worth a human noticing (run_library prints it)."""
    out = {}
    for spec in library:
        name = spec["name"]
        shadows_builtin = name in steering_mod.AXES
        sc.add_custom(name, spec["pos"], spec["neg"], mx=_LIBRARY_DEFAULT_MAX)
        out[name] = {"category": spec["category"], "predict": spec["predict"],
                     "pos": spec["pos"], "neg": spec["neg"],
                     "max": _LIBRARY_DEFAULT_MAX, "shadows_builtin": shadows_builtin}
    return out


# ================================================================================================ the sweep
_SWEEP_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
_SWEEP_FRACS_SMOKE = [0.0, 0.75, 1.5]

_DEGEN_THRESHOLD = 0.34   # a dose is flagged derailing once MORE than ~1-in-3 sampled prompts come back
                          # degenerate (counterfactual._coherence) -- copied from parliament.py's _DEGEN_OK.
                          # UNCHANGED by the direction-aware rewrite: coherence has nothing to do with the
                          # effect measure, so it needed no re-tuning.
# SCALE-RELATIVE floor: the effect metric is a raw projection magnitude that scales with the model+layer's
# residual norm, so a FIXED 2.0 that is right on the 7B (resid_norm 68.7) is ~4x too strict on Llama-3.1-8B
# @L16 (resid_norm ~11) and ~50x too strict on Gemma-4-E4B @L24 (resid_norm ~116). _scale_effect_eps() rewrites
# _EFFECT_EPS per run to a fixed FRACTION of the measured resid_norm, so "a real effect" means the same thing
# across models. _EFFECT_EPS_SCALE_K is that fraction; at the 7B reference (2.0/68.7) it reproduces 2.0 EXACTLY,
# leaving every existing Qwen2.5-7B calibration unchanged. --effect-eps pins an absolute floor and opts out.
_EFFECT_EPS_SCALE_K = 2.0 / 68.7    # 0.0291 effect-units per unit of residual norm
_EFFECT_EPS_OVERRIDE = None         # set by --effect-eps to pin an absolute floor (disables the scale rule)
_EFFECT_EPS = 2.0         # RE-TUNED for the new metric's scale (a raw projection delta, not a [0,1] Jaccard
                          # ratio -- full derivation in the module docstring's THRESHOLDS section). Picked,
                          # not derived, from this rig's own most recent real run against
                          # Qwen2.5-7B-Instruct nf4 at layer 14 (recorded steer_info.resid_norm = 68.7 in
                          # research/runs/dial_autocalibrate.json, hidden_size 3584): comfortably above a
                          # rough "uncorrelated vector" noise-floor estimate of resid_norm/sqrt(hidden_size)
                          # =~ 1.15, comfortably below the scale of an obvious tonal shift. An eyeballed,
                          # conservative cut like _DEGEN_THRESHOLD -- a module constant, not learned or fit
                          # -- but SCALE-TIED to this model/layer/quantization in a way the old dimensionless
                          # 0.03 never was: re-picking (not just re-deriving) this number is expected when
                          # this rig is run against a differently-scaled model/layer.


def _scale_effect_eps(resid_norm: float | None) -> float:
    """Rewrite the module-global _EFFECT_EPS for THIS run: an absolute pin if --effect-eps was given, else a
    fixed fraction (_EFFECT_EPS_SCALE_K) of the measured residual norm. Called once, right after compute_dials
    calibrates sc.resid_norm and before any curve is scored, so the whole sweep (and the saved effect_eps) use
    the scale-correct floor. Leaves _EFFECT_EPS at its 2.0 default when resid_norm is unavailable."""
    global _EFFECT_EPS
    if _EFFECT_EPS_OVERRIDE is not None:
        _EFFECT_EPS = float(_EFFECT_EPS_OVERRIDE)
        basis = "override (--effect-eps)"
    elif resid_norm and resid_norm > 0:
        _EFFECT_EPS = round(_EFFECT_EPS_SCALE_K * resid_norm, 4)
        basis = f"{_EFFECT_EPS_SCALE_K:.4f} x resid_norm {resid_norm:.1f}"
    else:
        basis = "default (resid_norm unavailable)"
    print(f"[eps] effect_eps = {_EFFECT_EPS}  ({basis})", flush=True)
    return _EFFECT_EPS


def _compute_calibration(curve: list[dict]) -> dict:
    """Pure function, no model, no I/O: derive derail_point / dead_below / usable_max / usable_range /
    range_valid from an already-generated `curve` (a list of {frac, real_degenerate_rate, effect,
    shuffled_effect, change_magnitude} dicts, one per swept dose, dose 0 included). Unit-testable directly
    against a fabricated curve -- this is where the "does the dial have a usable range" verdict actually
    lives; the GPU-touching calibrate_dial below only exists to produce `curve`'s numbers. UNCHANGED by the
    direction-aware rewrite: this function has always just compared `effect`/`shuffled_effect` numbers
    against thresholds, whatever those numbers mean -- it does not know or care that `effect` used to be a
    word-Jaccard ratio and is now a directional-alignment delta; `change_magnitude` (the old measure, kept
    as a diagnostic) is never read here.

    derail_point = the LOWEST frac (dose 0 included -- an unsteered model that is already degenerate on
                   this prompt sample is a real, if unlikely, finding, not a case to hide) whose
                   real_degenerate_rate exceeds _DEGEN_THRESHOLD. None if no swept dose derails.
    dead_below   = the LOWEST NONZERO frac whose effect exceeds _EFFECT_EPS. Dose 0 is excluded even though
                   its effect is always exactly 0.0 (it IS the baseline) -- "dead" here means "the dial
                   isn't doing anything yet", which is trivially true at 0 by construction, not a finding.
                   A NEGATIVE effect (the dial moved the reply toward its OPPOSITE pole -- possible now that
                   effect is a signed projection, never possible under the old [0,1]-bounded Jaccard
                   measure) also fails "> _EFFECT_EPS" and is correctly never picked here.
    usable_max   = the HIGHEST frac that is simultaneously: (a) coherent (real_degenerate_rate <=
                   _DEGEN_THRESHOLD), (b) has a real effect (> _EFFECT_EPS), AND (c) beats the shuffled
                   null AT THE SAME DOSE (effect > shuffled_effect) -- "the dial moved the reply toward its
                   own pole", not "any perturbation this size did", and not "the wording merely changed"
                   (change_magnitude plays no part in this test). None if no dose satisfies all three.
    usable_range = [dead_below, usable_max] (either or both may be None -- see range_valid).
    range_valid  = True iff both ends are present AND ordered (dead_below <= usable_max). A dial that never
                   clears the null/coherence bar at all, or only ever has an effect below _EFFECT_EPS --
                   INCLUDING a dial that changes the WORDING a great deal (a high change_magnitude) while
                   never moving toward its own pole, the reformat-not-steering case this module was
                   rewritten to catch -- reports range_valid=False. Read that as "no honestly-calibrated
                   usable range on this sample", never as a zero-width range at some arbitrary point.
    """
    derail_point = next((c["frac"] for c in curve if c["real_degenerate_rate"] > _DEGEN_THRESHOLD), None)
    dead_below = next((c["frac"] for c in curve if c["frac"] > 0 and c["effect"] > _EFFECT_EPS), None)
    usable_fracs = [c["frac"] for c in curve
                    if c["frac"] > 0
                    and c["real_degenerate_rate"] <= _DEGEN_THRESHOLD
                    and c["effect"] > _EFFECT_EPS
                    and c["effect"] > c["shuffled_effect"]]
    usable_max = max(usable_fracs) if usable_fracs else None
    range_valid = dead_below is not None and usable_max is not None and dead_below <= usable_max
    return {
        "derail_point": derail_point,
        "dead_below": dead_below,
        "usable_max": usable_max,
        "usable_range": [dead_below, usable_max],
        "range_valid": range_valid,
    }


def calibrate_dial(rig, sc, name: str, prompts: list[str], fracs: list[float], seed: int,
                   max_new: int = 100, batch_size: int = 1,
                   baseline_texts: list[str] | None = None) -> dict:
    """The heart of this module: sweep dial `name` over `fracs` (each a fraction of axis_max_of(sc, name))
    on `prompts`, against a matched-norm SHUFFLED-direction null at the IDENTICAL magnitude, at every dose.

    STRENGTH IS WRITTEN DIRECTLY INTO sc.strength[...], NOT via SteeringControl.set() (deliberate, and
    worth stating loud): .set() clamps its argument into [-axis_max, axis_max], which is exactly the
    ceiling this sweep needs to go PAST (fracs run up to 1.5x axis_max, by design, to find where a dial
    derails BEYOND its documented "safe" max) -- going through .set() would silently clamp frac=1.0/1.25/1.5
    down to the SAME strength, making every dose past 1.0x indistinguishable and derail_point unreachable.
    The shuffled null needs the identical bypass for the identical reason (parliament.py's own null already
    does this, for the same reason, at its narrower fracs<=1.0 sweep).

    At frac=0.0: the caller's shared unsteered baseline (or ONE greedy decode per prompt when called
    standalone) is both "the real arm" and "the shuffled arm" -- steering off is steering off, whichever
    vector isn't there. These `baseline_texts` are the fixed reference point for BOTH directional_effect
    and effect_vs_baseline at EVERY other dose (never a moving target). Their directional alignments are
    computed once per dial and cached across all real/shuffled dose comparisons.

    At each nonzero frac: real-direction decodes, then (after a full clear/disengage) shuffled-direction
    decodes, both at the SAME |strength| -- so directional_effect, effect_vs_baseline, and degenerate_rate
    are all computed on a like-for-like pair at every dose.

    Each curve row carries THREE effect-shaped numbers, deliberately, so a reader sees them together (see
    the module docstring's THE EFFECT MEASURE section):
      * effect             -- directional_effect(sc, name, baseline_texts, real_texts): the NEW,
                               direction-aware measure. Drives dead_below/usable_max/usable_range.
      * shuffled_effect    -- directional_effect(sc, name, baseline_texts, shuf_texts): the SAME projection,
                               applied to the shuffled-direction arm's replies, onto the SAME real dial axis.
      * change_magnitude   -- effect_vs_baseline(baseline_texts, real_texts): the OLD word-Jaccard measure,
                               kept as a diagnostic only (see effect_vs_baseline's own docstring) -- NOT
                               read by _compute_calibration.

    Returns {dial, axis_max, curve, derail_point, dead_below, usable_max, usable_range, range_valid,
    sample_replies} -- see _compute_calibration for exactly how the four calibration numbers are derived
    from `curve`, and the module docstring for the effect measure + thresholds' definitions and caveats.
    `sample_replies` keeps, per dose, the FIRST prompt's (prompt, baseline reply, steered reply) triple --
    enough for a human to eyeball whether a dose flagged "usable" is genuinely on-character, not a
    self-referential quirk of projecting onto the dial's own diff-of-means direction (see the module
    docstring's SELF-REFERENTIAL caveat).
    """
    axis_max = axis_max_of(sc, name)
    shuffle_vec = make_shuffle_unit_vector(sc.vecs[name], _dial_seed(seed, name))

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if baseline_texts is not None and len(baseline_texts) != len(prompts):
        raise ValueError("baseline_texts must match prompts length")
    baseline_texts = list(baseline_texts) if baseline_texts is not None else None
    baseline_scores: list[float] | None = None
    curve: list[dict] = []
    sample_replies: list[dict] = []
    for frac in fracs:
        strength = round(frac * axis_max, 4)
        sc.disengage()
        sc.clear()
        if frac == 0.0:
            if baseline_texts is None:
                baseline_texts = _generate_prompts(rig, prompts, max_new=max_new, batch_size=batch_size)
            real_texts = baseline_texts
            shuf_texts = real_texts        # steering off either way at frac=0 -- identical by construction
            baseline_scores = directional_alignments(sc, baseline_texts, name, batch_size=batch_size)
            real_effect = 0.0
            shuffled_effect = 0.0
        else:
            sc.strength[name] = strength   # direct write -- bypasses .set()'s clamp to axis_max (see above)
            sc.engage()
            real_texts = _generate_prompts(rig, prompts, max_new=max_new, batch_size=batch_size)
            sc.disengage()
            sc.clear()

            sc.vecs["_shuf_tmp"] = shuffle_vec
            sc.strength["_shuf_tmp"] = strength    # the null must land at EXACTLY the real dial's magnitude
            sc.engage()
            shuf_texts = _generate_prompts(rig, prompts, max_new=max_new, batch_size=batch_size)
            sc.disengage()
            sc.clear()
            del sc.vecs["_shuf_tmp"]

            # baseline_scores is computed once at dose zero and reused for both arms at every later dose.
            real_effect = directional_effect(
                sc, name, baseline_texts, real_texts,
                batch_size=batch_size, baseline_alignments=baseline_scores,
            )
            shuffled_effect = directional_effect(
                sc, name, baseline_texts, shuf_texts,
                batch_size=batch_size, baseline_alignments=baseline_scores,
            )

        curve.append({
            "frac": frac, "strength": strength,
            "real_degenerate_rate": degenerate_rate(real_texts),
            "shuffled_degenerate_rate": degenerate_rate(shuf_texts),
            "effect": real_effect,
            "shuffled_effect": shuffled_effect,
            "change_magnitude": effect_vs_baseline(baseline_texts, real_texts),
        })
        sample_replies.append({
            "frac": frac, "prompt": prompts[0] if prompts else "",
            "baseline_reply": baseline_texts[0] if baseline_texts else "",
            "steered_reply": real_texts[0] if real_texts else "",
        })

    calib = _compute_calibration(curve)
    return {"dial": name, "axis_max": axis_max, "curve": curve, "sample_replies": sample_replies, **calib}


_RESUME_COMPAT_KEYS = (
    "run_kind", "model", "four_bit", "seed", "smoke", "n_prompts", "prompt_source", "prompts",
    "n_prompts_requested", "prompts_file_path", "max_prompts", "max_new", "batch_size", "steer_layer",
    "sweep_fracs", "dial_order", "exemplars_path", "exemplars_sha256", "dial_source", "caa_pairs_used",
    "caa_pairs_skipped", "library_path", "dial_meta",
)


def _read_resume_checkpoint(out_path: str) -> dict:
    if not os.path.isfile(out_path):
        raise SystemExit(f"--resume requested but checkpoint does not exist: {out_path}")
    try:
        with open(out_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot resume unreadable checkpoint {out_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"cannot resume checkpoint {out_path}: root must be a JSON object")
    return checkpoint


def _resume_prompts(checkpoint: dict, out_path: str) -> tuple[list[str], str]:
    """Use the checkpoint's literal prompt sample; a changing runlog must not mutate a resumed experiment."""
    prompts = checkpoint.get("prompts")
    source = checkpoint.get("prompt_source")
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise SystemExit(f"cannot resume checkpoint {out_path}: prompts must be a JSON string array")
    if not prompts or checkpoint.get("n_prompts") != len(prompts) or not isinstance(source, str):
        raise SystemExit(f"cannot resume checkpoint {out_path}: invalid prompt metadata")
    return list(prompts), source


def _resume_from_checkpoint(out_path: str, fresh: dict, checkpoint: dict | None = None) -> tuple[dict, float]:
    """Load completed dials only when every result-affecting run setting matches exactly.

    Strict compatibility is deliberate: silently mixing prompt banks, steering layers, quantization,
    batch shapes, or direction recipes would make one JSON look like a single experiment when it was not.
    """
    checkpoint = checkpoint if checkpoint is not None else _read_resume_checkpoint(out_path)

    mismatches = [key for key in _RESUME_COMPAT_KEYS if checkpoint.get(key) != fresh.get(key)]
    if mismatches:
        raise SystemExit(
            f"cannot resume incompatible checkpoint {out_path}; differing field(s): {', '.join(mismatches)}"
        )
    old_dials = checkpoint.get("dials")
    if not isinstance(old_dials, dict):
        raise SystemExit(f"cannot resume checkpoint {out_path}: dials must be a JSON object")
    unknown = [name for name in old_dials if name not in fresh["dial_order"]]
    if unknown:
        raise SystemExit(f"cannot resume checkpoint {out_path}: unknown completed dial(s): {unknown}")

    fresh["dials"] = {name: old_dials[name] for name in fresh["dial_order"] if name in old_dials}
    try:
        prior_elapsed = float(checkpoint.get(
            "elapsed_wall_clock_sec", checkpoint.get("wall_clock_sec", 0.0)
        ) or 0.0)
    except (TypeError, ValueError):
        prior_elapsed = 0.0
    fresh["elapsed_wall_clock_sec"] = round(prior_elapsed, 1)
    fresh["resume_count"] = int(checkpoint.get("resume_count", 0) or 0) + 1
    print(f"[resume] {len(fresh['dials'])}/{len(fresh['dial_order'])} completed dial(s) loaded from "
          f"{out_path}", flush=True)
    return fresh, prior_elapsed


# ================================================================================================= run
def run(model_name: str, dials: list[str] | None = None, n_prompts: int = 6,
        out_path: str = "research/runs/dial_autocalibrate.json", four_bit_override: str = "auto",
        smoke: bool = False, seed: int = 0, layer: int | None = None, max_new: int = 100,
        prompts_file: str | None = None, max_prompts: int | None = None,
        exemplars: str | None = None, batch_size: int = 8, resume: bool = False) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    torch.manual_seed(seed)
    dial_names = list(dials) if dials else list(DEFAULT_DIALS)
    if smoke:
        dial_names = dial_names[:2]        # --smoke always caps to 1-2 dials, whatever was requested
    n_eff = 2 if smoke else n_prompts      # --smoke always uses 2 prompts
    fracs = _SWEEP_FRACS_SMOKE if smoke else _SWEEP_FRACS
    resume_checkpoint = _read_resume_checkpoint(out_path) if resume else None

    # A curated --prompts-file wins over the runlog sample (and ignores n_prompts): a fixed, model-
    # independent bank is what makes two models' calibrations comparable. --smoke still overrides to keep
    # the wiring-check cheap regardless of how big the file is.
    if resume_checkpoint is not None:
        prompts, prompt_source = _resume_prompts(resume_checkpoint, out_path)
    elif prompts_file and not smoke:
        prompts, prompt_source = load_prompts_file(prompts_file)
        if max_prompts and 0 < max_prompts < len(prompts):
            # Evenly-spaced subsample so a grouped-by-register bank stays balanced (take 1 from each
            # stretch, not the first N -- which would over-weight whatever group leads the file).
            idx = sorted({round(i * (len(prompts) - 1) / (max_prompts - 1)) for i in range(max_prompts)})
            prompts = [prompts[j] for j in idx]
            prompt_source += f"[{len(prompts)}/{'orig'}]"
    else:
        prompts, prompt_source = sample_prompts(n_eff, seed=seed)
    print(f"[prompts] {len(prompts)} prompt(s) from {prompt_source}", flush=True)

    rig = Rig(model_name, four_bit_override)
    sc = SingleTurnSteer(rig.model, rig.tok, layer=layer)
    print(f"[dials] computing {len(dial_names)} dial direction(s) at layer {sc.layer}: {dial_names}",
          flush=True)
    steer_info = compute_dials(sc, dial_names)
    print(f"[dials] {steer_info}", flush=True)
    _scale_effect_eps(steer_info.get("resid_norm"))   # scale the effect floor to THIS model's residual norm
    if steer_info["unknown_dials"]:
        print(f"[warn] unknown dial(s) ignored (not a steering.AXES built-in or a registered custom dial "
              f"-- known customs: {sorted(_CUSTOM_DIAL_DEFS)}): {steer_info['unknown_dials']}", flush=True)
    dial_names = [d for d in dial_names if d in sc.vecs]
    if not dial_names:
        raise SystemExit("no valid dials left to calibrate after filtering unknown --dials names")

    # --exemplars: ONLY after compute_dials has already run (sc.base/sc.resid_norm are calibrated above,
    # unconditionally, exactly as they were before this feature existed) does the CAA recipe get a chance to
    # REPLACE sc.vecs[name] for whichever dials the bank covers -- see the module docstring's "--exemplars
    # MODE" section for why that ordering is load-bearing. dial_source/caa_pairs_used/caa_pairs_skipped are
    # always present in the saved report (even when --exemplars was never passed, in which case every dial
    # is honestly "instructions" and every count is 0) so the JSON schema never silently changes shape
    # between an A and a B run -- the whole point is that the two are diffable side by side.
    caa_stats: dict[str, dict] = {}
    if exemplars:
        bank = dial_exemplars.load(exemplars)
        caa_directions, caa_stats = derive_caa_directions(sc, bank, dial_names)
        dial_source = apply_caa_directions(sc, dial_names, caa_directions)
        n_from_exemplars = len(caa_directions)
        n_fallback = len(dial_names) - n_from_exemplars
        warm_pairs = ", ".join(f"{n}:{s['pairs_used']}" for n, s in sorted(caa_stats.items()))
        print(f"[caa] derived {n_from_exemplars} dial(s) from {os.path.basename(exemplars)} "
              f"(pairs used -- {warm_pairs}); {n_fallback} fell back to instructions", flush=True)
    else:
        dial_source = {name: "instructions" for name in dial_names}

    res = {
        "run_kind": "dials", "model": model_name, "four_bit": rig.four_bit, "seed": seed, "smoke": smoke,
        "n_prompts": len(prompts), "prompt_source": prompt_source, "prompts": prompts,
        "n_prompts_requested": n_prompts, "prompts_file_path": prompts_file, "max_prompts": max_prompts,
        "max_new": max_new, "batch_size": batch_size, "steer_layer": sc.layer, "steer_info": steer_info,
        "sweep_fracs": fracs, "degen_threshold": _DEGEN_THRESHOLD, "effect_eps": _EFFECT_EPS,
        "exemplars_path": exemplars, "exemplars_sha256": _file_sha256(exemplars),
        "dial_source": dial_source,
        "caa_pairs_used": {n: caa_stats.get(n, {}).get("pairs_used", 0) for n in dial_names},
        "caa_pairs_skipped": {n: caa_stats.get(n, {}).get("pairs_skipped", 0) for n in dial_names},
        "dial_order": dial_names, "dials": {},
    }
    prior_elapsed = 0.0
    if resume:
        res, prior_elapsed = _resume_from_checkpoint(out_path, res, checkpoint=resume_checkpoint)
    else:
        _save(out_path, res)

    pending = [name for name in dial_names if name not in res["dials"]]
    shared_baseline = None
    if pending:
        sc.disengage()
        sc.clear()
        print(f"[baseline] generating {len(prompts)} shared unsteered replies in batches of {batch_size} ...",
              flush=True)
        shared_baseline = _generate_prompts(rig, prompts, max_new=max_new, batch_size=batch_size)

    print(f"[sweep] {len(pending)} pending of {len(dial_names)} dial(s) x {len(fracs)} doses x "
          f"{len(prompts)} prompts (batch_size={batch_size}) ...", flush=True)
    t0 = time.time()
    for name in dial_names:
        if name in res["dials"]:
            print(f"  [{name}] checkpoint complete -- skipped", flush=True)
            continue
        report = calibrate_dial(
            rig, sc, name, prompts, fracs, seed=seed, max_new=max_new,
            batch_size=batch_size, baseline_texts=shared_baseline,
        )
        res["dials"][name] = report
        res["elapsed_wall_clock_sec"] = round(prior_elapsed + time.time() - t0, 1)
        _save(out_path, res)
        print(f"  [{name}] usable_range={report['usable_range']} derail_point={report['derail_point']} "
              f"range_valid={report['range_valid']}", flush=True)
    res["wall_clock_sec"] = round(prior_elapsed + time.time() - t0, 1)
    res["elapsed_wall_clock_sec"] = res["wall_clock_sec"]

    sc.disengage()
    rig.free()
    del sc, rig
    _free_cuda()

    _summary(res)
    _save(out_path, res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _save(out_path, res):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def _summary(res):
    print("\n" + "=" * 78, flush=True)
    print(f"DIAL AUTO-CALIBRATION -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'}) -- "
          f"{res['n_prompts']} prompt(s) from {res['prompt_source']}", flush=True)
    # "source" (added alongside --exemplars) prints alongside the calibration numbers so an A/B reader never
    # has to cross-reference res["dial_source"] separately to know which recipe produced which dial's row --
    # see the module docstring's "--exemplars MODE" section.
    dial_source = res.get("dial_source", {})
    print(f"\n{'dial':12} {'dead_below':11} {'usable_max':11} {'derail_at':10} {'valid':6} {'source':12}",
          flush=True)
    for name in res["dial_order"]:
        c = res["dials"][name]
        print(f"{name:12} {str(c['dead_below']):11} {str(c['usable_max']):11} "
              f"{str(c['derail_point']):10} {str(c['range_valid']):6} "
              f"{dial_source.get(name, 'instructions'):12}", flush=True)
    invalid = [n for n in res["dial_order"] if not res["dials"][n]["range_valid"]]
    if invalid:
        print(f"\nno honestly-calibrated usable range on this sample: {invalid}", flush=True)
    print("\nNOTE: this is a RANGE calibration, not a recommended single value -- see the module docstring.",
          flush=True)


# ========================================================================================= the library sweep
def run_library(library_path: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct", n_prompts: int = 6,
                out_path: str = "research/runs/dial_library_sweep.json", four_bit_override: str = "auto",
                smoke: bool = False, seed: int = 0, layer: int | None = None, max_new: int = 100,
                batch_size: int = 8, resume: bool = False) -> dict:
    """--library mode: sweep an entire CANDIDATE LIBRARY (research/dial_library_candidates.json's ~70-dial,
    15-category format) through the IDENTICAL per-dial calibration path run() uses for steering.AXES/
    --dials -- same calibrate_dial, same _compute_calibration, same direction-aware effect measure, same
    shuffled null (see the module docstring for all of those). Two differences from run(): (1) every dial
    comes from add_custom via register_library_dials -- there is no steering.AXES built-in path here, only
    customs (even a library name that collides with a built-in -- see register_library_dials/axis_max_of);
    and (2) each dial's calibration report additionally carries its library `category`/`predict`/`pos`/
    `neg`, so --report can group/verdict/re-ship on them without ever re-reading the library file.

    sc.base/sc.resid_norm are calibrated via compute_dials(sc, []) BEFORE any library dial is registered --
    passing an empty dial_names list makes compute_dials leave steering_mod.AXES unnarrowed (so sc.compute()
    still has its full built-in set to average a resid_norm over, avoiding a divide-by-zero on an empty
    AXES -- see compute_dials's own docstring), which as a side effect also computes the 10 stock AXES
    directions into sc.vecs. Those 10 are never swept here (dial_order only ever lists library names) and
    the cost is negligible (~360 short forward passes, no generation) next to the ~70-dial sweep itself --
    an accepted, pre-existing cost this module already pays for ANY --dials invocation that requests zero
    built-ins (e.g. `--dials skeptical plain`), not something --library newly introduces.

    THIS IS A BIG RUN (~70 dials x 7 doses x n_prompts x ~2 generations -- an order of magnitude more than
    the default sweep), so, exactly like run(), the JSON at `out_path` is CHECKPOINT-SAVED after EVERY
    dial's calibration completes, not just at the end: a kill or OOM partway through still leaves every
    already-finished dial's full curve/calibration on disk, never just whatever was saved before the LAST
    dial that happened to be in flight when the process died. --resume reloads those completed rows after
    validating that every result-affecting setting matches the checkpoint."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    torch.manual_seed(seed)
    library = load_dial_library(library_path)
    if smoke:
        library = library[:2]          # --smoke caps to the first 2 library entries, matching run()'s cap
    n_eff = 2 if smoke else n_prompts
    fracs = _SWEEP_FRACS_SMOKE if smoke else _SWEEP_FRACS
    resume_checkpoint = _read_resume_checkpoint(out_path) if resume else None

    if resume_checkpoint is not None:
        prompts, prompt_source = _resume_prompts(resume_checkpoint, out_path)
    else:
        prompts, prompt_source = sample_prompts(n_eff, seed=seed)
    print(f"[prompts] {len(prompts)} prompt(s) from {prompt_source}", flush=True)

    rig = Rig(model_name, four_bit_override)
    sc = SingleTurnSteer(rig.model, rig.tok, layer=layer)
    print(f"[library] {len(library)} candidate dial(s) loaded from {library_path}", flush=True)
    steer_info = compute_dials(sc, [])     # calibrates sc.base/resid_norm; no built-ins requested for sweep
    _scale_effect_eps(steer_info.get("resid_norm"))   # scale the effect floor to THIS model's residual norm
    lib_meta = register_library_dials(sc, library)
    dial_names = list(lib_meta)
    print(f"[library] registered {len(dial_names)} custom dial(s) (max={_LIBRARY_DEFAULT_MAX} each)",
          flush=True)
    shadowed = sorted(n for n, m in lib_meta.items() if m["shadows_builtin"])
    if shadowed:
        print(f"[warn] {len(shadowed)} library dial name(s) shadow a steering.AXES built-in -- the "
              f"library's own pos/neg + max win (axis_max_of is custom-first): {shadowed}", flush=True)

    res = {
        "run_kind": "library", "model": model_name, "four_bit": rig.four_bit, "seed": seed, "smoke": smoke,
        "n_prompts": len(prompts), "prompt_source": prompt_source, "prompts": prompts,
        "n_prompts_requested": n_prompts, "prompts_file_path": None, "max_prompts": None,
        "max_new": max_new, "batch_size": batch_size, "steer_layer": sc.layer, "steer_info": steer_info,
        "library_path": library_path, "sweep_fracs": fracs,
        "degen_threshold": _DEGEN_THRESHOLD, "effect_eps": _EFFECT_EPS,
        "dial_order": dial_names, "dial_meta": lib_meta, "dials": {},
    }
    prior_elapsed = 0.0
    if resume:
        res, prior_elapsed = _resume_from_checkpoint(out_path, res, checkpoint=resume_checkpoint)
    else:
        _save(out_path, res)

    pending = [name for name in dial_names if name not in res["dials"]]
    shared_baseline = None
    if pending:
        sc.disengage()
        sc.clear()
        print(f"[baseline] generating {len(prompts)} shared unsteered replies in batches of {batch_size} ...",
              flush=True)
        shared_baseline = _generate_prompts(rig, prompts, max_new=max_new, batch_size=batch_size)

    print(f"[sweep] {len(pending)} pending of {len(dial_names)} dial(s) x {len(fracs)} doses x "
          f"{len(prompts)} prompts (batch_size={batch_size}) ...", flush=True)
    t0 = time.time()
    for i, name in enumerate(dial_names):
        if name in res["dials"]:
            print(f"  [{i + 1}/{len(dial_names)}] [{name}] checkpoint complete -- skipped", flush=True)
            continue
        report_row = calibrate_dial(
            rig, sc, name, prompts, fracs, seed=seed, max_new=max_new,
            batch_size=batch_size, baseline_texts=shared_baseline,
        )
        report_row["category"] = lib_meta[name]["category"]
        report_row["predict"] = lib_meta[name]["predict"]
        report_row["pos"] = lib_meta[name]["pos"]
        report_row["neg"] = lib_meta[name]["neg"]
        res["dials"][name] = report_row
        res["elapsed_wall_clock_sec"] = round(prior_elapsed + time.time() - t0, 1)
        _save(out_path, res)      # checkpoint after EVERY dial -- a kill/OOM keeps every finished one
        print(f"  [{i + 1}/{len(dial_names)}] [{name}] ({report_row['category']}/{report_row['predict']}) "
              f"usable_range={report_row['usable_range']} derail_point={report_row['derail_point']} "
              f"range_valid={report_row['range_valid']}", flush=True)
    res["wall_clock_sec"] = round(prior_elapsed + time.time() - t0, 1)
    res["elapsed_wall_clock_sec"] = res["wall_clock_sec"]

    sc.disengage()
    rig.free()
    del sc, rig
    _free_cuda()

    _summary_by_category(res)
    _save(out_path, res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _summary_by_category(res):
    print("\n" + "=" * 78, flush=True)
    print(f"DIAL LIBRARY SWEEP -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'}) -- "
          f"{res['n_prompts']} prompt(s) from {res['prompt_source']} -- {len(res['dial_order'])} dial(s)",
          flush=True)
    by_cat: dict = {}
    for name in res["dial_order"]:
        cat = res["dials"][name].get("category", "?")
        by_cat.setdefault(cat, []).append(name)
    for cat in sorted(by_cat):
        names = by_cat[cat]
        print(f"\n-- {cat} ({len(names)}) --", flush=True)
        print(f"{'dial':20} {'predict':10} {'dead_below':11} {'usable_max':11} {'derail_at':10} {'valid':6}",
              flush=True)
        for name in names:
            c = res["dials"][name]
            print(f"{name:20} {c.get('predict', '?'):10} {str(c['dead_below']):11} "
                  f"{str(c['usable_max']):11} {str(c['derail_point']):10} {str(c['range_valid']):6}",
                  flush=True)
    n_valid = sum(1 for n in res["dial_order"] if res["dials"][n]["range_valid"])
    print(f"\n{n_valid}/{len(res['dial_order'])} dial(s) got a real, honestly-calibrated usable range.",
          flush=True)
    print("\nNOTE: this is a RANGE calibration, not a recommended single value -- see the module docstring.",
          flush=True)


# =========================================================================================== --report mode
def _dial_mean_effect(dial_report: dict) -> float:
    """Pure, no model: mean of `effect` over every NONZERO-frac row in one dial's curve -- dose 0 is the
    baseline itself (effect always exactly 0.0 there by construction), excluded here for the same reason
    _compute_calibration's dead_below/usable_max only ever read nonzero-frac rows. 0.0 for a dial with no
    curve, or only its dose-0 row (never raises on a missing/malformed field)."""
    curve = dial_report.get("curve") or []
    vals = [c["effect"] for c in curve if c.get("frac", 0) > 0 and "effect" in c]
    return round(_mean(vals), 4) if vals else 0.0


def category_summary(sweep: dict) -> dict:
    """Pure function, no model/GPU: per-CATEGORY {n_usable, n_total, usable_rate, mean_effect} from a
    completed --library sweep JSON's "dials" map -- categories are read off each dial's own "category" field
    (stamped there by run_library), never re-derived from the original library file, so --report only ever
    needs the sweep JSON. mean_effect is the mean, across the category's dials, of each dial's OWN mean
    nonzero-dose effect (_dial_mean_effect) -- a category with a few strongly-steering dials and a few dead
    ones still shows an honest middling average, not just a pass/fail count. Sorted by category name for a
    stable report order."""
    dials = sweep.get("dials", {})
    by_cat: dict = {}
    for d in dials.values():
        by_cat.setdefault(d.get("category", "?"), []).append(d)
    out = {}
    for cat, rows in sorted(by_cat.items()):
        n_total = len(rows)
        n_usable = sum(1 for r in rows if r.get("range_valid"))
        mean_effect = round(_mean(_dial_mean_effect(r) for r in rows), 4) if rows else 0.0
        out[cat] = {"n_usable": n_usable, "n_total": n_total,
                    "usable_rate": round(n_usable / n_total, 3) if n_total else 0.0,
                    "mean_effect": mean_effect}
    return out


def hypothesis_verdict(sweep: dict) -> dict:
    """Pure function, no model/GPU: the candidate library's OWN pre-registered hypothesis (dial_library_
    candidates.json's _about/_predict_legend -- SURFACE-expressed qualities steer well, ABSTRACT-COGNITIVE
    stances don't), checked against a completed --library sweep: the usable RATE (n range_valid=True / n
    total) for predict="surface" dials vs predict="cognitive" dials, plus the gap (surface - cognitive).
    predict="uncertain" dials are reported (their own n/usable/rate) but excluded from the gap -- the
    library's own docstring calls that tag out as "the interesting middle", a deliberate third bucket, not a
    second hypothesis to average in. hypothesis_holds is True iff BOTH rates are defined and surface's rate
    is strictly higher -- a plain directional check, not a significance test (n-per-category is small and
    this sweep draws one sample of prompts; read the gap's SIZE, not just this boolean)."""
    dials = sweep.get("dials", {})
    by_predict: dict = {}
    for d in dials.values():
        by_predict.setdefault(d.get("predict", "?"), []).append(bool(d.get("range_valid")))

    def _bucket(tag):
        rows = by_predict.get(tag, [])
        n_usable = sum(rows)
        rate = round(n_usable / len(rows), 3) if rows else None
        return {"n_usable": n_usable, "n_total": len(rows), "usable_rate": rate}

    surface, cognitive, uncertain = _bucket("surface"), _bucket("cognitive"), _bucket("uncertain")
    sr, cr = surface["usable_rate"], cognitive["usable_rate"]
    gap = round(sr - cr, 3) if sr is not None and cr is not None else None
    return {
        "surface": surface, "cognitive": cognitive, "uncertain": uncertain,
        "gap_surface_minus_cognitive": gap,
        "hypothesis_holds": bool(gap is not None and gap > 0),
    }


def curated_library(sweep: dict) -> list[dict]:
    """Pure function, no model/GPU: the SHIPPABLE subset of a completed --library sweep -- every dial with
    range_valid=True (an honestly-calibrated, coherent, null-beating usable range on this sample; see
    _compute_calibration), reduced to exactly what a deploy-time caller needs to re-register and cap the
    SAME dial on a live model: name, category, usable_range ([dead_below, usable_max]), derail_point, and
    the pos/neg poles add_custom needs to recompute the identical direction. A dial with range_valid=False
    is silently dropped here -- not an error, see category_summary/hypothesis_verdict for what happened to
    it instead. Sorted by (category, name) -- matches the console report's own category grouping, name as a
    stable tie-breaker."""
    dials = sweep.get("dials", {})
    out = []
    for name, d in dials.items():
        if not d.get("range_valid"):
            continue
        out.append({"name": name, "category": d.get("category", "?"),
                    "usable_range": d.get("usable_range"), "derail_point": d.get("derail_point"),
                    "pos": d.get("pos"), "neg": d.get("neg")})
    out.sort(key=lambda r: (r["category"], r["name"]))
    return out


def report(sweep_path: str, curated_out: str = "research/runs/dial_library_curated.json") -> dict:
    """--report mode: PURE ANALYSIS, NO model/GPU -- reads a completed --library sweep JSON (produced by
    run_library) and prints (a) the per-CATEGORY summary (category_summary), (b) the pre-registered
    surface-vs-cognitive HYPOTHESIS VERDICT (hypothesis_verdict), and (c) the CURATED shippable list
    (curated_library) -- then writes that same curated list to `curated_out` as {"dials": [...]}, the file
    research/clozn_server.py's dial-calibration curator step is meant to read from (only range_valid=True
    winners -- see curated_library). Returns {"category_summary", "hypothesis", "curated"} so a caller (or a
    test) can assert on the numbers directly, without re-parsing stdout."""
    with open(sweep_path, encoding="utf-8") as f:
        sweep = json.load(f)

    cat_summary = category_summary(sweep)
    hyp = hypothesis_verdict(sweep)
    curated = curated_library(sweep)

    print("\n" + "=" * 78, flush=True)
    print(f"DIAL LIBRARY REPORT -- {sweep.get('model', '?')} -- "
          f"{len(sweep.get('dial_order', []))} dial(s) swept from {sweep.get('library_path', '?')}",
          flush=True)

    print(f"\n{'category':22} {'n_usable/n_total':17} {'usable_rate':12} {'mean_effect':11}", flush=True)
    for cat, s in cat_summary.items():
        frac = f"{s['n_usable']}/{s['n_total']}"
        print(f"{cat:22} {frac:17} {s['usable_rate']:<12} {s['mean_effect']:<11}", flush=True)

    print("\nHYPOTHESIS VERDICT (surface-expressed dials predicted to usable-calibrate more often than "
          "cognitive-stance ones):", flush=True)
    surf, cog, unc = hyp["surface"], hyp["cognitive"], hyp["uncertain"]
    print(f"  surface:   {surf['n_usable']}/{surf['n_total']} usable  (rate={surf['usable_rate']})", flush=True)
    print(f"  cognitive: {cog['n_usable']}/{cog['n_total']} usable  (rate={cog['usable_rate']})", flush=True)
    print(f"  uncertain: {unc['n_usable']}/{unc['n_total']} usable  (rate={unc['usable_rate']}) -- reported, "
          f"not part of the gap (the library's own deliberate middle)", flush=True)
    print(f"  gap (surface - cognitive) = {hyp['gap_surface_minus_cognitive']} -- hypothesis "
          f"{'HOLDS' if hyp['hypothesis_holds'] else 'DOES NOT HOLD'} on this sweep", flush=True)

    print(f"\nCURATED SHIPPABLE LIST ({len(curated)} dial(s) with a real, honestly-calibrated usable range):",
          flush=True)
    last_cat = None
    for r in curated:
        if r["category"] != last_cat:
            print(f"\n-- {r['category']} --", flush=True)
            last_cat = r["category"]
        print(f"  {r['name']:20} usable_range={r['usable_range']} derail_point={r['derail_point']}",
              flush=True)

    os.makedirs(os.path.dirname(curated_out) or ".", exist_ok=True)
    with open(curated_out, "w", encoding="utf-8") as f:
        json.dump({"dials": curated}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved curated library ({len(curated)} dial(s)) -> {curated_out}", flush=True)

    return {"category_summary": cat_summary, "hypothesis": hyp, "curated": curated}


def _default_out_path(library: str | None) -> str:
    """Pure, no I/O: --out's default depends on which mode is running -- the library sweep's own, much
    larger, output file (research/runs/dial_library_sweep.json) vs the plain --dials/built-in sweep's
    (research/runs/dial_autocalibrate.json). An explicit --out always overrides either default; this is
    only what --out defaults TO when the caller didn't pass one."""
    return "research/runs/dial_library_sweep.json" if library else "research/runs/dial_autocalibrate.json"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dials", nargs="+", default=None,
                    help="dial names to calibrate (default: every steering.AXES built-in + skeptical/plain)")
    ap.add_argument("--library", default=None, metavar="LIBRARY.json",
                    help="load a candidate-dial library (research/dial_library_candidates.json's "
                         "{'dials':[{name,category,pos,neg,predict}]} format) and sweep EVERY entry as a "
                         "custom dial, instead of steering.AXES/--dials -- checkpoint-saved after each dial")
    ap.add_argument("--report", default=None, metavar="SWEEP.json",
                    help="pure analysis, NO model/GPU: read a completed --library sweep JSON and print the "
                         "per-category summary + surface-vs-cognitive hypothesis verdict + curated winners "
                         "list (also written to --curated-out)")
    ap.add_argument("--curated-out", default="research/runs/dial_library_curated.json",
                    help="(--report only) where to write the curated, winners-only library JSON")
    ap.add_argument("--n-prompts", type=int, default=6,
                    help="recent runlog user turns to sample (else NEUTRAL_PROMPTS)")
    ap.add_argument("--prompts-file", default=None, metavar="PROMPTS.json",
                    help="curated, model-independent prompt bank (JSON list of strings, {'prompts':[...]} "
                         "object, or one-per-line text) used verbatim and in FULL instead of the runlog "
                         "sample -- the reusable knob: sweep the SAME file against any --model to get "
                         "comparable calibrations. Ignores --n-prompts; overridden by --smoke.")
    ap.add_argument("--max-prompts", type=int, default=None, metavar="N",
                    help="cap a --prompts-file to an EVENLY-SPACED N (keeps a grouped bank balanced) -- "
                         "e.g. take a representative 50 from the 97-prompt bank without editing the file.")
    ap.add_argument("--exemplars", nargs="?", const=dial_exemplars.DEFAULT_PATH, default=None,
                    metavar="EXEMPLARS.json",
                    help="derive each covered dial's steering direction from a matched-pair CAA exemplar "
                         "bank (dial_exemplars.py's schema) instead of the default one-instruction-pair "
                         "recipe -- runs AFTER compute_dials (which still calibrates sc.base/resid_norm "
                         "unconditionally; see the module docstring's '--exemplars MODE' section). Bare "
                         "'--exemplars' (no value) defaults to scripts/calibration/dial_exemplars.json; a "
                         "dial the bank doesn't cover with >= MIN_RECOMMENDED_PAIRS pairs honestly falls "
                         "back to its instruction-derived vector (see dial_source in the saved report).")
    ap.add_argument("--out", default=None,
                    help="output path (default: dial_library_sweep.json for --library, else "
                         "dial_autocalibrate.json -- see _default_out_path)")
    ap.add_argument("--four-bit", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--effect-eps", type=float, default=None, metavar="EPS",
                    help="pin an ABSOLUTE effect floor in projection units. Default (recommended): scale-"
                         "relative -- 0.0291 x this model's measured resid_norm, which reproduces the tuned "
                         "2.0 at the Qwen2.5-7B reference and scales correctly to other models' activation "
                         "magnitudes. Pass this only to override the auto floor with a fixed number.")
    ap.add_argument("--layer", type=int, default=None, help="steering layer override (default num_layers//2)")
    ap.add_argument("--max-new", type=int, default=100, help="max new tokens per generation")
    ap.add_argument("--batch-size", type=_positive_int, default=8,
                    help="prompts/replies per GPU batch (default 8; lower after OOM, raise with VRAM headroom)")
    ap.add_argument("--resume", action="store_true",
                    help="continue completed dials from an existing compatible --out checkpoint; every "
                         "result-affecting setting, including --batch-size, must match")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="1-2 dials, 3 doses, 2 prompts -- prove the wiring cheaply, not a finding")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.effect_eps is not None:
        _EFFECT_EPS_OVERRIDE = args.effect_eps   # pin the absolute floor; _scale_effect_eps honors it
    if args.report:
        report(args.report, curated_out=args.curated_out)
    elif args.library:
        run_library(args.library, model_name=args.model, n_prompts=args.n_prompts,
                    out_path=args.out or _default_out_path(args.library),
                    four_bit_override=args.four_bit, smoke=args.smoke, seed=args.seed, layer=args.layer,
                    max_new=args.max_new, batch_size=args.batch_size, resume=args.resume)
    else:
        run(args.model, dials=args.dials, n_prompts=args.n_prompts,
            out_path=args.out or _default_out_path(args.library),
            four_bit_override=args.four_bit, smoke=args.smoke, seed=args.seed, layer=args.layer,
            max_new=args.max_new, prompts_file=args.prompts_file, max_prompts=args.max_prompts,
            exemplars=args.exemplars, batch_size=args.batch_size, resume=args.resume)
