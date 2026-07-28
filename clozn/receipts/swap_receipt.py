"""swap_receipt.py -- SWAP-RECEIPTS: a MECHANISM-level causal
receipt, ported from the validated research script
(../clozn-jlens-work/scripts/run_j5a_swap.py -- J5a verdict: PASS, 93% coherent re-route at
scale~=0.5, random-null flat).

The receipt: READ what the model is disposed toward at the pre-answer position (J-lens top-1,
L21 by default), WRITE a different, contrasting concept `to_concept` into the residual DURING
regeneration (dir(to_concept), from clozn/behavior/steering/concept_dir.py's Build-1 primitive,
at the same validated operating point), and DIFF the regenerated answer against the un-swapped
baseline. "The model was leaning toward X; we swapped in Y at L21; the answer changed from ...
to ... -- a targeted, content-specific shift toward Y" is the claim this receipt is built to
either support or (honestly) fail to support.

Discipline (mirrors run_j5a_swap.py's own, and clozn/receipts/quant_receipts.py's "two honest
signals, never conflated" style):
  * BASELINE: an unsteered generation from the SAME rendered prompt.
  * SWAP: dir(to_concept) injected at the resolved layer, scale x median_norm (Build 1's
    validated operating point) -- see concept_dir.ConceptSteer.steer_toward.
  * NULL: a random, EQUAL-MAGNITUDE write at the same layer/coef (clozn.receipts.forced's own
    `_random_vector_of_norm`, reused unchanged) -- must NOT coherently steer toward `to_concept`
    any more than noise would. `targeted_shift` is only meaningful when the swap beats BOTH the
    baseline AND this null; see `null_note`.
  * COHERENCE: a fluency heuristic on the swapped reply (ported from run_j5a_swap.py's
    `coherence()`) -- distinguishes "the injection rerouted a real answer" from "the injection
    garbled the output."

Two measures, like the validated script, NEVER conflated:
  A) a GENERATION receipt (`lexicon_hits`): does the regenerated text mention `to_concept` more
     than the baseline/null did? See `_LEXICON_CAVEAT` -- this is a literal, any-concept
     generalization of the script's curated word-family lexicon (paris/france/french/eiffel/...
     for "Paris"), and is WEAKER than that curated version for concepts outside the original
     validated set.
  B) a QUANTITATIVE shift (`logprob_shift`): exact logprob(to_concept token | prompt), baseline
     vs swap vs null, via the engine's existing `/score` route -- unaffected by (A)'s limitation.

Never raises (mirrors quant_receipts.py's "never raise into the caller" contract): every failure
mode -- a bad run, no engine, the concept_dir unembed BLOCKER, a resolution failure, a generation
failure -- degrades to `causal_verified: False` plus a `blocked`/`note` explaining exactly why,
never a silent guess and never an exception escaping to the caller.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter

from . import rederive
from .forced import _random_vector_of_norm

_NULL_NOTE = (
    "the null arm injects a RANDOM direction of the SAME magnitude (coef) at the SAME layer as "
    "the real swap -- if it shifts the answer toward `to_concept` about as often/strongly as the "
    "real dir(to_concept) does, the effect is just 'perturbing the residual stream changes the "
    "text,' not a targeted concept swap. `targeted_shift` is only claimed True when the real swap "
    "beats BOTH the un-swapped baseline AND this null on at least one measure below (mirrors "
    "../clozn-jlens-work/scripts/run_j5a_swap.py's J5a discipline, validated PASS: null flat, "
    "real swap 93% coherent re-route at scale~=0.5). When the null arm itself could not be "
    "generated/scored, `null_control_available` is False and `targeted_shift` reflects the "
    "baseline comparison ONLY -- read it as weaker evidence in that case."
)

_LEXICON_CAVEAT = (
    "lexicon_hits counts LITERAL, case-insensitive, word-boundary mentions of `to_concept` "
    "itself -- a generalization of run_j5a_swap.py's curated word-family lexicon (e.g. "
    "paris/france/french/eiffel/seine for \"Paris\") to an ARBITRARY concept word (Build 1's "
    "whole point is zero-calibration, any-concept steering). It will UNDERCOUNT semantically "
    "related mentions (e.g. \"canine\" for \"dog\") for concepts outside that originally-"
    "validated set -- read `targeted_shift` alongside `logprob_shift` (unaffected by this "
    "limitation), never off lexicon_hits alone."
)


def _stable_seed(*parts) -> int:
    """A deterministic (cross-process) seed from arbitrary parts -- unlike Python's built-in
    hash() on strings, which is randomized per-process by default, this makes the null control's
    random direction reproducible for the SAME (run, concept, layer)."""
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8", "ignore")).hexdigest()
    return int(digest[:8], 16)


def _words_of(text: str) -> list:
    return re.findall(r"[a-z']+", (text or "").lower())


def _coherence(text: str):
    """Fluency heuristic in [0,1]: high ASCII-letter ratio AND no degenerate repetition. Ported
    from ../clozn-jlens-work/scripts/run_j5a_swap.py's `coherence()` (validated live: distinguishes
    a real rerouted answer from a garbled one at the J5a operating point). Returns
    (score: float, is_coherent: bool)."""
    t = (text or "").strip()
    if not t:
        return 0.0, False
    letters = sum(c.isascii() and (c.isalpha() or c in " ,.'\"-\n?!:;") for c in t)
    ascii_ratio = letters / len(t)
    words = _words_of(t)
    if words:
        top = Counter(words).most_common(1)[0][1]
        rep_ratio = top / len(words)
    else:
        rep_ratio = 1.0
    score = ascii_ratio * (1.0 - min(rep_ratio, 1.0) ** 0.5)
    is_coherent = ascii_ratio > 0.75 and rep_ratio < 0.5 and len(words) >= 3
    return round(score, 3), bool(is_coherent)


def _concept_mentions(text: str, concept: str) -> int:
    """See _LEXICON_CAVEAT -- a literal, word-boundary, case-insensitive count of `concept` in
    `text`, generalized to ANY concept word (not just the 8 curated ones the validated script
    shipped a hand lexicon for)."""
    concept = (concept or "").strip()
    if not concept or not text:
        return 0
    return len(re.findall(r"\b" + re.escape(concept.lower()) + r"\b", text.lower()))


def _baseline_lean(text: str) -> str:
    """The first salient content word of a generation -- ported from run_j5a_swap.py's
    `baseline_lean()`: a more reliable proxy for 'what the model actually leaned toward' than a
    J-lens read at a copula-final position (which can return blank-fill '____' artifacts)."""
    stop = {"a", "an", "the", "is", "are", "and", "of", "to", "in", "it", "this", "that", "which",
           "as", "was", "its", "also", "given", "these", "those", "facts", "what", "for", "with",
           "on", "at", "by", "or", "but", "so", "then", "there", "here", "he", "she", "they", "we",
           "you", "i", "his", "her", "their", "our", "your"}
    for w in _words_of(text):
        if w not in stop and len(w) >= 3:
            return w
    return "?"


def swap_receipt(run: dict, from_hint, to_concept: str, sub, *,
                 layer: int | None = None, strength: float | None = None,
                 max_new: int = 64, concept_steer=None, null_seed=None) -> dict:
    """The product entry point. Never raises.

    `run`: a stored run record -- the SAME shape `clozn.receipts.rederive.with_arm_conditions`
        already reconstructs a prompt from (messages / assembled_messages).
    `from_hint`: an OPTIONAL caller-supplied guess of what the model was disposed toward (e.g. a
        person's own read, from a UI). Recorded verbatim in `disposed["hint"]` as a LABEL only,
        alongside the two INDEPENDENT machine reads (J-lens top-1 at the pre-answer position, and
        the baseline generation's own lean) -- never fed into the computation. Pass None/'' when
        there is no hint.
    `to_concept`: the concept word to swap IN. Must resolve to a single vocab token (see
        concept_dir.ConceptSteer.resolve_token_id) -- a multi-token word degrades this receipt
        with `blocked: "token_resolution"`.
    `sub`: a substrate exposing `.jlens(text, layer=, topk=)` (EngineSubstrate's existing method;
        `{"available": False, "reason"}` when J-lens isn't loaded is handled cleanly) and
        `.engine` (the raw EngineClient: `.apply_template`, `.complete`, `.intervene`, `.score`)
        -- the SAME raw-vector steer path (`/intervene`) Build 1's ConceptSteer is built to feed.
    `concept_steer`: an optional pre-built concept_dir.ConceptSteer (e.g. one already pointed at a
        configured unembed source) -- a fresh one is built from `sub.engine` otherwise.

    Returns a dict always carrying: mode, causal_verified, run_id, disposed, swapped_to,
    baseline_reply, swapped_reply, null_reply, targeted_shift, null_control_available,
    lexicon_hits, logprob_shift, coherent, coherence_score, null_note, lexicon_note, blocked, note.
    """
    out = {
        "mode": "swap_receipt", "causal_verified": False, "run_id": None,
        "disposed": None,
        "swapped_to": {"concept": to_concept, "layer": layer, "strength": strength,
                       "token_id": None, "coef": None},
        "baseline_reply": None, "swapped_reply": None, "null_reply": None,
        "targeted_shift": None, "null_control_available": False,
        "lexicon_hits": None, "logprob_shift": None,
        "coherent": None, "coherence_score": None,
        "null_note": _NULL_NOTE, "lexicon_note": _LEXICON_CAVEAT,
        "blocked": None, "note": None,
    }
    try:
        # Imported HERE, not at module scope, and this placement is load-bearing rather than stylistic.
        #
        # concept_dir is steering math and needs numpy, which clozn does NOT install: pyproject declares
        # `dependencies = []` on purpose (docs/RUNTIME_SPLIT.md). This module is reachable from
        # `clozn/cli/main.py`'s import block via experiments/__init__ -> experiment.py, so a module-scope
        # import made numpy a hard requirement of `clozn --help` on a fresh `pip install clozn` --
        # breaking scripts/release/clean_room_install_test.py, and every install that has not separately
        # installed numpy. The product-minimal CI lane did not catch it because that lane asserts only
        # that torch is absent, and torch genuinely is.
        #
        # `layer`/`strength` default to None rather than concept_dir's constants for the same reason:
        # default arguments are evaluated at function-DEFINITION time, so reading them off concept_dir in
        # the signature would re-create the eager import no matter where this line sits.
        #
        # Inside the try on purpose: this function documents "Never raises", so a missing numpy has to
        # degrade to a blocked receipt like every other failure here, not escape as an ImportError.
        try:
            import clozn.behavior.steering.concept_dir as concept_dir
        except ImportError as e:
            out["blocked"] = "steering_unavailable"
            out["note"] = (f"concept steering is unavailable in this install ({e}). swap receipts need "
                           f"numpy, which clozn does not install by default.")
            return out
        if layer is None:
            layer = concept_dir.DEFAULT_LAYER
        if strength is None:
            strength = concept_dir.DEFAULT_STRENGTH

        if not run or not isinstance(run, dict):
            out["note"] = "no run record given"
            return out
        out["run_id"] = run.get("id")

        engine = getattr(sub, "engine", None)
        if engine is None or not all(hasattr(engine, m) for m in ("apply_template", "complete", "intervene")):
            out["blocked"] = "no_engine"
            out["note"] = ("substrate has no .engine (a raw EngineClient exposing apply_template/"
                           "complete/intervene) to render a prompt and inject a raw steer vector on")
            return out

        conditions = rederive.with_arm_conditions(run)
        messages = conditions.get("messages") or []
        if not messages:
            out["note"] = "run has no messages to reconstruct a prompt from"
            return out
        try:
            prompt = engine.apply_template(messages, add_assistant=True)
        except Exception as e:
            out["blocked"] = "template_render"
            out["note"] = f"could not render the run's messages into a prompt: {e}"
            return out

        # READ: the model's disposition at the pre-answer position (J-lens top-1), plus the
        # caller's optional hint -- kept as two visibly separate labels, never merged.
        disposed = {"hint": (from_hint or None), "jlens_available": False, "jlens_layer": layer,
                   "jlens_top1": None, "jlens_top5": None, "jlens_reason": None, "baseline_lean": None}
        jlens_fn = getattr(sub, "jlens", None)
        if callable(jlens_fn):
            try:
                jl = jlens_fn(prompt, layer=layer, topk=5) or {}
            except Exception as e:
                jl = {"available": False, "reason": str(e)}
            if jl.get("available"):
                disposed["jlens_available"] = True
                readouts = jl.get("readouts") or []
                if readouts:
                    last = readouts[-1] or []
                    disposed["jlens_top1"] = (str(last[0].get("piece") or "")).strip() if last else None
                    disposed["jlens_top5"] = [str(r.get("piece") or "").strip() for r in last[:5]]
            else:
                disposed["jlens_reason"] = jl.get("reason") or jl.get("error")
        else:
            disposed["jlens_reason"] = "substrate has no .jlens method"
        out["disposed"] = disposed

        # WRITE: dir(to_concept) via Build 1's ConceptSteer, at the SAME validated operating point.
        steer = concept_steer if concept_steer is not None else concept_dir.ConceptSteer(engine, layer=layer)
        built = steer.steer_toward(to_concept, strength, layer=layer)
        resolved_layer = built.get("layer", layer)
        out["swapped_to"] = {"concept": to_concept, "layer": resolved_layer, "strength": strength,
                             "token_id": built.get("token_id"), "coef": built.get("coef")}
        if not built.get("ok"):
            out["blocked"] = built.get("blocked")
            out["note"] = built.get("note")
            return out
        vec = built["vector"]
        coef = built["coef"]
        token_id = built.get("token_id")

        # BASELINE: unsteered generation from the SAME prompt.
        try:
            baseline_reply = concept_dir._text_of(engine.complete(prompt, max_tokens=max_new))
        except Exception as e:
            out["blocked"] = "generation_failed"
            out["note"] = f"baseline generation failed: {e}"
            return out
        out["baseline_reply"] = baseline_reply
        disposed["baseline_lean"] = _baseline_lean(baseline_reply)

        # SWAP: dir(to_concept) injected at resolved_layer during generation.
        try:
            swapped_reply = concept_dir._text_of(
                engine.intervene(prompt, vector=vec, coef=coef, layer=resolved_layer, max_tokens=max_new))
        except Exception as e:
            out["blocked"] = "generation_failed"
            out["note"] = f"swap generation failed: {e}"
            return out
        out["swapped_reply"] = swapped_reply

        # NULL: a random equal-magnitude write at the same layer/coef -- see _NULL_NOTE. Its
        # failure degrades ONLY the null control, not the whole receipt (baseline+swap already
        # succeeded); `null_control_available` records which case this is.
        seed = null_seed if null_seed is not None else _stable_seed(out["run_id"], to_concept, resolved_layer)
        null_vec = _random_vector_of_norm(len(vec), 1.0, seed)
        null_reply = None
        try:
            null_reply = concept_dir._text_of(
                engine.intervene(prompt, vector=null_vec, coef=coef, layer=resolved_layer, max_tokens=max_new))
            out["null_control_available"] = True
        except Exception:
            pass
        out["null_reply"] = null_reply

        # measure A: literal, any-concept mention count (the generation receipt).
        base_hits = _concept_mentions(baseline_reply, to_concept)
        swap_hits = _concept_mentions(swapped_reply, to_concept)
        null_hits = _concept_mentions(null_reply, to_concept) if null_reply is not None else None
        out["lexicon_hits"] = {"baseline": base_hits, "swap": swap_hits, "null": null_hits}

        # measure B: quantitative logprob(to_concept token) shift, baseline vs swap vs null.
        logprob_shift = None
        score_fn = getattr(engine, "score", None)
        if callable(score_fn) and isinstance(token_id, int):
            try:
                base_lp = score_fn(prompt=prompt, continuation_ids=[token_id], topk=0)["tokens"][0]["logprob"]
                swap_lp = score_fn(prompt=prompt, continuation_ids=[token_id], topk=0,
                                   steer_vec=vec, steer={"coef": coef, "layer": resolved_layer})["tokens"][0]["logprob"]
                null_lp = None
                try:
                    null_lp = score_fn(prompt=prompt, continuation_ids=[token_id], topk=0,
                                       steer_vec=null_vec, steer={"coef": coef, "layer": resolved_layer}
                                       )["tokens"][0]["logprob"]
                except Exception:
                    null_lp = None
                logprob_shift = {
                    "baseline": float(base_lp), "swap": float(swap_lp),
                    "null": (float(null_lp) if null_lp is not None else None),
                    "swap_over_baseline_nat": round(float(swap_lp) - float(base_lp), 4),
                    "swap_over_null_nat": (round(float(swap_lp) - float(null_lp), 4)
                                          if null_lp is not None else None),
                }
            except Exception:
                logprob_shift = None
        out["logprob_shift"] = logprob_shift

        coherence_score, is_coherent = _coherence(swapped_reply)
        out["coherence_score"] = coherence_score
        out["coherent"] = is_coherent

        beats_baseline_gen = swap_hits > base_hits
        beats_null_gen = null_hits is None or swap_hits > null_hits
        gen_shift = beats_baseline_gen and beats_null_gen
        score_shift = False
        if logprob_shift is not None:
            beats_baseline_score = logprob_shift["swap_over_baseline_nat"] > 1.0
            beats_null_score = (logprob_shift["swap_over_null_nat"] is None
                                or logprob_shift["swap_over_null_nat"] > 1.0)
            score_shift = beats_baseline_score and beats_null_score
        out["targeted_shift"] = bool(gen_shift or score_shift)

        if to_concept and from_hint and str(from_hint).strip().lower() == str(to_concept).strip().lower():
            out["note"] = ("from_hint and to_concept are the same word -- this is a no-op/null swap "
                           "by construction, not a counterfactual")

        out["causal_verified"] = True
        return out
    except Exception as e:
        out["blocked"] = out.get("blocked") or "error"
        out["note"] = out.get("note") or f"swap_receipt failed unexpectedly: {e}"
        return out


# ============================================================================== CLI: --demo (LIVE smoke, DEFERRED)
#
# Real, runnable code -- but deferred: needs a running engine, so it is not exercised by this module's
# own tests. tests/test_swap_receipt.py already
# covers the whole receipt model-free/GPU-free against fixtures + a FakeEngineClient; this is only
# for a human to run LATER against a real clozn-server, once BOTH exist:
#   1. a running clozn-server, started with --jlens <dir> (a real J-lens sidecar loaded);
#   2. an unembed export for concept_dir.ConceptDirSource (see concept_dir.py's BLOCKER_NOTE) --
#      e.g. --unembed-dir ../clozn-jlens-work/artifacts for local dev.
# Then: `python -m clozn.receipts.swap_receipt --port 8095 --unembed-dir ../clozn-jlens-work/artifacts
#   --prompt "The capital of France is" --to-concept ocean --from-hint Paris`

def _demo(args) -> int:
    here = os.path.dirname(os.path.abspath(__file__))                       # clozn/receipts
    engine_client_dir = os.path.abspath(os.path.join(here, "..", "..", "engine", "client"))
    import sys as _sys
    _sys.path.insert(0, engine_client_dir)
    from clozn_engine import EngineClient  # local import: only the --demo path needs the engine SDK

    class _DemoSub:
        """The minimal duck-typed substrate swap_receipt needs: .engine + .jlens -- mirrors
        clozn.server.app.EngineSubstrate's own shape (jlens() never raises; see its docstring)."""

        def __init__(self, engine_client, layer):
            self.engine = engine_client
            self._layer = layer

        def jlens(self, text, layer=None, topk=5):
            try:
                r = self.engine.jlens(text, layer=layer or self._layer, topk=topk)
                return {"available": True, **r}
            except Exception as e:
                return {"available": False, "reason": str(e)}

    ec = EngineClient(host=args.host, port=args.port)
    print(f"server: {ec.health().get('model')}")
    sub = _DemoSub(ec, args.layer)
    run = {"id": "demo", "messages": [{"role": "user", "content": args.prompt}]}
    source = concept_dir.ConceptDirSource(jlens_dir=args.jlens_dir, unembed_dir=args.unembed_dir)
    steer = concept_dir.ConceptSteer(ec, source=source, layer=args.layer)
    receipt = swap_receipt(run, args.from_hint, args.to_concept, sub, concept_steer=steer,
                           strength=args.strength, max_new=args.max_tokens)
    import json as _json
    print(_json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt.get("causal_verified") else 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="swap_receipt.py -- swap-receipts LIVE smoke (DEFERRED by default)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--jlens-dir", default=None, help="default: ~/.clozn/jlens or CLOZN_JLENS_DIR")
    ap.add_argument("--unembed-dir", default=None, help="see concept_dir.BLOCKER_NOTE")
    ap.add_argument("--layer", type=int, default=concept_dir.DEFAULT_LAYER)
    ap.add_argument("--strength", type=float, default=concept_dir.DEFAULT_STRENGTH)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--from-hint", default=None)
    ap.add_argument("--to-concept", default="ocean")
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args(argv)
    return _demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
