"""provenance.py -- did this answer come from the CONTEXT or from the model's own weights?

The question every RAG product needs answered and none can currently answer honestly: the model
cited a document, but did it actually *use* it? Today that is settled with attention heatmaps
(correlational -- attention weight is not influence) or by asking the model (which confabulates).

This measures it. Attention KNOCKOUT severs "the answering position may read position p" at every
layer and re-scores the answer token teacher-forced. If the answer survives having its context
access cut, it did not come from the context.

    context_dependence = 1 - P_cut(answer) / P_baseline(answer)

~1.0  the answer is carried by the context (cut it and the answer is gone)
~0.0  the answer is parametric (the model already knew it; the context was decoration)

Measured on Qwen2.5-7B (notes/CIRCUIT_TRACER_DESIGN.md section 5h) -- the dissociation is sharp:
an in-context key/value lookup and a name retrieval score ~1.0, while "the modern capital of Japan
is Tokyo" scores low: knock out every context position the search can find and the model STILL
says Tokyo, because it knows it parametrically.

WHY THIS AND NOT THE RESIDUAL TRACER: residual-site path patching cannot measure cross-position
influence at all on this stack (section 5f -- a flat 0.0% routed at every depth, because patching
one site lets the source re-supply downstream and the last layer is unpatchable). Cutting the
attention edge sidesteps both problems.

TWO THINGS THAT WILL GIVE YOU A WRONG ANSWER IF YOU SKIP THEM, both measured the hard way:
  1. RENORMALIZE. Position 0 is an attention sink holding a large share of the mass; zeroing it
     without rescaling shrinks the whole attention output -- a generic amplitude perturbation that
     looks like a routing result (it was the top-ranked position at +0.717 before renormalising,
     and vanishes from every span after). `renormalize=True` is the default here for that reason.
  2. GREEDY, NOT TOP-K. Single-position knockout loses to redundancy: cutting one token of a
     multi-token entity leaves its siblings to re-supply it, so every single position scores ~0.
     Greedy accumulation finds sets that are jointly decisive and individually invisible (measured:
     best single +0.11 vs the greedy span's +7.60 on the same prompt).

Requires a clozn-server started with --no-flash-attn (flash attention fuses the softmax so the
weights never materialize); `available()` checks, and the engine refuses rather than silently
no-opping. House style: the product-facing call never raises -- it returns a labeled
{"ok": False, "blocked": ...} dict.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Optional

DEFAULT_ENGINE = "http://127.0.0.1:8080"

# Product-facing scope note (presentation only -- read by clozn/cli/commands/provenance.py and
# clozn/server/routes/provenance.py; changes nothing about trace_provenance's own logic or its 41/41
# battery). States what this capability is, what it needs, and the maturity caveat that stays true no
# matter which verdict comes back, so a CONTEXT_CARRIED verdict is never read as louder than it is.
SCOPE_NOTE = ("attention-knockout provenance -- requires a clozn-server started with --no-flash-attn "
             "(flash attention fuses the softmax, so attention weights never materialize) -- validated "
             "two-family (Qwen2.5-7B + Llama-3.1-8B, 41/41 UNDER THE CURRENT GRADING; the stored "
             "provenance_battery_*.json summary blocks predate a grading change and read 23/23 + 21/26 "
             "-- re-grade the per-case data, do not quote the stale summaries); read a verdict as "
             "evidence, not proof")


@dataclass
class ProvenanceBudget:
    max_span: int = 6         # greedy steps (each = a few arms)
    candidates: int = 12      # strongest singles reconsidered per greedy step
    min_gain: float = 0.05    # stop when the best addition adds less than this
    n_controls: int = 3       # matched random-set draws per step
    renormalize: bool = True  # see the module docstring -- do not turn this off casually


# ============================================================== pure helpers (fixture-tested)

def context_dependence(base_logprob: float, cut_logprob: float) -> float:
    """1 - P_cut/P_base, clamped to [0, 1]. Both inputs are natural-log probabilities.
    1.0 = the context carried the answer; 0.0 = the answer survived losing its context."""
    import math
    if cut_logprob >= base_logprob:
        return 0.0
    ratio = math.exp(cut_logprob - base_logprob)     # P_cut / P_base, stable in log space
    return max(0.0, min(1.0, 1.0 - ratio))


def verdict(dependence: float, best_ratio: float) -> str:
    """CONTEXT_CARRIED / MIXED / PARAMETRIC / INCONCLUSIVE. `best_ratio` is the strongest
    span-vs-matched-random-control ratio: without separation from control, no verdict is earned
    no matter how large the raw effect."""
    if best_ratio < 3.0:
        return "INCONCLUSIVE"
    if dependence >= 0.8:
        return "CONTEXT_CARRIED"
    if dependence >= 0.3:
        return "MIXED"
    return "PARAMETRIC"


def _post(engine: str, path: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(engine.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def available(engine_url: str = DEFAULT_ENGINE) -> bool:
    """Is the engine started with --no-flash-attn (so attention weights materialize)?"""
    try:
        h = json.loads(urllib.request.urlopen(engine_url.rstrip("/") + "/health", timeout=10).read())
        return bool(h.get("capabilities", {}).get("attn_knockout"))
    except Exception:
        return False


# ==================================================================================== the API

def trace_provenance(prompt: str, continuation=None, *, engine_url: str = DEFAULT_ENGINE,
                     budget: Optional[ProvenanceBudget] = None, seed: int = 0,
                     focus: Optional[tuple] = None) -> dict:
    """Which context positions carry this answer, and is it context-carried or parametric?

    `continuation` defaults to whatever the model generates greedily (so we never grade it on an
    answer it did not give). Returns a receipt dict; {"ok": False, "blocked": ...} on failure.

    `focus=(start, end)` (token indices, end exclusive) scopes the question to ONE REGION of the
    prompt -- the RAG question in its honest form: "did the answer use THIS document?" rather than
    "did it use any context at all?". Measured motivation (R1 battery, 2nd family): total-context
    dependence CONFLATES "used the override/document" with "used the question" -- a model that
    IGNORED a counterfactual override and answered from weights still shows dependence ~1.0,
    because cutting the question tokens ('the capital of France is') kills any answer. With focus,
    the span search and its matched random controls are both restricted: candidate cut positions
    come only from [start, end), and control sets are drawn from OUTSIDE the focus region (same
    set size), so the ratio asks "is cutting inside the document worse than cutting the same
    number of positions elsewhere?". The verdict then speaks about the focus region specifically:
    PARAMETRIC now honestly means "the answer did not need this region", even when the question
    itself is load-bearing.

    FOCUS MODE IS EXPERIMENTAL. First measured caveat (Llama-3.1-8B live, 2026-07-20): with
    controls drawn from ALL outside-focus positions, the draws include the question's own
    load-bearing tokens, whose cuts are devastating -- the bar was nearly unclearable and small
    focus regions read INCONCLUSIVE even at meaningful dependence (a resisted override value read
    dep 0.49 -- the model plausibly using the override CONTRASTIVELY, reading it in order to
    contradict it -- yet failed the ratio). The null now shipped in response: the outside singles
    are scanned too and the TOP QUARTILE by delta is excluded from the control pool
    (`focus_trim` in the receipt records exactly which tokens were trimmed), so the null asks
    "more load-bearing than typical NON-CRITICAL context?" -- and a 12-draw rank test reports a
    permutation `focus_null.p_value` alongside the ratio. The p is evidence, not yet folded into
    the verdict; read a focused INCONCLUSIVE with a small p as "real dependence, ratio criterion
    unconvinced". Costs ~n_outside extra arms."""
    import numpy as np
    budget = budget or ProvenanceBudget()
    rng = np.random.default_rng(seed)
    try:
        health = json.loads(urllib.request.urlopen(engine_url.rstrip("/") + "/health",
                                                   timeout=10).read())
        if not health.get("capabilities", {}).get("attn_knockout"):
            return {"ok": False, "blocked": "engine lacks attn_knockout; start clozn-server with "
                                            "--no-flash-attn (flash attention fuses the softmax, so "
                                            "the attention weights never materialize)"}
        n_layer = int(health["n_layer"])

        if continuation is None:
            continuation = _post(engine_url, "/v1/completions",
                                 {"prompt": prompt, "max_tokens": 2,
                                  "temperature": 0})["choices"][0]["text"]
        base = _post(engine_url, "/score", {"prompt": prompt, "continuation": continuation,
                                            "topk": 3})
        n_p = int(base["n_prompt"])
        base_lp = float(base["tokens"][0]["logprob"])
        cont_ids = [int(base["tokens"][0]["id"])]
        answer = base["tokens"][0]["piece"]
        final = n_p - 1
        toks = _post(engine_url, "/harvest", {"text": prompt, "layer": 1})["tokens"]

        def cut(keys, topk=0):
            specs = [{"layer": L, "queries": [final], "keys": sorted(set(int(k) for k in keys)),
                      "renormalize": budget.renormalize} for L in range(n_layer)]
            r = _post(engine_url, "/score", {"prompt": prompt, "continuation_ids": cont_ids,
                                             "topk": topk, "attn_knockout": specs})
            lp = float(r["tokens"][0]["logprob"])
            return base_lp - lp, lp, r

        # Candidate pool + control pool. Default: every prompt position may be cut, controls drawn
        # from the not-yet-cut remainder. With `focus`, candidates come ONLY from the focus region
        # and controls ONLY from outside it (see the docstring for why that is the honest RAG
        # question). Focus bounds are clamped to the prompt; an empty/degenerate focus is refused.
        if focus is not None:
            f0, f1 = max(0, int(focus[0])), min(final, int(focus[1]))
            if f1 <= f0:
                return {"ok": False, "blocked": f"focus region ({focus}) is empty after clamping "
                                                f"to the prompt's {final} cuttable positions"}
            cand_pool = list(range(f0, f1))
            outside = [p for p in range(final) if p < f0 or p >= f1]
            if len(outside) < 2:
                return {"ok": False, "blocked": "focus covers (nearly) the whole prompt -- no "
                                                "outside-focus positions left to draw controls "
                                                "from; use the unfocused mode instead"}
            # TRIMMED control pool (the focus-mode null redesign, from the measured caveat in the
            # docstring): scan the OUTSIDE singles too and exclude the top quartile by delta --
            # those are typically the question's own load-bearing tokens, and a null that includes
            # them asks the wrong question ("is this region more critical than THE QUESTION?")
            # instead of the honest one ("is this region more load-bearing than typical
            # non-critical context?"). Costs len(outside) extra arms.
            outside_singles = sorted(((cut([p])[0], p) for p in outside), key=lambda x: -x[0])
            n_trim = max(1, len(outside_singles) // 4)
            trimmed_out = [p for _, p in outside_singles[n_trim:]]
            ctl_pool_base = trimmed_out if len(trimmed_out) >= 2 else outside
            focus_trim = {"n_outside": len(outside), "n_trimmed": n_trim,
                          "trimmed_positions": [int(p) for _, p in outside_singles[:n_trim]],
                          "trimmed_tokens": [toks[p] for _, p in outside_singles[:n_trim]]}
        else:
            cand_pool = list(range(final))
            ctl_pool_base = None   # controls drawn from all non-span positions (original behavior)
            focus_trim = None

        singles = sorted(((cut([p])[0], p) for p in cand_pool), key=lambda x: -x[0])
        span, trace, cur = [], [], 0.0
        remaining = [p for _, p in singles]

        # ADJACENT-BIGRAM SEEDING (found by the R1 battery, case kv_late): greedy accumulation
        # cannot START when no SINGLE position clears min_gain -- and a multi-token entity whose
        # pieces are individually redundant (a passcode split into two digit tokens, a name split
        # across BPE pieces) is exactly that case: cut either piece alone and the sibling re-
        # supplies it, so every single scores ~0 and the old code returned an empty span --
        # "PARAMETRIC" -- for an answer that exists ONLY in the prompt. Entities are contiguous,
        # so before giving up we scan ADJACENT PAIRS (n-2 extra arms) and, if the best pair
        # clears min_gain, seed the span with both positions at once; greedy then continues
        # normally from that seed.
        seeded_bigram = None
        if not singles or singles[0][0] <= budget.min_gain:
            bigrams = sorted((((cut([p, p + 1])[0]), p) for p in cand_pool[:-1]),
                             key=lambda x: -x[0])
            if bigrams and bigrams[0][0] > budget.min_gain:
                d2, p2 = bigrams[0]
                span = [p2, p2 + 1]
                cur = d2
                remaining = [p for p in remaining if p not in span]
                pool = ([p for p in ctl_pool_base if p not in span] if ctl_pool_base is not None
                        else [p for p in range(final) if p not in span])
                ctl = (max(cut(rng.choice(pool, size=2, replace=False).tolist())[0]
                           for _ in range(budget.n_controls))
                       if len(pool) >= 2 else float("nan"))
                seeded_bigram = {"step": 1, "added": [int(p2), int(p2 + 1)],
                                 "token": [toks[p2], toks[p2 + 1]], "joint": cur, "control": ctl,
                                 "ratio": (abs(cur) / abs(ctl)) if ctl and ctl == ctl
                                          and abs(ctl) > 1e-9 else None,
                                 "seeded": "adjacent_bigram"}
                trace.append(seeded_bigram)

        step_base = 1 if seeded_bigram else 0
        for step in range(budget.max_span):
            best = None
            for p in remaining[:budget.candidates]:
                d, _, _ = cut(span + [p])
                if best is None or d > best[0]:
                    best = (d, p)
            if best is None or best[0] <= cur + budget.min_gain:
                break
            cur, add = best
            span.append(add)
            remaining.remove(add)
            pool = ([p for p in ctl_pool_base if p not in span] if ctl_pool_base is not None
                    else [p for p in range(final) if p not in span])
            ctl = (max(cut(rng.choice(pool, size=len(span), replace=False).tolist())[0]
                       for _ in range(budget.n_controls))
                   if len(pool) >= len(span) else float("nan"))
            trace.append({"step": step_base + step + 1, "added": int(add), "token": toks[add],
                          "joint": cur, "control": ctl,
                          "ratio": (abs(cur) / abs(ctl)) if ctl and ctl == ctl and abs(ctl) > 1e-9
                                   else None})

        if not span:
            return {"ok": True, "answer": answer, "span": [], "dependence": 0.0,
                    "verdict": "PARAMETRIC",
                    "note": ("no focus-region position changed the answer -- the answer did not "
                             "need this region" if focus is not None
                             else "no context position changed the answer"),
                    "focus": list(focus) if focus is not None else None,
                    "baseline_logprob": base_lp}
        d_final, lp_cut, r_final = cut(span, topk=3)
        dep = context_dependence(base_lp, lp_cut)
        ratios = [t["ratio"] for t in trace if t["ratio"] is not None]
        best_ratio = max(ratios) if ratios else 0.0

        # Focus-mode rank test: where does the focus span's joint delta rank among same-size
        # random sets from the TRIMMED outside pool? A permutation p alongside the ratio -- 12
        # draws instead of max-of-3, so one lucky control can't sink a real effect. Reported as
        # evidence, not yet folded into the verdict (the verdict keeps the ratio criterion until
        # this null has its own battery).
        focus_null = None
        if focus is not None and span:
            pool = [p for p in ctl_pool_base if p not in span]
            k = len(span)
            if len(pool) >= k:
                n_draws = 12
                draws = [cut(rng.choice(pool, size=k, replace=False).tolist())[0]
                         for _ in range(n_draws)]
                worse = sum(1 for d in draws if d >= d_final)
                focus_null = {"p_value": round((1 + worse) / (1 + n_draws), 4),
                              "n_draws": n_draws,
                              "control_deltas": [round(float(d), 4) for d in draws],
                              "pool": "outside-focus, top-quartile singles trimmed"}
        return {
            "ok": True,
            "answer": answer, "baseline_logprob": base_lp, "cut_logprob": lp_cut,
            "span": sorted(int(p) for p in span),
            "span_tokens": [toks[p] for p in sorted(span)],
            "delta": d_final, "dependence": dep,
            "best_control_ratio": best_ratio,
            "verdict": verdict(dep, best_ratio),
            "top3_after_cut": [{"piece": t["piece"], "logprob": t["logprob"]}
                               for t in r_final["tokens"][0]["topk"]],
            "best_single": {"pos": int(singles[0][1]), "token": toks[singles[0][1]],
                            "delta": singles[0][0]},
            "trace": trace,
            "focus": list(focus) if focus is not None else None,
            "focus_trim": focus_trim,
            "focus_null": focus_null,
            "config": {"renormalize": budget.renormalize, "n_layer": n_layer, "seed": seed,
                       "units": "delta = logprob(baseline) - logprob(context access cut)",
                       "control_pool": ("outside-focus positions" if focus is not None
                                        else "all non-span positions")},
        }
    except Exception as e:
        return {"ok": False, "blocked": f"{type(e).__name__}: {e}"}
