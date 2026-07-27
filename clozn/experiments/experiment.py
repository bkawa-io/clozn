"""experiment.py -- the ONE experiment primitive.

clozn has six run-scoped HTTP endpoints that are all secretly the SAME operation -- "hold everything
constant, change one thing, compare, with a receipt" -- but each returns a different-shaped JSON:
POST /runs/<id>/replay, /counterfactual, /receipt, /receipts (prove-all, not folded in here), /branch,
/swap_receipt. This module is the dispatcher + normalizer: `run_experiment(run, change, method, sub)`
picks the right underlying op for a `change.type` and returns ONE envelope, so a UI "Experiment drawer"
can be a thin client over a single endpoint (POST /runs/<id>/experiment, wired in
clozn/server/routes/receipts.py) instead of six different shapes.

Registry (`REGISTRY`, also served read-only via `catalog()` / GET /experiments/types):
  ablate_card    {card_id}                -> receipt(mode)      -- real causal receipt (both-arms-greedy)
  ablate_memory  {}                       -> receipt(mode)      -- real causal receipt
  ablate_dial    {dial}                   -> receipt(mode)      -- real causal receipt
  set_dial       {dial, value}            -> counterfactual     -- what-if dial regen (decode-time, cheap)
  swap_concept   {to_concept, from_hint?} -> swap_receipt       -- read disposition, inject a concept dir,
                                                                    diff vs baseline AND a random-dir null
  edit_turn      {turn, alt_user?}        -> branch             -- fork the transcript at a turn
  reroll         {}                       -> replay({})         -- plain re-roll (sampled)
  toggle_greedy  {}                       -> replay({greedy})   -- same messages, forced greedy decode

HONESTY INVARIANTS (non-negotiable -- see module docstrings on receipts/deltas.py, replay/counterfactual.py,
receipts/swap_receipt.py for why these seams exist):
  * every null / random-direction control the underlying op returns is preserved in `result.null` -- NEVER
    dropped. When the underlying op has no null control at all (counterfactual, branch, replay), or the
    control could not be computed this time (e.g. forced-mode's null_floor needs a substrate with
    .steer_vector), `result.null` is `None` -- read as "missing", never as "no effect".
  * `result.has_effect` / `result.causal_verified` are copied VERBATIM from the underlying receipt's own
    field of that exact name when one exists. They are NEVER invented, inferred, or computed here from
    other data (e.g. `swap_receipt()` has no `has_effect` field -- its closest analogue, `targeted_shift`,
    stays exactly what it is, inside `result.receipt`, never relabeled). Ops that compute no verdict at all
    (branch, replay) leave both `None` -- a "no automatic verdict" experiment, not a fabricated "no effect".
  * `result.receipt` is always the FULL raw underlying result object, verbatim -- no information is ever
    dropped just because this module didn't have a normalized slot for it.
"""
from __future__ import annotations

from clozn.receipts.core import receipt as _receipt
from clozn.receipts.metrics import receipt_metrics as _receipt_metrics
from clozn.receipts.swap_receipt import swap_receipt as _swap_receipt
from clozn.replay.counterfactual import counterfactual as _counterfactual
from clozn.replay.replay import replay as _replay
from clozn.replay.timetravel import branch as _branch

_RECEIPT_MODES = ("regen", "forced", "both")


# ================================================================================================ registry
# type -> {label, needs (required change-dict keys), substrate ("chat" | "engine_jlens"), cost_hint, op
# (which underlying function this dispatches to, for the catalog only -- never consulted by the dispatcher,
# which switches on `type` explicitly below so a typo here can't silently misroute a real request).
REGISTRY: dict = {
    "ablate_card": {
        "label": "remove one memory card",
        "needs": ["card_id"],
        "substrate": "chat",
        "cost_hint": "expensive: a front-of-context memory ablation re-prefills the whole context (no KV reuse)",
        "op": "receipt (mode: regen|forced|both, default regen)",
    },
    "ablate_memory": {
        "label": "turn memory off entirely",
        "needs": [],
        "substrate": "chat",
        "cost_hint": "expensive: a front-of-context memory ablation re-prefills the whole context (no KV reuse)",
        "op": "receipt (mode: regen|forced|both, default regen)",
    },
    "ablate_dial": {
        "label": "zero one behavior dial",
        "needs": ["dial"],
        "substrate": "chat",
        "cost_hint": "cheap: a dial ablation is a decode-time hook, so the prompt KV stays reusable",
        "op": "receipt (mode: regen|forced|both, default regen)",
    },
    "set_dial": {
        "label": "set one behavior dial to a value",
        "needs": ["dial", "value"],
        "substrate": "chat",
        "cost_hint": "cheap: a dial override is a decode-time hook, so the prompt KV stays reusable",
        "op": "counterfactual",
    },
    "swap_concept": {
        "label": "swap the model's disposition toward a different concept",
        "needs": ["to_concept"],
        "substrate": "engine_jlens",
        "cost_hint": "moderate-to-expensive: 2-3 independent fresh generations, no KV reuse across arms",
        "op": "swap_receipt",
    },
    "edit_turn": {
        "label": "fork the conversation at a turn, optionally asking something different",
        "needs": ["turn"],
        "substrate": "chat",
        "cost_hint": "moderate: one generation from a re-prefilled truncated transcript (no fast-path KV reuse yet)",
        "op": "branch",
    },
    "reroll": {
        "label": "regenerate with the same settings",
        "needs": [],
        "substrate": "chat",
        "cost_hint": "moderate: one fresh generation, sampled by default (replay always regenerates from scratch)",
        "op": "replay({})",
    },
    "toggle_greedy": {
        "label": "regenerate under greedy (deterministic) decoding",
        "needs": [],
        "substrate": "chat",
        "cost_hint": "moderate: one fresh generation, forced greedy (replay always regenerates from scratch)",
        "op": 'replay({"greedy": true})',
    },
}

_SUBSTRATE_CHECKS = {
    "chat": lambda sub: bool(sub and getattr(sub, "chat", None)),
    "engine_jlens": lambda sub: bool(sub and getattr(sub, "engine", None) and getattr(sub, "jlens", None)),
}

_CONTROL_BY_TYPE = {
    "ablate_card": "with/without intervention; forced mode adds a matched filler control when available",
    "ablate_memory": "with/without intervention; forced mode adds a matched block filler control when available",
    "ablate_dial": "with/without intervention; forced mode adds an equal-norm random-vector control when available",
    "set_dial": "direct baseline comparison; no null control",
    "swap_concept": "equal-norm random-direction null control",
    "edit_turn": "direct branch comparison; no null control",
    "reroll": "direct regenerated comparison; no null control",
    "toggle_greedy": "direct regenerated comparison; no null control",
}


def substrate_ok(change_type: str, sub) -> bool:
    """Whether `sub` satisfies the substrate requirement REGISTRY records for `change_type`. False for an
    unknown type (nothing to check) as well as a genuinely missing capability -- callers that need to tell
    those apart should check `change_type in REGISTRY` first (the HTTP route does, for its 400 vs 503)."""
    entry = REGISTRY.get(change_type)
    if entry is None:
        return False
    check = _SUBSTRATE_CHECKS.get(entry["substrate"])
    return bool(check and check(sub))


def catalog() -> dict:
    """The preflight facts a UI needs before a person runs an experiment."""
    return {ctype: {"label": e["label"], "needs": list(e["needs"]), "cost_hint": e["cost_hint"],
                    "substrate": e["substrate"], "op": e["op"],
                    "control": _CONTROL_BY_TYPE[ctype]}
            for ctype, e in REGISTRY.items()}


# =================================================================================================== helpers

def _text_delta(a, b):
    """receipt_metrics(a, b), or None when there is truly nothing to compare (both sides absent) -- never
    fabricates a vacuous {0,0} delta for a fully-blocked op that generated neither reply."""
    if a is None and b is None:
        return None
    return _receipt_metrics(a, b)


def _grounded_est_seconds(run: dict, passes):
    """A rough per-experiment time estimate, grounded ONLY in this run's OWN recorded generation duration
    (never a fabricated constant): `passes` full generations at roughly the same cost as the run's own.
    Returns None -- and the envelope omits `cost.est_seconds` entirely -- whenever there's no recorded
    timing to ground the estimate in, or `passes` isn't a comparable-cost generation count (forced-mode
    scoring and swap_receipt's shorter completions are NOT estimated this way; see their handlers)."""
    if not passes:
        return None
    timing = (run or {}).get("timing") or {}
    dur_ms = timing.get("duration_ms")
    if not isinstance(dur_ms, (int, float)) or dur_ms <= 0:
        return None
    return round(passes * (dur_ms / 1000.0), 2)


def _forced_passes(forced: dict | None):
    """How many teacher-forced scoring calls a forced-mode receipt actually made: 2 (with/without) or 3
    (plus a null-floor control), read off what the receipt itself reports it computed. None when forced
    scoring didn't complete (causal_verified False) -- the exact call count at the point of failure isn't
    recoverable from the returned dict, so it is left unknown rather than guessed."""
    if not forced or not forced.get("causal_verified"):
        return None
    return 3 if forced.get("null_floor") is not None else 2


def _resolve_receipt_mode(method):
    if method is None:
        return "regen"
    if method in _RECEIPT_MODES:
        return method
    raise ValueError(f"method must be one of {_RECEIPT_MODES} for this change type (got {method!r})")


def _resolve_branch_sample(method):
    if method is None:
        return False
    m = str(method).strip().lower()
    if m == "greedy":
        return False
    if m in ("sample", "sampled"):
        return True
    raise ValueError(f"method must be 'greedy' or 'sample' for edit_turn (got {method!r})")


def _cap(s: str) -> str:
    return (s[:1].upper() + s[1:]) if s else s


def _trunc(s, n: int = 80) -> str:
    s = s if isinstance(s, str) else ("" if s is None else str(s))
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _first_note(raw: dict) -> str | None:
    for key in ("ablation_note", "override_note", "note"):
        v = (raw or {}).get(key)
        if v:
            return str(v)
    return None


def _plain_for(ctype: str, label: str, p: dict) -> str:
    """The one-line human summary. Mirrors the has_effect/causal_verified honesty rule exactly: it never
    asserts an effect the underlying op didn't verify, and it never claims "no effect" for a change whose
    verdict is simply missing (has_effect is None)."""
    baseline, changed = p.get("baseline_reply"), p.get("changed_reply")
    has_effect, causal_verified = p.get("has_effect"), p.get("causal_verified")
    if causal_verified is False:
        note = _first_note(p["raw"])
        return f"{_cap(label)}: not verified as applied" + (f" -- {note}" if note else "") + "."
    if has_effect is True:
        return f"{_cap(label)} changed the answer from \"{_trunc(baseline)}\" to \"{_trunc(changed)}\"."
    if has_effect is False:
        shown = changed if changed is not None else baseline
        return f"{_cap(label)} did not change the greedy answer (still \"{_trunc(shown)}\")."
    # has_effect is None: the underlying op computes no such verdict -- never invented here.
    if ctype == "swap_concept":
        raw = p["raw"]
        verdict = ("showed a targeted shift toward the swapped concept" if raw.get("targeted_shift")
                  else "did not show a targeted shift beyond the null control")
        return (f"{_cap(label)}: {verdict} -- baseline \"{_trunc(baseline)}\", "
               f"result \"{_trunc(changed)}\".")
    if baseline is not None and changed is not None:
        return (f"{_cap(label)}: no automatic effect verdict for this change type -- compare "
               f"\"{_trunc(baseline)}\" against \"{_trunc(changed)}\" directly.")
    return f"{_cap(label)}: no automatic effect verdict for this change type."


def _question_for(ctype: str, change: dict, target) -> str:
    if ctype == "ablate_card":
        return f"What if memory card {target} had not been applied?"
    if ctype == "ablate_memory":
        return "What if memory had been off entirely?"
    if ctype == "ablate_dial":
        return f"What if the '{target}' dial had not been applied?"
    if ctype == "set_dial":
        return f"What if the '{change.get('dial')}' dial had been set to {change.get('value')!r}?"
    if ctype == "swap_concept":
        hint = change.get("from_hint")
        lean = f" (instead of leaning toward '{hint}')" if hint else ""
        return f"What if the model had been steered toward '{change.get('to_concept')}'{lean}?"
    if ctype == "edit_turn":
        if change.get("alt_user"):
            return f"What if turn {target} had asked something different?"
        return f"What if turn {target} were re-generated from here?"
    if ctype == "reroll":
        return "What would a fresh generation produce under the same settings?"
    if ctype == "toggle_greedy":
        return "What would this run have produced under greedy (deterministic) decoding?"
    return "What if one thing about this run had been different?"


def _envelope(run: dict, change: dict, ctype: str, p: dict) -> dict:
    cost = {"passes": p.get("cost_passes"), "note": p["cost_note"]}
    if p.get("cost_est_seconds") is not None:
        cost["est_seconds"] = p["cost_est_seconds"]
    return {
        "run_id": run.get("id"),
        "question": _question_for(ctype, change, p.get("target")),
        "baseline": {"reply": p.get("baseline_reply")},
        "change": {"type": ctype, "target": p.get("target"), "label": p["label"]},
        "method": p["op"],
        "cost": cost,
        "result": {
            "changed_reply": p.get("changed_reply"),
            "delta": p.get("delta"),
            "has_effect": p.get("has_effect"),
            "causal_verified": p.get("causal_verified"),
            "null": p.get("null"),
            "receipt": p["raw"],
            "plain": _plain_for(ctype, p["label"], p),
        },
    }


# ========================================================================================= per-type handlers
# Each returns a "pieces" dict (see _envelope) or None if the underlying op ran but honestly couldn't
# produce a result (mirrors receipt()/counterfactual()/branch()/replay()'s own "never raise, return None on
# failure" contract -- run_experiment() maps that None straight through so the route can 500 exactly like
# every sibling route already does). A ValueError here means the CHANGE SPEC itself was malformed -- the
# route maps that to 400, mirroring the sibling routes' own influence/overrides/turn pre-validation.

def _ablate(run, method, sub, *, influence: dict, target, label: str) -> dict | None:
    mode = _resolve_receipt_mode(method)
    raw = _receipt(run, influence, sub, mode=mode)
    if raw is None:
        return None
    op = f"receipt:{mode}"
    if mode == "regen":
        return {
            "op": op, "raw": raw, "target": target, "label": label,
            "baseline_reply": raw.get("baseline_reply"), "changed_reply": raw.get("ablated_reply"),
            "delta": raw.get("delta"), "has_effect": raw.get("has_effect"),
            "causal_verified": raw.get("causal_verified"), "null": None,
            "cost_passes": 2, "cost_note": raw.get("cost_note") or "cost: unavailable",
            "cost_est_seconds": _grounded_est_seconds(run, 2),
        }
    if mode == "forced":
        forced_passes = _forced_passes(raw)
        note = ("cost: teacher-forced scoring re-derives per-token confidence for the answer this run "
                "already generated -- no new tokens are generated, so this is cheap relative to a "
                "regenerated (regen-mode) ablation.")
        return {
            "op": op, "raw": raw, "target": target, "label": label,
            # forced mode never generates new text -- it re-scores the SAME committed continuation, so
            # there is no separate "ablated reply" text; the run's own recorded response is the only text.
            "baseline_reply": run.get("response"), "changed_reply": None,
            "delta": None, "has_effect": raw.get("has_effect"), "causal_verified": raw.get("causal_verified"),
            "null": raw.get("null_floor"),
            "cost_passes": forced_passes, "cost_note": note, "cost_est_seconds": None,
        }
    # mode == "both": regen fields at the top level, forced nested under raw["forced"].
    forced = raw.get("forced") or {}
    forced_passes = _forced_passes(forced)
    note = (raw.get("cost_note") or "cost: unavailable") + (
        " Also runs a forced-mode (teacher-forced scoring) pass alongside the regen arms -- cheap, "
        "no new tokens generated.")
    return {
        "op": op, "raw": raw, "target": target, "label": label,
        "baseline_reply": raw.get("baseline_reply"), "changed_reply": raw.get("ablated_reply"),
        "delta": raw.get("delta"), "has_effect": raw.get("has_effect"),
        "causal_verified": raw.get("causal_verified"), "null": forced.get("null_floor"),
        "cost_passes": 2 + (forced_passes or 0), "cost_note": note,
        "cost_est_seconds": _grounded_est_seconds(run, 2),   # only the regen portion is duration-comparable
    }


def _handle_ablate_card(run, change, method, sub):
    card_id = change.get("card_id")
    if not card_id:
        raise ValueError("ablate_card needs a 'card_id'")
    return _ablate(run, method, sub, influence={"card_id": str(card_id)}, target=str(card_id),
                   label=f"removing memory card {card_id}")


def _handle_ablate_memory(run, change, method, sub):
    return _ablate(run, method, sub, influence={"memory_off": True}, target=None,
                   label="turning memory off")


def _handle_ablate_dial(run, change, method, sub):
    dial = change.get("dial")
    if not dial:
        raise ValueError("ablate_dial needs a 'dial'")
    return _ablate(run, method, sub, influence={"dial": str(dial)}, target=str(dial),
                   label=f"zeroing the '{dial}' dial")


def _handle_set_dial(run, change, method, sub):
    dial = change.get("dial")
    if not dial:
        raise ValueError("set_dial needs a 'dial'")
    if "value" not in change:
        raise ValueError("set_dial needs a 'value'")
    value = change["value"]
    raw = _counterfactual(run, {str(dial): value}, sub)
    if raw is None:
        return None
    return {
        "op": "counterfactual", "raw": raw, "target": str(dial), "label": f"setting '{dial}' to {value!r}",
        "baseline_reply": raw.get("baseline_reply"), "changed_reply": raw.get("counterfactual_reply"),
        "delta": raw.get("delta"), "has_effect": raw.get("has_effect"),
        "causal_verified": raw.get("causal_verified"),
        "null": None,   # counterfactual() has no null/random-direction control at all -- honestly absent
        "cost_passes": 2, "cost_note": raw.get("cost_note") or "cost: unavailable",
        "cost_est_seconds": _grounded_est_seconds(run, 2),
    }


def _handle_swap_concept(run, change, method, sub):
    to_concept = str(change.get("to_concept") or "").strip()
    if not to_concept:
        raise ValueError("swap_concept needs a 'to_concept'")
    from_hint = change.get("from_hint")
    raw = _swap_receipt(run, from_hint, to_concept, sub)   # never raises, never returns None (see module doc)
    lexicon = raw.get("lexicon_hits") or {}
    logprob = raw.get("logprob_shift") or {}
    null_obj = {
        "available": raw.get("null_control_available"),
        "reply": raw.get("null_reply"),
        "lexicon_hits": lexicon.get("null"),
        "logprob": logprob.get("null"),
        "swap_over_null_nat": logprob.get("swap_over_null_nat"),
        "note": raw.get("null_note"),
    }
    baseline_reply, changed_reply = raw.get("baseline_reply"), raw.get("swapped_reply")
    label = f"swapping toward '{to_concept}'" + (f" (from '{from_hint}')" if from_hint else "")
    n_gen = 2 + (1 if raw.get("null_control_available") else 0)
    note = ("cost: " + str(n_gen) + " independent fresh generations from the same rendered prompt "
           "(baseline, concept-swap" + (", null-control" if raw.get("null_control_available") else "") +
           ") -- no KV reuse across arms, comparable in cost to a memory-off ablation; the quantitative "
           "logprob-shift measure adds a few cheap teacher-forced scoring calls (no new tokens generated).")
    return {
        "op": "swap_receipt", "raw": raw, "target": to_concept, "label": label,
        "baseline_reply": baseline_reply, "changed_reply": changed_reply,
        "delta": _text_delta(baseline_reply, changed_reply),
        # swap_receipt() has no "has_effect" field -- never inferred from targeted_shift (see module doc).
        "has_effect": None, "causal_verified": raw.get("causal_verified"), "null": null_obj,
        "cost_passes": n_gen, "cost_note": note,
        "cost_est_seconds": None,   # shorter (max_new=64) completions aren't duration-comparable to the run
    }


def _handle_edit_turn(run, change, method, sub):
    if "turn" not in change:
        raise ValueError("edit_turn needs a 'turn'")
    try:
        turn = int(change["turn"])
    except (TypeError, ValueError):
        raise ValueError("edit_turn's 'turn' must be an integer")
    alt_user = change.get("alt_user")
    sample = _resolve_branch_sample(method)
    raw = _branch(run, turn, sub, alt_user=alt_user, sample=sample)
    if raw is None:
        return None
    label = f"editing turn {turn}" if alt_user else f"re-generating from turn {turn}"
    baseline_reply, changed_reply = run.get("response"), raw.get("response")
    kv = (raw.get("changes_applied") or {}).get("kv_snapshot")
    note = (f"cost: one generation from the truncated{' (edited)' if alt_user else ''} transcript -- v1 "
           f"always re-prefills the truncated context from scratch (no fast-path KV reuse yet; "
           f"kv_snapshot={kv} recorded on this branch).")
    return {
        "op": f"branch:{'sample' if sample else 'greedy'}", "raw": raw, "target": turn, "label": label,
        "baseline_reply": baseline_reply, "changed_reply": changed_reply,
        "delta": _text_delta(baseline_reply, changed_reply),
        # branch() computes no has_effect/causal_verified verdict at all -- never invented here.
        "has_effect": None, "causal_verified": None, "null": None,
        "cost_passes": 1, "cost_note": note, "cost_est_seconds": _grounded_est_seconds(run, 1),
    }


def _handle_reroll(run, change, method, sub):
    raw = _replay(run, {}, sub)
    if raw is None:
        return None
    baseline_reply, changed_reply = run.get("response"), raw.get("response")
    note = ("cost: one fresh generation from the same stored messages under the run's live settings "
           "(sampled by default) -- replay always regenerates from scratch (no KV cache reuse "
           "implemented yet).")
    return {
        "op": "replay:reroll", "raw": raw, "target": None, "label": "re-rolling the reply",
        "baseline_reply": baseline_reply, "changed_reply": changed_reply,
        "delta": _text_delta(baseline_reply, changed_reply),
        "has_effect": None, "causal_verified": None, "null": None,
        "cost_passes": 1, "cost_note": note, "cost_est_seconds": _grounded_est_seconds(run, 1),
    }


def _handle_toggle_greedy(run, change, method, sub):
    raw = _replay(run, {"greedy": True}, sub)
    if raw is None:
        return None
    baseline_reply, changed_reply = run.get("response"), raw.get("response")
    note = ("cost: one fresh generation from the same stored messages under forced greedy (deterministic) "
           "decoding -- isolates sampling randomness, not a content ablation; replay always regenerates "
           "from scratch (no KV cache reuse implemented yet).")
    return {
        "op": "replay:toggle_greedy", "raw": raw, "target": None, "label": "switching to greedy decoding",
        "baseline_reply": baseline_reply, "changed_reply": changed_reply,
        "delta": _text_delta(baseline_reply, changed_reply),
        "has_effect": None, "causal_verified": None, "null": None,
        "cost_passes": 1, "cost_note": note, "cost_est_seconds": _grounded_est_seconds(run, 1),
    }


_HANDLERS = {
    "ablate_card": _handle_ablate_card,
    "ablate_memory": _handle_ablate_memory,
    "ablate_dial": _handle_ablate_dial,
    "set_dial": _handle_set_dial,
    "swap_concept": _handle_swap_concept,
    "edit_turn": _handle_edit_turn,
    "reroll": _handle_reroll,
    "toggle_greedy": _handle_toggle_greedy,
}
assert set(_HANDLERS) == set(REGISTRY)   # every registry entry is dispatchable, and vice versa


# ======================================================================================================= API

def run_experiment(run: dict, change: dict, method: str | None, sub) -> dict | None:
    """The one experiment primitive. `run` is a full run record (as from runlog.get_run); `change` is
    `{"type": <one of REGISTRY>, ...type-specific fields}`; `method` selects among an op's own modes where
    one exists (regen|forced|both for the receipt-backed ablate_* types; greedy|sample for edit_turn) and is
    ignored where the underlying op has no such switch. `sub` is the live substrate, passed straight through
    to the underlying op exactly as the sibling routes already call it.

    Raises ValueError for a malformed request (no run, no/unknown change.type, a missing required field
    inside `change`, or an invalid `method`) -- the HTTP route maps that to 400, mirroring the sibling
    routes' own pre-validation (a bad `influence`/`overrides`/`turn` shape). Returns None when the change
    spec was well-formed but the underlying op honestly could not produce a result (mirrors
    receipt()/counterfactual()/branch()/replay()'s own "never raise, return None on failure" contract) --
    the route maps that to 500, exactly like every sibling route already does for its own op's None case.
    Never raises for any OTHER reason: a real exception from inside an underlying op propagates up
    unchanged (the route's own try/except turns that into 500 too)."""
    if not run or not isinstance(run, dict):
        raise ValueError("no run given")
    if not isinstance(change, dict) or not change:
        raise ValueError("need a change spec: {type, ...}")
    ctype = change.get("type")
    handler = _HANDLERS.get(ctype)
    if handler is None:
        raise ValueError(f"unknown change.type: {ctype!r} (know: {sorted(_HANDLERS)})")
    pieces = handler(run, change, method, sub)
    if pieces is None:
        return None
    return _envelope(run, change, ctype, pieces)
