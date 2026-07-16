"""Teacher-forced receipt scoring and null-floor controls.

SECTION ABLATION (prompt-section influence, fast path): a "section" influence (`{"section": name,
"source": "api"|"auto"}`, from deltas._section_influences -- memory_card-sourced sections are dedup'd out
there and never reach this module) is the one kind here whose ablation is a prompt-CONTENT edit rather
than a block/steer knob swap. `_forced_ablation`'s section branch reuses replay.py's OWN
`_apply_section_exclusions`/`_strip_message_parts` splice (never a second copy of that span-removal logic)
against `run["messages"]` -- the RAW, pre-block-injection list, which is exactly what an "api"/"auto"
section's `parts[].message_index` is built from (see clozn.runs.sections's docstring + clozn.server.
routes.openai's try_post). A part anchored to `final_prompt` (`message_index: null` -- a raw-prompt/CLI
run) can't be spliced through a messages list either way: `EngineSubstrate.score_tokens`'s only prompt
surface is `messages` -> `ctx._inject_block` -> `ctx._engine_tmpl` (the engine's chat-template render, see
clozn.server.app._engine_tmpl) -- there is no raw-prompt scoring entry point on this substrate at all, so
this is a real ceiling, not a shortcut we chose not to take. `_apply_section_exclusions` already reports
that honestly (a raw-anchored part it can't touch), and that note rides straight through as this
receipt's `ablation_note` -- no guessing, no silent no-op. No null-floor control is computed for a section
ablation (there's no register-matched "equal-sized irrelevant text" analogue for an arbitrary prompt
span the way a card has filler text or a dial has a random direction); the receipt still reports its raw
delta, just without a floor-clearing ratio.
"""
from __future__ import annotations

import math
import random

from . import rederive


_FORCED_MEAN_THRESHOLD = 0.05
_FORCED_SUM_THRESHOLD = 2.0
_NULL_FLOOR_RATIO_MIN = 5.0

_FORCED_CAVEAT = (
    "a nonzero delta means the influence changed the model's confidence in the answer it gave -- it "
    "does NOT mean the answer would have been different without it. Regen mode answers 'would the "
    "greedy answer have changed?' (counterfactual text); forced mode answers 'how much did THIS answer "
    "rely on it?' (dependence). Both are interventions; they measure different outcomes -- read them "
    "side by side, never interchangeably ('the sub-threshold receipt')."
)

_FORCED_NOTE = (
    "dial vectors (and, for a card ablation, the recompiled memory block) are computed from TODAY's "
    "steering library / card store at the run's recorded strengths and card texts -- the same "
    "limitation the regen receipt already carries. The with/without prompts differ in length by "
    "whatever was ablated; deltas align per CONTINUATION token position, which is what matters -- not "
    "per prompt token."
)

_FILLER_TEXT = (
    "The user prefers to schedule meetings in the early morning rather than the afternoon. The user "
    "always tips exactly twenty percent at restaurants without needing to calculate it by hand. The "
    "user set their phone's default browser to a different app than the one it shipped with. The user "
    "keeps their email inbox at zero and archives messages the same day they arrive. "
)


def _matched_length_filler(n_chars: int) -> str:
    n = max(1, int(n_chars))
    reps = n // len(_FILLER_TEXT) + 1
    return (_FILLER_TEXT * reps)[:n]


def _vector_norm(vec) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in (vec or [])))


def _random_vector_of_norm(dim: int, norm: float, seed) -> list:
    rng = random.Random(seed)
    dim = max(1, int(dim))
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    raw_norm = _vector_norm(raw)
    if raw_norm <= 0.0:
        raw = [1.0] + [0.0] * (dim - 1)
        raw_norm = 1.0
    scale = float(norm) / raw_norm
    return [x * scale for x in raw]


def _forced_deltas(with_tokens, without_tokens):
    if not with_tokens or not without_tokens or len(with_tokens) != len(without_tokens):
        return None
    out = []
    for w, wo in zip(with_tokens, without_tokens):
        if not isinstance(w, dict) or not isinstance(wo, dict):
            return None
        lw, lwo = w.get("logprob"), wo.get("logprob")
        if not isinstance(lw, (int, float)) or not isinstance(lwo, (int, float)):
            return None
        out.append(float(lw) - float(lwo))
    return out


def _delta_summary(deltas: list) -> dict:
    n = len(deltas) or 1
    return {
        "sum_nats": round(sum(deltas), 6),
        "mean_nats_per_token": round(sum(abs(d) for d in deltas) / n, 6),
    }


def _top_dependent(pieces: list, deltas: list, k: int = 5) -> list:
    order = sorted(range(len(deltas)), key=lambda i: -abs(deltas[i]))[:k]
    return [{"index": i, "piece": pieces[i] if i < len(pieces) else "", "delta": round(deltas[i], 6)}
            for i in order]


def _forced_ablation(run: dict, influence: dict, sub, conditions: dict):
    influence = influence or {}
    with_block = conditions.get("raw_block")
    with_strengths = dict(conditions.get("steer_strengths") or {})

    cid = influence.get("card_id")
    if cid:
        mem = run.get("memory") or {}
        ids = mem.get("applied_ids") or []
        texts = mem.get("cards_applied") or []
        pairs = list(zip(ids, texts))
        match = next((t for i, t in pairs if str(i) == str(cid)), None)
        if match is None:
            return {"without": None, "control": None,
                    "note": "this card was not recorded as applied on this run (internalized memory "
                            "mode fuses cards into a trained prefix, or the card simply wasn't active "
                            "this turn) -- nothing to ablate"}
        import clozn.memory.mode as memory_mode
        without_texts = [t for i, t in pairs if str(i) != str(cid)]
        without_block = memory_mode.compile_prompt_block(without_texts)
        control_texts = [t if str(i) != str(cid) else _matched_length_filler(len(match)) for i, t in pairs]
        control_block = memory_mode.compile_prompt_block(control_texts)
        return {"without": {"block": without_block, "steer_strengths": with_strengths},
                "control": {"block": control_block, "steer_strengths": with_strengths}, "note": None}

    if influence.get("memory_off"):
        control = ({"block": _matched_length_filler(len(with_block)), "steer_strengths": with_strengths}
                  if with_block else None)
        return {"without": {"block": None, "steer_strengths": with_strengths}, "control": control,
                "note": None if with_block else "no active memory block on this run -- nothing to ablate"}

    dial = influence.get("dial")
    if dial:
        without_strengths = dict(with_strengths)
        without_strengths.pop(dial, None)
        control = None
        steer = getattr(sub, "steer", None)
        if steer is not None and hasattr(steer, "steer_vector") and with_strengths.get(dial):
            try:
                isolated = steer.steer_vector({dial: with_strengths[dial]})
            except Exception:
                isolated = None
            norm = _vector_norm(isolated) if isolated else 0.0
            if norm > 0:
                seed = f"{run.get('id')}:dial:{dial}"
                rand_vec = _random_vector_of_norm(len(isolated), norm, seed)
                control = {"block": with_block, "steer_strengths": without_strengths, "steer_vec": rand_vec}
        return {"without": {"block": with_block, "steer_strengths": without_strengths}, "control": control,
                "note": None if with_strengths.get(dial) else
                       f"dial '{dial}' was not active on this run -- nothing to ablate"}

    if influence.get("behavior_off"):
        control = None
        steer = getattr(sub, "steer", None)
        if (steer is not None and hasattr(steer, "steer_vector") and with_strengths
                and any(with_strengths.values())):
            try:
                full_vec = steer.steer_vector(with_strengths)
            except Exception:
                full_vec = None
            norm = _vector_norm(full_vec) if full_vec else 0.0
            if norm > 0:
                seed = f"{run.get('id')}:behavior_off"
                rand_vec = _random_vector_of_norm(len(full_vec), norm, seed)
                control = {"block": with_block, "steer_strengths": {}, "steer_vec": rand_vec}
        return {"without": {"block": with_block, "steer_strengths": {}}, "control": control,
                "note": None if with_strengths else "no active dial on this run -- nothing to ablate"}

    section = influence.get("section")
    if section:
        # A prompt-CONTENT ablation, not a block/steer swap -- see this module's docstring. Reuse
        # replay.py's own splice (`_apply_section_exclusions` already does the by-name manifest lookup +
        # per-name honest notes, including the raw-anchored-part case) against the RAW message list, since
        # that's the list an "api"/"auto" section's offsets were computed against.
        #
        # Callers are expected to have already filtered to api/auto sources (deltas._section_influences's
        # dedup rule -- see this module's docstring); a "memory_card" section's offsets are into
        # assembled_messages instead, which splicing against the RAW list here would get subtly wrong
        # rather than honestly failing. Guarded directly (not just trusted from the caller) so a
        # misdirected call degrades to a note instead of a silently-bad splice.
        manifest = run.get("sections") if isinstance(run.get("sections"), list) else []
        entry = next((s for s in manifest if isinstance(s, dict) and s.get("name") == section), None)
        if entry is not None and entry.get("source") == "memory_card":
            return {"without": None, "control": None,
                    "note": f"section '{section}' is memory-card-sourced -- ablate it via its card_id "
                            "instead (a section-name ablation here only supports api/auto sections)"}
        from clozn.replay.replay import _apply_section_exclusions
        without_messages, notes, applied = _apply_section_exclusions(
            run, list(run.get("messages") or []), [str(section)])
        note = notes.get(str(section))
        if not applied:
            return {"without": None, "control": None,
                    "note": note or f"section '{section}' has no usable parts on this run -- nothing to ablate"}
        # steer_strengths/block are UNCHANGED from the with-arm -- only the message content differs.
        return {"without": {"messages": without_messages, "block": with_block,
                            "steer_strengths": with_strengths},
                "control": None, "note": note}

    return None


def forced_receipt(run: dict, influence: dict, sub) -> dict | None:
    """One teacher-forced dependence receipt for one influence."""
    try:
        if not run or not isinstance(run, dict):
            return None
        if not isinstance(influence, dict) or not influence:
            return None
        conditions = rederive.with_arm_conditions(run)
        ablation = _forced_ablation(run, influence, sub, conditions)
        if ablation is None:
            return None
        if ablation.get("without") is None:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": ablation.get("note"), "caveat": _FORCED_CAVEAT}

        with_tokens, with_ok = rederive.score_arm(
            sub, conditions, messages=conditions["raw_messages"], block=conditions["raw_block"],
            steer_strengths=conditions["steer_strengths"])
        if not with_ok:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": "forced scoring needs the engine substrate (score_tokens is not available "
                            "here)", "caveat": _FORCED_CAVEAT}

        # A section ablation's "without" dict carries its OWN spliced `messages` (a content edit, not a
        # block/steer swap); every other kind leaves that key absent and gets `raw_messages` unchanged --
        # popped rather than passed alongside so a section's `messages` key never collides with the
        # explicit kwarg below.
        without_kwargs = dict(ablation["without"])
        without_messages = without_kwargs.pop("messages", conditions["raw_messages"])
        without_tokens, without_ok = rederive.score_arm(
            sub, conditions, messages=without_messages, **without_kwargs)
        if not without_ok:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": "the ablated arm could not be scored", "caveat": _FORCED_CAVEAT}

        deltas = _forced_deltas(with_tokens, without_tokens)
        if deltas is None:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": "with/without arms did not align token-for-token (a scoring inconsistency)",
                    "caveat": _FORCED_CAVEAT}

        pieces = [str(t.get("piece", "")) for t in with_tokens]
        summary = _delta_summary(deltas)
        has_effect = (summary["mean_nats_per_token"] >= _FORCED_MEAN_THRESHOLD
                     or abs(summary["sum_nats"]) >= _FORCED_SUM_THRESHOLD)

        out = {
            "influence": influence,
            "mode": "forced",
            "retokenized": conditions["retokenized"],
            "causal_verified": True,
            "answer_tokens": pieces,
            "deltas": [round(d, 6) for d in deltas],
            "sum_nats": summary["sum_nats"],
            "mean_nats_per_token": summary["mean_nats_per_token"],
            "top_dependent": _top_dependent(pieces, deltas),
            "has_effect": has_effect,
            "threshold": {"mean_abs_nats_per_token": _FORCED_MEAN_THRESHOLD,
                         "abs_sum_nats": _FORCED_SUM_THRESHOLD},
            "note": _FORCED_NOTE,
            "caveat": _FORCED_CAVEAT,
        }
        if ablation.get("note"):
            out["ablation_note"] = ablation["note"]

        control = ablation.get("control")
        if control is not None:
            control_kwargs = dict(control)
            control_messages = control_kwargs.pop("messages", conditions["raw_messages"])
            control_tokens, control_ok = rederive.score_arm(
                sub, conditions, messages=control_messages, **control_kwargs)
            control_deltas = _forced_deltas(with_tokens, control_tokens) if control_ok else None
            if control_deltas is not None:
                c_summary = _delta_summary(control_deltas)
                floor_mean = c_summary["mean_nats_per_token"]
                ratio = (summary["mean_nats_per_token"] / floor_mean) if floor_mean > 0 else None
                out["null_floor"] = {
                    "kind": ("card_filler" if influence.get("card_id") else
                            "block_filler" if influence.get("memory_off") else
                            "behavior_off_random_vector" if influence.get("behavior_off") else
                            "dial_random_vector"),
                    "deltas": [round(d, 6) for d in control_deltas],
                    "sum_nats": c_summary["sum_nats"],
                    "mean_nats_per_token": floor_mean,
                    "ratio_real_over_floor": round(ratio, 3) if ratio is not None else None,
                    "exceeds_floor_by_order_of_magnitude": bool(ratio is not None
                                                                and ratio >= _NULL_FLOOR_RATIO_MIN),
                }
        return out
    except Exception:
        return None
