"""Teacher-forced receipt scoring and null-floor controls.

SECTION ABLATION (prompt-section influence, fast path): a "section" influence (`{"section": name,
"source": "api"|"auto"}`, from deltas._section_influences -- memory_card-sourced sections are dedup'd out
there and never reach this module) is the one kind here whose ablation is a prompt-CONTENT edit rather
than a block/steer knob swap. `_forced_ablation`'s section branch reuses replay.py's OWN
`_apply_section_exclusions`/`_strip_message_parts` splice (never a second copy of that span-removal logic)
against `run["messages"]` -- the RAW, pre-block-injection list, which is exactly what an "api"/"auto"
section's `parts[].message_index` is built from (see clozn.runs.sections's docstring + clozn.server.
routes.openai's try_post).

A section whose parts are ALL anchored to `final_prompt` (`message_index: null` -- a raw-prompt/CLI or
native-journaled run with no message breakdown) takes a DIFFERENT path (`_forced_raw_prompt_receipt` +
`_splice_final_prompt`, dispatched from `forced_receipt` via the `raw_prompt_ablation` sentinel
`_forced_ablation` returns for this case): there is no messages list to splice OR to feed
`EngineSubstrate.score_tokens` (its only prompt surface is `messages` -> `ctx._inject_block` ->
`ctx._engine_tmpl`), but `run["final_prompt"]` -- the exact string the model already saw -- IS the prompt,
so ablation means splicing the section's char spans directly out of THAT string (reusing replay.py's
`_strip_message_parts` verbatim by wrapping the string as a synthetic one-message list -- see
`_splice_final_prompt` -- never a second copy of the span-removal math) and teacher-forcing the stored
continuation against both versions via `EngineSubstrate.score_prompt_tokens` (the raw-prompt scoring
entry point this used to lack -- see that method's own docstring for why it's a separate seam from
score_tokens rather than an extra branch inside it). No block/steer reconstruction rides on this path: a
raw-prompt run's memory block and tone dials, if any, are already baked into its recorded `final_prompt`
text, so there's nothing separate left to hold constant between the two arms. A MIXED section (some
parts null, some message-anchored) is not expected from the producer (`clozn.runs.sections`'s schema);
if one shows up anyway, `_forced_ablation` prefers the message path, which already reports the raw-
anchored leftovers honestly via `_apply_section_exclusions`'s own partial-application note.

No null-floor control is computed for a section ablation on EITHER path (there's no register-matched
"equal-sized irrelevant text" analogue for an arbitrary prompt span the way a card has filler text or a
dial has a random direction); the receipt still reports its raw delta, just without a floor-clearing
ratio.
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

_FORCED_RAW_PROMPT_NOTE = (
    "this section is anchored to final_prompt (a raw-prompt/CLI or native-journaled run with no message "
    "breakdown), so both arms are scored directly against that recorded string via "
    "EngineSubstrate.score_prompt_tokens -- no block/steer reconstruction rides here (a raw run's memory "
    "block and tone dials, if any, are already baked into its final_prompt text; there's nothing separate "
    "left to hold constant between the two arms). The with/without prompts differ in length by exactly "
    "what was spliced out; deltas align per CONTINUATION token position, which is what matters -- not "
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


def _forced_result(influence: dict, conditions: dict, with_tokens: list, without_tokens: list,
                    note: str) -> dict:
    """with/without token-logprob arms -> the receipt's core shape (answer_tokens/deltas/sum_nats/
    mean_nats_per_token/top_dependent/has_effect/threshold/note/caveat) -- factored out so the
    message-anchored path (forced_receipt's own with/without scoring) and the raw-prompt sibling
    (_forced_raw_prompt_receipt, whose arms come from EngineSubstrate.score_prompt_tokens instead of
    score_tokens) build the IDENTICAL output shape from whatever tokens they scored, rather than each
    hand-rolling its own copy of this math. Degrades to a `causal_verified: False` dict (checkable via
    `not result.get("causal_verified")`) when the arms don't align token-for-token; never raises. Carries
    no `ablation_note`/`null_floor` -- callers that want those (only the generic message-anchored path
    does) add them to the returned dict themselves."""
    deltas = _forced_deltas(with_tokens, without_tokens)
    if deltas is None:
        return {"influence": influence, "mode": "forced", "causal_verified": False,
                "note": "with/without arms did not align token-for-token (a scoring inconsistency)",
                "caveat": _FORCED_CAVEAT}

    pieces = [str(t.get("piece", "")) for t in with_tokens]
    summary = _delta_summary(deltas)
    has_effect = (summary["mean_nats_per_token"] >= _FORCED_MEAN_THRESHOLD
                 or abs(summary["sum_nats"]) >= _FORCED_SUM_THRESHOLD)
    return {
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
        "note": note,
        "caveat": _FORCED_CAVEAT,
    }


def _splice_final_prompt(final_prompt: str, parts: list) -> str:
    """`final_prompt` with `parts`'s char spans (every one `message_index: null`) removed. Reuses
    replay.py's OWN `_strip_message_parts` verbatim -- never a second copy of the span-removal math -- by
    wrapping `final_prompt` as a synthetic ONE-message list (`_strip_message_parts` only cares that it
    gets a dict with a `content` string at the index its `parts_by_index` names; it has no idea, and no
    need to know, that the "message" here is really a whole raw prompt string). If the section's span(s)
    cover the WHOLE prompt, `_strip_message_parts`'s own whole-message rule drops that synthetic message
    entirely, which this reports as an empty ablated prompt -- an honest, if extreme, ablation, not a
    bug. Never raises: a part with a non-int start/end is skipped (mirrors `_apply_section_exclusions`'s
    own per-part tolerance)."""
    from clozn.replay.replay import _strip_message_parts
    spans = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        try:
            spans.append((int(p.get("start", 0)), int(p.get("end", 0))))
        except (TypeError, ValueError):
            continue
    if not spans:
        return final_prompt
    out = _strip_message_parts([{"content": final_prompt}], {0: spans})
    return out[0]["content"] if out else ""


def _score_raw_prompt(sub, prompt: str, continuation_ids, response_text: str):
    """rederive.score_arm's raw-prompt sibling: one `sub.score_prompt_tokens` call (never `score_tokens`
    -- there are no messages here to assemble), same continuation_ids-primary/response-text-fallback
    precedence. Returns (tokens, ok) -- ([], False) on any failure, including a substrate with no
    score_prompt_tokens (a torch lab substrate, or a test fake that only stubs score_tokens) -- never
    raises."""
    score = getattr(sub, "score_prompt_tokens", None)
    if not callable(score):
        return [], False
    try:
        if continuation_ids is not None:
            tokens = score(prompt, continuation_ids)
        else:
            if not response_text:
                return [], False
            tokens = score(prompt, None, continuation=response_text)
        return (tokens if isinstance(tokens, list) else []), True
    except Exception:
        return [], False


def _forced_raw_prompt_receipt(run: dict, influence: dict, sub, conditions: dict, raw: dict) -> dict:
    """The section ablation's raw-prompt sibling of forced_receipt's main with/without scoring: score the
    run's own stored continuation against `raw["prompt"]` (baseline = final_prompt unchanged) and
    `raw["ablated_prompt"]` (final_prompt with the section spliced out -- see _splice_final_prompt) via
    EngineSubstrate.score_prompt_tokens, then build the SAME receipt shape forced_receipt's generic path
    does (via the shared _forced_result) so a caller never has to know which path scored a given
    section."""
    ids = conditions.get("continuation_ids")
    response = conditions.get("response") or ""

    with_tokens, with_ok = _score_raw_prompt(sub, raw["prompt"], ids, response)
    if not with_ok:
        return {"influence": influence, "mode": "forced", "causal_verified": False,
                "note": "forced scoring needs the engine substrate (score_prompt_tokens is not available "
                        "here)", "caveat": _FORCED_CAVEAT}

    without_tokens, without_ok = _score_raw_prompt(sub, raw["ablated_prompt"], ids, response)
    if not without_ok:
        return {"influence": influence, "mode": "forced", "causal_verified": False,
                "note": "the ablated arm could not be scored", "caveat": _FORCED_CAVEAT}

    return _forced_result(influence, conditions, with_tokens, without_tokens, _FORCED_RAW_PROMPT_NOTE)


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

        # RAW-PROMPT section: every part anchored to final_prompt (message_index: null -- a raw-prompt/
        # CLI or native-journaled run with no message breakdown at all). A MIXED section (some parts
        # null, some message-anchored) is not expected from the producer (clozn.runs.sections's schema);
        # when it happens anyway, fall through to the message path below, which already reports any
        # raw-anchored leftovers honestly via _apply_section_exclusions's own partial-application note --
        # only an ALL-null section takes this branch. Returns a `raw_prompt_ablation` sentinel that
        # forced_receipt dispatches on BEFORE its generic messages-based with/without construction (see
        # _forced_raw_prompt_receipt) -- "without"/"control" stay None/None here since this ablation never
        # reconstructs block/steer arms the way every other kind does.
        if entry is not None:
            parts = entry.get("parts") if isinstance(entry.get("parts"), list) else []
            if parts and all(isinstance(p, dict) and p.get("message_index") is None for p in parts):
                final_prompt = run.get("final_prompt")
                if not isinstance(final_prompt, str) or not final_prompt:
                    return {"without": None, "control": None,
                            "note": f"section '{section}' is anchored to final_prompt offsets, but this "
                                    "run has no final_prompt recorded -- nothing to splice"}
                ablated_prompt = _splice_final_prompt(final_prompt, parts)
                return {"without": None, "control": None, "note": None,
                        "raw_prompt_ablation": {"prompt": final_prompt, "ablated_prompt": ablated_prompt}}

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

        # RAW-PROMPT section ablation (see _forced_ablation's section branch + this module's docstring):
        # a wholly different with/without construction (final_prompt splicing + score_prompt_tokens, no
        # messages/block/steer anywhere), so it's dispatched to its own builder and returned directly --
        # BEFORE the generic `ablation.get("without") is None` check below, since this sentinel also
        # carries "without": None (it never populates the messages-based ablation dict the generic path
        # expects).
        raw = ablation.get("raw_prompt_ablation")
        if raw is not None:
            return _forced_raw_prompt_receipt(run, influence, sub, conditions, raw)

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

        out = _forced_result(influence, conditions, with_tokens, without_tokens, _FORCED_NOTE)
        if not out.get("causal_verified"):
            return out
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
                ratio = (out["mean_nats_per_token"] / floor_mean) if floor_mean > 0 else None
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
