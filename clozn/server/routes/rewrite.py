"""POST /engine/rewrite -- Route D ("Rewrite (AR)"), the second edit mode alongside Route A's diffusion
Resolve (the previous shell's pin-and-resolve; studio/heavn is no longer served from "/"). NO CALLER
in the current studio/app UI: still reachable over HTTP -- rewire or retire is an open call.
See notes/EDIT_INSTRUCTIONS_DESIGN.md's Route D: AR models follow instructions natively, where a base
diffusion model cannot -- this is a SECOND, honestly-labeled edit mode, not a replacement for Resolve.

HONESTY INVARIANT (binding, from the design doc): "Route D's UI/API must state 'regenerates the unpinned
text -- not a bidirectional resolve.'" This is a genuinely DIFFERENT operation from Resolve: there is no
backward propagation and no bidirectional re-masking here -- pins are prompt-level constraints the model
is ASKED to honor, not an engine-enforced invariant the way Resolve's held spans are (Resolve never even
sends the held text to the engine; the engine literally cannot touch it). That gap is why every response
below carries the honest `note` field verbatim, and why pin fidelity is MEASURED, never assumed: a plain
post-hoc substring check on the model's actual output. A pin the model paraphrased, dropped, or reworded
is reported `kept: false` -- never silently accepted as if the constraint were enforced.

Reuses the EXISTING EngineSubstrate.chat() plumbing (memory cards, tone dials, anchored memory, S5
sampling) completely unchanged -- zero engine (C++) changes, zero new generation primitive; this is
"just a constrained AR chat call" per the design doc. Pins mirror Route A's ON-THE-WIRE shape
([{start, end}] CHARACTER offsets into `text`, not byte offsets -- this endpoint runs in Python, which
slices strings by Unicode codepoint, matching Route A's JS `text.slice(a, b)` for all but astral-plane
(surrogate-pair) characters) so the studio pin-selection UI's existing `pins` state (an array of [a, b]
char-offset pairs, unchanged from edit.mjs) can drive BOTH edit modes with the same selection mechanic;
only the endpoint and the extra `instruction` field differ.
"""
import time

from clozn.server import app as ctx

# The exact phrase notes/EDIT_INSTRUCTIONS_DESIGN.md requires Route D's UI/API to state -- carried
# verbatim (including the em dash, matching the design doc's own wording and edit.mjs's existing UI-copy
# style) in every response, so the honesty invariant lives on the WIRE, not just in UI copy that could
# drift out of sync with it.
NOTE = "regenerates the unpinned text — not a bidirectional resolve"


def _valid_pin(p, n):
    """A pin is a {start, end} object with 0 <= start < end <= len(text) -- integers, a real (non-empty)
    span, in bounds. Anything else is rejected with a 400, never silently clamped or dropped: a caller
    that sent a bad span asked for something specific and deserves to know it didn't happen."""
    return (isinstance(p, dict) and isinstance(p.get("start"), int) and not isinstance(p.get("start"), bool)
            and isinstance(p.get("end"), int) and not isinstance(p.get("end"), bool)
            and 0 <= p["start"] < p["end"] <= n)


def _rewrite_prompt(text, pinned_texts, instruction):
    """(system, user) messages for the constrained AR rewrite. Pins become an explicit "keep verbatim"
    list quoted back at the model; the instruction rides as the user's actual ask. No pins -> the
    constraint paragraph is omitted entirely -- a plain free rewrite, honestly, with nothing claimed
    that isn't there."""
    sys_lines = [
        "You are a precise text-rewriting assistant. Rewrite the passage the user gives you, following "
        "their instruction exactly.",
    ]
    if pinned_texts:
        quoted = "\n".join(f'  - "{t}"' for t in pinned_texts)
        sys_lines.append(
            "HARD CONSTRAINT: the following phrase(s) MUST appear in your rewritten output EXACTLY as "
            "written -- verbatim, word for word, unchanged (do not paraphrase, translate, reorder "
            "internally, or alter capitalization or punctuation within them):\n" + quoted +
            "\nWeave them naturally into the rewrite; do not simply append them at the end.")
    sys_lines.append(
        "Output ONLY the rewritten passage -- no preamble, no explanation, no surrounding quotation "
        "marks, no commentary.")
    system = "\n\n".join(sys_lines)
    user = f"Instruction: {instruction}\n\nOriginal text:\n{text}"
    return system, user


def _verify_pins(pins, pinned_texts, rewritten):
    """MEASURED pin fidelity (receipts house style: measured, never self-reported/assumed) -- a pin is
    `kept` iff its EXACT original substring is present, verbatim, somewhere in the model's actual output.
    A plain containment check, deliberately not fuzzy: a near-miss (reordered words, a dropped comma) is
    exactly the case that must NOT silently read as kept. Returns [{start, end, text, kept}, ...] in the
    caller's pin order, so the UI can highlight exactly which ones broke."""
    return [{"start": p["start"], "end": p["end"], "text": t, "kept": t in rewritten}
            for p, t in zip(pins, pinned_texts)]


def try_post(h, p, body):
    if p != "/engine/rewrite":
        return False
    sub = ctx.active_sub(h)
    if ctx.ENGINE is None or not (sub and getattr(sub, "chat", None)):
        h._json(502, {"error": "model worker unavailable (CLOZN_ENGINE_PORT)"})
        return True

    text = body.get("text")
    if not isinstance(text, str) or not text:
        h._json(400, {"error": "need non-empty 'text'"})
        return True
    instruction = body.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        h._json(400, {"error": "need a non-empty 'instruction' -- Rewrite is the free-text edit mode; "
                               "use Resolve (pin & resolve) for instruction-free, propagating edits"})
        return True
    raw_pins = body.get("pins", [])
    if not isinstance(raw_pins, list):
        h._json(400, {"error": "'pins' must be a list of {start, end} character-offset objects"})
        return True
    n = len(text)
    for pin in raw_pins:
        if not _valid_pin(pin, n):
            h._json(400, {"error": f"invalid pin {pin!r}: need integer 0 <= start < end <= len(text) ({n})"})
            return True
    pins = [{"start": int(pin["start"]), "end": int(pin["end"])} for pin in raw_pins]
    pinned_texts = [text[pin["start"]:pin["end"]] for pin in pins]
    try:
        mx = int(body.get("max_tokens", 512))
    except (TypeError, ValueError):
        h._json(400, {"error": "max_tokens must be an integer"})
        return True
    if mx < 1:
        h._json(400, {"error": "max_tokens must be at least 1"})
        return True
    sample = bool(body.get("sample", True))

    system, user = _rewrite_prompt(text, pinned_texts, instruction)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    t0 = time.time()
    trace_steps = []
    memout = {}
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    if isinstance(sub, ctx.EngineSubstrate):
        # Mirrors openai.py's /v1/chat/completions: live interactive generation opts into X7 anchored
        # memory; receipts/replay/forced-scoring paths (which never call this route) keep the
        # deterministic baseline.
        chat_kw["apply_anchored"] = True
        from clozn.server.generation_gateway import request_memory_scope
        chat_kw["memory_scope"] = request_memory_scope(h)
    try:
        rewritten = sub.chat(messages, mx, sample, **chat_kw)
    except Exception as e:
        h._log_run("engine_rewrite", messages, "", "clozn-engine (rewrite)", t0, error=str(e), mem_out=memout)
        h._json(502, {"error": f"engine-rewrite: {e}"})
        return True

    verified = _verify_pins(pins, pinned_texts, rewritten)
    all_kept = all(v["kept"] for v in verified) if verified else None
    fr = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
    openai_fr = ctx._openai_finish_reason(fr)

    rid = h._log_run("engine_rewrite", messages, rewritten, "clozn-engine (rewrite)", t0, trace=trace_steps,
                     mem_out=memout, finish_reason=fr, finish_reason_fallback=None if fr else openai_fr,
                     extra_meta={"edit_route": "rewrite_ar", "pins_total": len(verified),
                                 "pins_kept": sum(1 for v in verified if v["kept"])})

    h._json(200, {"text": rewritten, "pins": verified, "all_pins_kept": all_kept,
                 "finish_reason": openai_fr, "note": NOTE, "run_id": rid})
    return True
