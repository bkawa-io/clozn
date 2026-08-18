"""replay.py -- the Replay & Compare engine (roadmap issue F1).

Re-run a stored run under a modified state (memory off, behavior neutral, a nudged/overridden dial, ...)
and persist the result as a CHILD run (parent_run_id set), so a replay is itself an inspectable run the
Studio can diff against its parent.

The load-bearing rule here: the LIVE studio must never be left mutated. A replay only *temporarily* changes
the substrate to generate one reply, then restores it exactly in a finally. It applies changes by writing
the same dials/strength the normal chat path reads:

  * memory  -- nothing. It wrote sub.memory.memory_strength and sub.memory._exclude_card_ids (REAL
               per-card ablation) until memory cards were cut on 2026-07-27; `memory_off` and
               `disabled_memory_ids` are now answered with an honest "not applied" note.
  * behavior -- sub.steer.strength (sub.chat() engages the hook, which reads .strength)
  * sections -- changes["exclude_sections"] (a list of section NAMES against the run's `sections`
               manifest -- the fixed schema clozn.runs.sections/store produce: id/name/source/parts/
               char_count/preview) rebuilds the MESSAGE LIST itself, with each named section's parts
               removed (whole-message drop when a part's span covers the entire content, a char-range
               splice otherwise; a section with several parts has all of them removed, right-to-left
               within a message so earlier offsets stay valid), and regenerates against that modified
               prompt -- a real, content-level ablation, unlike the knob toggles above. Offsets are
               read against `assembled_messages` when the run has it (that's what the model actually
               saw); a part anchored to `final_prompt` (message_index: null -- a raw-prompt run with no
               message breakdown) can't be spliced through this path (chat() takes a message list, not
               a raw prompt override -- that's the Branch Fan engine.complete seam, not this one's) and is
               left in place with an honest note. Unknown section names, or a run with no `sections`
               manifest at all (predates section capture), are the SAME best-effort no-op + honest note
               `disabled_memory_ids` already uses for a card ablation that can't apply in the active mode.

so a replay is exactly "chat, but with these knobs (or, now, this prompt content) different for one turn".
The temporary dials are NEVER persisted (no save_state during replay) -- that would silently rewrite the
user's personality.

Stdlib + the sibling `runlog` (stdlib-only itself); the
substrate is passed in (the live SUB), never imported, so this module is unit-testable against a fake
substrate with no model.
"""
from __future__ import annotations

import time

import clozn.runs.store as runlog

NUDGE_STEP = 0.5            # a "nudge" bumps one dial this far toward its + pole (then set() caps it per-axis)


def _inject_prompt_instructions(messages: list[dict], instructions) -> list[dict]:
    """Add request-local system instructions without changing delivered messages.

    This mirrors prompt-memory assembly: caller system context remains first and the
    Clozn-owned block is appended to it, otherwise a system message is prepended.
    The journal still records ``messages`` unchanged; only assembled/final prompt
    evidence contains this intervention.
    """
    blocks = [str(value).strip() for value in (instructions or []) if str(value).strip()]
    if not blocks:
        return [dict(message) for message in messages]
    block = "\n\n".join(blocks)
    copied = [dict(message) for message in messages]
    for message in copied:
        if message.get("role") == "system":
            message["content"] = (str(message.get("content") or "") + "\n\n" + block).strip()
            return copied
    return [{"role": "system", "content": block}] + copied


def _mode() -> str:
    """Constant "none" since the 2026-07-27 cards cut removed memory from the product.

    This used to read the prompt-vs-internalized memory mode. Both mechanisms are gone; the value is
    kept on replay's own records (not the run's) so a reader can tell a post-cut replay apart from a
    pre-cut one rather than seeing a silently absent field."""
    return "none"


def _snapshot_strength(steer) -> dict:
    """A shallow copy of the dial dict we can restore verbatim later (values are floats)."""
    try:
        return dict(getattr(steer, "strength", {}) or {})
    except Exception:
        return {}


def _apply_changes(changes: dict, sub, mode: str) -> dict:
    """Mutate the live substrate's knobs per `changes`, in place. Returns a small dict of notes for the
    parts that CAN'T take effect in the active memory mode (honest, never silently pretended). Never
    raises."""
    notes: dict = {}
    steer = getattr(sub, "steer", None)

    # --- memory ---
    # (memory_off / disabled_memory_ids drove the strength dial and the per-card block ablation until
    # the 2026-07-27 cards cut. There is no memory to suppress or ablate now, so both are reported as
    # not-applied rather than silently accepted -- a change a caller asked for and did not get must
    # never read as a change that was made.)
    for _dead in ("memory_off", "disabled_memory_ids"):
        if changes.get(_dead):
            notes[_dead] = ("not applied: memory cards were removed from the product; "
                            "steering is what shapes a reply now")
    if changes.get("edited_memory"):
        notes["edited_memory"] = ("not applied: memory-card editing is not wired yet; "
                                  "use memory_off to compare with/without memory")

    # --- behavior / tone dials ---
    if steer is not None:
        if changes.get("behavior_off"):
            steer.clear()                                   # neutral: drop every dial for this turn

        overrides = changes.get("behavior_overrides")
        if isinstance(overrides, dict):
            for name, val in overrides.items():
                try:
                    steer.set(str(name), float(val))        # set() caps to the axis's per-axis max
                except Exception:
                    pass

        nudge = changes.get("nudge")
        if nudge:
            try:
                cur = float(getattr(steer, "strength", {}).get(str(nudge), 0.0))
                steer.set(str(nudge), cur + NUDGE_STEP)     # bump toward the + pole; set() caps it
            except Exception:
                pass

    return notes


def _effective_dials(sub) -> dict:
    """The dials actually in force after applying the changes (what shaped the child reply)."""
    steer = getattr(sub, "steer", None)
    if steer is None:
        return {}
    try:
        if hasattr(steer, "active"):
            return dict(steer.active())
        return {k: v for k, v in _snapshot_strength(steer).items() if v}
    except Exception:
        return {}


# --- section ablation (the fixed schema: clozn.runs.sections/store's `sections` manifest) -----------------
#
# A section is a named span of prompt content (id/name/source/parts/char_count/preview -- see the
# manifest producer). `parts` are char spans keyed by `message_index` into whichever message list the
# manifest was built against (assembled_messages when the run has one -- that's what the model actually
# saw -- else the raw messages); `message_index: null` means the span is anchored to `final_prompt`
# instead, which this module has no way to feed back into `sub.chat()` (its only surface is a MESSAGE
# list, not a raw prompt string -- that override exists on the engine's raw-completion seam, which is
# Branch Fan's territory, not replay's). We splice everything we CAN (message-anchored parts) and leave
# final_prompt-anchored parts in place with an honest note, rather than silently pretending they were
# removed.

def _strip_message_parts(messages: list, parts_by_index: dict) -> list:
    """A NEW message list with each message's listed char spans removed. `parts_by_index` is
    {message_index: [(start, end), ...]}. Per the schema: a span covering the WHOLE content drops that
    message entirely; otherwise every span for that message is spliced out, right-to-left (highest
    start first) so an earlier span's offsets are never invalidated by a later one's removal. Never
    raises: an out-of-range index, or a non-dict message, is simply left untouched."""
    if not parts_by_index:
        return list(messages or [])
    msgs = list(messages or [])
    out = []
    for i, m in enumerate(msgs):
        spans = parts_by_index.get(i)
        if not spans or not isinstance(m, dict):
            out.append(m)
            continue
        content = str(m.get("content", ""))
        whole = any(int(s) <= 0 and int(e) >= len(content) for s, e in spans)
        if whole:
            continue                                       # the whole message is one of this section's parts
        for s, e in sorted(spans, key=lambda p: p[0], reverse=True):
            s = max(0, min(int(s), len(content)))
            e = max(0, min(int(e), len(content)))
            if e < s:
                s, e = e, s
            content = content[:s] + content[e:]
        m = dict(m)
        m["content"] = content
        out.append(m)
    return out


def _apply_section_exclusions(run: dict, base_messages: list, names) -> tuple[list, dict, list]:
    """`base_messages` minus the parts of each named section in `run["sections"]`. Returns
    (new_messages, notes, applied): `notes` maps a requested name to an honest reason it did NOT apply
    in full (no manifest at all, an unknown name, no usable parts, or a final_prompt-anchored part this
    surface can't splice) -- exactly the `disabled_memory_ids` convention (a best-effort no-op is
    reported, never silently swallowed); `applied` lists the names that DID remove at least one part.
    Never raises."""
    notes: dict = {}
    applied: list = []
    if isinstance(names, (list, tuple, set)):
        names = [str(n) for n in names if n]
    elif names:
        names = [str(names)]
    else:
        names = []
    if not names:
        return list(base_messages or []), notes, applied

    manifest = run.get("sections") if isinstance(run, dict) else None
    if not isinstance(manifest, list) or not manifest:
        for n in names:
            notes[n] = "not applied: this run carries no section manifest (predates section capture)"
        return list(base_messages or []), notes, applied

    by_name: dict = {}
    for sec in manifest:
        if isinstance(sec, dict) and sec.get("name"):
            by_name.setdefault(str(sec["name"]), sec)

    n_messages = len(base_messages or [])
    parts_by_index: dict = {}
    for n in names:
        sec = by_name.get(n)
        if sec is None:
            notes[n] = f"not applied: no section named '{n}' on this run's manifest"
            continue
        parts = sec.get("parts")
        if not isinstance(parts, list) or not parts:
            notes[n] = f"not applied: section '{n}' has no parts recorded"
            continue
        removed_any = False
        raw_anchored = False
        for p in parts:
            if not isinstance(p, dict):
                continue
            idx = p.get("message_index")
            if idx is None:
                raw_anchored = True                        # anchored to final_prompt -- can't splice here
                continue
            try:
                idx = int(idx)
                s, e = int(p.get("start", 0)), int(p.get("end", 0))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= n_messages:
                continue
            parts_by_index.setdefault(idx, []).append((s, e))
            removed_any = True
        if removed_any:
            applied.append(n)
            if raw_anchored:
                notes[n] = (f"partially applied: section '{n}' has part(s) anchored to final_prompt "
                            "offsets (no message breakdown), which replay's chat(messages, ...) surface "
                            "cannot splice -- only its message-anchored parts were removed")
        elif raw_anchored:
            notes[n] = (f"not applied: section '{n}' is anchored entirely to final_prompt offsets "
                        "(a raw-prompt run) -- replay's chat(messages, ...) surface has no raw-prompt "
                        "override to splice against")
        else:
            notes[n] = f"not applied: section '{n}' has no usable parts"

    new_messages = _strip_message_parts(base_messages, parts_by_index)
    return new_messages, notes, applied


def replay(run: dict, changes: dict, sub, reference_tokens=None, *,
           prompt_instructions=None, max_new: int | None = None,
           messages_override: list[dict] | None = None,
           sampling_override: bool | dict | None = None) -> dict | None:
    """Re-run `run` under `changes` on the live substrate `sub`; record the result as a child run and return
    it. Returns None on any failure (a replay must never raise into the request handler).

    `run`      -- a run dict from runlog.get_run(id) (needs at least "id" and "messages").
    `changes`  -- the change spec (see module docstring): memory_off / behavior_off / nudge /
                  behavior_overrides / disabled_memory_ids / edited_memory / exclude_sections / plain.
                  {} == a plain re-roll.
    `sub`      -- the live substrate (SUB); must expose .chat(messages, max_new=, sample=). Its
                  .memory.memory_strength and .steer.strength are snapshotted and restored around generation.
    `reference_tokens` -- optional baseline reply token ids (prove-all early-stop): when the substrate's
                  chat() supports it, this ablated arm's generation HALTS at the first token that differs
                  from the baseline, so the child's `response` is a bit-exact prefix of the full reply and
                  the child carries `diverged`/`diverged_at`. A substrate whose chat() lacks the kwarg (torch
                  test fakes) simply generates fully -- correctness is preserved because the
                  receipt layer falls back to the string compare when `diverged` is absent. The returned
                  child ALSO always carries `generated_ids` (the committed token ids, tier-independent), so a
                  baseline replay can hand its own tokens to the ablated arms even at a trace-dropping tier."""
    try:
        if not run or not isinstance(run, dict):
            return None
        from clozn.runs.think_tags import sanitize_messages
        source_messages = run.get("messages") or []
        if messages_override is not None:
            if not isinstance(messages_override, list):
                return None
            source_messages = messages_override
        messages = sanitize_messages(source_messages)
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return None
        changes = changes or {}
        mode = _mode()

        # section ablation: rebuild `messages` itself (a content change, not a knob) BEFORE anything else
        # reads it -- offsets are defined against assembled_messages when the run has one (that's what the
        # model actually saw), else the raw messages. Never touches `messages` at all when this change key
        # is absent, so every existing (knob-only) replay path is byte-for-byte unchanged.
        section_notes: dict = {}
        sections_applied: list = []
        exclude_sections = changes.get("exclude_sections")
        if exclude_sections:
            base_messages = run.get("assembled_messages")
            if not isinstance(base_messages, list) or not base_messages:
                base_messages = messages
            messages, section_notes, sections_applied = _apply_section_exclusions(
                run, base_messages, exclude_sections)

        # Built AFTER section ablation (above), from whatever `messages` generation should actually see --
        # a request-local instruction rides on TOP OF an already-ablated prompt, never masking the
        # ablation by being derived from the pre-exclusion messages. The journal still records `messages`
        # itself unchanged by this step (see _inject_prompt_instructions' own docstring).
        generation_messages = _inject_prompt_instructions(messages, prompt_instructions)

        steer = getattr(sub, "steer", None)

        # snapshot the exact live state so we can restore it verbatim (never leave the studio mutated).
        # memory_strength / _exclude_card_ids were snapshotted here too, until the 2026-07-27 cards cut.
        saved_strength = _snapshot_strength(steer)

        t0 = time.time()
        notes = _apply_changes(changes, sub, mode)
        if section_notes:
            notes["exclude_sections"] = section_notes
        eff_dials = _effective_dials(sub)
        trace_steps: list = []          # per-token trace of the replay reply (B3) -- the baseline-vs-replay
        replay_memout: dict = {}        # exact post-change assembled/rendered prompt for the child receipt
        #                                 token diff needs it; replay previously never passed trace_out.
        try:
            # greedy:true (the receipts path) decodes deterministically, so the original-vs-replayed
            # difference is attributable to the CHANGE, not to sampling dice. Default stays sampled.
            # Capture the per-token trace when chat supports it (the real substrates do); fall back for a
            # chat that predates trace_out -- replay's sub contract is just (messages, max_new=, sample=).
            sampled = (sampling_override if isinstance(sampling_override, (bool, dict))
                       else not bool(changes.get("greedy")))
            # Build the call kwargs and drop any the substrate's chat() doesn't accept (a fake
            # / test fakes predate trace_out and/or reference_tokens). Progressive-degrade on the exact
            # unknown kwarg named in the TypeError, so the reply is never lost -- just less instrumented.
            budget = int(max_new) if isinstance(max_new, int) and max_new > 0 else 256
            call_kw = {"max_new": budget, "sample": sampled, "trace_out": trace_steps,
                       "mem_out": replay_memout}
            if reference_tokens:
                call_kw["reference_tokens"] = reference_tokens
            while True:
                try:
                    reply = chat(generation_messages, **call_kw)
                    break
                except TypeError as e:
                    msg = str(e)
                    dropped = next((k for k in ("reference_tokens", "trace_out", "mem_out")
                                    if k in call_kw and k in msg), None)
                    if dropped is None:
                        raise                                # a real TypeError from inside chat, not a kwarg
                    del call_kw[dropped]
        finally:
            # restore EXACTLY -- and never persist the temporary dials (no save_state here).
            if steer is not None:
                try:
                    steer.strength = dict(saved_strength)
                except Exception:
                    pass

        reply = reply if isinstance(reply, str) else str(reply)

        # Capture the committed token ids NOW, from the in-memory trace -- BEFORE the capture-tier logic
        # below may drop trace_steps to []. A baseline replay hands these to its ablated arms as the
        # early-stop reference, and they must survive even at a trace-dropping tier.
        generated_ids = [int(s["id"]) for s in (trace_steps or [])
                         if isinstance(s, dict) and s.get("id") is not None]
        # The early-stop verdict (prove-all ablated arms): (diverged, diverged_at) or (None, None).
        diverged = diverged_at = None
        if hasattr(sub, "last_divergence"):
            try:
                diverged, diverged_at = sub.last_divergence()
            except Exception:
                diverged = diverged_at = None

        # the replay's own stop cause + repro metadata (engine substrate) -- the SAME fields a live run
        # carries, read after generation (the finally above doesn't touch these stashes). Per-substrate
        # best-effort: a substrate without them (e.g. an HF stub) simply records None / {}.
        finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
        meta = None
        try:
            if hasattr(sub, "run_meta"):
                meta = sub.run_meta() or None
        except Exception:
            meta = None
        # capture tier: record it, and drop the trace at light -- the same record policy as the live path.
        try:
            from clozn.runs import capture_mode
            _tier = capture_mode.tier()
            meta = {**(meta or {}), "capture_tier": _tier}
            if not capture_mode.captures_trace(_tier):
                trace_steps = []
        except Exception:
            pass

        # child memory summary: what memory looked like *for this replay* (strength reflects memory_off).
        # In prompt mode the summary is card-store-based and honors the per-card ablation: cards_applied
        # is the ELIGIBLE set (active minus the disabled ids) + applied_ids for the per-card receipt UI.
        # (Eligible, not per-turn-gated: replay can't see inside sub.chat; same convention as live
        # internalized runs, which record the whole active set.)
        # The child's memory summary (eligible cards, ids, scope kinds, strength, has_prefix) was
        # assembled here. Memory cards were cut on 2026-07-27, so a replay has nothing to summarize;
        # `notes` still travels so a caller learns their memory_off/disabled_memory_ids was not applied.
        memd = {"mode": mode}
        if notes:
            memd["notes"] = notes

        meta = {**(meta or {}), "max_tokens": budget}
        identity = None
        try:
            if hasattr(sub, "identity_meta"):
                identity = sub.identity_meta() or None
        except Exception:
            identity = None
        rid = runlog.record(
            source="replay",
            client=run.get("client") or "studio",
            model=run.get("model"),
            substrate=run.get("substrate"),
            messages=messages,
            response=reply,
            memory=memd,
            behavior={"active_dials": eff_dials},
            trace=trace_steps,
            finish_reason=finish,
            meta=meta,
            parent_run_id=run.get("id"),
            changes_applied=changes,
            started=t0,
            assembled_messages=replay_memout.get("assembled_messages"),
            final_prompt=replay_memout.get("final_prompt"),
            identity=identity,
            session_key=run.get("session_key"),
            client_key=run.get("client_key"),
            client_key_source=run.get("client_key_source"),
            project_key=run.get("project_key"),
        )
        if rid is None:
            return None
        child = runlog.get_run(rid)
        if child is None:
            child = {"id": rid, "response": reply, "parent_run_id": run.get("id")}
        # Attach the early-stop bookkeeping to the returned child (not persisted in the run record's core
        # fields -- these are for the receipt orchestrator that called replay). `generated_ids` is the
        # committed-token reference a baseline hands to its ablated arms; `diverged`/`diverged_at` let the
        # receipt read the verdict without re-deriving it.
        child["generated_ids"] = generated_ids
        if diverged is not None:
            child["diverged"] = diverged
            child["diverged_at"] = diverged_at
        if exclude_sections:
            # names that actually removed something (vs. skipped -- see `section_notes`, which already
            # rode into memory.notes.exclude_sections above and is what the receipt layer's honesty check
            # reads); a convenience mirror for any caller inspecting the child directly.
            child["sections_excluded"] = sections_applied
        return child
    except Exception:
        return None
