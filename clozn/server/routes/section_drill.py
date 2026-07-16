"""section_drill.py -- POST /runs/<id>/section-drill: recursive drill-down for ONE high-influence section
found by /runs/<id>/section-influence -- "which sentence", not just "which section".

WHY THIS EXISTS. Section-influence (section_influence.py) answers "which SECTION of the prompt mattered"
-- a ranked list of shares over the run's own `sections` manifest. That's the right first cut, but a
section can be a whole RAG paragraph, a multi-sentence memory card, or a few-shot block, and "this whole
paragraph has 71% influence" is often not the end of the investigation -- the natural next question is
"which SENTENCE inside it actually carried that". This route answers that, at the granularity
clozn.runs.sections.drill_split finds (sentence/line-level spans), by re-running the EXACT SAME
teacher-forced machinery section-influence uses, just against finer synthetic sections instead of the
run's own recorded ones.

HOW IT REUSES SECTION-INFLUENCE'S MACHINERY, NOT A SECOND COPY OF IT:
  1. `drill_split` (clozn.runs.sections) splits the target section's OWN resolved text into finer
     (start, end) sub-spans -- pure text math, no knowledge of prompt coordinates at all.
  2. This route remaps each sub-span back into a REAL prompt-coordinate part -- `(message_index,
     part.start + a, part.start + b)` -- and packages it as a SYNTHETIC, schema-shaped section dict
     (`_synthetic_subsections`), named "<parent>.1", "<parent>.2", ... Only single-part sections are
     drilled: a multi-part section's parts can land in different messages entirely, so a sub-span found on
     the CONCATENATED text has no single, unambiguous `(message_index, start, end)` it maps to without
     re-deriving which physical part each character came from -- rather than guess and risk an incorrect
     ablation, this route declines multi-part sections outright, with an honest note (see _MULTIPART_NOTE).
  3. The synthetic sub-sections are appended to a SHALLOW COPY of the run's own `sections` manifest (never
     mutating the stored run) -- `clozn.receipts.forced.forced_receipt`'s section-ablation branch looks up
     an influence's `{"section": name}` by NAME in `run.get("sections")` (see forced.py's own module
     docstring), so putting the synthetic entries there is what lets `forced_receipt` splice/score them
     exactly as if they were real recorded sections -- no new ablation or splicing logic anywhere in this
     file. `message_index: None` sub-sections (a raw-prompt/CLI run) fall straight into forced_receipt's own
     raw-prompt sibling path automatically; this route never needs to know which path fired.
  4. Per-sub-section scoring reuses `section_influence._section_score` (the SAME forced_receipt(run,
     {"section": ..., "source": ...}, sub) call section-influence makes) and `_shares`/`_summary`
     (imported, never re-derived) -- `influence_share` here is scoped WITHIN the drilled section (shares
     over its own sub-sections sum to ~1), a DIFFERENT number from the parent's cross-section share.

Never raises into the request path: a missing/unknown `section`, a memory-card-sourced section (its parts
are anchored into `assembled_messages`, not the raw `messages` list this splices against -- see forced.py's
own dedup note), an unsplittable section, or a substrate that can't score all degrade to an honest JSON
body, never a stack trace.
"""
from __future__ import annotations

import clozn.runs.store as runlog
from clozn.receipts import rederive
from clozn.runs import sections as clozn_sections
from clozn.server import app as ctx
from clozn.server.routes import section_influence as si

_NOTE = ("Approximate influence via log-probability delta under teacher forcing, WITHIN this one section "
        "-- NOT causal proof, and NOT comparable across a different parent section's shares (each drill's "
        "shares sum to ~1 over its OWN sub-sections only). POST /runs/<id>/receipts for full ablation "
        "receipts.")

_UNSPLITTABLE_NOTE = (_NOTE + " This section has nothing finer to split (drill_split found no sentence "
                     "boundary or hard newline inside it) -- reporting it as a single, whole-section "
                     "sub-section rather than fabricating a split that isn't there.")

# Surfaced (with any_meaningful:false) when the section DID split but no sub-part measurably dominates:
# each sub-section's removal moved the stored answer's fit within noise, so the within-section shares are
# not a meaningful ranking (same honesty guard as section_influence, scoped to sub-sections).
_NO_SUBEFFECT_NOTE = ("No single part of this section measurably dominates: every sub-section's removal "
                     "moved the stored answer's fit within noise. The shares below are not a meaningful "
                     "ranking. " + _NOTE)

_MULTIPART_NOTE = ("This section has more than one part (it rode in more than one message/prompt region). "
                   "A sub-span found on its CONCATENATED text has no single, unambiguous prompt-coordinate "
                   "mapping without re-deriving which physical part each character came from -- this route "
                   "declines to guess rather than risk an incorrect ablation. Ablate the section as a "
                   "whole via POST /runs/<id>/section-influence instead.")


def _find_section(manifest: list, name: str) -> dict | None:
    """Match by `name` OR `id` (mirrors clozn.runs.sections.resolve's own "id or name" convention) -- a
    caller drilling a section it just saw in a /section-influence response may have either handy."""
    for s in manifest:
        if isinstance(s, dict) and (s.get("name") == name or s.get("id") == name):
            return s
    return None


def _synthetic_subsections(entry: dict, real_name: str, part: dict, text: str,
                           spans: list) -> list:
    """One `drill_split` span -> one schema-shaped SYNTHETIC section dict, offsets remapped from the
    section's own concatenated-text coordinates back to real prompt coordinates: `(a, b)` on `text`
    becomes `(part["start"] + a, part["start"] + b)` on whatever `part["message_index"]` already pointed
    at (a `messages` index, or None for a `final_prompt`-anchored/raw-prompt section) -- the COMMON case
    the module docstring describes, valid here because the caller has already confirmed `entry` has
    exactly one part before calling this. `source`/`message_index` are carried over from the parent
    unchanged: a sub-section of an "auto" section is still "auto" (same ablatability), and an already-
    None `message_index` still means "splice final_prompt", never a `messages` list."""
    source = entry.get("source") or "auto"
    mi = part.get("message_index")
    base_start = part["start"]
    parent_id = entry.get("id") or real_name
    out = []
    for i, (a, b) in enumerate(spans, start=1):
        piece = text[a:b]
        out.append({
            "id": f"{parent_id}_drill_{i}",
            "name": f"{real_name}.{i}",
            "source": source,
            "parts": [{"message_index": mi, "start": base_start + a, "end": base_start + b}],
            "char_count": b - a,
            "preview": piece[:clozn_sections.PREVIEW_CHARS],
        })
    return out


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith("/section-drill")):
        return False
    rid = p[len("/runs/"):-len("/section-drill")]
    run = runlog.get_run(rid)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    requested = (body or {}).get("section")
    if not isinstance(requested, str) or not requested.strip():
        h._json(400, {"error": "need a 'section' name to drill into"})
        return True
    requested = requested.strip()

    manifest = run.get("sections")
    if not isinstance(manifest, list) or not manifest:
        h._json(400, {"error": f"no section named '{requested}' on this run's manifest "
                              "(this run carries no section manifest at all)"})
        return True

    entry = _find_section(manifest, requested)
    if entry is None:
        h._json(400, {"error": f"no section named '{requested}' on this run's manifest"})
        return True
    real_name = entry.get("name") or requested

    if entry.get("source") == "memory_card":
        h._json(400, {"error": f"section '{real_name}' is memory-card-sourced -- drill-down only supports "
                             "api/auto sections (its parts are anchored into assembled_messages, not the "
                             "raw messages list this route splices against; ablate it via its card_id "
                             "through POST /runs/<id>/receipts instead)"})
        return True

    parts = entry.get("parts")
    if not isinstance(parts, list) or not parts:
        h._json(400, {"error": f"section '{real_name}' has no usable parts on this run -- nothing to drill"})
        return True

    if len(parts) != 1:
        h._json(200, {"run_id": rid, "method": "teacher_forced", "note": _MULTIPART_NOTE,
                     "parent_section": real_name, "sub_sections": []})
        return True

    text = clozn_sections.resolve(run, real_name)
    if not text:
        h._json(200, {"run_id": rid, "method": "teacher_forced",
                     "note": f"section '{real_name}' resolves to empty text on this run -- nothing to drill",
                     "parent_section": real_name, "sub_sections": []})
        return True

    spans = clozn_sections.drill_split(text)

    sub = ctx.active_sub(h)
    if not (sub and getattr(sub, "score_tokens", None)):
        h._json(503, {"error": "section-drill requires worker token scoring"})
        return True

    subs = _synthetic_subsections(entry, real_name, parts[0], text, spans)
    run_copy = dict(run)
    run_copy["sections"] = list(manifest) + subs

    # The baseline: the run's stored answer scored against the ORIGINAL, wholly-unablated prompt -- the
    # same "everything present" reference section-influence.py uses, computed identically (one call,
    # shared across every sub-section's delta below; see that module's own cost-note for why this is
    # deliberately redone rather than threaded through as a precomputed value).
    conditions = rederive.with_arm_conditions(run)
    with_tokens, with_ok = rederive.score_arm(
        sub, conditions, messages=conditions["raw_messages"], block=conditions["raw_block"],
        steer_strengths=conditions["steer_strengths"])
    if not with_ok:
        h._json(503, {"error": "section-drill requires worker token scoring"})
        return True
    baseline_logprob = round(sum(float(t.get("logprob") or 0.0) for t in with_tokens
                                if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))), 6)

    scored = []
    for sub_sec in subs:
        try:
            s = si._section_score(run_copy, sub_sec, sub)
        except Exception:
            s = None
        if s is not None:
            s["preview"] = sub_sec.get("preview", "")
            scored.append(s)

    for s, share in zip(scored, si._shares([s["log_prob_delta"] for s in scored])):
        s["influence_share"] = share
    scored.sort(key=lambda s: -s["influence_share"])   # biggest measured effect first, matches section-influence
    sub_sections = [{"name": s["name"], "preview": s["preview"], "influence_share": s["influence_share"],
                     "log_prob_delta": s["log_prob_delta"], "per_token_delta": s["per_token_delta"],
                     "summary": s["summary"]} for s in scored]

    meaningful = si.any_meaningful(scored)
    if len(spans) <= 1:
        note = _UNSPLITTABLE_NOTE
    elif not meaningful:
        note = _NO_SUBEFFECT_NOTE
    else:
        note = _NOTE
    h._json(200, {"run_id": rid, "method": "teacher_forced", "note": note,
                 "parent_section": real_name, "baseline_logprob": baseline_logprob,
                 "any_meaningful": meaningful,
                 "sub_sections": sub_sections})
    return True
