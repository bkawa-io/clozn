"""section_influence.py -- POST /runs/<id>/section-influence: quick, teacher-forced per-section influence
scores with NO regeneration (prompt-section influence, fast path).

WHY THIS EXISTS. `/runs/<id>/receipts` (mode=forced) already computes a rigorous teacher-forced receipt per
FIRED influence (clozn.receipts.forced.forced_receipt) -- including, for a card/dial, a null-floor control
-- but that's the "prove everything" batch call: one big object per influence (a per-token deltas array,
top_dependent, a caveat paragraph, ...), card/dial/behavior-off receipts mixed in alongside sections, and a
second /score call per receipt for the null floor nobody needs here. What a per-section overview (the
Studio's section panel; `clozn run --show-influence`; the REPL's `/influence`) actually wants is much
smaller: one ranked number per prompt SECTION, fast. This route is a THIN reshape of forced_receipt's own
machinery -- same splice helpers, same /score seam, same honesty rules -- into the flatter, FIXED contract
the UI is built against (do not change these field names):

    {"run_id", "method": "teacher_forced", "note": <the approximation disclaimer>,
     "baseline_logprob": <float>,
     "sections": [{"id", "name", "source", "log_prob_delta", "influence_share", "per_token_delta",
                   "summary"}, ...]}

`log_prob_delta` = (ablated total logprob) - (baseline total logprob) of the stored response tokens --
negative means the answer fits WORSE without that section. Note this is the OPPOSITE sign convention from
forced.py's own `sum_nats` (with-minus-without); the flip happens once, at `_section_score`, right where
forced_receipt's number comes in.  `influence_share` = this section's slice of the total measured |effect|
across every scored section (0.0 for every section, never NaN, when every delta is ~0 -- see the shared
`denom = max(total_abs, 1e-9)` below).  `per_token_delta` = the delta spread over the stored response's own
token count, so a long vs. short answer's deltas are comparable.  `summary` is a bucketed, honest phrase
("negligible"/"small"/"substantial" x "better"/"worse") -- never a causal claim.

DEDUP (why a "memory_card" section is silently absent from `sections`, never surfaced as an error): that
section is the SAME fired influence already covered, more richly, by a card_id receipt elsewhere
(clozn.receipts.deltas._section_influences's dedup rule -- restated here for the identical reason: scoring
it again here would double-count one real cause under two names). It also has a practical blocker: its
`parts[].message_index` point into `assembled_messages`, not the raw message list this route's splice
(reused from clozn.replay.replay, via forced.py) works against -- see clozn.receipts.forced's own module
docstring for the full raw-vs-assembled story. Only "api"/"auto" sourced sections are scored.

COST, stated honestly: this calls `forced_receipt()` once per ablatable section, and forced_receipt itself
always re-scores its own WITH arm internally -- so this route redoes that one (cheap: one forward pass over
the stored continuation, not a 256-token regeneration) baseline scoring call N+1 times rather than 1. Traded
deliberately for reusing forced_receipt() verbatim instead of threading a precomputed baseline through a
second, parallel scoring path -- correctness and one tested implementation over shaving a few /score calls
that are already the "fast" side of this feature's cost model (see deltas._cost_note: a section ablation is
priced like a memory ablation -- no KV reuse -- but scoring a short continuation is still far cheaper than
generating one).

Stdlib-only route glue: the scoring math is 100% clozn.receipts.forced / clozn.receipts.rederive, reused
verbatim -- this file never touches a splice or a /score call directly.
"""
from __future__ import annotations

import clozn.runs.store as runlog
from clozn.receipts import rederive
from clozn.receipts.forced import _FORCED_MEAN_THRESHOLD, forced_receipt
from clozn.server import app as ctx

_NOTE = ("Approximate influence via log-probability delta under teacher forcing; NOT causal proof. "
        "POST /runs/<id>/receipts for full ablation receipts.")
_NO_MANIFEST_NOTE = "no section manifest on this run"
# Surfaced when NOT any section cleared the per-token effect floor: manual probing showed that on a
# parametric-knowledge answer (the reply came from the model's own weights, not the prompt), every
# section's removal moves the fit within noise -- yet `influence_share` still normalizes to sum=1 and
# manufactures a confident-looking split (e.g. 54%/46%) out of ~0.1-nat deltas. A UI leading with that
# share would be lying. This note (paired with the top-level `any_meaningful:false` flag) says so plainly.
_NO_EFFECT_NOTE = ("No prompt section measurably changed this answer: every section's removal moved the "
                  f"stored answer's fit by less than {_FORCED_MEAN_THRESHOLD} nats/token, i.e. within noise "
                  "-- the answer likely came from the model's own knowledge or the query itself. The shares "
                  "below are not a meaningful ranking. " + _NOTE)

# `summary` bucket boundaries for |per_token_delta|. The negligible/small line reuses forced.py's own
# has-effect threshold (0.05 nats/token is already this codebase's one definition of "a real measured
# per-token effect" -- see clozn.receipts.forced._FORCED_MEAN_THRESHOLD) rather than invent a second,
# unrelated number; the small/substantial line is a judgment call with no existing precedent to match, set
# an order of magnitude higher so "substantial" means unambiguously large, not just "over the noise floor".
_NEGLIGIBLE = _FORCED_MEAN_THRESHOLD
_SUBSTANTIAL = 0.5


def _ablatable_sections(run: dict) -> list:
    """The run's own `sections` manifest, filtered to what this fast path can actually score -- "api"/
    "auto" sources only (see module docstring's dedup note); an entry with no `name` can't be ablated by
    name either (mirrors deltas._section_influences's own skip rule) so it's dropped here too. Never
    raises: a malformed manifest degrades to []."""
    manifest = run.get("sections")
    if not isinstance(manifest, list):
        return []
    return [s for s in manifest
            if isinstance(s, dict) and s.get("source") in ("api", "auto") and s.get("name")]


def _summary(per_token_delta: float) -> str:
    """A short, honest phrase bucketed by |per_token_delta| -- states an OBSERVED fit change, never a
    causal claim (mirrors _FORCED_CAVEAT's own "dependence, not counterfactual" framing, just shorter)."""
    mag = abs(per_token_delta)
    if mag < _NEGLIGIBLE:
        return "removing this section has a negligible effect on how well the stored answer fits"
    degree = "much" if mag >= _SUBSTANTIAL else "slightly"
    direction = "worse" if per_token_delta < 0 else "better"
    return f"removing this section makes the stored answer fit {degree} {direction}"


def _shares(deltas: list) -> list:
    """[log_prob_delta, ...] -> [influence_share, ...], same order: |delta_i| / max(sum_j |delta_j|,
    1e-9). Factored out as a pure function (no run, no substrate) specifically so the all-zero guard is
    unit-testable in isolation: when every delta is ~0, `denom` is still 1e-9 (never 0), so every share
    comes out exactly 0.0 -- never a 0/0 NaN. An empty list returns an empty list."""
    total = sum(abs(d) for d in deltas)
    denom = max(total, 1e-9)
    return [round(abs(d) / denom, 6) for d in deltas]


def any_meaningful(scored: list) -> bool:
    """True iff at least one scored section clears the per-token "real effect" floor (_NEGLIGIBLE). When
    False, every section's removal moved the stored answer's fit within noise -- the answer didn't
    measurably depend on ANY tagged part (it came from the model's own weights or the query). Exposed as a
    top-level flag (and shared with the drill route) so a UI never presents `influence_share` -- which
    always normalizes to sum=1 -- as a meaningful ranking of noise. Pure function of the scored rows, for
    unit-testability; an empty list is vacuously not-meaningful (False)."""
    return any(abs(s.get("per_token_delta") or 0.0) >= _NEGLIGIBLE for s in scored)


def _section_score(run: dict, sec: dict, sub) -> dict | None:
    """One section's {id, name, source, log_prob_delta, per_token_delta, summary} (influence_share is
    filled in by the caller once every section's delta is known). None when forced_receipt could not
    produce a verified delta for this section (no usable parts, engine hiccup, ...) -- an honest exclusion
    from the response, never a fabricated zero."""
    fr = forced_receipt(run, {"section": sec.get("name"), "source": sec.get("source")}, sub)
    if not fr or not fr.get("causal_verified"):
        return None
    n = len(fr.get("deltas") or []) or 1
    # forced.py's own convention is WITH-minus-WITHOUT (`sum_nats`); this route's contract is the opposite
    # sign, (ablated - baseline), so the answer's fit getting WORSE without the section reads as negative
    # -- the flip happens once, right here, at the seam between the two conventions.
    log_prob_delta = round(-float(fr.get("sum_nats") or 0.0), 6)
    per_token_delta = round(log_prob_delta / n, 6)
    return {"id": sec.get("id"), "name": sec.get("name"), "source": sec.get("source"),
            "log_prob_delta": log_prob_delta, "per_token_delta": per_token_delta,
            "summary": _summary(per_token_delta)}


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith("/section-influence")):
        return False
    rid = p[len("/runs/"):-len("/section-influence")]
    run = runlog.get_run(rid)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    sections = _ablatable_sections(run)
    if not sections:
        # No manifest at all (predates section capture), or a manifest with nothing this fast path can
        # ablate (memory_card-only) -- either way, an honest empty answer, not an engine-availability check
        # nobody needed: there is nothing to score regardless of substrate state.
        h._json(200, {"run_id": rid, "sections": [], "any_meaningful": False, "note": _NO_MANIFEST_NOTE})
        return True

    sub = ctx.active_sub(h)
    if not (sub and getattr(sub, "score_tokens", None)):
        h._json(503, {"error": "section-influence requires worker token scoring"})
        return True

    # The baseline (every section present) -- computed ONCE, shared by every section's delta below (each
    # forced_receipt() call redoes its own copy of this same call internally; see the module docstring's
    # cost note). Doubles as the "can we even score" probe: an engine-down or misconfigured worker fails
    # here, a single time, instead of once per section.
    conditions = rederive.with_arm_conditions(run)
    with_tokens, with_ok = rederive.score_arm(
        sub, conditions, messages=conditions["raw_messages"], block=conditions["raw_block"],
        steer_strengths=conditions["steer_strengths"])
    if not with_ok:
        h._json(503, {"error": "section-influence requires worker token scoring"})
        return True
    baseline_logprob = round(sum(float(t.get("logprob") or 0.0) for t in with_tokens
                                if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))), 6)

    scored = []
    for sec in sections:
        try:
            s = _section_score(run, sec, sub)
        except Exception:
            s = None
        if s is not None:
            scored.append(s)

    for s, share in zip(scored, _shares([s["log_prob_delta"] for s in scored])):
        s["influence_share"] = share
    scored.sort(key=lambda s: -s["influence_share"])  # biggest measured effect first (matches the spec example)
    # Reorder keys to match the contract's own field order (cosmetic only -- JSON key order carries no
    # meaning -- but a human reading a raw response should see it laid out the way it's documented).
    scored = [{"id": s["id"], "name": s["name"], "source": s["source"], "log_prob_delta": s["log_prob_delta"],
              "influence_share": s["influence_share"], "per_token_delta": s["per_token_delta"],
              "summary": s["summary"]} for s in scored]

    meaningful = any_meaningful(scored)
    h._json(200, {"run_id": rid, "method": "teacher_forced",
                 "note": _NOTE if meaningful else _NO_EFFECT_NOTE,
                 "any_meaningful": meaningful,
                 "baseline_logprob": baseline_logprob, "sections": scored})
    return True
