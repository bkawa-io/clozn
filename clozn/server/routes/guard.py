"""Generation-guard persisted default: GET/POST /guard/mode.

Reads/writes the server-wide `generation_guard` setting (clozn.server.generation_guard.GUARD_SETTING,
persisted through clozn.memory.mode's atomic settings store -- see clozn._io.atomic_write_json) that
generation_guard.parse_guard_spec falls back to on every /v1/chat/completions request whose body omits
`clozn_guard` entirely. Before this route existed, GUARD_SETTING was only ever READ by that fallback, never
WRITTEN by any HTTP route -- there was no way to toggle the guard and have it stick (see
studio/app/behavior.mjs's prior "declared skeleton" note for the disposition-guard section).

PRECEDENCE (see generation_guard.parse_guard_spec's own docstring -- restated here since it is exactly
what this route can and cannot change): a request's own explicit `clozn_guard` field on
POST /v1/chat/completions ALWAYS wins over whatever is persisted here, including an explicit falsy value
(meaning "opted out this one call" even when this persisted default is on). This route only ever changes
what happens on a request that says nothing about clozn_guard at all.

Shape mirrors /timetravel/mode and /sampling/mode (routes/timetravel.py, routes/engine.py): GET the live
persisted config; POST any subset of fields, get the resolved config + whether anything changed back.
Every accepted field is validated through generation_guard._normalize_guard_spec -- the SAME validator a
live per-request clozn_guard value goes through via set_persisted_guard_spec/get_persisted_guard_spec, so
this route invents no knobs generation_guard.py doesn't already support.
"""
from clozn.server import generation_guard


def try_get(h, p):
    if p == "/guard/mode":
        spec = generation_guard.get_persisted_guard_spec()
        h._json(200, {"enabled": spec is not None, "guard": spec})
        return True
    return False


# Every key this route accepts: "enabled" (its own on/off shorthand) + exactly the spec fields
# generation_guard._normalize_guard_spec reads. Kept explicit so an unknown key is REJECTED rather
# than silently dropped -- same stance as the OpenAI compat normalizer (routes/openai.py). The trap
# this closes: `clozn_guard` is the name of the PER-REQUEST field, so posting
# {"clozn_guard": {...}} here is the natural mistake, and it used to return 200 + changed:true
# while persisting nothing at all.
ACCEPTED_FIELDS = frozenset({
    "enabled", "concepts", "threshold", "counter_strength", "max_fires", "layer",
})


def try_post(h, p, body):
    if p != "/guard/mode":
        return False
    concepts = body.get("concepts")
    if isinstance(concepts, (list, tuple)) and len(concepts) > 64:
        # Every concept resolves to a steer direction at generation time; an unbounded list is a
        # self-DoS. 64 is far above any real use (the calibrated set is single digits).
        h._json(400, {"error": f"too many concepts ({len(concepts)}); the guard accepts at most 64"})
        return True
    unknown = sorted(k for k in body if k not in ACCEPTED_FIELDS)
    if unknown:
        h._json(400, {"error": {
            "message": f"unknown field(s) for /guard/mode: {', '.join(unknown)}. Accepted: "
                       f"{', '.join(sorted(ACCEPTED_FIELDS))}. (The per-request field on "
                       f"/v1/chat/completions is named 'clozn_guard' and wraps these same keys; "
                       f"here they go at the top level.)",
            "type": "invalid_request_error", "param": unknown[0],
        }})
        return True
    before = generation_guard.get_persisted_guard_spec()
    enabled = body.get("enabled")
    if enabled is False:
        # An explicit "turn it off" always wins over any other field in the same call -- mirrors
        # parse_guard_spec's own "an explicit falsy value means opted out" rule for the per-request field.
        raw: dict = {}
    else:
        raw = {k: v for k, v in body.items() if k != "enabled"}
        if enabled is True and not raw.get("concepts"):
            h._json(400, {"error": {
                "message": "enabling the guard requires at least one concept in 'concepts' (a list of "
                           "non-empty strings) -- an empty/omitted concepts list is what turns it off, "
                           "not what enables it with nothing to guard against.",
                "type": "invalid_request_error", "param": "concepts",
            }})
            return True
    try:
        spec = generation_guard.set_persisted_guard_spec(raw)
    except ValueError as exc:
        h._json(400, {"error": {"message": str(exc), "type": "invalid_request_error", "param": "clozn_guard"}})
        return True
    # "changed" is MEASURED, not asserted. It used to be a hardcoded True, so a no-op write reported
    # a change that never happened -- a success signal that carried no information.
    h._json(200, {"enabled": spec is not None, "guard": spec, "changed": spec != before})
    return True
