"""GET /runs/<id>/context-tension -- did different pieces of context exert opposing MEASURED effects on
the same part of this answer? (E8, built on the same shared `clozn.runs.influence_geometry` primitives
as `clozn.runs.influence_query`'s "Why this?", E7.)

CONTEXT TENSION, NOT CONFLICT OR CONTRADICTION
--------------------------------------------------
A tension record means exactly: two DISTINCT context spans each have a `causally_supported` link to the
SAME answer span, with opposite `effect` values. It is controlled intervention evidence that two spans
pulled one recorded answer span in opposite measured directions -- never a claim that the underlying
source TEXTS semantically contradict each other (that needs textual/semantic comparison, which this
route never performs; see `clozn.runs.context_tension`'s own module docstring for the full boundary
against `clozn.runs.claim_support`'s separate, deliberately un-reused contradiction heuristics).

A PURE PROJECTION, NEVER A NEW MEASUREMENT
--------------------------------------------
`clozn.runs.context_tension.build_context_tension` reads only the run already on disk and its already-
persisted `run["influence_map"]` (`clozn.context_answer_influence.v1`). It never calls
`context_answer_influence(...)`, never starts an influence-map job, never touches `select_control_model_
for_run`/`score_tokens`/an engine client/a worker, and never mutates the run. When no measurement is
stored, this route says so (`measurement.state == "not_measured"`) rather than computing one.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4), spliced in before the generic GET /runs/<id>
fallback -- see clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not
cosmetic, for every /runs/<id>/<suffix> family including this one.

Fixed at `privacy="metadata_only"`, matching `influence_query.py`, `claim_support.py`, and
`span_addresses.py`: the response never embeds prompt or answer text, only real
`clozn.text-span-addresses.v1` `span_...` address IDs a caller already has a route to resolve.

Wire shape:
  GET /runs/<id>/context-tension[?limit=<int>]
      -> 200 <clozn.context-tension.v1> -- whole-answer mode: every resolvable measured answer span is
         examined.
  GET /runs/<id>/context-tension?start=<int>&end=<int>[&limit=<int>]
      -> 200 <clozn.context-tension.v1> -- ranged mode: only answer spans overlapping [start, end).
      -> 404 the run was not found
      -> 400 {"error": ..., "code": "incomplete_output_range"} -- exactly one of start/end supplied.
      -> 400 {"error": ..., "code": "invalid_output_range"} -- malformed/negative/inverted start or
         end, or end beyond the recorded answer's length.
      -> 400 {"error": ..., "code": "invalid_limit"} -- limit outside [1, 100].
      -> 500 {"error": ..., "code": "context_tension_contract_invalid"} -- a malformed legacy run/
         influence artifact could not be composed into a valid response. Text-free (mirrors
         influence_query.py's, claim_support.py's, and span_addresses.py's own contract-failure
         handlers): an exception raised while reading malformed legacy data may itself carry private
         source literals.

`start`/`end` are Unicode code-point offsets, half-open `[start, end)`, into the exact recorded answer
string -- the identical coordinate convention `GET /runs/<id>/influence-query` and
`clozn.text-span-addresses.v1` already established.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/context-tension"

_CONTRACT_ERROR = {
    "error": "run context tension could not be composed",
    "code": "context_tension_contract_invalid",
}


def _parse_int(raw: str | None) -> int | None:
    """A strict base-10 integer parse -- `None` for missing/empty/non-integer text (including a float-
    looking string like "3.0"). Query values arrive as strings; this never silently truncates or
    coerces a malformed value into a number that merely happens to work."""
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(h.path).query)
    start_raw = (query.get("start") or [None])[0]
    end_raw = (query.get("end") or [None])[0]
    if (start_raw is None) != (end_raw is None):
        h._json(400, {
            "error": "start and end must both be supplied together, or both omitted for whole-answer "
                     "mode",
            "code": "incomplete_output_range",
        })
        return True

    output_start = output_end = None
    if start_raw is not None:
        output_start = _parse_int(start_raw)
        output_end = _parse_int(end_raw)
        if output_start is None or output_end is None or output_start < 0 or output_end <= output_start:
            h._json(400, {
                "error": "start and end must be non-negative integers with end > start, into the "
                         "recorded answer's Unicode code points",
                "code": "invalid_output_range",
            })
            return True

    limit_raw = (query.get("limit") or [None])[0]
    if limit_raw is None:
        limit = 25
    else:
        limit = _parse_int(limit_raw)
        if limit is None or not (1 <= limit <= 100):
            h._json(400, {
                "error": "limit must be an integer from 1 to 100",
                "code": "invalid_limit",
            })
            return True

    from clozn import schemas
    from clozn.runs.context_tension import build_context_tension

    try:
        document = build_context_tension(
            run, output_start=output_start, output_end=output_end, limit=limit,
        )
    except ValueError as exc:
        # build_context_tension raises ValueError only for structurally invalid arguments -- by this
        # point the route has already validated the range/limit shape, so the only realistic case left
        # is "end beyond the recorded answer's known length". Never for a measurement-availability
        # outcome, which is always a typed 200 body instead. The message never contains selected
        # answer/source text (see the function's own docstring).
        h._json(400, {"error": str(exc), "code": "invalid_output_range"})
        return True
    except (TypeError, UnicodeError, schemas.ValidationError):
        # Metadata-only route. An exception raised while reading a malformed legacy run or influence
        # artifact may itself contain private source literals, so the public failure stays generic and
        # text-free (mirrors influence_query.py's, claim_support.py's, and span_addresses.py's own
        # contract-failure handlers).
        h._json(500, _CONTRACT_ERROR)
        return True

    h._json(200, document)
    return True
