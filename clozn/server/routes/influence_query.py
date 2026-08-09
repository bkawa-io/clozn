"""GET /runs/<id>/influence-query -- "Why this?": which MEASURED context spans affected a caller-
selected range of the recorded answer (E7, the debugger's read-side counterpart to E2's claim support).

A PURE PROJECTION, NEVER A NEW MEASUREMENT
--------------------------------------------
`clozn.runs.influence_query.build_influence_query` reads only the run already on disk and its already-
persisted `run["influence_map"]` (`clozn.context_answer_influence.v1`). It never calls
`context_answer_influence(...)`, never starts an influence-map job, never touches `select_control_model_
for_run`/`score_tokens`/an engine client/a worker, and never mutates the run. When no measurement is
stored, this route says so (`measurement.state == "not_measured"`) rather than computing one -- a caller
that wants a fresh measurement uses the existing `POST /runs/<id>/influence-map[/jobs]` explicitly (left
completely unchanged by this file). That is exactly why this is a GET: it creates nothing, mutates
nothing, starts no job, and runs no model -- a pure projection of already-recorded evidence, matching
`GET /runs/<id>/claim-support` and `GET /runs/<id>/span-addresses`'s own rationale.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4), spliced in before the generic GET /runs/<id>
fallback -- see clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not
cosmetic, for every /runs/<id>/<suffix> family including this one.

Fixed at `privacy="metadata_only"`, matching every other derived-artifact GET route in this codebase
(claim_support.py, span_addresses.py, diagnosis_findings.py): the response never embeds prompt or answer
text, only real `clozn.text-span-addresses.v1` `span_...` address IDs a caller already has a route to
resolve (`GET /runs/<id>/span-addresses`).

Wire shape:
  GET /runs/<id>/influence-query?start=<int>&end=<int>[&limit=<int>]
      -> 200 <clozn.influence-query.v1>
      -> 404 the run was not found
      -> 400 {"error": ..., "code": "invalid_output_range" | "invalid_limit"} -- malformed/missing
         start or end, a non-integer, a negative value, end <= start, end beyond the recorded answer's
         length, or a limit outside [1, 50]. Never echoes selected answer/source text.
      -> 500 {"error": ..., "code": "influence_query_contract_invalid"} -- a malformed legacy run/
         influence artifact could not be composed into a valid response. Text-free: an exception raised
         while reading malformed legacy data may itself carry private source literals (mirrors
         claim_support.py's and span_addresses.py's own contract-failure handlers).

`start`/`end` are Unicode code-point offsets, half-open `[start, end)`, into the exact recorded answer
string -- the same coordinate convention `clozn.text-span-addresses.v1` already established. This is not
a byte offset, a token offset, or a JavaScript UTF-16 offset.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/influence-query"

_CONTRACT_ERROR = {
    "error": "run influence query could not be composed",
    "code": "influence_query_contract_invalid",
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
    start = _parse_int((query.get("start") or [None])[0])
    end = _parse_int((query.get("end") or [None])[0])
    if start is None or end is None or start < 0 or end <= start:
        h._json(400, {
            "error": "start and end must be non-negative integers with end > start, into the "
                     "recorded answer's Unicode code points",
            "code": "invalid_output_range",
        })
        return True

    limit_raw = (query.get("limit") or [None])[0]
    if limit_raw is None:
        limit = 12
    else:
        limit = _parse_int(limit_raw)
        if limit is None or not (1 <= limit <= 50):
            h._json(400, {
                "error": "limit must be an integer from 1 to 50",
                "code": "invalid_limit",
            })
            return True

    from clozn import schemas
    from clozn.runs.influence_query import build_influence_query

    try:
        document = build_influence_query(run, output_start=start, output_end=end, limit=limit)
    except ValueError as exc:
        # build_influence_query raises ValueError only for structurally invalid arguments (bad
        # start/end/limit shape, or end beyond the recorded answer's known length) -- never for a
        # measurement-availability outcome, which is always a typed 200 body instead. The message never
        # contains selected answer/source text (see the function's own docstring).
        h._json(400, {"error": str(exc), "code": "invalid_output_range"})
        return True
    except (TypeError, UnicodeError, schemas.ValidationError):
        # Metadata-only route. An exception raised while reading a malformed legacy run or influence
        # artifact may itself contain private source literals, so the public failure stays generic and
        # text-free (mirrors claim_support.py's and span_addresses.py's own contract-failure handlers).
        h._json(500, _CONTRACT_ERROR)
        return True

    h._json(200, document)
    return True
