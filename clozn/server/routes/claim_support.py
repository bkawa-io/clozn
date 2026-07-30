"""GET /runs/<id>/claim-support -- E3's on-demand view: E1's `clozn.answer-claims.v1` claim segmentation
(clozn/runs/claims.py) plus E2's `clozn.claim-support.v1` per-claim verification status
(clozn/runs/claim_support.py), derived fresh on every request.

Both artifacts are PURE, deterministic projections over a run already recorded -- E1 reads only
`run["response"]`; E2 additionally reads the run's already-persisted `run["influence_map"]`, never
starting a measurement of its own. Neither module touches the engine, calls a model, or mutates the run
(see both modules' own docstrings: "Never reads or writes anything but its... argument(s)"). A
synchronous GET is honest here for exactly that reason -- this route is a pure composition of the two,
nothing more, matching clozn/server/routes/diagnosis_findings.py's own "findings AND narrative, together
in one response" shape for a comparable pair of derived artifacts.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4), spliced in before the generic GET /runs/<id>
fallback -- see clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not
cosmetic, for every /runs/<id>/<suffix> family including this one.

Fixed at `privacy="metadata_only"` for both artifacts, matching every other derived-artifact GET route in
this codebase (context_receipt.py, span_addresses.py, diagnosis_findings.py -- none ever requests
`privacy="full"` on demand): a claim's `text_span` carries offsets and hashes, never the literal answer
substring, and `clozn.claim-support.v1`'s own `results` never embed source text at all regardless of
privacy (only `source_span_ids`, real `clozn.text-span-addresses.v1` address_id strings a caller resolves
separately via GET /runs/<id>/span-addresses). A caller that already has the recorded answer text (Studio
already shows it elsewhere on the same page) recovers each claim's substring locally via its own
`text_span.resolution.canonical.start`/`end` code-point offsets -- the same "no second addressing scheme"
discipline both source modules document.

Wire shape:
  GET /runs/<id>/claim-support
      -> 200 {"claims": <clozn.answer-claims.v1>, "support": <clozn.claim-support.v1>}
      -> 404 the run was not found
      -> 500 the derived pair could not be composed for this run (malformed/legacy run data) --
         a text-free, generic body; see _CONTRACT_ERROR below (same discipline as span_addresses.py's
         own contract-failure handler, for the same reason: exception text from a malformed legacy
         artifact may itself contain private source literals).
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/claim-support"

_CONTRACT_ERROR = {
    "error": "run claim support could not be composed",
    "code": "claim_support_contract_invalid",
}


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    from clozn import schemas
    from clozn.runs.claim_support import build_claim_support
    from clozn.runs.claims import build_answer_claims

    try:
        claims = build_answer_claims(run, privacy="metadata_only")
        support = build_claim_support(run, claims, privacy="metadata_only")
    except (TypeError, ValueError, UnicodeError, schemas.ValidationError):
        # Metadata-only route. Validation exceptions raised while reading malformed legacy run data may
        # contain source literals, so the public failure stays generic and text-free (mirrors
        # span_addresses.py's own contract-failure handler for the identical reason).
        h._json(500, _CONTRACT_ERROR)
        return True

    h._json(200, {"claims": claims, "support": support})
    return True
