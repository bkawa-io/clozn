"""HTTP surface for the retired F5/F6 durable correction store ("Teach Once").

Durable, auto-applied corrections were retired: Clozn no longer lets a saved correction silently
shape a later, unrelated generation. See docs/CAPABILITIES.md. Every path that used to
create/draft/confirm/promote/enable/disable/resolve/export a persisted correction now returns a
typed HTTP 410 -- distinguishable from a plain 404 route miss -- rather than appearing to still work.

Use a one-shot corrective retry instead (``POST /runs/<id>/retry``, or the
``/runs/<id>/corrective-actions`` preview/confirm/keep flow): both compare a correction against one
specific run and leave no standing policy behind for a later request to pick up.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_MESSAGE = (
    "durable corrections were retired -- Clozn no longer persists a correction and auto-applies it "
    "to future requests. Use a one-shot corrective retry (POST /runs/<id>/retry, or the "
    "/runs/<id>/corrective-actions preview/confirm/keep flow) to compare a correction against a "
    "specific run instead."
)


def _retired(h) -> bool:
    h._json(410, {"error": _MESSAGE, "code": "durable_corrections_retired"})
    return True


def try_get(h, p):
    if p == "/corrections" or p.startswith("/corrections/"):
        return _retired(h)
    return False


def try_post(h, p, body):
    if p == "/corrections" or p.startswith("/corrections/"):
        return _retired(h)
    return False
