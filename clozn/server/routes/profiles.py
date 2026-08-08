"""HTTP surface for the retired named-persona-bundle ("Profiles") feature.

Clozn is a debugger and Model CI system, not a persona manager. The useful primitive was always
steering (`/steer/axes`, `/steer/set`, `/steer/check`), which is untouched by this retirement -- see
docs/CAPABILITIES.md. Every request that used to list/save/switch/export/import/delete a named
profile bundle now returns a typed HTTP 410 rather than silently doing nothing or 404ing.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_MESSAGE = (
    "named behavior profiles were retired -- configure steering directly through Clozn's tone-dial "
    "controls (/steer/axes, /steer/set) instead of switching between saved persona bundles."
)


def _retired(h) -> bool:
    h._json(410, {"error": _MESSAGE, "code": "profiles_retired"})
    return True


def try_get(h, p):
    if p.startswith("/profiles/"):
        return _retired(h)
    return False


def try_post(h, p, body):
    if p.startswith("/profiles/"):
        return _retired(h)
    return False
