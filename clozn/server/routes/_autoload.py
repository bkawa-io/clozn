"""Opt-in discovery for NEW route families, so adding one is a pure file addition.

Same problem and same shape as clozn/cli/commands/_autoload.py, one layer down: clozn/server/app.py
hand-wires ~25 route modules into `_GET_ROUTES` and `_POST_ROUTES`, so every feature that adds an
endpoint edits the same two lists and every parallel branch conflicts there.

A module that opts in is dispatched without app.py changing:

    # clozn/server/routes/experiments.py
    CLOZN_ROUTE_AUTOLOAD = True

    def try_get(h, p):
        if p == "/experiments":
            h._json(200, {"experiments": [...]})
            return True
        return False

`try_get(h, p)` and `try_post(h, p, body)` are both optional -- a module is added to whichever lists it
has a handler for. Returning truthy means "I handled this"; falsy means "not mine, keep looking", which
is exactly the existing contract every hand-wired module already follows.

ORDER IS SEMANTIC HERE, NOT COSMETIC
------------------------------------
`_GET_ROUTES` ends with the generic `GET /runs/<id>` fallback, and app.py's comment is explicit that
this is deliberate: every more-specific `/runs/<id>/<suffix>` family must get first refusal, because
they all share the prefix the fallback also matches. So autoloaded GET modules are spliced in BEFORE
that fallback, never appended after it. Appending would make a new `/runs/<id>/anything` route
unreachable -- and it would fail as a wrong-shaped 200 from the fallback rather than a 404, which is
the kind of bug that takes an afternoon to see.

Within the autoloaded group, order is by module name, so the dispatch order is deterministic and a
route family's behavior does not depend on filesystem iteration order.

FAILURE BEHAVIOR
----------------
A module that opts in and then fails to import is recorded in LOAD_FAILURES and reported on stderr --
never swallowed (roadmap rule 3). The server still starts, so a broken route family does not take down
the gateway that `clozn doctor` and every other route need. tests/test_route_autoload.py asserts
LOAD_FAILURES is empty, which is what makes a broken route a hard CI failure rather than a warning.
"""
from __future__ import annotations

import importlib
import os
import sys

MARKER = "CLOZN_ROUTE_AUTOLOAD"

_PACKAGE = "clozn.server.routes"
_DIR = os.path.dirname(os.path.abspath(__file__))

# (module_name, exception) for every route module that opted in but could not be loaded.
LOAD_FAILURES: list[tuple[str, BaseException]] = []


def _candidates() -> list[str]:
    """Module names in this package whose SOURCE mentions the marker, sorted for deterministic
    dispatch order. Private and test modules are skipped outright."""
    names = []
    try:
        entries = sorted(os.listdir(_DIR))
    except OSError:
        return []
    for entry in entries:
        if not entry.endswith(".py") or entry.startswith("_") or entry.startswith("test_"):
            continue
        try:
            with open(os.path.join(_DIR, entry), encoding="utf-8") as handle:
                if MARKER not in handle.read():
                    continue
        except OSError:
            continue
        names.append(entry[:-3])
    return names


def discover() -> list:
    """Every route module that sets `CLOZN_ROUTE_AUTOLOAD = True`, in deterministic order.

    Never raises: a broken module is recorded in LOAD_FAILURES and announced on stderr, leaving the
    rest of the gateway serving.
    """
    found = []
    for name in _candidates():
        dotted = f"{_PACKAGE}.{name}"
        try:
            module = importlib.import_module(dotted)
            if getattr(module, MARKER, False) is not True:
                continue
            if not hasattr(module, "try_get") and not hasattr(module, "try_post"):
                raise AttributeError(
                    f"sets {MARKER} = True but defines neither try_get(h, p) nor try_post(h, p, body)")
            found.append(module)
        except BaseException as exc:            # noqa: BLE001 -- deliberately broad; see module docstring
            LOAD_FAILURES.append((dotted, exc))
            print(f"clozn: route module {dotted} failed to load: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
    return found


def with_try_get(modules) -> list:
    """Those of `modules` that handle GET. Splice BEFORE the /runs/<id> fallback -- see the docstring."""
    return [m for m in modules if hasattr(m, "try_get")]


def with_try_post(modules) -> list:
    """Those of `modules` that handle POST."""
    return [m for m in modules if hasattr(m, "try_post")]
