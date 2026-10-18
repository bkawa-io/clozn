"""Explicit registration manifest for server route modules.

Only modules named in :data:`ROUTE_MODULES` enter the gateway dispatch tables. The module-local
``CLOZN_ROUTE_AUTOLOAD`` marker remains a second assertion: a mistaken manifest entry is diagnosed,
not silently exposed. Merely adding or restoring a source file cannot create a public endpoint.

Order is semantic. Manifested GET handlers are inserted before the generic ``GET /runs/<id>``
fallback, and module names are sorted for deterministic first-refusal behavior.
"""
from __future__ import annotations

from collections.abc import Iterable
import importlib
import sys

MARKER = "CLOZN_ROUTE_AUTOLOAD"
_PACKAGE = "clozn.server.routes"

# The v1 product has one context reader/effects family, one reduction family, the context receipt
# disclosure, and parent/reduced-child comparison. Restoring any other route file does not expose it.
ROUTE_MODULES = (
    "context_investigation_v1",
    "context_receipt",
    "engine",
    "minimal_context",
    "run_compare",
)

# (module_name, exception) for every manifested route module that could not be loaded.
LOAD_FAILURES: list[tuple[str, BaseException]] = []


def _candidates(module_names: Iterable[str] | None = None) -> list[str]:
    """Return a validated, deterministic route registration list."""
    names = list(ROUTE_MODULES if module_names is None else module_names)
    invalid = [name for name in names if not isinstance(name, str) or not name.isidentifier()]
    if invalid:
        raise ValueError(f"invalid route module names: {invalid!r}")
    if len(names) != len(set(names)):
        raise ValueError("duplicate route module names in registration manifest")
    return sorted(names)


def discover(module_names: Iterable[str] | None = None) -> list:
    """Import and validate every route module in the explicit manifest.

    A broken manifested module is recorded in :data:`LOAD_FAILURES` and announced on stderr, leaving
    the rest of the gateway available for diagnosis.
    """
    found = []
    for name in _candidates(module_names):
        dotted = f"{_PACKAGE}.{name}"
        try:
            module = importlib.import_module(dotted)
            if getattr(module, MARKER, False) is not True:
                raise AttributeError(f"is manifested but does not set {MARKER} = True")
            if not hasattr(module, "try_get") and not hasattr(module, "try_post"):
                raise AttributeError(
                    f"sets {MARKER} = True but defines neither "
                    "try_get(h, p) nor try_post(h, p, body)")
            found.append(module)
        except BaseException as exc:  # noqa: BLE001 -- diagnostics must survive broken extensions
            LOAD_FAILURES.append((dotted, exc))
            print(
                f"clozn: route module {dotted} failed to load: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return found


def with_try_get(modules) -> list:
    """Return manifested GET handlers for insertion before the generic run fallback."""
    return [module for module in modules if hasattr(module, "try_get")]


def with_try_post(modules) -> list:
    """Return manifested POST handlers."""
    return [module for module in modules if hasattr(module, "try_post")]


__all__ = [
    "LOAD_FAILURES",
    "MARKER",
    "ROUTE_MODULES",
    "discover",
    "with_try_get",
    "with_try_post",
]
