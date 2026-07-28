"""Namespaced extension points for the run identity block, so features add facets without collisions.

THE PROBLEM THIS SOLVES
-----------------------
Roadmap rule 2 ("Immutable identity") wants every run to record the exact model, adapter, tokenizer,
template, engine, protocol, sampling configuration, context assembly, and application configuration.
Those facets are owned by different features -- the engine artifact by managed setup, the adapter by the
LoRA workflow, machine/runtime identity by performance diagnosis -- and if each one adds its fields by
editing runtime_identity() directly, every feature branch rewrites the same function body.

A provider is a file dropped in clozn/runs/identity_providers/ that names itself and returns a dict:

    # clozn/runs/identity_providers/engine_artifact.py
    NAME = "engine_artifact"

    def identity(context):
        health = (context or {}).get("engine_health") or {}
        out = {}
        if health.get("protocol_version") is not None:
            out["protocol_version"] = health["protocol_version"]
        return out            # {} or None -> this namespace is omitted entirely

Its result lands at `identity["ext"]["engine_artifact"]`. Nothing shared is edited, so N features can
add N facets on N branches and merge additively.

THE OMISSION RULE IS INHERITED, NOT OPTIONAL
--------------------------------------------
clozn/runs/identity.py's contract is that a key is OMITTED, never null-padded, when it cannot be
honestly measured -- absence must stay visible rather than be faked as null. That rule applies inside
`ext` too, and collect() enforces the outer half of it: a provider returning None, {}, or a non-dict
contributes no namespace at all, and `ext` itself is absent when no provider produced anything. A
provider that null-pads its OWN fields is violating the contract in a way this module cannot see, so
don't.

A PROVIDER MUST NEVER BE ABLE TO BREAK A RUN
--------------------------------------------
runtime_identity() is called on the path that records a real user's run. A provider that raises, hangs
on I/O, or returns garbage must cost that run its identity FACET, never the run itself -- so collect()
catches everything and records it in COLLECT_FAILURES for `clozn doctor` rather than propagating. Being
loud about a broken provider is the test suite's job (tests/test_identity_ext.py), not the failing run's.

Providers are called on the run-record path. Keep them cheap: read already-computed values out of
`context`, never hash a multi-GB file or make a network call.
"""
from __future__ import annotations

import importlib
import os

_PACKAGE = "clozn.runs.identity_providers"
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity_providers")

# (provider_module, exception) for every provider that failed to import or raised when called. Surfaced
# by `clozn doctor` and asserted empty by the test suite; never cleared.
COLLECT_FAILURES: list[tuple[str, BaseException]] = []

# Discovery result, cached per process: the filesystem walk and the imports happen once, not on every
# recorded run. None means "not yet scanned".
_PROVIDERS: list[tuple[str, object]] | None = None


def _discover() -> list[tuple[str, object]]:
    """(namespace, callable) for every provider module, sorted by namespace for deterministic output."""
    global _PROVIDERS
    if _PROVIDERS is not None:
        return _PROVIDERS

    found: list[tuple[str, object]] = []
    try:
        entries = sorted(os.listdir(_DIR))
    except OSError:
        _PROVIDERS = []
        return _PROVIDERS

    for entry in entries:
        if not entry.endswith(".py") or entry.startswith("_") or entry.startswith("test_"):
            continue
        dotted = f"{_PACKAGE}.{entry[:-3]}"
        try:
            module = importlib.import_module(dotted)
            namespace = getattr(module, "NAME", None)
            provider = getattr(module, "identity", None)
            if not isinstance(namespace, str) or not namespace:
                raise AttributeError("defines no NAME string")
            if not callable(provider):
                raise AttributeError("defines no identity(context) callable")
            found.append((namespace, provider))
        except BaseException as exc:            # noqa: BLE001 -- deliberately broad; see module docstring
            COLLECT_FAILURES.append((dotted, exc))

    found.sort(key=lambda pair: pair[0])
    _PROVIDERS = found
    return _PROVIDERS


def collect(context=None) -> dict:
    """Every provider's contribution, keyed by namespace. `{}` when nothing was contributed.

    `context` is passed through to each provider untouched; callers in clozn.runs.identity supply the
    same keyword arguments runtime_identity() received (model_path, engine_health, ...) so a provider
    can reuse an already-computed value instead of recomputing it. Never raises.
    """
    out: dict = {}
    for namespace, provider in _discover():
        try:
            contributed = provider(context)
        except BaseException as exc:            # noqa: BLE001 -- a broken provider must not fail the run
            COLLECT_FAILURES.append((namespace, exc))
            continue
        # Omission, not null-padding: an empty or non-dict result means this facet could not be
        # established, and an absent namespace says exactly that.
        if isinstance(contributed, dict) and contributed:
            out[namespace] = contributed
    return out


def reset_cache() -> None:
    """Drop the discovery cache. For tests that add a provider file after this module was imported."""
    global _PROVIDERS
    _PROVIDERS = None
