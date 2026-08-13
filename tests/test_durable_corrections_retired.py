"""Regression coverage for the "retire durable personalization / Teach Once" cut.

The invariant this file exists to pin down (docs/CAPABILITIES.md): nothing persisted by the old
personalization system may influence a new generation. One-shot corrective retries remain fully
functional; only the DURABLE, auto-applying machinery (session/profile-scoped corrective_retries,
the F5/F6 correction registry) was removed.
"""
from __future__ import annotations

import pytest



# ------------------------------------------------------------------ the persistent modules are gone

def test_persistent_corrective_retries_module_no_longer_exists():
    """The session/profile-scoped policy store (~/.clozn/corrective_retries.json,
    effective_presets/inject/activate/undo) is fully removed, not merely disconnected -- a
    reawakened import can never resurrect it."""
    with pytest.raises(ImportError):
        import clozn.behavior.corrective_retries  # noqa: F401


def test_durable_correction_registry_modules_no_longer_exist():
    """F5 (clozn.runs.corrections) and F6 (clozn.runs.teaching_loop) -- the durable "Teach Once"
    draft/confirm/promote/enable/disable lifecycle -- are fully removed."""
    with pytest.raises(ImportError):
        import clozn.runs.corrections  # noqa: F401
    with pytest.raises(ImportError):
        import clozn.runs.teaching_loop  # noqa: F401


def test_generation_gateway_has_no_persistent_injection_functions():
    """The three functions that used to look up and splice a saved policy into a live request
    (apply_corrective_policy, apply_scoped_corrections, reapply_scoped_resolution) are gone from the
    generation path, not merely unreachable."""
    from clozn.server import generation_gateway as gw
    for name in ("apply_corrective_policy", "apply_scoped_corrections", "reapply_scoped_resolution"):
        assert not hasattr(gw, name), f"generation_gateway still exposes {name}"


def test_generation_routes_no_longer_import_the_retired_correction_store():
    """openai.py/ollama.py must not still import clozn.runs.corrections or reference a saved
    correction resolution on the request handler."""
    import clozn.server.routes.openai as openai_route
    import clozn.server.routes.ollama as ollama_route
    for module in (openai_route, ollama_route):
        source = open(module.__file__, encoding="utf-8").read()
        assert "clozn.runs.corrections" not in source
        assert "_correction_resolution" not in source


def test_corrections_cli_command_no_longer_registered():
    """`clozn corrections ...` (F5/F6) is fully removed from the CLI, not just the HTTP surface."""
    import clozn.cli.commands._autoload as autoload
    assert "corrections" not in autoload._candidates()
