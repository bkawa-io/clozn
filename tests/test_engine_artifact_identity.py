"""clozn.runs.identity_providers.engine_artifact + clozn.server.substrates._engine_discovery_context --
the roadmap feature 01 identity facet end to end: CLOZN_ENGINE_* env vars (set by
clozn.cli.runtime_process.spawn_runtime on the gateway subprocess) -> extra_context ->
identity["ext"]["engine_artifact"]. See tests/test_identity_ext.py for the generic extra_context
plumbing this builds on; this file is specific to the engine_artifact provider itself.
"""
from __future__ import annotations

from clozn.runs import identity
from clozn.runs.identity_providers import engine_artifact

import clozn.server.app as _app_bootstrap   # noqa: F401 -- resolves clozn.server.app <-> .substrates's
                                            # circular import before the next line (see test_engine_substrate.py)
from clozn.server import substrates


# --------------------------------------------------------------------------------- engine_artifact.identity

def test_identity_reports_only_the_fields_present_in_context():
    result = engine_artifact.identity({"discovery_source": "managed", "backend": "cuda"})
    assert result == {"discovery_source": "managed", "backend": "cuda"}


def test_identity_omits_everything_when_context_is_empty():
    assert engine_artifact.identity({}) == {}
    assert engine_artifact.identity(None) == {}


def test_identity_reads_protocol_version_out_of_engine_health():
    result = engine_artifact.identity({"engine_health": {"protocol_version": "1.0", "other": "x"}})
    assert result == {"protocol_version": "1.0"}


def test_identity_never_invents_a_field_not_present():
    result = engine_artifact.identity({
        "discovery_source": "repo_dev_build", "backend": "gpu",
        "artifact_sha256": None, "engine_version": None,
    })
    assert result == {"discovery_source": "repo_dev_build", "backend": "gpu"}
    assert "artifact_sha256" not in result and "engine_version" not in result


def test_identity_ignores_unrelated_context_keys():
    result = engine_artifact.identity({"model_path": "/x/model.gguf", "discovery_source": "managed"})
    assert result == {"discovery_source": "managed"}


# ------------------------------------------------------------------------------------- end-to-end via seam

def test_runtime_identity_with_full_extra_context_populates_engine_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    # engine_health is passed as runtime_identity()'s OWN kwarg, not via extra_context: a measured
    # argument always wins over a same-named extra_context key (see identity.py's docstring), so putting
    # it in extra_context here would be overwritten by the (unset) engine_health=None default -- this is
    # exactly the protection test_extra_context_cannot_overwrite_a_measured_value in
    # tests/test_identity_ext.py asserts, working as intended.
    block = identity.runtime_identity(
        engine_health={"protocol_version": "1.0"},
        extra_context={
            "discovery_source": "managed", "backend": "cuda",
            "artifact_sha256": "a" * 64, "engine_version": "1.0.0",
        })
    assert block["ext"]["engine_artifact"] == {
        "discovery_source": "managed", "backend": "cuda",
        "artifact_sha256": "a" * 64, "engine_version": "1.0.0",
        "protocol_version": "1.0",
    }


def test_runtime_identity_with_no_extra_context_omits_engine_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    block = identity.runtime_identity()
    assert "ext" not in block or "engine_artifact" not in block.get("ext", {})


# ----------------------------------------------------------------------------- _engine_discovery_context

def test_engine_discovery_context_reads_set_env_vars(monkeypatch):
    monkeypatch.setenv("CLOZN_ENGINE_DISCOVERY_SOURCE", "managed")
    monkeypatch.setenv("CLOZN_ENGINE_BACKEND", "cuda")
    monkeypatch.setenv("CLOZN_ENGINE_ARTIFACT_SHA256", "c" * 64)
    monkeypatch.setenv("CLOZN_ENGINE_VERSION", "1.2.3")
    assert substrates._engine_discovery_context() == {
        "discovery_source": "managed", "backend": "cuda",
        "artifact_sha256": "c" * 64, "engine_version": "1.2.3",
    }


def test_engine_discovery_context_omits_unset_vars(monkeypatch):
    for var in ("CLOZN_ENGINE_DISCOVERY_SOURCE", "CLOZN_ENGINE_BACKEND",
                "CLOZN_ENGINE_ARTIFACT_SHA256", "CLOZN_ENGINE_VERSION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLOZN_ENGINE_DISCOVERY_SOURCE", "repo_dev_build")
    assert substrates._engine_discovery_context() == {"discovery_source": "repo_dev_build"}


def test_engine_discovery_context_empty_when_nothing_set(monkeypatch):
    for var in ("CLOZN_ENGINE_DISCOVERY_SOURCE", "CLOZN_ENGINE_BACKEND",
                "CLOZN_ENGINE_ARTIFACT_SHA256", "CLOZN_ENGINE_VERSION"):
        monkeypatch.delenv(var, raising=False)
    assert substrates._engine_discovery_context() == {}
