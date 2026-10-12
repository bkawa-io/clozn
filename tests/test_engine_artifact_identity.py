"""Engine executable and build identity from supervisor and worker facts."""
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
            "build_id": "release-1.0.0-cuda",
            "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
        })
    assert block["ext"]["engine_artifact"] == {
        "discovery_source": "managed", "backend": "cuda",
        "artifact_sha256": "a" * 64, "engine_version": "1.0.0",
        "build_id": "release-1.0.0-cuda",
        "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
        "protocol_version": "1.0",
    }


def test_runtime_identity_with_no_extra_context_omits_engine_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    block = identity.runtime_identity()
    assert "ext" not in block or "engine_artifact" not in block.get("ext", {})


# ----------------------------------------------------------------------------- _engine_discovery_context

def test_engine_discovery_context_combines_worker_and_supervisor_facts():
    assert substrates._engine_discovery_context(
        {
            "backend": "cuda",
            "engine_version": "1.2.3",
            "build_id": "release-1.2.3-cuda",
            "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
        },
        {"discovery_source": "managed", "artifact_sha256": "c" * 64},
    ) == {
        "discovery_source": "managed", "backend": "cuda",
        "artifact_sha256": "c" * 64, "engine_version": "1.2.3",
        "build_id": "release-1.2.3-cuda",
        "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
    }


def test_engine_discovery_context_omits_unavailable_facts():
    assert substrates._engine_discovery_context(
        {"backend": "cpu", "engine_version": ""},
        {"discovery_source": "repo_dev_build", "artifact_sha256": None},
    ) == {"discovery_source": "repo_dev_build", "backend": "cpu"}


def test_engine_discovery_context_empty_without_facts():
    assert substrates._engine_discovery_context() == {}
