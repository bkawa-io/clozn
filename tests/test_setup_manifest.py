"""Pure tests for clozn/setup/manifest.py: parse_manifest() + select_artifact() + install_key().

Every case here is a plain dict in, plain dict/exception out -- no filesystem, no network, no subprocess.
Fixture hardware profiles are synthetic clozn/setup/platform_detect.py-shaped dicts, never a real probe.
"""
from __future__ import annotations

import pytest

from clozn.setup import manifest as m
from clozn.setup.errors import ManifestError, SelectionError

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
LLAMA_COMMIT = "88a39274ecf88ba11686acd357b59685b1cbf03d"


def _artifact(**kw):
    base = {
        "os": "linux", "arch": "x86_64", "backend": "cpu",
        "url": "https://example.invalid/clozn-engine-1.0.0-linux-x86_64-cpu.tar.gz",
        "sha256": SHA_A, "size_bytes": 1000, "entrypoint": "bin/clozn-server",
        "build_id": "release-1.0.0-test",
        "llama_cpp_commit": LLAMA_COMMIT,
        "feature_flags": {"lora": True, "sae": False},
    }
    base.update(kw)
    return base


def _manifest(artifacts, **kw):
    base = {
        "schema_version": "clozn.engine-manifest.v1",
        "clozn_version": "1.0.0",
        "protocol_version": "1.0",
        "artifacts": artifacts,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------------------- parse_manifest

def test_parse_manifest_accepts_a_valid_document():
    doc = _manifest([_artifact()])
    assert m.parse_manifest(doc) is doc


def test_parse_manifest_rejects_a_non_dict():
    with pytest.raises(ManifestError):
        m.parse_manifest("not a manifest")


def test_parse_manifest_rejects_schema_violations():
    doc = _manifest([_artifact(sha256="too-short")])
    with pytest.raises(ManifestError, match="invalid"):
        m.parse_manifest(doc)


def test_parse_manifest_rejects_an_unsupported_protocol_major():
    doc = _manifest([_artifact()], protocol_version="99.0")
    with pytest.raises(ManifestError, match="protocol_version"):
        m.parse_manifest(doc)


def test_parse_manifest_accepts_a_compatible_minor_bump():
    """A newer worker MINOR is fine per clozn.protocol's own contract -- this manifest schema reuses that
    exact check rather than re-deriving it."""
    doc = _manifest([_artifact()], protocol_version="1.7")
    assert m.parse_manifest(doc) is doc


# --------------------------------------------------------------------------------------- select_artifact

WINDOWS_CPU = {"os": "windows", "arch": "x86_64", "gpu_backend": None, "cuda_major": None}
WINDOWS_CUDA12 = {"os": "windows", "arch": "x86_64", "gpu_backend": "cuda", "cuda_major": 12}
LINUX_CPU = {"os": "linux", "arch": "x86_64", "gpu_backend": None, "cuda_major": None}
MACOS_METAL = {"os": "macos", "arch": "arm64", "gpu_backend": "metal", "cuda_major": None}


def test_select_artifact_exact_platform_match():
    doc = _manifest([_artifact(os="linux", arch="x86_64", backend="cpu")])
    chosen = m.select_artifact(doc, LINUX_CPU)
    assert chosen["os"] == "linux" and chosen["backend"] == "cpu"


def test_select_artifact_auto_prefers_detected_gpu_backend():
    doc = _manifest([
        _artifact(os="windows", arch="x86_64", backend="cpu", entrypoint="bin/clozn-server.exe"),
        _artifact(os="windows", arch="x86_64", backend="cuda", cuda_major=12, sha256=SHA_B,
                  entrypoint="bin/clozn-server.exe"),
    ])
    chosen = m.select_artifact(doc, WINDOWS_CUDA12)
    assert chosen["backend"] == "cuda" and chosen["cuda_major"] == 12


def test_select_artifact_auto_falls_back_to_cpu_when_no_gpu_artifact_exists():
    doc = _manifest([_artifact(os="windows", arch="x86_64", backend="cpu", entrypoint="bin/clozn-server.exe")])
    chosen = m.select_artifact(doc, WINDOWS_CUDA12)   # platform HAS a gpu, manifest doesn't offer one
    assert chosen["backend"] == "cpu"


def test_select_artifact_explicit_backend_overrides_auto_detection():
    doc = _manifest([
        _artifact(os="windows", arch="x86_64", backend="cpu", entrypoint="bin/clozn-server.exe"),
        _artifact(os="windows", arch="x86_64", backend="cuda", cuda_major=12, sha256=SHA_B,
                  entrypoint="bin/clozn-server.exe"),
    ])
    chosen = m.select_artifact(doc, WINDOWS_CUDA12, backend_pref="cpu")
    assert chosen["backend"] == "cpu"


def test_select_artifact_explicit_backend_not_offered_raises():
    doc = _manifest([_artifact(os="windows", arch="x86_64", backend="cpu", entrypoint="bin/clozn-server.exe")])
    with pytest.raises(SelectionError, match="no cuda artifact"):
        m.select_artifact(doc, WINDOWS_CPU, backend_pref="cuda")


def test_select_artifact_no_platform_match_raises():
    doc = _manifest([_artifact(os="linux", arch="x86_64", backend="cpu")])
    with pytest.raises(SelectionError):
        m.select_artifact(doc, MACOS_METAL)


def test_select_artifact_rejects_an_unknown_backend_pref():
    doc = _manifest([_artifact()])
    with pytest.raises(SelectionError, match="--backend"):
        m.select_artifact(doc, LINUX_CPU, backend_pref="vulkan")


def test_select_artifact_cuda_major_exact_match_wins():
    doc = _manifest([
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=11, sha256=SHA_A),
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=12, sha256=SHA_B),
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=13, sha256=SHA_C),
    ])
    platform = {"os": "linux", "arch": "x86_64", "gpu_backend": "cuda", "cuda_major": 12}
    chosen = m.select_artifact(doc, platform)
    assert chosen["cuda_major"] == 12


def test_select_artifact_cuda_major_falls_back_to_highest_not_newer():
    """Driver advertises CUDA 12; the manifest only offers 11 and 13 -- an artifact newer than the
    driver cannot be trusted to load, so 11 (the highest that does not exceed 12) wins."""
    doc = _manifest([
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=11, sha256=SHA_A),
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=13, sha256=SHA_C),
    ])
    platform = {"os": "linux", "arch": "x86_64", "gpu_backend": "cuda", "cuda_major": 12}
    chosen = m.select_artifact(doc, platform)
    assert chosen["cuda_major"] == 11


def test_select_artifact_cuda_major_with_no_detected_driver_picks_highest():
    doc = _manifest([
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=11, sha256=SHA_A),
        _artifact(os="linux", arch="x86_64", backend="cuda", cuda_major=12, sha256=SHA_B),
    ])
    platform = {"os": "linux", "arch": "x86_64", "gpu_backend": "cuda", "cuda_major": None}
    chosen = m.select_artifact(doc, platform, backend_pref="cuda")
    assert chosen["cuda_major"] == 12


def test_select_artifact_ambiguous_non_cuda_duplicates_raise():
    doc = _manifest([
        _artifact(os="linux", arch="x86_64", backend="cpu", sha256=SHA_A,
                  url="https://example.invalid/a.tar.gz"),
        _artifact(os="linux", arch="x86_64", backend="cpu", sha256=SHA_B,
                  url="https://example.invalid/b.tar.gz"),
    ])
    with pytest.raises(SelectionError, match="ambiguous"):
        m.select_artifact(doc, LINUX_CPU)


# --------------------------------------------------------------------------------------------- install_key

def test_install_key_cpu():
    assert m.install_key("1.0.0", _artifact(os="linux", arch="x86_64", backend="cpu")) == \
        "1.0.0/linux-x86_64-cpu"


def test_install_key_cuda_includes_major():
    assert m.install_key("1.0.0", _artifact(os="windows", arch="x86_64", backend="cuda", cuda_major=12)) == \
        "1.0.0/windows-x86_64-cuda12"
