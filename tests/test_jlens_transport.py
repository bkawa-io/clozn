"""test_jlens_transport.py -- clozn/analysis/jlens_transport.py (J-transport an
ALREADY-BUILT steer direction: notes/JLENS_SAE_FINDINGS.md finding #1, "J-transported SAE steering
is 1.5-2x more stable than raw").

Model-free and GPU-free throughout (mirrors test_concept_dir.py's guardrails): every J artifact
exercised here is a tiny SYNTHETIC fixture this suite writes itself (never the real
~/.clozn/jlens or ~/.clozn/artifacts/jlens/); no engine, no GPU, no network.

Coverage (per the task's minimum bar):
  * the compact math (compact_transport / apply_compact_jlens) is CORRECT against a known dense J
    (a symmetric construction J = Q diag(sv) Q.T makes "compact transport reconstructs J @ dir
    exactly at full rank" a mathematical guarantee, not a probabilistic echo -- same style as
    test_concept_dir.py's orthogonal-J self-consistency check).
  * transport CHANGES the direction (cosine != 1, and provably concentrates energy into the
    live/top-k subspace when the input has energy outside it).
  * the no-J path is an EXACT no-op (element-wise-equal vector back, applied=False).
  * a WRONG-MODEL artifact is refused, not silently used.
  * the applied/not-applied flag is correct in every branch (available, no artifact, wrong model,
    dim mismatch, corrupt/missing sidecar file).
  * a dimension mismatch is refused (JTransportError from the low-level math; "dimension_mismatch"/
    "dim_mismatch" from the product-facing entry points) rather than silently broadcast/truncated.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.analysis.jlens_transport as jt  # noqa: E402
from clozn.artifacts import contracts  # noqa: E402


# ==================================================================================== fixtures

def _orthonormal(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q.astype(np.float32)


def _symmetric_J(d_model: int, sv, seed: int = 0):
    """J = Q diag(sv) Q.T -- symmetric (J == J.T), so compact_transport's V S V^T reconstruction
    equals the TRUE J^T @ dir exactly at full rank, regardless of any per-row sign ambiguity in
    the recovered singular vectors (v v^T is sign-invariant). Returns (J, Q)."""
    q = _orthonormal(seed, d_model)
    sv = np.asarray(sv, dtype=np.float64)
    J = (q * sv) @ q.T
    return J.astype(np.float32), q


def _write_jlens_fixture(tmp_path, *, d_model=16, layers=(21,), model="fixture-model", seed=0,
                         subdir="jlens"):
    """A tiny synthetic dense J-lens sidecar: manifest.json + J_layer{L}.f16, the same file
    layout concept_dir.py's load_jlens_jacobians / the real ~/.clozn/jlens directory use."""
    jdir = tmp_path / subdir
    jdir.mkdir(exist_ok=True)
    sv = np.linspace(10.0, 1.0, d_model)
    Js = {}
    for i, layer in enumerate(layers):
        J, _ = _symmetric_J(d_model, sv, seed=seed + i)
        Js[layer] = J
        J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    manifest = {"model": model, "d_model": d_model, "vocab": d_model, "layers": list(layers),
                "engine_default_tap_layer": layers[0]}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(jdir), Js


# ==================================================================================== compact_transport (the math)

def test_compact_transport_matches_dense_J_at_full_rank():
    """k == d_model: no truncation, so Vh.T @ (S * (Vh @ dir)) must equal J @ dir (== J.T @ dir,
    J symmetric by construction) to floating-point precision -- a mathematical guarantee, not an
    approximation."""
    d_model = 16
    sv = np.linspace(10.0, 1.0, d_model)
    J, _ = _symmetric_J(d_model, sv, seed=1)
    Vh, S = jt.fit_compact_from_dense(J, k=d_model)
    rng = np.random.default_rng(2)
    direction = rng.standard_normal(d_model).astype(np.float32)
    got = jt.compact_transport(direction, Vh, S)
    expected = J @ direction
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_compact_transport_truncated_k_still_exact_for_a_direction_inside_the_kept_subspace():
    """A direction that lives ENTIRELY inside the top-k kept subspace loses nothing to truncation
    -- proof that the compact form concentrates rather than merely approximates."""
    d_model = 16
    k = 4
    sv = np.linspace(10.0, 1.0, d_model)
    J, Q = _symmetric_J(d_model, sv, seed=3)
    Vh, S = jt.fit_compact_from_dense(J, k=k)
    # Build a direction as a combination of the FIRST k singular vectors (the ones kept) --
    # np.linalg.svd's Vh rows may carry a sign flip vs Q's columns, so use Vh itself here, not Q.
    coeffs = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    direction = (coeffs @ Vh).astype(np.float32)
    got = jt.compact_transport(direction, Vh, S)
    expected = J @ direction
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_compact_transport_truncated_k_drops_energy_outside_the_kept_subspace():
    """The complementary case: a direction OUTSIDE the top-k subspace gets attenuated/rotated by
    truncation -- this is the whole point (null-space energy is dropped), so the compact-k result
    must differ from the full-rank result for such a direction."""
    d_model = 16
    sv = np.linspace(10.0, 1.0, d_model)
    J, Q = _symmetric_J(d_model, sv, seed=4)
    Vh_full, S_full = jt.fit_compact_from_dense(J, k=d_model)
    Vh_k, S_k = jt.fit_compact_from_dense(J, k=4)
    # a direction built from a LOW singular vector (index 12, well outside top-4) -- the low
    # channel's own contribution should show up in full-rank but be dropped by the top-4 compact.
    direction = Vh_full[12].copy()
    full = jt.compact_transport(direction, Vh_full, S_full)
    compact = jt.compact_transport(direction, Vh_k, S_k)
    assert not np.allclose(full, compact, atol=1e-2)


def test_compact_transport_dimension_mismatch_raises_not_broadcasts():
    Vh, S = _orthonormal(5, 8)[:4], np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    with pytest.raises(jt.JTransportError, match="d_model"):
        jt.compact_transport(np.zeros(5, dtype=np.float32), Vh, S)


def test_compact_transport_S_Vh_row_mismatch_raises():
    Vh = _orthonormal(6, 8)[:4]
    S = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # only 3 entries for Vh's 4 rows
    with pytest.raises(jt.JTransportError, match="k=4"):
        jt.compact_transport(np.zeros(8, dtype=np.float32), Vh, S)


# ==================================================================================== apply_compact_jlens (norm modes)

def _small_compact(d_model=8, k=4, seed=7):
    sv = np.linspace(5.0, 1.0, d_model)
    J, _ = _symmetric_J(d_model, sv, seed=seed)
    Vh, S = jt.fit_compact_from_dense(J, k=k)
    return jt.CompactJLens(Vh=Vh, S=S, layer=21, d_model=d_model, model="fixture")


def test_apply_compact_jlens_unit_norm_returns_unit_vector():
    compact = _small_compact()
    rng = np.random.default_rng(9)
    direction = rng.standard_normal(8).astype(np.float32) * 5.0
    out = jt.apply_compact_jlens(direction, compact, norm="unit")
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-4)


def test_apply_compact_jlens_preserve_norm_keeps_original_magnitude():
    compact = _small_compact()
    rng = np.random.default_rng(10)
    direction = rng.standard_normal(8).astype(np.float32) * 3.7
    original_norm = float(np.linalg.norm(direction))
    out = jt.apply_compact_jlens(direction, compact, norm="preserve")
    assert np.linalg.norm(out) == pytest.approx(original_norm, rel=1e-4)


def test_apply_compact_jlens_changes_the_direction():
    """The core product claim: J-transport is NOT a no-op on a generic direction -- cosine to the
    original must be meaningfully less than 1 (mirrors the measured 0.05-0.07 cosine in
    notes/JLENS_SAE_FINDINGS.md, though this synthetic fixture won't reproduce that exact number)."""
    compact = _small_compact()
    rng = np.random.default_rng(11)
    direction = rng.standard_normal(8).astype(np.float32)
    out = jt.apply_compact_jlens(direction, compact, norm="unit")
    cos = float(np.dot(out, direction) / (np.linalg.norm(out) * np.linalg.norm(direction)))
    assert cos < 0.999   # genuinely rotated, not a rounding tweak
    assert not np.allclose(out, direction / np.linalg.norm(direction))


def test_apply_compact_jlens_degenerate_direction_raises():
    """A direction entirely in the null space of this compact J (orthogonal to every kept Vh row)
    transports to a zero vector -- refused, never silently returned as a zero/garbage vector."""
    d_model = 8
    Vh = np.zeros((1, d_model), dtype=np.float32)
    Vh[0, 0] = 1.0                       # the only kept direction is e_0
    S = np.array([5.0], dtype=np.float32)
    compact = jt.CompactJLens(Vh=Vh, S=S, layer=21, d_model=d_model, model="fixture")
    direction = np.zeros(d_model, dtype=np.float32)
    direction[1] = 1.0                   # e_1 -- orthogonal to the kept subspace
    with pytest.raises(jt.JTransportError, match="degenerate"):
        jt.apply_compact_jlens(direction, compact, norm="unit")


def test_apply_compact_jlens_unknown_norm_mode_raises():
    compact = _small_compact()
    with pytest.raises(jt.JTransportError, match="norm mode"):
        jt.apply_compact_jlens(np.ones(8, dtype=np.float32), compact, norm="bogus")


# ==================================================================================== CompactJLens validation

def test_compact_jlens_rejects_mismatched_S_length():
    with pytest.raises(jt.JTransportError):
        jt.CompactJLens(Vh=np.zeros((4, 8), dtype=np.float32), S=np.zeros(3, dtype=np.float32),
                        layer=1, d_model=8)


def test_compact_jlens_rejects_d_model_not_matching_Vh_columns():
    with pytest.raises(jt.JTransportError):
        jt.CompactJLens(Vh=np.zeros((4, 8), dtype=np.float32), S=np.zeros(4, dtype=np.float32),
                        layer=1, d_model=16)


# ==================================================================================== fit_compact_from_dense

def test_fit_compact_from_dense_rejects_non_square():
    with pytest.raises(jt.JTransportError, match="square"):
        jt.fit_compact_from_dense(np.zeros((4, 8), dtype=np.float32), k=2)


def test_fit_compact_from_dense_singular_values_are_descending_and_clamped_to_k():
    d_model = 10
    sv = np.linspace(9.0, 0.5, d_model)
    J, _ = _symmetric_J(d_model, sv, seed=5)
    Vh, S = jt.fit_compact_from_dense(J, k=1000)     # k larger than d_model -> clamp, not error
    assert Vh.shape == (d_model, d_model)
    assert S.shape == (d_model,)
    assert list(S) == sorted(S, reverse=True)


# ==================================================================================== resolve_compact_jlens / transport_direction: discovery + honesty

def test_transport_direction_is_an_exact_noop_when_no_jlens_artifact_exists(tmp_path):
    missing_dir = str(tmp_path / "nothing_here")
    direction = [0.1, -0.2, 0.3, 0.4]
    result = jt.transport_direction(direction, jlens_dir=missing_dir, layer=21)
    assert result["applied"] is False
    assert result["reason"] == "no_jlens_artifact"
    assert result["vector"] == direction          # EXACT no-op: same values back, not "close"


def test_transport_direction_applies_when_a_matching_artifact_exists(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,), model="qwen2.5-7b-v1")
    rng = np.random.default_rng(20)
    direction = rng.standard_normal(16).astype(np.float32).tolist()
    result = jt.transport_direction(direction, jlens_dir=jdir, layer=21, model_id="qwen2.5-7b-v1")
    assert result["applied"] is True
    assert result["layer"] == 21
    assert result["model"] == "qwen2.5-7b-v1"
    assert len(result["vector"]) == 16
    assert result["vector"] != direction


def test_transport_direction_preserve_norm_keeps_the_callers_calibrated_magnitude(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,), model="fixture-model")
    rng = np.random.default_rng(21)
    direction = (rng.standard_normal(16) * 4.2).astype(np.float32)
    original_norm = float(np.linalg.norm(direction))
    result = jt.transport_direction(direction, jlens_dir=jdir, layer=21, norm="preserve")
    assert result["applied"] is True
    assert np.linalg.norm(result["vector"]) == pytest.approx(original_norm, rel=1e-3)


def test_transport_direction_refuses_a_wrong_model_artifact(tmp_path):
    """The honesty-critical case: a jlens sidecar that claims a DIFFERENT model than the one
    active must be REFUSED, never silently used as if it were a match."""
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,), model="qwen2.5-7b-v1")
    direction = [0.5] * 16
    result = jt.transport_direction(direction, jlens_dir=jdir, layer=21, model_id="llama-3.2-1b")
    assert result["applied"] is False
    assert result["reason"] == "wrong_model"
    assert result["vector"] == direction


def test_transport_direction_refuses_dim_mismatch_against_the_manifest(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,))
    direction = [0.0] * 16
    result = jt.transport_direction(direction, jlens_dir=jdir, layer=21, expected_d_model=4096)
    assert result["applied"] is False
    assert result["reason"] == "dim_mismatch"
    assert result["vector"] == direction


def test_transport_direction_refuses_a_direction_whose_length_does_not_match_the_sidecar(tmp_path):
    """Even with no expected_d_model check requested, a direction that simply has the WRONG
    number of dimensions for this J must be refused by the underlying math, not silently
    truncated/padded to fit."""
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,))
    direction = [0.1, 0.2, 0.3, 0.4]           # only 4 dims, this J is fitted for 16
    result = jt.transport_direction(direction, jlens_dir=jdir, layer=21)
    assert result["applied"] is False
    assert result["reason"] == "dimension_mismatch"
    assert result["vector"] == direction


def test_transport_direction_refuses_an_unfitted_layer(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,))
    direction = [0.1] * 16
    result = jt.transport_direction(direction, jlens_dir=jdir, layer=99)
    assert result["applied"] is False
    assert result["reason"] == "layer_not_fitted"
    assert result["vector"] == direction


def test_transport_direction_defaults_to_the_manifests_default_tap_layer_when_layer_omitted(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(2, 14), model="m")
    direction = [0.2] * 8
    result = jt.transport_direction(direction, jlens_dir=jdir)   # no layer= given
    assert result["applied"] is True
    assert result["layer"] == 2       # engine_default_tap_layer in the fixture manifest


def test_transport_direction_corrupt_manifest_is_refused_not_raised(tmp_path):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    (jdir / "manifest.json").write_text("{not json", encoding="utf-8")
    result = jt.transport_direction([0.1, 0.2], jlens_dir=str(jdir))
    assert result["applied"] is False
    assert result["reason"] == "corrupt_manifest"


def test_transport_direction_missing_sidecar_file_is_refused_not_raised(tmp_path):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "m", "d_model": 8, "vocab": 8, "layers": [21], "engine_default_tap_layer": 21}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # no J_layer21.f16 written at all
    result = jt.transport_direction([0.1] * 8, jlens_dir=str(jdir), layer=21)
    assert result["applied"] is False
    assert result["reason"] == "missing_sidecar_file"


def test_resolve_compact_jlens_caches_across_calls(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=16, layers=(21,))
    cache: dict = {}
    first = jt.resolve_compact_jlens(jlens_dir=jdir, layer=21, cache=cache)
    second = jt.resolve_compact_jlens(jlens_dir=jdir, layer=21, cache=cache)
    assert first["available"] is True and first["cached"] is False
    assert second["available"] is True and second["cached"] is True
    assert first["compact"] is second["compact"]     # same object -- no J_layer21.f16 re-read/re-fit


def test_resolve_compact_jlens_honors_clozn_jlens_dir_env_var_default(monkeypatch, tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,), model="env-fixture")
    monkeypatch.setenv("CLOZN_JLENS_DIR", jdir)
    result = jt.resolve_compact_jlens(layer=21)     # no jlens_dir= at all -> env var default
    assert result["available"] is True
    assert result["model"] == "env-fixture"


# ==================================================================================== resolve_compact_jlens: model_identity tier
# (reuses clozn.artifacts.contracts wholesale -- full GGUF identity, every artifact FILE checksum
# verified, not just the manifest's claims)

def _fake_gguf_header(path, *, d_model, n_layers, vocab):
    return {
        "name": "Fixture Model", "arch": "fixture", "embedding_length": d_model,
        "n_layers": n_layers, "quant": "Q4_K_M", "file_size_bytes": path.stat().st_size,
        "metadata": {"tokenizer.ggml.model": "fixture-bpe", "tokenizer.ggml.tokens": ["a"] * vocab,
                    "tokenizer.chat_template": "{{ messages }}"},
    }


def _fake_identity(tmp_path, monkeypatch, *, d_model=8, n_layers=4, vocab=8, name="model.gguf"):
    """Mirrors test_artifact_contracts.py's `model` fixture: a fake GGUF file + a monkeypatched
    header reader, so contracts.gguf_identity() returns a REAL sha256 of REAL bytes with no actual
    GGUF parsing."""
    path = tmp_path / name
    path.write_bytes(b"exact gguf bytes for " + name.encode())
    monkeypatch.setattr(contracts, "gguf_header_from_path",
                        lambda _p: _fake_gguf_header(path, d_model=d_model, n_layers=n_layers, vocab=vocab))
    return contracts.gguf_identity(path)


def _write_contract_jlens_artifact(directory, identity, *, d_model=8, layer=21,
                                   compatible_sha256=None, corrupt_checksum=False):
    """A CONTRACT-style jlens artifact directory (the ~/.clozn/artifacts/jlens/<model>/ shape):
    manifest.json with the nested model.compatible_gguf_sha256 contracts.py validates, PLUS the
    top-level d_model/layers/engine_default_tap_layer fields concept_dir.py's loader reads, PLUS a
    REAL dense J_layer{L}.f16 whose checksum the manifest declares."""
    d = directory
    d.mkdir(parents=True, exist_ok=True)
    sv = np.linspace(5.0, 1.0, d_model)
    J, _ = _symmetric_J(d_model, sv, seed=42)
    payload_path = d / f"J_layer{layer}.f16"
    J.astype("<f2").tofile(str(payload_path))
    payload_bytes = payload_path.read_bytes()
    manifest = {
        "contract_version": 1, "artifact_type": "jlens", "artifact_version": 1,
        "model": {
            "source_id": "owner/fixture-model", "architecture": identity["architecture"],
            "hidden_size": identity["hidden_size"], "layer_count": identity["layer_count"],
            "vocab_size": identity["vocab_size"], "tokenizer_sha256": identity["tokenizer_sha256"],
            "compatible_gguf_sha256": (compatible_sha256 if compatible_sha256 is not None
                                       else [identity["sha256"]]),
        },
        "d_model": d_model, "vocab": identity["vocab_size"], "layers": [layer],
        "engine_default_tap_layer": layer,
        "files": {
            f"J_layer{layer}.f16": {
                "bytes": len(payload_bytes),
                "sha256": ("0" * 64 if corrupt_checksum else hashlib.sha256(payload_bytes).hexdigest()),
            }
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(d)


def test_resolve_compact_jlens_model_identity_tier_succeeds_via_explicit_dir(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    jdir = _write_contract_jlens_artifact(tmp_path / "artifact", identity, d_model=8, layer=21)
    result = jt.resolve_compact_jlens(jlens_dir=jdir, layer=21, model_identity=identity)
    assert result["available"] is True
    assert result["model"] == "owner/fixture-model"


def test_resolve_compact_jlens_model_identity_tier_auto_discovers_under_artifact_root(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "artifacts"
    _write_contract_jlens_artifact(root / "jlens" / "fixture-model", identity, d_model=8, layer=21)
    result = jt.resolve_compact_jlens(layer=21, model_identity=identity, artifact_root=str(root))
    assert result["available"] is True


def test_resolve_compact_jlens_model_identity_tier_refuses_wrong_gguf_sha256(tmp_path, monkeypatch):
    """The rigorous-tier honesty case: an artifact whose manifest claims a DIFFERENT GGUF sha256
    than the one actually loaded is refused via contracts.py's own validation -- not silently
    used just because someone pointed jlens_dir= at it."""
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    jdir = _write_contract_jlens_artifact(tmp_path / "artifact", identity, d_model=8, layer=21,
                                          compatible_sha256=["0" * 64])   # NOT this GGUF's sha256
    result = jt.resolve_compact_jlens(jlens_dir=jdir, layer=21, model_identity=identity)
    assert result["available"] is False
    assert result["reason"] == "artifact_contract_error"


def test_resolve_compact_jlens_model_identity_tier_refuses_tampered_payload_file(tmp_path, monkeypatch):
    """contracts.py's manifest validation checksums every declared FILE, not just the model
    fields -- a J_layer payload that doesn't match its declared sha256 is refused."""
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    jdir = _write_contract_jlens_artifact(tmp_path / "artifact", identity, d_model=8, layer=21,
                                          corrupt_checksum=True)
    result = jt.resolve_compact_jlens(jlens_dir=jdir, layer=21, model_identity=identity)
    assert result["available"] is False
    assert result["reason"] == "artifact_contract_error"


def test_resolve_compact_jlens_model_identity_tier_no_artifact_at_all_is_a_noop(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "empty_artifacts"
    result = jt.resolve_compact_jlens(layer=21, model_identity=identity, artifact_root=str(root))
    assert result["available"] is False
    assert result["reason"] == "no_jlens_artifact"


# ==================================================================================== resolve_compact_jlens: model_sha256 tier
# (lighter than model_identity -- just the GGUF digest, e.g. all EngineSubstrate ever has -- but
# still a real cryptographic match against the manifest's compatible_gguf_sha256, plus a checksum
# of the one sidecar file this call actually reads)

def test_resolve_compact_jlens_model_sha256_tier_matches_and_verifies_payload(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "artifacts"
    _write_contract_jlens_artifact(root / "jlens" / "fixture-model", identity, d_model=8, layer=21)
    result = jt.resolve_compact_jlens(layer=21, model_sha256=identity["sha256"], artifact_root=str(root))
    assert result["available"] is True
    assert result["model"] == "owner/fixture-model"


def test_resolve_compact_jlens_model_sha256_tier_refuses_a_non_matching_digest(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "artifacts"
    _write_contract_jlens_artifact(root / "jlens" / "fixture-model", identity, d_model=8, layer=21)
    result = jt.resolve_compact_jlens(layer=21, model_sha256="f" * 64, artifact_root=str(root))
    assert result["available"] is False
    assert result["reason"] == "no_jlens_artifact"


def test_resolve_compact_jlens_model_sha256_tier_refuses_a_tampered_sidecar_file(tmp_path, monkeypatch):
    """Even though the MANIFEST claims the right GGUF sha256, the specific J_layer bytes on disk
    no longer match what the manifest declares -- refused, not read as if trustworthy."""
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "artifacts"
    jdir = _write_contract_jlens_artifact(root / "jlens" / "fixture-model", identity, d_model=8, layer=21)
    # tamper the payload AFTER the (correct) manifest was written
    with open(os.path.join(jdir, "J_layer21.f16"), "r+b") as f:
        f.write(b"\xff\xff\xff\xff")
    result = jt.resolve_compact_jlens(layer=21, model_sha256=identity["sha256"], artifact_root=str(root))
    assert result["available"] is False
    assert result["reason"] == "corrupt_sidecar"


def test_resolve_compact_jlens_model_sha256_tier_refuses_ambiguous_multiple_matches(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "artifacts"
    _write_contract_jlens_artifact(root / "jlens" / "copy-a", identity, d_model=8, layer=21)
    _write_contract_jlens_artifact(root / "jlens" / "copy-b", identity, d_model=8, layer=21)
    result = jt.resolve_compact_jlens(layer=21, model_sha256=identity["sha256"], artifact_root=str(root))
    assert result["available"] is False
    assert result["reason"] == "ambiguous_artifact"


def test_transport_direction_model_sha256_tier_end_to_end(tmp_path, monkeypatch):
    identity = _fake_identity(tmp_path, monkeypatch, d_model=8)
    root = tmp_path / "artifacts"
    _write_contract_jlens_artifact(root / "jlens" / "fixture-model", identity, d_model=8, layer=21)
    direction = [0.3, -0.1, 0.2, 0.4, -0.2, 0.1, 0.05, -0.05]
    result = jt.transport_direction(direction, layer=21, model_sha256=identity["sha256"],
                                    artifact_root=str(root))
    assert result["applied"] is True
    assert result["vector"] != direction
