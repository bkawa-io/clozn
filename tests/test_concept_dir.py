"""test_concept_dir.py -- clozn/analysis/concept_dir.py (the any-concept dial: dir(c) =
normalize(J_l^T @ W_U[c])).

Model-free and GPU-free throughout, per the task's guardrails (a GPU experiment is running
elsewhere; this module boots no engine):
  * the loaders (load_jlens_jacobians / load_unembed / ConceptDirSource) are exercised against
    on-disk FIXTURE files this suite writes itself (a tiny synthetic J-lens sidecar + a tiny
    synthetic unembed export), never the real ~/.clozn/jlens or the research lab's artifacts/.
  * the dir(c) self-consistency check ("read dir(c) back through the SAME lens and c comes back
    at top-1") uses a DETERMINISTIC fixture (an orthogonal J + orthonormal W_U rows) so the
    result is a mathematical guarantee of correct transpose/indexing wiring, not a probabilistic
    echo of the real semantic result already validated live (see
    ../clozn-jlens-work/artifacts/dirc_selfconsistency_result.txt) -- this suite proves the CODE
    is wired right; it does not re-derive the science.
  * ConceptSteer is exercised against a FakeEngineClient (mirrors test_engine_add_custom.py's
    _FakeEC pattern) -- no real clozn-server, no socket; FakeEngineClient now ALSO stands in for
    the engine's /jlens/unembed_row route (see its `unembed_rows` param) so the DEFAULT
    in-product path (no lab export at all) is covered without booting a real engine.
  * The FIX (engine/core/serve/routes_jlens.cpp's /jlens/unembed_row) is unit-tested for both the
    success path (test_compute_succeeds_via_engine_unembed_row_with_no_lab_export_configured) and
    the degrade path (test_compute_blocked_unembed_unavailable_when_engine_has_no_row_and_no_lab_
    export): with no unembed_dir/CLOZN_DIRC_UNEMBED_DIR configured AND no working engine row,
    every product-facing call still degrades to a labeled `blocked: "unembed_unavailable"` dict
    instead of raising or silently guessing.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.analysis.concept_dir as cd  # noqa: E402


# ==================================================================================== fixtures

def _orthogonal(seed: int, n: int) -> np.ndarray:
    """A genuinely orthogonal [n, n] float64 matrix via QR -- Q @ Q.T == Q.T @ Q == I exactly (up
    to float64 precision), which is what makes the self-consistency check below a MATHEMATICAL
    guarantee rather than a probabilistic echo."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def _write_jlens_fixture(tmp_path, *, d_model=32, layers=(21,), seed=1):
    """A tiny synthetic J-lens sidecar: manifest.json + J_layer{L}.f16, same file shapes/names the
    real ~/.clozn/jlens directory uses (see engine/core/serve/server_shared.hpp's JlensServe::load
    and the module docstring)."""
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "fixture", "d_model": d_model, "vocab": d_model, "layers": list(layers),
                "engine_default_tap_layer": layers[0]}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    matrices = {}
    for i, layer in enumerate(layers):
        J = _orthogonal(seed + i, d_model)
        matrices[layer] = J.astype(np.float32)
        J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    return str(jdir), matrices


def _write_unembed_fixture(tmp_path, *, d_model=32, vocab=32, seed=2):
    """A tiny synthetic unembed export: norm_weight.npy / lm_head_weight.npy / unembed_meta.json,
    the SAME 3-file shape ../clozn-jlens-work/scripts/dirc.py's load_unembed already produces.
    lm_head_weight's rows are ORTHONORMAL (vocab <= d_model) so w_c . w_v is EXACTLY 1 for v==c
    and EXACTLY 0 otherwise -- the self-consistency guarantee below follows from this by
    construction, not by luck."""
    udir = tmp_path / "unembed"
    udir.mkdir()
    q = _orthogonal(seed, d_model)[:vocab].astype(np.float32)
    np.save(str(udir / "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
    np.save(str(udir / "lm_head_weight.npy"), q)
    (udir / "unembed_meta.json").write_text(json.dumps({"rms_norm_eps": 1e-6}), encoding="utf-8")
    return str(udir), q


class FakeEngineClient:
    """Stands in for clozn_engine.EngineClient: ConceptSteer.resolve_token_id calls
    `.score(prompt=, continuation=, topk=)` (reusing /score as a tokenizer), and
    ConceptSteer.compute (via ConceptDirSource.unembed_row) calls `.unembed_row(token_id)` -- the
    NEW /jlens/unembed_row round trip that closed the W_U blocker (see the module docstring).
    `vocab` maps a leading-space concept string -> a list of token ids (len>1 simulates a
    multi-token word); unknown words default to a single fresh id. `unembed_rows` maps
    token_id -> a [d_model] row (simulating what the real engine's /jlens/unembed_row would
    return for that token); a token_id with no configured row raises, mirroring a real engine
    round-trip failure (no J-lens loaded, connection error, ...) -- exercised by the existing
    "no unembed configured" tests, which never populate this dict."""

    def __init__(self, vocab=None, raises=False, unembed_rows=None, unembed_raises=False):
        self.vocab = dict(vocab or {})
        self.raises = raises
        self.unembed_rows = {int(k): list(v) for k, v in (unembed_rows or {}).items()}
        self.unembed_raises = unembed_raises
        self.calls = []
        self.unembed_calls = []
        self._next_id = 1000

    def score(self, prompt=None, continuation=None, topk=0, **kw):
        self.calls.append({"prompt": prompt, "continuation": continuation, "topk": topk})
        if self.raises:
            raise RuntimeError("engine unreachable")
        ids = self.vocab.get(continuation)
        if ids is None:
            ids = [self._next_id]
            self._next_id += 1
            self.vocab[continuation] = ids
        return {"tokens": [{"id": i, "piece": continuation} for i in ids]}

    def unembed_row(self, token_id):
        self.unembed_calls.append(int(token_id))
        if self.unembed_raises:
            raise RuntimeError("engine unreachable")
        row = self.unembed_rows.get(int(token_id))
        if row is None:
            raise RuntimeError(f"no /jlens/unembed_row fixture configured for token {token_id} "
                               "(simulates: J-lens not loaded / engine unreachable)")
        return {"token_id": int(token_id), "piece": None, "d_model": len(row), "vector": list(row)}


# ==================================================================================== loaders: load_jlens_jacobians

def test_load_jlens_jacobians_reads_raw_fp16_sidecar_matching_manifest(tmp_path):
    jdir, matrices = _write_jlens_fixture(tmp_path, d_model=8, layers=(21, 25))
    out = cd.load_jlens_jacobians(jdir)
    assert sorted(out) == [21, 25]
    for layer, J in matrices.items():
        assert out[layer].shape == (8, 8)
        assert out[layer].dtype == np.float32
        np.testing.assert_allclose(out[layer], J, atol=2e-3)  # fp16 round-trip tolerance


def test_load_jlens_jacobians_can_select_a_subset_of_layers(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21, 25))
    out = cd.load_jlens_jacobians(jdir, layers=[21])
    assert list(out) == [21]


def test_load_jlens_jacobians_unfitted_layer_raises_valueerror(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    with pytest.raises(ValueError, match="not a fitted"):
        cd.load_jlens_jacobians(jdir, layers=[99])


def test_load_jlens_jacobians_missing_sidecar_file_raises(tmp_path):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "fixture", "d_model": 8, "vocab": 8, "layers": [21], "engine_default_tap_layer": 21}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        cd.load_jlens_jacobians(str(jdir))


def test_default_jlens_dir_honors_clozn_jlens_dir_env_var(monkeypatch):
    monkeypatch.setenv("CLOZN_JLENS_DIR", r"C:\somewhere\jlens")
    assert cd._default_jlens_dir() == r"C:\somewhere\jlens"


def test_default_jlens_dir_falls_back_to_home_dot_clozn(monkeypatch):
    monkeypatch.delenv("CLOZN_JLENS_DIR", raising=False)
    expected = os.path.join(os.path.expanduser("~"), ".clozn", "jlens")
    assert cd._default_jlens_dir() == expected


# ==================================================================================== loaders: load_unembed (the BLOCKER)

def test_load_unembed_raises_unembedunavailable_with_nothing_configured(monkeypatch):
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    with pytest.raises(cd.UnembedUnavailable):
        cd.load_unembed(None)


def test_load_unembed_blocker_note_names_the_actual_gap():
    # The note must be self-explanatory to whoever hits this in production -- it should name BOTH
    # halves of dir(c) (J_l ships; W_U doesn't) so a reader isn't left guessing which artifact.
    assert "W_U" in cd.BLOCKER_NOTE or "unembed" in cd.BLOCKER_NOTE.lower()
    assert "J_l" in cd.BLOCKER_NOTE or "jlens" in cd.BLOCKER_NOTE.lower()


def test_load_unembed_missing_files_in_configured_dir_raises_unembedunavailable(tmp_path):
    empty_dir = tmp_path / "empty_unembed"
    empty_dir.mkdir()
    with pytest.raises(cd.UnembedUnavailable):
        cd.load_unembed(str(empty_dir))


def test_load_unembed_reads_fixture_correctly(tmp_path):
    udir, q = _write_unembed_fixture(tmp_path, d_model=8, vocab=8)
    uw = cd.load_unembed(udir)
    assert uw.lm_head_weight.shape == (8, 8)
    assert uw.norm_weight.shape == (8,)
    assert uw.eps == 1e-6
    np.testing.assert_allclose(uw.lm_head_weight, q)


def test_load_unembed_honors_clozn_dirc_unembed_dir_env_var(tmp_path, monkeypatch):
    udir, _ = _write_unembed_fixture(tmp_path, d_model=8, vocab=8)
    monkeypatch.setenv("CLOZN_DIRC_UNEMBED_DIR", udir)
    uw = cd.load_unembed(None)   # no explicit arg -- must fall back to the env var
    assert uw.lm_head_weight.shape == (8, 8)


# ==================================================================================== dir(c) self-consistency (the math)

def test_dir_c_self_consistency_recovers_c_exactly_in_memory():
    """Pure in-memory guarantee, no file I/O: an orthogonal J + orthonormal W_U rows makes
    unembed(J_l @ dir(c)) recover c at EXACT rank 0 for every c, by construction (see
    _write_unembed_fixture's docstring for the algebra) -- proves the transpose/indexing
    convention in dir_c/read_through_lens is wired correctly, independent of any real model."""
    d_model, vocab = 24, 24
    J = _orthogonal(10, d_model).astype(np.float32)
    W = _orthogonal(11, d_model)[:vocab].astype(np.float32)
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32), lm_head_weight=W, eps=1e-6)
    J_by_layer = {21: J}
    for c in range(vocab):
        vec = cd.dir_c(c, 21, J_by_layer, uw, scale=1.0)
        logits = cd.read_through_lens(vec, 21, J_by_layer, uw)
        assert cd.rank_of(logits, c) == 0, f"token {c} did not recover at top-1"


def test_dir_c_self_consistency_through_the_real_file_loaders(tmp_path):
    """SAME guarantee, but exercising the actual on-disk loaders (fp16 J sidecar + npy unembed) --
    covers the file-reading/reshape code path, not just the pure math."""
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=32, layers=(21,), seed=5)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=32, vocab=32, seed=6)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    J_by_layer = source.jacobians()
    uw = source.unembed()
    for c in (0, 5, 17, 31):
        vec = cd.dir_c(c, 21, J_by_layer, uw, scale=1.0)
        logits = cd.read_through_lens(vec, 21, J_by_layer, uw)
        assert cd.rank_of(logits, c) == 0


def test_dir_c_scale_invariance_of_rank():
    """RMSNorm-then-argmax is invariant to any positive rescaling of dir(c)'s input -- the rank
    must be identical across scales (a code sanity check, mirrors run_dirc_selfconsistency.py's
    check B)."""
    d_model = 16
    J = _orthogonal(20, d_model).astype(np.float32)
    W = _orthogonal(21, d_model).astype(np.float32)
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32), lm_head_weight=W, eps=1e-6)
    J_by_layer = {21: J}
    ranks = []
    for scale in (0.1, 1.0, 10.0, 100.0):
        vec = cd.dir_c(3, 21, J_by_layer, uw, scale=scale)
        logits = cd.read_through_lens(vec, 21, J_by_layer, uw)
        ranks.append(cd.rank_of(logits, 3))
    assert len(set(ranks)) == 1


def test_dir_c_typical_norm_scales_the_vector_magnitude():
    d_model = 8
    J = _orthogonal(30, d_model).astype(np.float32)
    W = _orthogonal(31, d_model).astype(np.float32)
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32), lm_head_weight=W, eps=1e-6)
    unit = cd.dir_c(0, 21, {21: J}, uw, scale=1.0, typical_norm=None)
    scaled = cd.dir_c(0, 21, {21: J}, uw, scale=0.5, typical_norm=146.68)
    assert abs(float(np.linalg.norm(unit)) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(scaled)) - 0.5 * 146.68) < 1e-2


def test_dir_c_raises_on_unfitted_layer():
    d_model = 4
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32),
                           lm_head_weight=np.eye(d_model, dtype=np.float32))
    with pytest.raises(ValueError, match="no loaded J_l"):
        cd.dir_c(0, 99, {21: np.eye(d_model, dtype=np.float32)}, uw)


def test_dir_c_raises_on_out_of_range_token_id():
    d_model = 4
    J = np.eye(d_model, dtype=np.float32)
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32),
                           lm_head_weight=np.eye(d_model, dtype=np.float32))
    with pytest.raises(ValueError, match="out of range"):
        cd.dir_c(99, 21, {21: J}, uw)


def test_dir_c_raises_on_degenerate_direction():
    """A token whose W_U row is entirely in J_l's null space produces a ~zero raw direction --
    must raise, never silently return a garbage/zero vector."""
    d_model = 4
    J = np.zeros((d_model, d_model), dtype=np.float32)   # the zero map: every raw direction is 0
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32),
                           lm_head_weight=np.eye(d_model, dtype=np.float32))
    with pytest.raises(ValueError, match="degenerate"):
        cd.dir_c(0, 21, {21: J}, uw)


# ==================================================================================== dir_c_from_row + fetch_unembed_row_from_engine
# (the engine-route path's own math/plumbing -- the fix that lets dir(c) work from
# ONE already-fetched W_U row instead of the full [vocab, d_model] matrix.)

def test_dir_c_from_row_matches_dir_c_given_the_same_row():
    """dir_c_from_row (engine-route path) must be the SAME math as dir_c (lab-export path) -- they
    only differ in whether the caller already extracted the row or hands over the whole matrix."""
    d_model = 16
    J = _orthogonal(70, d_model).astype(np.float32)
    W = _orthogonal(71, d_model).astype(np.float32)
    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32), lm_head_weight=W, eps=1e-6)
    for c in (0, 5, 15):
        a = cd.dir_c(c, 21, {21: J}, uw, scale=0.7)
        b = cd.dir_c_from_row(W[c], 21, {21: J}, scale=0.7)
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_dir_c_from_row_raises_on_unfitted_layer():
    with pytest.raises(ValueError, match="no loaded J_l"):
        cd.dir_c_from_row(np.ones(4, dtype=np.float32), 99, {21: np.eye(4, dtype=np.float32)})


def test_dir_c_from_row_raises_on_d_model_mismatch():
    J = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        cd.dir_c_from_row(np.ones(3, dtype=np.float32), 21, {21: J})


def test_dir_c_from_row_raises_on_degenerate_direction():
    d_model = 4
    J = np.zeros((d_model, d_model), dtype=np.float32)
    with pytest.raises(ValueError, match="degenerate"):
        cd.dir_c_from_row(np.ones(d_model, dtype=np.float32), 21, {21: J})


def test_fetch_unembed_row_from_engine_calls_client_and_returns_array():
    ec = FakeEngineClient(unembed_rows={5: [1.0, 2.0, 3.0]})
    row = cd.fetch_unembed_row_from_engine(ec, 5)
    np.testing.assert_allclose(row, [1.0, 2.0, 3.0])
    assert ec.unembed_calls == [5]


def test_fetch_unembed_row_from_engine_propagates_client_failure():
    ec = FakeEngineClient(unembed_raises=True)
    with pytest.raises(RuntimeError, match="unreachable"):
        cd.fetch_unembed_row_from_engine(ec, 5)


def test_fetch_unembed_row_from_engine_raises_unembedunavailable_on_malformed_response():
    class BadEngineClient:
        def unembed_row(self, token_id):
            return {"token_id": token_id}   # no "vector" -- a server contract violation

    with pytest.raises(cd.UnembedUnavailable):
        cd.fetch_unembed_row_from_engine(BadEngineClient(), 5)


# ==================================================================================== ConceptDirSource

def test_concept_dir_source_available_layers(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(2, 14, 21, 25))
    source = cd.ConceptDirSource(jlens_dir=jdir)
    assert source.available_layers() == [2, 14, 21, 25]


def test_concept_dir_source_unembed_available_false_without_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    source = cd.ConceptDirSource(jlens_dir=jdir)
    assert source.unembed_available() is False


def test_concept_dir_source_unembed_available_true_when_configured(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    udir, _ = _write_unembed_fixture(tmp_path, d_model=8, vocab=8)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    assert source.unembed_available() is True


def test_concept_dir_source_unembed_row_prefers_explicit_lab_export_over_engine(tmp_path):
    """An explicit unembed_dir/CLOZN_DIRC_UNEMBED_DIR always wins, matching every other config
    knob in this module -- the engine route is the DEFAULT, not an override."""
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    udir, W = _write_unembed_fixture(tmp_path, d_model=8, vocab=8)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(unembed_rows={0: [999.0] * 8})   # would be WRONG if ever used

    row = source.unembed_row(0, engine_client=ec)

    np.testing.assert_allclose(row, W[0])
    assert ec.unembed_calls == []   # lab export wins; engine never called


def test_concept_dir_source_unembed_row_falls_back_to_engine_without_lab_export(tmp_path, monkeypatch):
    """THE fix: with NO unembed_dir/env configured at all, unembed_row() still succeeds -- via the
    engine's /jlens/unembed_row (fetch_unembed_row_from_engine)."""
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    source = cd.ConceptDirSource(jlens_dir=jdir)   # no unembed_dir at all
    ec = FakeEngineClient(unembed_rows={3: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})

    row = source.unembed_row(3, engine_client=ec)

    np.testing.assert_allclose(row, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert ec.unembed_calls == [3]


def test_concept_dir_source_unembed_row_raises_when_neither_available(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    source = cd.ConceptDirSource(jlens_dir=jdir)
    with pytest.raises(cd.UnembedUnavailable):
        source.unembed_row(0, engine_client=None)


def test_concept_dir_source_jacobians_caches_and_only_loads_missing_layers(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21, 25))
    source = cd.ConceptDirSource(jlens_dir=jdir)
    first = source.jacobians(layers=[21])
    assert list(first) == [21]
    second = source.jacobians(layers=[21, 25])
    assert sorted(second) == [21, 25]
    assert second[21] is source._J[21]  # not reloaded


# ==================================================================================== ConceptSteer.resolve_token_id

def test_resolve_token_id_single_token_ok():
    ec = FakeEngineClient(vocab={" ocean": [777]})
    steer = cd.ConceptSteer(ec)
    out = steer.resolve_token_id("ocean")
    assert out == {"ok": True, "token_id": 777, "piece": " ocean"}


def test_resolve_token_id_multi_token_is_blocked_not_truncated():
    ec = FakeEngineClient(vocab={" antidisestablishmentarianism": [1, 2, 3]})
    steer = cd.ConceptSteer(ec)
    out = steer.resolve_token_id("antidisestablishmentarianism")
    assert out["ok"] is False
    assert "not a single token" in out["note"]


def test_resolve_token_id_empty_concept_is_graceful():
    steer = cd.ConceptSteer(FakeEngineClient())
    out = steer.resolve_token_id("   ")
    assert out == {"ok": False, "note": "empty concept"}


def test_resolve_token_id_engine_failure_is_graceful_never_raises():
    ec = FakeEngineClient(raises=True)
    steer = cd.ConceptSteer(ec)
    out = steer.resolve_token_id("ocean")
    assert out["ok"] is False
    assert "failed" in out["note"]


# ==================================================================================== ConceptSteer.compute / steer_toward

def _steer_with_fixtures(tmp_path, *, layer=21, d_model=32, vocab=32, vocab_map=None, median_norm=None):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=d_model, layers=(layer,), seed=40)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=d_model, vocab=vocab, seed=41)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab=vocab_map or {" ocean": [3]})
    return cd.ConceptSteer(ec, source=source, layer=layer, median_norm=median_norm), ec


def test_compute_blocked_unembed_unavailable_when_engine_has_no_row_and_no_lab_export(tmp_path, monkeypatch):
    """The degrade path THIS module keeps tested even after the fix: no lab export configured AND
    the engine_client can't produce the row either (here: FakeEngineClient with no unembed_rows
    configured, simulating "engine unreachable" / "no J-lens sidecar loaded") -- must still
    degrade to a labeled blocked dict, never raise."""
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    source = cd.ConceptDirSource(jlens_dir=jdir)   # no unembed_dir
    ec = FakeEngineClient(vocab={" ocean": [3]})   # no unembed_rows configured
    steer = cd.ConceptSteer(ec, source=source, layer=21)

    out = steer.compute("ocean")

    assert out["ok"] is False
    assert out["blocked"] == "unembed_unavailable"
    assert out["concept"] == "ocean"


def test_compute_succeeds_via_engine_unembed_row_with_no_lab_export_configured(tmp_path, monkeypatch):
    """THE fix: dir(c) now works with ONLY the shipped J-lens
    sidecar + a running engine -- no lab-only unembed export needed at all. Hands ConceptSteer a
    full orthogonal W_U ONLY via the fake engine's /jlens/unembed_row (never via
    unembed_dir/CLOZN_DIRC_UNEMBED_DIR, which stay unset), then checks the resulting dir(c)
    recovers the token at rank 0 through the SAME self-consistency check the lab-export path
    already proves (test_dir_c_self_consistency_through_the_real_file_loaders) -- using the full
    W ONLY for this offline verification, never handed to ConceptSteer itself."""
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    d_model = 32
    jdir, J_by_layer = _write_jlens_fixture(tmp_path, d_model=d_model, layers=(21,), seed=60)
    W = _orthogonal(61, d_model).astype(np.float32)   # a full W_U, held ONLY by the test/"engine"
    source = cd.ConceptDirSource(jlens_dir=jdir)      # NOTE: no unembed_dir at all
    token_id = 7
    ec = FakeEngineClient(vocab={" ocean": [token_id]},
                         unembed_rows={token_id: W[token_id].tolist()})
    steer = cd.ConceptSteer(ec, source=source, layer=21)

    out = steer.compute("ocean")

    assert out["ok"] is True, out
    assert out["token_id"] == token_id
    assert ec.unembed_calls == [token_id]   # fetched exactly the ONE row it needed

    uw = cd.UnembedWeights(norm_weight=np.ones(d_model, dtype=np.float32), lm_head_weight=W, eps=1e-6)
    logits = cd.read_through_lens(np.asarray(out["vector"]), 21, J_by_layer, uw)
    assert cd.rank_of(logits, token_id) == 0


def test_compute_blocked_token_resolution_propagates_the_reason(tmp_path):
    steer, _ec = _steer_with_fixtures(tmp_path, vocab_map={" weird": [1, 2]})
    out = steer.compute("weird")
    assert out["ok"] is False
    assert out["blocked"] == "token_resolution"


def test_compute_success_returns_unit_vector_and_caches(tmp_path):
    steer, ec = _steer_with_fixtures(tmp_path, vocab_map={" ocean": [3]})

    out = steer.compute("ocean")

    assert out["ok"] is True
    assert out["token_id"] == 3
    assert out["layer"] == 21
    assert len(out["vector"]) == 32
    assert abs(float(np.linalg.norm(out["vector"])) - 1.0) < 1e-4

    calls_before = len(ec.calls)
    out2 = steer.compute("ocean")
    assert out2["vector"] == out["vector"]
    assert len(ec.calls) == calls_before   # cached -- no second /score round trip


def test_compute_bad_layer_reports_available_layers(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    udir, _ = _write_unembed_fixture(tmp_path, d_model=8, vocab=8)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    steer = cd.ConceptSteer(ec, source=source, layer=21)

    out = steer.compute("ocean", layer=99)

    assert out["ok"] is False
    assert out["blocked"] == "bad_layer"
    assert out["note"] and "21" in out["note"]


def test_steer_toward_returns_intervene_ready_payload(tmp_path):
    steer, _ec = _steer_with_fixtures(tmp_path, vocab_map={" ocean": [3]})

    out = steer.steer_toward("ocean", 0.4)

    assert out["ok"] is True
    assert out["concept"] == "ocean"
    assert out["token_id"] == 3
    assert out["layer"] == 21
    assert out["strength"] == 0.4
    assert out["coef"] == pytest.approx(0.4 * cd.VALIDATED_MEDIAN_RESID_NORM[21])
    assert abs(float(np.linalg.norm(out["vector"])) - 1.0) < 1e-4
    assert out["note"] is None
    assert steer.strength["ocean"] == 0.4   # persisted, like a tone dial


def test_steer_toward_warns_when_strength_over_validated_range(tmp_path):
    steer, _ec = _steer_with_fixtures(tmp_path, vocab_map={" ocean": [3]})
    out = steer.steer_toward("ocean", 1.5)
    assert out["ok"] is True
    assert "over-injects" in out["note"]


def test_steer_toward_propagates_blocked_reason_and_keeps_strength(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(21,))
    source = cd.ConceptDirSource(jlens_dir=jdir)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    steer = cd.ConceptSteer(ec, source=source, layer=21)

    out = steer.steer_toward("ocean", 0.4)

    assert out["ok"] is False
    assert out["blocked"] == "unembed_unavailable"
    assert out["strength"] == 0.4
    assert "ocean" not in steer.strength   # never persisted on failure


def test_steer_toward_blocked_when_no_median_norm_calibration(tmp_path):
    """Layer 14 is a real fitted J-lens layer but VALIDATED_MEDIAN_RESID_NORM only covers 21/25 --
    a legitimate 'we don't have a calibrated injection magnitude for this layer yet' degrade."""
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=8, layers=(14,), seed=50)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=8, vocab=8, seed=51)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    steer = cd.ConceptSteer(ec, source=source, layer=14)

    out = steer.steer_toward("ocean", 0.4)

    assert out["ok"] is False
    assert out["blocked"] == "no_norm_calibration"


# ==================================================================================== persistent-dial surface

def test_set_clamps_like_enginesteer():
    steer = cd.ConceptSteer(FakeEngineClient())
    steer.set("ocean", 99)
    assert steer.strength["ocean"] == 1.5
    steer.set("ocean", -99)
    assert steer.strength["ocean"] == -1.5


def test_clear_and_active():
    steer = cd.ConceptSteer(FakeEngineClient())
    steer.set("ocean", 0.4)
    steer.set("dog", 0.0)
    assert steer.active() == {"ocean": 0.4}
    steer.clear()
    assert steer.active() == {}


def test_steer_vector_returns_none_when_nothing_active(tmp_path):
    steer, _ec = _steer_with_fixtures(tmp_path)
    assert steer.steer_vector() is None


def test_steer_vector_sums_active_concepts_pre_scaled(tmp_path):
    steer, _ec = _steer_with_fixtures(tmp_path, vocab_map={" ocean": [3], " dog": [5]})
    steer.set("ocean", 0.4)
    steer.set("dog", -0.3)

    vec = steer.steer_vector()

    assert vec is not None
    assert len(vec) == 32
    ocean = np.asarray(steer.compute("ocean")["vector"]) * (0.4 * cd.VALIDATED_MEDIAN_RESID_NORM[21])
    dog = np.asarray(steer.compute("dog")["vector"]) * (-0.3 * cd.VALIDATED_MEDIAN_RESID_NORM[21])
    np.testing.assert_allclose(vec, (ocean + dog).tolist(), atol=1e-3)


def test_steer_vector_skips_a_concept_that_fails_to_build(tmp_path):
    steer, ec = _steer_with_fixtures(tmp_path, vocab_map={" ocean": [3], " broken": [1, 2]})
    steer.set("ocean", 0.4)
    steer.set("broken", 0.5)   # multi-token -> can't build, must be skipped, not fatal

    vec = steer.steer_vector()

    assert vec is not None
    expected = np.asarray(steer.compute("ocean")["vector"]) * (0.4 * cd.VALIDATED_MEDIAN_RESID_NORM[21])
    np.testing.assert_allclose(vec, expected.tolist(), atol=1e-3)


# ==================================================================================== per-model concept-dial
# calibration: the schema, the path convention, and the missing-file=no-op fallback discipline (mirroring
# clozn.server.app._dial_calibration()'s tone-dial calibration file). All model-free -- the live
# measurement sweep (scripts/calibration/concept_dial_autocalibrate.py) is deferred (needs a real engine).

def test_concept_calibration_path_scopes_per_exact_digest():
    p = cd.concept_calibration_path("abc123")
    assert p == os.path.join(os.path.expanduser("~"), ".clozn", "models", "abc123",
                             "concept_dial_calibration.json")


def test_concept_calibration_path_falls_back_to_legacy_root_without_a_digest():
    p = cd.concept_calibration_path(None)
    assert p == os.path.join(os.path.expanduser("~"), ".clozn", "concept_dial_calibration.json")


def test_load_concept_dial_calibration_missing_file_returns_none(tmp_path):
    assert cd.load_concept_dial_calibration(path=str(tmp_path / "nope.json")) is None


def test_load_concept_dial_calibration_wrong_schema_returns_none(tmp_path):
    p = tmp_path / "concept_dial_calibration.json"
    p.write_text(json.dumps({"schema": "something.else.v1",
                             "layers": {"21": {"median_resid_norm": 100.0}}}), encoding="utf-8")
    assert cd.load_concept_dial_calibration(path=str(p)) is None


def test_load_concept_dial_calibration_no_layers_returns_none(tmp_path):
    p = tmp_path / "concept_dial_calibration.json"
    p.write_text(json.dumps({"schema": cd.CONCEPT_CALIBRATION_SCHEMA, "layers": {}}), encoding="utf-8")
    assert cd.load_concept_dial_calibration(path=str(p)) is None


def test_load_concept_dial_calibration_corrupt_json_returns_none(tmp_path):
    p = tmp_path / "concept_dial_calibration.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert cd.load_concept_dial_calibration(path=str(p)) is None


def test_save_then_load_concept_dial_calibration_round_trips(tmp_path):
    p = str(tmp_path / "concept_dial_calibration.json")
    written = cd.save_concept_dial_calibration(
        "modelsha", {21: {"median_resid_norm": 200.5, "usable_scale_range": [0.2, 0.6],
                          "derail_point": 0.8, "works": True, "n_samples": 12}},
        path=p, note="smoke sweep",
    )
    assert written == p
    out = cd.load_concept_dial_calibration(path=p)
    assert out == {21: {"median_resid_norm": 200.5, "usable_scale_range": (0.2, 0.6),
                        "derail_point": 0.8, "works": True}}


def test_save_concept_dial_calibration_raises_on_missing_median_norm(tmp_path):
    with pytest.raises(ValueError, match="median_resid_norm"):
        cd.save_concept_dial_calibration("m", {21: {"works": True}}, path=str(tmp_path / "x.json"))


def test_save_concept_dial_calibration_raises_on_bad_scale_range_shape(tmp_path):
    with pytest.raises(ValueError, match="usable_scale_range"):
        cd.save_concept_dial_calibration(
            "m", {21: {"median_resid_norm": 10.0, "usable_scale_range": [0.1]}}, path=str(tmp_path / "x.json"))


def test_load_concept_dial_calibration_skips_malformed_layer_entries_keeps_the_rest(tmp_path):
    p = tmp_path / "concept_dial_calibration.json"
    p.write_text(json.dumps({
        "schema": cd.CONCEPT_CALIBRATION_SCHEMA,
        "layers": {
            "21": {"median_resid_norm": 150.0},
            "not-an-int": {"median_resid_norm": 999.0},
            "25": {"median_resid_norm": None},          # wrong type -- dropped
            "16": "not-a-dict",                          # wrong shape -- dropped
        },
    }), encoding="utf-8")
    out = cd.load_concept_dial_calibration(path=str(p))
    assert out == {21: {"median_resid_norm": 150.0, "usable_scale_range": None,
                        "derail_point": None, "works": True}}


# ==================================================================================== ConceptSteer wiring:
# per-model calibration present -> used; absent -> global fallback; both cases say which source applied.

def test_concept_steer_defaults_to_global_when_no_model_sha256_configured(tmp_path):
    """No model_sha256/calibration passed at all -- existing callers must behave EXACTLY as before this
    feature existed (Law #6)."""
    steer, _ec = _steer_with_fixtures(tmp_path)   # layer=21 default
    assert steer.median_norm == cd.VALIDATED_MEDIAN_RESID_NORM[21]
    assert steer.median_norm_source == "global_default"
    assert steer.scale_range == cd.VALIDATED_SCALE_RANGE
    assert steer.scale_range_source == "global_default"


def test_concept_steer_explicit_median_norm_still_wins_over_everything(tmp_path):
    steer, _ec = _steer_with_fixtures(tmp_path, median_norm=999.0)
    assert steer.median_norm == 999.0
    assert steer.median_norm_source == "explicit"


def test_concept_steer_uses_injected_per_model_calibration_dict(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=32, layers=(21,), seed=40)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=32, vocab=32, seed=41)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    calibration = {21: {"median_resid_norm": 55.5, "usable_scale_range": (0.1, 0.2),
                        "derail_point": 0.3, "works": True}}

    steer = cd.ConceptSteer(ec, source=source, layer=21, calibration=calibration)

    assert steer.median_norm == 55.5
    assert steer.median_norm_source == "per_model_calibration"
    assert steer.scale_range == (0.1, 0.2)
    assert steer.scale_range_source == "per_model_calibration"

    out = steer.steer_toward("ocean", 0.4)
    assert out["ok"] is True
    assert out["coef"] == pytest.approx(0.4 * 55.5)
    assert out["norm_source"] == "per_model_calibration"


def test_concept_steer_note_reflects_the_per_model_scale_range_when_present(tmp_path):
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=32, layers=(21,), seed=40)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=32, vocab=32, seed=41)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    calibration = {21: {"median_resid_norm": 55.5, "usable_scale_range": (0.1, 0.2)}}
    steer = cd.ConceptSteer(ec, source=source, layer=21, calibration=calibration)

    out = steer.steer_toward("ocean", 1.5)

    assert out["ok"] is True
    assert "0.1" in out["note"] and "0.2" in out["note"]


def test_concept_steer_model_sha256_reads_the_real_loader_via_concept_calibration_path(tmp_path, monkeypatch):
    """Full wiring, not just the injected-dict shortcut: ConceptSteer(model_sha256=...) actually calls
    load_concept_dial_calibration, which resolves concept_calibration_path -- proven by monkeypatching that
    resolver into tmp_path instead of touching the real ~/.clozn."""
    calib_path = tmp_path / "concept_dial_calibration.json"
    monkeypatch.setattr(cd, "concept_calibration_path", lambda model_sha256=None: str(calib_path))
    cd.save_concept_dial_calibration("digest123", {21: {"median_resid_norm": 77.0}}, path=str(calib_path))

    jdir, _ = _write_jlens_fixture(tmp_path, d_model=32, layers=(21,), seed=40)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=32, vocab=32, seed=41)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab={" ocean": [3]})

    steer = cd.ConceptSteer(ec, source=source, layer=21, model_sha256="digest123")

    assert steer.median_norm == 77.0
    assert steer.median_norm_source == "per_model_calibration"


def test_concept_steer_model_sha256_falls_back_to_global_when_calibration_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "concept_calibration_path", lambda model_sha256=None: str(tmp_path / "nope.json"))
    jdir, _ = _write_jlens_fixture(tmp_path, d_model=32, layers=(21,), seed=40)
    udir, _ = _write_unembed_fixture(tmp_path, d_model=32, vocab=32, seed=41)
    source = cd.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
    ec = FakeEngineClient(vocab={" ocean": [3]})

    steer = cd.ConceptSteer(ec, source=source, layer=21, model_sha256="digest123")

    assert steer.median_norm == cd.VALIDATED_MEDIAN_RESID_NORM[21]
    assert steer.median_norm_source == "global_default"
    assert steer.scale_range == cd.VALIDATED_SCALE_RANGE
    assert steer.scale_range_source == "global_default"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
