"""Tests for the SAE-sparsify top-k CPU reference (ROADMAP 3.3).

Self-contained unit tests of the kernel contract (shapes, top-k correctness, tie rule,
ReLU gating, k clamping, the dense round-trip) — numpy only, no torch, no clozn_lab. These
pin the semantics the CUDA kernel must reproduce; validate.py then proves the kernel matches.

Run (from this directory, with the project venv active):
    python -m pytest test_reference.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from reference import SaeTopKResult, sae_topk

ROWS = 12
N_FEATURES = 64


def _random_pre(seed: int = 0, rows: int = ROWS, n_features: int = N_FEATURES) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, n_features)).astype(np.float32)


# --------------------------------------------------------------------------- #
# (a) contract / shape
# --------------------------------------------------------------------------- #


def test_result_shapes_and_types() -> None:
    pre = _random_pre()
    res = sae_topk(pre, k=8)
    assert isinstance(res, SaeTopKResult)
    assert res.indices.shape == (ROWS, 8)
    assert res.values.shape == (ROWS, 8)
    assert res.indices.dtype == np.int64
    assert res.values.dtype == np.float64
    assert res.k_eff == 8
    assert res.indices.min() >= 0 and res.indices.max() < N_FEATURES


def test_indices_ascending_per_row() -> None:
    pre = _random_pre(seed=1)
    res = sae_topk(pre, k=5)
    for r in range(ROWS):
        row = res.indices[r, : res.k_eff]
        assert list(row) == sorted(row)
        assert len(set(row.tolist())) == res.k_eff  # distinct features


def test_bad_args_rejected() -> None:
    pre = _random_pre()
    with pytest.raises(ValueError):
        sae_topk(pre, k=0)
    with pytest.raises(ValueError):
        sae_topk(np.zeros(5), k=2)  # not 2-D


def test_pre_acts_not_mutated() -> None:
    pre = _random_pre(seed=7)
    before = pre.copy()
    sae_topk(pre, k=4)
    np.testing.assert_array_equal(pre, before)


# --------------------------------------------------------------------------- #
# (b) top-k correctness — the selected set is the k largest, values aligned
# --------------------------------------------------------------------------- #


def test_topk_is_k_largest_features() -> None:
    pre = _random_pre(seed=2)
    k = 6
    res = sae_topk(pre, k=k, relu=False)  # signed: pure top-k over raw values
    for r in range(ROWS):
        # The reference's selected set must equal the true k-largest set for the row.
        true_topk = set(np.argsort(-pre[r], kind="stable")[:k].tolist())
        assert set(res.indices[r].tolist()) == true_topk
        # Values are the raw pre-acts at those indices.
        for c in range(k):
            feat = res.indices[r, c]
            assert res.values[r, c] == pytest.approx(float(pre[r, feat]))


def test_values_aligned_with_indices_relu() -> None:
    pre = _random_pre(seed=3)
    res = sae_topk(pre, k=5, relu=True)
    for r in range(ROWS):
        for c in range(res.k_eff):
            feat = res.indices[r, c]
            expected = max(float(pre[r, feat]), 0.0)
            assert res.values[r, c] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# (c) tie rule — equal values resolve toward the LOWER feature index
# --------------------------------------------------------------------------- #


def test_tie_breaks_toward_lower_index_all_equal() -> None:
    # Every feature equal -> the top-k must be the k LOWEST indices.
    pre = np.ones((1, 32), dtype=np.float32)
    res = sae_topk(pre, k=4)
    assert res.indices[0].tolist() == [0, 1, 2, 3]


def test_tie_breaks_on_plateau() -> None:
    # Features 10..19 tied at the max; k=3 picks the three lowest of the plateau.
    pre = np.zeros((1, 64), dtype=np.float32)
    pre[0, 10:20] = 5.0
    res = sae_topk(pre, k=3)
    assert res.indices[0].tolist() == [10, 11, 12]


# --------------------------------------------------------------------------- #
# (d) ReLU gating — non-positive picks emit 0.0; ranking uses the ReLU'd value
# --------------------------------------------------------------------------- #


def test_relu_gates_negative_picks_to_zero() -> None:
    # Only 2 positive features; k=5 forces 3 non-positive picks, which must emit 0.0.
    pre = np.full((1, 16), -1.0, dtype=np.float32)
    pre[0, 3] = 2.0
    pre[0, 9] = 1.0
    res = sae_topk(pre, k=5, relu=True)
    # The two positives are selected with their values; the rest are gated to 0.0.
    idx = res.indices[0].tolist()
    assert 3 in idx and 9 in idx
    vals = {res.indices[0, c]: res.values[0, c] for c in range(res.k_eff)}
    assert vals[3] == pytest.approx(2.0)
    assert vals[9] == pytest.approx(1.0)
    for feat, v in vals.items():
        if feat not in (3, 9):
            assert v == 0.0  # gated


def test_relu_off_keeps_signed_values() -> None:
    pre = np.full((1, 16), -1.0, dtype=np.float32)
    pre[0, 3] = 2.0
    res = sae_topk(pre, k=3, relu=False)
    # Top value is feature 3; the other two are the LOWEST-index -1.0 features (tie rule).
    assert res.indices[0, 0] == 3 or 3 in res.indices[0].tolist()
    vals = {int(res.indices[0, c]): float(res.values[0, c]) for c in range(res.k_eff)}
    assert vals[3] == pytest.approx(2.0)
    # signed: the negative picks keep their raw -1.0 (not gated).
    assert all(v == pytest.approx(-1.0) for f, v in vals.items() if f != 3)


# --------------------------------------------------------------------------- #
# (e) k clamping + dense round-trip
# --------------------------------------------------------------------------- #


def test_k_larger_than_n_features_clamps_and_pads() -> None:
    pre = _random_pre(seed=5, n_features=8)
    res = sae_topk(pre, k=20)
    assert res.k_eff == 8
    # The first 8 columns cover all features (a full permutation per row); pad cols carry 0.
    for r in range(ROWS):
        assert set(res.indices[r, :8].tolist()) == set(range(8))
        assert np.all(res.values[r, 8:] == 0.0)


def test_to_dense_scatters_sparse_code() -> None:
    pre = _random_pre(seed=6)
    res = sae_topk(pre, k=10, relu=True)
    dense = res.to_dense(N_FEATURES)
    assert dense.shape == (ROWS, N_FEATURES)
    # Exactly k_eff nonzeros per row at most (could be fewer if some gated to 0).
    for r in range(ROWS):
        nz = np.flatnonzero(dense[r])
        assert set(nz.tolist()) <= set(res.indices[r, : res.k_eff].tolist())
    # Reconstructed dense matches a direct ReLU+top-k mask of the input.
    relu_pre = np.maximum(pre.astype(np.float64), 0.0)
    for r in range(ROWS):
        keep = set(res.indices[r, : res.k_eff].tolist())
        for f in range(N_FEATURES):
            if f in keep:
                assert dense[r, f] == pytest.approx(relu_pre[r, f])
            else:
                assert dense[r, f] == 0.0
