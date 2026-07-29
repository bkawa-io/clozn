"""test_bisect_acceptance_helpers -- the two pure, model-free helpers in
scripts/smoke/bisect_acceptance.py (`norm`, `scaled_random_vector`) that construct the synthetic
ground-truth perturbations for batteries 1 and 3. No engine, no GPU: these are the same tiny stdlib
vector-math primitives clozn.analysis.transplant's own `_norm`/`_random_equal_norm_vector` use
(deliberately reimplemented here, not imported -- this script is a caller/consumer, not an internal of
clozn.analysis), so this test protects the one property the whole synthetic-perturbation design in
bisect_acceptance.py depends on: `scaled_random_vector` actually produces a vector at the REQUESTED
multiple of the reference row's own norm, deterministically for a given seed, and always finite.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "smoke"))

import bisect_acceptance as ba  # noqa: E402


def test_norm_matches_euclidean_definition():
    assert ba.norm([3.0, 4.0]) == 5.0
    assert ba.norm([0.0, 0.0, 0.0]) == 0.0
    assert abs(ba.norm([1.0, 1.0, 1.0, 1.0]) - 2.0) < 1e-12


def test_scaled_random_vector_hits_the_requested_norm_multiple():
    reference_row = [1.0, -2.0, 3.0, 0.5, -0.5, 4.0, -1.0, 2.5]
    ref_norm = ba.norm(reference_row)
    for scale in (0.5, 1.0, 3.0, 8.0):
        vec = ba.scaled_random_vector(reference_row, seed=123, scale=scale)
        assert len(vec) == len(reference_row)
        assert all(math.isfinite(x) for x in vec), "a well-formed plant vector must never carry NaN/inf"
        assert abs(ba.norm(vec) - scale * ref_norm) < 1e-9 * max(1.0, scale * ref_norm)


def test_scaled_random_vector_is_deterministic_given_a_seed():
    reference_row = [2.0, -1.0, 0.5, 3.0]
    a = ba.scaled_random_vector(reference_row, seed=42, scale=2.0)
    b = ba.scaled_random_vector(reference_row, seed=42, scale=2.0)
    assert a == b, "the same seed must reproduce the exact same planted vector (reproducibility)"


def test_scaled_random_vector_differs_across_seeds():
    reference_row = [2.0, -1.0, 0.5, 3.0, 1.5]
    a = ba.scaled_random_vector(reference_row, seed=1, scale=1.0)
    b = ba.scaled_random_vector(reference_row, seed=2, scale=1.0)
    assert a != b


def test_scaled_random_vector_zero_row_is_never_a_crash():
    # A degenerate (all-zero) reference row has no norm to match -- scaled_random_vector must still
    # return a finite, well-formed vector rather than raising (division by ~zero would explode) or
    # producing NaN/inf, matching the same discipline transplant.py's own _random_equal_norm_vector
    # documents for a zero reference vector.
    zero_row = [0.0, 0.0, 0.0]
    vec = ba.scaled_random_vector(zero_row, seed=0, scale=5.0)
    assert len(vec) == 3
    assert all(math.isfinite(x) for x in vec)
    assert abs(ba.norm(vec)) < 1e-9
