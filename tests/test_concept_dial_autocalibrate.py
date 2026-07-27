"""test_concept_dial_autocalibrate.py -- model-free tests for
scripts/calibration/concept_dial_autocalibrate.py's PURE math (median_resid_norm_from_harvests,
_compute_scale_calibration). No engine, no GPU, no model: the live-measurement functions
(measure_median_resid_norm, measure_concept_scale_sweep, calibrate_layer, main) all talk to a real
clozn-server and are deliberately DEFERRED from this suite -- see the module's own docstring for why (the
same "live path deferred, pure math model-free tested" split this codebase already uses for quant-check and
research/dial_autocalibrate_engine.py's own _compute_calibration).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts", "calibration")
sys.path.insert(0, SCRIPTS)

import concept_dial_autocalibrate as cdac   # noqa: E402


# ==================================================================================== median_resid_norm_from_harvests

def test_median_resid_norm_single_prompt():
    acts = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])   # row norms: 5.0, 5.0
    assert cdac.median_resid_norm_from_harvests([acts]) == pytest.approx(5.0)


def test_median_resid_norm_pools_across_prompts():
    a = np.array([[3.0, 4.0, 0.0]])          # norm 5.0
    b = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0]])  # norms 1.0, 9.0
    # pooled sample: [5.0, 1.0, 9.0] -> median 5.0
    assert cdac.median_resid_norm_from_harvests([a, b]) == pytest.approx(5.0)


def test_median_resid_norm_empty_input_is_zero():
    assert cdac.median_resid_norm_from_harvests([]) == 0.0


def test_median_resid_norm_skips_malformed_rows():
    good = np.array([[3.0, 4.0]])            # norm 5.0
    bad_1d = np.array([1.0, 2.0])            # wrong ndim -- skipped, never crashes
    bad_empty = np.zeros((0, 2))              # zero rows -- skipped
    assert cdac.median_resid_norm_from_harvests([good, bad_1d, bad_empty]) == pytest.approx(5.0)


# ==================================================================================== _compute_scale_calibration

def test_compute_scale_calibration_normal_case():
    curve = [
        {"scale": 0.0, "logprob_delta": 0.0, "degenerate_rate": 0.0},
        {"scale": 0.25, "logprob_delta": 3.0, "degenerate_rate": 0.0},
        {"scale": 0.5, "logprob_delta": 5.0, "degenerate_rate": 0.1},
        {"scale": 1.0, "logprob_delta": 6.0, "degenerate_rate": 0.6},   # derails here
    ]
    out = cdac._compute_scale_calibration(curve, degen_threshold=0.34, logprob_rise_min=2.0)
    assert out["works"] is True
    assert out["usable_scale_range"] == [0.25, 0.5]
    assert out["derail_point"] == 1.0


def test_compute_scale_calibration_never_clears_logprob_floor():
    curve = [
        {"scale": 0.0, "logprob_delta": 0.0, "degenerate_rate": 0.0},
        {"scale": 0.25, "logprob_delta": 0.5, "degenerate_rate": 0.0},
        {"scale": 0.5, "logprob_delta": 1.0, "degenerate_rate": 0.0},
    ]
    out = cdac._compute_scale_calibration(curve, degen_threshold=0.34, logprob_rise_min=2.0)
    assert out["works"] is False
    assert out["usable_scale_range"] is None
    assert out["derail_point"] is None


def test_compute_scale_calibration_derails_immediately():
    curve = [
        {"scale": 0.0, "logprob_delta": 0.0, "degenerate_rate": 0.0},
        {"scale": 0.25, "logprob_delta": 5.0, "degenerate_rate": 0.9},   # derails at the very first dose
    ]
    out = cdac._compute_scale_calibration(curve, degen_threshold=0.34, logprob_rise_min=2.0)
    assert out["derail_point"] == 0.25
    assert out["usable_scale_range"] is None
    assert out["works"] is False


def test_compute_scale_calibration_baseline_row_never_a_candidate():
    """scale=0.0 is the shared baseline row -- even if it somehow carried a nonzero logprob_delta, it must
    never be treated as a usable dose (usable_scales filters scale > 0)."""
    curve = [{"scale": 0.0, "logprob_delta": 99.0, "degenerate_rate": 0.0}]
    out = cdac._compute_scale_calibration(curve, degen_threshold=0.34, logprob_rise_min=2.0)
    assert out["usable_scale_range"] is None
    assert out["works"] is False


# ==================================================================================== degenerate_rate (reused gate)

def test_degenerate_rate_empty_list_is_zero():
    assert cdac.degenerate_rate([]) == 0.0


def test_degenerate_rate_flags_repeat_3gram():
    assert cdac.degenerate_rate(["loop loop loop forever"]) == 1.0


def test_degenerate_rate_coherent_text_is_zero():
    assert cdac.degenerate_rate(["A perfectly ordinary sentence about the weather today."]) == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
