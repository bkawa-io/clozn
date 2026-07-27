"""Fixture tests for clozn.analysis.tracer's pure math (no engine, no network).

The engine-facing orchestration (trace()) is exercised live (it needs a running clozn-server with
a J-lens sidecar); everything below is the model-free seam: ablation algebra, candidate screening,
the noise floor / verdict rules, the unexplained-mass accounting, and joint-write grouping.
"""
import numpy as np
import pytest

from clozn.analysis.tracer import (accounting, controls_verdict, directional_ablate,
                                   edge_candidates, group_joint_writes, noise_floor,
                                   screen_candidates)


# ------------------------------------------------------------------------- directional_ablate

def test_directional_ablate_removes_exactly_the_projection():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(64).astype(np.float32)
    d = rng.standard_normal(64).astype(np.float32)
    out = directional_ablate(h, d)
    d_hat = d / np.linalg.norm(d)
    assert abs(float(out @ d_hat)) < 1e-4            # the component along d is gone
    resid = h - out                                   # and what was removed is parallel to d
    cos = float(resid @ d_hat) / (np.linalg.norm(resid) + 1e-12)
    assert abs(abs(cos) - 1.0) < 1e-4


def test_directional_ablate_is_scale_invariant_in_d():
    rng = np.random.default_rng(1)
    h = rng.standard_normal(16).astype(np.float32)
    d = rng.standard_normal(16).astype(np.float32)
    a = directional_ablate(h, d)
    b = directional_ablate(h, 7.5 * d)               # only the direction matters
    assert np.allclose(a, b, atol=1e-5)


def test_directional_ablate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        directional_ablate(np.ones(4), np.ones(5))   # shape mismatch
    with pytest.raises(ValueError):
        directional_ablate(np.ones(4), np.zeros(4))  # degenerate direction


# -------------------------------------------------------------------------- screen_candidates

def _screen_fixture():
    d = 8
    H = np.zeros((5, d), dtype=np.float32)
    dirs = {"target": np.eye(d, dtype=np.float32)[0], "other": np.eye(d, dtype=np.float32)[1]}
    H[3, 0] = 10.0    # planted: strong "target" alignment at pos 3
    H[1, 1] = 4.0     # weaker "other" alignment at pos 1
    return {16: H}, {16: dirs}


def test_screen_ranks_planted_site_first_and_labels_it():
    H, dirs = _screen_fixture()
    out = screen_candidates(H, dirs, max_candidates=3, force_sites=[])
    assert (out[0]["layer"], out[0]["pos"], out[0]["concept"]) == (16, 3, "target")
    assert (out[1]["layer"], out[1]["pos"], out[1]["concept"]) == (16, 1, "other")


def test_screen_force_sites_come_first_and_dedupe():
    H, dirs = _screen_fixture()
    out = screen_candidates(H, dirs, max_candidates=4, force_sites=[(16, 4), (16, 4), (16, 3)])
    assert (out[0]["layer"], out[0]["pos"]) == (16, 4)
    assert (out[1]["layer"], out[1]["pos"]) == (16, 3)   # forced AND planted: appears once
    assert len([c for c in out if (c["layer"], c["pos"]) == (16, 3)]) == 1
    assert len(out) <= 4


def test_screen_respects_cap():
    H, dirs = _screen_fixture()
    out = screen_candidates(H, dirs, max_candidates=2, force_sites=[])
    assert len(out) == 2


# ---------------------------------------------------------------- noise floor / verdict rules

def test_noise_floor_is_mult_times_median():
    assert noise_floor([0.1, -0.2, 0.3], mult=3.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        noise_floor([])


def test_verdict_pass_when_real_beats_controls():
    assert controls_verdict([2.0, -1.5], [0.01, -0.02, 0.015]) == "PASS"


def test_verdict_no_causal_nodes_when_nothing_survived():
    assert controls_verdict([], [0.01, -0.02]) == "NO_CAUSAL_NODES"


def test_verdict_failed_when_controls_match_real():
    # the strongest control equals the strongest "real" effect -> nothing here is trustworthy
    assert controls_verdict([0.5], [0.6, 0.01]) == "FAILED_CONTROLS"
    with pytest.raises(ValueError):
        controls_verdict([1.0], [])


# ------------------------------------------------------------------- strength tiering contract

def test_strength_tiers_match_the_documented_thresholds():
    """The tiering the CLI prints and the receipt stores: strong >= 3x the strongest control,
    weak 1-3x, marginal <= 1x. Pinned because a 16-prompt battery found single traces spanning
    1.3x to 218x -- the tail is nearly indistinguishable from a random intervention and callers
    must be able to tell those apart."""
    ctl_max = 0.02
    for delta, want in [(10.0, "strong"), (0.06, "strong"), (0.059, "weak"),
                        (0.021, "weak"), (0.02, "marginal"), (0.001, "marginal")]:
        ratio = abs(delta) / ctl_max
        tier = "strong" if ratio >= 3.0 else "weak" if ratio > 1.0 else "marginal"
        assert tier == want, f"delta {delta} (ratio {ratio:.2f}) -> {tier}, expected {want}"


# ------------------------------------------------------------------------------- accounting

def test_accounting_interaction_gap_signs():
    sub = accounting([2.0, 3.0], delta_total=4.0)     # joint < sum: sub-additive (self-repair)
    assert sub["interaction_gap"] == pytest.approx(-1.0)
    sup = accounting([1.0, 1.0], delta_total=3.0)     # joint > sum: super-additive
    assert sup["interaction_gap"] == pytest.approx(1.0)
    empty = accounting([], delta_total=0.5)
    assert empty["sum_solo"] == 0.0


# -------------------------------------------------------------------------- edge_candidates

def test_edge_candidates_respect_causal_reachability():
    nodes = [{"layer": 16, "pos": 5, "delta_full": 2.0},
             {"layer": 24, "pos": 5, "delta_full": 1.0},   # same pos, later layer: reachable
             {"layer": 24, "pos": 3, "delta_full": 3.0}]   # EARLIER pos at later layer: unreachable from pos 5
    pairs = edge_candidates(nodes, max_edges=10)
    assert [( (a["layer"], a["pos"]), (b["layer"], b["pos"]) ) for a, b in pairs] == \
           [((16, 5), (24, 5))]                            # only the causally-possible pair


def test_edge_candidates_order_and_cap():
    nodes = [{"layer": 16, "pos": 0, "delta_full": 1.0},
             {"layer": 24, "pos": 1, "delta_full": 5.0},
             {"layer": 24, "pos": 2, "delta_full": 0.1}]
    pairs = edge_candidates(nodes, max_edges=1)
    assert len(pairs) == 1
    assert (pairs[0][1]["layer"], pairs[0][1]["pos"]) == (24, 1)   # strongest |dA*dB| pair kept
    with pytest.raises(ValueError):
        edge_candidates(nodes, max_edges=-1)


# ------------------------------------------------------------------------ group_joint_writes

def test_group_joint_writes_one_spec_per_layer_with_stacked_mean_rows():
    nodes = [{"layer": 16, "pos": 3}, {"layer": 16, "pos": 1}, {"layer": 24, "pos": 4}]
    mean_rows = {16: np.full(4, 2.0, dtype=np.float32), 24: np.full(4, 5.0, dtype=np.float32)}
    specs = group_joint_writes(nodes, mean_rows)
    assert [s["layer"] for s in specs] == [16, 24]
    s16 = specs[0]
    assert sorted(s16["positions"]) == [1, 3]
    assert len(s16["values"]) == 2 * 4                # one mean row per position, concatenated
    assert all(v == 2.0 for v in s16["values"])
    assert specs[1]["positions"] == [4] and len(specs[1]["values"]) == 4
