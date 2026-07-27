"""Tests for the confidence-select CPU reference (DESIGN.md §4.3).

Two layers:

* Self-contained unit tests of the kernel contract (shapes, greedy, top_p,
  confidence variants, selection) — numpy only, no torch, no clozn_lab.
* PARITY tests that import ``clozn_lab`` (TEST-ONLY) and assert the reference
  reproduces ``generate.sample_candidates`` (token ids + confidences, greedy AND
  sampled, seeding both sides identically) and the ``ConfidenceTopK`` /
  ``Threshold`` selection policies exactly.

Run (from this directory, with the project venv active):
    python -m pytest test_reference.py -q

Parity is one-directional: if a parity test fails, the *reference* is wrong and
must be fixed to match the lab — never the reverse.
"""

from __future__ import annotations

import numpy as np
import pytest

from reference import (
    ConfidenceSelectResult,
    confidence_select,
    host_transfer_bytes,
)

VOCAB = 64
N_MASKED = 12


def _random_logits(seed: int = 0, n: int = N_MASKED, vocab: int = VOCAB) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, vocab)).astype(np.float32)


# --------------------------------------------------------------------------- #
# (a) contract / shape tests
# --------------------------------------------------------------------------- #


def test_result_shapes_and_types() -> None:
    logits = _random_logits()
    res = confidence_select(logits, k_commit=3)
    assert isinstance(res, ConfidenceSelectResult)
    assert res.token_ids.shape == (N_MASKED,)
    assert res.confidences.shape == (N_MASKED,)
    assert res.token_ids.dtype == np.int64
    assert res.confidences.dtype == np.float64
    assert isinstance(res.selected, list)
    assert all(isinstance(i, int) for i in res.selected)
    # token ids are valid vocab indices
    assert res.token_ids.min() >= 0 and res.token_ids.max() < VOCAB


def test_selected_is_sorted_ascending_and_subset() -> None:
    logits = _random_logits(seed=1)
    res = confidence_select(logits, k_commit=5)
    assert res.selected == sorted(res.selected)
    assert len(res.selected) == 5
    assert set(res.selected) <= set(range(N_MASKED))


def test_exactly_one_of_k_or_tau_required() -> None:
    logits = _random_logits()
    with pytest.raises(ValueError):
        confidence_select(logits)  # neither
    with pytest.raises(ValueError):
        confidence_select(logits, k_commit=2, tau=0.5)  # both


def test_sampling_requires_rng() -> None:
    logits = _random_logits()
    with pytest.raises(ValueError):
        confidence_select(logits, temperature=1.0, k_commit=2)  # no rng


def test_bad_confidence_variant_rejected() -> None:
    logits = _random_logits()
    with pytest.raises(ValueError):
        confidence_select(logits, k_commit=2, confidence="entropy")


def test_logits_not_mutated() -> None:
    logits = _random_logits(seed=7)
    before = logits.copy()
    confidence_select(logits, top_p=0.5, k_commit=3)
    np.testing.assert_array_equal(logits, before)


def test_host_transfer_bytes() -> None:
    # 2 values (int + float) per masked position, 4 bytes each.
    assert host_transfer_bytes(0) == 0
    assert host_transfer_bytes(10) == 2 * 10 * 4
    # ~10,000x smaller than the naive full-logits transfer for a real vocab.
    naive = N_MASKED * 50_000 * 4
    assert host_transfer_bytes(N_MASKED) * 10_000 < naive * 2  # order-of-magnitude check
    with pytest.raises(ValueError):
        host_transfer_bytes(-1)


# --------------------------------------------------------------------------- #
# (b) greedy: token == argmax, confidence == softmax prob of the pick
# --------------------------------------------------------------------------- #


def test_greedy_token_is_argmax_and_conf_is_softmax_prob() -> None:
    logits = _random_logits(seed=2)
    res = confidence_select(logits, temperature=0.0, k_commit=4)

    # token == row argmax
    np.testing.assert_array_equal(res.token_ids, logits.argmax(axis=1))

    # confidence == float64 softmax prob of the picked token
    x = logits.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    probs = np.exp(x)
    probs /= probs.sum(axis=1, keepdims=True)
    expected = probs[np.arange(N_MASKED), res.token_ids]
    np.testing.assert_allclose(res.confidences, expected, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# (c) top_p actually restricts support
# --------------------------------------------------------------------------- #


def test_top_p_greedy_pick_respects_nucleus() -> None:
    # One row, one dominant token: with a tiny top_p only that token survives,
    # and the greedy pick must be it (it is also the argmax here).
    logits = np.array([[0.0, 5.0, 0.1, 0.2]], dtype=np.float32)
    res = confidence_select(logits, top_p=0.5, temperature=0.0, k_commit=1)
    assert res.token_ids[0] == 1


def test_top_p_never_samples_outside_nucleus() -> None:
    # Token 0 dominates; with top_p just above its mass, only token 0 (and the
    # token that first crosses the threshold) survive. We make token 0's prob
    # alone exceed top_p so nothing else may ever be drawn.
    logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(123)
    drawn = set()
    for _ in range(500):
        res = confidence_select(logits, top_p=0.5, temperature=1.0, k_commit=1, rng=rng)
        drawn.add(int(res.token_ids[0]))
    # Only the dominant token (whose prob alone > top_p) can be sampled.
    assert drawn == {0}


def test_top_p_keeps_token_that_first_crosses_threshold() -> None:
    # Mirror open-dCoder: the first token whose cumulative mass crosses top_p is
    # KEPT (mask shifted right by one). Two near-equal tokens, top_p between one
    # and two of them -> both should remain samplable.
    logits = np.array([[2.0, 1.99, -20.0, -20.0]], dtype=np.float32)
    rng = np.random.default_rng(7)
    drawn = set()
    for _ in range(800):
        res = confidence_select(logits, top_p=0.6, temperature=1.0, k_commit=1, rng=rng)
        drawn.add(int(res.token_ids[0]))
    assert drawn == {0, 1}  # the far-negative tokens are filtered out


# --------------------------------------------------------------------------- #
# (d) margin and neg_entropy confidence variants
# --------------------------------------------------------------------------- #


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def test_margin_confidence_is_top1_minus_top2() -> None:
    logits = _random_logits(seed=3)
    res = confidence_select(logits, k_commit=4, confidence="margin")
    probs = _softmax(logits)
    srt = np.sort(probs, axis=1)[:, ::-1]
    expected = srt[:, 0] - srt[:, 1]
    np.testing.assert_allclose(res.confidences, expected, rtol=0, atol=0)


def test_neg_entropy_confidence_matches_sum_p_logp() -> None:
    logits = _random_logits(seed=4)
    res = confidence_select(logits, k_commit=4, confidence="neg_entropy")
    probs = _softmax(logits)
    log_probs = np.log(np.clip(probs, 1e-10, None))
    expected = (probs * log_probs).sum(axis=1)
    np.testing.assert_allclose(res.confidences, expected, rtol=0, atol=0)
    # negative entropy is <= 0 for proper distributions
    assert (res.confidences <= 1e-9).all()


def test_neg_entropy_more_peaked_is_more_confident() -> None:
    peaked = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    flat = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    c_peaked = confidence_select(peaked, k_commit=1, confidence="neg_entropy").confidences[0]
    c_flat = confidence_select(flat, k_commit=1, confidence="neg_entropy").confidences[0]
    assert c_peaked > c_flat  # closer to 0 = more confident


# --------------------------------------------------------------------------- #
# (e) selection: fixed k_commit and tau path with min_commit rail
# --------------------------------------------------------------------------- #


def test_topk_picks_exactly_k_highest_confidence() -> None:
    # Hand-built confidences via one-hot-ish logits so the order is unambiguous.
    confs = [0.9, 0.1, 0.5, 0.7, 0.3]
    logits = _logits_with_max_probs(confs)
    res = confidence_select(logits, k_commit=3)
    # top-3 by confidence are positions 0 (.9), 3 (.7), 2 (.5) -> ascending
    assert res.selected == [0, 2, 3]


def test_topk_tie_breaks_toward_lower_index() -> None:
    # Two positions with identical confidence; k=1 must pick the lower index.
    confs = [0.6, 0.6, 0.2]
    logits = _logits_with_max_probs(confs)
    res = confidence_select(logits, k_commit=1)
    assert res.selected == [0]
    # k=2 picks both tied (0,1), never position 2.
    res2 = confidence_select(logits, k_commit=2)
    assert res2.selected == [0, 1]


def test_topk_clamps_to_n_masked() -> None:
    logits = _random_logits(seed=5, n=4)
    res = confidence_select(logits, k_commit=99)
    assert res.selected == [0, 1, 2, 3]


def test_threshold_commits_those_at_or_above_tau() -> None:
    # tau sits in a gap between the target probs (0.4 < 0.75 < 0.8) so the
    # selection is robust to the float32 round-trip of _logits_with_max_probs.
    confs = [0.9, 0.4, 0.8, 0.2, 0.85]
    logits = _logits_with_max_probs(confs)
    res = confidence_select(logits, tau=0.75, min_commit=1)
    assert res.selected == [0, 2, 4]


def test_threshold_min_commit_rail_forces_top_when_none_clear() -> None:
    confs = [0.3, 0.1, 0.25]
    logits = _logits_with_max_probs(confs)
    # nothing clears 0.9; rail forces the single most confident (pos 0).
    res = confidence_select(logits, tau=0.9, min_commit=1)
    assert res.selected == [0]
    # min_commit=2 forces the top two by confidence (0 -> .3, 2 -> .25).
    res2 = confidence_select(logits, tau=0.9, min_commit=2)
    assert res2.selected == [0, 2]


def _logits_with_max_probs(target_probs: list[float], vocab: int = VOCAB) -> np.ndarray:
    """Build logits whose per-row softmax max-prob equals each target.

    For a row we want argmax prob == p: put logit ``log(p)`` on the winner and
    spread the remaining ``1 - p`` over the other ``vocab - 1`` tokens, each
    ``log((1 - p) / (vocab - 1))``. Softmax of these logs reproduces the
    distribution exactly, so max_prob confidence == p.
    """
    n = len(target_probs)
    logits = np.empty((n, vocab), dtype=np.float64)
    rest = vocab - 1
    for i, p in enumerate(target_probs):
        logits[i, :] = np.log((1.0 - p) / rest)
        logits[i, 0] = np.log(p)  # winner is token 0 in every row
    return logits.astype(np.float32)


# --------------------------------------------------------------------------- #
# (f) PARITY tests against clozn_lab (TEST-ONLY imports)
# --------------------------------------------------------------------------- #

from clozn_lab.generate import sample_candidates  # noqa: E402
from clozn_lab.scheduler.policies import (  # noqa: E402
    Candidate,
    ConfidenceTopK,
    StepContext,
    Threshold,
)


def _candidates_from(res: ConfidenceSelectResult) -> list[Candidate]:
    """Lab Candidates from a reference result (pos == row index, max_prob conf)."""
    return [
        Candidate(pos=i, token_id=int(res.token_ids[i]), confidence=float(res.confidences[i]))
        for i in range(res.token_ids.shape[0])
    ]


def test_parity_greedy_tokens_and_confidences() -> None:
    logits = _random_logits(seed=11)
    positions = list(range(logits.shape[0]))

    lab = sample_candidates(logits, positions, temperature=0.0)
    ref = confidence_select(logits, temperature=0.0, k_commit=1)

    lab_tokens = np.array([c.token_id for c in lab], dtype=np.int64)
    lab_conf = np.array([c.confidence for c in lab], dtype=np.float64)
    np.testing.assert_array_equal(ref.token_ids, lab_tokens)
    np.testing.assert_array_equal(ref.confidences, lab_conf)  # EXACT, same float64 path


def test_parity_sampled_tokens_and_confidences() -> None:
    logits = _random_logits(seed=12)
    positions = list(range(logits.shape[0]))
    temperature = 0.8

    # Seed BOTH sides identically; sample_candidates and the reference both loop
    # rng.choice(vocab, p=row) in row order, so the draw sequences must match.
    lab = sample_candidates(
        logits, positions, temperature=temperature, rng=np.random.default_rng(99)
    )
    ref = confidence_select(
        logits, temperature=temperature, k_commit=1, rng=np.random.default_rng(99)
    )

    lab_tokens = np.array([c.token_id for c in lab], dtype=np.int64)
    lab_conf = np.array([c.confidence for c in lab], dtype=np.float64)
    np.testing.assert_array_equal(ref.token_ids, lab_tokens)
    np.testing.assert_array_equal(ref.confidences, lab_conf)


def test_parity_topk_selection_fixed_k() -> None:
    logits = _random_logits(seed=13)
    K = 5
    ref = confidence_select(logits, temperature=0.0, k_commit=K)

    cands = _candidates_from(ref)
    sel = ConfidenceTopK(k=K).select(cands, StepContext(step=0, steps_total=8))
    lab_positions = sorted(c.pos for c in sel.commit)
    assert ref.selected == lab_positions


def test_parity_topk_quota_mode_matches_precomputed_k() -> None:
    # The quota-ramp k=None case computes k from step context OUTSIDE the kernel.
    # Compute the same k the policy would (ceil(n / steps_remaining)) and pass it.
    logits = _random_logits(seed=14)
    n = logits.shape[0]
    ctx = StepContext(step=2, steps_total=8)
    steps_remaining = ctx.steps_remaining
    assert steps_remaining is not None
    k = -(-n // steps_remaining)  # ceil(n / steps_remaining)

    ref = confidence_select(logits, temperature=0.0, k_commit=k)
    cands = _candidates_from(ref)
    sel = ConfidenceTopK(k=None).select(cands, ctx)  # quota mode resolves k internally
    lab_positions = sorted(c.pos for c in sel.commit)
    assert ref.selected == lab_positions


def test_parity_threshold_selection() -> None:
    logits = _random_logits(seed=15)
    tau, min_commit = 0.15, 1
    ref = confidence_select(logits, temperature=0.0, tau=tau, min_commit=min_commit)

    cands = _candidates_from(ref)
    sel = Threshold(tau=tau, min_commit=min_commit).select(cands, StepContext(step=0))
    lab_positions = sorted(c.pos for c in sel.commit)
    assert ref.selected == lab_positions


def test_parity_threshold_rail_path() -> None:
    # Pick a tau no position can clear so the min_commit rail decides — must still
    # match the lab's rail (top min_commit by confidence, ties toward lower pos).
    logits = _random_logits(seed=16)
    tau, min_commit = 0.999, 3
    ref = confidence_select(logits, temperature=0.0, tau=tau, min_commit=min_commit)

    cands = _candidates_from(ref)
    sel = Threshold(tau=tau, min_commit=min_commit).select(cands, StepContext(step=0))
    lab_positions = sorted(c.pos for c in sel.commit)
    assert ref.selected == lab_positions
    assert len(ref.selected) == min_commit
