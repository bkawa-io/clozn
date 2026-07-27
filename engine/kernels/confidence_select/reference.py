"""Confidence-select kernel — CPU numpy reference oracle (Clozn DESIGN.md §4.3).

This is the *tested* reference path for the one new kernel Clozn ships. It is
self-contained numpy (it deliberately does NOT import ``clozn_lab``) so the
correctness contract lives independently of the lab and can later validate a
Metal/CUDA port the same way the golden tests validate the scheduler.

Contract (DESIGN.md §4.3, verbatim):

    inputs : logits buffer [n_masked, vocab] (device-resident),
             temperature, top_p, k_commit (or threshold tau), rng state
    outputs: per masked position -> (sampled_token_id, confidence) ;
             plus the indices of the top-k_commit positions by confidence
    transfer to host: 2 x n_masked ints + floats   (~10,000x smaller)

Why this kernel exists (the motivation it fuses away):

    The naive denoise loop computes logits for every masked position and ships
    the full ``[n_masked x vocab]`` float buffer GPU->CPU every step. The
    upstream llama.cpp diffusion PR measured that transfer at **~87% of GPU
    wall time**. Fusing sample + confidence + selection device-side means only
    ``2 * n_masked`` ints/floats cross the bus per step (a token id and a
    confidence per masked position), which the host then turns into commits.

Confidence = probability of the sampled token after temperature/top-p scaling.
Default "max_prob"; "margin" (top1 - top2 prob) and "neg_entropy"
(sum p*log p, already negative) are selectable — DESIGN open question #3.

Semantics are pinned to two lab sources, mirrored exactly (see test_reference.py
for the parity assertions):

* ``clozn_lab.generate.sample_candidates`` — the sample + max_prob confidence
  step. Greedy (temperature == 0): token = argmax, confidence = its softmax
  prob. Sampled (temperature > 0): draw from softmax(logits / T) with the rng;
  confidence = the drawn token's post-temperature probability. The softmax is
  computed in float64. NOTE: ``sample_candidates`` does NOT apply top_p, so the
  greedy/sampled *parity* path here also leaves top_p unset.
* ``clozn_lab.scheduler.policies`` — the selection step. ``ConfidenceTopK(k)``
  commits ``min(k, n_masked)`` highest-confidence positions, ties broken toward
  the LOWER index (``sorted(key=(-conf, pos))``). ``Threshold(tau, min_commit)``
  commits every position with ``conf >= tau``; if fewer than ``min_commit``
  clear tau, it instead commits the ``min_commit`` most confident (the
  min-one-commit progress rail). The selected indices are returned ascending.

The open-dCoder ``generation_utils.top_p_logits`` is mirrored for the
nucleus-filtering path (sort desc, cumulative softmax, drop the tokens *after*
the one that first crosses top_p, always keep the top-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

# Confidence variants (DESIGN §5.2 / open question #3). Mirror open-dCoder's
# sample_tokens flags: max_prob is the gathered prob of the pick; margin is
# top1 - top2 prob; neg_entropy is sum_v p_v * log p_v (already negative).
_CONFIDENCE_VARIANTS = ("max_prob", "margin", "neg_entropy")

# Matches open-dCoder's probs.clamp(min=1e-10) before log, so neg_entropy never
# hits log(0) on a filtered/zero-probability token.
_ENTROPY_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class ConfidenceSelectResult:
    """The kernel's host-side payload for one denoise step.

    Mirrors the §4.3 transfer: a token id and a confidence per masked position
    (the per-row arrays), plus the indices the host should commit this step.

    Attributes:
        token_ids: shape ``(n_masked,)`` int64; the sampled token per position.
        confidences: shape ``(n_masked,)`` float64; the selected confidence
            variant per position (max_prob by default).
        selected: row indices (into ``0..n_masked-1``) to commit this step,
            sorted ascending — the positions a top-k / threshold policy picks.
    """

    token_ids: IntArray
    confidences: FloatArray
    selected: list[int]


def host_transfer_bytes(n_masked: int) -> int:
    """Bytes crossing GPU->host per step under the fused kernel (§4.3).

    Two values per masked position survive the fusion: a token id (int) and a
    confidence (float). At 4 bytes each that is ``2 * n_masked * 4`` bytes,
    versus ``n_masked * vocab * 4`` for the naive full-logits transfer — the
    ~10,000x reduction the design quotes (vocab ~= 32k-150k for real dLLMs).
    """
    if n_masked < 0:
        raise ValueError(f"n_masked must be >= 0, got {n_masked}")
    bytes_per_value = 4  # int32 token id + float32 confidence
    return 2 * n_masked * bytes_per_value


def _top_p_filter(logits: FloatArray, top_p: float) -> FloatArray:
    """Nucleus filtering, mirroring open-dCoder ``top_p_logits``.

    Sort each row descending, take the cumulative softmax mass, and remove every
    token whose *predecessors* already reached ``top_p`` (the mask is shifted
    right by one so the token that first crosses the threshold is kept; the
    top-1 is always kept). Removed logits are set to the float min so they get
    ~0 probability after the subsequent softmax.

    Operates row-wise on a copy; returns a new array.
    """
    out = logits.copy()
    order = np.argsort(-out, axis=1, kind="stable")  # descending per row
    sorted_logits = np.take_along_axis(out, order, axis=1)

    # Softmax over the sorted row (stable: subtract the row max, which is col 0).
    shifted = sorted_logits - sorted_logits[:, :1]
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)

    remove_sorted = cumulative > top_p
    # Shift right by one so the first token to cross the threshold is retained,
    # and force-keep the top-1 (exactly open-dCoder's two index lines).
    remove_sorted[:, 1:] = remove_sorted[:, :-1].copy()
    remove_sorted[:, 0] = False

    # Scatter the per-rank removal flags back to vocab positions.
    remove = np.zeros_like(remove_sorted)
    np.put_along_axis(remove, order, remove_sorted, axis=1)

    out[remove] = np.finfo(out.dtype).min
    return out


def _softmax_rows(logits: FloatArray) -> FloatArray:
    """Row-wise softmax in float64 (matches ``sample_candidates`` precision)."""
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def _confidences(probs: FloatArray, token_ids: IntArray, variant: str) -> FloatArray:
    """Per-row confidence under the requested variant (mirrors open-dCoder).

    * ``max_prob``: probability of the sampled token (the gathered pick prob).
    * ``margin``: top1 - top2 probability over the row.
    * ``neg_entropy``: sum_v p_v * log p_v (already negative; higher = more
      peaked = more confident), with the same 1e-10 clamp open-dCoder uses.
    """
    n = probs.shape[0]
    if variant == "max_prob":
        return probs[np.arange(n), token_ids]
    if variant == "margin":
        # Partition is enough for the two largest; sort the top slice for top1/top2.
        sorted_desc = np.sort(probs, axis=1)[:, ::-1]
        top1 = sorted_desc[:, 0]
        top2 = sorted_desc[:, 1] if probs.shape[1] > 1 else np.zeros(n)
        return top1 - top2
    if variant == "neg_entropy":
        log_probs = np.log(np.clip(probs, _ENTROPY_EPS, None))
        return (probs * log_probs).sum(axis=1)
    raise ValueError(f"confidence must be one of {_CONFIDENCE_VARIANTS}, got {variant!r}")


def _select_topk(confidences: FloatArray, k_commit: int) -> list[int]:
    """Indices of the ``k`` highest-confidence rows, ties toward the lower index.

    Matches ``policies._by_confidence`` (sort by ``(-confidence, pos)``) followed
    by ``[:take]`` with ``take = min(k, n)``; the committed indices are then
    returned ascending (``policies._commit`` sorts by pos).
    """
    n = confidences.shape[0]
    take = min(k_commit, n)
    # Stable sort on the *negated* confidence keeps original (lower) index order
    # among ties — exactly the (-conf, pos) ordering the lab uses.
    order = np.argsort(-confidences, kind="stable")
    chosen = order[:take]
    return sorted(int(i) for i in chosen)


def _select_threshold(confidences: FloatArray, tau: float, min_commit: int) -> list[int]:
    """Indices with ``conf >= tau``; else the ``min_commit`` most confident.

    Mirrors ``policies.Threshold.select``: commit everything clearing tau, but if
    fewer than ``min_commit`` clear it, fall back to the top ``min_commit`` by
    confidence (the min-one-commit progress rail). Returned ascending.
    """
    above = [int(i) for i in np.flatnonzero(confidences >= tau)]
    if len(above) >= min_commit:
        return sorted(above)
    return _select_topk(confidences, min_commit)


def confidence_select(
    logits: NDArray[np.floating],
    *,
    temperature: float = 0.0,
    top_p: float | None = None,
    k_commit: int | None = None,
    tau: float | None = None,
    min_commit: int = 1,
    confidence: str = "max_prob",
    rng: np.random.Generator | None = None,
) -> ConfidenceSelectResult:
    """Fused sample + confidence + selection over masked-position logits (§4.3).

    Args:
        logits: ``[n_masked, vocab]`` logits, one row per masked position. Read,
            never mutated (filtering operates on a copy).
        temperature: 0 => greedy (argmax). > 0 => scale logits by ``1/T`` and
            sample from the resulting softmax with ``rng``.
        top_p: nucleus threshold in (0, 1]; ``None`` (or ``>= 1``) disables it.
            Applied after temperature, before softmax — mirrors open-dCoder.
        k_commit: fixed top-k selection — commit the ``min(k, n_masked)`` highest
            confidence positions. Exactly one of ``k_commit`` / ``tau`` is given.
            (The policy's quota-ramp ``k=None`` case is resolved to a concrete k
            by the caller using step context; pass that already-computed value.)
        tau: threshold selection — commit positions with ``conf >= tau``, with a
            ``min_commit`` rail. Exactly one of ``k_commit`` / ``tau`` is given.
        min_commit: min positions to force when fewer than ``min_commit`` clear
            ``tau`` (threshold path only). Must be >= 1.
        confidence: "max_prob" (default), "margin", or "neg_entropy".
        rng: required when ``temperature > 0`` (sampling); unused when greedy.

    Returns:
        ConfidenceSelectResult with per-position token ids and confidences and
        the ascending list of selected (to-commit) row indices.
    """
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [n_masked, vocab], got shape {logits.shape}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature > 0 and rng is None:
        raise ValueError("sampling (temperature > 0) requires an rng")
    if confidence not in _CONFIDENCE_VARIANTS:
        raise ValueError(f"confidence must be one of {_CONFIDENCE_VARIANTS}, got {confidence!r}")
    if (k_commit is None) == (tau is None):
        raise ValueError("provide exactly one of k_commit / tau")
    if k_commit is not None and k_commit < 1:
        raise ValueError(f"k_commit must be >= 1, got {k_commit}")
    if tau is not None and not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    if min_commit < 1:
        raise ValueError(f"min_commit must be >= 1, got {min_commit}")

    n_masked, vocab = logits.shape

    # Everything downstream runs in float64 to match sample_candidates' precision
    # (its softmax is float64; bitwise parity on the pick depends on this).
    x = logits.astype(np.float64)
    if temperature > 0:
        x = x / temperature
    if top_p is not None and top_p < 1.0:
        x = _top_p_filter(x, top_p)

    probs = _softmax_rows(x)

    token_ids = np.empty(n_masked, dtype=np.int64)
    if temperature == 0:
        token_ids[:] = probs.argmax(axis=1)
    else:
        assert rng is not None  # guarded above
        # Draw per row with the rng, mirroring sample_candidates' loop exactly
        # (rng.choice(vocab, p=row)) so the draw sequence is identical.
        for row in range(n_masked):
            token_ids[row] = rng.choice(vocab, p=probs[row])

    confidences = _confidences(probs, token_ids, confidence)

    if k_commit is not None:
        selected = _select_topk(confidences, k_commit)
    else:
        assert tau is not None  # guarded by the xor check above
        selected = _select_threshold(confidences, tau, min_commit)

    return ConfidenceSelectResult(
        token_ids=token_ids,
        confidences=confidences,
        selected=selected,
    )
