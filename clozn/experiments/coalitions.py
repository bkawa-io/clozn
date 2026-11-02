"""Text-blind, reproducible coalition samples for exploratory attribution."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
import random


def shapley_pairs(
    source_ids: Sequence[str], *, seed: int,
) -> Iterator[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Yield (source, without_source, with_source), in canonical prompt order.

    Visit sources in shuffled rounds for coverage. For each source independently,
    draw k uniformly from 0..n-1, then a uniform k-subset of the other sources.
    This is the distribution of predecessors in a uniform random permutation:
    each subset has weight 1 / (n * choose(n-1, k)). Independent 50/50 inclusion
    instead targets Banzhaf, not Shapley. Sampling never inspects source text.

    These are sampling draws, not certified Shapley values. In particular, a
    compute-budget stop with cache-dependent costs is not a fixed sample count.
    """
    if isinstance(source_ids, (str, bytes)):
        raise ValueError("source_ids must be a sequence of IDs, not text")
    ids = tuple(source_ids)
    if not ids or any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("source_ids must be non-empty, unique strings")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    while True:
        order = list(ids)
        rng.shuffle(order)
        for source_id in order:
            others = [sid for sid in ids if sid != source_id]
            chosen = set(rng.sample(others, rng.randrange(len(ids))))
            without = tuple(sid for sid in ids if sid in chosen)
            chosen.add(source_id)
            yield source_id, without, tuple(sid for sid in ids if sid in chosen)
