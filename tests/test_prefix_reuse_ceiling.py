"""Model-free tests for the exact-token prefix ceiling diagnostic."""

from scripts.bench.prefix_reuse_ceiling import _pack_resident, exact_trie_rows


def test_exact_trie_rows_counts_shared_edges_only():
    assert exact_trie_rows([(1, 2, 3, 4, 10),
                            (1, 2, 3, 4, 11),
                            (1, 2, 3, 5, 12)]) == 8
    # The equal suffix is not re-merged after the X/Y histories diverge.
    assert exact_trie_rows([(1, 2, 10, 20, 30),
                            (1, 2, 11, 20, 30)]) == 8
    assert exact_trie_rows([]) == 0


def test_resident_packing_matches_arm_and_context_limits():
    prompts = [(index,) for index in range(33)]
    batches = _pack_resident(prompts, n_ctx=4)
    assert [len(batch) for batch in batches] == [4] * 8 + [1]

    long_prompts = [(1, 2), (3, 4), (5, 6)]
    batches = _pack_resident(long_prompts, n_ctx=4)
    assert batches == [[(1, 2), (3, 4)], [(5, 6)]]
