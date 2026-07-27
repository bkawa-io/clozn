"""test_timetravel_determinism -- the DETERMINISM RECEIPT for the time-travel debugger.

The load-bearing scientific claim v1 owes: a branch (rewind to an earlier turn from a CPU-offloaded KV
SNAPSHOT, splice an alternate user turn, continue WITHOUT re-prefilling the shared past) reproduces a
FRESH full recompute of the identical transcript BYTE-FOR-BYTE. This is what makes "rewind & branch"
trustworthy -- the branched future is EXACTLY the future you'd have gotten typing it fresh, not a drifted
approximation.

kv_timetravel.py already PROVED this at the mechanism level (Phase 1: byte-identical at depth 2/5/10). This
test re-proves it THROUGH the product's own store: each turn's KV is offloaded via timetravel.offload_cache
into a real timetravel.SnapshotStore (the bounded ring), and the branch RESTORES from the store's payload
-- so the receipt covers the store's snapshot round-trip, not just KVChat's internal one. The rewind point
+ the alt-user substitution come from timetravel.branch_messages (the product transform).

Branch semantics: to branch AT conversational turn `t` (i.e. replace turn t's user message and continue),
rewind to the KV state right AFTER turn t-1 -- the snapshot at ring index t-1 -- then append the alt user
as a fresh turn. The fresh reference recomputes base_users[:t] + [alt] from an empty cache.

Gated behind -m model (loads Qwen2.5-1.5B on the GPU; ~1 min): mirrors test_prompt_vs_prefix_ab.py. Skips
cleanly with no CUDA. Pins the same stack kv_timetravel.py pins (cache internals are not a stable contract).

Run as the gated test:
    C:/Users/brigi/src/clozn/.venv/Scripts/python.exe -m pytest \
        research/tests/test_timetravel_determinism.py -m model -q
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.replay.timetravel as tt  # noqa: E402  (stdlib-only import; torch is lazy)


def _lazy():
    """The heavy rig, imported only when a run needs it (so plain collection stays model-free)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    import kv_timetravel as kvt                          # the proven KVChat harness + fresh_recompute
    return torch, AutoModelForCausalLM, AutoTokenizer, DynamicCache, kvt


def _restore_cache_from(kv_cpu, DynamicCache, dev):
    """Rebuild an on-device DynamicCache from a store snapshot's offloaded (k,v)-per-layer payload -- the
    same construction KVChat._restore_cache_from uses, but reading the STORE's tensors (the round-trip
    under test)."""
    dc = DynamicCache()
    for L, (k, v) in enumerate(kv_cpu):
        dc.update(k.to(dev).clone(), v.to(dev).clone(), L)
    return dc


RUN_ID = "run_ttdet"


@pytest.mark.model
def test_branch_from_stored_snapshot_byte_matches_fresh_recompute(tmp_path):
    torch, AutoModelForCausalLM, AutoTokenizer, DynamicCache, kvt = _lazy()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: the determinism receipt runs a real KV cache on the GPU")

    dev = "cuda"
    path = kvt.resolve_model_path("Qwen/Qwen2.5-1.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base_users = [
        "Hi! I just moved to a new city and I'm feeling a bit overwhelmed.",
        "What's a good first thing to do to settle in?",
        "I work from home, so I don't meet coworkers naturally.",
        "How do people usually make friends as adults?",
    ]
    alt_user = "Actually, let me change topic -- can you recommend a comforting recipe for tonight?"
    branch_turn = 2               # replace turn 2's user with alt_user; rewind to the state AFTER turn 1
    rewind_idx = branch_turn - 1
    max_new = 40

    # 1) Build the base conversation with KVChat, offloading EACH turn's KV into the product's store.
    #    A snapshot at ring index i holds the cache + n_tok + messages right AFTER turn i (user+assistant).
    store = tt.SnapshotStore(cap=8, budget_mb=512)
    chat = kvt.KVChat(model, tok)
    snap_messages = {}                                    # turn -> the transcript as of that snapshot
    for i, u in enumerate(base_users):
        info = chat.generate_turn(u, max_new=max_new)
        kv_cpu = tt.offload_cache(chat.cache)             # == KVChat._snapshot's payload shape
        snap = store.snapshot_turn(RUN_ID, i, n_tok=info["n_tok"], kv=kv_cpu)
        snap_messages[i] = [dict(m) for m in chat.checkpoints[i]["messages"]]
        assert snap.has_cache and snap.nbytes > 0         # a real offloaded payload, sized honestly

    # the store's per-snapshot byte size must equal the pure accounting formula for this model's shapes
    cfg = model.config
    n_kv, hd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    rewind_snap = store.get(RUN_ID, rewind_idx)
    expect_bytes = tt.kv_snapshot_bytes(rewind_snap.n_tok, cfg.num_hidden_layers, n_kv, hd, bytes_per_elt=2)
    assert rewind_snap.nbytes == expect_bytes

    # the product transform: branching turn `branch_turn` of the FULL transcript replaces that user turn.
    full_msgs = [dict(m) for m in chat.checkpoints[-1]["messages"]]
    branched_msgs = tt.branch_messages(full_msgs, branch_turn, alt_user=alt_user)
    assert branched_msgs[-1]["content"] == alt_user

    # 2) BRANCH via the store: restore the shared past from the STORE's snapshot at rewind_idx, then append
    #    the alt user as a fresh turn -- NO re-prefill of the shared prefix.
    bchat = kvt.KVChat(model, tok)
    bchat.cache = _restore_cache_from(rewind_snap.kv, DynamicCache, dev)   # <-- restore from STORE payload
    bchat.n_tok = rewind_snap.n_tok
    bchat.messages = list(snap_messages[rewind_idx])      # the transcript exactly as of that snapshot
    tok_before_branch = bchat.n_tok
    binfo = bchat.generate_turn(alt_user, max_new=max_new)
    branch_ids = bchat.checkpoints[-1]["gen_ids"]

    # 3) FRESH full recompute of the identical branched transcript, from an empty cache.
    fresh_users = base_users[:branch_turn] + [alt_user]
    fr = kvt.fresh_recompute(model, tok, fresh_users, max_new=max_new)

    # 4) THE RECEIPT: byte-for-byte identical generated tokens (stronger than agreement >= 0.98).
    agree = kvt.token_agreement(branch_ids, fr["gen_ids"])
    assert agree["identical"] is True, (
        f"branch-from-stored-snapshot diverged from fresh recompute: {agree}\n"
        f"branch:  {bchat.checkpoints[-1]['reply']!r}\nfresh:   {fr['reply']!r}")

    # savings receipt (the cheap, assumption-free column): the branch did NOT re-prefill the shared prefix.
    assert tok_before_branch > 0
    assert binfo["new_prefill_tokens"] < tok_before_branch   # only the alt user turn was prefilled

    print(f"\n[ttdet] byte-identical={agree['identical']} tokens={agree['len_a']} "
          f"| shared prefix NOT reprefilled={tok_before_branch} tok, branch prefilled "
          f"{binfo['new_prefill_tokens']} | snapshot={rewind_snap.nbytes} B "
          f"({round(rewind_snap.nbytes/1048576,3)} MB)", flush=True)
