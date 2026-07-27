"""test_generation_meta -- honest reproducibility metadata (backlog #1): the pure per-substrate
generation-meta builders in clozn/clozn_server.py (_qwen_generation_meta, _engine_generation_meta,
_without_unknowns).

Model-free: these are plain dict builders, no torch/HF/engine process involved. EngineSubstrate.run_meta()
already has thorough coverage in test_engine_substrate.py (health-derived n_ctx/device/gpu_layers, greedy
0-valued fields preserved, never-raises-on-bad-health) -- this file covers the Qwen/HF side of the same
contract, which had no dedicated test before: the sampling regime it actually uses (temperature=0.7,
top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3 -- QwenMemory._generate's real generate() kwargs)
is honestly persisted; greedy mode's temperature 0.0 survives the "drop unknowns" filter (0 is not
"unknown"); and fields the Qwen path genuinely never sets (seed -- HF sampling here uses no fixed seed;
top_k -- generate() passes no explicit top_k, so its value is an implicit, transformers-version-dependent
default this code cannot honestly report) are never invented.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from clozn.server import app as cs   # noqa: E402


def test_without_unknowns_drops_none_but_keeps_honest_falsy_values():
    d = {"temperature": 0.0, "seed": 0, "top_p": None, "max_tokens": None, "stream": False}
    assert cs._without_unknowns(d) == {"temperature": 0.0, "seed": 0, "stream": False}


def test_qwen_generation_meta_sampling_regime_is_honest():
    """The Qwen chat/chat_stream sampling call: do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.3, no_repeat_ngram_size=3 -- exactly QwenMemory._generate's real generate() kwargs
    (self_teach_server.py) -- must be exactly what's persisted, not a guess."""
    meta = cs._qwen_generation_meta(256, sample=True, stream=False)
    assert meta["temperature"] == 0.7
    assert meta["top_p"] == 0.9
    assert meta["repetition_penalty"] == 1.3
    assert meta["no_repeat_ngram_size"] == 3
    assert meta["sampler_mode"] == "sample" and meta["sampling"] == "sample"
    assert meta["max_tokens"] == 256
    assert meta["stream"] is False
    # never invented: the HF call sets no explicit seed anywhere on this path, and no explicit top_k either
    assert "seed" not in meta
    assert "top_k" not in meta


def test_qwen_generation_meta_greedy_keeps_the_honest_zero_temperature():
    """sample=False -> temperature 0.0, and _without_unknowns must NOT drop it just because it's falsy.
    repetition_penalty/no_repeat_ngram_size still apply even in greedy mode (they're unconditional logits
    processors in _generate, not sampling-only params) -- so they're still honestly reported here too."""
    meta = cs._qwen_generation_meta(64, sample=False, stream=False)
    assert meta["temperature"] == 0.0
    assert meta["sampler_mode"] == "greedy" and meta["sampling"] == "greedy"
    assert meta["repetition_penalty"] == 1.3 and meta["no_repeat_ngram_size"] == 3
    # top_p has no effect once do_sample=False -- honestly omitted, not guessed at 0.9 or fabricated None
    assert "top_p" not in meta
    assert "seed" not in meta and "top_k" not in meta


def test_engine_generation_meta_forced_greedy_regime_is_honest():
    """The engine chat path forces temperature=0.0, rep_penalty=1.0, seed=0 -- greedy, deterministic,
    reproducible -- and all three honest-falsy values must survive _without_unknowns."""
    meta = cs._engine_generation_meta(128, stream=True)
    assert meta["temperature"] == 0.0
    assert meta["repetition_penalty"] == 1.0
    assert meta["seed"] == 0
    assert meta["sampler_mode"] == "greedy" and meta["sampling"] == "greedy"
    assert meta["max_tokens"] == 128
    assert meta["stream"] is True
    # top_p/top_k do not participate in the forced-greedy path; no_repeat_ngram_size is not an engine knob
    # at all. None should be fabricated in the metadata for this call.
    assert "top_p" not in meta and "top_k" not in meta and "no_repeat_ngram_size" not in meta


def test_engine_decode_block_is_the_reproducible_greedy_regime():
    """S2 self-describing `decode`: engine chat is greedy/temp0/seed0 -- reproducible by construction, the
    exact values passed to the engine (not a guess). Nested block survives (it's not None, so
    _without_unknowns keeps it; the meaningful nested values stay)."""
    meta = cs._engine_generation_meta(128, stream=True)
    assert meta["decode"] == {"mode": "greedy", "temperature": 0.0, "seed": 0}


def test_qwen_decode_block_marks_sampling_as_not_reproducible():
    """S2 self-describing `decode`: a sampled Qwen run carries seed=None ON PURPOSE -- HF sets no fixed
    seed, so it is not exactly reproducible; that honest null is the whole point of the block (contrast
    the engine's seed 0). Greedy Qwen: temperature 0.0, top_p null (N/A without sampling), seed still null."""
    sampled = cs._qwen_generation_meta(256, sample=True, stream=False)["decode"]
    assert sampled == {"mode": "sample", "temperature": 0.7, "top_p": 0.9, "seed": None}
    greedy = cs._qwen_generation_meta(64, sample=False, stream=False)["decode"]
    assert greedy == {"mode": "greedy", "temperature": 0.0, "top_p": None, "seed": None}
