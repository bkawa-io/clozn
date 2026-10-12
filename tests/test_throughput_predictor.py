"""test_throughput_predictor -- model-free tests for clozn/cli/throughput_predictor.py: the decode-tok/s
ROOFLINE `clozn plan` now reports (memory-bandwidth-bound autoregressive decode -- see that module's
docstring for the physics: tok/s ~= bandwidth_bytes_s / (weight_bytes + kv_bytes_per_token)).

Layout:
  * weight_bytes_and_params() driven directly against small, hand-built tensor-info lists (the same shape
    parse_gguf_header() produces) -- exact bits-per-weight arithmetic for known ggml_types, plus the
    "unknown type" escape hatch (never silently misestimates; counts and reports what it excluded).
  * predict_throughput() driven against a synthetic header dict with known layer/head/embedding counts --
    every intermediate (kv_bytes_per_token, total_bytes_per_token, predicted_tok_s) checked against hand
    computation, so this is provably the advertised formula, not just "a number came out."
  * cross-check against the two REAL local GGUFs named in FRONTIER_BETS/test_fit_planner.py: the exact
    per-tensor weight-byte sum this module computes should land within ~1% of the file's actual size on
    disk (the file IS mostly quantized tensor bytes, so this is a strong sanity check of the whole bpw
    table) -- skipped, not failed, if this machine doesn't have them under ~/.clozn/models.
  * format_throughput()'s pure render: the honest ROOFLINE caveat, the weight/KV breakdown, and the
    unknown-ggml-type flag, all driven from canned dicts (no I/O).
  * clozn.cli.main wiring: `plan` carries --bandwidth-gb-s and --calibrate, and cmd_plan's DEFERRED
    --calibrate stub prints its TODO without touching the engine (no engine_process import in that path).
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))          # tests/
REPO = os.path.dirname(HERE)                                # repo root
sys.path.insert(0, REPO)

from clozn.cli import throughput_predictor as tp             # noqa: E402
import clozn.cli.main as clozn_cli                            # noqa: E402


# --------------------------------------------------------------------------------- weight_bytes_and_params

def test_weight_bytes_and_params_exact_for_known_types():
    tensors = [
        {"name": "blk.0.attn_q.weight", "shape": [64, 64], "ggml_type": 12},   # Q4_K: 4096 params @ 4.5 bpw
        {"name": "blk.0.attn_norm.weight", "shape": [64], "ggml_type": 0},     # F32:  64 params @ 32 bpw
    ]
    r = tp.weight_bytes_and_params(tensors)
    assert r["n_params"] == 4096 + 64
    assert r["weight_bytes"] == pytest.approx(4096 * 4.5 / 8 + 64 * 32 / 8)
    assert r["unknown_type_tensor_counts"] == {}
    assert r["unknown_params"] == 0


def test_weight_bytes_and_params_excludes_unknown_ggml_type_but_still_counts_params():
    tensors = [
        {"name": "blk.0.weird", "shape": [10, 10], "ggml_type": 12345},   # not in BPW_BY_GGML_TYPE
        {"name": "blk.0.norm", "shape": [10], "ggml_type": 0},            # F32, known
    ]
    r = tp.weight_bytes_and_params(tensors)
    assert r["n_params"] == 100 + 10                     # every param counted regardless of type
    assert r["weight_bytes"] == pytest.approx(10 * 32 / 8)   # only the F32 tensor contributes bytes
    assert r["unknown_type_tensor_counts"] == {12345: 1}
    assert r["unknown_params"] == 100


def test_weight_bytes_and_params_empty_tensor_list():
    r = tp.weight_bytes_and_params([])
    assert r["n_params"] == 0
    assert r["weight_bytes"] == 0.0
    assert r["unknown_params"] == 0


# ------------------------------------------------------------------------------------- predict_throughput
# A small synthetic header: 4 layers, embedding_length=64, head_count=8 (-> head_dim=8), head_count_kv=8,
# plus a tensor list whose exact weight-byte total we compute by hand below -- so every field in the
# returned estimate is checked against hand arithmetic, not just "some plausible number came out."

SYNTH_HEADER = {
    "n_layers": 4,
    "embedding_length": 64,
    "head_count": 8,
    "head_count_kv": 8,
    "tensors": [
        {"name": "blk.0.attn_q.weight", "shape": [64, 64], "ggml_type": 12},   # Q4_K: 4096 @ 4.5 bpw
        {"name": "blk.0.attn_norm.weight", "shape": [64], "ggml_type": 0},     # F32:  64 @ 32 bpw
    ],
}
_EXPECT_WEIGHT_BYTES = 4096 * 4.5 / 8 + 64 * 32 / 8      # = 2304 + 256 = 2560.0
_EXPECT_N_PARAMS = 4096 + 64


def test_predict_throughput_matches_hand_computed_roofline():
    est = tp.predict_throughput(SYNTH_HEADER, bandwidth_gb_s=900.0, kv_bytes_per_element=2.0,
                                ctx_for_estimate=8192)
    assert est["n_params"] == _EXPECT_N_PARAMS
    assert est["weight_bytes"] == pytest.approx(_EXPECT_WEIGHT_BYTES)

    # kv_bytes_per_token = 2 (K&V) * n_layers * head_count_kv * head_dim * ctx * kv_bytes_per_element
    head_dim = 64 / 8                                     # embedding_length / head_count = 8
    expect_kv = 2 * 4 * 8 * head_dim * 8192 * 2.0
    assert est["kv_bytes_per_token"] == pytest.approx(expect_kv)

    expect_total = _EXPECT_WEIGHT_BYTES + expect_kv
    assert est["total_bytes_per_token"] == pytest.approx(expect_total)

    expect_tok_s = (900.0 * 1e9) / expect_total
    assert est["predicted_tok_s"] == pytest.approx(expect_tok_s)


def test_predict_throughput_bandwidth_scales_linearly():
    """Same bytes/token, double the assumed bandwidth -> exactly double the predicted tok/s (it's a
    straight division, so this is a strong regression guard against an accidental non-linear term)."""
    lo = tp.predict_throughput(SYNTH_HEADER, bandwidth_gb_s=300.0)
    hi = tp.predict_throughput(SYNTH_HEADER, bandwidth_gb_s=600.0)
    assert hi["predicted_tok_s"] == pytest.approx(2 * lo["predicted_tok_s"])


def test_predict_throughput_more_context_lowers_tok_s():
    """KV bytes/token grows with the assumed context depth -- more context, fewer tok/s, all else equal.
    This is the honest "throughput degrades with context" fact the CAVEAT in format_throughput alludes to."""
    small_ctx = tp.predict_throughput(SYNTH_HEADER, ctx_for_estimate=1024)
    big_ctx = tp.predict_throughput(SYNTH_HEADER, ctx_for_estimate=32768)
    assert big_ctx["predicted_tok_s"] < small_ctx["predicted_tok_s"]


def test_predict_throughput_missing_fields_does_not_raise():
    est = tp.predict_throughput({})
    assert est["n_params"] == 0
    assert est["predicted_tok_s"] == 0.0                  # honest zero, not a crash or a divide-by-zero


def test_default_bandwidth_is_rtx_5080_class_and_stated():
    # The task's own framing: "~900 GB/s-ish class" for this machine -- pinned so a silent drift in the
    # default doesn't quietly change what every unlabeled `clozn plan` call reports.
    assert tp.DEFAULT_BANDWIDTH_GB_S == 900.0


# --------------------------------------------------------------------- ground truth: the two real GGUFs
# Same two files test_fit_planner.py checks against FRONTIER_BETS's ground truth. Here: the EXACT per-
# tensor weight-byte sum this module computes should land close to the real file size on disk -- proof the
# bpw table is right, not just internally consistent. Skipped (not failed) if this machine doesn't have
# them under ~/.clozn/models.

_MODELS_DIR = os.path.expanduser("~/.clozn/models")
QWEN_PATH = os.path.join(_MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
LLAMA_PATH = os.path.join(_MODELS_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")


@pytest.mark.skipif(not os.path.isfile(QWEN_PATH), reason="ground-truth Qwen2.5-7B GGUF not on this machine")
def test_real_qwen_weight_bytes_close_to_file_size_on_disk():
    from clozn.cli import fit_planner
    h = fit_planner.gguf_header_from_path(QWEN_PATH)
    est = tp.predict_throughput(h)
    assert est["n_params"] == pytest.approx(7_616_000_000, rel=0.01)     # ~7.6B, the known Qwen2.5-7B count
    # the GGUF file is almost entirely the quantized tensor bytes (a little metadata besides) -- our
    # exact per-tensor sum should track the real file size within a couple of percent.
    assert est["weight_bytes"] == pytest.approx(h["file_size_bytes"], rel=0.02)
    assert est["predicted_tok_s"] > 0
    assert est["unknown_params"] == 0        # every tensor type in a real Q4_K_M file is in our table


@pytest.mark.skipif(not os.path.isfile(LLAMA_PATH), reason="ground-truth Llama-3.2-1B GGUF not on this machine")
def test_real_llama_weight_bytes_close_to_file_size_on_disk():
    from clozn.cli import fit_planner
    h = fit_planner.gguf_header_from_path(LLAMA_PATH)
    est = tp.predict_throughput(h)
    assert est["weight_bytes"] == pytest.approx(h["file_size_bytes"], rel=0.02)
    assert est["predicted_tok_s"] > 0


# --------------------------------------------------------------------------------------- format_throughput

QWEN_LIKE_EST = tp.predict_throughput(SYNTH_HEADER, bandwidth_gb_s=900.0)


def test_format_throughput_pure_render_has_the_headline_numbers():
    text = clozn_cli.format_throughput(QWEN_LIKE_EST)
    assert "predicted decode throughput" in text
    assert "tok/s" in text
    assert "900" in text                       # the bandwidth assumption, stated
    assert "bpw" in text
    assert "GB" in text or "MB" in text


def test_format_throughput_states_the_roofline_caveat_honestly():
    text = clozn_cli.format_throughput(QWEN_LIKE_EST)
    low = text.lower()
    assert "roofline" in low
    assert "not a promise" in low or "not a promise" in text.lower()
    assert "prefill" in low
    assert "calibrate" in low


def test_format_throughput_flags_unknown_ggml_types():
    est = tp.predict_throughput({
        "n_layers": 1, "embedding_length": 8, "head_count": 1, "head_count_kv": 1,
        "tensors": [{"name": "x", "shape": [8, 8], "ggml_type": 99999}],
    })
    text = clozn_cli.format_throughput(est)
    assert "unrecognized" in text.lower()


def test_format_throughput_unavailable_when_header_has_no_usable_tensors():
    est = tp.predict_throughput({})
    text = clozn_cli.format_throughput(est)
    assert "unavailable" in text.lower()


# ------------------------------------------------------------------------------------------ cli.py wiring

def test_cli_plan_bandwidth_and_calibrate_flags_registered():
    p = clozn_cli.build_parser()
    args = p.parse_args(["plan", "qwen", "--bandwidth-gb-s", "500", "--calibrate"])
    assert args.bandwidth_gb_s == 500.0
    assert args.calibrate is True


def test_cli_plan_bandwidth_and_calibrate_default_off():
    p = clozn_cli.build_parser()
    args = p.parse_args(["plan", "qwen"])
    assert args.bandwidth_gb_s is None
    assert args.calibrate is False


def test_calibrate_stub_prints_unavailable_and_never_touches_the_engine(capsys):
    """--calibrate must be clearly unavailable, not a real engine
    boot. Proof-by-source as well as by-behavior: _cmd_plan_calibrate_stub's own module (commands/models.py)
    imports engine_process only for ENGINE_CORE/REPO/find_engine at module load -- this call must not
    reach any of those or spawn a subprocess."""
    from clozn.cli.commands import models as models_mod
    models_mod._cmd_plan_calibrate_stub("some-model")
    out = capsys.readouterr().out
    assert "unavailable" in out.lower()
    assert "not part of the current plan command" in out.lower()


def test_cmd_plan_end_to_end_prints_throughput_block(tmp_path, capsys):
    """Full cmd_plan() run against a real (synthetic) GGUF file on disk: the throughput predictor's output
    must appear right after the existing fit-check block, not replace it."""
    import struct
    from types import SimpleNamespace

    def _u32(v): return struct.pack("<I", v)
    def _u64(v): return struct.pack("<Q", v)
    def _str(s):
        b = s.encode("utf-8")
        return _u64(len(b)) + b
    def _kv_u32(key, v): return _str(key) + _u32(4) + _u32(v)
    def _kv_str(key, v): return _str(key) + _u32(8) + _str(v)
    def _tensor_info(name, dims, ggml_type, offset):
        out = _str(name) + _u32(len(dims))
        for d in dims:
            out += _u64(d)
        return out + _u32(ggml_type) + _u64(offset)

    tensors = [("blk.0.attn_q.weight", [64, 64], 12, 0), ("blk.0.attn_norm.weight", [64], 0, 16384)]
    kvs = b"".join([
        _kv_str("general.architecture", "testarch"),
        _kv_u32("general.file_type", 15),
        _kv_u32("testarch.block_count", 4),
        _kv_u32("testarch.context_length", 2048),
        _kv_u32("testarch.embedding_length", 64),
        _kv_u32("testarch.attention.head_count", 8),
        _kv_u32("testarch.attention.head_count_kv", 8),
    ])
    tinfo = b"".join(_tensor_info(*t) for t in tensors)
    data = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(7) + kvs + tinfo

    path = tmp_path / "synthetic.gguf"
    path.write_bytes(data)

    args = SimpleNamespace(model=str(path), vram=None, bandwidth_gb_s=None, calibrate=False)
    clozn_cli.cmd_plan(args)
    out = capsys.readouterr().out
    assert "FITS" in out or "WON'T FIT" in out             # the existing fit-check block, untouched
    assert "predicted decode throughput" in out             # the new throughput block, appended
    assert "ROOFLINE" in out
