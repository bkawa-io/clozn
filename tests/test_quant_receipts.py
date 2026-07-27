"""test_quant_receipts -- clozn/receipts/quant_receipts.py (QUANT-RECEIPTS): "did Q4 lobotomize my
model?"

Model-free throughout, on FIXTURE /score-shaped arrays (`[{"id","piece","logprob","topk"?}, ...]` --
exactly what engine/core/serve/routes_whitebox.cpp's /score handler and
engine/client/clozn_engine.py's score() return) -- no C++ engine, no GPU, no real socket. The live-path
seam (`quant_receipt_for_run`) is exercised only against a FakeScoreSub (mirrors test_rederive.py's own
fake), never against a real engine -- the two-real-quant-files run stays deferred.

Covers:
  * diff_quant_scores -- a preserved case (no flips), a broken case (argmax flips at k positions), the
    unknown-flip-status case (missing/empty topk), and the graceful failure modes (length mismatch, id
    misalignment, missing logprob, malformed input).
  * the flip-detail cap and its "showing first N of M" note.
  * quant_receipt_for_run -- the live-path seam's wiring: bad run / no continuation ids / a substrate
    that can't score / the happy path delegating to diff_quant_scores.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.receipts.quant_receipts as qr  # noqa: E402


# ==================================================================================== fixtures / fakes

def _tok(id_, piece, logprob, topk=None):
    t = {"id": id_, "piece": piece, "logprob": logprob}
    if topk is not None:
        t["topk"] = topk
    return t


class FakeScoreSub:
    """Exposes exactly `.score_tokens` -- mirrors test_rederive.py's own fake exactly, so
    quant_receipt_for_run's wiring is tested the same model-free way rederive.py's is."""

    def __init__(self, tokens=None, raises=False):
        self.calls = []
        self._tokens = tokens if tokens is not None else []
        self._raises = raises

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        if self._raises:
            raise RuntimeError("boom")
        self.calls.append({"messages": messages, "continuation_ids": continuation_ids, "topk": topk,
                           "block": block, "steer_strengths": steer_strengths})
        return self._tokens


class NoScoreSub:
    """A substrate with no score_tokens at all -- score_arm's ([], False) contract."""
    pass


# ==================================================================================== diff_quant_scores: preserved

def test_preserved_case_no_flips():
    answer = [11, 22, 33, 44]
    tokens_a = [
        _tok(11, "Hello", -0.05, [_tok(11, "Hello", -0.05), _tok(99, "Hi", -3.0)]),
        _tok(22, " there", -0.10, [_tok(22, " there", -0.10), _tok(88, " friend", -2.5)]),
        _tok(33, "!", -0.01, [_tok(33, "!", -0.01)]),
        _tok(44, " ok", -0.20, [_tok(44, " ok", -0.20), _tok(77, " sure", -1.0)]),
    ]
    tokens_b = [
        _tok(11, "Hello", -0.07, [_tok(11, "Hello", -0.07), _tok(99, "Hi", -2.9)]),
        _tok(22, " there", -0.15, [_tok(22, " there", -0.15), _tok(88, " friend", -2.6)]),
        _tok(33, "!", -0.02, [_tok(33, "!", -0.02)]),
        _tok(44, " ok", -0.25, [_tok(44, " ok", -0.25), _tok(77, " sure", -1.1)]),
    ]
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b, label_a="Q8_0", label_b="Q4_K_M")
    assert out is not None
    assert out["causal_verified"] is True
    assert out["label_a"] == "Q8_0" and out["label_b"] == "Q4_K_M"
    assert out["n_tokens"] == 4
    s = out["summary"]
    assert s["n_flipped"] == 0
    assert s["n_preserved"] == 4
    assert s["n_unknown"] == 0
    assert s["flipped_positions"] == []
    assert "4/4 tokens preserved" in s["summary_text"]
    assert "no argmax flips" in s["summary_text"]
    # every position's argmax equals the recorded token -- and equals across arms
    for p in out["positions"]:
        assert p["status"] == "preserved"
        assert p["argmax_flip"] is False
        assert p["argmax_a_id"] == p["token_id"] == p["argmax_b_id"]
    # deltas are real and directionally b - a
    assert out["positions"][0]["delta_nats"] == round(-0.07 - (-0.05), 6)
    assert s["mean_abs_delta_nats_all"] is not None
    assert s["mean_abs_delta_nats_preserved"] == s["mean_abs_delta_nats_all"]
    assert out["caveat"] == qr._QUANT_CAVEAT


# ==================================================================================== diff_quant_scores: broken

def test_broken_case_argmax_flips_at_k_positions():
    answer = [1, 2, 3, 4, 5]
    # positions 1 and 3 flip: quant A's greedy pick stays on the recorded token, quant B's argmax moves
    # to a DIFFERENT token (e.g. a formatting/refusal token breaking) -- positions 0,2,4 agree.
    tokens_a = [
        _tok(1, "Sure", -0.1, [_tok(1, "Sure", -0.1)]),
        _tok(2, ", here", -0.1, [_tok(2, ", here", -0.1), _tok(50, ". I", -1.5)]),
        _tok(3, " is", -0.1, [_tok(3, " is", -0.1)]),
        _tok(4, " the", -0.2, [_tok(4, " the", -0.2), _tok(60, " a", -0.9)]),
        _tok(5, " code", -0.1, [_tok(5, " code", -0.1)]),
    ]
    tokens_b = [
        _tok(1, "Sure", -0.3, [_tok(1, "Sure", -0.3)]),
        _tok(2, ", here", -2.0, [_tok(50, ". I", -0.2), _tok(2, ", here", -2.0)]),   # FLIP: argmax -> 50
        _tok(3, " is", -0.15, [_tok(3, " is", -0.15)]),
        _tok(4, " the", -1.8, [_tok(61, " cannot", -0.1), _tok(4, " the", -1.8)]),    # FLIP: argmax -> 61
        _tok(5, " code", -0.12, [_tok(5, " code", -0.12)]),
    ]
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b, label_a="Q8_0", label_b="Q3_K_S")
    assert out["causal_verified"] is True
    s = out["summary"]
    assert s["n_flipped"] == 2
    assert s["n_preserved"] == 3
    assert s["flipped_positions"] == [1, 3]
    assert "diverged at 2 position(s)" in s["summary_text"]
    assert "[1, 3]" in s["summary_text"]
    # flip detail names what each quant would actually have said
    detail_by_index = {d["index"]: d for d in s["flipped_detail"]}
    assert detail_by_index[1]["Q8_0_would_say"] == ", here"
    assert detail_by_index[1]["Q3_K_S_would_say"] == ". I"
    assert detail_by_index[3]["Q8_0_would_say"] == " the"
    assert detail_by_index[3]["Q3_K_S_would_say"] == " cannot"
    # the non-flipped positions are still correctly "preserved"
    preserved_idxs = {p["index"] for p in out["positions"] if p["status"] == "preserved"}
    assert preserved_idxs == {0, 2, 4}


def test_flip_detail_is_capped_with_a_note():
    n = 25
    answer = list(range(n))
    tokens_a = [_tok(i, f"a{i}", -0.1, [_tok(i, f"a{i}", -0.1)]) for i in range(n)]
    tokens_b = [_tok(i, f"a{i}", -0.1, [_tok(1000 + i, f"b{i}", -0.1)]) for i in range(n)]  # all flip
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    s = out["summary"]
    assert s["n_flipped"] == n
    assert len(s["flipped_positions"]) == n           # the cheap index list is NOT truncated
    assert len(s["flipped_detail"]) == qr._FLIP_DETAIL_CAP  # the detail rendering IS capped
    assert f"showing first {qr._FLIP_DETAIL_CAP} of {n}" in s["summary_text"]


# ==================================================================================== diff_quant_scores: unknown / topk

def test_missing_topk_is_unknown_not_preserved():
    answer = [1, 2, 3]
    tokens_a = [
        _tok(1, "a", -0.1, [_tok(1, "a", -0.1)]),
        _tok(2, "b", -0.1),               # no topk key at all on this arm
        _tok(3, "c", -0.1, [_tok(3, "c", -0.1)]),
    ]
    tokens_b = [
        _tok(1, "a", -0.2, [_tok(1, "a", -0.2)]),
        _tok(2, "b", -0.2, [_tok(2, "b", -0.2)]),   # topk present on THIS arm, but not the other
        _tok(3, "c", -0.2, []),                      # empty topk list on this arm
    ]
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    s = out["summary"]
    assert s["n_unknown"] == 2
    assert s["n_preserved"] == 1
    assert s["n_flipped"] == 0
    positions_by_index = {p["index"]: p for p in out["positions"]}
    assert positions_by_index[1]["status"] == "unknown"
    assert positions_by_index[1]["argmax_flip"] is None
    assert positions_by_index[2]["status"] == "unknown"
    # delta_nats is still reported even when flip status is unknown
    assert positions_by_index[1]["delta_nats"] == round(-0.2 - (-0.1), 6)
    assert "unknown flip status" in s["summary_text"]
    # unknown positions are excluded from the preserved-only mean but present in the all-positions mean
    assert s["mean_abs_delta_nats_preserved"] == round(abs(-0.2 - (-0.1)), 6)  # only position 0
    assert s["mean_abs_delta_nats_all"] is not None


def test_rank_fields_when_actual_token_is_not_rank_zero():
    answer = [7]
    tokens_a = [_tok(7, "x", -0.5, [_tok(9, "y", -0.05), _tok(7, "x", -0.5)])]      # actual token at rank 1
    tokens_b = [_tok(7, "x", -0.5, [_tok(9, "y", -0.05)])]                          # actual token NOT in topk
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    p = out["positions"][0]
    assert p["rank_a"] == 1
    assert p["rank_b"] is None
    assert p["rank_change"] is None      # can't compute a rank change when one side is unknown
    assert p["argmax_a_id"] == 9 and p["argmax_b_id"] == 9
    assert p["argmax_flip"] is False     # both arms agree the argmax is 9 -- no flip despite differing ranks


# ==================================================================================== diff_quant_scores: graceful failures

def test_length_mismatch_is_graceful_not_a_raise():
    out = qr.diff_quant_scores([1, 2, 3], [_tok(1, "a", -0.1)], [_tok(1, "a", -0.1)])
    assert out["causal_verified"] is False
    assert "length" in out["note"] or "align" in out["note"]
    assert out["caveat"] == qr._QUANT_CAVEAT


def test_id_misalignment_is_graceful_not_a_raise():
    answer = [1, 2]
    tokens_a = [_tok(1, "a", -0.1), _tok(2, "b", -0.1)]
    tokens_b = [_tok(1, "a", -0.1), _tok(999, "DIFFERENT", -0.1)]   # arm B scored a different continuation
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    assert out["causal_verified"] is False
    assert "position 1" in out["note"]


def test_missing_logprob_is_graceful_not_a_raise():
    answer = [1]
    tokens_a = [{"id": 1, "piece": "a"}]   # no "logprob" key
    tokens_b = [_tok(1, "a", -0.1)]
    out = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    assert out["causal_verified"] is False
    assert "logprob" in out["note"]


def test_malformed_input_returns_none():
    assert qr.diff_quant_scores([], [], []) is None
    assert qr.diff_quant_scores(None, [_tok(1, "a", -0.1)], [_tok(1, "a", -0.1)]) is None
    assert qr.diff_quant_scores([1], "not a list", [_tok(1, "a", -0.1)]) is None
    assert qr.diff_quant_scores([1], [_tok(1, "a", -0.1)], []) is None


def test_non_dict_score_entry_is_graceful_not_a_raise():
    out = qr.diff_quant_scores([1], ["not a dict"], [_tok(1, "a", -0.1)])
    assert out["causal_verified"] is False
    assert "not a dict" in out["note"]


def test_never_raises_on_garbage_types():
    # garbage that would otherwise blow up int()/dict access is caught (either by the per-position
    # id check treating an un-int-able id as a mismatch, or by the outer try/except) -- never raises.
    out = qr.diff_quant_scores([object()], [{"id": object(), "logprob": -0.1}],
                               [{"id": object(), "logprob": -0.1}])
    assert out is None or out.get("causal_verified") is False


def test_deterministic_across_repeated_calls():
    answer = [1, 2]
    tokens_a = [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)]), _tok(2, "b", -0.2, [_tok(2, "b", -0.2)])]
    tokens_b = [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)]), _tok(2, "b", -0.2, [_tok(2, "b", -0.2)])]
    a = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    b = qr.diff_quant_scores(answer, tokens_a, tokens_b)
    assert a == b


# ==================================================================================== quant_receipt_for_run (live seam, fake subs only)

RUN = {
    "id": "run_1",
    "messages": [{"role": "user", "content": "hi"}],
    "assembled_messages": [{"role": "user", "content": "hi"}],
    "response": "ok",
    "behavior": {"active_dials": {}},
    "trace": {"token_ids": [1, 2]},
}

TOKENS_A = [_tok(1, "o", -0.1, [_tok(1, "o", -0.1)]), _tok(2, "k", -0.1, [_tok(2, "k", -0.1)])]
TOKENS_B = [_tok(1, "o", -0.2, [_tok(1, "o", -0.2)]), _tok(2, "k", -0.2, [_tok(2, "k", -0.2)])]


def test_quant_receipt_for_run_returns_none_for_bad_run():
    assert qr.quant_receipt_for_run(None, FakeScoreSub(), FakeScoreSub(), label_a="a", label_b="b") is None
    assert qr.quant_receipt_for_run({}, FakeScoreSub(), FakeScoreSub(), label_a="a", label_b="b") is None


def test_quant_receipt_for_run_flags_missing_continuation_ids():
    run = {"messages": [{"role": "user", "content": "hi"}], "response": "ok"}   # no trace ids
    out = qr.quant_receipt_for_run(run, FakeScoreSub(tokens=TOKENS_A), FakeScoreSub(tokens=TOKENS_B),
                                   label_a="Q8", label_b="Q4")
    assert out["causal_verified"] is False
    assert "continuation ids" in out["note"]


def test_quant_receipt_for_run_flags_a_substrate_that_cannot_score():
    out = qr.quant_receipt_for_run(RUN, NoScoreSub(), FakeScoreSub(tokens=TOKENS_B),
                                   label_a="Q8", label_b="Q4")
    assert out["causal_verified"] is False
    assert "Q8" in out["note"]


def test_quant_receipt_for_run_happy_path_delegates_to_diff_quant_scores():
    sub_a = FakeScoreSub(tokens=TOKENS_A)
    sub_b = FakeScoreSub(tokens=TOKENS_B)
    out = qr.quant_receipt_for_run(RUN, sub_a, sub_b, label_a="Q8_0", label_b="Q4_K_M", topk=8)
    assert out["causal_verified"] is True
    assert out["label_a"] == "Q8_0" and out["label_b"] == "Q4_K_M"
    assert out["n_tokens"] == 2
    # both subs were called with the SAME run conditions -- continuation ids, topk passthrough
    assert sub_a.calls[-1]["continuation_ids"] == [1, 2]
    assert sub_b.calls[-1]["continuation_ids"] == [1, 2]
    assert sub_a.calls[-1]["topk"] == 8
    assert sub_b.calls[-1]["topk"] == 8


def test_quant_receipt_for_run_never_raises_when_a_substrate_raises():
    sub_a = FakeScoreSub(raises=True)
    sub_b = FakeScoreSub(tokens=TOKENS_B)
    out = qr.quant_receipt_for_run(RUN, sub_a, sub_b, label_a="Q8", label_b="Q4")
    assert out["causal_verified"] is False
