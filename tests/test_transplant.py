"""test_transplant -- clozn/analysis/transplant.py (`clozn.transplant.v1`), slice 3.4: the controlled
transplant primitive and its five-arm control harness.

Model-free throughout (roadmap rule 8): `reference_loader`/`candidate_loader` are fake context-manager
factories wrapping a `FakeEngine` whose `.score(...)` returns hand-built /score-shaped dicts (or raises a
queued exception) -- no real engine, no GPU, no network. Sequencing (reference loaded, captured, and torn
down BEFORE the candidate is ever loaded; the candidate torn down even when an arm mid-sequence fails) is
verified directly against an ordered event log, mirroring tests/test_mechanistic_diff.py's own discipline
for the sibling slice 3.2 module.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from clozn import schemas  # noqa: E402
from clozn.analysis import tensor_store  # noqa: E402
from clozn.analysis import transplant as tp  # noqa: E402


# ==================================================================================== fakes

class FakeEngine:
    """`responses` is an ordered list of items, one per expected `.score(...)` call: either a dict to
    return, or an exception INSTANCE to raise. Every call (kwargs) is recorded in `.calls` regardless of
    outcome, so a test can assert both the exact wire shape sent and the exact sequence of calls made."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeEngine.score called more times than responses were queued")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _loader(engine, name, events):
    @contextlib.contextmanager
    def _cm():
        events.append(f"enter:{name}")
        try:
            yield engine
        finally:
            events.append(f"exit:{name}")
    return _cm


_COMPATIBLE = {
    "schema_version": "clozn.pair-compatibility.v1",
    "model_a": {"label": "reference"},
    "model_b": {"label": "candidate"},
    "hidden_size": {"state": "same", "value_a": 4, "value_b": 4},
    "layer_count": {"state": "same", "value_a": 6, "value_b": 6},
    "verdict": {
        "overall": "compatible", "reasons": [],
        "operations": {
            "per_token_comparison": {"permitted": True, "reason": "tokenizers match exactly."},
            "residual_transplant": {"permitted": True, "reason": "hidden_size matches exactly (4)."},
        },
    },
}

_SITE = {"hook": "residual", "layer": 2}

# hidden_size=4/head_count=2 -> d_head=2, a non-trivial slice width for the head-site tests below.
_COMPATIBLE_HEAD = {
    "schema_version": "clozn.pair-compatibility.v1",
    "model_a": {"label": "reference"},
    "model_b": {"label": "candidate"},
    "hidden_size": {"state": "same", "value_a": 4, "value_b": 4},
    "layer_count": {"state": "same", "value_a": 6, "value_b": 6},
    "head_count": {"state": "same", "value_a": 2, "value_b": 2},
    "verdict": {
        "overall": "compatible", "reasons": [],
        "operations": {
            "per_token_comparison": {"permitted": True, "reason": "tokenizers match exactly."},
            "residual_transplant": {"permitted": True, "reason": "hidden_size matches exactly (4)."},
            "head_transplant": {"permitted": True, "reason": "head_count matches exactly (2)."},
        },
    },
}

_HEAD_SITE = {"hook": "head", "layer": 2, "head": 1}


def _run_kwargs(**over):
    base = dict(
        pair_compat=_COMPATIBLE, prompt_ids=[1, 2], continuation_ids=[9], site=_SITE,
        shuffled_layer=1, write_positions=[2], readout_position=2, target_token_id=5, topk=3, seed=0,
    )
    base.update(over)
    return base


# ==================================================================================== response builders

def _resp(*, sum_logprob, top1_id, top1_piece, top1_logprob, second_id=None, second_piece=None,
          second_logprob=None, captured_layer=None, captured_field="captured", position=2,
          write_applied=None, applied_field="write_applied"):
    topk = [{"id": top1_id, "piece": top1_piece, "logprob": top1_logprob}]
    if second_id is not None:
        topk.append({"id": second_id, "piece": second_piece, "logprob": second_logprob})
    forced_logprob = top1_logprob if second_id is None else min(top1_logprob, second_logprob) - 1.0
    out = {
        "n_prompt": 2, "n_cont": 1,
        "tokens": [{"id": top1_id, "piece": top1_piece, "logprob": top1_logprob, "topk": topk}],
        "sum_logprob": sum_logprob,
    }
    if captured_layer is not None:
        out[captured_field] = {str(captured_layer): {str(position): [1.0, 0.0, 0.0, 0.0]}}
    if write_applied is not None:
        out[applied_field] = write_applied
    return out


def _vec(*xs):
    return [float(x) for x in xs]


def _head_capture_resp(*, sum_logprob, top1_id, top1_piece, top1_logprob, second_id=None,
                       second_piece=None, second_logprob=None, layer, full_row, position=2,
                       n_head=2, d_head=2):
    """A head_capture-bearing response: head_dims + head_rows (the merged [ne0] row), NOT the generic
    `captured`/`ffn_captured` shape residual/ffn use -- see tp._read_head_vectors."""
    out = _resp(sum_logprob=sum_logprob, top1_id=top1_id, top1_piece=top1_piece, top1_logprob=top1_logprob,
               second_id=second_id, second_piece=second_piece, second_logprob=second_logprob)
    out["head_dims"] = {"ne0": n_head * d_head, "n_head": n_head, "d_head": d_head}
    out["head_rows"] = {str(layer): {str(position): full_row}}
    return out


def _head_write_resp(*, sum_logprob, top1_id, top1_piece, top1_logprob, second_id=None,
                     second_piece=None, second_logprob=None, write_applied=None):
    return _resp(sum_logprob=sum_logprob, top1_id=top1_id, top1_piece=top1_piece, top1_logprob=top1_logprob,
               second_id=second_id, second_piece=second_piece, second_logprob=second_logprob,
               write_applied=write_applied, applied_field="head_write_applied")


# ==================================================================================== tiny math / helpers

def test_random_equal_norm_vector_matches_reference_norm():
    ref = _vec(3.0, 4.0, 0.0, 0.0)          # norm = 5
    rnd = tp._random_equal_norm_vector(ref, random.Random(0))
    got_norm = tp._norm(rnd)
    assert got_norm == pytest.approx(5.0, abs=1e-6)
    assert rnd != ref                       # a random direction, not a copy


def test_random_equal_norm_vector_zero_reference_returns_zero():
    ref = _vec(0.0, 0.0, 0.0, 0.0)
    rnd = tp._random_equal_norm_vector(ref, random.Random(0))
    assert rnd == [0.0, 0.0, 0.0, 0.0]


def test_random_equal_norm_vector_deterministic_given_a_fresh_seeded_rng():
    ref = _vec(1.0, 2.0, 3.0, 4.0)
    a = tp._random_equal_norm_vector(ref, random.Random(42))
    b = tp._random_equal_norm_vector(ref, random.Random(42))
    assert a == b


def test_random_equal_norm_vector_different_seeds_differ():
    ref = _vec(1.0, 2.0, 3.0, 4.0)
    a = tp._random_equal_norm_vector(ref, random.Random(1))
    b = tp._random_equal_norm_vector(ref, random.Random(2))
    assert a != b


def test_flipped_to_target_true_when_baseline_missed_and_arm_hits():
    assert tp._flipped_to_target({"top1_is_target": False}, {"top1_is_target": True}) is True


def test_flipped_to_target_false_when_arm_also_misses():
    assert tp._flipped_to_target({"top1_is_target": False}, {"top1_is_target": False}) is False


def test_flipped_to_target_false_when_baseline_already_hit():
    assert tp._flipped_to_target({"top1_is_target": True}, {"top1_is_target": True}) is False


def test_flipped_to_target_none_when_either_side_missing():
    assert tp._flipped_to_target({}, {"top1_is_target": True}) is None
    assert tp._flipped_to_target({"top1_is_target": False}, {}) is None


def test_writable_range_residual_excludes_layer_zero_and_final_layer():
    assert tp._writable_range("residual", 6) == (1, 6)


def test_writable_range_ffn_includes_layer_zero():
    assert tp._writable_range("ffn", 6) == (0, 6)


def test_writable_range_head_includes_layer_zero():
    assert tp._writable_range("head", 6) == (0, 6)


def test_capture_kwargs_head_requests_full_rows():
    kwargs = tp._capture_kwargs("head", 3, [1, 2])
    assert kwargs == {"head_capture_layers": [3], "head_capture_positions": [1, 2],
                      "head_capture_rows": True}


def test_write_kwargs_head_includes_head_field():
    kwargs = tp._write_kwargs("head", layer=3, positions=[1, 2], values=[1.0, 2.0], head=5)
    assert kwargs == {"head_write": [{"layer": 3, "positions": [1, 2], "values": [1.0, 2.0], "head": 5}]}


def test_write_kwargs_residual_omits_head_field_when_none():
    kwargs = tp._write_kwargs("residual", layer=3, positions=[1, 2], values=[1.0, 2.0])
    assert "head" not in kwargs["write"][0]


# ==================================================================================== _read_head_vectors

def test_read_head_vectors_slices_correct_head_from_merged_row():
    response = {"head_dims": {"ne0": 8, "n_head": 2, "d_head": 4},
               "head_rows": {"3": {"5": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}}}
    out = tp._read_head_vectors(response, layer=3, head=1, positions=[5])
    assert out[5] == pytest.approx([5.0, 6.0, 7.0, 8.0])
    out0 = tp._read_head_vectors(response, layer=3, head=0, positions=[5])
    assert out0[5] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_read_head_vectors_head_out_of_range_is_missing():
    response = {"head_dims": {"ne0": 8, "n_head": 2, "d_head": 4},
               "head_rows": {"3": {"5": [1.0] * 8}}}
    out = tp._read_head_vectors(response, layer=3, head=5, positions=[5])
    assert out[5] is None


def test_read_head_vectors_d_head_zero_is_missing():
    """The 'division did not divide evenly' case (hook_vocabulary's d_head_probe): the engine reports
    d_head=0 and applies nothing -- this must read as every requested position missing, never a guessed
    slice width."""
    response = {"head_dims": {"ne0": 0, "n_head": 2, "d_head": 0}, "head_rows": {"3": {"5": []}}}
    out = tp._read_head_vectors(response, layer=3, head=0, positions=[5])
    assert out[5] is None


def test_read_head_vectors_missing_head_dims_is_missing():
    response = {"head_rows": {"3": {"5": [1.0] * 8}}}
    out = tp._read_head_vectors(response, layer=3, head=0, positions=[5])
    assert out[5] is None


def test_read_head_vectors_missing_position_is_missing_others_present():
    response = {"head_dims": {"ne0": 4, "n_head": 2, "d_head": 2},
               "head_rows": {"3": {"5": [1.0, 2.0, 3.0, 4.0]}}}
    out = tp._read_head_vectors(response, layer=3, head=0, positions=[5, 6])
    assert out[5] == pytest.approx([1.0, 2.0])
    assert out[6] is None


def test_read_head_vectors_short_row_is_missing():
    response = {"head_dims": {"ne0": 8, "n_head": 2, "d_head": 4},
               "head_rows": {"3": {"5": [1.0, 2.0, 3.0]}}}    # shorter than hi=8 for head=1
    out = tp._read_head_vectors(response, layer=3, head=1, positions=[5])
    assert out[5] is None


def test_target_metrics_reads_forced_token_logprob_directly_even_when_topk_omits_it():
    response = {"n_prompt": 2, "n_cont": 1,
               "tokens": [{"id": 5, "piece": "y", "logprob": -0.4, "topk": [{"id": 9, "piece": "x", "logprob": -0.1}]}]}
    out = tp._target_metrics(response, n_prompt=2, n_cont=1, readout_position=2, target_token_id=5)
    assert out["metrics"]["target_token_logprob"] == pytest.approx(-0.4)
    assert out["metrics"]["target_token_piece"] == "y"
    # not in the returned top-k list -> rank is honestly omitted, not guessed
    assert "target_token_rank" not in out["metrics"]
    assert any(o["metric"] == "target_token_rank" for o in out["omitted"])


def test_target_metrics_position_outside_continuation_omits_everything_with_reason():
    response = {"n_prompt": 2, "n_cont": 1, "sum_logprob": -0.1,
               "tokens": [{"id": 5, "piece": "y", "logprob": -0.1, "topk": []}]}
    out = tp._target_metrics(response, n_prompt=2, n_cont=1, readout_position=100, target_token_id=5)
    assert out["metrics"] == {"sum_logprob": -0.1}
    assert len(out["omitted"]) == 5
    assert all("outside the scored continuation range" in o["reason"] for o in out["omitted"])


# ==================================================================================== run_site(): preflight

def test_refuses_non_dict_pair_compat():
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat="not a dict",
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert events == []


def test_refuses_bad_site_hook():
    events = []
    out = tp.run_site(**_run_kwargs(
        site={"hook": "bogus", "layer": 2},
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "site.hook must be one of" in out["error"]
    assert events == []


def test_refuses_when_hidden_size_not_same():
    pair_compat = dict(_COMPATIBLE, hidden_size={"state": "differs", "value_a": 4, "value_b": 8},
                       verdict={"overall": "incompatible", "reasons": [],
                                "operations": {"per_token_comparison": {"permitted": True, "reason": "ok"},
                                               "residual_transplant": {"permitted": False,
                                                                       "reason": "hidden_size differs (4 vs 8)."}}})
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=pair_compat,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "hidden_size" in out["error"]
    assert events == []


def test_refuses_when_layer_count_unknown():
    pair_compat = dict(_COMPATIBLE, layer_count={"state": "unknown"})
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=pair_compat,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "layer_count is unknown" in out["error"]
    assert events == []


def test_refuses_site_layer_outside_writable_range():
    events = []
    out = tp.run_site(**_run_kwargs(
        site={"hook": "residual", "layer": 0},        # layer 0 reserved (l_out's read-tap sentinel)
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "outside the writable range" in out["error"]
    assert events == []


def test_refuses_shuffled_layer_equal_to_site_layer():
    events = []
    out = tp.run_site(**_run_kwargs(
        shuffled_layer=2,           # same as _SITE's layer
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "must differ from site.layer" in out["error"]
    assert events == []


def test_refuses_shuffled_layer_outside_writable_range():
    events = []
    out = tp.run_site(**_run_kwargs(
        shuffled_layer=6,           # layer_count is 6 -> valid range [1,6)
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "shuffled_layer must be an integer in" in out["error"]
    assert events == []


def test_refuses_empty_write_positions():
    events = []
    out = tp.run_site(**_run_kwargs(
        write_positions=[],
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "write_positions must not be empty" in out["error"]
    assert events == []


def test_refuses_topk_less_than_one():
    events = []
    out = tp.run_site(**_run_kwargs(
        topk=0,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "topk >= 1" in out["error"]
    assert events == []


def test_refuses_empty_continuation():
    events = []
    out = tp.run_site(**_run_kwargs(
        continuation_ids=[],
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "non-empty continuation" in out["error"]
    assert events == []


# ==================================================================================== run_site(): head-site preflight
# head_transplant is gated INDEPENDENTLY of residual_transplant (pair_compatibility's own head_count
# dimension/operation) -- these mirror the hidden_size/layer_count preflight tests above, for head_count.

def test_refuses_head_transplant_when_not_permitted():
    pair_compat = dict(_COMPATIBLE_HEAD, head_count={"state": "differs", "value_a": 2, "value_b": 4},
                       verdict={"overall": "compatible_with_caveats", "reasons": ["head_count differs (2 vs 4)."],
                                "operations": {
                                    "per_token_comparison": {"permitted": True, "reason": "ok"},
                                    "residual_transplant": {"permitted": True, "reason": "ok"},
                                    "head_transplant": {"permitted": False,
                                                        "reason": "head_count differs (2 vs 4)."}}})
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=pair_compat, site=_HEAD_SITE, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "head_count" in out["error"]
    assert events == []


def test_refuses_head_site_when_head_count_unknown():
    pair_compat = dict(_COMPATIBLE_HEAD, head_count={"state": "unknown"})
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=pair_compat, site=_HEAD_SITE, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "head_count is unknown" in out["error"]
    assert events == []


def test_refuses_head_site_missing_head_field():
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site={"hook": "head", "layer": 2}, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "site.head must be an integer in [0, 2)" in out["error"]
    assert events == []


def test_refuses_head_site_head_out_of_range():
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site={"hook": "head", "layer": 2, "head": 5}, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "site.head must be an integer in [0, 2)" in out["error"]
    assert events == []


def test_refuses_head_site_head_is_bool():
    # isinstance(True, int) is True in Python -- a stray bool must not be treated as a head index
    # (mirrors this module's own int-vs-bool discipline for site.layer/topk elsewhere).
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site={"hook": "head", "layer": 2, "head": True}, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "site.head must be an integer" in out["error"]
    assert events == []


def test_refuses_head_site_head_is_negative():
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site={"hook": "head", "layer": 2, "head": -1}, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "site.head must be an integer in [0, 2)" in out["error"]
    assert events == []


def test_head_site_layer_zero_is_valid_unlike_residual():
    """head_write's layer range is [0, n_layer), same as ffn_write -- unlike residual write's [1,
    n_layer). Layer 0 must pass this module's OWN preflight (the reference engine call is attempted and
    only fails because the fake engine has nothing queued)."""
    events = []
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site={"hook": "head", "layer": 0, "head": 0}, shuffled_layer=1,
        reference_loader=_loader(FakeEngine([RuntimeError("stop after preflight")]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "reference capture failed" in out["error"]
    assert events == ["enter:ref", "exit:ref"]


# ==================================================================================== run_site(): sequencing

def test_reference_capture_requests_topk_zero_not_the_callers_topk():
    """The reference forward is only ever used to capture the site's vector -- its own token
    distribution is never consulted (target_token_id is interpreted purely under the CANDIDATE's
    vocabulary, see the module docstring), so it should not pay for topk it will never read."""
    events = []
    ref_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0, captured_layer=2),
    ])
    cand_engine = FakeEngine([RuntimeError("stop after reference")])
    tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert ref_engine.calls[0]["topk"] == 0


def test_reference_capture_failure_never_touches_candidate_loader():
    events = []
    ref_engine = FakeEngine([RuntimeError("engine boom")])
    cand_engine = FakeEngine([])
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is False
    assert "reference capture failed" in out["error"] and "engine boom" in out["error"]
    assert events == ["enter:ref", "exit:ref"]


def test_reference_capture_missing_row_refuses_without_touching_candidate():
    events = []
    ref_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0, captured_layer=None),
    ])
    cand_engine = FakeEngine([])
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is False
    assert "reference capture produced no row" in out["error"]
    assert events == ["enter:ref", "exit:ref"]
    assert cand_engine.calls == []


def test_candidate_self_capture_missing_row_still_tears_down_candidate():
    events = []
    ref_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0, captured_layer=2),
    ])
    cand_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0, captured_layer=None),
    ])
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is False
    assert "candidate self-capture produced no row" in out["error"]
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]
    assert len(cand_engine.calls) == 1        # never proceeded to any arm


def test_candidate_arm_failure_mid_sequence_still_tears_down_candidate():
    events = []
    ref_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0, captured_layer=2),
    ])
    cand_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0, captured_layer=2),  # baseline
        RuntimeError("reference_transplant arm boom"),                                              # arm 1
    ])
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is False
    assert "reference_transplant arm failed" in out["error"]
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]
    assert len(cand_engine.calls) == 2        # baseline + the one failed arm call, nothing further


# ==================================================================================== run_site(): happy path

def _happy_engines(*, self_flips=False, random_flips=False, write_applied=True, baseline_top1=9):
    """A scenario where the reference transplant flips the candidate's top-1 to target_token_id=5 and
    (by default) the random-equal-norm control does not -- the textbook reference-specific case."""
    ref_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=5, top1_piece="y", top1_logprob=-0.05, captured_layer=2),
    ])
    baseline = _resp(sum_logprob=-1.0, top1_id=baseline_top1, top1_piece="x", top1_logprob=-1.0,
                     second_id=5, second_piece="y", second_logprob=-2.0, captured_layer=2)
    ref_transplant = _resp(sum_logprob=-0.1, top1_id=5, top1_piece="y", top1_logprob=-0.1,
                           second_id=9, second_piece="x", second_logprob=-1.5, write_applied=write_applied)
    self_top1, self_second = (5, 9) if self_flips else (baseline_top1, 5)
    self_transplant = _resp(sum_logprob=-1.0, top1_id=self_top1, top1_piece="y" if self_flips else "x",
                            top1_logprob=-1.0 if not self_flips else -0.2,
                            second_id=self_second, second_piece="x" if self_flips else "y",
                            second_logprob=-2.0, write_applied=write_applied)
    rand_top1 = 5 if random_flips else baseline_top1
    rand_arm = _resp(sum_logprob=-1.1, top1_id=rand_top1, top1_piece="y" if random_flips else "x",
                     top1_logprob=-1.1 if not random_flips else -0.2,
                     second_id=9 if random_flips else 5, second_piece="x" if random_flips else "y",
                     second_logprob=-2.1, write_applied=write_applied)
    shuffled_arm = _resp(sum_logprob=-1.05, top1_id=baseline_top1, top1_piece="x", top1_logprob=-1.05,
                         second_id=5, second_piece="y", second_logprob=-2.05, write_applied=write_applied)
    no_write = _resp(sum_logprob=-1.0, top1_id=baseline_top1, top1_piece="x", top1_logprob=-1.0,
                     second_id=5, second_piece="y", second_logprob=-2.0)
    cand_engine = FakeEngine([baseline, ref_transplant, self_transplant, rand_arm, shuffled_arm, no_write])
    return ref_engine, cand_engine


def _run_happy(tmp_path, monkeypatch, **scenario_kwargs):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    events = []
    ref_engine, cand_engine = _happy_engines(**scenario_kwargs)
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    return out, ref_engine, cand_engine, events


def test_happy_path_validates_and_arm_order(tmp_path, monkeypatch):
    out, ref_engine, cand_engine, events = _run_happy(tmp_path, monkeypatch)
    assert out["ok"] is True
    doc = out["document"]
    schemas.validate(doc)
    assert doc["schema_version"] == "clozn.transplant.v1"
    assert [arm["name"] for arm in doc["arms"]] == [
        "reference_transplant", "candidate_self_transplant", "random_equal_norm", "shuffled_layer",
        "no_write_replay",
    ]
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]
    assert len(ref_engine.calls) == 1
    assert len(cand_engine.calls) == 6        # baseline + 4 write arms + no_write_replay
    assert "write" not in doc["arms"][4]      # no_write_replay carries no write block at all


def test_happy_path_reference_specific_true(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is True
    assert doc["analysis"]["reference_moved_toward_reference"] is True
    assert doc["analysis"]["random_moved_toward_reference"] is False
    assert doc["analysis"]["reference_specific"] is True


def test_happy_path_random_also_flips_is_not_reference_specific(tmp_path, monkeypatch):
    """The exact control this project's prior transplant-localization study needed to add after its
    first pass overclaimed (docs/research/DISTRIBUTED_FUNCTION.md) -- reproduced here structurally."""
    out, *_ = _run_happy(tmp_path, monkeypatch, random_flips=True)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is True
    assert doc["analysis"]["reference_moved_toward_reference"] is True
    assert doc["analysis"]["random_moved_toward_reference"] is True
    assert doc["analysis"]["reference_specific"] is False


def test_happy_path_instrument_not_sane_when_self_transplant_flips(tmp_path, monkeypatch):
    """If writing the candidate's OWN state back into itself moves the answer, the instrument is
    broken -- reference_specific must be OMITTED, never computed anyway."""
    out, *_ = _run_happy(tmp_path, monkeypatch, self_flips=True)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is False
    assert "reference_specific" not in doc["analysis"]
    assert "reference_moved_toward_reference" not in doc["analysis"]
    assert any("candidate_self_transplant changed the top-1 token" in r for r in doc["analysis"]["reasons"])


def test_happy_path_write_not_applied_marks_instrument_not_sane(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch, write_applied=False)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is False
    assert "reference_specific" not in doc["analysis"]
    assert any("write_applied was not confirmed true" in r for r in doc["analysis"]["reasons"])


def test_happy_path_baseline_already_matches_target(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch, baseline_top1=5)   # baseline ALREADY predicts target=5
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is True
    assert doc["analysis"]["baseline_already_matches_target"] is True
    assert "reference_specific" not in doc["analysis"]
    assert "reference_moved_toward_reference" not in doc["analysis"]


def test_happy_path_tensor_refs_are_stored_and_loadable(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    doc = out["document"]
    ref_arm = next(a for a in doc["arms"] if a["name"] == "reference_transplant")
    entry = ref_arm["write"]["vectors"][0]
    loaded = tensor_store.load_tensor(entry["tensor"])
    assert loaded["ok"] is True
    assert loaded["values"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-5)
    assert loaded["provenance"]["arm"] == "reference_transplant"
    assert loaded["provenance"]["role"] == "residual_transplant_write"


def test_shuffled_layer_arm_writes_the_same_reference_vector_at_a_different_layer(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    doc = out["document"]
    ref_arm = next(a for a in doc["arms"] if a["name"] == "reference_transplant")
    shuf_arm = next(a for a in doc["arms"] if a["name"] == "shuffled_layer")
    assert shuf_arm["write"]["layer"] == 1 and ref_arm["write"]["layer"] == 2
    # same reference vector -> identical content-addressed tensor (same sha256), different layer
    assert (shuf_arm["write"]["vectors"][0]["tensor"]["sha256"]
           == ref_arm["write"]["vectors"][0]["tensor"]["sha256"])


def test_store_tensors_false_omits_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    events = []
    ref_engine, cand_engine = _happy_engines()
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events), store_tensors=False))
    doc = out["document"]
    for arm in doc["arms"]:
        if "write" in arm:
            assert "vectors" not in arm["write"]


def test_random_seed_is_recorded_and_reproducible_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    events1, events2 = [], []
    ref1, cand1 = _happy_engines()
    ref2, cand2 = _happy_engines()
    out1 = tp.run_site(**_run_kwargs(reference_loader=_loader(ref1, "ref", events1),
                                     candidate_loader=_loader(cand1, "cand", events1), seed=99))
    out2 = tp.run_site(**_run_kwargs(reference_loader=_loader(ref2, "ref", events2),
                                     candidate_loader=_loader(cand2, "cand", events2), seed=99))
    assert out1["document"]["random_seed"] == 99 == out2["document"]["random_seed"]
    rand1 = next(a for a in out1["document"]["arms"] if a["name"] == "random_equal_norm")
    rand2 = next(a for a in out2["document"]["arms"] if a["name"] == "random_equal_norm")
    assert (rand1["write"]["vectors"][0]["tensor"]["sha256"]
           == rand2["write"]["vectors"][0]["tensor"]["sha256"])


def test_ffn_hook_site_uses_ffn_capture_and_ffn_write_wire_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    events = []
    ref_engine = FakeEngine([
        _resp(sum_logprob=-1.0, top1_id=5, top1_piece="y", top1_logprob=-0.05,
             captured_layer=0, captured_field="ffn_captured"),
    ])
    baseline = _resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0,
                     second_id=5, second_piece="y", second_logprob=-2.0,
                     captured_layer=0, captured_field="ffn_captured")
    write_resp = _resp(sum_logprob=-0.5, top1_id=9, top1_piece="x", top1_logprob=-0.5,
                       second_id=5, second_piece="y", second_logprob=-1.0,
                       write_applied=True, applied_field="ffn_write_applied")
    cand_engine = FakeEngine([baseline] + [write_resp] * 4 + [baseline])
    out = tp.run_site(**_run_kwargs(
        site={"hook": "ffn", "layer": 0},          # layer 0 IS valid for ffn (unlike residual)
        shuffled_layer=1,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is True
    # capture wire shape
    assert "ffn_capture_layers" in ref_engine.calls[0] and "capture_layers" not in ref_engine.calls[0]
    assert "ffn_capture_layers" in cand_engine.calls[0]
    # write wire shape on every write-bearing arm (calls[1:5])
    for call in cand_engine.calls[1:5]:
        assert "ffn_write" in call and "write" not in call
    doc = out["document"]
    ref_arm = next(a for a in doc["arms"] if a["name"] == "reference_transplant")
    assert ref_arm["metrics"]["write_applied"] is True
    assert doc["site"] == {"hook": "ffn", "layer": 0}


# ==================================================================================== run_site(): head-site happy path

def _happy_head_engines(*, self_flips=False, random_flips=False, write_applied=True, baseline_top1=9):
    """The head-site mirror of _happy_engines: reference/candidate rows are FULL [ne0]=4 merged rows
    (n_head=2, d_head=2); _HEAD_SITE's head=1 selects the SECOND half of each row -- reference [10,20,
    30,40] slices to [30,40] (norm 50), candidate baseline [1,2,3,4] slices to [3,4]."""
    ref_engine = FakeEngine([
        _head_capture_resp(sum_logprob=-1.0, top1_id=5, top1_piece="y", top1_logprob=-0.05,
                           layer=2, full_row=[10.0, 20.0, 30.0, 40.0]),
    ])
    baseline = _head_capture_resp(sum_logprob=-1.0, top1_id=baseline_top1, top1_piece="x", top1_logprob=-1.0,
                                  second_id=5, second_piece="y", second_logprob=-2.0,
                                  layer=2, full_row=[1.0, 2.0, 3.0, 4.0])
    ref_transplant = _head_write_resp(sum_logprob=-0.1, top1_id=5, top1_piece="y", top1_logprob=-0.1,
                                      second_id=9, second_piece="x", second_logprob=-1.5,
                                      write_applied=write_applied)
    self_top1, self_second = (5, 9) if self_flips else (baseline_top1, 5)
    self_transplant = _head_write_resp(sum_logprob=-1.0, top1_id=self_top1,
                                       top1_piece="y" if self_flips else "x",
                                       top1_logprob=-1.0 if not self_flips else -0.2,
                                       second_id=self_second, second_piece="x" if self_flips else "y",
                                       second_logprob=-2.0, write_applied=write_applied)
    rand_top1 = 5 if random_flips else baseline_top1
    rand_arm = _head_write_resp(sum_logprob=-1.1, top1_id=rand_top1, top1_piece="y" if random_flips else "x",
                                top1_logprob=-1.1 if not random_flips else -0.2,
                                second_id=9 if random_flips else 5, second_piece="x" if random_flips else "y",
                                second_logprob=-2.1, write_applied=write_applied)
    shuffled_arm = _head_write_resp(sum_logprob=-1.05, top1_id=baseline_top1, top1_piece="x",
                                    top1_logprob=-1.05, second_id=5, second_piece="y", second_logprob=-2.05,
                                    write_applied=write_applied)
    no_write = _resp(sum_logprob=-1.0, top1_id=baseline_top1, top1_piece="x", top1_logprob=-1.0,
                     second_id=5, second_piece="y", second_logprob=-2.0)
    cand_engine = FakeEngine([baseline, ref_transplant, self_transplant, rand_arm, shuffled_arm, no_write])
    return ref_engine, cand_engine


def _run_happy_head(tmp_path, monkeypatch, **scenario_kwargs):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    events = []
    ref_engine, cand_engine = _happy_head_engines(**scenario_kwargs)
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site=_HEAD_SITE, shuffled_layer=1,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    return out, ref_engine, cand_engine, events


def test_head_site_uses_head_capture_and_head_write_wire_fields(tmp_path, monkeypatch):
    out, ref_engine, cand_engine, events = _run_happy_head(tmp_path, monkeypatch)
    assert out["ok"] is True
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]
    # capture wire shape: head_capture_*, never capture_layers/ffn_capture_layers
    assert ref_engine.calls[0]["head_capture_layers"] == [2]
    assert ref_engine.calls[0]["head_capture_positions"] == [2]
    assert ref_engine.calls[0]["head_capture_rows"] is True
    assert "capture_layers" not in ref_engine.calls[0] and "ffn_capture_layers" not in ref_engine.calls[0]
    assert cand_engine.calls[0]["head_capture_layers"] == [2]
    # write wire shape on every write-bearing arm (calls[1:5]): head_write, carrying "head"
    for call in cand_engine.calls[1:5]:
        assert "head_write" in call and "write" not in call and "ffn_write" not in call
        assert call["head_write"][0]["head"] == 1
    doc = out["document"]
    ref_arm = next(a for a in doc["arms"] if a["name"] == "reference_transplant")
    assert ref_arm["metrics"]["write_applied"] is True
    assert ref_arm["write"]["head"] == 1
    assert doc["site"] == {"hook": "head", "layer": 2, "head": 1}
    schemas.validate(doc)


def test_head_site_random_equal_norm_scaled_to_per_head_slice_not_full_residual(tmp_path, monkeypatch):
    """The exact requirement this addendum calls out: the random-equal-norm control for a head site must
    be scaled to the d_head-wide per-head slice's own norm, not the full n_embd residual's."""
    out, *_ = _run_happy_head(tmp_path, monkeypatch)
    doc = out["document"]
    ref_arm = next(a for a in doc["arms"] if a["name"] == "reference_transplant")
    rand_arm = next(a for a in doc["arms"] if a["name"] == "random_equal_norm")
    ref_vec = tensor_store.load_tensor(ref_arm["write"]["vectors"][0]["tensor"])["values"]
    rand_vec = tensor_store.load_tensor(rand_arm["write"]["vectors"][0]["tensor"])["values"]
    assert ref_vec == pytest.approx([30.0, 40.0], abs=1e-5)   # full_row[2:4] -- head=1, d_head=2
    assert len(ref_vec) == 2      # d_head, NOT ne0=4
    assert len(rand_vec) == 2
    assert tp._norm(rand_vec) == pytest.approx(tp._norm(ref_vec), abs=1e-5)   # == 50.0


def test_head_site_reference_specific_true(tmp_path, monkeypatch):
    out, *_ = _run_happy_head(tmp_path, monkeypatch)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is True
    assert doc["analysis"]["reference_moved_toward_reference"] is True
    assert doc["analysis"]["random_moved_toward_reference"] is False
    assert doc["analysis"]["reference_specific"] is True


def test_head_site_random_also_flips_is_not_reference_specific(tmp_path, monkeypatch):
    out, *_ = _run_happy_head(tmp_path, monkeypatch, random_flips=True)
    doc = out["document"]
    assert doc["analysis"]["reference_moved_toward_reference"] is True
    assert doc["analysis"]["random_moved_toward_reference"] is True
    assert doc["analysis"]["reference_specific"] is False


def test_head_site_instrument_not_sane_when_self_transplant_flips(tmp_path, monkeypatch):
    """instrument_sane/reference_specific gating is NOT bypassable for the head kind -- it runs through
    the exact same _derive_analysis() this module uses for residual/ffn sites."""
    out, *_ = _run_happy_head(tmp_path, monkeypatch, self_flips=True)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is False
    assert "reference_specific" not in doc["analysis"]
    assert "reference_moved_toward_reference" not in doc["analysis"]


def test_head_site_write_not_applied_marks_instrument_not_sane(tmp_path, monkeypatch):
    out, *_ = _run_happy_head(tmp_path, monkeypatch, write_applied=False)
    doc = out["document"]
    assert doc["analysis"]["instrument_sane"] is False
    assert "reference_specific" not in doc["analysis"]
    assert any("write_applied was not confirmed true" in r for r in doc["analysis"]["reasons"])


def test_head_reference_capture_d_head_zero_refuses_without_touching_candidate():
    events = []
    ref_engine = FakeEngine([
        _head_capture_resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0,
                           layer=2, full_row=[1.0, 2.0, 3.0, 4.0], n_head=2, d_head=0),
    ])
    cand_engine = FakeEngine([])
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site=_HEAD_SITE, shuffled_layer=1,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is False
    assert "reference capture produced no row at layer=2, head=1, positions=[2]" in out["error"]
    assert events == ["enter:ref", "exit:ref"]
    assert cand_engine.calls == []


def test_head_candidate_self_capture_missing_row_still_tears_down_candidate():
    events = []
    ref_engine = FakeEngine([
        _head_capture_resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0,
                           layer=2, full_row=[10.0, 20.0, 30.0, 40.0]),
    ])
    cand_engine = FakeEngine([
        _head_capture_resp(sum_logprob=-1.0, top1_id=9, top1_piece="x", top1_logprob=-1.0,
                           layer=2, full_row=[1.0, 2.0, 3.0, 4.0], n_head=2, d_head=0),
    ])
    out = tp.run_site(**_run_kwargs(
        pair_compat=_COMPATIBLE_HEAD, site=_HEAD_SITE, shuffled_layer=1,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is False
    assert "candidate self-capture produced no row at layer=2, head=1" in out["error"]
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]
    assert len(cand_engine.calls) == 1


def test_validate_false_skips_schema_check(monkeypatch, tmp_path):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    calls = []
    monkeypatch.setattr(tp.schemas, "validate", lambda doc: calls.append(doc))
    events = []
    ref_engine, cand_engine = _happy_engines()
    out = tp.run_site(**_run_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events), validate=False))
    assert out["ok"] is True
    assert calls == []


# ==================================================================================== no verdict language

_BANNED = ("caused", "because", "responsible for", "localiz", "distributed")

# Legitimate citations that happen to CONTAIN a banned stem as a substring of a proper-noun file path --
# not a verdict claim. "distribut" is deliberately NOT banned as a bare stem (only the exact word
# "distributed"): this module's own vocabulary legitimately needs "distribution" (a token/output
# probability distribution), an unrelated word that would otherwise collide with the stem.
_FILENAME_CITATIONS = ("transplant_localize.py", "DISTRIBUTED_FUNCTION.md")


def test_document_never_contains_banned_vocabulary(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    text = json.dumps(out["document"]).lower()
    for word in _BANNED:
        assert word not in text, f"banned vocabulary {word!r} leaked into the transplant artifact"


def test_module_source_never_contains_banned_vocabulary():
    """The module docstring legitimately NAMES the banned words in quotes to state the rule against
    using them, and legitimately CITES scripts/tracer/transplant_localize.py and
    docs/research/DISTRIBUTED_FUNCTION.md by their real filenames (the prior transplant study this
    module's control rule is modeled on) -- both stripped first. What must never appear afterward is the
    bare word used as this module's OWN verdict vocabulary: in a docstring sentence, a comment, or
    (most importantly) a message string this module could actually emit into a stored artifact."""
    path = os.path.join(REPO_ROOT, "clozn", "analysis", "transplant.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    remainder = text
    for citation in _FILENAME_CITATIONS:
        assert citation in remainder, f"expected the module to cite {citation} by its real filename"
        remainder = remainder.replace(citation, "")
    quoted_mentions = ('"localize"/"localized"/"localizing"', '"distributed"')
    for mention in quoted_mentions:
        assert mention in remainder, f"expected the module docstring to name {mention} as banned vocabulary"
        remainder = remainder.replace(mention, "", 1)
    lowered = remainder.lower()
    for word in _BANNED:
        if word in ("caused", "because", "responsible for"):
            continue     # this module never claims these at all, quoted or not -- nothing to strip first
        assert word not in lowered, f"banned vocabulary {word!r} used outside its citations/quoted mention"
