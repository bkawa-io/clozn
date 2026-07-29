"""test_causal_bisect -- clozn/analysis/causal_bisect.py (`clozn.causal-bisect.v1`), slice 3.5: the
coarse-to-fine causal bisect built on top of `clozn.analysis.transplant`'s single-site five-arm primitive.

Model-free throughout (roadmap rule 8): the window-phase engine is a `FakeEngine` returning hand-built
`/score`-shaped dicts (mirrors `tests/test_transplant.py`'s own fixture exactly), and the single-site
confirmation step -- which this module delegates to `clozn.analysis.transplant.run_site()` -- is stubbed
directly (monkeypatching `causal_bisect.transplant.run_site`) rather than simulated through a fake engine,
per this slice's own instruction to "stub the transplant layer, do not boot an engine."
"""
from __future__ import annotations

import contextlib
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from clozn import schemas  # noqa: E402
from clozn.analysis import causal_bisect as cb  # noqa: E402


# ==================================================================================== fakes / fixtures

class FakeEngine:
    """Identical contract to tests/test_transplant.py's own FakeEngine: `responses` is consumed in
    order, one item per `.score(...)` call, either a dict to return or an exception instance to raise."""

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
    "hidden_size": {"state": "same", "value_a": 1, "value_b": 1},
    "layer_count": {"state": "same", "value_a": 4, "value_b": 4},
    "verdict": {
        "overall": "compatible", "reasons": [],
        "operations": {
            "per_token_comparison": {"permitted": True, "reason": "tokenizers match exactly."},
            "residual_transplant": {"permitted": True, "reason": "hidden_size matches exactly (1)."},
        },
    },
}


def _resp(*, top1_id, top1_piece, top1_logprob, target_id=None, target_piece=None, target_logprob=None,
          sum_logprob=-1.0, n_prompt=2, n_cont=1, captured=None, applied=None,
          applied_field="ffn_write_applied"):
    topk = [{"id": top1_id, "piece": top1_piece, "logprob": top1_logprob}]
    if target_id is not None and target_id != top1_id:
        topk.append({"id": target_id, "piece": target_piece, "logprob": target_logprob})
    out = {"n_prompt": n_prompt, "n_cont": n_cont,
          "tokens": [{"id": top1_id, "piece": top1_piece, "logprob": top1_logprob, "topk": topk}],
          "sum_logprob": sum_logprob}
    if captured is not None:
        out["ffn_captured"] = captured
    if applied is not None:
        out[applied_field] = applied
    return out


def _captured(layers, position, value):
    return {str(l): {str(position): [float(value)]} for l in layers}


def _base_kwargs(**over):
    base = dict(
        pair_compat=_COMPATIBLE, prompt_ids=[1, 2], continuation_ids=[9], write_positions=[2],
        readout_position=2, target_token_id=5, primary_metric="reference_token_logprob_recovery",
        topk=3, seed=0,
    )
    base.update(over)
    return base


# ==================================================================================== pure helpers

def test_writable_range_ffn_includes_layer_zero():
    assert cb._writable_range("ffn", 8) == (0, 8)


def test_writable_range_residual_excludes_layer_zero_and_final_layer():
    assert cb._writable_range("residual", 8) == (1, 8)


def test_writable_range_head_matches_ffn():
    assert cb._writable_range("head", 8) == (0, 8)


def test_tile_splits_into_window_size_chunks_last_partial():
    assert cb._tile([0, 1, 2, 3, 4, 5, 6], 4) == [[0, 1, 2, 3], [4, 5, 6]]


def test_tile_window_size_one_is_all_singletons():
    assert cb._tile([0, 1, 2], 1) == [[0], [1], [2]]


def test_pick_shuffled_layers_returns_disjoint_same_size_set():
    picked = cb._pick_shuffled_layers([0, 1], [0, 1, 2, 3])
    assert picked == [2, 3]


def test_pick_shuffled_layers_none_when_no_room():
    assert cb._pick_shuffled_layers([0, 1, 2, 3], [0, 1, 2, 3]) is None


def test_pick_any_other_layer_avoids_the_given_layer():
    assert cb._pick_any_other_layer(0, 0, 4) == 1
    assert cb._pick_any_other_layer(2, 0, 4) == 0


def test_pick_any_other_layer_none_when_range_too_small():
    assert cb._pick_any_other_layer(0, 0, 1) is None


def test_random_equal_norm_vector_matches_reference_norm():
    ref = [3.0, 4.0, 0.0, 0.0]  # norm = 5
    rnd = cb._random_equal_norm_vector(ref, random.Random(0))
    assert cb._norm(rnd) == pytest.approx(5.0, abs=1e-6)


def test_flipped_to_target_true_when_baseline_missed_and_arm_hits():
    assert cb._flipped_to_target({"top1_is_target": False}, {"top1_is_target": True}) is True


def test_flipped_to_target_none_when_missing():
    assert cb._flipped_to_target({}, {"top1_is_target": True}) is None


def test_read_captured_multi_reports_none_for_missing_layer():
    response = {"ffn_captured": {"0": {"2": [1.0]}}}
    out = cb._read_captured_multi(response, "ffn_captured", [0, 1], [2])
    assert out[0][2] == [1.0]
    assert out[1][2] is None


def test_read_arm_metrics_reads_top1_and_target():
    response = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, target_id=5, target_piece="y",
                     target_logprob=-2.0)
    out = cb._read_arm_metrics(response, n_prompt=2, n_cont=1, readout_position=2, target_token_id=5)
    assert out["metrics"]["top1_token_id"] == 9
    assert out["metrics"]["top1_is_target"] is False
    assert out["metrics"]["target_token_logprob"] == pytest.approx(-2.0)


# ==================================================================================== _derive_verdict

def _sane_site(hook, layer, *, moved, beat_control):
    return {"kind": "site", "hook": hook, "layer": layer, "instrument_sane": True, "moved": moved,
           "beat_control": beat_control}


def _sane_window(hook, layers, depth, *, moved, beat_control):
    return {"kind": "window", "hook": hook, "layers": layers, "depth": depth, "instrument_sane": True,
           "moved": moved, "beat_control": beat_control}


def test_derive_verdict_localized_site_when_a_site_beats_control():
    window_tests = [{"hook": "ffn", "layers": [0, 1, 2, 3], "depth": 0, "instrument_sane": True,
                     "moved": True, "beat_control": True, "retained": True, "reasons": []}]
    single_site_tests = [
        {"hook": "ffn", "layer": 2, "source": "bisection_leaf", "ok": True,
         "transplant": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": True,
                                     "reference_specific": True}}},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=single_site_tests,
                           composable_kinds_searched={"ffn"}, search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "localized_site"
    assert v["evidence"]["sites"] == [{"hook": "ffn", "layer": 2}]


def test_derive_verdict_localized_window_when_a_narrowed_window_beats_control_and_no_site_does():
    window_tests = [
        {"hook": "ffn", "layers": [0, 1, 2, 3], "depth": 0, "instrument_sane": True, "moved": True,
         "beat_control": True, "retained": True, "reasons": []},
        {"hook": "ffn", "layers": [0, 1], "depth": 1, "instrument_sane": True, "moved": False,
         "beat_control": False, "retained": False, "reasons": []},
        {"hook": "ffn", "layers": [2, 3], "depth": 1, "instrument_sane": True, "moved": True,
         "beat_control": True, "retained": True, "reasons": []},
    ]
    single_site_tests = [
        {"hook": "ffn", "layer": 2, "source": "bisection_leaf", "ok": True,
         "transplant": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": False,
                                     "reference_specific": False}}},
        {"hook": "ffn", "layer": 3, "source": "bisection_leaf", "ok": True,
         "transplant": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": False,
                                     "reference_specific": False}}},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=single_site_tests,
                           composable_kinds_searched={"ffn"}, search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "localized_window"
    assert v["evidence"]["windows"] == [{"hook": "ffn", "layers": [2, 3]}]


def test_derive_verdict_distributed_restoration_when_only_the_coarse_window_beats_control():
    """The broad, UNBISECTED (depth=0) window beats control; the narrower window/sites tested inside it
    do not -- the exact 'broad restores, no narrow subset does' pattern distributed_restoration names."""
    window_tests = [
        {"hook": "ffn", "layers": [0, 1, 2, 3], "depth": 0, "instrument_sane": True, "moved": True,
         "beat_control": True, "retained": True, "reasons": []},
        {"hook": "ffn", "layers": [0, 1], "depth": 1, "instrument_sane": True, "moved": False,
         "beat_control": False, "retained": False, "reasons": []},
        {"hook": "ffn", "layers": [2, 3], "depth": 1, "instrument_sane": True, "moved": False,
         "beat_control": False, "retained": False, "reasons": []},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=[],
                           composable_kinds_searched={"ffn"}, search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "distributed_restoration"
    assert v["evidence"]["windows"] == [{"hook": "ffn", "layers": [0, 1, 2, 3]}]


def test_derive_verdict_distributed_restoration_asserts_without_composable_kinds():
    """Defense-in-depth: even if window_tests somehow carried a depth-0 beaten window while
    composable_kinds_searched claims nothing composable ran (an internal inconsistency that should never
    happen from real run_bisect() data), the module refuses to emit distributed_restoration silently --
    it raises rather than mislabel."""
    window_tests = [
        {"hook": "ffn", "layers": [0, 1, 2, 3], "depth": 0, "instrument_sane": True, "moved": True,
         "beat_control": True, "retained": True, "reasons": []},
    ]
    with pytest.raises(AssertionError):
        cb._derive_verdict(window_tests=window_tests, single_site_tests=[], composable_kinds_searched=set(),
                           search_kinds=("ffn",), hooks_unavailable=[])


def test_derive_verdict_perturbation_sensitive_when_nothing_beats_control_but_something_moved():
    """REQUIRED PROOF: a perturbation-sensitive case (reference AND random both flip the answer) must
    never be reported as any localized_* verdict."""
    window_tests = [
        {"hook": "ffn", "layers": [0, 1], "depth": 0, "instrument_sane": True, "moved": True,
         "beat_control": False, "retained": False, "reasons": ["the random equal-norm control ALSO "
                                                                "flipped..."]},
    ]
    single_site_tests = [
        {"hook": "ffn", "layer": 5, "source": "explicit_residual", "ok": True,
         "transplant": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": True,
                                     "reference_specific": False}}},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=single_site_tests,
                           composable_kinds_searched={"ffn"}, search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "perturbation_sensitive"
    assert v["label"] not in ("localized_site", "localized_window", "distributed_restoration")


def test_derive_verdict_residual_only_search_cannot_return_distributed_restoration():
    """REQUIRED PROOF: with search_kinds=("residual",), window_tests is ALWAYS empty (residual sites are
    never windowed -- run_bisect() never populates window_tests for a residual-only search), so
    distributed_restoration is unreachable no matter how many single sites "look" broad or extreme."""
    single_site_tests = [
        {"hook": "residual", "layer": layer, "source": "explicit_residual", "ok": True,
         "transplant": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": True,
                                     "reference_specific": True}}}
        for layer in (1, 2, 3, 4, 5, 6)
    ]
    v = cb._derive_verdict(window_tests=[], single_site_tests=single_site_tests,
                           composable_kinds_searched=set(), search_kinds=("residual",), hooks_unavailable=[])
    assert v["label"] != "distributed_restoration"
    assert v["label"] == "localized_site"      # sites beat control -> localized_site, never distributed


def test_derive_verdict_inconclusive_when_instrument_insane_everywhere():
    """REQUIRED PROOF: an insane instrument yields inconclusive, never a substantive verdict."""
    window_tests = [
        {"hook": "ffn", "layers": [0, 1], "depth": 0, "instrument_sane": False, "retained": False,
         "reasons": ["candidate_self_transplant changed the top-1 token..."]},
    ]
    single_site_tests = [
        {"hook": "ffn", "layer": 5, "source": "bisection_leaf", "ok": True,
         "transplant": {"analysis": {"instrument_sane": False}}},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=single_site_tests,
                           composable_kinds_searched={"ffn"}, search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "inconclusive"


def test_derive_verdict_inconclusive_mixed_sane_and_insane_still_uses_the_sane_subset():
    """instrument_sane is checked PER result, not globally: one insane window must not block a verdict
    the SANE evidence already supports."""
    window_tests = [
        {"hook": "ffn", "layers": [0, 1], "depth": 0, "instrument_sane": False, "retained": False,
         "reasons": ["x"]},
    ]
    single_site_tests = [
        {"hook": "ffn", "layer": 5, "source": "bisection_leaf", "ok": True,
         "transplant": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": True,
                                     "reference_specific": True}}},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=single_site_tests,
                           composable_kinds_searched={"ffn"}, search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "localized_site"


def test_derive_verdict_no_restoration_when_sane_but_nothing_moved():
    window_tests = [
        {"hook": "ffn", "layers": [0, 1], "depth": 0, "instrument_sane": True, "moved": False,
         "beat_control": False, "retained": False, "reasons": ["x"]},
    ]
    v = cb._derive_verdict(window_tests=window_tests, single_site_tests=[], composable_kinds_searched={"ffn"},
                           search_kinds=("ffn",), hooks_unavailable=[])
    assert v["label"] == "no_restoration"


def test_derive_verdict_unavailable_when_every_requested_kind_is_unavailable():
    v = cb._derive_verdict(window_tests=[], single_site_tests=[], composable_kinds_searched=set(),
                           search_kinds=("ffn",),
                           hooks_unavailable=[{"hook": "ffn", "reason": "ffn_out absent on this architecture"}])
    assert v["label"] == "unavailable"


def test_derive_verdict_inconclusive_when_nothing_tested_and_not_all_kinds_unavailable():
    v = cb._derive_verdict(window_tests=[], single_site_tests=[], composable_kinds_searched=set(),
                           search_kinds=("residual",), hooks_unavailable=[])
    assert v["label"] == "inconclusive"


# ==================================================================================== _run_window

_USABLE = [0, 1, 2, 3, 4, 5]


def _window_kwargs(**over):
    base = dict(
        hook="ffn", layers=[0, 1], depth=0,
        ref_vectors_by_layer={l: {2: [1.0]} for l in _USABLE},
        self_vectors_by_layer={l: {2: [0.5]} for l in _USABLE},
        usable_layers=_USABLE, baseline_metrics={"top1_token_id": 9, "top1_is_target": False},
        positions=[2], prompt_ids=[1, 2], continuation_ids=[9], n_prompt=2, n_cont=1, readout_position=2,
        target_token_id=5, topk=3, rng=random.Random(0), reference_target_logprob=None,
        primary_metric="reference_token_logprob_recovery",
    )
    base.update(over)
    return base


def test_run_window_retained_when_reference_flips_and_random_does_not():
    engine = FakeEngine([
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),                     # reference_transplant
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True),       # candidate_self_transplant
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),                     # random_equal_norm
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05),                    # shuffled_window
    ])
    result = cb._run_window(candidate_engine=engine, **_window_kwargs())
    assert result["instrument_sane"] is True
    assert result["moved"] is True
    assert result["beat_control"] is True
    assert result["retained"] is True
    assert "shuffled_window" in result["arms"]
    assert len(engine.calls) == 4
    for call in engine.calls:
        assert "ffn_write" in call


def test_run_window_not_retained_when_random_also_flips_perturbation_sensitive():
    engine = FakeEngine([
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),                    # reference flips
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True),      # self: sane
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.2),                    # random ALSO flips
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05),                   # shuffled
    ])
    result = cb._run_window(candidate_engine=engine, **_window_kwargs())
    assert result["instrument_sane"] is True
    assert result["moved"] is True
    assert result["beat_control"] is False
    assert result["retained"] is False


def test_run_window_instrument_not_sane_when_self_transplant_flips():
    engine = FakeEngine([
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.2, applied=True),   # self ALSO flips: broken instrument
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05),
    ])
    result = cb._run_window(candidate_engine=engine, **_window_kwargs())
    assert result["instrument_sane"] is False
    assert result["retained"] is False
    assert "beat_control" not in result
    assert "moved" not in result


def test_run_window_instrument_not_sane_when_write_not_applied():
    engine = FakeEngine([
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=False),   # write never confirmed
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05),
    ])
    result = cb._run_window(candidate_engine=engine, **_window_kwargs())
    assert result["instrument_sane"] is False


def test_run_window_omits_shuffled_arm_when_no_disjoint_layers_available():
    engine = FakeEngine([
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),
    ])
    kwargs = _window_kwargs(layers=[0, 1, 2, 3, 4, 5], usable_layers=[0, 1, 2, 3, 4, 5])   # full range: no room
    result = cb._run_window(candidate_engine=engine, **kwargs)
    assert "shuffled_window" not in result["arms"]
    assert len(engine.calls) == 3


def test_run_window_arm_call_failure_is_reported_not_raised():
    engine = FakeEngine([RuntimeError("engine boom")])
    result = cb._run_window(candidate_engine=engine, **_window_kwargs())
    assert result["instrument_sane"] is False
    assert result["retained"] is False
    assert "engine boom" in result["reasons"][0]


def test_run_window_uses_head_write_field_for_head_hook():
    engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True, applied_field="head_write_applied"),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05),
    ])
    result = cb._run_window(candidate_engine=engine, **_window_kwargs(hook="head"))
    assert result["hook"] == "head"
    for call in engine.calls:
        assert "head_write" in call and "ffn_write" not in call


# ==================================================================================== run_bisect(): preflight

def test_refuses_non_dict_pair_compat():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        pair_compat="not a dict",
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert events == []


def test_refuses_when_hidden_size_not_same():
    pair_compat = dict(_COMPATIBLE, hidden_size={"state": "differs", "value_a": 4, "value_b": 8},
                       verdict={"overall": "incompatible", "reasons": [],
                                "operations": {"per_token_comparison": {"permitted": True, "reason": "ok"},
                                               "residual_transplant": {"permitted": False,
                                                                       "reason": "hidden_size differs (4 vs 8)."}}})
    events = []
    out = cb.run_bisect(**_base_kwargs(
        pair_compat=pair_compat,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "hidden_size" in out["error"]
    assert events == []


def test_refuses_empty_search_kinds():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        search_kinds=(),
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "search_kinds must not be empty" in out["error"]


def test_refuses_unknown_search_kind():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        search_kinds=("laser",),
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "subset of" in out["error"]


def test_refuses_window_size_less_than_one():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        window_size=0,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "window_size" in out["error"]


def test_refuses_empty_write_positions():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        write_positions=[],
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "write_positions" in out["error"]


def test_refuses_topk_less_than_one():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        topk=0,
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "topk >= 1" in out["error"]


def test_refuses_empty_continuation():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        continuation_ids=[],
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "non-empty continuation" in out["error"]


def test_refuses_empty_primary_metric():
    events = []
    out = cb.run_bisect(**_base_kwargs(
        primary_metric="",
        reference_loader=_loader(FakeEngine([]), "ref", events),
        candidate_loader=_loader(FakeEngine([]), "cand", events)))
    assert out["ok"] is False
    assert "primary_metric" in out["error"]


# ==================================================================================== run_bisect(): integration

def _happy_ffn_scenario():
    """layer_count=4, window_size=2 -> two coarse windows [0,1] and [2,3]. [0,1] never flips (inert);
    [2,3] flips on the reference arm only and bisects down to leaf layers 2 and 3, which are then
    confirmed via a stubbed transplant.run_site()."""
    ref_engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 1.0)),
    ])
    baseline = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, target_id=5, target_piece="y",
                     target_logprob=-3.0, captured=_captured([0, 1, 2, 3], 2, 0.5))
    # window [0,1]: reference does NOT flip -> not retained, no bisection.
    w01_ref = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0)
    w01_self = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True)
    w01_rand = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1)
    w01_shuf = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05)
    # window [2,3]: reference FLIPS, random does not -> retained -> bisects to leaves 2, 3.
    w23_ref = _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1)
    w23_self = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True)
    w23_rand = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1)
    w23_shuf = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05)
    cand_engine = FakeEngine([baseline, w01_ref, w01_self, w01_rand, w01_shuf,
                              w23_ref, w23_self, w23_rand, w23_shuf])
    return ref_engine, cand_engine


def test_happy_path_localized_site_end_to_end(monkeypatch):
    events = []
    ref_engine, cand_engine = _happy_ffn_scenario()

    def fake_run_site(*, site, **kwargs):
        assert site["hook"] == "ffn"
        if site["layer"] == 2:
            analysis = {"instrument_sane": True, "reference_moved_toward_reference": True,
                       "reference_specific": True, "reasons": ["localizes"]}
        else:
            analysis = {"instrument_sane": True, "reference_moved_toward_reference": False,
                       "reasons": ["does not localize"]}
        return {"ok": True, "document": {"schema_version": "clozn.transplant.v1", "analysis": analysis,
                                         "site": dict(site)}}

    monkeypatch.setattr(cb.transplant, "run_site", fake_run_site)

    out = cb.run_bisect(**_base_kwargs(
        window_size=2,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))

    assert out["ok"] is True
    doc = out["document"]
    schemas.validate(doc)
    assert doc["verdict"]["label"] == "localized_site"
    assert doc["verdict"]["evidence"]["sites"] == [{"hook": "ffn", "layer": 2}]
    assert doc["search"]["composable_kinds_searched"] == ["ffn"]
    assert doc["coverage"]["bounds_applied"]                 # coverage limit always present
    assert len(doc["window_tests"]) == 2                     # both coarse windows tested
    assert {tuple(w["layers"]) for w in doc["window_tests"]} == {(0, 1), (2, 3)}
    assert len(doc["single_site_tests"]) == 2                # leaves 2 and 3 confirmed
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]


def test_happy_path_reference_and_candidate_never_resident_together(monkeypatch):
    """Sequential model orchestration: the candidate loader is never entered before the reference loader
    has fully exited."""
    events = []
    ref_engine, cand_engine = _happy_ffn_scenario()
    monkeypatch.setattr(cb.transplant, "run_site",
                        lambda **kw: {"ok": True,
                                     "document": {"analysis": {"instrument_sane": True}}})
    cb.run_bisect(**_base_kwargs(
        window_size=2,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    ref_exit_index = events.index("exit:ref")
    cand_enter_index = events.index("enter:cand")
    assert ref_exit_index < cand_enter_index


def test_coverage_limit_always_present_across_scenarios(monkeypatch):
    """REQUIRED PROOF: the coverage bound is present regardless of outcome -- exercised across an
    unavailable-hook scenario and a max_windows-capped scenario."""
    events = []
    ref_engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0),   # no ffn_captured at all -> ffn unavailable
    ])
    cand_engine = FakeEngine([])
    out = cb.run_bisect(**_base_kwargs(
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is True
    assert out["document"]["coverage"]["bounds_applied"]
    assert out["document"]["verdict"]["label"] == "unavailable"
    assert out["document"]["search"]["hooks_unavailable"][0]["hook"] == "ffn"


def test_max_windows_caps_coarse_windows_and_records_it(monkeypatch):
    events = []
    ref_engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 1.0)),
    ])
    baseline = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 0.5))
    # only ONE window (window_size=2 -> 2 candidate tiles, capped to max_windows=1) is ever tested, and it
    # is designed to NOT retain so no bisection/single-site calls follow.
    inert_arm = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0)
    self_arm = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True)
    rand_arm = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1)
    shuf_arm = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05)
    cand_engine = FakeEngine([baseline, inert_arm, self_arm, rand_arm, shuf_arm])

    out = cb.run_bisect(**_base_kwargs(
        window_size=2, max_windows=1,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))

    assert out["ok"] is True
    doc = out["document"]
    assert len(doc["window_tests"]) == 1
    assert doc["coverage"]["max_windows"] == 1
    assert any("max_windows=1" in b for b in doc["coverage"]["bounds_applied"])


def test_residual_only_search_never_windows_and_verdict_is_never_distributed(monkeypatch):
    events = []
    ref_engine = FakeEngine([])   # ffn not requested -> reference is never even entered for ffn capture
    cand_engine = FakeEngine([])

    def fake_run_site(*, site, **kwargs):
        assert site["hook"] == "residual"
        return {"ok": True,
               "document": {"analysis": {"instrument_sane": True, "reference_moved_toward_reference": True,
                                        "reference_specific": True}}}

    monkeypatch.setattr(cb.transplant, "run_site", fake_run_site)
    out = cb.run_bisect(**_base_kwargs(
        search_kinds=("residual",), residual_layers=[1, 2, 3],
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))

    assert out["ok"] is True
    doc = out["document"]
    schemas.validate(doc)
    assert doc["window_tests"] == []
    assert doc["search"]["composable_kinds_searched"] == []
    assert doc["verdict"]["label"] != "distributed_restoration"
    assert doc["verdict"]["label"] == "localized_site"


def test_head_hook_refused_by_transplant_is_recorded_as_unavailable_not_a_crash(monkeypatch):
    events = []
    ref_engine = FakeEngine([])
    cand_engine = FakeEngine([])

    def fake_run_site(*, site, **kwargs):
        return {"ok": False, "error": f"site.hook must be one of ['ffn', 'residual'], got {site['hook']!r}"}

    monkeypatch.setattr(cb.transplant, "run_site", fake_run_site)
    out = cb.run_bisect(**_base_kwargs(
        search_kinds=("head",), head_layers=[1, 2],
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))

    assert out["ok"] is True
    doc = out["document"]
    assert doc["verdict"]["label"] == "unavailable"
    assert any(h["hook"] == "head" for h in doc["search"]["hooks_unavailable"])
    assert all(not s["ok"] for s in doc["single_site_tests"])


def test_batched_screen_requested_is_honestly_recorded_as_unused(monkeypatch):
    events = []
    ref_engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0),
    ])
    cand_engine = FakeEngine([])
    out = cb.run_bisect(**_base_kwargs(
        use_batched_screen=True,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is True
    screening = out["document"]["search"]["screening"]
    assert screening["requested"] is True
    assert screening["used"] is False
    assert "reason" in screening


def test_single_site_engine_failure_mid_search_does_not_abort_the_whole_search(monkeypatch):
    """A window-level engine hiccup on ONE window must not take down the whole multi-window search --
    the failing window is recorded as untestable and the rest of the search proceeds."""
    events = []
    ref_engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 1.0)),
    ])
    baseline = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 0.5))
    cand_engine = FakeEngine([
        baseline,
        RuntimeError("window [0,1] boom"),                                       # window [0,1] arm 1 fails
        _resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),                     # window [2,3] reference flips
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True),       # self
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),                     # random
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05),                    # shuffled
    ])
    monkeypatch.setattr(cb.transplant, "run_site",
                        lambda **kw: {"ok": True,
                                     "document": {"analysis": {"instrument_sane": True,
                                                              "reference_moved_toward_reference": True,
                                                              "reference_specific": True}}})
    out = cb.run_bisect(**_base_kwargs(
        window_size=2,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is True
    doc = out["document"]
    failed = [w for w in doc["window_tests"] if w["layers"] == [0, 1]]
    assert failed and failed[0]["instrument_sane"] is False
    assert doc["verdict"]["label"] == "localized_site"


# ==================================================================================== no verdict overclaim

def test_reference_specific_is_read_verbatim_never_recomputed(monkeypatch):
    """Never bypass or recompute clozn.transplant.v1's own reference_specific field -- the stubbed
    document's analysis is embedded byte-for-byte."""
    events = []
    ref_engine = FakeEngine([
        _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 1.0)),
    ])
    baseline = _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, captured=_captured([0, 1, 2, 3], 2, 0.5))
    w01 = [_resp(top1_id=9, top1_piece="x", top1_logprob=-1.0),
          _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True),
          _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),
          _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05)]
    w23 = [_resp(top1_id=5, top1_piece="y", top1_logprob=-0.1),
          _resp(top1_id=9, top1_piece="x", top1_logprob=-1.0, applied=True),
          _resp(top1_id=9, top1_piece="x", top1_logprob=-1.1),
          _resp(top1_id=9, top1_piece="x", top1_logprob=-1.05)]
    cand_engine = FakeEngine([baseline] + w01 + w23)

    sentinel_analysis = {"instrument_sane": True, "reference_moved_toward_reference": True,
                         "reference_specific": True, "reasons": ["sentinel value for identity check"]}

    monkeypatch.setattr(cb.transplant, "run_site",
                        lambda **kw: {"ok": True,
                                     "document": {"schema_version": "clozn.transplant.v1",
                                                 "analysis": dict(sentinel_analysis)}})
    out = cb.run_bisect(**_base_kwargs(
        window_size=2,
        reference_loader=_loader(ref_engine, "ref", events),
        candidate_loader=_loader(cand_engine, "cand", events)))
    assert out["ok"] is True
    confirmed = [s for s in out["document"]["single_site_tests"] if s["ok"]]
    assert confirmed
    for s in confirmed:
        assert s["transplant"]["analysis"]["reasons"] == ["sentinel value for identity check"]
