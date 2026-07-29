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
        site={"hook": "head", "layer": 2},
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
