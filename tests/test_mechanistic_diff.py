"""test_mechanistic_diff -- clozn/analysis/mechanistic_diff.py (`clozn.mechanistic-diff.v1`), slice 3.2.

Model-free throughout (roadmap rule 8): `reference_loader`/`candidate_loader` are fake context-manager
factories wrapping a `FakeEngine` whose `.score(...)` returns hand-built /score-shaped dicts -- no real
engine, no GPU, no network. Sequencing (reference loaded+captured+torn down BEFORE candidate is ever
loaded) is verified directly against an ordered event log, which is the one property a schema fixture
alone could never pin down.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from clozn import schemas  # noqa: E402
from clozn.analysis import mechanistic_diff as md  # noqa: E402
from clozn.analysis import tensor_store  # noqa: E402


# ==================================================================================== fakes

class FakeEngine:
    """Wraps a plain callable so each test controls the exact /score-shaped response (or exception)
    for its own scenario, while still recording every call for assertions."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def score(self, *, prompt_ids=None, continuation_ids=None, topk=0, capture_layers=None,
              capture_positions=None, **_kw):
        call = {"prompt_ids": prompt_ids, "continuation_ids": continuation_ids, "topk": topk,
               "capture_layers": capture_layers, "capture_positions": capture_positions}
        self.calls.append(call)
        return self.responder(call)


def _loader(engine, name, events):
    @contextlib.contextmanager
    def _cm():
        events.append(f"enter:{name}")
        try:
            yield engine
        finally:
            events.append(f"exit:{name}")
    return _cm


def _ok(value):
    return lambda call: value


def _raise(exc):
    def _r(call):
        raise exc
    return _r


_COMPATIBLE = {
    "schema_version": "clozn.pair-compatibility.v1",
    "model_a": {"label": "reference", "filename": "ref.gguf", "sha256": "a" * 64},
    "model_b": {"label": "candidate", "filename": "cand.gguf", "sha256": "b" * 64},
    "tokenizer": {"state": "exact", "method": "hash"},
    "hidden_size": {"state": "same", "value_a": 4, "value_b": 4},
    "layer_count": {"state": "same", "value_a": 6, "value_b": 6},
    "verdict": {
        "overall": "compatible", "reasons": [],
        "operations": {
            "per_token_comparison": {"permitted": True, "reason": "tokenizers match exactly."},
            "residual_transplant": {"permitted": True, "reason": "hidden_size matches."},
        },
    },
}


# ==================================================================================== math primitives

def test_cosine_similarity_identical_vectors_is_one():
    assert md._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    assert md._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_none_not_zero():
    assert md._cosine_similarity([0.0, 0.0], [1.0, 2.0]) is None


def test_l2_normalized_identical_vectors_is_zero():
    assert md._l2_normalized([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_l2_normalized_both_zero_is_none():
    assert md._l2_normalized([0.0, 0.0], [0.0, 0.0]) is None


def test_rank_of_finds_index():
    topk = [{"id": 5}, {"id": 9}, {"id": 3}]
    assert md._rank_of(topk, 9) == 1
    assert md._rank_of(topk, 999) is None
    assert md._rank_of(None, 9) is None


# ==================================================================================== compare(): preflight

def test_refuses_when_tokenizer_not_exact():
    pair_compat = dict(_COMPATIBLE, tokenizer={"state": "differs", "method": "hash"},
                       verdict={"overall": "incompatible", "reasons": [],
                                "operations": {"per_token_comparison": {"permitted": False, "reason": "tokenizers differ."},
                                               "residual_transplant": {"permitted": True, "reason": "ok"}}})
    events = []
    ref_engine = FakeEngine(_ok({}))
    cand_engine = FakeEngine(_ok({}))
    out = md.compare(pair_compat=pair_compat, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1, 2], continuation_ids=[3], layers=[1], positions=[2])
    assert out["ok"] is False
    assert "tokenizers differ" in out["error"]
    assert events == []          # refused before either loader was ever touched
    assert ref_engine.calls == []


def test_refuses_when_hidden_size_not_same():
    pair_compat = dict(_COMPATIBLE, hidden_size={"state": "differs", "value_a": 4, "value_b": 8})
    events = []
    ref_engine = FakeEngine(_ok({}))
    cand_engine = FakeEngine(_ok({}))
    out = md.compare(pair_compat=pair_compat, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1, 2], continuation_ids=[3], layers=[1], positions=[2])
    assert out["ok"] is False
    assert "hidden_size" in out["error"]
    assert events == []


def test_refuses_malformed_pair_compat():
    events = []
    out = md.compare(pair_compat="not a dict", reference_loader=_loader(FakeEngine(_ok({})), "ref", events),
                     candidate_loader=_loader(FakeEngine(_ok({})), "cand", events),
                     prompt_ids=[1], continuation_ids=[2], layers=[1], positions=[1])
    assert out["ok"] is False
    assert events == []


def test_refuses_empty_layers_or_positions():
    events = []
    engine = FakeEngine(_ok({}))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(engine, "ref", events),
                     candidate_loader=_loader(engine, "cand", events),
                     prompt_ids=[1], continuation_ids=[2], layers=[], positions=[1])
    assert out["ok"] is False
    assert events == []


# ==================================================================================== compare(): sequencing

def test_reference_is_fully_torn_down_before_candidate_is_loaded():
    events = []
    ref_engine = FakeEngine(_ok({"n_prompt": 1, "n_cont": 1, "tokens": [], "captured": {}}))
    cand_engine = FakeEngine(_ok({"n_prompt": 1, "n_cont": 1, "tokens": [], "captured": {}}))
    md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
              candidate_loader=_loader(cand_engine, "cand", events),
              prompt_ids=[1], continuation_ids=[2], layers=[1], positions=[1], store_tensors=False,
              validate=False)
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]


def test_reference_capture_failure_never_touches_candidate_loader():
    events = []
    ref_engine = FakeEngine(_raise(RuntimeError("engine boom")))
    cand_engine = FakeEngine(_ok({}))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1], continuation_ids=[2], layers=[1], positions=[1])
    assert out["ok"] is False
    assert "reference capture failed" in out["error"]
    assert "engine boom" in out["error"]
    assert events == ["enter:ref", "exit:ref"]     # candidate loader never entered


def test_candidate_capture_failure_after_reference_already_succeeded():
    events = []
    ref_engine = FakeEngine(_ok({"n_prompt": 1, "n_cont": 1, "tokens": [], "captured": {}}))
    cand_engine = FakeEngine(_raise(RuntimeError("candidate boom")))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1], continuation_ids=[2], layers=[1], positions=[1])
    assert out["ok"] is False
    assert "candidate capture failed" in out["error"]
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]


# ==================================================================================== compare(): happy path

def _vec(*xs):
    return [float(x) for x in xs]


def _happy_responses():
    reference = {
        "n_prompt": 3, "n_cont": 2,
        "tokens": [
            {"id": 10, "piece": "A", "logprob": -0.1,
             "topk": [{"id": 10, "piece": "A", "logprob": -0.1}, {"id": 20, "piece": "B", "logprob": -1.0},
                     {"id": 30, "piece": "C", "logprob": -2.0}]},
            {"id": 11, "piece": "B", "logprob": -0.5,
             "topk": [{"id": 11, "piece": "B", "logprob": -0.5}, {"id": 21, "piece": "X", "logprob": -1.5},
                     {"id": 31, "piece": "Y", "logprob": -2.5}]},
        ],
        "sum_logprob": -0.6,
        "n_embd": 4,
        "captured": {
            "2": {"3": _vec(1, 0, 0, 0), "4": _vec(0, 1, 0, 0)},
            "3": {"3": _vec(1, 0, 0, 0), "4": _vec(0, 1, 0, 0)},
        },
    }
    candidate = {
        "n_prompt": 3, "n_cont": 2,
        "tokens": [
            {"id": 10, "piece": "A", "logprob": -0.3,
             "topk": [{"id": 10, "piece": "A", "logprob": -0.3}, {"id": 20, "piece": "B", "logprob": -0.9},
                     {"id": 30, "piece": "C", "logprob": -2.2}]},
            {"id": 11, "piece": "B", "logprob": -1.2,
             "topk": [{"id": 21, "piece": "X", "logprob": -0.4}, {"id": 11, "piece": "B", "logprob": -1.2},
                     {"id": 31, "piece": "Y", "logprob": -2.1}]},
        ],
        "sum_logprob": -1.5,
        "n_embd": 4,
        "captured": {
            "2": {"3": _vec(0.9, 0.1, 0, 0), "4": _vec(0, 0.9, 0.1, 0)},
            "3": {"3": _vec(0.8, 0.2, 0, 0), "4": _vec(0, 0.8, 0.2, 0)},
        },
    }
    return reference, candidate


def _run_happy(tmp_path, monkeypatch, *, topk=3, store_tensors=True):
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    reference, candidate = _happy_responses()
    events = []
    ref_engine = FakeEngine(_ok(reference))
    cand_engine = FakeEngine(_ok(candidate))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1, 2, 3], continuation_ids=[10, 11], layers=[2, 3], positions=[3, 4],
                     topk=topk, store_tensors=store_tensors)
    return out, ref_engine, cand_engine, events


def test_happy_path_validates_and_has_expected_shape(tmp_path, monkeypatch):
    out, ref_engine, cand_engine, events = _run_happy(tmp_path, monkeypatch)
    assert out["ok"] is True
    doc = out["document"]
    schemas.validate(doc)          # compare() already validated internally; re-assert explicitly
    assert doc["schema_version"] == "clozn.mechanistic-diff.v1"
    assert doc["reference_model"] == _COMPATIBLE["model_a"]
    assert doc["candidate_model"] == _COMPATIBLE["model_b"]
    assert doc["pair_compatibility"] == _COMPATIBLE
    assert doc["continuation"] == {"n_prompt": 3, "n_cont": 2}
    assert events == ["enter:ref", "exit:ref", "enter:cand", "exit:cand"]
    # exactly one capture-bearing /score call per engine
    assert len(ref_engine.calls) == 1 and len(cand_engine.calls) == 1
    assert ref_engine.calls[0]["capture_layers"] == [2, 3]
    assert ref_engine.calls[0]["capture_positions"] == [3, 4]


def test_happy_path_layer_capture_all_true(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    doc = out["document"]
    assert doc["layer_capture"] == [
        {"layer": 2, "reference_captured": True, "candidate_captured": True},
        {"layer": 3, "reference_captured": True, "candidate_captured": True},
    ]
    assert "layer_change" in doc      # two captured layers per position -> at least one consecutive pair
    ref_changes = [c for c in doc["layer_change"] if c["model"] == "reference"]
    assert len(ref_changes) == 2      # one per position (3 and 4), from_layer=2 -> to_layer=3


def test_happy_path_residual_metrics_match_hand_computed_values(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    doc = out["document"]
    reference, candidate = _happy_responses()

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    def l2n(a, b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        return diff / ((na + nb) / 2.0)

    points = {(p["layer"], p["position"]): p for p in doc["residual_points"]}
    assert len(points) == 4
    for layer in (2, 3):
        for position in (3, 4):
            a = reference["captured"][str(layer)][str(position)]
            b = candidate["captured"][str(layer)][str(position)]
            point = points[(layer, position)]
            assert point["metrics"]["residual_cosine_similarity"] == pytest.approx(cos(a, b), abs=1e-6)
            assert point["metrics"]["residual_l2_normalized"] == pytest.approx(l2n(a, b), abs=1e-6)
            assert point["omitted"] == []


def test_happy_path_position_metrics_reference_token():
    reference, candidate = _happy_responses()
    entry = md._position_metrics_entry(3, 3, 2, reference, candidate, topk=3)
    assert entry["reference_token_id"] == 10
    assert entry["metrics"]["reference_token_logit_delta"] == pytest.approx(-0.3 - (-0.1))
    assert entry["metrics"]["reference_token_rank_reference"] == 0
    assert entry["metrics"]["reference_token_rank_candidate"] == 0
    assert entry["metrics"]["reference_token_rank_movement"] == 0
    assert entry["omitted"] == []

    entry4 = md._position_metrics_entry(4, 3, 2, reference, candidate, topk=3)
    assert entry4["reference_token_id"] == 11
    assert entry4["metrics"]["reference_token_rank_reference"] == 0
    assert entry4["metrics"]["reference_token_rank_candidate"] == 1     # id 11 is candidate's 2nd choice
    assert entry4["metrics"]["reference_token_rank_movement"] == 1


def test_happy_path_position_metrics_candidate_token_when_it_differs_from_reference():
    reference, candidate = _happy_responses()
    entry = md._position_metrics_entry(4, 3, 2, reference, candidate, topk=3)
    # candidate's own top-1 at this position is id 21 ("X"), not the forced id 11
    assert entry["candidate_token_id"] == 21
    assert entry["metrics"]["candidate_token_rank_candidate"] == 0
    assert entry["metrics"]["candidate_token_rank_reference"] == 1       # id 21 is reference's 2nd choice
    assert entry["metrics"]["candidate_token_rank_movement"] == -1
    assert entry["metrics"]["candidate_token_logit_delta"] == pytest.approx(-0.4 - (-1.5))
    assert entry["omitted"] == []


def test_happy_path_tensors_stored_and_loadable(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch, store_tensors=True)
    doc = out["document"]
    point = next(p for p in doc["residual_points"] if p["layer"] == 2 and p["position"] == 3)
    assert "tensors" in point
    ref_tensor = tensor_store.load_tensor(point["tensors"]["reference"])
    assert ref_tensor["ok"] is True
    assert ref_tensor["values"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-5)
    assert ref_tensor["provenance"]["layer"] == 2
    assert ref_tensor["provenance"]["position"] == 3
    assert ref_tensor["provenance"]["model"] == "reference"


def test_store_tensors_false_omits_tensors_field(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch, store_tensors=False)
    doc = out["document"]
    for point in doc["residual_points"]:
        assert "tensors" not in point


def test_topk_zero_omits_candidate_token_metrics(tmp_path, monkeypatch):
    # A dedicated response (not _happy_responses(), whose canned topk lists are non-empty regardless of
    # what `topk` compare() was called with -- a fake engine, unlike a real one, doesn't derive its
    # response from the request). This response has no `topk` field at all on either token, exactly
    # what a real engine returns when topk=0 was requested.
    monkeypatch.setattr(tensor_store, "TENSOR_ROOT", str(tmp_path / "tensors"))
    reference = {"n_prompt": 3, "n_cont": 2, "captured": {},
                "tokens": [{"id": 10, "piece": "A", "logprob": -0.1},
                          {"id": 11, "piece": "B", "logprob": -0.5}]}
    candidate = {"n_prompt": 3, "n_cont": 2, "captured": {},
                "tokens": [{"id": 10, "piece": "A", "logprob": -0.3},
                          {"id": 11, "piece": "B", "logprob": -1.2}]}
    events = []
    ref_engine = FakeEngine(_ok(reference))
    cand_engine = FakeEngine(_ok(candidate))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1, 2, 3], continuation_ids=[10, 11], layers=[2], positions=[3],
                     topk=0, store_tensors=False)
    assert out["ok"] is True
    doc = out["document"]
    for entry in doc["position_metrics"]:
        omitted_names = {o["metric"] for o in entry["omitted"]}
        assert "candidate_token_logit_delta" in omitted_names
        reason = next(o["reason"] for o in entry["omitted"] if o["metric"] == "candidate_token_logit_delta")
        assert "topk" in reason


# ==================================================================================== missing-layer honesty

def test_last_layer_missing_is_reported_never_as_zero_divergence():
    reference, candidate = _happy_responses()
    # simulate the engine's own honest report: layer 5 was armed but yielded nothing on both sides
    reference["capture_missing"] = [5]
    candidate["capture_missing"] = [5]
    events = []
    ref_engine = FakeEngine(_ok(reference))
    cand_engine = FakeEngine(_ok(candidate))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1, 2, 3], continuation_ids=[10, 11], layers=[2, 5], positions=[3],
                     store_tensors=False)
    assert out["ok"] is True
    doc = out["document"]
    layer5 = next(entry for entry in doc["layer_capture"] if entry["layer"] == 5)
    assert layer5["reference_captured"] is False
    assert layer5["candidate_captured"] is False
    assert "inp_out_ids" in layer5["note"]
    # no residual_points entry at all for (layer=5, position=3) -- nothing to report, never a fake zero
    assert not any(p["layer"] == 5 for p in doc["residual_points"])


def test_position_outside_continuation_is_recorded_with_reason():
    reference, candidate = _happy_responses()
    entry = md._position_metrics_entry(100, 3, 2, reference, candidate, topk=3)
    assert entry["metrics"] == {}
    assert len(entry["omitted"]) == len(md._POSITION_METRIC_NAMES)
    assert all("outside the scored continuation range" in o["reason"] for o in entry["omitted"])
    assert "reference_token_id" not in entry


# ==================================================================================== no causal language

_BANNED = ("caused", "because", "responsible for", "localiz")   # localize/localized/localization


def test_document_never_contains_causal_vocabulary(tmp_path, monkeypatch):
    out, *_ = _run_happy(tmp_path, monkeypatch)
    text = json.dumps(out["document"]).lower()
    for word in _BANNED:
        assert word not in text, f"causal vocabulary {word!r} leaked into the observational artifact"


def test_module_source_never_contains_causal_vocabulary():
    """A stronger guard than scanning one document: no docstring/comment/message anywhere in the
    module may use this vocabulary either, since ANY string built from it could reach an artifact.

    The module docstring legitimately NAMES the banned words, in quotes, to state the rule against
    using them (see its own warning). This test removes exactly those one-time quoted mentions, then
    asserts the bare words appear nowhere else in the file -- not in another docstring, a comment, or
    (most importantly) a message string this module could actually emit into a stored artifact.
    """
    path = os.path.join(REPO_ROOT, "clozn", "analysis", "mechanistic_diff.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    quoted_mentions = ('"caused"', '"because"', '"responsible for"', '"localized"')
    remainder = text
    for mention in quoted_mentions:
        assert mention in remainder, f"expected the module docstring to name {mention} as banned vocabulary"
        remainder = remainder.replace(mention, "", 1)
    lowered = remainder.lower()
    for word in ("caused", "because", "responsible for", "localiz"):
        assert word not in lowered, f"causal vocabulary {word!r} used outside its one quoted mention"


# ==================================================================================== validate flag

def test_validate_false_skips_schema_check(monkeypatch):
    calls = []
    monkeypatch.setattr(md.schemas, "validate", lambda doc: calls.append(doc))
    events = []
    ref_engine = FakeEngine(_ok({"n_prompt": 1, "n_cont": 1, "tokens": [], "captured": {}}))
    cand_engine = FakeEngine(_ok({"n_prompt": 1, "n_cont": 1, "tokens": [], "captured": {}}))
    out = md.compare(pair_compat=_COMPATIBLE, reference_loader=_loader(ref_engine, "ref", events),
                     candidate_loader=_loader(cand_engine, "cand", events),
                     prompt_ids=[1], continuation_ids=[2], layers=[1], positions=[1],
                     store_tensors=False, validate=False)
    assert out["ok"] is True
    assert calls == []
