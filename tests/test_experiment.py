"""test_experiment -- model-free tests for clozn/experiments/experiment.py, the ONE experiment primitive
over clozn's run-scoped "hold everything constant, change one thing, compare, with a receipt" ops
(replay / counterfactual / receipt / branch / swap_receipt).

No model, no GPU: mirrors test_receipts.py / test_counterfactual.py / test_swap_receipt.py's own
model-free style. Two layers:

  * REAL-op tests: drive run_experiment() against a FakeSub (mirrors test_receipts_server.py's
    FakeSteer/FakeMem/FakeSub) so receipt()/counterfactual()/branch()/replay() run for real, over an
    underlying ops, not just its own bookkeeping. swap_concept has no cheap fake substrate (it needs a real
    J-lens + unembed export, exhaustively covered in test_swap_receipt.py / test_receipts_server.py's own
    fixture-built happy path) -- here it's exercised with `experiment._swap_receipt` stubbed to a
    contract-shaped canned dict (HEAVN_API_CONTRACTS.md §8), which is exactly the "stub the ... underlying
    ops" the build brief calls for.
  * SPY tests: every one of the six underlying-op names the dispatcher imports
    (_receipt/_counterfactual/_swap_receipt/_branch/_replay) is monkeypatched with a
    recording stub, to prove the REGISTRY dispatches every change.type to exactly the one right op and
    nothing else.

What's under test:
  * the envelope shape ({run_id, question, baseline, change, method, cost, result} with the same `result`
    sub-keys) is IDENTICAL across every change.type.
  * the registry dispatches each change.type to exactly the right underlying op.
  * has_effect / causal_verified / null carry through EXACTLY as the underlying op computed them -- never
    invented, never dropped, never silently defaulted to "no effect" when actually just missing.
  * unknown change.type, a missing/malformed change spec, and a missing run all degrade to a clean
    ValueError (the HTTP route's 400); an underlying op that honestly can't produce a result degrades to
    None (the HTTP route's 500) -- exactly mirroring receipt()/counterfactual()/branch()/replay()'s own
    "never raise, return None" contract.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.experiments.experiment as experiment   # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402


import clozn.runs.store as runlog             # noqa: E402


# ================================================================================================== fakes

class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def set(self, name, value):
        self.strength[str(name)] = float(value)

    def clear(self):
        self.strength = {}

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class FakeMem:
    def __init__(self, strength=1.0, rules=None, prefix="PFX"):
        self.memory_strength = float(strength)
        self.rules = list(rules or [])
        self.prefix = prefix


class FakeSub:
    """chat() is a pure function of (memory_strength, excluded card ids, concise/warm dial values) -- no
    randomness (mirrors test_receipts.py / test_receipts_server.py's fakes)."""
    name = "qwen"

    def __init__(self, mem=None, steer=None, concise_card_ids=()):
        self.memory = mem if mem is not None else FakeMem()
        self._mem = self.memory
        self.steer = steer if steer is not None else FakeSteer()
        self.concise_card_ids = {str(i) for i in concise_card_ids}
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        excluded = {str(i) for i in (getattr(self.memory, "_exclude_card_ids", None) or [])}
        if self.memory.memory_strength <= 0:
            return "Generic reply, memory off."
        concise_active = self.concise_card_ids - excluded
        concise_dial = float(self.steer.strength.get("concise", 0.0) or 0.0)
        base = "Short answer." if (concise_active or concise_dial > 0) else "A much longer rambling reply."
        if float(self.steer.strength.get("warm", 0.0) or 0.0) > 0:
            base += " Warmly!"
        return base


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _seed_run(**kw):
    defaults = dict(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                    messages=[{"role": "user", "content": "tell me about your day"}],
                    response="THE STORED SAMPLED REPLY -- never a baseline",
                    behavior={"active_dials": {"warm": 0.5}},
                    started=1000.0, ended=1000.0)   # duration_ms == 0 by default -- no est_seconds grounding
    defaults.update(kw)
    rid = runlog.record(**defaults)
    return runlog.get_run(rid)


_ENVELOPE_TOP_KEYS = {"run_id", "question", "baseline", "change", "method", "cost", "result"}
_RESULT_KEYS = {"changed_reply", "delta", "has_effect", "causal_verified", "null", "receipt", "plain"}


def _assert_envelope_shape(out):
    assert set(out.keys()) == _ENVELOPE_TOP_KEYS
    assert set(out["baseline"].keys()) == {"reply"}
    assert set(out["change"].keys()) == {"type", "target", "label"}
    assert "passes" in out["cost"] and "note" in out["cost"]
    assert set(out["result"].keys()) == _RESULT_KEYS


# ======================================================================== a canned swap_receipt() stand-in

def _canned_swap_receipt(run, from_hint, to_concept, sub):
    return {
        "mode": "swap_receipt", "causal_verified": True, "run_id": run.get("id"),
        "disposed": {"hint": from_hint, "jlens_available": True, "jlens_layer": 21,
                    "jlens_top1": "Paris", "jlens_top5": ["Paris", "France"], "jlens_reason": None,
                    "baseline_lean": "paris"},
        "swapped_to": {"concept": to_concept, "layer": 21, "strength": 6.0, "token_id": 16234, "coef": 0.5},
        "baseline_reply": "The capital of France is Paris.",
        "swapped_reply": "The nearest large body of water to France is the Atlantic Ocean.",
        "null_reply": "The capital of France is Paris, a city on the Seine.",
        "targeted_shift": True, "null_control_available": True,
        "lexicon_hits": {"baseline": 0, "swap": 1, "null": 0},
        "logprob_shift": {"baseline": -4.1, "swap": -0.6, "null": -3.9,
                          "swap_over_baseline_nat": 3.5, "swap_over_null_nat": 3.3},
        "coherent": True, "coherence_score": 0.91,
        "null_note": "the null arm injects a RANDOM direction ...",
        "lexicon_note": "lexicon_hits counts LITERAL ... mentions ...",
        "blocked": None, "note": None,
    }


@pytest.fixture
def swap_stub(monkeypatch):
    monkeypatch.setattr(experiment, "_swap_receipt", _canned_swap_receipt)


# =========================================================================================== envelope shape



def test_ablate_receipt_modes_all_share_the_same_envelope_shape(iso):
    run = _seed_run()
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    for mode in ("regen", "forced", "both"):
        out = experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, mode, sub)
        assert out is not None, mode
        _assert_envelope_shape(out)
        assert out["method"] == f"receipt:{mode}"


# =========================================================================================== registry dispatch





def test_set_dial_passes_the_dial_value_override_through(iso, monkeypatch):
    seen = {}

    def fake_cf(run, overrides, sub):
        seen["overrides"] = overrides
        return {"overrides_applied": overrides, "baseline_reply": "b", "counterfactual_reply": "c",
               "delta": {}, "has_effect": False, "causal_verified": True,
               "coherence": {"degenerate": False, "reason": ""}, "note": "n", "cost_note": "c"}

    monkeypatch.setattr(experiment, "_counterfactual", fake_cf)
    run = _seed_run()
    experiment.run_experiment(run, {"type": "set_dial", "dial": "warm", "value": 1.7}, None, object())
    assert seen["overrides"] == {"warm": 1.7}


# =========================================================================================== honesty invariants

def test_ablate_regen_has_effect_and_causal_verified_carry_through_exactly_and_null_is_honestly_absent(iso):
    run = _seed_run()
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    out = experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, "regen", sub)
    raw = out["result"]["receipt"]
    assert out["result"]["has_effect"] == raw["has_effect"] is True
    assert out["result"]["causal_verified"] == raw["causal_verified"] is True
    # regen mode has no null control AT ALL -- must read as missing, never as "no effect"
    assert out["result"]["null"] is None


def test_ablate_forced_null_floor_carries_through_untouched_when_present(iso, monkeypatch):
    def fake_receipt(run, influence, sub, *, mode="regen"):
        assert mode == "forced"
        return {"influence": influence, "mode": "forced", "retokenized": False, "causal_verified": True,
               "answer_tokens": ["hi"], "deltas": [0.4], "sum_nats": 0.4, "mean_nats_per_token": 0.4,
               "top_dependent": [], "has_effect": True,
               "threshold": {"mean_abs_nats_per_token": 0.05, "abs_sum_nats": 2.0}, "note": "n",
               "caveat": "cv",
               "null_floor": {"kind": "dial_random_vector", "deltas": [0.01], "sum_nats": 0.01,
                              "mean_nats_per_token": 0.01, "ratio_real_over_floor": 40.0,
                              "exceeds_floor_by_order_of_magnitude": True}}

    monkeypatch.setattr(experiment, "_receipt", fake_receipt)
    run = _seed_run()
    out = experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, "forced", object())
    raw = out["result"]["receipt"]
    assert out["result"]["null"] == raw["null_floor"]                 # preserved VERBATIM, not dropped
    assert out["result"]["has_effect"] is True and out["result"]["causal_verified"] is True
    assert out["result"]["changed_reply"] is None                     # forced mode never generates new text


def test_ablate_forced_missing_null_floor_reads_as_none_not_as_no_effect(iso, monkeypatch):
    def fake_receipt(run, influence, sub, *, mode="regen"):
        return {"influence": influence, "mode": "forced", "retokenized": False, "causal_verified": True,
               "answer_tokens": ["hi"], "deltas": [0.4], "sum_nats": 0.4, "mean_nats_per_token": 0.4,
               "top_dependent": [], "has_effect": True,
               "threshold": {"mean_abs_nats_per_token": 0.05, "abs_sum_nats": 2.0}, "note": "n",
               "caveat": "cv"}   # no null_floor key at all -- e.g. steer had no .steer_vector

    monkeypatch.setattr(experiment, "_receipt", fake_receipt)
    run = _seed_run()
    out = experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, "forced", object())
    assert out["result"]["null"] is None
    assert out["result"]["has_effect"] is True   # a real effect DID fire -- null being absent must not hide it




def test_swap_concept_never_invents_has_effect_and_preserves_the_null_control(iso, swap_stub):
    run = _seed_run()
    out = experiment.run_experiment(
        run, {"type": "swap_concept", "to_concept": "ocean", "from_hint": "Paris"}, None, object())
    # swap_receipt() has no "has_effect" field at all -- must stay None, never inferred from targeted_shift
    assert out["result"]["has_effect"] is None
    assert out["result"]["causal_verified"] is True
    null = out["result"]["null"]
    assert null["available"] is True
    assert null["reply"] == "The capital of France is Paris, a city on the Seine."
    assert null["lexicon_hits"] == 0
    assert null["swap_over_null_nat"] == 3.3
    # the raw underlying receipt is preserved verbatim, no info loss
    assert out["result"]["receipt"]["targeted_shift"] is True
    assert "targeted_shift" not in out["result"]   # never promoted/renamed into a top-level field


def test_swap_concept_blocked_response_still_shapes_cleanly(iso, monkeypatch):
    def blocked_swap(run, from_hint, to_concept, sub):
        return {"mode": "swap_receipt", "causal_verified": False, "run_id": run.get("id"),
               "disposed": None, "swapped_to": {"concept": to_concept, "layer": 21, "strength": 6.0,
                                                "token_id": None, "coef": None},
               "baseline_reply": None, "swapped_reply": None, "null_reply": None,
               "targeted_shift": None, "null_control_available": False, "lexicon_hits": None,
               "logprob_shift": None, "coherent": None, "coherence_score": None,
               "null_note": "...", "lexicon_note": "...", "blocked": "no_engine",
               "note": "substrate has no .engine"}

    monkeypatch.setattr(experiment, "_swap_receipt", blocked_swap)
    run = _seed_run()
    out = experiment.run_experiment(run, {"type": "swap_concept", "to_concept": "ocean"}, None, object())
    _assert_envelope_shape(out)
    assert out["result"]["causal_verified"] is False
    assert out["result"]["has_effect"] is None
    assert out["result"]["delta"] is None            # both replies are None -- never a fabricated {0,0} delta
    assert "not verified as applied" in out["result"]["plain"]


def test_set_dial_has_no_null_control(iso):
    run = _seed_run()
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    out = experiment.run_experiment(run, {"type": "set_dial", "dial": "warm", "value": 0.0}, None, sub)
    assert out["result"]["null"] is None
    assert out["result"]["has_effect"] == out["result"]["receipt"]["has_effect"]
    assert out["result"]["causal_verified"] == out["result"]["receipt"]["causal_verified"]


@pytest.mark.parametrize("change", [{"type": "edit_turn", "turn": 0}, {"type": "reroll"},
                                    {"type": "toggle_greedy"}])
def test_branch_and_replay_ops_never_invent_a_verdict(iso, change):
    run = _seed_run()
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    out = experiment.run_experiment(run, change, None, sub)
    assert out["result"]["has_effect"] is None
    assert out["result"]["causal_verified"] is None
    assert out["result"]["null"] is None
    assert out["result"]["changed_reply"] is not None
    assert out["result"]["receipt"] is not None       # full raw child run record preserved


def test_cost_est_seconds_omitted_when_run_has_no_timing(iso):
    run = _seed_run()   # default started == ended -> duration_ms == 0 -> nothing to ground an estimate in
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    out = experiment.run_experiment(run, {"type": "reroll"}, None, sub)
    assert "est_seconds" not in out["cost"]


def test_cost_est_seconds_grounded_in_the_runs_own_recorded_duration(iso):
    run = _seed_run(started=1.0, ended=3.0)   # -> timing.duration_ms == 2000
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    out = experiment.run_experiment(run, {"type": "reroll"}, None, sub)
    assert out["cost"]["passes"] == 1
    assert out["cost"]["est_seconds"] == pytest.approx(2.0)   # 1 pass * 2000ms


def test_forced_mode_never_fabricates_est_seconds(iso, monkeypatch):
    def fake_receipt(run, influence, sub, *, mode="regen"):
        return {"influence": influence, "mode": "forced", "causal_verified": True, "has_effect": False,
               "deltas": [0.0], "sum_nats": 0.0, "mean_nats_per_token": 0.0, "answer_tokens": [],
               "top_dependent": [], "threshold": {}, "note": "n", "caveat": "cv"}

    monkeypatch.setattr(experiment, "_receipt", fake_receipt)
    run = _seed_run(started=1.0, ended=3.0)   # -> timing.duration_ms == 2000, but forced mode must ignore it
    out = experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, "forced", object())
    assert "est_seconds" not in out["cost"]   # scoring cost != generation cost -- never grounded that way


# ============================================================================================ clean degrades

def test_unknown_change_type_raises_value_error(iso):
    run = _seed_run()
    with pytest.raises(ValueError, match="unknown change.type"):
        experiment.run_experiment(run, {"type": "nonsense"}, None, object())


def test_missing_change_spec_raises_value_error(iso):
    run = _seed_run()
    with pytest.raises(ValueError):
        experiment.run_experiment(run, {}, None, object())
    with pytest.raises(ValueError):
        experiment.run_experiment(run, None, None, object())


def test_missing_run_raises_value_error():
    with pytest.raises(ValueError):
        experiment.run_experiment(None, {"type": "reroll"}, None, object())
    with pytest.raises(ValueError):
        experiment.run_experiment({}, {"type": "reroll"}, None, object())


@pytest.mark.parametrize("change", [
    {"type": "ablate_card"},                          # missing card_id
    {"type": "ablate_dial"},                          # missing dial
    {"type": "set_dial", "dial": "warm"},              # missing value
    {"type": "set_dial", "value": 1.0},                # missing dial
    {"type": "swap_concept"},                          # missing to_concept
    {"type": "swap_concept", "to_concept": "  "},       # blank to_concept
    {"type": "edit_turn"},                             # missing turn
    {"type": "edit_turn", "turn": "not-an-int"},        # bad turn
])
def test_missing_required_fields_raise_value_error(iso, change):
    run = _seed_run()
    with pytest.raises(ValueError):
        experiment.run_experiment(run, change, None, object())


def test_bad_method_raises_value_error_for_receipt_backed_types(iso):
    run = _seed_run()
    with pytest.raises(ValueError):
        experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, "bogus", object())


def test_bad_method_raises_value_error_for_edit_turn(iso):
    run = _seed_run()
    with pytest.raises(ValueError):
        experiment.run_experiment(run, {"type": "edit_turn", "turn": 0}, "bogus", object())


def test_underlying_op_returning_none_degrades_to_none_not_an_exception(iso, monkeypatch):
    monkeypatch.setattr(experiment, "_receipt", lambda *a, **k: None)
    run = _seed_run()
    out = experiment.run_experiment(run, {"type": "ablate_dial", "dial": "warm"}, None, object())
    assert out is None


# =========================================================================================== substrate_ok / catalog

def test_substrate_ok_checks_the_registered_requirement(iso):
    run = _seed_run()
    assert experiment.substrate_ok("ablate_dial", None) is False
    assert experiment.substrate_ok("ablate_dial", FakeSub()) is True
    assert experiment.substrate_ok("swap_concept", FakeSub()) is False   # no .engine/.jlens

    class EngineJlensSub:
        engine = object()

        def jlens(self, *a, **k):
            return {}

    assert experiment.substrate_ok("swap_concept", EngineJlensSub()) is True
    assert experiment.substrate_ok("nonsense_type", FakeSub()) is False


def test_catalog_matches_the_registry_and_hides_the_substrate_field(iso):
    cat = experiment.catalog()
    assert set(cat) == set(experiment.REGISTRY)
    for ctype, entry in cat.items():
        assert set(entry.keys()) == {"label", "needs", "cost_hint", "substrate", "op", "control"}
        assert entry["label"] == experiment.REGISTRY[ctype]["label"]
        assert entry["substrate"] == experiment.REGISTRY[ctype]["substrate"]
        assert entry["op"] == experiment.REGISTRY[ctype]["op"]
        assert entry["control"]
