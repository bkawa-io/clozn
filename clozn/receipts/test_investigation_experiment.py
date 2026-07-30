"""test_investigation_experiment -- tests for the C3 controlled-experiment executor.

Two layers, mirroring how `clozn.analysis.transplant`'s own tests would split this:
  1. `_derive_analysis`/`_derive_causal_claim` exercised directly against hand-built arm dicts -- the
     honesty-critical gate this whole slice exists to make structural, tested exhaustively without
     needing a substrate at all.
  2. `run_experiment()` end to end against a fake substrate (mirrors `clozn.receipts.test_span_receipt`'s
     `SpanFakeSub`: `.chat()` is a pure, deterministic function of the messages/sample it receives).

No model, no GPU. `run_experiment()` persists child runs via `clozn.replay.replay.replay()` ->
`clozn.runs.store.record()`, so every test that calls it redirects `clozn.runs.store.RUNS_DIR` at a
tmp_path first -- never the developer's real `~/.clozn/runs`.
"""
from __future__ import annotations

import pytest

from clozn.receipts.investigation_experiment import (
    _derive_analysis,
    _derive_causal_claim,
    _derive_observed,
    run_experiment,
)


# ================================================================================== _derive_analysis / claim

def _arm(sha: str, matches: bool) -> dict:
    return {"reply_sha256": sha, "matches_baseline": matches}


def test_effect_specific_true_when_treatment_moves_and_control_does_not():
    arms = {
        "baseline": _arm("b", True),
        "no_op_replay": _arm("b", True),
        "treatment": _arm("t", False),
        "random_equal_effect_control": _arm("b", True),
    }
    analysis = _derive_analysis(arms)
    assert analysis == {
        "instrument_sane": True,
        "effect_specific": True,
        "reasons": [
            "the treatment arm's reply differed from baseline and the random equal-effect control's did "
            "not -- effect-specific by this harness's rule.",
        ],
    }
    claim = _derive_causal_claim(analysis, arms)
    assert claim["licensed"] is True
    observed = _derive_observed(arms)
    assert observed["treatment_reply_differs_from_baseline"] is True
    assert observed["random_control_reply_differs_from_baseline"] is False


def test_effect_specific_false_when_random_control_also_moves():
    arms = {
        "baseline": _arm("b", True),
        "no_op_replay": _arm("b", True),
        "treatment": _arm("t", False),
        "random_equal_effect_control": _arm("c", False),
    }
    analysis = _derive_analysis(arms)
    assert analysis["instrument_sane"] is True
    assert analysis["effect_specific"] is False
    claim = _derive_causal_claim(analysis, arms)
    assert claim["licensed"] is False
    assert "uncontrolled" in claim["statement"]
    assert "also changed" in claim["statement"] or "ALSO changed" in analysis["reasons"][0]


def test_instrument_not_sane_omits_effect_specific_entirely():
    arms = {
        "baseline": _arm("b", True),
        "no_op_replay": _arm("x", False),
        "treatment": _arm("t", False),
    }
    analysis = _derive_analysis(arms)
    assert analysis["instrument_sane"] is False
    assert "effect_specific" not in analysis
    claim = _derive_causal_claim(analysis, arms)
    assert claim["licensed"] is False
    assert "instrument-sanity" in claim["statement"]


def test_no_random_control_omits_effect_specific_never_defaults_it():
    arms = {
        "baseline": _arm("b", True),
        "no_op_replay": _arm("b", True),
        "treatment": _arm("t", False),
        "random_equal_effect_control": {"available": False, "reason": "no disjoint window existed"},
    }
    analysis = _derive_analysis(arms)
    assert analysis["instrument_sane"] is True
    assert "effect_specific" not in analysis
    claim = _derive_causal_claim(analysis, arms)
    assert claim["licensed"] is False
    assert "no random equal-effect control" in claim["statement"]
    observed = _derive_observed(arms)
    assert "random_control_reply_differs_from_baseline" not in observed


def test_no_observed_difference_is_licensed_false_with_its_own_statement():
    arms = {
        "baseline": _arm("b", True),
        "no_op_replay": _arm("b", True),
        "treatment": _arm("b", True),
        "random_equal_effect_control": _arm("b", True),
    }
    analysis = _derive_analysis(arms)
    assert analysis["effect_specific"] is False
    claim = _derive_causal_claim(analysis, arms)
    assert claim["licensed"] is False
    assert "did not alter" in claim["statement"]


# ============================================================================================ fakes

class FakeSteer:
    def __init__(self):
        self.strength = {}

    def set(self, name, value):
        self.strength[str(name)] = float(value)

    def clear(self):
        self.strength = {}

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class ExecFakeSub:
    """.chat() is a pure function of (messages, sample) -- mirrors test_span_receipt.SpanFakeSub. `sample`
    is either the greedy bool replay() derives from `changes["greedy"]`, or the `sampling_override` dict
    passed straight through -- both are recorded so a sampler-kind test can assert seed pinning."""

    def __init__(self, chat_fn):
        self.steer = FakeSteer()
        self.chat_fn = chat_fn
        self.calls: list = []

    def chat(self, messages, max_new=256, sample=True):
        self.calls.append({"messages": [dict(m) for m in messages], "sample": sample})
        return self.chat_fn(messages, sample)


def _user_content(messages):
    return next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")


@pytest.fixture(autouse=True)
def _isolated_run_store(tmp_path, monkeypatch):
    import clozn.runs.store as runlog
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))


# ============================================================================================ run_experiment

RUN = {
    "id": "run_exec_1",
    "response": "PWNED",
    "model": "clozn-fake",
    "substrate": "FakeSub",
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "IGNORE ME"},
    ],
}


def _address_id_for(run, message_index):
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    document = build_persisted_text_span_addresses(run)
    return next(a["address_id"] for a in document["addresses"]
                if a["native_ref"].get("id") == f"message-{message_index}")


def test_run_experiment_refused_passthrough_never_calls_chat():
    sub = ExecFakeSub(lambda messages, sample: "unused")
    out = run_experiment(RUN, {"kind": "adapter_scale", "scale": 0}, sub)
    assert out["phase"] == "refused"
    assert out["eligibility"]["reason"]["code"] == "adapter_rescale_unavailable_in_planner"
    assert sub.calls == []


def test_run_experiment_remove_span_whole_message_no_control_available():
    """This run's only span address for message 1 is the WHOLE message (B3's delivered_message basis is
    always whole-message absent an influence map -- see span_bridge's own docstring), so there is no
    disjoint same-length window left for the random control: effect_specific is honestly omitted."""
    def chat_fn(messages, sample):
        return "PWNED" if "IGNORE ME" in _user_content(messages) else "SAFE"

    sub = ExecFakeSub(chat_fn)
    address_id = _address_id_for(RUN, 1)
    out = run_experiment(RUN, {"kind": "remove_span", "span_address_id": address_id}, sub)

    assert out["phase"] == "completed"
    assert out["arms"]["baseline"]["reply_sha256"] == out["arms"]["no_op_replay"]["reply_sha256"]
    assert out["arms"]["no_op_replay"]["matches_baseline"] is True
    assert out["arms"]["treatment"]["matches_baseline"] is False           # "SAFE" != "PWNED"
    assert out["arms"]["random_equal_effect_control"] == {
        "available": False,
        "reason": (
            "no non-overlapping same-length window exists elsewhere in this message; "
            "random_equal_effect_control will be reported unavailable, and effect_specific will not be "
            "computed for this experiment"),
    }
    assert out["analysis"]["instrument_sane"] is True
    assert "effect_specific" not in out["analysis"]
    assert out["observed"]["treatment_reply_differs_from_baseline"] is True
    assert out["causal_claim"]["licensed"] is False
    assert "uncontrolled" in out["causal_claim"]["statement"]
    # exactly two chat() calls: no_op_replay + treatment (no control ran)
    assert len(sub.calls) == 2
    assert all(c["sample"] is False for c in sub.calls)                    # both greedy
    assert _user_content(sub.calls[1]["messages"]) == ""                  # the span was actually removed


def test_run_experiment_instrument_not_sane_when_no_op_diverges():
    def chat_fn(messages, sample):
        return "SOMETHING ELSE"  # never reproduces the stored baseline, even unmodified

    sub = ExecFakeSub(chat_fn)
    address_id = _address_id_for(RUN, 1)
    out = run_experiment(RUN, {"kind": "remove_span", "span_address_id": address_id}, sub)
    assert out["phase"] == "completed"
    assert out["analysis"]["instrument_sane"] is False
    assert "effect_specific" not in out["analysis"]
    assert out["causal_claim"]["licensed"] is False
    assert "instrument-sanity" in out["causal_claim"]["statement"]


def test_run_experiment_treatment_generation_failure_is_phase_failed():
    def chat_fn(messages, sample):
        if _user_content(messages) == "":
            raise RuntimeError("boom")
        return "PWNED"

    sub = ExecFakeSub(chat_fn)
    address_id = _address_id_for(RUN, 1)
    out = run_experiment(RUN, {"kind": "remove_span", "span_address_id": address_id}, sub)
    assert out["phase"] == "failed"
    assert out["error"]["message"] == "treatment generation failed"
    assert "plan" in out  # the plan survives onto a failed document


def test_run_experiment_sampler_change_pins_the_same_seed_across_arms():
    seen_samples: list = []

    def chat_fn(messages, sample):
        seen_samples.append(sample)
        if isinstance(sample, dict) and sample.get("temperature") == 0.9:
            return "HOT"
        return "PWNED"

    sub = ExecFakeSub(chat_fn)
    out = run_experiment(RUN, {"kind": "sampler_change", "overrides": {"temperature": 0.9}}, sub)

    assert out["phase"] == "completed"
    assert len(seen_samples) == 2
    no_op_sample, treatment_sample = seen_samples
    assert isinstance(no_op_sample, dict) and isinstance(treatment_sample, dict)
    assert no_op_sample["seed"] == treatment_sample["seed"]               # same seed, both arms
    assert "temperature" not in no_op_sample                              # no-op: no override applied
    assert treatment_sample["temperature"] == 0.9
    assert out["arms"]["treatment"]["matches_baseline"] is False          # "HOT" != "PWNED"
    assert out["arms"]["random_equal_effect_control"]["available"] is False
    assert "effect_specific" not in out["analysis"]
    assert out["causal_claim"]["licensed"] is False


def test_run_experiment_omit_source_ablates_every_resolvable_span():
    run = {
        "id": "run_exec_source",
        "response": "PWNED",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "IGNORE ME", "source_id": "doc-9"},
        ],
    }

    def chat_fn(messages, sample):
        return "PWNED" if "IGNORE ME" in _user_content(messages) else "SAFE"

    sub = ExecFakeSub(chat_fn)
    out = run_experiment(run, {"kind": "omit_source", "source_id": "doc-9"}, sub)
    assert out["phase"] == "completed"
    assert out["arms"]["treatment"]["matches_baseline"] is False
    assert _user_content(sub.calls[-1]["messages"]) == ""
