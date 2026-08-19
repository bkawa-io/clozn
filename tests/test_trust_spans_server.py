"""Trust-spans route: separate proxy, truth-temperature, and opt-in NLI support channels.

All tests are model-free. The NLI matcher is injected with a tiny function; no optional checkpoint, Torch,
or GPU is loaded.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from clozn.eval import store as eval_store
from clozn.receipts import semantic_matcher
from clozn.runs import store as runlog
from clozn.runs.actuary import Calibration, CalibrationBin
from clozn.server.routes import journal


class Handler:
    def _json(self, status, obj):
        self.status, self.obj = status, obj


def _curve():
    bins = [CalibrationBin(i / 10, (i + 1) / 10, 30, .8, i / 10 + .05, -.1) for i in range(10)]
    return Calibration(bins=bins, n_runs=300, n_scored=300, ece_proxy=.1)


@pytest.fixture
def run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "eval_report.json"))
    monkeypatch.setattr(journal, "_report", lambda: (SimpleNamespace(calibration=_curve()), .2))
    eval_store.save({"model": "m1", "set": "arith", "score": "mean", "n": 80,
                     "report": {"temperature_scaling": {
                         "available": True, "temperature": 2.0, "n": 80}}})
    return runlog.record(source="openai_api", model="m1",
                         messages=[{"role": "user", "content": "where?"}],
                         response="Kyoto gardens.",
                         trace={"tokens": ["Kyoto", " gardens", "."],
                                "confidence": [.9, .8, .95]})


def _post(rid, body):
    h = Handler()
    assert journal.try_post(h, f"/runs/{rid}/trust_spans", body) is True
    assert h.status == 200
    return h.obj


def test_default_read_is_model_free_and_carries_proxy_plus_truth(run_id, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("automatic trust fetch must not load NLI")
    monkeypatch.setattr(semantic_matcher, "nli_support_matcher", must_not_run)
    out = _post(run_id, {})
    assert out["available"] is True
    assert out["proxy"]["available"] is True and out["truth"]["available"] is True
    assert out["support"]["requested"] is False
    assert out["spans"][0]["trusted_rate_estimate"] == .8
    assert out["spans"][0]["truth_correctness_estimate"] is not None
    assert "support" not in out["spans"][0]


def test_support_true_runs_injected_nli_and_keeps_it_separate(run_id, monkeypatch):
    calls = []
    def fake_nli(claim, explanation):
        calls.append((claim, explanation))
        return {"supported": True, "score": .91, "threshold": .5, "matched_id": "card1",
                "closest_id": "card1", "contradiction": .01, "method": "nli-deberta-v3"}
    monkeypatch.setattr(semantic_matcher, "nli_support_matcher", fake_nli)
    out = _post(run_id, {"support": True})
    assert calls and out["support"]["requested"] is True and out["support"]["available"] is True
    assert out["support"]["evidence_tier"] == "active_manifest"
    assert "presence, not causal effect" in out["support"]["evidence_note"]
    assert out["spans"][0]["support"]["entailed"] is True
    assert out["spans"][0]["support"]["method"] == "nli-deberta-v3"
    assert out["spans"][0]["truth_correctness_estimate"] is not None


def test_model_mismatch_refuses_truth_mapping_but_keeps_proxy(run_id):
    saved = eval_store.load()
    saved["model"] = "other-model"
    eval_store.save(saved)
    out = _post(run_id, {})
    assert out["truth"]["available"] is False and "does not match" in out["truth"]["reason"]
    assert "truth_correctness_estimate" not in out["spans"][0]
    assert out["proxy"]["available"] is True and out["available"] is True


def test_stored_receipts_switch_support_to_causal_only_premises(run_id, monkeypatch):
    rec = runlog.get_run(run_id)
    rec["receipts"] = {"receipts": [{"influence": {"card_id": "card1"}, "has_effect": True}]}
    assert runlog.replace_run(rec) is True
    seen = []
    def fake_nli(claim, explanation):
        seen.append(explanation)
        return {"supported": False, "score": .1, "method": "nli-deberta-v3"}
    monkeypatch.setattr(semantic_matcher, "nli_support_matcher", fake_nli)
    out = _post(run_id, {"support": True})
    assert seen and out["support"]["evidence_tier"] == "causal_receipts"
    assert "stored leave-one-out receipt" in out["support"]["evidence_note"]
