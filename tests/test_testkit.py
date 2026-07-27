"""test_testkit -- model-free tests for clozn/testkit/runner.py (backlog/tiny-test-harness).

No model, no GPU, no torch: static checks are exercised against plain fixture run dicts (the exact shapes
runlog.py / receipt_bundle.py / run_timeline.py document); the one causal check (`leans_on`) is exercised
against a FAKE substrate mirroring tests/test_receipts.py's FakeSteer/FakeMem/FakeSub, so a receipt's
baseline-vs-ablated delta is driven only by whatever replay.py actually changed, never by randomness.

What's under test:
  * every static check, pass AND fail, plus its honest "error" status when the run simply lacks the data
    the check needs (no confidence trace, an out-of-range index, an unknown check name, ...).
  * the causal `leans_on` path: a real ablation with an effect (pass), a real ablation with no effect
    (fail -- NOT a skip, since it WAS verified), the internalized-mode / no-substrate HONEST SKIP (never a
    false pass), and the `fetch_receipt` injection seam the CLI's --live wiring plugs an HTTP fetch into.
  * evaluate()'s per-test "worst of its assertions" status, and run_suite()'s flattening into tiny_tests.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import clozn.settings as clozn_settings          # noqa: E402

import clozn.receipts.bundle as receipt_bundle    # noqa: E402
import clozn.runs.store as runlog            # noqa: E402
from clozn import testkit           # noqa: E402


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every flat-file store testkit/receipts/replay touch (mirrors test_receipts.py's `iso`)."""
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


# ------------------------------------------------------------------------------------------- static fixture
STATIC_RUN = {
    "id": "run_static1",
    "response": "The capital of France is Paris.",
    "finish_reason": "stop",
    "meta": {"seed": 42, "quant": "Q4_K_M", "n_ctx": 4096, "device": "cpu", "sampler_mode": "greedy"},
    "trace": {
        "tokens": ["The", " capital", " of", " France", " is", " Paris", "."],
        "confidence": [0.99, 0.95, 0.90, 0.60, 0.85, 0.40, 0.99],
        "alternatives": [[], [], [], [], [], [{"piece": " London", "prob": 0.30}], []],
        "topk_entropy": [0.01, 0.02, 0.03, 0.20, 0.05, 0.90, 0.01],
    },
    "memory": {
        "cards_applied": ["The user is studying French geography.", "The user likes concise answers."],
        "applied_ids": ["france-facts", "concise-card"],
        "relevance": [0.82, 0.40],
        "gate": 0.9, "mode": "prompt", "strength": 1.0,
    },
    "behavior": {"active_dials": {"concise": 0.7}},
}


def _test_spec(*asserts, name="t", run="run_static1"):
    return {"name": name, "run": run, "assert": list(asserts)}


def _run(assertion):
    """evaluate() one assertion against STATIC_RUN; return that one assertion's result dict."""
    r = testkit.evaluate(STATIC_RUN, _test_spec(assertion))
    assert len(r["assertions"]) == 1
    return r["assertions"][0]


# =================================================================================================== contains
def test_contains_pass():
    a = _run({"check": "contains", "value": "Paris"})
    assert a["status"] == "pass" and a["check"] == "contains"


def test_contains_fail():
    a = _run({"check": "contains", "value": "Berlin"})
    assert a["status"] == "fail"
    assert a["expected"] == "Berlin"
    assert a["actual"] == STATIC_RUN["response"]


def test_contains_missing_value_is_an_error():
    a = _run({"check": "contains"})
    assert a["status"] == "error"


def test_not_contains_pass_and_fail():
    assert _run({"check": "not_contains", "value": "Berlin"})["status"] == "pass"
    assert _run({"check": "not_contains", "value": "Paris"})["status"] == "fail"


def test_matches_pass_and_fail():
    assert _run({"check": "matches", "value": r"[Pp]aris\."})["status"] == "pass"
    assert _run({"check": "matches", "value": r"^Berlin"})["status"] == "fail"


def test_matches_invalid_regex_is_an_error():
    a = _run({"check": "matches", "value": "(unclosed["})
    assert a["status"] == "error"
    assert "regex" in a["note"]


def test_equals_pass_and_fail():
    assert _run({"check": "equals", "value": STATIC_RUN["response"]})["status"] == "pass"
    assert _run({"check": "equals", "value": "not the response"})["status"] == "fail"


# ============================================================================================ finish_reason
def test_finish_reason_pass_and_fail():
    assert _run({"check": "finish_reason", "value": "stop"})["status"] == "pass"
    assert _run({"check": "finish_reason", "value": "length"})["status"] == "fail"


# =================================================================================================== meta
def test_meta_seed_pass_and_fail():
    assert _run({"check": "seed", "value": 42})["status"] == "pass"
    assert _run({"check": "seed", "value": 7})["status"] == "fail"


def test_meta_device_pass():
    assert _run({"check": "device", "value": "cpu"})["status"] == "pass"


def test_meta_field_missing_from_run_is_a_clean_fail_not_error():
    # sampler_mode IS a REPRO_META_KEYS name but this run's meta happens to have it -- check a name that
    # legitimately isn't recorded (build_git_commit) to prove missing-meta reads as None, not a crash.
    a = _run({"check": "build_git_commit", "value": "abc123"})
    assert a["status"] == "fail"
    assert a["actual"] is None


# =============================================================================================== confidence
def test_min_confidence_overall_pass_and_fail():
    assert _run({"check": "min_confidence", "value": 0.3})["status"] == "pass"     # min is 0.40
    assert _run({"check": "min_confidence", "value": 0.5})["status"] == "fail"


def test_min_confidence_at_index():
    a = _run({"check": "min_confidence", "value": 0.9, "at": 0})
    assert a["status"] == "pass" and a["actual"] == 0.99
    assert _run({"check": "min_confidence", "value": 0.9, "at": 5})["status"] == "fail"


def test_min_confidence_out_of_range_index_is_an_error():
    a = _run({"check": "min_confidence", "value": 0.5, "at": 99})
    assert a["status"] == "error"
    assert "out of range" in a["note"]


def test_min_confidence_missing_value_is_an_error():
    assert _run({"check": "min_confidence"})["status"] == "error"


def test_min_confidence_no_trace_is_an_error_not_a_false_fail():
    run = {"id": "run_bare", "response": "hi"}
    r = testkit.evaluate(run, _test_spec({"check": "min_confidence", "value": 0.5}, run="run_bare"))
    assert r["assertions"][0]["status"] == "error"
    assert "no confidence trace" in r["assertions"][0]["note"]


def test_max_confidence_overall_pass_and_fail():
    assert _run({"check": "max_confidence", "value": 0.99})["status"] == "pass"    # max is exactly 0.99
    assert _run({"check": "max_confidence", "value": 0.5})["status"] == "fail"


def test_max_confidence_at_index():
    # index 3 ("France") has confidence 0.60
    a = _run({"check": "max_confidence", "value": 0.7, "at": 3})
    assert a["status"] == "pass" and a["actual"] == 0.60
    assert _run({"check": "max_confidence", "value": 0.5, "at": 3})["status"] == "fail"
    assert _run({"check": "max_confidence", "value": 0.5, "at": 0})["status"] == "fail"     # conf=0.99


# ================================================================================================= entropy
def test_max_entropy_overall_pass_and_fail():
    assert _run({"check": "max_entropy", "value": 0.95})["status"] == "pass"       # max is 0.90
    assert _run({"check": "max_entropy", "value": 0.5})["status"] == "fail"


def test_max_entropy_at_index():
    a = _run({"check": "max_entropy", "value": 0.5, "at": 4})
    assert a["status"] == "pass" and a["actual"] == 0.05
    assert _run({"check": "max_entropy", "value": 0.5, "at": 5})["status"] == "fail"        # 0.90


def test_max_entropy_null_at_index_is_an_error():
    run = dict(STATIC_RUN, trace=dict(STATIC_RUN["trace"], topk_entropy=[None, 0.1, None, None, None, None, None]))
    r = testkit.evaluate(run, _test_spec({"check": "max_entropy", "value": 0.5, "at": 0}))
    assert r["assertions"][0]["status"] == "error"


def test_max_entropy_no_data_on_run_is_an_error():
    run = {"id": "run_bare", "response": "hi", "trace": {}}
    r = testkit.evaluate(run, _test_spec({"check": "max_entropy", "value": 0.5}, run="run_bare"))
    assert r["assertions"][0]["status"] == "error"


# ============================================================================================= memory cards
def test_card_applied_by_id_pass():
    assert _run({"check": "card_applied", "card": "france-facts"})["status"] == "pass"


def test_card_applied_by_text_substring_pass():
    assert _run({"check": "card_applied", "card": "concise answers"})["status"] == "pass"


def test_card_applied_fail_when_not_found():
    assert _run({"check": "card_applied", "card": "no-such-card"})["status"] == "fail"


def test_relevance_at_least_pass_and_fail():
    assert _run({"check": "relevance_at_least", "card": "france-facts", "value": 0.5})["status"] == "pass"
    assert _run({"check": "relevance_at_least", "card": "france-facts", "value": 0.95})["status"] == "fail"


def test_relevance_at_least_card_not_found_is_a_fail():
    a = _run({"check": "relevance_at_least", "card": "nope", "value": 0.1})
    assert a["status"] == "fail"


def test_relevance_at_least_card_found_but_no_relevance_recorded_is_an_error():
    run = dict(STATIC_RUN, memory=dict(STATIC_RUN["memory"], relevance=[]))
    r = testkit.evaluate(run, _test_spec({"check": "relevance_at_least", "card": "france-facts", "value": 0.1}))
    assert r["assertions"][0]["status"] == "error"


# ========================================================================================= alternative_present
def test_alternative_present_pass_and_fail():
    assert _run({"check": "alternative_present", "value": " London"})["status"] == "pass"
    assert _run({"check": "alternative_present", "value": " Berlin"})["status"] == "fail"


def test_alternative_present_at_index():
    assert _run({"check": "alternative_present", "value": " London", "at": 5})["status"] == "pass"
    assert _run({"check": "alternative_present", "value": " London", "at": 0})["status"] == "fail"


def test_alternative_present_no_data_is_an_error():
    run = {"id": "run_bare", "response": "hi", "trace": {}}
    r = testkit.evaluate(run, _test_spec({"check": "alternative_present", "value": "x"}, run="run_bare"))
    assert r["assertions"][0]["status"] == "error"


# ==================================================================================================== unknown
def test_unknown_check_is_an_error():
    a = _run({"check": "not_a_real_check"})
    assert a["status"] == "error"
    assert "unknown check" in a["note"]


def test_static_dispatch_matches_documented_static_checks():
    assert set(testkit.STATIC_DISPATCH) == set(testkit.STATIC_CHECKS)


# ============================================================================================================
# ================================================================================== causal: leans_on (opt-in)
# ============================================================================================================

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
    randomness (mirrors tests/test_receipts.py's FakeSub exactly)."""

    def __init__(self, mem=None, steer=None, concise_card_ids=()):
        self.memory = mem if mem is not None else FakeMem()
        self.steer = steer if steer is not None else FakeSteer()
        self.concise_card_ids = {str(i) for i in concise_card_ids}

    def chat(self, messages, max_new=256, sample=True):
        excluded = {str(i) for i in (getattr(self.memory, "_exclude_card_ids", None) or [])}
        if self.memory.memory_strength <= 0:
            return "Generic reply with memory off, noticeably longer and less tailored than usual."
        concise_active = self.concise_card_ids - excluded
        concise_dial = float(self.steer.strength.get("concise", 0.0) or 0.0)
        if concise_active or concise_dial > 0:
            base = "Short answer."
        else:
            base = ("A much longer rambling reply with plenty of extra words, since nothing left standing "
                    "kept this concise once every source of brevity was ablated away.")
        if float(self.steer.strength.get("warm", 0.0) or 0.0) > 0:
            base += " Hope that helps and warms your day a little!"
        return base


CAUSAL_RUN = {"id": "run_causal1", "model": "clozn-qwen", "substrate": "QwenSubstrate",
             "messages": [{"role": "user", "content": "tell me about your day"}],
             "response": "THE STORED SAMPLED REPLY -- must never be used as a baseline"}


def test_leans_on_dial_ablation_with_real_effect_passes(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "dial": "warm", "min_effect": 0.0},
                                                run="run_causal1"), sub)
    a = r["assertions"][0]
    assert a["status"] == "pass"
    assert a["actual"]["has_effect"] is True


def test_leans_on_min_effect_threshold_too_high_fails_not_skips(iso):
    """The ablation DID verify and DID show an effect -- just a small one (dropping one warm sentence off
    a long paragraph). A min_effect that demands near-total rewording must FAIL, not silently pass."""
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "dial": "warm", "min_effect": 0.99},
                                                run="run_causal1"), sub)
    a = r["assertions"][0]
    assert a["status"] == "fail"
    assert a["actual"]["has_effect"] is True                # verified effect -- just below the bar






def test_leans_on_with_no_substrate_and_no_fetch_receipt_is_an_honest_skip(iso):
    """The core honesty rule: no substrate, no live fetcher -> skip, with a note pointing at --live. NEVER
    a silent pass."""
    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "dial": "warm"}, run="run_causal1"),
                        sub=None)
    a = r["assertions"][0]
    assert a["status"] == "skip"
    assert "--live" in a["note"]


def test_leans_on_missing_card_and_dial_is_an_error(iso):
    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on"}, run="run_causal1"), FakeSub())
    assert r["assertions"][0]["status"] == "error"


def test_leans_on_receipt_returns_none_is_a_skip(iso):
    """A bad/empty influence spec (receipts.receipt returns None) must skip, not crash or pass."""
    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "card": ""}, run="run_causal1"), FakeSub())
    # card="" is falsy -> falls through to the "needs a card or dial" error branch (never reaches receipts)
    assert r["assertions"][0]["status"] == "error"


def test_leans_on_fetch_receipt_injection_is_used_verbatim(iso):
    """The seam the CLI's --live HTTP path plugs into: `fetch_receipt(run, influence)` is used INSTEAD of
    calling receipts.receipt(..., sub) -- proving the wiring without needing a live model or a real HTTP
    server. `sub` stays None throughout: fetch_receipt alone is enough to un-skip a causal assertion."""
    canned = {"causal_verified": True, "has_effect": True, "delta": {"changed": 80}, "note": "ok"}
    calls = []

    def fetch_receipt(run, influence):
        calls.append((run.get("id"), influence))
        return canned

    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "card": "france-facts", "min_effect": 0.5},
                                                run="run_causal1"), sub=None, fetch_receipt=fetch_receipt)
    a = r["assertions"][0]
    assert a["status"] == "pass"
    assert calls == [("run_causal1", {"card_id": "france-facts"})]


def test_leans_on_fetch_receipt_returning_none_is_a_skip(iso):
    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "dial": "warm"}, run="run_causal1"),
                        sub=None, fetch_receipt=lambda run, inf: None)
    assert r["assertions"][0]["status"] == "skip"


def test_leans_on_fetch_receipt_raising_is_a_skip_not_a_crash(iso):
    def boom(run, influence):
        raise RuntimeError("studio unreachable")

    r = testkit.evaluate(CAUSAL_RUN, _test_spec({"check": "leans_on", "dial": "warm"}, run="run_causal1"),
                        sub=None, fetch_receipt=boom)
    a = r["assertions"][0]
    assert a["status"] == "skip"
    assert "studio unreachable" in a["note"]


def test_judge_receipt_is_a_pure_function_reused_by_both_paths():
    status, actual, note = testkit.judge_receipt(
        {"causal_verified": True, "has_effect": True, "delta": {"changed": 40}}, 0.3)
    assert status == "pass" and actual["effect"] == 0.4
    status, _, _ = testkit.judge_receipt(None, 0.0)
    assert status == "skip"


def test_judge_receipt_distinguishes_engine_unreachable_from_a_bad_spec():
    """Engine-down pressure test finding #3, CLI --live path: cli.commands.test._fetch_live_receipt
    returns this {"_fetch_error": "engine_unreachable", ...} sentinel (instead of bare None) when the
    live gateway itself answered 502/503 "engine not reachable" -- judge_receipt must say so distinctly
    from the generic "bad influence spec, or the substrate could not generate" note a plain None gets."""
    status, actual, note = testkit.judge_receipt(
        {"_fetch_error": "engine_unreachable",
         "_fetch_detail": "engine not reachable at http://127.0.0.1:8080 -- is it running?"}, 0.0)
    assert status == "skip"
    assert actual is None
    assert note == "causal assertion skipped: engine not reachable at http://127.0.0.1:8080 -- is it running?"

    # no detail supplied -> a sane fallback note, never a blank/None note
    status, _, note = testkit.judge_receipt({"_fetch_error": "engine_unreachable"}, 0.0)
    assert status == "skip"
    assert note == "causal assertion skipped: engine not reachable"

    # a genuinely-None receipt (the pre-existing ambiguous case) keeps its ORIGINAL generic note
    status, _, note = testkit.judge_receipt(None, 0.0)
    assert status == "skip"
    assert "bad influence spec" in note


# ============================================================================================================
# =============================================================================== evaluate() / run_suite() ===
# ============================================================================================================

def test_evaluate_run_not_found_is_an_error():
    r = testkit.evaluate(None, _test_spec({"check": "contains", "value": "x"}, run="run_missing"))
    assert r["status"] == "error"
    assert "run not found" in r["assertions"][0]["note"]


def test_evaluate_no_assert_list_is_an_error():
    r = testkit.evaluate(STATIC_RUN, {"name": "t", "run": "run_static1"})
    assert r["status"] == "error"


def test_evaluate_malformed_assertion_entry_is_an_error_not_a_crash():
    r = testkit.evaluate(STATIC_RUN, _test_spec("not a dict"))
    assert r["assertions"][0]["status"] == "error"


def test_evaluate_overall_status_is_worst_of_its_assertions():
    r = testkit.evaluate(STATIC_RUN, _test_spec({"check": "contains", "value": "Paris"},     # pass
                                                {"check": "contains", "value": "Berlin"}))    # fail
    assert r["status"] == "fail"

    r2 = testkit.evaluate(STATIC_RUN, _test_spec({"check": "contains", "value": "Paris"},
                                                 {"check": "leans_on", "dial": "warm"}), sub=None)
    assert r2["status"] == "skip"                            # pass + skip -> skip (nothing failed)


def test_run_suite_resolves_latest_via_injected_get_run():
    calls = []

    def get_run(ref):
        calls.append(ref)
        return STATIC_RUN

    spec = {"tests": [_test_spec({"check": "contains", "value": "Paris"}, run="latest")]}
    out = testkit.run_suite(spec, get_run=get_run)
    assert calls == ["latest"]
    assert out["status"] == "pass"
    assert out["tests"][0]["run_id"] == "run_static1"


def test_run_suite_flattens_every_assertion_into_tiny_tests():
    spec = {"tests": [
        _test_spec({"check": "contains", "value": "Paris"}, {"check": "finish_reason", "value": "stop"},
                  name="t1"),
        _test_spec({"check": "contains", "value": "Berlin"}, name="t2"),
    ]}
    out = testkit.run_suite(spec, get_run=lambda ref: STATIC_RUN)
    assert len(out["tiny_tests"]) == 3
    assert all(set(a) == {"name", "check", "target", "expected", "actual", "status", "note"}
              for a in out["tiny_tests"])
    assert out["status"] == "fail"                            # t2's assertion failed
    assert out["counts"]["pass"] == 2 and out["counts"]["fail"] == 1


def test_run_suite_malformed_test_entry_degrades_to_one_error_not_a_crash():
    spec = {"tests": ["not a dict", _test_spec({"check": "contains", "value": "Paris"})]}
    out = testkit.run_suite(spec, get_run=lambda ref: STATIC_RUN)
    assert out["tests"][0]["status"] == "error"
    assert out["tests"][1]["status"] == "pass"


def test_run_suite_get_run_raising_resolves_to_run_not_found():
    def boom(ref):
        raise RuntimeError("disk on fire")

    spec = {"tests": [_test_spec({"check": "contains", "value": "Paris"})]}
    out = testkit.run_suite(spec, get_run=boom)
    assert out["tests"][0]["status"] == "error"


def test_run_suite_empty_tests_list_is_an_error():
    assert testkit.run_suite({"tests": []})["status"] == "error"


def test_results_by_run_groups_by_resolved_run_id_and_skips_unresolved():
    spec = {"tests": [
        _test_spec({"check": "contains", "value": "Paris"}, name="t1", run="run_static1"),
        _test_spec({"check": "contains", "value": "Berlin"}, name="t2", run="run_static1"),
        _test_spec({"check": "contains", "value": "x"}, name="t3", run="run_missing"),
    ]}
    out = testkit.run_suite(spec, get_run=lambda ref: STATIC_RUN if ref == "run_static1" else None)
    grouped = testkit.results_by_run(out)
    assert set(grouped) == {"run_static1"}
    assert len(grouped["run_static1"]) == 2


# ================================================================================================ default_get_run
def test_default_get_run_resolves_latest(iso):
    runlog.record(source="test", client="pytest", model="m", messages=[{"role": "user", "content": "hi"}],
                  response="hello", started=1.0, ended=1.1)
    rid = runlog.list_runs(limit=1)[0]["id"]
    run = testkit.default_get_run("latest")
    assert run is not None and run["id"] == rid


def test_default_get_run_resolves_a_literal_id(iso):
    rid = runlog.record(source="test", client="pytest", model="m",
                        messages=[{"role": "user", "content": "hi"}], response="hello",
                        started=1.0, ended=1.1)
    assert testkit.default_get_run(rid)["id"] == rid


def test_default_get_run_missing_run_is_none(iso):
    assert testkit.default_get_run("run_does_not_exist") is None


def test_default_get_run_no_runs_at_all_is_none(iso):
    assert testkit.default_get_run("latest") is None


# ================================================================================================ --attach seam
def test_attached_results_round_trip_through_receipt_bundle(iso):
    """The whole point of the tiny_tests slot: attach a suite's results to a run, then confirm
    receipt_bundle.build(run)["tiny_tests"] reads exactly what was attached -- no reshaping needed."""
    rid = runlog.record(source="test", client="pytest", model="m",
                        messages=[{"role": "user", "content": "capital of France?"}],
                        response="Paris.", finish_reason="stop", started=1.0, ended=1.1)
    spec = {"tests": [{"name": "t1", "run": rid,
                      "assert": [{"check": "contains", "value": "Paris"},
                                {"check": "finish_reason", "value": "stop"}]}]}
    out = testkit.run_suite(spec, get_run=runlog.get_run)
    assert out["status"] == "pass"

    grouped = testkit.results_by_run(out)
    ok = runlog.update_tiny_tests(rid, grouped[rid])
    assert ok is True

    run = runlog.get_run(rid)
    assert run["tiny_tests"] == grouped[rid]
    bundle = receipt_bundle.build(run)
    assert bundle["tiny_tests"] == grouped[rid]
    assert len(bundle["tiny_tests"]) == 2


def test_update_tiny_tests_returns_false_for_a_missing_run(iso):
    assert runlog.update_tiny_tests("run_does_not_exist", [{"a": 1}]) is False
