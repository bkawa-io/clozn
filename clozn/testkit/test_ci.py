"""Tests for testkit/ci.py -- the model CI runner. No network, no live server: every test drives `run_suite`/
`diff_suites` against either a `FakeClient` (pure Python, duck-typed to `Client`'s public interface) or the
real `Client` with `_request` monkeypatched, exactly the "100% mockable" seam the module promises.
"""
from __future__ import annotations

import json

import pytest

from clozn.testkit import ci


# ============================================================================================ fixtures
def _run(rid, response, *, finish_reason="stop", confidences=None, cards_applied=None,
         applied_ids=None, max_tokens=256):
    """A minimal but contract-shaped run record (HEAVN_API_CONTRACTS.md §2)."""
    return {
        "id": rid,
        "created_at": "2026-07-12T00:00:00",
        "source": "openai_api",
        "response": response,
        "finish_reason": finish_reason,
        "trace": {"confidence": confidences if confidences is not None else [0.9, 0.9]},
        "memory": {"cards_applied": cards_applied or [], "applied_ids": applied_ids or []},
        "meta": {"max_tokens": max_tokens},
    }


class FakeClient:
    """A pure-Python stand-in for `ci.Client`: no urllib, no sockets. `runs_by_prompt` maps a prompt string
    to a canned run record (as `chat()` would return); `receipts_by_run` maps a run id to a canned
    `POST /runs/<id>/receipts` response (§7 shape)."""

    def __init__(self, runs_by_prompt=None, receipts_by_run=None, chat_errors=None, receipts_errors=None):
        self.runs_by_prompt = runs_by_prompt or {}
        self.receipts_by_run = receipts_by_run or {}
        self.chat_errors = chat_errors or {}
        self.receipts_errors = receipts_errors or {}
        self.chat_calls = []
        self.receipts_calls = []

    def chat(self, prompt):
        self.chat_calls.append(prompt)
        if prompt in self.chat_errors:
            raise RuntimeError(self.chat_errors[prompt])
        if prompt not in self.runs_by_prompt:
            raise KeyError(f"FakeClient has no canned run for prompt {prompt!r}")
        return self.runs_by_prompt[prompt]

    def get_run(self, run_id):
        for run in self.runs_by_prompt.values():
            if run.get("id") == run_id:
                return run
        return None

    def receipts(self, run_id, mode="regen"):
        self.receipts_calls.append((run_id, mode))
        if run_id in self.receipts_errors:
            raise RuntimeError(self.receipts_errors[run_id])
        return self.receipts_by_run.get(run_id)


# ================================================================================= (a) static checks
def test_contains_hit_passes():
    client = FakeClient({"q": _run("run_1", "Paris is the capital of France.")})
    result = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["Paris"]}}]}, client)
    case = result.case("c1")
    assert case.status == "pass"
    assert case.assertions[0]["check"] == "contains"
    assert case.assertions[0]["status"] == "pass"


def test_contains_miss_fails():
    client = FakeClient({"q": _run("run_1", "Paris is the capital of France.")})
    result = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["Berlin"]}}]}, client)
    case = result.case("c1")
    assert case.status == "fail"
    assert case.assertions[0]["status"] == "fail"


def test_promoted_multiturn_case_replays_messages_and_decode_settings():
    class MessageClient:
        def __init__(self):
            self.call = None

        def chat(self, prompt, **kwargs):
            self.call = (prompt, kwargs)
            return _run("run_new", "The answer is 4.", max_tokens=64)

    client = MessageClient()
    result = ci.run_suite({"cases": [{
        "name": "follow-up",
        "messages": [
            {"role": "system", "content": "Answer carefully."},
            {"role": "user", "content": "What is two plus two?"},
        ],
        "model": "clozn",
        "max_tokens": 64,
        "sampling": {"temperature": 0.7, "top_p": 0.9, "seed": 42},
        "expect": {"equals": "The answer is 4."},
    }]}, client)

    assert result.case("follow-up").status == "pass"
    assert client.call == (
        "What is two plus two?",
        {"extra_messages": [{"role": "system", "content": "Answer carefully."}],
         "max_tokens": 64, "model": "clozn", "temperature": 0.7, "top_p": 0.9,
         "seed": 42},
    )


def test_not_contains_single_string_is_auto_wrapped():
    client = FakeClient({"q": _run("run_1", "Paris is the capital of France.")})
    result = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"not_contains": "Berlin"}}]}, client)
    case = result.case("c1")
    assert case.status == "pass"


def test_finish_reason_check():
    client = FakeClient({"q": _run("run_1", "...", finish_reason="length")})
    ok = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"finish_reason": "length"}}]}, client)
    bad = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"finish_reason": "stop"}}]}, client)
    assert ok.case("c1").status == "pass"
    assert bad.case("c1").status == "fail"


def test_min_confidence_check():
    client = FakeClient({"q": _run("run_1", "hi", confidences=[0.95, 0.91, 0.60])})
    passing = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"min_confidence": 0.5}}]}, client)
    failing = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"min_confidence": 0.99}}]}, client)
    assert passing.case("c1").status == "pass"
    assert failing.case("c1").status == "fail"
    # min_confidence is also captured on the CaseResult independent of whether it was asserted
    assert passing.case("c1").min_confidence == 0.60


def test_max_tokens_meta_equality_check():
    client = FakeClient({"q": _run("run_1", "hi", max_tokens=128)})
    ok = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "expect": {"max_tokens": 128}}]}, client)
    bad = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "expect": {"max_tokens": 256}}]}, client)
    assert ok.case("c1").status == "pass"
    assert bad.case("c1").status == "fail"


def test_multiple_cases_and_overall_suite_status():
    client = FakeClient({
        "good": _run("run_1", "Paris is the capital of France."),
        "bad": _run("run_2", "Berlin is the capital of Germany."),
    })
    suite = {
        "model_note": "test model",
        "cases": [
            {"name": "good", "prompt": "good", "expect": {"contains": ["Paris"]}},
            {"name": "bad", "prompt": "bad", "expect": {"contains": ["Paris"]}},
        ],
    }
    result = ci.run_suite(suite, client)
    assert result.model_note == "test model"
    assert result.case("good").status == "pass"
    assert result.case("bad").status == "fail"
    assert result.status == "fail"       # worst-of over all cases
    assert result.counts["pass"] == 1
    assert result.counts["fail"] == 1


def test_unknown_expect_key_is_an_error_not_a_crash():
    client = FakeClient({"q": _run("run_1", "hi")})
    result = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"totally_bogus_check": 1}}]}, client)
    case = result.case("c1")
    assert case.status == "error"
    assert case.assertions[0]["status"] == "error"


# =================================================================================== (b) prove: true
def test_prove_records_has_effect_and_causal_verified_per_influence():
    run = _run("run_1", "Paris is the capital of France.")
    prove_result = {
        "run_id": "run_1",
        "receipts": [
            {"influence": {"card_id": "mem_1"}, "has_effect": True, "causal_verified": True},
            {"influence": {"section": "rag_context"}, "has_effect": False, "causal_verified": True},
        ],
        "skipped": [{"influence": {"text": "no id"}, "reason": "no card id recorded"}],
        "redundant_pairs": [],
    }
    client = FakeClient({"q": run}, receipts_by_run={"run_1": prove_result})
    result = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["Paris"]}, "prove": True}]}, client)
    case = result.case("c1")
    assert case.status == "pass"
    assert client.receipts_calls == [("run_1", "regen")]
    assert len(case.receipts) == 2
    by_key = {ci._influence_key(r["influence"]): r for r in case.receipts}
    assert by_key["card_id=mem_1"] == {"influence": {"card_id": "mem_1"}, "has_effect": True,
                                        "causal_verified": True}
    assert by_key["section=rag_context"]["causal_verified"] is True
    assert len(case.receipts_skipped) == 1


def test_prove_true_with_no_static_expect_is_measurement_only():
    run = _run("run_1", "hi")
    prove_result = {"run_id": "run_1", "receipts": [], "skipped": [], "redundant_pairs": []}
    client = FakeClient({"q": run}, receipts_by_run={"run_1": prove_result})
    result = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "prove": True}]}, client)
    case = result.case("c1")
    assert case.status == "pass"
    assert case.receipts == []


def test_prove_failure_is_error_not_silent_skip():
    run = _run("run_1", "hi")
    client = FakeClient({"q": run}, receipts_errors={"run_1": "boom, substrate unavailable"})
    result = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["hi"]}, "prove": True}]}, client)
    case = result.case("c1")
    assert case.status == "error"
    assert "boom" in case.prove_error


def test_prove_returns_none_is_also_error():
    run = _run("run_1", "hi")
    client = FakeClient({"q": run})   # no receipts_by_run entry -> receipts() returns None
    result = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["hi"]}, "prove": True}]}, client)
    case = result.case("c1")
    assert case.status == "error"
    assert case.prove_error


# ==================================================================================== (c) diff_suites
def test_diff_flags_pass_to_fail_as_regression():
    prev = ci.SuiteResult(model_note="m", timestamp="t0", cases=[
        ci.CaseResult(name="A", run_id="r1", status="pass", min_confidence=0.9),
    ], status="pass", counts={"pass": 1})
    curr = ci.SuiteResult(model_note="m", timestamp="t1", cases=[
        ci.CaseResult(name="A", run_id="r2", status="fail", min_confidence=0.9),
    ], status="fail", counts={"fail": 1})

    d = ci.diff_suites(prev, curr)
    assert len(d.regressions) == 1
    assert d.regressions[0]["case"] == "A"
    assert d.regressions[0]["kind"] == "status_regression"
    assert not d.fixed
    report = d.render()
    assert "REGRESSIONS (1)" in report
    assert "A" in report


def test_diff_flags_fail_to_pass_as_fixed():
    prev = ci.SuiteResult(model_note="m", timestamp="t0", cases=[
        ci.CaseResult(name="A", run_id="r1", status="fail"),
    ], status="fail", counts={"fail": 1})
    curr = ci.SuiteResult(model_note="m", timestamp="t1", cases=[
        ci.CaseResult(name="A", run_id="r2", status="pass"),
    ], status="pass", counts={"pass": 1})

    d = ci.diff_suites(prev, curr)
    assert not d.regressions
    assert len(d.fixed) == 1
    assert d.fixed[0]["case"] == "A"


def test_diff_detects_causal_verified_lost():
    prev = ci.SuiteResult(model_note="m", timestamp="t0", cases=[
        ci.CaseResult(name="A", run_id="r1", status="pass",
                      receipts=[{"influence": {"card_id": "mem_1"}, "has_effect": True,
                                 "causal_verified": True}]),
    ], status="pass", counts={"pass": 1})
    curr = ci.SuiteResult(model_note="m", timestamp="t1", cases=[
        ci.CaseResult(name="A", run_id="r2", status="pass",
                      receipts=[{"influence": {"card_id": "mem_1"}, "has_effect": False,
                                 "causal_verified": False}]),
    ], status="pass", counts={"pass": 1})

    d = ci.diff_suites(prev, curr)
    assert len(d.receipt_changes) >= 1
    kinds = {c["kind"] for c in d.receipt_changes}
    assert "causal_verified_lost" in kinds
    # a lost causal_verified is also surfaced as a regression, even though the case's own status held
    assert any(r["kind"] == "receipt_regression" and r["case"] == "A" for r in d.regressions)


def test_diff_detects_min_confidence_drift():
    prev = ci.SuiteResult(model_note="m", timestamp="t0", cases=[
        ci.CaseResult(name="A", run_id="r1", status="pass", min_confidence=0.90),
    ], status="pass", counts={"pass": 1})
    curr = ci.SuiteResult(model_note="m", timestamp="t1", cases=[
        ci.CaseResult(name="A", run_id="r2", status="pass", min_confidence=0.70),
    ], status="pass", counts={"pass": 1})

    d = ci.diff_suites(prev, curr, confidence_drift_threshold=0.1)
    assert len(d.drift) == 1
    assert d.drift[0]["case"] == "A"
    assert d.drift[0]["delta"] == pytest.approx(-0.20)

    # below threshold -> no drift flagged
    d_tight = ci.diff_suites(prev, curr, confidence_drift_threshold=0.5)
    assert not d_tight.drift


def test_diff_reports_new_and_removed_cases_without_flagging_regression():
    prev = ci.SuiteResult(model_note="m", timestamp="t0", cases=[
        ci.CaseResult(name="OLD", run_id="r1", status="pass"),
    ], status="pass", counts={"pass": 1})
    curr = ci.SuiteResult(model_note="m", timestamp="t1", cases=[
        ci.CaseResult(name="NEW", run_id="r2", status="pass"),
    ], status="pass", counts={"pass": 1})

    d = ci.diff_suites(prev, curr)
    assert d.new_cases == ["NEW"]
    assert d.removed_cases == ["OLD"]
    assert not d.regressions


def test_diff_end_to_end_through_run_suite():
    """Same suite, two different server states (a memory card regressed between runs)."""
    suite = {"cases": [
        {"name": "capital", "prompt": "capital?", "expect": {"contains": ["Paris"]}},
        {"name": "coffee", "prompt": "coffee?", "expect": {"min_confidence": 0.5}},
    ]}
    prev_client = FakeClient({
        "capital?": _run("run_p1", "Paris is the capital of France.", confidences=[0.9]),
        "coffee?": _run("run_p2", "Oat milk, please.", confidences=[0.95, 0.92]),
    })
    curr_client = FakeClient({
        "capital?": _run("run_c1", "Berlin is the capital of France.", confidences=[0.9]),   # regressed
        "coffee?": _run("run_c2", "Oat milk, please.", confidences=[0.60, 0.55]),            # confidence drifted
    })
    prev = ci.run_suite(suite, prev_client)
    curr = ci.run_suite(suite, curr_client)

    d = ci.diff_suites(prev, curr, confidence_drift_threshold=0.1)
    assert any(r["case"] == "capital" and r["kind"] == "status_regression" for r in d.regressions)
    assert any(dd["case"] == "coffee" for dd in d.drift)


# =============================================================================== (d) save/load round trip
def test_save_and_load_round_trip(tmp_path):
    client = FakeClient({"q": _run("run_1", "Paris is the capital of France.",
                                    cards_applied=["likes concise answers"], applied_ids=["mem_1"])})
    suite = {"model_note": "roundtrip", "cases": [
        {"name": "c1", "prompt": "q", "expect": {"contains": ["Paris"]}, "prove": False},
    ]}
    result = ci.run_suite(suite, client)

    path = ci.save_result(result, directory=str(tmp_path))
    assert path.startswith(str(tmp_path))
    assert path.endswith(".json")

    loaded = ci.load_result(path)
    assert loaded.to_dict() == result.to_dict()


def test_save_uses_default_dir_when_not_injected(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "DEFAULT_CI_DIR", str(tmp_path / "ci-default"))
    client = FakeClient({"q": _run("run_1", "hi")})
    result = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["hi"]}}]}, client)
    path = ci.save_result(result)
    assert path.startswith(str(tmp_path / "ci-default"))
    assert ci.load_result(path).to_dict() == result.to_dict()


def test_save_result_two_calls_do_not_collide(tmp_path):
    client = FakeClient({"q": _run("run_1", "hi")})
    result = ci.run_suite({"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["hi"]}}]}, client)
    p1 = ci.save_result(result, directory=str(tmp_path))
    p2 = ci.save_result(result, directory=str(tmp_path))
    assert p1 != p2


# ============================================================================ (e) malformed cases don't crash
def test_malformed_cases_degrade_to_error_status_without_crashing():
    client = FakeClient({"hi": _run("run_1", "hello there")})
    suite = {"cases": [
        None,
        "just a string, not a dict",
        {"name": "no-prompt"},
        {"name": "bad-prompt-type", "prompt": 12345},
        {"name": "bad-expect-shape", "prompt": "hi", "expect": "not-a-dict"},
        {"name": "empty-case", "prompt": "hi", "expect": {}},
        {"name": "good", "prompt": "hi", "expect": {"contains": ["hello"]}},
    ]}
    result = ci.run_suite(suite, client)   # must not raise

    assert len(result.cases) == 7
    statuses = [c.status for c in result.cases]
    assert statuses.count("error") == 6
    assert result.case("good").status == "pass"
    assert result.status == "error"   # worst-of


def test_run_suite_tolerates_missing_or_non_list_cases():
    client = FakeClient({})
    assert ci.run_suite({}, client).cases == []
    assert ci.run_suite({"cases": "not-a-list"}, client).cases == []
    assert ci.run_suite({"cases": None}, client).cases == []


def test_chat_raising_is_an_error_case_not_a_crash():
    client = FakeClient({}, chat_errors={"q": "connection refused"})
    result = ci.run_suite(
        {"cases": [{"name": "c1", "prompt": "q", "expect": {"contains": ["x"]}}]}, client)
    case = result.case("c1")
    assert case.status == "error"
    assert "connection refused" in case.error


# ======================================================================= Client: the mockable HTTP seam
def test_client_chat_uses_clozn_run_id_when_present():
    client = ci.Client("http://fake.local")
    calls = []

    def fake_request(method, path, body=None):
        calls.append((method, path))
        if (method, path) == ("POST", "/v1/chat/completions"):
            return {"id": "chatcmpl-clozn", "clozn_run_id": "run_direct",
                     "choices": [{"finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "Paris."}}]}
        if (method, path) == ("GET", "/runs/run_direct"):
            return _run("run_direct", "Paris.")
        raise AssertionError(f"unexpected request {method} {path}")

    client._request = fake_request
    run = client.chat("What is the capital of France?")
    assert run["id"] == "run_direct"
    assert ("GET", "/runs") not in calls    # resolved directly -- no polling needed


def test_client_chat_forwards_frozen_sampling_fields():
    client = ci.Client("http://fake.local")
    request_body = {}

    def fake_request(method, path, body=None):
        if (method, path) == ("POST", "/v1/chat/completions"):
            request_body.update(body)
            return {"clozn_run_id": "run_direct"}
        if (method, path) == ("GET", "/runs/run_direct"):
            return _run("run_direct", "answer")
        raise AssertionError(f"unexpected request {method} {path}")

    client._request = fake_request
    client.chat("question", temperature=0.7, top_p=0.9, seed=42)

    assert request_body["temperature"] == 0.7
    assert request_body["top_p"] == 0.9
    assert request_body["seed"] == 42


def test_client_chat_falls_back_to_newest_run_polling_when_run_id_absent():
    client = ci.Client("http://fake.local")

    def fake_request(method, path, body=None):
        if (method, path) == ("POST", "/v1/chat/completions"):
            # no clozn_run_id -- e.g. `_log_run` failed server-side (contracts §18)
            return {"id": "chatcmpl-clozn",
                     "choices": [{"finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "Paris."}}]}
        if (method, path) == ("GET", "/runs"):
            return {"runs": [{"id": "run_newest", "created_at": "now"}]}
        if (method, path) == ("GET", "/runs/run_newest"):
            return _run("run_newest", "Paris.")
        raise AssertionError(f"unexpected request {method} {path}")

    client._request = fake_request
    run = client.chat("What is the capital of France?")
    assert run["id"] == "run_newest"


def test_client_get_run_404_returns_none():
    client = ci.Client("http://fake.local")

    def fake_request(method, path, body=None):
        raise ci.ClientHTTPError(404, {"error": "run not found"})

    client._request = fake_request
    assert client.get_run("nope") is None


def test_client_receipts_404_returns_none_other_errors_propagate():
    client = ci.Client("http://fake.local")

    def not_found(method, path, body=None):
        raise ci.ClientHTTPError(404, {"error": "run not found"})

    client._request = not_found
    assert client.receipts("nope") is None

    def server_error(method, path, body=None):
        raise ci.ClientHTTPError(503, {"error": "receipts need the qwen substrate"})

    client._request = server_error
    with pytest.raises(ci.ClientHTTPError):
        client.receipts("run_1")


def test_client_chat_raises_when_no_run_can_be_associated():
    client = ci.Client("http://fake.local")

    def fake_request(method, path, body=None):
        if (method, path) == ("POST", "/v1/chat/completions"):
            return {"id": "chatcmpl-clozn", "choices": []}
        if (method, path) == ("GET", "/runs"):
            return {"runs": []}
        raise AssertionError(f"unexpected request {method} {path}")

    client._request = fake_request
    with pytest.raises(ci.CIError):
        client.chat("anything")
