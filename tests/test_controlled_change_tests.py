"""Model-free budget/classification tests for controlled run swaps."""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.replay import controlled


def _sampling(temp: float, seed: int) -> dict:
    return {
        "sampling": "sample",
        "temperature": temp,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "seed": seed,
        "max_tokens": 32,
    }


def _run(rid: str, response: str, *, messages=None, meta=None) -> dict:
    return {
        "id": rid,
        "response": response,
        "messages": list(messages or [{"role": "user", "content": rid}]),
        "meta": dict(meta or _sampling(0.8, 1)),
        "identity": {},
        "context_receipt": {},
        "behavior": {"active_dials": {}},
        "trace": {},
    }


class Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value


class Runner:
    def __init__(self, clock, replies):
        self.clock = clock
        self.replies = dict(replies)
        self.calls = []
        self.cancelled = 0

    def qualify(self, kind, _baseline, _candidate):
        return {"ok": True, "checked": {"fake": kind}}

    def run_arm(self, kind, arm, _baseline, _candidate, *, timeout_seconds):
        self.calls.append((kind, arm, timeout_seconds))
        value = self.replies[(kind, arm)]
        if isinstance(value, tuple):
            response, advance = value
            self.clock.value += advance
        else:
            response = value
        return {"run": {"id": f"run_child_{kind}_{arm}", "response": response,
                        "context_receipt": {}, "trace": {}}}

    def cancel_current(self):
        self.cancelled += 1
        return {"requested": True, "terminated": True, "reason": "fake worker terminated"}


def test_plan_is_schema_valid_and_starts_no_runs():
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}])
    doc = controlled.plan_change_tests(a, b, tests=["context"], max_runs=2, max_seconds=30)
    schemas.validate(doc, controlled.SCHEMA_VERSION)
    assert doc["status"] == "planned"
    assert doc["budget"]["runs_used"] == 0
    assert doc["tests"][0]["stop_reason"] == "planned"


def test_context_control_and_treatment_can_support_one_cause():
    clock = Clock()
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}])
    runner = Runner(clock, {("context", "control"): "bad", ("context", "treatment"): "good"})
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context"], max_runs=2, max_seconds=30, clock=clock
    )
    assert doc["tests"][0]["status"] == "causally_supported"
    assert doc["summary"]["classification"] == "context"
    assert doc["budget"]["runs_used"] == 2
    assert [e["run_id"] for e in doc["tests"][0]["evidence"]] == [
        "run_child_context_control", "run_child_context_treatment",
    ]


def test_treatment_that_does_not_recover_is_not_called_causal():
    clock = Clock()
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}])
    runner = Runner(clock, {("context", "control"): "bad", ("context", "treatment"): "bad"})
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context"], max_runs=2, max_seconds=30, clock=clock
    )
    assert doc["tests"][0]["status"] == "eliminated"
    assert doc["summary"]["classification"] == "undetermined"


def test_budget_never_starts_half_of_a_two_arm_test():
    clock = Clock()
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}])
    runner = Runner(clock, {("context", "control"): "bad", ("context", "treatment"): "good"})
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context"], max_runs=1, max_seconds=30, clock=clock
    )
    assert runner.calls == []
    assert doc["status"] == "budget_exhausted"
    assert doc["tests"][0]["stop_reason"] == "budget_exhausted"
    assert doc["budget"]["runs_used"] == 0


def test_remaining_steps_become_budget_exhausted_after_first_test():
    clock = Clock()
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}],
             meta=_sampling(0.2, 3))
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}],
             meta=_sampling(0.8, 7))
    runner = Runner(clock, {
        ("context", "control"): "bad", ("context", "treatment"): "good",
        ("sampling", "control"): "bad", ("sampling", "treatment"): "good",
    })
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context", "sampling"],
        max_runs=2, max_seconds=30, clock=clock,
    )
    assert len(runner.calls) == 2
    assert doc["tests"][1]["ran"] is False
    assert doc["tests"][1]["stop_reason"] == "budget_exhausted"
    assert doc["status"] == "budget_exhausted"


def test_wall_budget_requests_cancellation_and_blocks_the_treatment():
    clock = Clock()
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}])
    runner = Runner(clock, {
        ("context", "control"): ("bad", 2.0),
        ("context", "treatment"): "good",
    })
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context"], max_runs=2, max_seconds=1, clock=clock
    )
    assert len(runner.calls) == 1
    assert runner.cancelled == 1
    assert doc["budget"]["runs_used"] == 1
    assert doc["status"] == "budget_exhausted"
    assert doc["tests"][0]["status"] == "inconclusive"


def test_two_successful_swaps_are_reported_as_entangled():
    clock = Clock()
    a = _run("run_a", "good", messages=[{"role": "user", "content": "full"}],
             meta=_sampling(0.2, 3))
    b = _run("run_b", "bad", messages=[{"role": "user", "content": "short"}],
             meta=_sampling(0.8, 7))
    runner = Runner(clock, {
        ("context", "control"): "bad", ("context", "treatment"): "good",
        ("sampling", "control"): "bad", ("sampling", "treatment"): "good",
    })
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context", "sampling"],
        max_runs=4, max_seconds=30, clock=clock,
    )
    assert doc["summary"]["classification"] == "entangled"
    assert doc["summary"]["entangled"] is True
    assert doc["summary"]["causally_supported"] == ["context", "sampling"]


def test_template_fingerprint_without_exact_material_is_explicitly_unavailable():
    a = _run("run_a", "good")
    b = _run("run_b", "bad")
    a["identity"]["template_fingerprint"] = "a" * 16
    b["identity"]["template_fingerprint"] = "b" * 16
    doc = controlled.plan_change_tests(a, b, tests=["template"])
    assert doc["tests"][0]["stop_reason"] == "unavailable"
    assert "fingerprint" in doc["tests"][0]["reason"]


def test_no_run_starts_when_selected_metric_has_no_recorded_regression():
    clock = Clock()
    a = _run("run_a", "same", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", "same", messages=[{"role": "user", "content": "short"}])
    runner = Runner(clock, {
        ("context", "control"): "same", ("context", "treatment"): "same",
    })
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["context"], max_runs=2, max_seconds=30, clock=clock
    )
    assert runner.calls == []
    assert doc["budget"]["runs_used"] == 0
    assert doc["tests"][0]["stop_reason"] == "unavailable"
    assert "no measured regression" in doc["tests"][0]["reason"]


def test_sampling_test_requires_an_actual_exact_sampling_change():
    clock = Clock()
    a = _run("run_a", "good")
    b = _run("run_b", "bad")
    a["meta"]["max_tokens"] = 16
    b["meta"]["max_tokens"] = 32
    runner = Runner(clock, {
        ("sampling", "control"): "bad", ("sampling", "treatment"): "good",
    })
    doc = controlled.execute_change_tests(
        a, b, runner=runner, tests=["sampling"], max_runs=2, max_seconds=30, clock=clock
    )
    assert runner.calls == []
    assert doc["tests"][0]["stop_reason"] == "unavailable"
    assert doc["tests"][0]["reason"] == "exact sampling configuration did not change"


def test_invalid_budgets_and_semantic_matcher_are_refused():
    a, b = _run("run_a", "a"), _run("run_b", "b")
    with pytest.raises(controlled.ControlledTestError):
        controlled.plan_change_tests(a, b, max_runs=-1)
    with pytest.raises(controlled.ControlledTestError):
        controlled.plan_change_tests(a, b, match_criterion="semantic_similarity")
