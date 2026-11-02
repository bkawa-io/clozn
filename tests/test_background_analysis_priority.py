from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import pytest

from clozn.experiments.multi_arm import BatchCancelled
from clozn.server.model_routing import (
    BackgroundScoringSubstrate,
    ModelSelection,
    select_control_model_for_run,
)
from clozn.server.request_gate import RequestGate, WorkerGateRegistry


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


def test_queued_foreground_generation_wins_the_next_turn_from_background():
    gate = RequestGate(capacity=2, wait_timeout=1)
    assert gate.acquire_background() is None
    order = []
    foreground_acquired = threading.Event()
    release_foreground = threading.Event()
    background_acquired = threading.Event()

    def foreground():
        assert gate.acquire() is None
        order.append("foreground")
        foreground_acquired.set()
        release_foreground.wait(1)
        gate.release()

    def background():
        assert gate.acquire_background() is None
        order.append("background")
        background_acquired.set()
        gate.release_background()

    foreground_thread = threading.Thread(target=foreground)
    background_thread = threading.Thread(target=background)
    foreground_thread.start()
    _wait_for(lambda: gate.snapshot()["waiting"] == 1)
    background_thread.start()
    gate.release_background()

    assert foreground_acquired.wait(1)
    assert order == ["foreground"]
    assert not background_acquired.is_set()
    release_foreground.set()
    foreground_thread.join(1)
    background_thread.join(1)
    assert not foreground_thread.is_alive()
    assert not background_thread.is_alive()
    assert order == ["foreground", "background"]


def test_background_wait_is_cancellable_without_consuming_foreground_capacity():
    gate = RequestGate(capacity=1, wait_timeout=1)
    assert gate.acquire() is None
    cancel = threading.Event()
    cancel.set()

    assert gate.acquire_background(cancel_check=cancel.is_set, poll_interval=.01) == "cancelled"
    assert gate.snapshot() == {
        "active": 1,
        "waiting": 0,
        "capacity": 1,
        "wait_timeout_seconds": 1.0,
    }
    gate.release()


def test_background_substrate_gates_each_score_and_preserves_capabilities():
    registry = WorkerGateRegistry(("model",), capacity=2, wait_timeout=1)
    outcome, release_generation = registry.acquire_generation("model")
    assert outcome is None
    calls = []

    class Target:
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "scalar_http_concurrent"

        def score_tokens(self, value):
            calls.append(("one", value))
            return [value]

        def score_tokens_many(self, values, *, cancel=None):
            calls.append(("many", values))
            return list(values)

    wrapped = BackgroundScoringSubstrate(Target(), registry, "model")
    result = []
    thread = threading.Thread(target=lambda: result.append(wrapped.score_tokens("score")))
    thread.start()
    time.sleep(.03)
    assert calls == []
    release_generation()
    thread.join(1)

    assert result == [["score"]]
    assert calls == [("one", "score")]
    assert wrapped.score_tokens_many_proof_grade is True
    assert wrapped.score_tokens_many_execution_strategy == "scalar_serial"


def test_background_substrate_gates_analysis_generation():
    registry = WorkerGateRegistry(("model",), capacity=2, wait_timeout=1)
    outcome, release_generation = registry.acquire_generation("model")
    assert outcome is None
    calls = []

    class Target:
        def chat(self, messages, max_new, sample, **kwargs):
            calls.append((messages, max_new, sample, kwargs))
            return "generated"

    wrapped = BackgroundScoringSubstrate(Target(), registry, "model")
    result = []
    thread = threading.Thread(target=lambda: result.append(
        wrapped.chat([{"role": "user", "content": "hi"}], 12, False, stop=[]),
    ))
    thread.start()
    time.sleep(.03)
    assert calls == []
    release_generation()
    thread.join(1)

    assert result == ["generated"]
    assert calls == [([{"role": "user", "content": "hi"}], 12, False, {"stop": []})]


def test_background_many_cancels_while_waiting_for_foreground():
    registry = WorkerGateRegistry(("model",), capacity=2, wait_timeout=1)
    outcome, release_generation = registry.acquire_generation("model")
    assert outcome is None
    called = []

    class Target:
        def score_tokens(self, value):
            called.append(value)
            return value

    wrapped = BackgroundScoringSubstrate(Target(), registry, "model")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(BatchCancelled):
        wrapped.score_tokens_many([{"value": 1}], cancel=cancel)
    assert called == []
    release_generation()


def test_fast_background_scoring_uses_native_batch_and_falls_back_to_scalar():
    registry = WorkerGateRegistry(("model",), capacity=2, wait_timeout=1)
    calls = []

    class Target:
        score_tokens_many_proof_grade = True

        def score_tokens(self, value):
            calls.append(("scalar", value))
            return [value]

        def score_tokens_many_native(self, arms, *, cancel=None):
            calls.append(("native", [arm["value"] for arm in arms]))
            return {
                "results": [[arm["value"]] for arm in arms],
                "numerical_regime": "native_prefix_trie_target_logprob_v1",
                "qualified": False,
                "execution": {
                    "kind": "clozn.teacher-forced-score-many.native-prefix-trie.v1",
                    "approximate_contract": "float32_multi_sequence_batch_shape",
                },
            }

    target = Target()
    wrapped = BackgroundScoringSubstrate(target, registry, "model", scoring_mode="fast")
    result = wrapped.score_tokens_many([{"value": "a"}, {"value": "b"}])
    assert result["evidence_use"] == "approximate"
    assert result["results"] == [["a"], ["b"]]
    assert calls == [("native", ["a", "b"])]

    def fail_native(_arms, *, cancel=None):
        raise RuntimeError("native unavailable")

    target.score_tokens_many_native = fail_native
    result = wrapped.score_tokens_many([{"value": "a"}, {"value": "b"}])
    assert result["evidence_use"] == "trusted"
    assert result["execution_strategy"] == "scalar_serial"
    assert result["native_fallback"]["used"] is True
    assert calls[-2:] == [("scalar", "a"), ("scalar", "b")]


def test_background_many_yields_to_foreground_between_scores():
    registry = WorkerGateRegistry(("model",), capacity=2, wait_timeout=1)
    first_started = threading.Event()
    finish_first = threading.Event()
    foreground_acquired = threading.Event()
    finish_foreground = threading.Event()
    calls = []

    class Target:
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "scalar_http_concurrent"

        def score_tokens(self, value):
            calls.append(value)
            if value == "first":
                first_started.set()
                finish_first.wait(1)
            return value

    wrapped = BackgroundScoringSubstrate(Target(), registry, "model")
    result = []
    background = threading.Thread(target=lambda: result.extend(
        wrapped.score_tokens_many([{"value": "first"}, {"value": "second"}]),
    ))

    def foreground():
        outcome, release = registry.acquire_generation("model")
        assert outcome is None
        foreground_acquired.set()
        finish_foreground.wait(1)
        release()

    background.start()
    assert first_started.wait(1)
    foreground_thread = threading.Thread(target=foreground)
    foreground_thread.start()
    _wait_for(lambda: registry.snapshot()["model"]["waiting"] == 1)
    finish_first.set()

    assert foreground_acquired.wait(1)
    assert calls == ["first"]
    finish_foreground.set()
    foreground_thread.join(1)
    background.join(1)

    assert result == ["first", "second"]
    assert calls == ["first", "second"]


def test_run_selection_wraps_scoring_but_control_requests_remain_direct(monkeypatch):
    from clozn.server import app

    target = SimpleNamespace(score_tokens=lambda *_args, **_kwargs: [])
    direct = ModelSelection("model", target, object(), None)

    class Router:
        _gate = WorkerGateRegistry(("model",), capacity=2, wait_timeout=1)

        def select_control_model(self, _model, *, route):
            assert route == "/analysis"
            return direct

    monkeypatch.setattr(app, "MODEL_ROUTER", Router())
    selected = select_control_model_for_run(object(), "model", route="/analysis")

    assert isinstance(selected.sub, BackgroundScoringSubstrate)
    assert selected.engine is direct.engine
