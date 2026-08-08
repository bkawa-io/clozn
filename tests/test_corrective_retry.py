"""Model-free tests for prompt-first corrective retry comparisons."""
from copy import deepcopy

import pytest

from clozn.replay import corrective


def test_preset_vocabulary_is_bounded_and_named():
    assert set(corrective.CORRECTION_PRESETS) == {
        "less-verbose", "more-concrete", "use-context", "ask-before-guessing",
        "preserve-formatting", "stop-repeating",
    }
    with pytest.raises(TypeError):
        corrective.CORRECTION_PRESETS["custom"] = "unbounded instruction"


def test_new_presets_inject_their_own_instruction():
    messages = [{"role": "user", "content": "x"}]
    for preset in ("preserve-formatting", "stop-repeating"):
        injected = corrective.inject_correction(messages, preset)
        assert corrective.CORRECTION_PRESETS[preset] in injected[0]["content"]


def test_injection_preserves_caller_messages_and_nested_payloads():
    messages = [
        {"role": "system", "content": "Caller policy"},
        {"role": "user", "content": "Use this", "metadata": {"items": [1, 2]}},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    original = deepcopy(messages)

    injected = corrective.inject_correction(messages, "use-context")

    assert messages == original
    assert injected[0] == original[0]
    assert injected[1]["role"] == "system"
    assert "Clozn corrective retry" in injected[1]["content"]
    assert injected[2:] == original[1:]
    injected[2]["metadata"]["items"].append(3)
    assert messages[1]["metadata"]["items"] == [1, 2]


def test_retry_compare_uses_two_greedy_children_and_records_exact_correction(monkeypatch):
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "Explain it"}]}
    calls = []
    sub = object()

    def fake_replay(arm_run, changes, actual_sub, **kwargs):
        calls.append((deepcopy(arm_run), deepcopy(changes), actual_sub, deepcopy(kwargs)))
        index = len(calls)
        child = {"id": f"run-child-{index}", "response": "long answer" if index == 1 else "short"}
        if kwargs.get("prompt_instructions"):
            child["assembled_messages"] = [{"role": "system", "content": kwargs["prompt_instructions"][0]}]
        return child

    monkeypatch.setattr(corrective, "replay_run", fake_replay)
    result = corrective.retry_compare(run, "less-verbose", sub)

    assert len(calls) == 2
    assert all(call[1]["greedy"] is True for call in calls)
    assert calls[0][0]["messages"] == run["messages"]
    assert calls[1][0]["messages"] == run["messages"]
    assert calls[0][3] == {"max_new": 256}
    assert calls[1][3]["max_new"] == 256
    assert calls[1][3]["prompt_instructions"][0].startswith("Clozn active corrective response policy:")
    assert calls[1][1]["corrective_retry"] == {
        "arm": "corrected",
        "preset": "less-verbose",
        "method": "system_instruction",
        "instruction": corrective.CORRECTION_PRESETS["less-verbose"],
    }
    assert calls[0][2] is sub and calls[1][2] is sub
    assert result["baseline_reply"] == "long answer"
    assert result["corrected_reply"] == "short"
    assert result["child_ids"] == {"baseline": "run-child-1", "corrected": "run-child-2"}
    assert result["changed"] is True
    assert result["intervention_observed"] is True
    assert isinstance(result["delta"], dict)


def test_retry_compare_has_no_scope_or_active_presets_parameter():
    """Durable session/profile scoping was retired (docs/CAPABILITIES.md) -- retry_compare accepts
    no `scope` or `active_presets` argument, so a caller has no way to ask for persisted behavior;
    every retry is request-local by construction."""
    import inspect
    params = inspect.signature(corrective.retry_compare).parameters
    assert "scope" not in params
    assert "active_presets" not in params


def test_retry_compare_stops_when_an_arm_fails(monkeypatch):
    calls = []

    def fail_baseline(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(corrective, "replay_run", fail_baseline)
    assert corrective.retry_compare(
        {"id": "run-parent", "messages": []}, "more-concrete", object()
    ) is None
    assert len(calls) == 1


def test_unknown_preset_fails_before_generation(monkeypatch):
    monkeypatch.setattr(corrective, "replay_run", lambda *_: pytest.fail("must not generate"))
    with pytest.raises(ValueError, match="unknown corrective preset"):
        corrective.retry_compare(
            {"id": "run-parent", "messages": [{"role": "user", "content": "x"}]},
            "invent-a-new-policy",
            object(),
        )


def test_invalid_backend_value_raises(monkeypatch):
    monkeypatch.setattr(corrective, "replay_run", lambda *_: pytest.fail("must not generate"))
    with pytest.raises(ValueError, match="backend must be"):
        corrective.retry_compare(
            {"id": "run-parent", "messages": [{"role": "user", "content": "x"}]},
            "less-verbose", object(), backend="not-a-real-backend",
        )


class _FakeCalibratedSteer:
    def __init__(self, table):
        self._table = table

    def ceiling_for(self, name):
        return self._table.get(name, (0.6, False))


class _FakeSub:
    def __init__(self, steer):
        self.steer = steer


def test_control_vector_backend_applies_the_calibrated_dial(monkeypatch):
    """backend="control_vector" on a calibrated model routes through replay's OWN
    behavior_overrides/snapshot-restore mechanism (clozn/replay/replay.py) -- no prompt text at all
    for the corrected arm -- and the return value is honestly labeled, not disguised as a preset."""
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "Explain it"}]}
    calls = []

    def fake_replay(arm_run, changes, actual_sub, **kwargs):
        calls.append(deepcopy(changes))
        index = len(calls)
        if index == 1:
            return {"id": "run-child-1", "response": "long answer"}
        return {"id": "run-child-2", "response": "short",
                "behavior": {"active_dials": {"concise": 0.804}}}

    monkeypatch.setattr(corrective, "replay_run", fake_replay)
    sub = _FakeSub(_FakeCalibratedSteer({"concise": (1.2, True)}))
    result = corrective.retry_compare(run, "less-verbose", sub, backend="control_vector")

    assert calls[1]["behavior_overrides"] == {"concise": 0.804}
    assert calls[1]["corrective_retry"]["method"] == "control_vector"
    assert "instruction" not in calls[1]["corrective_retry"]
    assert result["backend"] == "control_vector"
    assert result["backend_fallback"] is False
    assert result["intervention_observed"] is True
    ident = result["execution_identity"]["ext"]["behavior_intervention"]
    assert ident["backend"] == "control_vector"
    assert ident["qualification"] == "model_build_exact"
    assert ident["qualified"] is True
    assert ident["parameters"] == {"dial": "concise", "strength": 0.804}


def test_control_vector_falls_back_to_prompt_policy_when_uncalibrated(monkeypatch):
    """No calibration for THIS exact model (e959477's fail-closed contract) -> honest fallback, never
    a silent prompt-policy substitution presented as control_vector."""
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "Explain it"}]}

    def fake_replay(arm_run, changes, actual_sub, **kwargs):
        return {"id": f"run-child-{len(calls_seen)}", "response": "reply"}
    calls_seen = []
    monkeypatch.setattr(corrective, "replay_run",
                        lambda *a, **k: (calls_seen.append(1), fake_replay(*a, **k))[1])

    sub = _FakeSub(_FakeCalibratedSteer({}))              # ceiling_for -> (0.6, False): uncalibrated
    result = corrective.retry_compare(run, "less-verbose", sub, backend="control_vector")

    assert result["backend"] == "prompt_policy"
    assert result["backend_fallback"] is True
    assert result["execution_identity"]["ext"]["behavior_intervention"]["fallback"] is True


def test_control_vector_falls_back_for_an_action_with_no_matching_dial(monkeypatch):
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "x"}]}
    monkeypatch.setattr(corrective, "replay_run",
                        lambda *a, **k: {"id": "c", "response": "r"})
    sub = _FakeSub(_FakeCalibratedSteer({"concise": (1.2, True)}))
    result = corrective.retry_compare(run, "use-context", sub, backend="control_vector")
    assert result["backend"] == "prompt_policy"
    assert result["backend_fallback"] is True


def test_default_backend_is_prompt_policy_even_with_a_calibrated_steer(monkeypatch):
    """The spec: raw dials must never be the default interaction. Omitting `backend` must NOT
    silently prefer control_vector just because the loaded model happens to be calibrated."""
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "x"}]}
    monkeypatch.setattr(corrective, "replay_run",
                        lambda *a, **k: {"id": "c", "response": "r"})
    sub = _FakeSub(_FakeCalibratedSteer({"concise": (1.2, True)}))
    result = corrective.retry_compare(run, "less-verbose", sub)
    assert result["backend"] == "prompt_policy"
    assert result["backend_fallback"] is False
