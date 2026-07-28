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
    assert calls[0][3] == {"prompt_instructions": [], "max_new": 256}
    assert calls[1][3]["max_new"] == 256
    assert calls[1][3]["prompt_instructions"][0].startswith("Clozn active corrective response policy:")
    assert calls[1][1]["corrective_retry"] == {
        "arm": "corrected",
        "preset": "less-verbose",
        "method": "system_instruction",
        "instruction": corrective.CORRECTION_PRESETS["less-verbose"],
        "scope": "once",
    }
    assert calls[0][2] is sub and calls[1][2] is sub
    assert result["baseline_reply"] == "long answer"
    assert result["corrected_reply"] == "short"
    assert result["child_ids"] == {"baseline": "run-child-1", "corrected": "run-child-2"}
    assert result["changed"] is True
    assert result["intervention_observed"] is True
    assert isinstance(result["delta"], dict)


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
