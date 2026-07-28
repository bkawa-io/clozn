"""Model-free tests for clozn.runs.identity_providers.behavior_intervention (Seam 3) and its wiring
into clozn.replay.corrective.retry_compare()'s execution_identity field."""
from __future__ import annotations

from copy import deepcopy

from clozn.replay import corrective
from clozn.runs import identity_ext
from clozn.runs.identity_providers import behavior_intervention


def test_provider_normalizes_a_well_formed_fact():
    out = behavior_intervention.identity({"behavior_intervention": {
        "action_id": "less-verbose", "backend": "prompt_policy", "registry_version": "1",
        "parameters": {"preset": "less-verbose"}, "qualification": "generic", "qualified": None,
        "fallback": False,
    }})
    assert out == {
        "action_id": "less-verbose", "backend": "prompt_policy", "registry_version": "1",
        "parameters": {"preset": "less-verbose"}, "qualification": "generic", "fallback": False,
    }


def test_provider_omits_qualified_when_not_a_bool():
    out = behavior_intervention.identity(
        {"behavior_intervention": {"action_id": "less-verbose", "backend": "prompt_policy",
                                    "qualified": None}})
    assert "qualified" not in out


def test_provider_contributes_nothing_without_the_fact():
    assert behavior_intervention.identity({}) is None
    assert behavior_intervention.identity({"unrelated": True}) is None
    assert behavior_intervention.identity(None) is None


def test_provider_requires_both_action_id_and_backend():
    assert behavior_intervention.identity({"behavior_intervention": {"action_id": "less-verbose"}}) is None
    assert behavior_intervention.identity({"behavior_intervention": {"backend": "prompt_policy"}}) is None


def test_provider_is_discovered_by_the_shipped_seam():
    """The real shipped identity_ext scan (not a throwaway shim) must find this provider."""
    identity_ext.reset_cache()
    try:
        out = identity_ext.collect({"behavior_intervention": {
            "action_id": "use-context", "backend": "prompt_policy"}})
    finally:
        identity_ext.reset_cache()
    assert out["behavior_intervention"] == {"action_id": "use-context", "backend": "prompt_policy"}
    assert identity_ext.COLLECT_FAILURES == []


def test_retry_compare_attaches_execution_identity(monkeypatch):
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "Explain it"}]}

    def fake_replay(arm_run, changes, actual_sub, **kwargs):
        index = fake_replay.calls
        fake_replay.calls += 1
        child = {"id": f"run-child-{index}", "response": "long answer" if index == 0 else "short"}
        if kwargs.get("prompt_instructions"):
            child["assembled_messages"] = [{"role": "system", "content": kwargs["prompt_instructions"][0]}]
        return child
    fake_replay.calls = 0

    monkeypatch.setattr(corrective, "replay_run", fake_replay)
    result = corrective.retry_compare(run, "less-verbose", object())

    ident = result["execution_identity"]
    assert ident["parent_run_id"] == "run-parent"
    assert ident["action_id"] == "less-verbose"
    assert ident["backend"] == "prompt_policy"
    assert ident["before_hash"] == corrective._sha256("long answer")
    assert ident["after_hash"] == corrective._sha256("short")
    assert ident["ext"]["behavior_intervention"]["action_id"] == "less-verbose"
    assert ident["ext"]["behavior_intervention"]["qualification"] == "generic"
    assert "qualified" not in ident["ext"]["behavior_intervention"]
    assert result["backend"] == "prompt_policy"


def test_execution_identity_hashes_are_deterministic_and_content_addressed():
    a = corrective._sha256("same text")
    b = corrective._sha256("same text")
    c = corrective._sha256("different text")
    assert a == b and a != c
    assert len(a) == 64


def test_retry_compare_does_not_mutate_caller_run(monkeypatch):
    run = {"id": "run-parent", "messages": [{"role": "user", "content": "x"}]}
    original = deepcopy(run)

    def fake_replay(arm_run, changes, actual_sub, **kwargs):
        return {"id": "child", "response": "y"}

    monkeypatch.setattr(corrective, "replay_run", fake_replay)
    corrective.retry_compare(run, "more-concrete", object())
    assert run == original
