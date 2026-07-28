from __future__ import annotations

import pytest

from clozn.behavior import corrective_retries as policy
from clozn.profiles import store as profiles


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "_PATH", str(tmp_path / "corrective.json"))
    profile_store = profiles.ProfileStore(str(tmp_path / "profiles"))
    monkeypatch.setattr(policy, "_profile_store", lambda: profile_store)
    return profile_store


def test_session_policy_activates_injects_and_undoes(isolated):
    key = "session_0123456789abcdef01234567"
    activated = policy.activate("session", key, "less-verbose", now=100.0)
    assert policy.session_presets(key, now=101.0) == ["less-verbose"]
    delivered = [{"role": "user", "content": "Tell me"}]
    effective = policy.inject(delivered, policy.effective_presets(session_key=key, now=101.0))
    assert delivered == [{"role": "user", "content": "Tell me"}]
    assert effective[0]["role"] == "system" and "answer concisely" in effective[0]["content"]

    undone = policy.undo(activated["undo_id"], now=102.0)
    assert undone["status"] == "undone"
    assert policy.session_presets(key, now=103.0) == []


def test_stale_session_undo_refuses_newer_policy(isolated):
    key = "session_0123456789abcdef01234567"
    first = policy.activate("session", key, "less-verbose", now=100.0)
    policy.activate("session", key, "more-concrete", now=101.0)
    with pytest.raises(policy.CorrectivePolicyError, match="stale undo"):
        policy.undo(first["undo_id"], now=102.0)
    assert policy.session_presets(key, now=103.0) == ["less-verbose", "more-concrete"]


def test_profile_policy_travels_in_bundle_and_undoes(isolated):
    isolated.save(profiles.new_profile("work"))
    activated = policy.activate("profile", "work", "use-context", now=100.0)
    assert isolated.load("work")["response_policies"] == ["use-context"]
    assert policy.profile_presets("work") == ["use-context"]

    policy.undo(activated["undo_id"], now=101.0)
    assert isolated.load("work")["response_policies"] == []


# ---- undo must be PROVEN to restore prior state (not just report "undone"), including restoring a
# NON-EMPTY prior state, and proven to REFUSE -- leaving state untouched -- when the target drifted. ----

def test_undoing_the_second_of_two_session_activations_restores_the_first_not_empty(isolated):
    """The existing coverage only ever undoes back to an empty initial state, which cannot
    distinguish "restored the recorded `before`" from "just cleared everything". Undo the SECOND
    (newest, non-stale) transaction and assert the surviving state is exactly the first preset --
    proof the restore used the transaction's own `before` snapshot, not a blanket reset."""
    key = "session_0123456789abcdef01234567"
    policy.activate("session", key, "less-verbose", now=100.0)
    second = policy.activate("session", key, "more-concrete", now=101.0)
    assert policy.session_presets(key, now=101.5) == ["less-verbose", "more-concrete"]

    undone = policy.undo(second["undo_id"], now=102.0)
    assert undone["status"] == "undone"
    assert undone["presets"] == ["less-verbose"]
    assert policy.session_presets(key, now=103.0) == ["less-verbose"]     # restored, not wiped


def test_undoing_the_second_of_two_profile_activations_restores_the_first_not_empty(isolated):
    isolated.save(profiles.new_profile("work"))
    policy.activate("profile", "work", "less-verbose", now=100.0)
    second = policy.activate("profile", "work", "more-concrete", now=101.0)
    assert isolated.load("work")["response_policies"] == ["less-verbose", "more-concrete"]

    policy.undo(second["undo_id"], now=102.0)
    assert isolated.load("work")["response_policies"] == ["less-verbose"]  # restored, not wiped


def test_stale_profile_undo_refuses_and_leaves_the_drifted_bundle_untouched(isolated):
    """Mirrors test_stale_session_undo_refuses_newer_policy for the OTHER scope corrective_retries
    supports. The compare-and-swap check at corrective_retries.undo() (comparing the transaction's
    `after` snapshot against the profile's LIVE response_policies) had no direct test for the
    profile branch -- only the session branch was exercised. Proves both the refusal AND that the
    bundle a second, independent activation produced is left completely intact."""
    isolated.save(profiles.new_profile("work"))
    first = policy.activate("profile", "work", "less-verbose", now=100.0)
    policy.activate("profile", "work", "more-concrete", now=101.0)   # drifts the target out from under `first`
    drifted = isolated.load("work")["response_policies"]
    assert drifted == ["less-verbose", "more-concrete"]

    with pytest.raises(policy.CorrectivePolicyError, match="stale undo"):
        policy.undo(first["undo_id"], now=102.0)

    # the refused undo must not have touched the bundle AT ALL, not even a partial write.
    assert isolated.load("work")["response_policies"] == drifted


def test_stale_profile_undo_also_refuses_when_drifted_by_a_direct_external_save(isolated):
    """Drift doesn't only come from a second corrective_retries.activate() call -- any external
    write to the SAME bundle (a hand edit, a different client, /profiles/save) must be caught too,
    since undo's contract is "restores the prior profile only if it has not drifted externally"."""
    isolated.save(profiles.new_profile("work"))
    activated = policy.activate("profile", "work", "less-verbose", now=100.0)

    bundle = isolated.load("work")
    bundle["response_policies"] = ["ask-before-guessing"]     # an unrelated external edit
    isolated.save(bundle)

    with pytest.raises(policy.CorrectivePolicyError, match="stale undo"):
        policy.undo(activated["undo_id"], now=101.0)
    assert isolated.load("work")["response_policies"] == ["ask-before-guessing"]


def test_undo_is_not_reusable_once_applied(isolated):
    """A transaction id must be single-use: undoing twice must not silently succeed a second time
    (which would let a caller replay an undo against whatever state has accumulated since)."""
    key = "session_0123456789abcdef01234567"
    activated = policy.activate("session", key, "less-verbose", now=100.0)
    policy.undo(activated["undo_id"], now=101.0)
    with pytest.raises(policy.CorrectivePolicyError, match="already undone"):
        policy.undo(activated["undo_id"], now=102.0)
